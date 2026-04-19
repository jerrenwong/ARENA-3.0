import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Silence the C++ "unbatched P2P op" warning NCCL emits on lazy sub-comm init.
# Must be set before torch is imported.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Make exercises dir importable so part2_cnns / part3_optimization packages resolve.
chapter = "chapter0_fundamentals"
root_dir = next(p for p in Path(__file__).resolve().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# part2_cnns.solutions uses Path.cwd() to locate the repo root on import, which
# fails unless cwd is inside the repo. Temporarily chdir so the import works
# from any cwd (script or notebook).
with contextlib.chdir(exercises_dir):
    import part3_optimization.tests as tests  # noqa: F401
    from part2_cnns.solutions import Linear, ResNet34
    from part3_optimization.solutions import WandbResNetFinetuningArgs, get_cifar

WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12345")


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    t.cuda.set_device(rank)
    device = t.device(f"cuda:{rank}")

    if rank == 0:
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)
    elif rank == 1:
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()


def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """Broadcast `tensor` from `src` rank to all other ranks."""
    if rank == src:
        for other_rank in range(world_size):
            if other_rank != src:
                dist.send(tensor, dst=other_rank)
    else:
        dist.recv(tensor, src=src)


def reduce(tensor, rank, world_size, dst=0, op="sum"):
    """Reduce `tensor` across ranks into `dst` (sum or mean)."""
    if rank != dst:
        dist.send(tensor, dst=dst)
    else:
        for other_rank in range(world_size):
            if other_rank != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=other_rank)
                tensor += received_tensor
        if op == "mean":
            tensor /= world_size


def all_reduce(tensor, rank, world_size, op="sum"):
    """Allreduce via reduce-to-0 then broadcast-from-0."""
    reduce(tensor, rank, world_size, dst=0, op=op)
    broadcast(tensor, rank, world_size, src=0)


def ring_all_reduce(tensor: Tensor, rank: int, world_size: int, op: str = "sum") -> None:
    """Bandwidth-optimal ring all-reduce.

    Splits `tensor` into `world_size` chunks along dim 0 and rotates ONE chunk
    per round. Total bytes moved per rank ≈ 2·(ws-1)/ws · |tensor|, independent
    of `world_size`. A naive "rotate the whole tensor" ring sends `|tensor|` per
    round and so scales badly (≈ 2·(ws-1)·|tensor|).

    Chunk-index bookkeeping: at reduce-scatter step `k`, rank `r` sends the
    chunk indexed (r - k) mod ws to rank+1 and receives the chunk indexed
    (r - k - 1) mod ws from rank-1 (into its own slot at that index). After
    ws-1 such rounds, rank r holds the full sum of chunk (r+1) mod ws. The
    all-gather phase uses the same rotation but overwrites instead of adding.

    Uses isend/irecv so every rank can post its send + recv concurrently
    without deadlocking.
    """
    # Flatten so we can chunk along a single dim regardless of the input shape.
    flat = tensor.view(-1)
    chunks = list(flat.tensor_split(world_size))  # views — writes flow back to `tensor`

    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size

    # --- Reduce-scatter: each round each rank is responsible for one chunk ---
    for k in range(world_size - 1):
        send_idx = (rank - k) % world_size
        recv_idx = (rank - k - 1) % world_size
        recv_buf = t.zeros_like(chunks[recv_idx])
        send_req = dist.isend(chunks[send_idx].contiguous(), dst=send_to)
        recv_req = dist.irecv(recv_buf, src=recv_from)
        send_req.wait()
        recv_req.wait()
        chunks[recv_idx] += recv_buf

    # --- All-gather: rotate the now-summed chunks around the ring ---
    for k in range(world_size - 1):
        send_idx = (rank - k + 1) % world_size
        recv_idx = (rank - k) % world_size
        recv_buf = t.zeros_like(chunks[recv_idx])
        send_req = dist.isend(chunks[send_idx].contiguous(), dst=send_to)
        recv_req = dist.irecv(recv_buf, src=recv_from)
        send_req.wait()
        recv_req.wait()
        chunks[recv_idx].copy_(recv_buf)

    if op == "mean":
        tensor /= world_size


class SimpleModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    t.cuda.set_device(rank)
    device = t.device(f"cuda:{rank}")

    model = SimpleModel().to(device)
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)
    print(f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}")

    optimizer.step()
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()


def run_simple_model_ddp(rank, world_size):
    """DDP equivalent of run_simple_model: gradient sync happens inside loss.backward()."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    t.cuda.set_device(rank)
    device = t.device(f"cuda:{rank}")

    # DDP broadcasts initial weights from rank 0 and registers hooks that all-reduce
    # grads automatically when backward() runs. No manual sync needed.
    model = DDP(SimpleModel().to(device), device_ids=[rank])
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # DDP hooks fire here, averaging grads across ranks

    optimizer.step()
    # Access the underlying module via `.module` since DDP wraps it.
    print(f"Rank {rank}, new param (DDP): {model.module.param.data}")

    dist.destroy_process_group()


def get_untrained_resnet(n_classes: int) -> ResNet34:
    """Untrained ResNet34 from part2_cnns with its classifier head swapped for `n_classes`."""
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"
    reduce_mode: str = "manual"  # "manual" = our two-stage all_reduce, "ring" = ring_all_reduce


def _cuda_time():
    """GPU-accurate timestamp. Sync the device first so prior kernels are done."""
    if t.cuda.is_available():
        t.cuda.synchronize()
    return time.perf_counter()


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        # Sync initial weights: rank 0 broadcasts each parameter to all other ranks.
        for param in self.model.parameters():
            broadcast(param.data, self.rank, self.args.world_size, src=0)

    def pre_training_setup(self):
        self.optimizer = t.optim.SGD(self.model.parameters(), lr=0.1)
        # Rank 0 downloads CIFAR first; others wait to avoid a parallel-download race.
        if self.rank == 0:
            self.trainset, self.testset = get_cifar()
        dist.barrier()
        if self.rank != 0:
            self.trainset, self.testset = get_cifar()
        dist.barrier()
        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset, num_replicas=self.args.world_size, rank=self.rank
        )
        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset, num_replicas=self.args.world_size, rank=self.rank
        )
        loader_kwargs = dict(batch_size=self.args.batch_size, num_workers=2, pin_memory=True)
        self.train_loader = t.utils.data.DataLoader(self.trainset, sampler=self.train_sampler, **loader_kwargs)
        self.test_loader = t.utils.data.DataLoader(self.testset, sampler=self.test_sampler, **loader_kwargs)
        self.examples_seen = 0

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        reduce_fn = ring_all_reduce if self.args.reduce_mode == "ring" else all_reduce

        t0 = _cuda_time()
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        t1 = _cuda_time()
        loss.backward()
        t2 = _cuda_time()
        for param in self.model.parameters():
            reduce_fn(param.grad, self.rank, self.args.world_size, op="mean")
        t3 = _cuda_time()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Record per-step component times (seconds).
        self._step_times["fwd"] += t1 - t0
        self._step_times["bwd"] += t2 - t1
        self._step_times["dist"] += t3 - t2
        self._step_times["n"] += 1
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        pbar = tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0, leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        # Sum across ranks once at the end (cheaper than per-batch all_reduce).
        stats = t.tensor([total_correct, total_samples], device=self.device)
        all_reduce(stats, self.rank, self.args.world_size, op="sum")
        total_correct, total_samples = stats.tolist()
        return total_correct / total_samples

    def train(self):
        self.pre_training_setup()
        for epoch in range(self.args.epochs):
            self.train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            self._step_times = {"fwd": 0.0, "bwd": 0.0, "dist": 0.0, "n": 0}
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch} [{self.args.reduce_mode}]",
                disable=self.rank != 0,
            )
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                total_loss += loss.item() * imgs.shape[0]
                self.examples_seen += imgs.shape[0] * self.args.world_size
                n = self._step_times["n"]
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    fwd=f"{1000 * self._step_times['fwd'] / n:.1f}ms",
                    bwd=f"{1000 * self._step_times['bwd'] / n:.1f}ms",
                    dist=f"{1000 * self._step_times['dist'] / n:.1f}ms",
                )
            accuracy = self.evaluate()
            if self.rank == 0:
                n = self._step_times["n"]
                avg_loss = total_loss / (len(self.train_loader) * self.args.world_size)
                print(
                    f"Epoch {epoch} [{self.args.reduce_mode}] | "
                    f"avg_loss={avg_loss:.4f} | accuracy={accuracy:.4f} | "
                    f"per-step: fwd={1000 * self._step_times['fwd'] / n:.1f}ms "
                    f"bwd={1000 * self._step_times['bwd'] / n:.1f}ms "
                    f"dist={1000 * self._step_times['dist'] / n:.1f}ms"
                )


def dist_train_resnet_from_scratch(rank, world_size, reduce_mode: str = "manual"):
    t.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size, reduce_mode=reduce_mode)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


def dist_train_resnet_from_scratch_ring(rank, world_size):
    """Same as dist_train_resnet_from_scratch but uses ring_all_reduce."""
    dist_train_resnet_from_scratch(rank, world_size, reduce_mode="ring")


class DDPResNetTrainer:
    """DDP version of DistResNetTrainer.

    Differences vs. the manual version:
      * Model is wrapped in DDP; initial weights are broadcast automatically.
      * No manual all_reduce of gradients in training_step — DDP's backward
        hooks do it, overlapping comm with compute.
      * evaluate() uses the built-in dist.all_reduce (NCCL) rather than our
        send/recv-based version.
    """

    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")
        base_model = get_untrained_resnet(self.args.n_classes).to(self.device)
        self.model = DDP(base_model, device_ids=[rank])

    def pre_training_setup(self):
        self.optimizer = t.optim.SGD(self.model.parameters(), lr=0.1)
        if self.rank == 0:
            self.trainset, self.testset = get_cifar()
        dist.barrier()
        if self.rank != 0:
            self.trainset, self.testset = get_cifar()
        dist.barrier()
        self.train_sampler = t.utils.data.DistributedSampler(
            self.trainset, num_replicas=self.args.world_size, rank=self.rank
        )
        self.test_sampler = t.utils.data.DistributedSampler(
            self.testset, num_replicas=self.args.world_size, rank=self.rank
        )
        loader_kwargs = dict(batch_size=self.args.batch_size, num_workers=2, pin_memory=True)
        self.train_loader = t.utils.data.DataLoader(self.trainset, sampler=self.train_sampler, **loader_kwargs)
        self.test_loader = t.utils.data.DataLoader(self.testset, sampler=self.test_sampler, **loader_kwargs)
        self.examples_seen = 0

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        t0 = _cuda_time()
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        t1 = _cuda_time()
        # DDP fuses backward + gradient all-reduce (hooks fire during backward,
        # overlapping comm with compute) — they can't be separated cleanly.
        loss.backward()
        t2 = _cuda_time()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self._step_times["fwd"] += t1 - t0
        self._step_times["bwd_dist"] += t2 - t1
        self._step_times["n"] += 1
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        pbar = tqdm(self.test_loader, desc="Evaluating", disable=self.rank != 0, leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        stats = t.tensor([total_correct, total_samples], device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_correct, total_samples = stats.tolist()
        return total_correct / total_samples

    def train(self):
        self.pre_training_setup()
        for epoch in range(self.args.epochs):
            self.train_sampler.set_epoch(epoch)
            self.model.train()
            total_loss = 0.0
            self._step_times = {"fwd": 0.0, "bwd_dist": 0.0, "n": 0}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [ddp]", disable=self.rank != 0)
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                total_loss += loss.item() * imgs.shape[0]
                self.examples_seen += imgs.shape[0] * self.args.world_size
                n = self._step_times["n"]
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    fwd=f"{1000 * self._step_times['fwd'] / n:.1f}ms",
                    bwd_dist=f"{1000 * self._step_times['bwd_dist'] / n:.1f}ms",
                )
            accuracy = self.evaluate()
            if self.rank == 0:
                n = self._step_times["n"]
                avg_loss = total_loss / (len(self.train_loader) * self.args.world_size)
                print(
                    f"Epoch {epoch} [ddp] | "
                    f"avg_loss={avg_loss:.4f} | accuracy={accuracy:.4f} | "
                    f"per-step: fwd={1000 * self._step_times['fwd'] / n:.1f}ms "
                    f"bwd+dist={1000 * self._step_times['bwd_dist'] / n:.1f}ms"
                )


def dist_train_resnet_from_scratch_ddp(rank, world_size):
    t.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DDPResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


# TRAIN_MODE selects the gradient-sync strategy for the training run.
#   "manual" — our hand-rolled two-stage all_reduce (rank-0 bottleneck)
#   "ring"   — our chunked ring all_reduce (bandwidth-optimal)
#   "ddp"    — torch's DistributedDataParallel (bucketed + comm/compute overlap)
ENTRYPOINT = os.environ.get("TRAIN_MODE", "ddp")

if __name__ == "__main__":
    world_size = t.cuda.device_count()
    target = {
        "manual": dist_train_resnet_from_scratch,
        "ring": dist_train_resnet_from_scratch_ring,
        "ddp": dist_train_resnet_from_scratch_ddp,
    }[ENTRYPOINT]
    mp.spawn(target, args=(world_size,), nprocs=world_size, join=True)
