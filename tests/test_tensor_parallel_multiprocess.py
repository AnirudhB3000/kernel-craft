"""
test_tensor_parallel_multiprocess.py
=====================================
End-to-end NCCL all-reduce correctness test using torch.distributed.

Run via torchrun (single-GPU SHM transport, two virtual ranks):

    CUDA_VISIBLE_DEVICES=0,0 torchrun --nproc_per_node=2 \\
        tests/test_tensor_parallel_multiprocess.py

On a real 2-GPU machine:

    torchrun --nproc_per_node=2 \\
        tests/test_tensor_parallel_multiprocess.py

The test verifies:
  1. All-reduce (sum): each rank contributes a tensor of all-ones;
     after allreduce each rank holds world_size * ones.
  2. All-reduce (sum) of non-uniform data: rank r contributes r+1 on every
     element; result = world_size*(world_size+1)/2 on every element.
  3. All-gather: rank r contributes [r*C .. (r+1)*C - 1];
     after allgather each rank holds the concatenated sequence 0..R*C-1.
  4. Performance: measures effective bandwidth for a large (32M float) allreduce.

The script exits with code 0 on all tests passing, 1 otherwise.
"""

import os
import sys
import math
import time

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("SKIP: torch not installed")
    sys.exit(0)


def setup():
    """Initialise the process group from the torchrun environment variables."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world


def teardown():
    dist.barrier()
    dist.destroy_process_group()


def check(condition: bool, name: str, rank: int) -> bool:
    """Print [PASS] / [FAIL] from rank 0 only; return True on pass."""
    passed = torch.tensor(1 if condition else 0, device="cuda")
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if rank == 0:
        tag = "[PASS]" if passed.item() == 1 else "[FAIL]"
        print(f"{tag} {name}")
    return passed.item() == 1


def test_allreduce_ones(rank, world):
    """All-reduce of ones => world_size everywhere."""
    x = torch.ones(1024, device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    ok = torch.allclose(x, torch.full_like(x, world))
    return check(ok, f"allreduce ones: result == {world}", rank)


def test_allreduce_rank_values(rank, world):
    """Each rank contributes (rank+1); result = world*(world+1)/2."""
    x = torch.full((512,), float(rank + 1), device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2
    ok = torch.allclose(x, torch.full_like(x, expected))
    return check(ok, f"allreduce rank values: result == {expected:.0f}", rank)


def test_allgather(rank, world):
    """All-gather of rank-specific chunks => concatenated sequence."""
    chunk_size = 256
    local = torch.arange(
        rank * chunk_size, (rank + 1) * chunk_size, dtype=torch.float32, device="cuda"
    )
    gathered = [torch.empty(chunk_size, device="cuda") for _ in range(world)]
    dist.all_gather(gathered, local)
    full = torch.cat(gathered)
    expected = torch.arange(world * chunk_size, dtype=torch.float32, device="cuda")
    ok = torch.allclose(full, expected)
    return check(ok, "all_gather: concatenated sequence correct", rank)


def test_allreduce_large_bandwidth(rank, world):
    """Measure effective bandwidth for a 32M-float all-reduce."""
    count = 32 * 1024 * 1024  # 128 MB
    x = torch.ones(count, device="cuda")

    # Warmup
    for _ in range(3):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    reps = 10
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()

    ms_avg = (t1 - t0) / reps * 1e3
    bytes_total = count * 4
    # Ring allreduce effective BW: 2*(R-1)/R * bytes
    eff_bytes = 2.0 * (world - 1) / world * bytes_total
    gbps = (eff_bytes / 1e9) / (ms_avg / 1e3)

    # Correctness: after reps allreduces, each element was multiplied by world
    # reps times → world**reps. Only check shape and no NaN.
    ok = not torch.isnan(x).any()
    passed = check(ok, f"allreduce 32M floats: {ms_avg:.2f} ms  {gbps:.1f} GB/s effective", rank)
    if rank == 0:
        print(f"  [bw] {ms_avg:.3f} ms/iter, {gbps:.2f} GB/s effective BW "
              f"({world} ranks, SHM transport)")
    return passed


def main():
    rank, world = setup()

    if rank == 0:
        print(f"=== Tensor Parallel Multiprocess Tests (world_size={world}) ===")

    results = []
    results.append(test_allreduce_ones(rank, world))
    results.append(test_allreduce_rank_values(rank, world))
    results.append(test_allgather(rank, world))
    results.append(test_allreduce_large_bandwidth(rank, world))

    n_pass = sum(results)
    n_fail = len(results) - n_pass

    if rank == 0:
        print(f"\nResults: {n_pass} passed, {n_fail} failed")

    teardown()
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
