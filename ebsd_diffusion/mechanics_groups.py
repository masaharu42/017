"""Contiguous same-alloy groups, without introducing new independent labels."""
from __future__ import annotations
import torch
from torch.utils.data import Sampler


class AlloyGroupBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, group_size, seed):
        if group_size < 1 or batch_size % group_size:
            raise ValueError("Batch size must be divisible by K")
        self.dataset, self.batch_size, self.group_size = dataset, batch_size, group_size
        self.seed, self.epoch = int(seed), 0
        ids = [int(row["alloy_id"]) for row in dataset.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Group sampler requires one source view per alloy")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        n = len(self.dataset.rows)
        count = len(self) * (self.batch_size // self.group_size)
        rows = torch.cat([torch.randperm(n, generator=rng) for _ in range((count+n-1)//n)])[:count]
        batch = []
        for row in rows.tolist():
            batch.extend([row] * self.group_size)  # dataset draws a fresh crop per access
            if len(batch) == self.batch_size:
                yield batch
                batch = []


def validate_groups(ids, conditions, k):
    if len(ids) % k:
        raise ValueError("Incomplete mechanics group")
    i = ids.reshape(-1, k)
    c = conditions.reshape(-1, k, conditions.shape[-1])
    if not bool((i == i[:, :1]).all()) or not bool((c == c[:, :1]).all()):
        raise ValueError("Mechanics groups must share alloy and composition")


def group_random_inputs(conditions, k, num_timesteps, jitter_std, dropout):
    """Same RNG consumption for C0/C1; independent image noise is sampled elsewhere."""
    groups = len(conditions) // k
    t = torch.randint(num_timesteps, (groups,), device=conditions.device).repeat_interleave(k)
    jitter = torch.randn(groups, conditions.shape[-1], device=conditions.device) * jitter_std
    condition = conditions + jitter.repeat_interleave(k, 0)
    drop = (torch.rand(groups, device=conditions.device) < dropout).repeat_interleave(k)
    return t, condition, drop
