from __future__ import annotations

import json
from pathlib import Path

from brian_sphere_llm.data.pack import read_token_bin

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None
    DistributedSampler = None
    Dataset = object


class PackedTokenDataset(Dataset):
    def __init__(self, bin_path: str | Path, idx_path: str | Path) -> None:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required for dataloaders.")
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        with self.idx_path.open("r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.sequence_length = int(self.index["sequence_length"])
        self.tokens = torch.tensor(read_token_bin(self.bin_path), dtype=torch.long)
        expected = int(self.index["num_sequences"]) * self.sequence_length
        if self.tokens.numel() != expected:
            raise ValueError(f"Token file has {self.tokens.numel()} tokens; expected {expected}")

    def __len__(self) -> int:
        return int(self.index["num_sequences"])

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.sequence_length
        end = start + self.sequence_length
        return self.tokens[start:end]


def build_dataloader(
    *,
    tokenized_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> "DataLoader":
    if torch is None or DataLoader is None:
        raise ModuleNotFoundError("PyTorch is required for dataloaders.")
    tokenized_dir = Path(tokenized_dir)
    dataset = PackedTokenDataset(tokenized_dir / f"{split}.bin", tokenized_dir / f"{split}.idx")
    sampler = None
    if distributed:
        if DistributedSampler is None:
            raise ModuleNotFoundError("PyTorch DistributedSampler is required for distributed dataloaders.")
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
    )
