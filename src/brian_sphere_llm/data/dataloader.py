from __future__ import annotations

import json
from pathlib import Path

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
        expected = int(self.index["num_sequences"]) * self.sequence_length
        actual = self.bin_path.stat().st_size // 4
        if self.bin_path.stat().st_size % 4 != 0:
            raise ValueError(f"Token file byte size must be divisible by 4: {self.bin_path}")
        if actual != expected:
            raise ValueError(f"Token file has {actual} tokens; expected {expected}")
        self.tokens = torch.from_file(
            str(self.bin_path),
            shared=False,
            size=actual,
            dtype=torch.int32,
        )

    def __len__(self) -> int:
        return int(self.index["num_sequences"])

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.sequence_length
        end = start + self.sequence_length
        return self.tokens[start:end].to(dtype=torch.long)


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
    if distributed or shuffle:
        if DistributedSampler is None:
            raise ModuleNotFoundError("PyTorch DistributedSampler is required for distributed dataloaders.")
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=shuffle,
            seed=seed,
            drop_last=distributed,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
    )
