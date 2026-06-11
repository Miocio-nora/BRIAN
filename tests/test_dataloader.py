from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brian_sphere_llm.data.dataloader import PackedTokenDataset, build_dataloader
from brian_sphere_llm.data.pack import write_index, write_token_bin


def test_validation_dataloader_preserves_fixed_order(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    sequences = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    write_token_bin(sequences, tokenized / "val.bin")
    write_index(tokenized / "val.idx", sequence_length=3, num_sequences=len(sequences))

    first_loader = build_dataloader(tokenized_dir=tokenized, split="val", batch_size=2, shuffle=False)
    second_loader = build_dataloader(tokenized_dir=tokenized, split="val", batch_size=2, shuffle=False)

    first_batches = [batch.tolist() for batch in first_loader]
    second_batches = [batch.tolist() for batch in second_loader]
    assert first_batches == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
    assert second_batches == first_batches


def test_train_dataloader_uses_distributed_sampler_when_requested(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    sequences = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    write_token_bin(sequences, tokenized / "train.bin")
    write_index(tokenized / "train.idx", sequence_length=3, num_sequences=len(sequences))

    loader = build_dataloader(
        tokenized_dir=tokenized,
        split="train",
        batch_size=1,
        shuffle=False,
        distributed=True,
        rank=1,
        world_size=2,
        seed=7,
    )

    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 1
    assert [batch.tolist() for batch in loader] == [[[4, 5, 6]], [[10, 11, 12]]]


def test_shuffled_train_dataloader_supports_epoch_deterministic_order(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    sequences = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
    ]
    write_token_bin(sequences, tokenized / "train.bin")
    write_index(tokenized / "train.idx", sequence_length=3, num_sequences=len(sequences))

    first_loader = build_dataloader(tokenized_dir=tokenized, split="train", batch_size=1, shuffle=True, seed=11)
    second_loader = build_dataloader(tokenized_dir=tokenized, split="train", batch_size=1, shuffle=True, seed=11)

    first_loader.sampler.set_epoch(3)
    second_loader.sampler.set_epoch(3)
    assert [batch.tolist() for batch in first_loader] == [batch.tolist() for batch in second_loader]

    first_loader.sampler.set_epoch(4)
    second_loader.sampler.set_epoch(3)
    assert [batch.tolist() for batch in first_loader] != [batch.tolist() for batch in second_loader]


def test_packed_dataset_rejects_token_index_mismatch(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    write_token_bin([[1, 2, 3]], tokenized / "train.bin")
    write_index(tokenized / "train.idx", sequence_length=3, num_sequences=2)

    with pytest.raises(ValueError, match="Token file has 3 tokens; expected 6"):
        PackedTokenDataset(tokenized / "train.bin", tokenized / "train.idx")
