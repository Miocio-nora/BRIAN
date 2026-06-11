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


def test_packed_dataset_rejects_token_index_mismatch(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    write_token_bin([[1, 2, 3]], tokenized / "train.bin")
    write_index(tokenized / "train.idx", sequence_length=3, num_sequences=2)

    with pytest.raises(ValueError, match="Token file has 3 tokens; expected 6"):
        PackedTokenDataset(tokenized / "train.bin", tokenized / "train.idx")
