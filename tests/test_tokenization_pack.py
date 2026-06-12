import json
from pathlib import Path

from brian_sphere_llm.data.pack import FixedLengthTokenBinWriter, pack_fixed_length, read_token_bin, write_index, write_token_bin
from brian_sphere_llm.data.tokenize import SimpleByteTokenizer, tokenizer_metadata


def test_simple_byte_tokenizer_metadata() -> None:
    tokenizer = SimpleByteTokenizer()
    meta = tokenizer_metadata(tokenizer, name="simple-byte-tokenizer", revision="local", license="test")
    assert meta.vocab_size == 260
    assert tokenizer.encode("A", add_special_tokens=True)[0] == tokenizer.bos_token_id


def test_simple_byte_tokenizer_saves_reproducible_artifacts(tmp_path: Path) -> None:
    tokenizer = SimpleByteTokenizer()
    paths = tokenizer.save_pretrained(tmp_path)
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_json = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))

    assert {Path(path).name for path in paths} == {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    assert tokenizer_config["tokenizer_class"] == "SimpleByteTokenizer"
    assert tokenizer_json["format"] == "simple-byte-tokenizer-v1"
    assert tokenizer_json["model"]["byte_tokens"]["<0x41>"] == 65


def test_pack_and_bin_roundtrip(tmp_path: Path) -> None:
    sequences = pack_fixed_length([[1, 2, 3], [4]], sequence_length=3, pad_token_id=0)
    assert sequences == [[1, 2, 3], [4, 0, 0]]
    bin_path = tmp_path / "train.bin"
    idx_path = tmp_path / "train.idx"
    write_token_bin(sequences, bin_path)
    write_index(idx_path, sequence_length=3, num_sequences=2)
    assert read_token_bin(bin_path) == [1, 2, 3, 4, 0, 0]


def test_streaming_token_bin_writer_matches_fixed_length_pack(tmp_path: Path) -> None:
    bin_path = tmp_path / "stream.bin"
    with FixedLengthTokenBinWriter(bin_path, sequence_length=3, pad_token_id=0, flush_sequences=1) as writer:
        writer.add_document([1, 2])
        writer.add_document([3, 4])
        assert writer.close() == 2

    assert read_token_bin(bin_path) == [1, 2, 3, 4, 0, 0]
