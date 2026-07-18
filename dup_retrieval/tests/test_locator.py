"""Tests for the Stage-A tar-offset locator (manifest.py).

Exercises, without GPU and without real shar data:
  - _scan_tar_offsets: seek(offset)+read(size) returns byte-identical audio
  - meta (.json) + placeholder (.nodata/.nometa) members are excluded
  - cut_ids containing dots are parsed correctly
  - _shard_num pairs a cuts shard with its recording tar by number
  - discover_shards + _process_shard end-to-end populate the locator columns,
    and a text-only cut (no audio member) gets null offsets

Run from the repo root:

    PYTHONPATH=. python3 audio_tokenization/utils/data_selection/dup_retrieval/tests/test_locator.py
"""

import gzip
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

# Make the dup_retrieval package importable when run as a script.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]  # ../../../../../..
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_tokenization.utils.data_selection.dup_retrieval.core.manifest import (
    _scan_tar_offsets, _shard_num, discover_shards, _process_shard,
)


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


def _build_fixture() -> tuple:
    """Build a Lhotse-style shar dir: recording tar (data+meta pairs, one
    placeholder cut, one dotted cut_id) + shar_index + cuts shard."""
    shar = Path(tempfile.mkdtemp(prefix="locator_test_"))
    payloads = {
        "c1":          b"FAKEFLAC-c1-" + bytes(range(60)),
        "weird.id.v2": b"FAKEFLAC-weird-" + bytes(range(120)),
    }
    tar_path = shar / "recording.000000.tar"
    with tarfile.open(tar_path, "w") as tar:          # uncompressed -> seekable
        _add_bytes(tar, "c1.flac", payloads["c1"])
        _add_bytes(tar, "c1.json", b'{"meta":"c1"}')
        _add_bytes(tar, "weird.id.v2.flac", payloads["weird.id.v2"])
        _add_bytes(tar, "weird.id.v2.json", b'{"meta":"weird"}')
        _add_bytes(tar, "c3.nodata", b"")             # placeholder: no audio
        _add_bytes(tar, "c3.nometa", b"")
    (shar / "shar_index.json").write_text(json.dumps({
        "fields": {"cuts": ["cuts.000000.jsonl.gz"],
                   "recording": ["recording.000000.tar"]}}))
    cuts = [
        {"id": "c1", "duration": 1.0,
         "supervisions": [{"text": "hello world this is cut one"}],
         "recording": {"id": "rec_c1", "num_samples": 16000}},
        {"id": "weird.id.v2", "duration": 2.0,
         "supervisions": [{"text": "second cut dotted identifier here"}],
         "recording": {"id": "rec_w", "num_samples": 32000}},
        {"id": "c3", "duration": 1.5,                 # text-only: no audio member
         "supervisions": [{"text": "third cut has text but no audio payload"}],
         "recording": {"id": "rec_c3", "num_samples": 24000}},
    ]
    with gzip.open(shar / "cuts.000000.jsonl.gz", "wt") as f:
        for c in cuts:
            f.write(json.dumps(c) + "\n")
    return shar, tar_path, payloads


def test_scan_tar_offsets_byte_identity():
    _shar, tar_path, payloads = _build_fixture()
    offs = _scan_tar_offsets(str(tar_path))
    # meta (.json) + placeholder (.nodata/.nometa) excluded; dotted id kept.
    assert set(offs) == {"c1", "weird.id.v2"}, sorted(offs)
    with open(tar_path, "rb") as fh:
        for cid, (off, size, fmt) in offs.items():
            fh.seek(off)
            assert fh.read(size) == payloads[cid], f"byte mismatch for {cid}"
            assert fmt == "flac"


def test_shard_num():
    assert _shard_num("cuts.000007.jsonl.gz") == "000007"
    assert _shard_num("recording.000007.tar") == "000007"
    assert _shard_num("no_number_here.tar") is None


def test_process_shard_join():
    shar, tar_path, _payloads = _build_fixture()
    offs = _scan_tar_offsets(str(tar_path))

    shards = discover_shards([str(shar)])
    assert len(shards) == 1
    assert Path(shards[0].recording_path).resolve() == tar_path.resolve()

    rows = _process_shard((shards[0], {"min_text_chars": 4}))["rows"]
    idx = {cid: i for i, cid in enumerate(rows["cut_id"])}
    assert set(idx) == {"c1", "weird.id.v2", "c3"}

    for cid in ("c1", "weird.id.v2"):
        i = idx[cid]
        assert rows["tar_path"][i] == "recording.000000.tar"
        assert rows["tar_offset"][i] == offs[cid][0]
        assert rows["tar_size"][i] == offs[cid][1]
        assert rows["audio_format"][i] == "flac"

    i3 = idx["c3"]  # text-only cut -> no audio member -> null locator
    assert rows["tar_path"][i3] is None
    assert rows["tar_offset"][i3] is None


if __name__ == "__main__":
    test_scan_tar_offsets_byte_identity()
    test_shard_num()
    test_process_shard_join()
    print("ALL PASS")
