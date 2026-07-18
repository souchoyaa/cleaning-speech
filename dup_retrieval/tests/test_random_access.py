"""Tests for the Stage-D random-access planner (loader_audio._plan_tar_reads)
and the M1->M2 read path (manifest offsets -> plan -> seek+read byte identity).

Pure planning logic + raw tar I/O; no GPU, no real audio decode.  (The decode /
resample / batch tensor path needs torch+soundfile and is validated by the
cluster smoke run.)

Run from the repo root:

    PYTHONPATH=. python3 audio_tokenization/utils/data_selection/dup_retrieval/tests/test_random_access.py
"""

import io
import sys
import tarfile
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_tokenization.utils.data_selection.dup_retrieval.core.loader_audio import (
    _plan_tar_reads,
)
from audio_tokenization.utils.data_selection.dup_retrieval.core.manifest import (
    _scan_tar_offsets,
)


def test_plan_rank_split_and_sort():
    loc = {
        "a": ("recording.000000.tar", 100, 10, "flac", 5),
        "b": ("recording.000000.tar", 50, 10, "flac", 3),
        "c": ("recording.000001.tar", 200, 10, "flac", 9),
        "d": ("recording.000002.tar", 0, 10, "flac", 1),
    }
    p0 = _plan_tar_reads(loc, "ds", set(), rank=0, world_size=2)
    p1 = _plan_tar_reads(loc, "ds", set(), rank=1, world_size=2)
    assert [t for t, _ in p0] == ["recording.000000.tar", "recording.000002.tar"]
    assert [t for t, _ in p1] == ["recording.000001.tar"]
    # within a tar: sort by (num_samples, offset) -> b (ns 3) before a (ns 5)
    assert [m[0] for m in dict(p0)["recording.000000.tar"]] == ["b", "a"]
    # rank split is a partition: full cover, no overlap
    t0, t1 = {t for t, _ in p0}, {t for t, _ in p1}
    assert t0 | t1 == {f"recording.00000{i}.tar" for i in (0, 1, 2)}
    assert not (t0 & t1)


def test_plan_skip_keys():
    loc = {
        "a": ("recording.000000.tar", 100, 10, "flac", 5),
        "b": ("recording.000000.tar", 50, 10, "flac", 3),
    }
    plan = _plan_tar_reads(loc, "ds", {("ds", "a")}, rank=0, world_size=1)
    assert [m[0] for m in dict(plan)["recording.000000.tar"]] == ["b"]


def test_plan_to_read_byte_identity():
    shar = Path(tempfile.mkdtemp(prefix="ra_test_"))
    payloads = {"c1": b"FLAC-c1-" + bytes(range(80)),
                "weird.id.v2": b"FLAC-w-" + bytes(range(140))}
    tar_path = shar / "recording.000000.tar"

    def _add(tar, name, data):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))

    with tarfile.open(tar_path, "w") as tar:
        _add(tar, "c1.flac", payloads["c1"]); _add(tar, "c1.json", b"{}")
        _add(tar, "weird.id.v2.flac", payloads["weird.id.v2"])
        _add(tar, "weird.id.v2.json", b"{}")

    offs = _scan_tar_offsets(str(tar_path))
    locator = {cid: ("recording.000000.tar", off, sz, fmt, 16000)
               for cid, (off, sz, fmt) in offs.items()}

    got = {}
    for rel, members in _plan_tar_reads(locator, "ds", set(), 0, 1):
        with open(shar / rel, "rb") as fh:          # same seek/read the reader does
            for cid, off, sz, fmt, ns in members:
                fh.seek(off)
                got[cid] = fh.read(sz)
    assert got == payloads


if __name__ == "__main__":
    test_plan_rank_split_and_sort()
    test_plan_skip_keys()
    test_plan_to_read_byte_identity()
    print("ALL PASS")
