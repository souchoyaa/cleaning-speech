"""Smoke tests for the algorithmic cores of Stages C and D.

Exercises constellation-map keypoint extraction, triplet hash pack + Hough
voting, text-cluster retention, and the cudf/dask_cudf Stage-C backends (which
skip when no GPU is present).

Run from the repo root:

    PYTHONPATH=. python3 audio_tokenization/utils/data_selection/dup_retrieval/tests/test_algorithms.py
"""

import os
import sys
from pathlib import Path

# Make the dup_retrieval package importable when run as a script.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]  # ../../../../../..
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from audio_tokenization.utils.data_selection.dup_retrieval.core.audio_fingerprint import (
    _pack_hash, encode_signatures, constellation_map,
    extract_keypoints_per_cut,
)
from audio_tokenization.utils.data_selection.dup_retrieval.core.audio_match import (
    _hough_match_score, _shared_hash_count,
)
from audio_tokenization.utils.data_selection.dup_retrieval.core.retention import (
    select_text_cluster_drops,
)


# ---------------------------------------------------------------------------
# Stage C tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage D + E tests
# ---------------------------------------------------------------------------


def test_hash_pack_roundtrip():
    """The 26-bit packed hash should preserve all five fields."""
    for fb, fk, ff, dtb, dtf in [
        (0, 0, 0, 0, 0),
        (63, 63, 63, 15, 15),
        (1, 2, 3, 4, 5),
        (33, 17, 50, 9, 11),
    ]:
        h = _pack_hash(fb, fk, ff, dtb, dtf)
        assert (h >> 20) & 0x3F == fb
        assert (h >> 14) & 0x3F == fk
        assert (h >>  8) & 0x3F == ff
        assert (h >>  4) & 0x0F == dtb
        assert  h        & 0x0F == dtf
    print(f"  test_hash_pack_roundtrip: all 4 cases OK")


def test_encode_signatures_window():
    """Synthetic keypoints: verify only those with both partners in [m,M) emit."""
    # m=4, M=20.  Keypoints at t = 0, 5, 10, 25, 30.
    kp = [(0, 1), (5, 2), (10, 3), (25, 4), (30, 5)]
    sigs = encode_signatures(kp, m=4, M=20)
    # Center=5: backward t=0 (5-0=5 in [4,20)) ✓; forward t=10 (10-5=5) ✓ -> emit.
    # Center=10: backward t=5 (5 in [4,20)) ✓; forward t=25 (25-10=15) ✓ -> emit.
    # Center=25: backward t=10 (15) ✓; forward t=30 (5) ✓ -> emit.
    # Center=0: no backward -> drop.
    # Center=30: no forward -> drop.
    print(f"  test_encode_signatures_window: emitted {len(sigs)} signatures, "
          f"expected 3.  ts={[s[1] for s in sigs]}")
    assert len(sigs) == 3, f"expected 3 sigs, got {len(sigs)}"
    assert sorted(s[1] for s in sigs) == [5, 10, 25]


def test_hough_voting_aligned_pair():
    """Two identical hash sequences with constant offset → max_bin == n."""
    hashes = np.array([100, 200, 300, 400], dtype=np.int64)
    times_a = np.array([0, 10, 20, 30], dtype=np.int64)
    times_b = times_a + 5  # B is shifted +5 frames vs A
    max_bin, span_a, span_b = _hough_match_score(hashes, times_a, hashes, times_b)
    print(f"  test_hough_voting_aligned_pair: max_bin={max_bin}, "
          f"span_a={span_a}, span_b={span_b}")
    assert max_bin == 4, f"expected max_bin=4, got {max_bin}"
    assert span_a == 31  # 30 - 0 + 1
    assert span_b == 31


def test_hough_voting_unaligned():
    """Random hash overlap with no temporal coherence → low max_bin."""
    rng = np.random.default_rng(0)
    h = rng.integers(0, 10000, size=20, dtype=np.int64)
    ta = rng.integers(0, 100, size=20, dtype=np.int64)
    tb = rng.integers(0, 100, size=20, dtype=np.int64)
    max_bin, _, _ = _hough_match_score(h, ta, h, tb)
    print(f"  test_hough_voting_unaligned: max_bin={max_bin} (should be small)")
    assert max_bin <= 4, f"expected low max_bin, got {max_bin}"


def test_constellation_map_silent():
    """Silent input produces no constellation peaks (all-zero mel → all False)."""
    import torch
    mel = torch.zeros(1, 64, 100)  # B=1, F=64, T=100
    lengths = torch.tensor([100], dtype=torch.int64)
    mask = constellation_map(mel, lengths, energy_factor=1.0, time_window=9)
    n_peaks = int(mask.sum().item())
    print(f"  test_constellation_map_silent: n_peaks={n_peaks} (should be 0)")
    assert n_peaks == 0


def test_constellation_map_synthetic():
    """A synthetic mel with isolated tone peaks should produce keypoints."""
    import torch
    mel = torch.full((1, 64, 100), 0.001)
    # Drop two strong "tones": (freq=20, t=10) and (freq=40, t=60).
    mel[0, 20, 10] = 5.0
    mel[0, 40, 60] = 5.0
    lengths = torch.tensor([100], dtype=torch.int64)
    mask = constellation_map(mel, lengths, energy_factor=1.0, time_window=9)
    kp_per = extract_keypoints_per_cut(mask, mel, hash_frame_rate=12.5,
                                       mel_frame_rate=40.0)
    print(f"  test_constellation_map_synthetic: keypoints={kp_per[0]}")
    # Both peaks should be detected (one keypoint each), but they collapse
    # to nearby t_hash bins; expect at least 2 keypoints.
    assert len(kp_per[0]) >= 2, f"expected ≥2 keypoints, got {kp_per[0]}"


def test_shared_hash_count():
    a = np.array([1, 2, 3, 3, 4], dtype=np.int64)
    b = np.array([3, 3, 4, 5], dtype=np.int64)
    # Multisets: a={1,2,3,3,4}, b={3,3,4,5}.  Intersection: {3,3,4} -> count 3.
    n = _shared_hash_count(a, b)
    print(f"  test_shared_hash_count: a∩b = {n} (expected 3)")
    assert n == 3, f"expected 3, got {n}"


# ---------------------------------------------------------------------------
# Stage F tests
# ---------------------------------------------------------------------------


def test_text_cluster_cap_keeps_top_x():
    # 5 cuts in one text cluster; cap to 2.  scores rank: idx1 > idx3 > idx0 >
    # idx4 > idx2(None).  Keep {1,3}, drop {0,2,4}.
    scores   = [0.5, 0.9, None, 0.7, 0.1]
    datasets = ["d"] * 5
    durations = [1.0] * 5
    drops = select_text_cluster_drops([0, 1, 2, 3, 4], scores, datasets,
                                      durations, pref_rank={}, cap=2)
    print(f"  test_text_cluster_cap_keeps_top_x: drops={sorted(drops)} (expected [0, 2, 4])")
    assert sorted(drops) == [0, 2, 4], f"got {sorted(drops)}"


def test_text_cluster_cap_noop_under_cap():
    # Cluster already within cap -> nothing dropped.
    drops = select_text_cluster_drops([0, 1], [0.1, 0.2], ["d", "d"],
                                      [1.0, 1.0], pref_rank={}, cap=5)
    print(f"  test_text_cluster_cap_noop_under_cap: drops={drops} (expected [])")
    assert drops == [], f"got {drops}"


def test_text_cluster_cap_tiebreak_pref_then_duration():
    # Equal scores -> preferred dataset wins; then longer duration.  cap=1.
    scores   = [0.5, 0.5, 0.5]
    datasets = ["other", "preferred", "other"]
    durations = [9.0, 1.0, 2.0]
    pref_rank = {"preferred": 0}
    drops = select_text_cluster_drops([0, 1, 2], scores, datasets, durations,
                                      pref_rank, cap=1)
    print(f"  test_text_cluster_cap_tiebreak: kept idx1, drops={sorted(drops)} (expected [0, 2])")
    assert sorted(drops) == [0, 2], f"got {sorted(drops)}"


def test_cudf_backend_clusters():
    """Single-GPU cudf Stage C backend groups near-dups and separates distinct
    text.  Skips when cudf / a GPU is unavailable (e.g. local dev)."""
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"  test_cudf_backend_clusters: SKIP (no cudf/GPU: {repr(e)[:60]})")
        return
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from audio_tokenization.utils.data_selection.dup_retrieval.core.text_dedup import (
        _run_cudf_backend,
    )

    base = "the quick brown fox jumps over the lazy dog near the river bank at dawn"
    texts = [
        base,
        base + " today",                     # near-dup: short append keeps all n-grams
        "today " + base,                     # near-dup: short prepend keeps all n-grams
        "an entirely unrelated sentence about astronomy and very distant galaxies",
        "yet another wholly different line concerning marine biology field research",
    ]
    cfg_c = dict(num_permutations=128, ngram_size=12, num_bands=16, band_width=8,
                 jaccard_threshold=0.5, small_bucket_threshold=100,
                 max_bucket_size=1000, seed=1)
    with tempfile.TemporaryDirectory() as td:
        mdir = Path(td) / "manifest"
        mdir.mkdir()
        tbl = pa.table({"dataset": ["d"] * len(texts),
                        "cut_id": [f"c{i}" for i in range(len(texts))],
                        "normalized_text": texts})
        pq.write_table(tbl, str(mdir / "part_00000.parquet"))
        _ds, cid, roots, _edges = _run_cudf_backend(mdir, cfg_c)

    g = {c: int(r) for c, r in zip(cid, np.asarray(roots).tolist())}
    near = {g["c0"], g["c1"], g["c2"]}
    print(f"  test_cudf_backend_clusters: near-dup roots={sorted(near)}, "
          f"distinct=({g.get('c3')},{g.get('c4')})")
    assert len(near) == 1, f"near-dups not merged into one component: {near}"
    assert g["c3"] != g["c0"] and g["c4"] != g["c0"], "distinct lines wrongly merged"
    assert g["c3"] != g["c4"], "two unrelated lines wrongly merged"


def test_dask_cudf_backend_clusters():
    """Multi-GPU dask_cudf Stage C backend (cuGraph MG CC if present, else scipy
    on gathered edges).  Skips when dask-cuda / a GPU is unavailable."""
    try:
        import cudf  # noqa: F401
        import dask_cudf  # noqa: F401
        from dask_cuda import LocalCUDACluster  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"  test_dask_cudf_backend_clusters: SKIP (no dask-cuda/GPU: {repr(e)[:60]})")
        return
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from audio_tokenization.utils.data_selection.dup_retrieval.core.text_dedup import (
        _run_dask_cudf_backend,
    )

    base = "the quick brown fox jumps over the lazy dog near the river bank at dawn"
    texts = [base, base + " today", "today " + base,
             "an entirely unrelated sentence about astronomy and distant galaxies",
             "yet another wholly different line concerning marine biology research"]
    cfg_c = dict(num_permutations=128, ngram_size=12, num_bands=16, band_width=8,
                 jaccard_threshold=0.5, max_bucket_size=1000, seed=1,
                 rmm_pool_size="6GB", use_cugraph_cc=True)
    with tempfile.TemporaryDirectory() as td:
        mdir = Path(td) / "manifest"
        mdir.mkdir()
        pq.write_table(
            pa.table({"dataset": ["d"] * len(texts),
                      "cut_id": [f"c{i}" for i in range(len(texts))],
                      "normalized_text": texts}),
            str(mdir / "part_00000.parquet"))
        _ds, cid, roots, _edges = _run_dask_cudf_backend(mdir, cfg_c)

    g = {c: int(r) for c, r in zip(cid, np.asarray(roots).tolist())}
    near = {g["c0"], g["c1"], g["c2"]}
    print(f"  test_dask_cudf_backend_clusters: near-dup roots={sorted(near)}, "
          f"distinct=({g.get('c3')},{g.get('c4')})")
    assert len(near) == 1, f"near-dups not merged into one component: {near}"
    assert g["c3"] != g["c0"] and g["c4"] != g["c0"], "distinct lines wrongly merged"
    assert g["c3"] != g["c4"], "two unrelated lines wrongly merged"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_hash_pack_roundtrip,
        test_encode_signatures_window,
        test_hough_voting_aligned_pair,
        test_hough_voting_unaligned,
        test_constellation_map_silent,
        test_constellation_map_synthetic,
        test_shared_hash_count,
        test_text_cluster_cap_keeps_top_x,
        test_text_cluster_cap_noop_under_cap,
        test_text_cluster_cap_tiebreak_pref_then_duration,
        test_cudf_backend_clusters,
        test_dask_cudf_backend_clusters,
    ]
    n_pass = 0
    for t in tests:
        print(f"[*] {t.__name__}")
        t()
        n_pass += 1
    print(f"\n{n_pass}/{len(tests)} smoke tests passed.")


if __name__ == "__main__":
    main()
