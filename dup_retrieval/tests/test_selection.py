"""Tests for the quality bridge (quality_ingest._pivot_quality) and the
flag-only unique-pair policy (retention._selection_columns).

Pure logic; no parquet I/O, no GPU.  Run from the repo root:

    PYTHONPATH=. python3 audio_tokenization/utils/data_selection/dup_retrieval/tests/test_selection.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_tokenization.utils.data_selection.dup_retrieval.core.quality_ingest import (
    _pivot_quality,
)
from audio_tokenization.utils.data_selection.dup_retrieval.core.retention import (
    _selection_columns, load_quality_for_datasets,
    _flatten_metrics, _quality_components, _component_stats,
    raw_score, rank_score, gate_score, axes_for_mode,
)


def test_pivot_quality_union_and_nulls():
    quality = {
        ("ds1", "c1"): {"utmos": 4.1, "dnsmos_nisqa": 3.5},
        ("ds1", "c2"): {"utmos": 2.0, "audiobox_OVL": 7.2, "rover_primary_fallbacks": 3},
    }
    ds, cid, keys, cols = _pivot_quality(quality)
    assert ds == ["ds1", "ds1"] and cid == ["c1", "c2"]
    assert keys == ["audiobox_OVL", "dnsmos_nisqa", "rover_primary_fallbacks", "utmos"]
    assert cols["utmos"] == [4.1, 2.0]
    assert cols["dnsmos_nisqa"] == [3.5, None]   # c2 lacks dnsmos
    assert cols["audiobox_OVL"] == [None, 7.2]   # c1 lacks audiobox
    assert cols["rover_primary_fallbacks"] == [None, 3]


def test_pivot_quality_empty():
    ds, cid, keys, cols = _pivot_quality({})
    assert ds == [] and cid == [] and keys == [] and cols == {}


def test_selection_columns_flag_only():
    datasets = ["d1", "d1", "d2", "d1"]
    scores = [4.5, 3.0, None, 2.0]
    floor = {"d1": 4.0, "d2": 5.0}
    cluster_sz = [2, 2, 1, 1]   # cuts 0,1 share an audio cluster; 2,3 are alone
    is_kept = [True, False, True, True]
    retention = ["quality", "quality_dropped", "singleton", "singleton"]

    dup, low, keep_reason = _selection_columns(
        datasets, scores, floor, cluster_sz, is_kept, retention)

    assert dup == [True, True, False, False]              # is_duplicate = cluster_size > 1
    assert low == [False, True, False, True]              # below floor (None score -> False)
    assert keep_reason == ["kept", "quality_dropped", "kept", "low_quality"]
    # flag-only policy: is_kept must be untouched
    assert is_kept == [True, False, True, True]


def test_selection_columns_gate_flag():
    """gate_flagged ORs into low_quality even when no floor trips it."""
    dup, low, keep_reason = _selection_columns(
        ["d1", "d1"], [None, None], {}, [1, 1], [True, True],
        ["singleton", "singleton"], gate_flagged=[False, True])
    assert low == [False, True]
    assert keep_reason == ["kept", "low_quality"]


def test_selection_columns_missing_floor():
    dup, low, keep_reason = _selection_columns(
        ["dX"], [0.1], {}, [1], [True], ["singleton"])
    assert low == [False] and keep_reason == ["kept"]


def test_flatten_excludes_pc_from_audiobox_ovl():
    """PC is extracted as a raw axis but EXCLUDED from audiobox_OVL (it inverts
    the quality signal — see retention docstring / mos/eval/FINDINGS.md)."""
    rec = {"cut_id": "c", "metrics": {"audiobox": {"score":
            {"CE": 3.0, "CU": 6.0, "PC": 9.0, "PQ": 9.0}}}}
    flat = _flatten_metrics(rec)
    assert flat["audiobox_PC"] == 9.0                       # raw axis still kept
    assert flat["audiobox_OVL"] == (3.0 + 6.0 + 9.0) / 3.0  # mean of CE/CU/PQ, no PC


def test_raw_score_native_scale():
    # aes_ovl maps to the "audiobox" weight key; raw uses native (un-z) scale.
    comp = {"utmos": 4.0, "dnsmos": 3.0, "aes_ovl": 5.0}
    axes = axes_for_mode("mos")
    w = {"utmos": 0.3, "dnsmos": 0.4, "audiobox": 0.2}
    assert abs(raw_score(comp, axes, w) - (0.3 * 4 + 0.4 * 3 + 0.2 * 5)) < 1e-9


def test_rank_is_mean_of_z():
    """rank_score = mean of active z; z-scoring equalizes axes of different scale
    so a high-variance axis can't dominate a low-variance one."""
    comps = [{"utmos": 3.00, "dnsmos": 1.0},
             {"utmos": 3.10, "dnsmos": 5.0}]
    axes = axes_for_mode("mos")
    stats = _component_stats(comps, axes)
    s0 = rank_score(comps[0], axes, stats, {})
    s1 = rank_score(comps[1], axes, stats, {})
    assert s1 > s0 and abs(s1 + s0) < 1e-9       # symmetric z (2 pts -> +/-1), mean
    assert abs(s1 - 1.0) < 1e-9                  # mean of (+1, +1)


def test_gate_uses_worst_axis():
    """gate_score = 0.5*mean + 0.5*min; a single bad axis pulls it below the mean
    so the worst failure mode is caught even if the average looks fine."""
    axes = axes_for_mode("mos")
    # corpus: one cut good on both, one bad only on dnsmos
    comps = [{"utmos": 4.0, "dnsmos": 4.0}, {"utmos": 4.0, "dnsmos": 1.0}]
    stats = _component_stats(comps, axes)
    g_bad = gate_score(comps[1], axes, stats, {})       # utmos z=0, dnsmos z=-1
    r_bad = rank_score(comps[1], axes, stats, {})        # mean = -0.5
    assert g_bad < r_bad                                 # min term pulls gate down
    assert abs(g_bad - (0.5 * -0.5 + 0.5 * -1.0)) < 1e-9


def test_mode_selects_axes():
    assert "rover" not in axes_for_mode("mos")
    assert axes_for_mode("asr") == ("rover",)
    assert "utmos" in axes_for_mode("both") and "rover" in axes_for_mode("both")


def test_load_quality_prefers_in_row_dataset(tmp_path=None):
    """M3: an explicit in-row ``dataset`` overrides the path-derived tag;
    rows without it (or with null) fall back to the path heuristic."""
    import json, tempfile
    d = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    rows = [
        {"cut_id": "c1", "dataset": "stage_2/granary_ytc/en",
         "metrics": {"utmos": {"score": {"utmos": 4.0}}}},
        {"cut_id": "c2", "metrics": {"utmos": {"score": {"utmos": 3.0}}}},
        {"cut_id": "c3", "dataset": None,
         "metrics": {"utmos": {"score": {"utmos": 2.0}}}},
    ]
    (d / "mos_rank_0000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    out = load_quality_for_datasets([d])
    assert ("stage_2/granary_ytc/en", "c1") in out      # explicit tag wins
    assert ("default", "c2") in out                      # path fallback (no field)
    assert ("default", "c3") in out                      # path fallback (null)
    assert out[("stage_2/granary_ytc/en", "c1")].get("utmos") == 4.0


if __name__ == "__main__":
    test_pivot_quality_union_and_nulls()
    test_pivot_quality_empty()
    test_selection_columns_flag_only()
    test_selection_columns_gate_flag()
    test_selection_columns_missing_floor()
    test_flatten_excludes_pc_from_audiobox_ovl()
    test_raw_score_native_scale()
    test_rank_is_mean_of_z()
    test_gate_uses_worst_axis()
    test_mode_selects_axes()
    test_load_quality_prefers_in_row_dataset()
    print("ALL PASS")
