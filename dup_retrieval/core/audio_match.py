"""Stage E — audio matching within text clusters (Hough 1D voting).

For each text cluster, run pairwise Hough voting on the Stage-D constellation
fingerprints.  Confirmed pairs feed a per-cluster union-find whose components
become the audio_cluster_id.

Multi-rank: each rank owns 1/world of the clusters (round-robin over the
largest-first order) and vectorised-loads only its clusters' fingerprints.
``run_rank`` writes ``part_{rank:04d}_clusters.parquet`` + a ``.done`` marker;
``merge`` (rank 0) re-bases the per-rank local ids into a single global
``audio_cluster_id`` and writes ``clusters.parquet`` + ``_SUCCESS``.

Thresholds:
- ``min_keypoints``    : pairs with too few keypoints are skipped.
- ``match_threshold = max(min_abs, frac * min(|fp_a|, |fp_b|))``.
- ``min_relative_match``: matched span (max-bin width / shorter clip dur) must
  be >= this fraction.

Cluster-size handling:
- ``size <= small_cluster_max``: full pairwise.
- up to ``huge_cluster_max``: shared-hash pre-filter -> sub-cluster components
  -> pairwise on each.
- ``size > huge_cluster_max``: flagged in ``huge_clusters.parquet``, not matched.

Inputs  : fingerprint/part_*.parquet (Stage D), text_dedup/clusters.parquet (Stage C)
Outputs : audio_match/clusters.parquet
              (dataset, cut_id, text_cluster_id, audio_cluster_id,
               max_match_score, matched_span_secs)
          audio_match/huge_clusters.parquet, audio_match/_SUCCESS
          (optional) audio_match/match_edges.parquet  (debug)
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"


# ---------------------------------------------------------------------------
# Cluster-level Hough matching primitives
# ---------------------------------------------------------------------------


def _hough_match_score(
    hashes_a: np.ndarray, t_a: np.ndarray,
    hashes_b: np.ndarray, t_b: np.ndarray,
    hist_tolerance: int = 0,
) -> Tuple[int, int, int]:
    """Return (max_bin_count, span_a_frames, span_b_frames).

    Build the time-offset histogram between matched hashes.  Each bin is one
    12.5 Hz frame (~80 ms); bins of width 1 capture exact alignment.  When
    ``hist_tolerance > 0`` we widen the count by a moving sum.

    Returns the maximum bin count *and* the time spans (in 12.5 Hz frames)
    over the matched signatures so the caller can compute matched_span /
    duration ratios.
    """
    if hashes_a.size == 0 or hashes_b.size == 0:
        return 0, 0, 0

    # Vectorized hash join: sort B's hashes once, then for every A-hash take the
    # [lo, hi) run of equal B-hashes and materialize all matched (t_a, t_b) pairs
    # with array ops (no Python per-match loop).  This yields the SAME multiset of
    # deltas as the dict approach, so max_bin / peak_delta / spans are identical.
    order_b = np.argsort(hashes_b, kind="stable")
    hb_s = hashes_b[order_b]
    tb_s = np.asarray(t_b)[order_b]
    lo = np.searchsorted(hb_s, hashes_a, side="left")
    hi = np.searchsorted(hb_s, hashes_a, side="right")
    cnt = hi - lo
    total = int(cnt.sum())
    if total == 0:
        return 0, 0, 0

    a_match = np.repeat(np.asarray(t_a), cnt)                  # t_a per match
    seg_start = np.repeat(lo, cnt)
    excl_prefix = np.repeat(np.cumsum(cnt) - cnt, cnt)         # segment start in flat array
    b_match = tb_s[seg_start + (np.arange(total) - excl_prefix)]   # t_b per match
    deltas = (a_match - b_match).astype(np.int64)

    unique, counts = np.unique(deltas, return_counts=True)
    if hist_tolerance > 0:
        # Windowed sum over the (sorted) delta histogram: total count within ±tol
        # of each bin.  Exact equivalent of the old moving-sum, vectorized.
        lo_w = np.searchsorted(unique, unique - hist_tolerance, side="left")
        hi_w = np.searchsorted(unique, unique + hist_tolerance, side="right")
        csum = np.concatenate(([0], np.cumsum(counts)))
        weight = csum[hi_w] - csum[lo_w]
        i_max = int(np.argmax(weight))
        max_bin = int(weight[i_max])
    else:
        i_max = int(np.argmax(counts))
        max_bin = int(counts[i_max])
    peak_delta = int(unique[i_max])

    # Span over matches at EXACTLY the peak delta (tolerance only widens the
    # count, not the span window — matches the original).
    on_peak = deltas == peak_delta
    if not on_peak.any():
        return max_bin, 0, 0
    a_on = a_match[on_peak]
    b_on = b_match[on_peak]
    span_a = int(a_on.max() - a_on.min() + 1)
    span_b = int(b_on.max() - b_on.min() + 1)
    return max_bin, span_a, span_b


def _shared_hash_count(hashes_a: np.ndarray, hashes_b: np.ndarray) -> int:
    """Cheap pre-filter: number of hashes A and B share (with multiplicity)."""
    if hashes_a.size == 0 or hashes_b.size == 0:
        return 0
    # multiset intersection sized by hash counts.
    a_unique, a_counts = np.unique(hashes_a, return_counts=True)
    b_unique, b_counts = np.unique(hashes_b, return_counts=True)
    common, ai, bi = np.intersect1d(a_unique, b_unique, return_indices=True,
                                    assume_unique=True)
    if common.size == 0:
        return 0
    return int(np.minimum(a_counts[ai], b_counts[bi]).sum())


# ---------------------------------------------------------------------------
# Per-cluster worker (multiprocessing-friendly)
# ---------------------------------------------------------------------------


# Fingerprint tables are large and shared read-only by every cluster task, so we
# load them into each worker process ONCE via the pool initializer instead of
# pickling them with every submit (the latter is O(n_clusters * corpus_fp_size)
# and stalls on dense data — 43k clusters x the full fingerprint set).
_GFP_HASHES: Optional[List[np.ndarray]] = None
_GFP_TIMES:  Optional[List[np.ndarray]] = None
_GFP_DUR:    Optional[np.ndarray] = None


def _pool_init(fp_hashes, fp_times, fp_durations) -> None:
    global _GFP_HASHES, _GFP_TIMES, _GFP_DUR
    _GFP_HASHES, _GFP_TIMES, _GFP_DUR = fp_hashes, fp_times, fp_durations


def _process_one_cluster(
    cluster_id: int,
    members: List[int],                  # global row indices into the fp tables
    cfg_e: dict,
) -> dict:
    """Cluster the members of one text_cluster into audio_cluster ids.

    Reads the shared fingerprint tables from process globals (set by
    ``_pool_init``); ``members`` indexes into them.
    """
    fp_hashes, fp_times, fp_durations = _GFP_HASHES, _GFP_TIMES, _GFP_DUR
    min_keypoints       = int(cfg_e.get("min_keypoints", 4))
    match_min_abs       = int(cfg_e.get("match_min_abs", 4))
    match_frac          = float(cfg_e.get("match_frac", 0.30))
    min_rel_match       = float(cfg_e.get("min_relative_match", 0.30))
    hist_tolerance      = int(cfg_e.get("hist_tolerance", 0))
    small_cluster_max   = int(cfg_e.get("small_cluster_max", 200))
    huge_cluster_max    = int(cfg_e.get("huge_cluster_max", 10_000))
    pre_filter_min      = int(cfg_e.get("pre_filter_min_shared", 4))
    # When at least this fraction of the shorter clip's keypoints align at one
    # offset, the match is treated as a whole-clip duplicate and bypasses the
    # relative-span gate (sparse fingerprints can be temporally clustered, so an
    # identical clip may legitimately have a small matched span).
    strong_match_frac   = float(cfg_e.get("strong_match_frac", 0.85))
    save_match_edges    = bool(cfg_e.get("save_match_edges", False))

    sz = len(members)
    out = {
        "cluster_id":      int(cluster_id),
        "members":         members,
        "size":            sz,
        "is_huge":         False,
        "audio_cluster":   {},      # row_idx -> sub-cluster id (within this cluster)
        "max_match_score": {},      # row_idx -> int
        "matched_span":    {},      # row_idx -> int (12.5 Hz frames, longest pair)
        "edges":           [],      # only when save_match_edges
        "stats":           {"pairs_checked": 0, "pairs_confirmed": 0,
                            "pre_filtered": 0},
    }

    if sz > huge_cluster_max:
        out["is_huge"] = True
        return out

    # Decide which pairs to verify.
    pairs_to_verify: List[Tuple[int, int]] = []
    if sz <= small_cluster_max:
        for i in range(sz):
            for j in range(i + 1, sz):
                pairs_to_verify.append((i, j))
    else:
        # Pre-filter: shared-hash count -> seed sub-clusters via union-find, then
        # verify only pairs within seeded sub-clusters.  Inline a tiny UF here to
        # avoid a cross-module import in the workers.
        parent = list(range(sz))
        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        # Cheap pre-filter on shared-hash count.
        for i in range(sz):
            for j in range(i + 1, sz):
                shared = _shared_hash_count(fp_hashes[members[i]],
                                            fp_hashes[members[j]])
                if shared < pre_filter_min:
                    out["stats"]["pre_filtered"] += 1
                    continue
                _union(i, j)
        # Build sub-clusters and run pairwise within each (capped at small_cluster_max).
        sub: Dict[int, List[int]] = {}
        for i in range(sz):
            sub.setdefault(_find(i), []).append(i)
        for sub_members in sub.values():
            if len(sub_members) > small_cluster_max:
                # Even the sub-cluster is too big — flag, but still process
                # (huge_cluster_max should have caught the whole cluster).
                logger.warning("Sub-cluster of size %d in cluster %d exceeds "
                               "small_cluster_max; processing anyway.",
                               len(sub_members), cluster_id)
            for ii in range(len(sub_members)):
                for jj in range(ii + 1, len(sub_members)):
                    pairs_to_verify.append((sub_members[ii], sub_members[jj]))

    # Confirm pairs via Hough voting + adaptive threshold + span check.
    parent = list(range(sz))
    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    max_score: Dict[int, int]   = {i: 0 for i in range(sz)}
    max_span:  Dict[int, int]   = {i: 0 for i in range(sz)}

    HASH_FRAME_RATE = float(cfg_e.get("hash_frame_rate", 12.5))

    for i, j in pairs_to_verify:
        a_idx = members[i]; b_idx = members[j]
        ha = fp_hashes[a_idx]; ta = fp_times[a_idx]
        hb = fp_hashes[b_idx]; tb = fp_times[b_idx]
        if min(ha.size, hb.size) < min_keypoints:
            continue
        out["stats"]["pairs_checked"] += 1

        threshold = max(match_min_abs, int(match_frac * min(ha.size, hb.size)))
        max_bin, span_a, span_b = _hough_match_score(ha, ta, hb, tb, hist_tolerance)
        if max_bin < threshold:
            continue

        # Relative span check: matched span (in seconds) must be at least
        # min_relative_match * shorter clip duration -- UNLESS the match is
        # near-total (most of the shorter clip's keypoints align at one offset),
        # which is an unambiguous whole-clip duplicate even when its sparse
        # keypoints are temporally concentrated.  Without this bypass, identical
        # clips with few, clustered keypoints fail the span gate.
        dur_a = float(fp_durations[a_idx])
        dur_b = float(fp_durations[b_idx])
        shorter_dur = min(dur_a, dur_b) if dur_a > 0 and dur_b > 0 else 0.0
        matched_span_secs = max(span_a, span_b) / HASH_FRAME_RATE
        match_fraction = max_bin / max(1, min(ha.size, hb.size))
        if (shorter_dur > 0 and match_fraction < strong_match_frac
                and (matched_span_secs / shorter_dur) < min_rel_match):
            continue

        _union(i, j)
        out["stats"]["pairs_confirmed"] += 1

        max_score[i] = max(max_score[i], max_bin)
        max_score[j] = max(max_score[j], max_bin)
        max_span[i]  = max(max_span[i],  max(span_a, span_b))
        max_span[j]  = max(max_span[j],  max(span_a, span_b))

        if save_match_edges:
            out["edges"].append((a_idx, b_idx, int(max_bin), float(matched_span_secs)))

    # Component map: row_idx -> sub-cluster id.
    sub_id = {}
    next_id = 0
    for i in range(sz):
        r = _find(i)
        if r not in sub_id:
            sub_id[r] = next_id
            next_id += 1
        out["audio_cluster"][members[i]] = sub_id[r]
        out["max_match_score"][members[i]] = max_score[i]
        out["matched_span"][members[i]]    = max_span[i]
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _atomic_write_parquet(out_path: Path, table: pa.Table) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, str(tmp), compression="zstd", compression_level=3,
                   row_group_size=200_000)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)


def _verify_upstream(output_dir: Path) -> None:
    if not (output_dir / "text_dedup" / SUCCESS_MARKER).exists():
        raise RuntimeError("Stage C not finalized.")
    if not (output_dir / "fingerprint" / SUCCESS_MARKER).exists():
        raise RuntimeError("Stage D not finalized.")


def _load_fingerprints_filtered(fp_dir: Path, wanted_cut_ids: set) -> dict:
    """Vectorised filtered load. Keeps only rows whose cut_id is in *wanted_cut_ids*
    (and drops null/sentinel rows), then groups per (dataset, cut_id) by SORTING:
    one stable argsort reorders hash and t_hash by the SAME permutation, so each
    keypoint's (hash, time) stays paired (the v3-first bug was two independent
    group_by('list') aggregations whose intra-group order was not co-ordered).
    Per-cut arrays are then O(1) slices of the sorted buffers."""
    parts = sorted(fp_dir.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError(f"No fingerprint parts in {fp_dir}")
    wanted_arr = pa.array(sorted(wanted_cut_ids), type=pa.string())
    tables = []
    for p in parts:
        t = pq.read_table(p, columns=["dataset", "cut_id", "hash", "t_hash"])
        mask = pc.and_(pc.is_in(t.column("cut_id"), value_set=wanted_arr),
                       pc.and_(pc.is_valid(t.column("hash")),
                               pc.greater_equal(t.column("t_hash"),
                                                pa.scalar(0, pa.int64()))))
        tf = t.filter(mask)
        if tf.num_rows:
            tables.append(tf)
    if not tables:
        return {"row_keys": [], "key_to_row": {}, "fp_hashes": [], "fp_times": []}
    full = pa.concat_tables(tables)

    # integer codes per (dataset, cut_id) so the sort + boundary scan are numeric.
    # Cast strings to large_string before combining: 540M rows overflow the int32
    # offsets of the default `string` type on combine_chunks.
    cid_arr = full.column("cut_id").cast(pa.large_string()).combine_chunks()
    ds_arr = full.column("dataset").cast(pa.large_string()).combine_chunks()
    cid_dict = pc.dictionary_encode(cid_arr)
    ds_dict = pc.dictionary_encode(ds_arr)
    cut_codes = cid_dict.indices.to_numpy().astype(np.int64)
    ds_codes = ds_dict.indices.to_numpy().astype(np.int64)
    n_ds = len(ds_dict.dictionary)
    combined = cut_codes * (n_ds + 1) + ds_codes        # unique per (cut, dataset)

    hash_np = np.asarray(full.column("hash").combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
    time_np = np.asarray(full.column("t_hash").combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)

    order = np.argsort(combined, kind="stable")
    sc = combined[order]
    sh = hash_np[order]                                  # hash and time reordered
    st = time_np[order]                                  # by the SAME permutation
    bnd = np.flatnonzero(np.diff(sc)) + 1
    starts = np.concatenate(([0], bnd))
    ends = np.concatenate((bnd, [len(sc)]))
    n = len(starts)

    fp_hashes = [sh[starts[i]:ends[i]] for i in range(n)]
    fp_times = [st[starts[i]:ends[i]] for i in range(n)]

    group_combined = sc[starts]
    group_cut_code = (group_combined // (n_ds + 1))
    group_ds_code = (group_combined % (n_ds + 1))
    cut_at = cid_dict.dictionary.take(pa.array(group_cut_code)).to_pylist()
    ds_at = ds_dict.dictionary.take(pa.array(group_ds_code)).to_pylist()
    row_keys = list(zip(ds_at, cut_at))
    return {
        "row_keys": row_keys,
        "key_to_row": {k: i for i, k in enumerate(row_keys)},
        "fp_hashes": fp_hashes,
        "fp_times": fp_times,
    }


def _durations(output_dir: Path, fp: dict) -> np.ndarray:
    wanted = set(fp["row_keys"])
    dur_map = {}
    for p in sorted((output_dir / "manifest").glob("part_*.parquet")):
        t = pq.read_table(p, columns=["dataset", "cut_id", "duration_secs"])
        for d, c, du in zip(t.column("dataset").to_pylist(),
                            t.column("cut_id").to_pylist(),
                            t.column("duration_secs").to_pylist()):
            if (d, c) in wanted:
                dur_map[(d, c)] = float(du) if du is not None else 0.0
    arr = np.zeros(len(fp["row_keys"]), dtype=np.float64)
    for i, key in enumerate(fp["row_keys"]):
        arr[i] = dur_map.get(key, 0.0)
    return arr


def run_rank(cfg: dict, rank: int, world: int) -> None:
    output_dir = Path(cfg["output_dir"])
    stage_dir = output_dir / "audio_match"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _verify_upstream(output_dir)

    cfg_e = cfg.get("audio_match", {})
    hfr = float((cfg.get("audio_fingerprint") or {}).get("hash_frame_rate", 12.5))
    cfg_e = {**cfg_e, "hash_frame_rate": hfr}
    save_edges = bool(cfg_e.get("save_match_edges", False))
    t0 = time.time()

    tc = pq.read_table(output_dir / "text_dedup" / "clusters.parquet",
                       columns=["dataset", "cut_id", "text_cluster_id"])
    cluster_keys = {}
    for d, c, cl in zip(tc.column("dataset").to_pylist(),
                        tc.column("cut_id").to_pylist(),
                        tc.column("text_cluster_id").to_pylist()):
        cluster_keys.setdefault(cl, []).append((d, c))
    sorted_clusters = sorted(cluster_keys.items(), key=lambda kv: -len(kv[1]))
    mine = [cm for i, cm in enumerate(sorted_clusters) if i % world == rank]
    wanted = {c for _, keys in mine for (_, c) in keys}
    logger.info("[rank %d/%d] %d of %d clusters assigned, %d cuts to load (%.0fs)",
                rank, world, len(mine), len(sorted_clusters), len(wanted), time.time() - t0)

    fp = _load_fingerprints_filtered(output_dir / "fingerprint", wanted)
    fp_dur = _durations(output_dir, fp)
    logger.info("[rank %d] loaded %d cuts' fingerprints (VECTORISED) (%.0fs)",
                rank, len(fp["row_keys"]), time.time() - t0)

    cluster_to_rows = {}
    for cl, keys in mine:
        rows = [fp["key_to_row"][k] for k in keys if k in fp["key_to_row"]]
        if rows:
            cluster_to_rows[cl] = rows
    mine_items = sorted(cluster_to_rows.items(), key=lambda kv: -len(kv[1]))
    total = len(mine_items)

    workers = int(cfg_e.get("workers", min(32, os.cpu_count() or 4)))
    results = []
    done = 0
    t1 = time.time()
    if workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_pool_init,
                                 initargs=(fp["fp_hashes"], fp["fp_times"], fp_dur)) as pool:
            futs = [pool.submit(_process_one_cluster, cl, mem, cfg_e) for cl, mem in mine_items]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 2000 == 0 or done == total:
                    el = time.time() - t1
                    logger.info("[rank %d] %d/%d clusters (%.0f%%, %.0fs, %.0f cl/s)",
                                rank, done, total, 100.0 * done / total, el,
                                done / el if el > 0 else 0)
    else:
        _pool_init(fp["fp_hashes"], fp["fp_times"], fp_dur)
        for cl, mem in mine_items:
            results.append(_process_one_cluster(cl, mem, cfg_e))
            done += 1

    out = {k: [] for k in ("ds", "cid", "tcid", "acid", "score", "span")}
    huge, edges, next_local = [], [], 0
    for r in results:
        cl = r["cluster_id"]
        if r["is_huge"]:
            for ri in r["members"]:
                ds, cid = fp["row_keys"][ri]
                huge.append((ds, cid, cl, len(r["members"])))
            continue
        l2g = {}
        for ri, sub in r["audio_cluster"].items():
            if sub not in l2g:
                l2g[sub] = next_local
                next_local += 1
            ds, cid = fp["row_keys"][ri]
            out["ds"].append(ds); out["cid"].append(cid); out["tcid"].append(int(cl))
            out["acid"].append(int(l2g[sub]))
            out["score"].append(int(r["max_match_score"][ri]))
            out["span"].append(float(r["matched_span"][ri]) / hfr)
        if save_edges and r["edges"]:
            for a, b, mb, ms in r["edges"]:
                a_ds, a_cid = fp["row_keys"][a]
                b_ds, b_cid = fp["row_keys"][b]
                edges.append((a_ds, a_cid, b_ds, b_cid, int(mb), float(ms)))

    _atomic_write_parquet(stage_dir / f"part_{rank:04d}_clusters.parquet", pa.table({
        "dataset": pa.array(out["ds"], pa.string()),
        "cut_id": pa.array(out["cid"], pa.string()),
        "text_cluster_id": pa.array(out["tcid"], pa.int64()),
        "audio_cluster_local": pa.array(out["acid"], pa.int64()),
        "max_match_score": pa.array(out["score"], pa.int32()),
        "matched_span_secs": pa.array(out["span"], pa.float32()),
    }))
    if huge:
        _atomic_write_parquet(stage_dir / f"part_{rank:04d}_huge.parquet", pa.table({
            "dataset": pa.array([h[0] for h in huge], pa.string()),
            "cut_id": pa.array([h[1] for h in huge], pa.string()),
            "text_cluster_id": pa.array([h[2] for h in huge], pa.int64()),
            "text_cluster_size": pa.array([h[3] for h in huge], pa.int64()),
        }))
    if save_edges and edges:
        _atomic_write_parquet(stage_dir / f"part_{rank:04d}_edges.parquet", pa.table({
            "src_dataset": pa.array([e[0] for e in edges], pa.string()),
            "src_cut_id": pa.array([e[1] for e in edges], pa.string()),
            "dst_dataset": pa.array([e[2] for e in edges], pa.string()),
            "dst_cut_id": pa.array([e[3] for e in edges], pa.string()),
            "match_score": pa.array([e[4] for e in edges], pa.int32()),
            "matched_span_secs": pa.array([e[5] for e in edges], pa.float32()),
        }))
    (stage_dir / f"part_{rank:04d}.done").write_text(
        json.dumps({"rank": rank, "rows": len(out["ds"]),
                    "local_clusters": next_local, "huge_rows": len(huge)}))
    logger.info("[rank %d] DONE: %d rows, %d local audio-clusters, %d huge rows (%.0fs total)",
                rank, len(out["ds"]), next_local, len(huge), time.time() - t0)


def merge(cfg: dict, world: int) -> None:
    stage_dir = Path(cfg["output_dir"]) / "audio_match"
    cl_tables, edge_tables, huge_tables = [], [], []
    offset, total_clusters = 0, 0
    for rank in range(world):
        done = stage_dir / f"part_{rank:04d}.done"
        if not done.exists():
            raise RuntimeError(f"rank {rank} part not finished")
        expected_rows = int(json.loads(done.read_text())["rows"])
        p = stage_dir / f"part_{rank:04d}_clusters.parquet"
        if not p.exists():
            raise RuntimeError(f"rank {rank}: {p.name} missing "
                               f"(.done reports {expected_rows} rows)")
        t = pq.read_table(p)
        if t.num_rows != expected_rows:
            raise RuntimeError(f"rank {rank}: {p.name} has {t.num_rows} rows, "
                               f".done reports {expected_rows}")
        local = t.column("audio_cluster_local").to_numpy()
        n_local = int(local.max()) + 1 if len(local) else 0
        cl_tables.append(pa.table({
            "dataset": t.column("dataset"),
            "cut_id": t.column("cut_id"),
            "text_cluster_id": t.column("text_cluster_id"),
            "audio_cluster_id": pa.array(local + offset, pa.int64()),
            "max_match_score": t.column("max_match_score"),
            "matched_span_secs": t.column("matched_span_secs"),
        }))
        offset += n_local
        total_clusters += n_local
        ep = stage_dir / f"part_{rank:04d}_edges.parquet"
        if ep.exists():
            edge_tables.append(pq.read_table(ep))
        hp = stage_dir / f"part_{rank:04d}_huge.parquet"
        if hp.exists():
            huge_tables.append(pq.read_table(hp))
    clusters = pa.concat_tables(cl_tables) if cl_tables else None
    _atomic_write_parquet(stage_dir / "clusters.parquet", clusters)
    if edge_tables:
        _atomic_write_parquet(stage_dir / "match_edges.parquet", pa.concat_tables(edge_tables))
    huge_rows = 0
    if huge_tables:
        ht = pa.concat_tables(huge_tables)
        huge_rows = ht.num_rows
        _atomic_write_parquet(stage_dir / "huge_clusters.parquet", ht)
    from . import run_layout
    run_layout.finalize_stage(stage_dir.parent, "audio_match",
                              rows=clusters.num_rows if clusters is not None else 0,
                              extra={"backend": "multi_rank", "world": world,
                                     "audio_clusters": total_clusters,
                                     "huge_cluster_rows": huge_rows})
    logger.info("MERGE done: %d rows, %d audio clusters, %d huge rows",
                clusters.num_rows if clusters is not None else 0, total_clusters, huge_rows)


def _load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--world", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _load_cfg(args.config)
    if args.merge:
        merge(cfg, args.world or int(os.environ.get("SLURM_NTASKS", "1")))
    else:
        run_rank(cfg, int(os.environ.get("SLURM_PROCID", "0")),
                 int(os.environ.get("SLURM_NTASKS", "1")))


if __name__ == "__main__":
    main()
