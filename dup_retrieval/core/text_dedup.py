"""Stage C — fuzzy text dedup (NeMo-Curator-inspired, verified-pair single-linkage).

GPU backends only (the container always provides RAPIDS):
  backend: cudf       single-GPU cuDF MinHash + LSH groupby + chunked GPU
                      Jaccard verify + connected components (cluster default).
  backend: dask_cudf  multi-GPU dask-cuda for larger corpora.

Inputs  : <output_dir>/manifest/part_*.parquet
Outputs : <output_dir>/text_dedup/clusters.parquet
          (dataset, cut_id, text_cluster_id, cluster_size; size>1 rows only)
          <output_dir>/text_dedup/_SUCCESS
          (optional, save_candidate_edges) candidate_edges.parquet

Composite primary key: (dataset, cut_id) everywhere.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SUCCESS_MARKER = "_SUCCESS"

# 64-bit Mersenne prime
MERSENNE61 = (1 << 61) - 1
HASH_MASK_32 = (1 << 32) - 1


# ---------------------------------------------------------------------------
# MinHash core
# ---------------------------------------------------------------------------


def _stable_hash64(b: bytes) -> int:
    """Stable 64-bit hash of bytes.

    We use blake2b(digest_size=8) so signatures are reproducible across
    Python startups (Python's built-in hash() is randomized via PYTHONHASHSEED).
    """
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big",
                          signed=False)


# Sentinel used to right-pad texts shorter than the n-gram width so they still
# produce a (single) shingle instead of being dropped.  Must not occur in
# normalized_text; NUL never does.
_PAD_CHAR = "\x00"


def _ngrams_chars(text: str, n: int) -> np.ndarray:
    """Char n-grams of *text* hashed to uint64 via blake2b (stable across runs).

    For multilingual text, character n-grams are robust to whitespace
    conventions (CJK has none).  Text shorter than *n* is right-padded with
    ``_PAD_CHAR`` to a single n-gram so identical short utterances still collide
    (and cluster) instead of being silently dropped; different short texts pad
    to different shingles.  Empty text -> empty array.
    """
    if not text:
        return np.zeros(0, dtype=np.uint64)
    if len(text) < n:
        text = text.ljust(n, _PAD_CHAR)
    L = len(text) - n + 1
    out = np.empty(L, dtype=np.uint64)
    # Pure-Python loop; for very large corpora replace with a compiled kernel
    # (xxhash is the obvious 5-10x speedup).  Correctness is the same.
    for i in range(L):
        out[i] = _stable_hash64(text[i:i + n].encode("utf-8"))
    return out


# ---------------------------------------------------------------------------
# Exact-Jaccard candidate verification (B2)
# ---------------------------------------------------------------------------
#
# The MinHash signature only ESTIMATES Jaccard (std ~ sqrt(j(1-j)/K)); near the
# threshold this noise both invents false near-dups and drops true ones.  When
# ``text_dedup.exact_jaccard_verify`` is on, LSH still generates candidates but
# the final keep/drop is decided by the TRUE char-n-gram Jaccard, recomputed on
# the (small, LSH-prefiltered) candidate set in parallel across CPU workers.

# Shingle table, set in the parent before forking the verify pool so fork workers
# inherit it COW instead of it being pickled per task (the Stage-E lesson).
_EXACT_SHINGLES: Optional[Dict[int, frozenset]] = None


def _verify_chunk_lowmem(args):
    """Worker: exact char-n-gram similarity for a chunk of (a,b) row-index pairs.
    Builds shingle SETs for ONLY this chunk\'s rows (passed in ``row_texts``) — no
    shared/forked global table, so fork never COW-copies a multi-GB shingle map
    (the thing that made the verify thrash/swap at 10M+ pairs and use 1 node).
    metric: "jaccard"=|A∩B|/|A∪B|; "containment"=|A∩B|/min(|A|,|B|) (the permissive
    method). ``min_shared`` drops pairs sharing fewer than that many shingles."""
    pa_, pb_, row_texts, ngram_size, threshold, metric, min_shared = args
    sh = {r: frozenset(_ngrams_chars(t, ngram_size).tolist()) for r, t in row_texts.items()}
    n = len(pa_)
    keep = np.zeros(n, dtype=bool)
    score = np.zeros(n, dtype=np.float32)
    cont = (metric == "containment")
    for k in range(n):
        A = sh.get(int(pa_[k])); B = sh.get(int(pb_[k]))
        if not A or not B:
            continue
        inter = len(A & B)
        if inter == 0 or inter < min_shared:
            continue
        s = inter / min(len(A), len(B)) if cont else inter / (len(A) + len(B) - inter)
        score[k] = s
        if s >= threshold:
            keep[k] = True
    return keep, score


def exact_jaccard_verify(ea: np.ndarray, eb: np.ndarray, texts: List[str],
                         ngram_size: int, threshold: float,
                         workers: int, metric: str = "jaccard",
                         min_shared: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Filter candidate row-index pairs by EXACT char-n-gram similarity (Jaccard
    OR containment), in parallel.  The containment/"permissive" method routes
    through here (it forces this verify).  Each worker builds shingle sets for
    ONLY its own pair-chunk\'s rows, so there is no giant forked global table —
    it scales to 10M+ pairs without COW memory thrash and saturates all
    ``workers`` cores (set exact_verify_workers to the node core count, e.g. 288).
    Identical result to the previous global-table implementation.
    """
    ea = np.asarray(ea); eb = np.asarray(eb)
    n = int(ea.shape[0])
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    workers = max(1, int(workers))
    nchunks = workers if (n >= 2 * workers) else 1
    args = []
    for c in np.array_split(np.arange(n), nchunks):
        if not len(c):
            continue
        eac = ea[c]; ebc = eb[c]
        rows = np.unique(np.concatenate([eac, ebc]))
        row_texts = {int(r): texts[int(r)] for r in rows}
        args.append((eac, ebc, row_texts, ngram_size, threshold, metric, min_shared))
    if len(args) == 1:
        results = [_verify_chunk_lowmem(args[0])]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(len(args)) as pool:
            results = pool.map(_verify_chunk_lowmem, args)
    keep = np.concatenate([r[0] for r in results])
    jac = np.concatenate([r[1] for r in results])
    return keep, jac


def _dedup_pairs(ea: np.ndarray, eb: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Canonicalize (min,max) and drop duplicate candidate pairs.  N < ~3e9 so
    the lo*N+hi key fits int64."""
    ea = np.asarray(ea, dtype=np.int64); eb = np.asarray(eb, dtype=np.int64)
    if ea.size == 0:
        return ea, eb
    lo = np.minimum(ea, eb); hi = np.maximum(ea, eb)
    _, idx = np.unique(lo * np.int64(N) + hi, return_index=True)
    return lo[idx], hi[idx]


def _texts_for_rows(manifest_dir: Path, ds_all: List[str],
                    cid_all: List[str]) -> List[str]:
    """normalized_text aligned with the signature row order (ds_all/cid_all)."""
    text_by_key: Dict[Tuple[str, str], str] = {}
    for p in sorted(manifest_dir.glob("part_*.parquet")):
        t = pq.read_table(p, columns=["dataset", "cut_id", "normalized_text"])
        for d, c, x in zip(t.column("dataset").to_pylist(),
                            t.column("cut_id").to_pylist(),
                            t.column("normalized_text").to_pylist()):
            text_by_key[(d, c)] = x or ""
    return [text_by_key.get((ds_all[i], cid_all[i]), "") for i in range(len(ds_all))]


# ---------------------------------------------------------------------------
# Stage C.1 — chunked MinHash compute
# ---------------------------------------------------------------------------


def _atomic_write_parquet(out_path: Path, table: pa.Table) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(table, str(tmp), compression="zstd", compression_level=3,
                   row_group_size=200_000)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)


def _cleanup_tmp(d: Path) -> None:
    for p in d.glob("*.parquet.tmp"):
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Stage C orchestration
# ---------------------------------------------------------------------------


def _verify_upstream(manifest_dir: Path) -> None:
    if not (manifest_dir / SUCCESS_MARKER).exists():
        raise RuntimeError(
            f"Upstream Stage A not done: missing {manifest_dir/SUCCESS_MARKER}.")


def _finalize_and_write(stage_dir: Path, ds_all: List[str], cid_all: List[str],
                        roots, cfg_c: dict, edge_log: Optional[List[tuple]],
                        t0: float, backend: str) -> None:
    """Renumber components, enforce the candidate-fraction guard, and write
    clusters.parquet / candidate_edges.parquet / _SUCCESS.

    ``roots`` is an int array (len N) where members of the same cluster share a
    value (union-find root or connected-components label).  Singletons keep a
    unique value and are dropped (text_clusters holds size>1 rows only).
    """
    n_perms                = int(cfg_c.get("num_permutations", 256))
    num_bands              = int(cfg_c.get("num_bands", 20))
    band_width             = int(cfg_c.get("band_width", 13))
    jaccard_threshold      = float(cfg_c.get("jaccard_threshold", 0.80))
    small_bucket_threshold = int(cfg_c.get("small_bucket_threshold", 200))
    max_bucket_size        = int(cfg_c.get("max_bucket_size", 10_000))
    candidate_fraction_max = float(cfg_c.get("candidate_fraction_max", 0.30))
    save_candidate_edges   = bool(cfg_c.get("save_candidate_edges", False))

    roots = np.asarray(roots)
    N = int(roots.shape[0])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    inverse = inverse.reshape(-1)
    cluster_sizes = np.bincount(inverse)
    cluster_ids = np.where(cluster_sizes[inverse] > 1, inverse, -1)

    # Write clusters.parquet (only rows in clusters of size > 1).
    keep_mask = cluster_ids >= 0
    n_clustered = int(keep_mask.sum())
    candidate_fraction = n_clustered / max(N, 1)
    logger.info("Stage C (%s): %d cuts in clusters (size>1); candidate_fraction=%.4f",
                backend, n_clustered, candidate_fraction)
    if candidate_fraction > candidate_fraction_max:
        raise RuntimeError(
            f"candidate_fraction={candidate_fraction:.3f} > candidate_fraction_max="
            f"{candidate_fraction_max:.3f}.  Inspect why text dedup is so lenient "
            "before paying audio decode.  Tune ngram_size / jaccard_threshold / "
            "max_bucket_size, or raise candidate_fraction_max if intentional.")

    out_size = cluster_sizes[inverse]
    text_clusters_path = stage_dir / "clusters.parquet"
    schema = pa.schema([
        pa.field("dataset",          pa.string(), nullable=False),
        pa.field("cut_id",           pa.string(), nullable=False),
        pa.field("text_cluster_id",  pa.int64(),  nullable=False),
        pa.field("cluster_size",     pa.int64(),  nullable=False),
    ])
    rows_idx = np.flatnonzero(keep_mask)
    table = pa.Table.from_arrays(
        [
            pa.array([ds_all[i]  for i in rows_idx]),
            pa.array([cid_all[i] for i in rows_idx]),
            pa.array(cluster_ids[rows_idx].astype(np.int64), type=pa.int64()),
            pa.array(out_size[rows_idx].astype(np.int64),    type=pa.int64()),
        ],
        schema=schema,
    )
    _atomic_write_parquet(text_clusters_path, table)

    if save_candidate_edges and edge_log is not None:
        edges_path = stage_dir / "candidate_edges.parquet"
        edge_schema = pa.schema([
            pa.field("src_dataset", pa.string()), pa.field("src_cut_id", pa.string()),
            pa.field("dst_dataset", pa.string()), pa.field("dst_cut_id", pa.string()),
            pa.field("jaccard",      pa.float32()),
        ])
        if edge_log:
            sd, sc, dd, dc, jv = [], [], [], [], []
            for a, b, jacc in edge_log:
                sd.append(ds_all[a]); sc.append(cid_all[a])
                dd.append(ds_all[b]); dc.append(cid_all[b])
                jv.append(float(jacc))
            etbl = pa.Table.from_arrays(
                [pa.array(sd), pa.array(sc), pa.array(dd), pa.array(dc),
                 pa.array(jv, type=pa.float32())],
                schema=edge_schema,
            )
        else:
            etbl = pa.Table.from_arrays(
                [pa.array([], type=f.type) for f in edge_schema], schema=edge_schema)
        _atomic_write_parquet(edges_path, etbl)
        logger.info("Wrote %d candidate edges to %s", len(edge_log), edges_path.name)

    elapsed = time.time() - t0
    n_clusters = int((cluster_sizes > 1).sum())
    logger.info("Stage C: %d cuts in %d clusters (%.1fs)", n_clustered, n_clusters, elapsed)

    from . import run_layout
    run_layout.finalize_stage(stage_dir.parent, "text_dedup", rows=n_clustered, extra={
        "backend":            backend,
        "clusters":           n_clusters,
        "candidate_fraction": candidate_fraction,
        "config_used":        {
            "num_permutations":       n_perms,
            "num_bands":              num_bands,
            "band_width":             band_width,
            "jaccard_threshold":      jaccard_threshold,
            "small_bucket_threshold": small_bucket_threshold,
            "max_bucket_size":        max_bucket_size,
        },
    })
    logger.info("Wrote %s", stage_dir / SUCCESS_MARKER)


def _maybe_init_rmm_pool(cfg_c: dict) -> None:
    """Switch cudf/cupy onto an RMM pool allocator to avoid cudaMalloc churn
    during the LSH groupby / Jaccard verify.  Best-effort; no-op on failure or
    when ``text_dedup.rmm_pool`` is false."""
    if not bool(cfg_c.get("rmm_pool", True)):
        return
    try:
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        import cupy as cp
        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=bool(cfg_c.get("rmm_managed_memory", False)),
            initial_pool_size=cfg_c.get("rmm_initial_pool_size"),
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
        logger.info("Stage C: RMM pool allocator enabled (managed_memory=%s).",
                    bool(cfg_c.get("rmm_managed_memory", False)))
    except Exception as exc:                                   # pragma: no cover
        logger.warning("Stage C: RMM pool init failed (%s); default allocator.",
                       repr(exc)[:100])


def _release_rmm_pool(cfg_c: dict) -> None:
    """Free the RMM pool created by :func:`_maybe_init_rmm_pool`.

    The pipeline runs all stages in ONE process, so the Stage-C RMM pool (which
    grows to tens of GB on large manifests) would otherwise stay resident on the
    GPU and starve the next GPU stage (audio_fingerprint → CUDA OOM). Reverting
    to the default allocator releases the pool without throttling anything.
    No-op when ``rmm_pool`` is off. Best-effort."""
    if not bool(cfg_c.get("rmm_pool", True)):
        return
    try:
        import gc
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.set_allocator(None)        # detach the RMM allocator from cupy
        except Exception:
            pass
        import rmm
        rmm.reinitialize(pool_allocator=False)  # revert to cudaMalloc → frees the pool
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Stage C: RMM pool released (GPU freed for downstream stages).")
    except Exception as exc:                                   # pragma: no cover
        logger.warning("Stage C: RMM pool release failed (%s).", repr(exc)[:100])


def _connected_components_from_edges(ea: np.ndarray, eb: np.ndarray, N: int,
                                     cfg_c: dict) -> np.ndarray:
    """Connected components over an edge list.  Uses single-GPU cuGraph when
    importable and ``use_cugraph_cc`` is set (post-rebuild), else scipy on CPU.

    Returns ``roots[N]`` where same-component nodes share a value.  cuGraph
    component labels are shifted by +N so they never collide with the singleton
    ids of edge-less nodes.
    """
    if len(ea) == 0:
        return np.arange(N, dtype=np.int64)
    if bool(cfg_c.get("use_cugraph_cc", True)):
        try:
            import cudf
            import cugraph
            g = cugraph.Graph(directed=False)
            g.from_cudf_edgelist(
                cudf.DataFrame({"src": ea, "dst": eb}),
                source="src", destination="dst", renumber=True)
            comp = cugraph.connected_components(g)
            roots = np.arange(N, dtype=np.int64)
            roots[comp["vertex"].to_numpy()] = N + comp["labels"].to_numpy()
            logger.info("Stage C: connected components via cuGraph.")
            return roots
        except Exception as exc:
            logger.warning("Stage C: cuGraph CC unavailable (%s); using scipy.",
                           repr(exc)[:90])
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(ea), dtype=np.int8), (ea, eb)), shape=(N, N))
    _, roots = connected_components(graph, directed=False)
    return roots


def _run_cudf_backend(manifest_dir: Path, cfg_c: dict):
    """Single-GPU Stage C: cudf MinHash + LSH banding + chunked GPU Jaccard
    verify + connected components (cuGraph if available, else scipy).

    Returns ``(ds_all, cid_all, roots, edge_log)``.  Holds the full N x K
    signature table on one GPU, so it targets up to ~50M cuts on a 120 GB
    GH200; beyond that, use the dask_cudf backend.
    """
    import cudf
    import cupy as cp

    _maybe_init_rmm_pool(cfg_c)

    n_perms                = int(cfg_c.get("num_permutations", 256))
    ngram_size             = int(cfg_c.get("ngram_size", 24))
    num_bands              = int(cfg_c.get("num_bands", 20))
    band_width             = int(cfg_c.get("band_width", 13))
    jaccard_threshold      = float(cfg_c.get("jaccard_threshold", 0.80))
    small_bucket_threshold = int(cfg_c.get("small_bucket_threshold", 200))
    max_bucket_size        = int(cfg_c.get("max_bucket_size", 10_000))
    seed                   = int(cfg_c.get("seed", 1))
    save_candidate_edges   = bool(cfg_c.get("save_candidate_edges", False))
    verify_chunk           = int(cfg_c.get("gpu_verify_chunk", 2_000_000))
    # B2 — exact-Jaccard verify.  When on, the GPU MinHash estimate is only a
    # RECALL-SAFE pre-filter (kept loose by ``exact_verify_margin``); the final
    # keep/drop is the true char-n-gram Jaccard, computed on CPU in parallel.
    exact          = bool(cfg_c.get("exact_jaccard_verify", False))
    exact_margin   = float(cfg_c.get("exact_verify_margin", 0.10))
    exact_workers  = int(cfg_c.get("exact_verify_workers", os.cpu_count() or 4))
    collect_thr    = max(0.0, jaccard_threshold - exact_margin) if exact else jaccard_threshold

    # similarity: "jaccard" (default) or "containment" (the MinHash-perm method —
    # truncation/slight-edit tolerant: |A∩B|/min(|A|,|B|)).  Containment cannot be
    # read from the GPU MinHash Jaccard estimate, so it forces the exact CPU verify
    # and keeps the GPU estimate as a LOW recall-safe pre-filter (truncations have
    # low Jaccard).  ``min_shared`` guards against generic-phrase mega-merges.
    similarity    = str(cfg_c.get("similarity", "jaccard")).lower()
    min_shared    = int(cfg_c.get("min_shared", 0))
    sim_threshold = jaccard_threshold
    if similarity == "containment":
        sim_threshold = float(cfg_c.get("containment_threshold", jaccard_threshold))
        exact         = True
        collect_thr   = float(cfg_c.get("containment_prefilter", 0.20))
        logger.info("Stage C (cudf): containment mode (thr=%.2f, min_shared=%d, "
                    "prefilter=%.2f, ngram=%dc).", sim_threshold, min_shared,
                    collect_thr, ngram_size)

    if num_bands * band_width > n_perms:
        raise ValueError(
            f"num_bands * band_width = {num_bands*band_width} > num_permutations = {n_perms}.")

    # Permutation coefficients for cudf's (hv*a + b) % mersenne_prime minhash.
    rng = np.random.default_rng(seed)
    a = cp.asarray(rng.integers(1, 2 ** 32, size=n_perms, dtype=np.uint64).astype(np.uint32))
    b = cp.asarray(rng.integers(0, 2 ** 32, size=n_perms, dtype=np.uint64).astype(np.uint32))

    parts = sorted(manifest_dir.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError(f"No manifest parts in {manifest_dir}")

    ds_all: List[str] = []
    cid_all: List[str] = []
    txt_all: List[str] = []     # normalized_text per row (only kept for exact verify)
    sig_chunks = []
    for p in parts:
        g = cudf.read_parquet(str(p), columns=["dataset", "cut_id", "normalized_text"])
        g = g[g["normalized_text"].str.len() > 0]
        if len(g) == 0:
            continue
        # Pad short texts to the n-gram width so identical shorts still cluster.
        sig = g["normalized_text"].str.pad(
            width=ngram_size, side="right", fillchar=_PAD_CHAR).str.minhash(
            seed, a, b, ngram_size)  # list<uint32>[K]
        leaves = cp.asarray(sig.list.leaves.values).reshape(len(g), n_perms).astype(cp.uint32)
        sig_chunks.append(leaves)
        ds_all.extend(g["dataset"].to_arrow().to_pylist())
        cid_all.extend(g["cut_id"].to_arrow().to_pylist())
        if exact:
            txt_all.extend(g["normalized_text"].to_arrow().to_pylist())
    if not sig_chunks:
        raise RuntimeError("Stage C (cudf): no signatures (all texts empty?).")

    sigs = cp.concatenate(sig_chunks, axis=0)
    del sig_chunks
    N = int(sigs.shape[0])
    logger.info("Stage C (cudf): %d signatures, %.2f GB on GPU", N, sigs.nbytes / 1e9)

    row_idx = cp.arange(N, dtype=cp.int64)
    e_src: List[np.ndarray] = []
    e_dst: List[np.ndarray] = []
    e_jac: List[np.ndarray] = []

    def _verify_and_collect(ca, cb) -> None:
        """Chunked GPU Jaccard verify of candidate pairs; keep edges with estimate
        >= ``collect_thr``.  Routing EVERY candidate through this means a hash
        collision (band hash or full-signature hash) can never become a false
        union — it is just a candidate that fails verification.  In exact-verify
        mode ``collect_thr`` is loose (recall-safe pre-filter) and the real
        decision happens on CPU afterwards."""
        for s in range(0, int(ca.shape[0]), verify_chunk):
            ia = ca[s:s + verify_chunk]
            ib = cb[s:s + verify_chunk]
            jac = (sigs[ia] == sigs[ib]).mean(axis=1)     # Jaccard estimate over K
            keep = jac >= collect_thr
            if not bool(keep.any()):
                continue
            e_src.append(cp.asnumpy(ia[keep]))
            e_dst.append(cp.asnumpy(ib[keep]))
            if save_candidate_edges and not exact:
                e_jac.append(cp.asnumpy(jac[keep]))

    def _anchor_fanout(frame):
        """(anchor, other) candidate pairs within each bucket of *frame* (cols
        row, bucket): smallest row per bucket fans out to the rest.  O(bucket)."""
        anchor = frame.groupby("bucket")["row"].min().reset_index()
        anchor.columns = ["bucket", "anchor"]
        mm = frame.merge(anchor, on="bucket")
        mm = mm[mm["row"] != mm["anchor"]]
        if not len(mm):
            return None, None
        return mm["anchor"].values, mm["row"].values

    # ---- Full-signature groups -> candidate pairs (the per-band anchor fan-out
    # below misses pairs that are near-dups of *each other* but not of the band's
    # one anchor; rows with an identical signature are Jaccard 1.0 and must merge).
    # hash_values over all K columns groups identical signatures; a 32-bit hash
    # collision is harmless because the pairs are Jaccard-verified.
    sig_id = cudf.DataFrame(sigs).hash_values()
    sg = cudf.DataFrame({"row": cudf.Series(row_idx), "bucket": sig_id})
    sg_sizes = sg.groupby("bucket").size().reset_index()
    sg_sizes.columns = ["bucket", "sz"]
    sg_sizes = sg_sizes[sg_sizes["sz"] >= 2]
    if len(sg_sizes):
        sg = sg.merge(sg_sizes[["bucket"]], on="bucket")
        sa_pairs, sb_pairs = _anchor_fanout(sg[["row", "bucket"]])
        if sa_pairs is not None:
            _verify_and_collect(sa_pairs, sb_pairs)

    # ---- LSH banding (per band) -> candidate pairs -> GPU Jaccard verify ----
    # Process one band at a time so candidate-pair tables stay bounded.  Buckets
    # are no longer DROPPED for size: an oversized bucket just takes the anchor
    # fan-out path (O(bucket), like mid) instead of all-pairs, so the most-
    # duplicated content is clustered rather than silently lost.
    for bi in range(num_bands):
        c0 = bi * band_width
        band = cudf.DataFrame(sigs[:, c0:c0 + band_width])
        bucket = band.hash_values()                       # uint32 per row
        df = cudf.DataFrame({"row": cudf.Series(row_idx), "bucket": bucket})
        sizes = df.groupby("bucket").size().reset_index()
        sizes.columns = ["bucket", "sz"]
        sizes = sizes[sizes["sz"] >= 2]
        if len(sizes) == 0:
            continue
        df = df.merge(sizes, on="bucket", how="inner")
        small = df[df["sz"] <= small_bucket_threshold]
        mid   = df[df["sz"] >  small_bucket_threshold]

        ca_parts = []
        cb_parts = []
        if len(small):                                    # all-pairs within bucket
            mm = small[["row", "bucket"]].merge(small[["row", "bucket"]], on="bucket")
            mm = mm[mm["row_x"] < mm["row_y"]]
            if len(mm):
                ca_parts.append(mm["row_x"].values)
                cb_parts.append(mm["row_y"].values)
        if len(mid):                                      # anchor fan-out (incl. oversized)
            a_an, b_an = _anchor_fanout(mid[["row", "bucket"]])
            if a_an is not None:
                ca_parts.append(a_an)
                cb_parts.append(b_an)
        if not ca_parts:
            continue
        ca = cp.concatenate(ca_parts)
        cb = cp.concatenate(cb_parts)
        _verify_and_collect(ca, cb)

    # ---- Connected components (cuGraph if available, else scipy) ----
    edge_log = [] if save_candidate_edges else None
    if e_src:
        ea = np.concatenate(e_src)
        eb = np.concatenate(e_dst)
        if exact:
            # B2: replace the noisy MinHash estimate with the TRUE Jaccard on the
            # deduped candidate set (CPU, parallel).  Only the survivors form edges.
            ea, eb = _dedup_pairs(ea, eb, N)
            logger.info("Stage C (cudf): exact-Jaccard verify on %d unique candidate "
                        "pairs (%d workers).", len(ea), exact_workers)
            keep, jac = exact_jaccard_verify(ea, eb, txt_all, ngram_size,
                                             sim_threshold, exact_workers,
                                             metric=similarity, min_shared=min_shared)
            ea, eb, kept_jac = ea[keep], eb[keep], jac[keep]
            logger.info("Stage C (cudf): %d pairs confirmed (%s >= %.2f, min_shared=%d).",
                        len(ea), similarity, sim_threshold, min_shared)
            roots = _connected_components_from_edges(ea, eb, N, cfg_c)
            if save_candidate_edges:
                edge_log = [(int(ea[k]), int(eb[k]), float(kept_jac[k]))
                            for k in range(ea.shape[0])]
        else:
            roots = _connected_components_from_edges(ea, eb, N, cfg_c)
            if save_candidate_edges:
                ej = np.concatenate(e_jac)
                edge_log = [(int(ea[k]), int(eb[k]), float(ej[k])) for k in range(ea.shape[0])]
    else:
        roots = np.arange(N, dtype=np.int64)

    return ds_all, cid_all, roots, edge_log


def _run_dask_cudf_backend(manifest_dir: Path, cfg_c: dict):
    """Multi-GPU Stage C via dask-cuda: distributed cudf MinHash + per-band LSH
    + Jaccard verify across all visible GPUs, connected components via cuGraph
    (multi-GPU when available, else scipy on the gathered edge list).

    For 50M+ cuts where the N x K signature table won't fit one GPU.  Candidate
    pairs use anchor fan-out within each LSH bucket (single-linkage equivalent
    for connected components; much smaller shuffle than all-pairs).

    Returns ``(ds_all, cid_all, roots, edge_log)``.
    """
    import cudf
    import cupy as cp
    import dask_cudf
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    n_perms              = int(cfg_c.get("num_permutations", 256))
    ngram_size           = int(cfg_c.get("ngram_size", 24))
    num_bands            = int(cfg_c.get("num_bands", 20))
    band_width           = int(cfg_c.get("band_width", 13))
    jaccard_threshold    = float(cfg_c.get("jaccard_threshold", 0.80))
    max_bucket_size      = int(cfg_c.get("max_bucket_size", 10_000))
    seed                 = int(cfg_c.get("seed", 1))
    save_candidate_edges = bool(cfg_c.get("save_candidate_edges", False))

    if num_bands * band_width > n_perms:
        raise ValueError(
            f"num_bands * band_width = {num_bands*band_width} > num_permutations = {n_perms}.")

    rng = np.random.default_rng(seed)
    a_np = rng.integers(1, 2 ** 32, size=n_perms, dtype=np.uint64).astype(np.uint32)
    b_np = rng.integers(0, 2 ** 32, size=n_perms, dtype=np.uint64).astype(np.uint32)
    scol = [f"s{k}" for k in range(n_perms)]

    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=cfg_c.get("dask_cuda_visible_devices"),
        rmm_pool_size=cfg_c.get("rmm_pool_size", "70GB"),
        rmm_managed_memory=bool(cfg_c.get("rmm_managed_memory", False)),
    )
    client = Client(cluster)
    logger.info("Stage C (dask_cudf): cluster up with %d GPU worker(s).",
                len(cluster.workers))
    try:
        ddf = dask_cudf.read_parquet(
            str(manifest_dir / "part_*.parquet"),
            columns=["dataset", "cut_id", "normalized_text"])
        ddf = ddf[ddf["normalized_text"].str.len() > 0]   # shorts padded, not dropped

        # ---- Distributed MinHash -> wide signature frame (s0..sK-1) ----
        def _mh(part):
            if len(part) == 0:
                cols = {c: cp.zeros(0, dtype="uint32") for c in scol}
                out = cudf.DataFrame(cols)
                out["dataset"] = cudf.Series([], dtype="object")
                out["cut_id"] = cudf.Series([], dtype="object")
                return out
            sig = part["normalized_text"].str.pad(
                width=int(ngram_size), side="right", fillchar=_PAD_CHAR).str.minhash(
                int(seed), cudf.Series(a_np), cudf.Series(b_np), int(ngram_size))
            arr = cp.asarray(sig.list.leaves).reshape(len(part), n_perms)
            out = cudf.DataFrame({scol[k]: arr[:, k] for k in range(n_perms)})
            out["dataset"] = part["dataset"].reset_index(drop=True)
            out["cut_id"] = part["cut_id"].reset_index(drop=True)
            return out

        meta = cudf.DataFrame({c: cp.zeros(0, dtype="uint32") for c in scol})
        meta["dataset"] = cudf.Series([], dtype="object")
        meta["cut_id"] = cudf.Series([], dtype="object")
        wide = ddf.map_partitions(_mh, meta=meta)
        # Stable global row id 0..N-1 (cumulative count of ones across partitions).
        wide = wide.reset_index(drop=True)
        wide["row"] = wide.map_partitions(
            lambda p: cudf.Series(cp.ones(len(p), dtype="int64")),
            meta=("row", "int64")).cumsum() - 1
        wide = wide.persist()

        idmap = wide[["row", "dataset", "cut_id"]].compute().sort_values("row")
        ds_all = idmap["dataset"].to_arrow().to_pylist()
        cid_all = idmap["cut_id"].to_arrow().to_pylist()
        N = len(ds_all)
        logger.info("Stage C (dask_cudf): %d signatures across %d partitions.",
                    N, wide.npartitions)

        sig_wide = wide[["row"] + scol].persist()

        # ---- Per-band LSH -> anchor fan-out candidate pairs ----
        pair_frames = []
        for bi in range(num_bands):
            cols = scol[bi * band_width:(bi + 1) * band_width]

            def _bucket(part, cols=cols):
                return cudf.DataFrame({"row": part["row"].values,
                                       "bucket": part[cols].hash_values()})

            bmeta = cudf.DataFrame({"row": cp.zeros(0, "int64"),
                                    "bucket": cp.zeros(0, "uint32")})
            bdf = sig_wide.map_partitions(_bucket, meta=bmeta)
            sizes = (bdf.groupby("bucket").row.count().reset_index()
                     .rename(columns={"row": "sz"}))
            # B3: no longer drop oversized buckets — anchor fan-out below is
            # O(bucket), so big buckets are clustered, not lost.  (B4 full-signature
            # collapse is implemented for the cudf backend; not ported to dask here
            # since dask_cudf is the not-yet-used 50M+ path.)
            sizes = sizes[sizes["sz"] >= 2]
            bsel = bdf.merge(sizes[["bucket"]], on="bucket")
            anchor = (bsel.groupby("bucket").row.min().reset_index()
                      .rename(columns={"row": "a"}))
            pp = bsel.merge(anchor, on="bucket")
            pp = pp[pp["row"] != pp["a"]][["a", "row"]].rename(columns={"row": "b"})
            pair_frames.append(pp)
        pairs = dask_cudf.concat(pair_frames).drop_duplicates().persist()

        # ---- Distributed Jaccard verify (join both signatures, compare K cols) ----
        sa = sig_wide.rename(columns={"row": "a", **{c: f"a_{c}" for c in scol}})
        sb = sig_wide.rename(columns={"row": "b", **{c: f"b_{c}" for c in scol}})
        merged = pairs.merge(sa, on="a").merge(sb, on="b")
        acols = [f"a_{c}" for c in scol]
        bcols = [f"b_{c}" for c in scol]

        def _verify(part):
            if len(part) == 0:
                return cudf.DataFrame({"a": cp.zeros(0, "int64"),
                                       "b": cp.zeros(0, "int64"),
                                       "jac": cp.zeros(0, "float32")})
            jac = (part[acols].values == part[bcols].values).mean(axis=1).astype("float32")
            keep = jac >= jaccard_threshold
            return cudf.DataFrame({"a": part["a"].values[keep],
                                   "b": part["b"].values[keep],
                                   "jac": jac[keep]})

        vmeta = cudf.DataFrame({"a": cp.zeros(0, "int64"),
                                "b": cp.zeros(0, "int64"),
                                "jac": cp.zeros(0, "float32")})
        verified = merged.map_partitions(_verify, meta=vmeta).persist()

        # ---- Connected components ----
        roots = None
        if bool(cfg_c.get("use_cugraph_cc", True)):
            try:
                import cugraph
                import cugraph.dask as dask_cugraph
                from cugraph.dask.comms import comms as Comms
                Comms.initialize(p2p=True)
                edf = verified[["a", "b"]].rename(columns={"a": "src", "b": "dst"})
                g = cugraph.Graph(directed=False)
                g.from_dask_cudf_edgelist(edf, source="src", destination="dst",
                                          renumber=True)
                cdf = dask_cugraph.weakly_connected_components(g).compute().to_pandas()
                Comms.destroy()
                roots = np.arange(N, dtype=np.int64)
                roots[cdf["vertex"].to_numpy()] = N + cdf["labels"].to_numpy()
                logger.info("Stage C (dask_cudf): connected components via cuGraph MG.")
            except Exception as exc:
                logger.warning("Stage C (dask_cudf): cuGraph MG CC unavailable (%s); "
                               "scipy fallback on gathered edges.", repr(exc)[:90])
                roots = None
        if roots is None:
            edf = verified[["a", "b"]].compute()
            ea = edf["a"].to_numpy()
            eb = edf["b"].to_numpy()
            roots = _connected_components_from_edges(
                ea, eb, N, {**cfg_c, "use_cugraph_cc": False})

        edge_log = None
        if save_candidate_edges:
            edf = verified.compute().to_pandas()
            edge_log = [(int(r.a), int(r.b), float(r.jac))
                        for r in edf.itertuples(index=False)]
        return ds_all, cid_all, roots, edge_log
    finally:
        client.close()
        cluster.close()


def run(cfg: dict) -> None:
    output_dir   = Path(cfg["output_dir"])
    manifest_dir = output_dir / "manifest"
    stage_dir    = output_dir / "text_dedup"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_tmp(stage_dir)
    _verify_upstream(manifest_dir)

    cfg_c = cfg.get("text_dedup", {})
    n_perms    = int(cfg_c.get("num_permutations", 256))
    num_bands  = int(cfg_c.get("num_bands", 20))
    band_width = int(cfg_c.get("band_width", 13))
    if num_bands * band_width > n_perms:
        raise ValueError(
            f"num_bands * band_width = {num_bands*band_width} > num_permutations = {n_perms}.")

    backend = str(cfg_c.get("backend", "cudf")).lower().replace("-", "_")
    if backend not in ("cudf", "dask_cudf"):
        raise ValueError(
            f"text_dedup.backend={backend!r} unsupported; use 'cudf' or 'dask_cudf'.")

    t0 = time.time()
    if backend == "cudf":
        ds_all, cid_all, roots, edge_log = _run_cudf_backend(manifest_dir, cfg_c)
    else:
        ds_all, cid_all, roots, edge_log = _run_dask_cudf_backend(manifest_dir, cfg_c)
    _finalize_and_write(stage_dir, ds_all, cid_all, roots, cfg_c, edge_log, t0, backend)
    _release_rmm_pool(cfg_c)   # free the GPU pool so the next same-process stage (fingerprint) does not OOM


def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage C: fuzzy text dedup")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(_load_cfg(args.config))


if __name__ == "__main__":
    main()
