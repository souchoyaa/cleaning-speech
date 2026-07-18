#!/usr/bin/env python3
"""Bag-of-Words text dedup — EXPERIMENT (standalone; does NOT touch core/).

An alternative to the char-n-gram MinHash (core/text_dedup.py): represent each
transcript as its SET OF WORDS (order-independent), cluster near-duplicates via
word-set MinHash + LSH + union-find, and report how many duplicates it flags vs.
the current method's text_dedup/clusters.parquet.

Why this clusters better on `text_slight_change`: char-16-gram MinHash keys on
exact character substrings, so reordered / slightly-edited transcripts fall
below threshold; word-set Jaccard is order-independent and tolerant of edits.
A *non-strict* policy (low word-Jaccard threshold, e.g. 0.5) catches more.

Scales to millions of cuts:
  - candidate generation is fully vectorized (per-band sort → consecutive
    same-key pairs; no Python per-bucket loop, no set-of-tuples),
  - pair de-dup via a single np.unique on encoded ids,
  - a vectorized MinHash-estimate pre-filter cuts the pair set down to the
    near-duplicate neighbourhood, then EXACT word-Jaccard is computed only on
    those survivors (a small set), at several thresholds in one pass.

Run (in container):
  python -m audio_tokenization.utils.data_selection.dup_retrieval.eval.bow_dedup \
    --manifest <dedup_out>/manifest \
    --baseline <dedup_out>/text_dedup/clusters.parquet \
    --ngram 1 --num-perms 128 --bands 32 --band-width 4 \
    --thresholds 0.5,0.6,0.7
"""
from __future__ import annotations
import argparse, glob, os, re, time
from collections import Counter
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_WORD = re.compile(r"\w+", re.UNICODE)
_PRIME = (1 << 61) - 1


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
def load_texts(manifest_dir):
    cid, txt, ds = [], [], []
    for f in sorted(glob.glob(f"{manifest_dir}/part_*.parquet")):
        t = pq.read_table(f, columns=["dataset", "cut_id", "normalized_text"])
        ds  += t.column("dataset").to_pylist()
        cid += t.column("cut_id").to_pylist()
        txt += t.column("normalized_text").to_pylist()
    return cid, txt, ds


def write_text_clusters(path, labels, ds, cid):
    """Write a text_dedup-format clusters.parquet (+_SUCCESS) for the dup cuts
    (cluster_size>1), so the core audio stages can use BoW clusters as their
    whitelist. Columns: dataset, cut_id, text_cluster_id, cluster_size."""
    cnt = Counter(labels.tolist())
    remap = {}
    DS, CID, TC, SZ = [], [], [], []
    for i in range(len(labels)):
        lab = int(labels[i])
        if cnt[lab] <= 1:
            continue
        if lab not in remap:
            remap[lab] = len(remap)
        DS.append(ds[i]); CID.append(cid[i]); TC.append(remap[lab]); SZ.append(cnt[lab])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table({
        "dataset": DS, "cut_id": CID,
        "text_cluster_id": pa.array(TC, pa.int64()),
        "cluster_size": pa.array(SZ, pa.int64()),
    }), path)
    open(os.path.join(os.path.dirname(path), "_SUCCESS"), "w").close()
    return len(CID), len(remap)


def build_csr(texts, ngram):
    """Tokenize → per-text SET of word(-gram) ids, as a CSR (flat ids + offsets)."""
    vocab = {}
    flat = []
    offs = [0]
    for s in texts:
        words = _WORD.findall((s or "").lower())
        if ngram > 1:
            words = [" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)] or words
        seen = set()
        for w in words:
            if w in seen:
                continue
            seen.add(w)
            wid = vocab.get(w)
            if wid is None:
                wid = len(vocab); vocab[w] = wid
            flat.append(wid)
        offs.append(len(flat))
    return np.asarray(flat, dtype=np.uint64), np.asarray(offs, dtype=np.int64), len(vocab)


def minhash(flat, offs, n, k, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=k, dtype=np.uint64)
    b = rng.integers(0, _PRIME, size=k, dtype=np.uint64)
    sig = np.full((n, k), _PRIME, dtype=np.uint64)
    lengths = np.diff(offs)
    ne = np.where(lengths > 0)[0]              # non-empty texts
    starts = offs[:-1][ne]                     # segment starts in flat (contiguous)
    for i in range(k):
        ph = (a[i] * flat + b[i]) % _PRIME     # permuted hash over all tokens
        sig[ne, i] = np.minimum.reduceat(ph, starts)
    return sig


def lsh_candidate_pairs(sig, bands, width):
    """Vectorized: per band, sort by band-key and emit consecutive same-key pairs.

    Consecutive same-key pairs chain every bucket (union-find closes it
    transitively), so a bucket of size m costs m-1 pairs — O(n) per band, no
    per-bucket Python loop and no giant-bucket blow-up. Returns de-duplicated
    (I, J) with I < J.
    """
    n = sig.shape[0]
    pw = np.uint64(1099511628211)              # FNV-ish band mixer
    Is, Js = [], []
    for bi in range(bands):
        band = sig[:, bi * width:(bi + 1) * width]
        key = np.zeros(n, dtype=np.uint64)
        for j in range(width):
            key = (key * pw) ^ band[:, j]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = ks[1:] == ks[:-1]               # consecutive identical band-keys
        ii = order[:-1][same]
        jj = order[1:][same]
        Is.append(np.minimum(ii, jj))
        Js.append(np.maximum(ii, jj))
    I = np.concatenate(Is); J = np.concatenate(Js)
    code = I.astype(np.int64) * np.int64(n) + J.astype(np.int64)
    uniq = np.unique(code)
    return (uniq // np.int64(n)).astype(np.int64), (uniq % np.int64(n)).astype(np.int64)


def estimate_jaccard(sig, I, J, chunk=2_000_000):
    """Vectorized MinHash-estimate Jaccard for each (I,J) pair, chunked for RAM."""
    k = sig.shape[1]
    out = np.empty(len(I), dtype=np.float32)
    for s in range(0, len(I), chunk):
        e = min(s + chunk, len(I))
        out[s:e] = (sig[I[s:e]] == sig[J[s:e]]).sum(axis=1) / k
    return out


def exact_jaccard(flat, offs, I, J):
    """EXACT word-set Jaccard for the (small) survivor pair set.

    Builds each involved text's word-set once (cache), then intersects per pair.
    """
    if len(I) == 0:
        return np.empty(0, dtype=np.float32)
    involved = np.unique(np.concatenate([I, J]))
    sets = {int(idx): set(flat[offs[idx]:offs[idx + 1]].tolist()) for idx in involved.tolist()}
    out = np.empty(len(I), dtype=np.float32)
    for t in range(len(I)):
        si = sets[int(I[t])]; sj = sets[int(J[t])]
        if not si or not sj:
            out[t] = 0.0; continue
        inter = len(si & sj)
        out[t] = inter / (len(si) + len(sj) - inter)
    return out


def union_find_labels(n, I, J):
    p = np.arange(n, dtype=np.int64)

    def find(x):
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    for a, b in zip(I.tolist(), J.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            p[ra] = rb
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def cluster_stats(labels):
    _, counts = np.unique(labels, return_counts=True)
    dup = counts[counts > 1]
    return int(len(dup)), int(dup.sum()), (int(dup.max()) if len(dup) else 1)


def dup_cids(labels, cids):
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    mask = counts[inv] > 1
    return {cids[i] for i in np.flatnonzero(mask).tolist()}


def baseline_dup_cids(path):
    t = pq.read_table(path)
    names = t.column_names
    cids = t.column("cut_id").to_pylist() if "cut_id" in names else []
    if "cluster_size" in names:
        sizes = t.column("cluster_size").to_pylist()
        return {cids[i] for i in range(len(cids)) if sizes[i] > 1}, names
    return set(cids), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ngram", type=int, default=1)
    ap.add_argument("--num-perms", type=int, default=128)
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--band-width", type=int, default=4)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7",
                    help="comma-separated word-Jaccard thresholds to report (non-strict = low)")
    ap.add_argument("--prefilter-margin", type=float, default=0.12,
                    help="keep candidates with estimate >= min(thr)-margin before exact verify")
    ap.add_argument("--write-clusters", default=None,
                    help="write a text_dedup-format clusters.parquet (+_SUCCESS) for --cluster-threshold")
    ap.add_argument("--cluster-threshold", type=float, default=None,
                    help="threshold at which to write clusters (must be one of --thresholds)")
    a = ap.parse_args()
    assert a.bands * a.band_width <= a.num_perms, "bands*width must be <= num_perms"
    thrs = sorted(float(x) for x in a.thresholds.split(","))

    t0 = time.time()
    cid, txt, ds = load_texts(a.manifest)
    n = len(cid)
    log(f"[load] {n:,} cuts in {time.time()-t0:.1f}s")

    t1 = time.time()
    flat, offs, vsize = build_csr(txt, a.ngram)
    log(f"[tokens] vocab={vsize:,} total_tokens={len(flat):,} ngram={a.ngram} in {time.time()-t1:.1f}s")

    t2 = time.time()
    sig = minhash(flat, offs, n, a.num_perms)
    log(f"[minhash] {a.num_perms} perms in {time.time()-t2:.1f}s")

    t3 = time.time()
    I, J = lsh_candidate_pairs(sig, a.bands, a.band_width)
    log(f"[lsh] {len(I):,} unique candidate pairs ({a.bands}x{a.band_width} bands) in {time.time()-t3:.1f}s")

    t4 = time.time()
    est = estimate_jaccard(sig, I, J)
    keep = est >= (min(thrs) - a.prefilter_margin)
    I, J = I[keep], J[keep]
    log(f"[prefilter] {len(I):,} pairs >= {min(thrs)-a.prefilter_margin:.2f} (estimate) in {time.time()-t4:.1f}s")

    t5 = time.time()
    exj = exact_jaccard(flat, offs, I, J)
    log(f"[exact] verified {len(exj):,} pairs in {time.time()-t5:.1f}s")

    base_dup, bnames = (baseline_dup_cids(a.baseline) if a.baseline else (None, None))

    log("\n================ BAG-OF-WORDS RESULT ================")
    log(f"  cuts total : {n:,}   (ngram={a.ngram}, {a.bands}x{a.band_width} LSH)")
    if base_dup is not None:
        log(f"  baseline (char-16-gram) flags : {len(base_dup):,}  ({100*len(base_dup)/n:.2f}%)")
    log(f"  {'thr':>5} {'dup_clusters':>13} {'cuts_flagged':>13} {'%corpus':>8} {'largest':>8}"
        + ("  | new_vs_base  only_base  Jaccard" if base_dup is not None else ""))
    for thr in thrs:
        m = exj >= thr
        labels = union_find_labels(n, I[m], J[m])
        if a.write_clusters and a.cluster_threshold is not None and abs(thr - a.cluster_threshold) < 1e-9:
            nrows, ncl_w = write_text_clusters(a.write_clusters, labels, ds, cid)
            log(f"  [wrote {nrows:,} dup cuts / {ncl_w:,} clusters @ thr={thr} → {a.write_clusters}]")
        nc, nd, lg = cluster_stats(labels)
        line = f"  {thr:5.2f} {nc:>13,} {nd:>13,} {100*nd/n:>7.2f}% {lg:>8,}"
        if base_dup is not None:
            bd = dup_cids(labels, cid)
            new = len(bd - base_dup); ob = len(base_dup - bd)
            jacc = len(bd & base_dup) / max(len(bd | base_dup), 1)
            line += f"  | {new:>10,} {ob:>9,} {jacc:>8.3f}"
        log(line)
    log("====================================================")
    if base_dup is not None:
        log("new_vs_base = cuts BoW flags that char-16-gram missed; "
            "only_base = cuts char-16-gram flags that BoW missed.")


if __name__ == "__main__":
    main()
