#!/usr/bin/env python3
"""Permissive near-duplicate text clustering — EXPERIMENT (standalone).

Goal vs. bow_dedup.py: cluster transcripts together even under SLIGHT edits and
TRUNCATION (one transcript is a shortened/contained version of another). Plain
Jaccard tanks on truncation (the longer text has many extra words → small
intersection/union ratio), so here we score with CONTAINMENT:

    sim(A,B) = |A ∩ B| / min(|A|, |B|)

A truncated/contained transcript scores ~1.0 against its longer original, so it
clusters in. To stop containment from fusing everything through generic short
phrases ("thank you", "yeah"), two guards:
  --min-words   : texts with fewer unique words are left as singletons.
  --min-shared  : a pair needs at least this many shared words to link.

Candidate generation is the same vectorized word-set MinHash + LSH as bow_dedup,
but defaults are more permissive (more bands → lower LSH threshold → more
recall). Verification is EXACT containment on the survivors.

Caveat: MinHash-LSH still under-recalls EXTREME truncation (a tiny snippet
inside a very long text shares too few band-keys); catching those needs an
inverted word index, not done here.

Run (in container):
  python -m audio_tokenization.utils.data_selection.dup_retrieval.eval.bow_dedup_permissive \
    --manifest <dedup_out>/manifest \
    --baseline <dedup_out>/text_dedup/clusters.parquet \
    --ngram 1 --num-perms 128 --bands 64 --band-width 2 \
    --thresholds 0.8,0.9 --min-words 5 --min-shared 4
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


def load_texts(manifest_dir):
    cid, txt, ds = [], [], []
    for f in sorted(glob.glob(f"{manifest_dir}/part_*.parquet")):
        t = pq.read_table(f, columns=["dataset", "cut_id", "normalized_text"])
        ds  += t.column("dataset").to_pylist()
        cid += t.column("cut_id").to_pylist()
        txt += t.column("normalized_text").to_pylist()
    return cid, txt, ds


def build_csr(texts, ngram):
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
    ne = np.where(lengths > 0)[0]
    starts = offs[:-1][ne]
    for i in range(k):
        ph = (a[i] * flat + b[i]) % _PRIME
        sig[ne, i] = np.minimum.reduceat(ph, starts)
    return sig


def lsh_candidate_pairs(sig, bands, width, eligible):
    """Vectorized consecutive-same-key pairs per band, restricted to eligible rows
    (passing the --min-words guard) so short generic texts can't seed clusters."""
    n = sig.shape[0]
    pw = np.uint64(1099511628211)
    Is, Js = [], []
    for bi in range(bands):
        band = sig[:, bi * width:(bi + 1) * width]
        key = np.zeros(n, dtype=np.uint64)
        for j in range(width):
            key = (key * pw) ^ band[:, j]
        key = np.where(eligible, key, np.uint64(0))   # ineligible → bucket 0, dropped below
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = (ks[1:] == ks[:-1]) & (ks[1:] != np.uint64(0))
        ii = order[:-1][same]; jj = order[1:][same]
        Is.append(np.minimum(ii, jj)); Js.append(np.maximum(ii, jj))
    I = np.concatenate(Is); J = np.concatenate(Js)
    code = I.astype(np.int64) * np.int64(n) + J.astype(np.int64)
    uniq = np.unique(code)
    return (uniq // np.int64(n)).astype(np.int64), (uniq % np.int64(n)).astype(np.int64)


def estimate_jaccard(sig, I, J, chunk=2_000_000):
    k = sig.shape[1]
    out = np.empty(len(I), dtype=np.float32)
    for s in range(0, len(I), chunk):
        e = min(s + chunk, len(I))
        out[s:e] = (sig[I[s:e]] == sig[J[s:e]]).sum(axis=1) / k
    return out


def exact_containment(flat, offs, I, J, min_shared):
    """EXACT max-containment |A∩B|/min(|A|,|B|) + the absolute intersection size,
    so the caller can apply both a ratio threshold and a --min-shared floor."""
    if len(I) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
    involved = np.unique(np.concatenate([I, J]))
    sets = {int(idx): set(flat[offs[idx]:offs[idx + 1]].tolist()) for idx in involved.tolist()}
    cont = np.empty(len(I), dtype=np.float32)
    shared = np.empty(len(I), dtype=np.int32)
    for t in range(len(I)):
        si = sets[int(I[t])]; sj = sets[int(J[t])]
        if not si or not sj:
            cont[t] = 0.0; shared[t] = 0; continue
        inter = len(si & sj)
        shared[t] = inter
        cont[t] = inter / min(len(si), len(sj))
    return cont, shared


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


def write_text_clusters(path, labels, ds, cid):
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


def baseline_dup_cids(path):
    t = pq.read_table(path)
    names = t.column_names
    cids = t.column("cut_id").to_pylist() if "cut_id" in names else []
    if "cluster_size" in names:
        sizes = t.column("cluster_size").to_pylist()
        return {cids[i] for i in range(len(cids)) if sizes[i] > 1}
    return set(cids)


def freq_cap_report(txt, caps):
    """Audio-INDEPENDENT cap: count EXACT normalized text; for each cap K report
    how many cuts would be dropped keeping at most K copies of each sentence.
    Keyed on exact transcript (NOT cluster) — robust to permissive over-merging."""
    tc = Counter(t for t in txt if t)
    n = len(txt)
    log("\n  --- frequent-sentence cap (exact-text frequency, audio-independent) ---")
    top = tc.most_common(5)
    log("  top sentences: " + " | ".join(f"{c}x {repr((s or '')[:25])}" for s, c in top))
    for K in caps:
        drop = sum(c - K for c in tc.values() if c > K)
        nsent = sum(1 for c in tc.values() if c > K)
        log(f"  cap@{K:<4}: {nsent:,} sentences exceed {K}x → drop {drop:,} cuts ({100*drop/max(n,1):.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ngram", type=int, default=1)
    ap.add_argument("--num-perms", type=int, default=128)
    ap.add_argument("--bands", type=int, default=64)
    ap.add_argument("--band-width", type=int, default=2)
    ap.add_argument("--thresholds", default="0.8,0.9",
                    help="comma-separated CONTAINMENT thresholds to report")
    ap.add_argument("--min-words", type=int, default=5,
                    help="texts with fewer unique words are not clustered (anti generic-phrase merge)")
    ap.add_argument("--min-shared", type=int, default=4,
                    help="a pair needs at least this many shared words to link")
    ap.add_argument("--prefilter-estimate", type=float, default=0.05,
                    help="drop candidate pairs below this MinHash-Jaccard estimate before exact verify")
    ap.add_argument("--cap", default="10,50,100", help="comma-separated K for the frequent-sentence cap report")
    ap.add_argument("--write-clusters", default=None)
    ap.add_argument("--cluster-threshold", type=float, default=None)
    a = ap.parse_args()
    caps = [int(x) for x in a.cap.split(",")] if a.cap else []
    assert a.bands * a.band_width <= a.num_perms, "bands*width must be <= num_perms"
    thrs = sorted(float(x) for x in a.thresholds.split(","))

    t0 = time.time()
    cid, txt, ds = load_texts(a.manifest)
    n = len(cid)
    log(f"[load] {n:,} cuts in {time.time()-t0:.1f}s")

    t1 = time.time()
    flat, offs, vsize = build_csr(txt, a.ngram)
    sizes = np.diff(offs)
    eligible = sizes >= a.min_words
    log(f"[tokens] vocab={vsize:,} ngram={a.ngram} | eligible(>= {a.min_words} words)="
        f"{int(eligible.sum()):,}/{n:,} in {time.time()-t1:.1f}s")

    t2 = time.time()
    sig = minhash(flat, offs, n, a.num_perms)
    log(f"[minhash] {a.num_perms} perms in {time.time()-t2:.1f}s")

    t3 = time.time()
    I, J = lsh_candidate_pairs(sig, a.bands, a.band_width, eligible)
    log(f"[lsh] {len(I):,} candidate pairs ({a.bands}x{a.band_width} bands) in {time.time()-t3:.1f}s")

    t4 = time.time()
    est = estimate_jaccard(sig, I, J)
    keep = est >= a.prefilter_estimate
    I, J = I[keep], J[keep]
    log(f"[prefilter] {len(I):,} pairs >= {a.prefilter_estimate} (estimate) in {time.time()-t4:.1f}s")

    t5 = time.time()
    cont, shared = exact_containment(flat, offs, I, J, a.min_shared)
    log(f"[exact] containment for {len(cont):,} pairs in {time.time()-t5:.1f}s")

    base_dup = baseline_dup_cids(a.baseline) if a.baseline else None

    log("\n============ PERMISSIVE (containment) RESULT ============")
    log(f"  cuts total : {n:,}   (ngram={a.ngram}, {a.bands}x{a.band_width} LSH, "
        f"min_words={a.min_words}, min_shared={a.min_shared})")
    if base_dup is not None:
        log(f"  baseline (char-16-gram) flags : {len(base_dup):,}  ({100*len(base_dup)/n:.2f}%)")
    log(f"  {'cont>=':>7} {'dup_clusters':>13} {'cuts_flagged':>13} {'%corpus':>8} {'largest':>8}"
        + ("  | new_vs_base" if base_dup is not None else ""))
    for thr in thrs:
        m = (cont >= thr) & (shared >= a.min_shared)
        labels = union_find_labels(n, I[m], J[m])
        if a.write_clusters and a.cluster_threshold is not None and abs(thr - a.cluster_threshold) < 1e-9:
            nr, ncl_w = write_text_clusters(a.write_clusters, labels, ds, cid)
            log(f"  [wrote {nr:,} dup cuts / {ncl_w:,} clusters @ cont>={thr} → {a.write_clusters}]")
        nc, nd, lg = cluster_stats(labels)
        line = f"  {thr:7.2f} {nc:>13,} {nd:>13,} {100*nd/n:>7.2f}% {lg:>8,}"
        if base_dup is not None:
            bd = dup_cids(labels, cid)
            line += f"  | {len(bd - base_dup):>10,}"
        log(line)
    if caps:
        freq_cap_report(txt, caps)
    log("=========================================================")


if __name__ == "__main__":
    main()
