#!/usr/bin/env python3
"""Diagnose whether Stage-F refine_chains (fix "B5") is worth enabling.

refine_chains only helps when single-linkage clustering produces *chains* —
clusters held together by a path of pairwise-similar links rather than being a
tight clique. This reads a Stage-C output dir and reports:

  - the text-cluster size distribution (a heavy tail is the *prerequisite* for
    chaining; with no large clusters there are no chains to break);
  - if candidate_edges.parquet is present (run Stage C with
    save_candidate_edges: true), the per-cluster edge DENSITY = edges /
    clique_edges — low density in large clusters is the direct chaining signal.

Verdict:
  - max/p99 sizes small               -> refine_chains NOT useful.
  - large clusters + no edges          -> inconclusive; re-run with edges.
  - large clusters + low edge density  -> refine_chains USEFUL.

Usage:
  python diagnose_chaining.py --stage-c-dir <output_dir>/text_dedup
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-c-dir", required=True,
                    help="<output_dir>/text_dedup")
    args = ap.parse_args()
    d = Path(args.stage_c_dir)

    succ = d / "_SUCCESS"
    if succ.exists():
        s = json.loads(succ.read_text())
        print(f"_SUCCESS: backend={s.get('backend')} rows_in_clusters={s.get('rows')} "
              f"clusters={s.get('clusters')} candidate_fraction={s.get('candidate_fraction')}")

    tc = pq.read_table(d / "text_clusters.parquet",
                       columns=["text_cluster_id", "cluster_size"])
    # per-cluster size (one row per cluster)
    seen = {}
    for cid, sz in zip(tc.column("text_cluster_id").to_pylist(),
                       tc.column("cluster_size").to_pylist()):
        seen[cid] = sz
    sizes = sorted(seen.values())
    n_clusters = len(sizes)
    n_cuts = sum(sizes)
    if n_clusters == 0:
        print("No clusters (size>1). Nothing to chain. refine_chains NOT useful.")
        return
    mx = sizes[-1]
    mean = n_cuts / n_clusters
    print(f"\nText clusters: {n_clusters} | clustered cuts: {n_cuts}")
    print(f"size  mean={mean:.1f}  median={_pct(sizes,0.5)}  p90={_pct(sizes,0.90)} "
          f" p99={_pct(sizes,0.99)}  max={mx}")
    for thr in (10, 50, 100, 1000):
        n = sum(1 for s in sizes if s > thr)
        frac = sum(s for s in sizes if s > thr) / n_cuts
        print(f"  clusters >{thr:>4}: {n:>6}   (hold {frac*100:5.1f}% of clustered cuts)")

    # --- direct chaining signal from candidate edges, if available ---
    edges_path = d / "candidate_edges.parquet"
    verdict_from_edges = None
    if edges_path.exists():
        et = pq.read_table(edges_path, columns=["src_cut_id", "dst_cut_id"])
        # We don't have cut->cluster here cheaply without a join; approximate by
        # counting edges per connected blob is out of scope. Report edge count.
        n_edges = et.num_rows
        # clique edges if every cluster were complete:
        clique = sum(s * (s - 1) // 2 for s in sizes)
        density = n_edges / max(clique, 1)
        print(f"\ncandidate_edges: {n_edges} edges; overall density vs cliques="
              f"{density:.3f}")
        verdict_from_edges = density
        if density < 0.3 and mx > 50:
            print("=> LOW density + large clusters: chains likely. refine_chains USEFUL.")
        elif density >= 0.6:
            print("=> HIGH density: clusters are near-cliques, not chains. NOT useful.")
        else:
            print("=> Mixed; inspect largest clusters individually.")

    # --- size-only screen ---
    print("\nVERDICT (size screen):")
    if mx <= 50 and _pct(sizes, 0.99) <= 10:
        print("  Clusters are small and tight-tailed -> NO chains possible -> "
              "refine_chains NOT useful. Skip B5.")
    elif verdict_from_edges is None:
        print("  Heavy tail present but no candidate_edges to measure density.")
        print("  -> INCONCLUSIVE. Re-run Stage C with save_candidate_edges: true,")
        print("     then re-run this to get edge density before enabling B5.")
    else:
        print("  See edge-density verdict above.")


if __name__ == "__main__":
    main()
