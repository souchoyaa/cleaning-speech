"""Join MOS scores with synthetic ground truth; report per-metric discrimination,
inter-metric correlation, and a couple of combination strategies.

Pure-stdlib (no numpy/pandas) so it runs on the login node.
"""
import json, math, sys
from collections import defaultdict

GT   = "/capstor/scratch/cscs/sgodey/dedup_pipeline_test/synthetic_shar/cv22_synth/en/ground_truth.jsonl"
MOS  = "/capstor/scratch/cscs/sgodey/dedup_pipeline_test/mos_synth/mos_rank_0000.jsonl"

# ---- load ground truth ----
gt = {}
for l in open(GT):
    d = json.loads(l)
    gt[d["cut_id"]] = d

# ---- load mos, flatten ----
rows = []
for l in open(MOS):
    d = json.loads(l)
    cid = d["cut_id"]
    if cid not in gt:
        continue
    m = d["metrics"]
    flat = {
        "utmos":   m["utmos"]["score"]["utmos"],
        "stoi":    m["squim"]["score"]["stoi"],
        "pesq":    m["squim"]["score"]["pesq"],
        "si_sdr":  m["squim"]["score"]["si_sdr"],
        "dnsmos":  m["dnsmos_nisqa"]["score"]["nisqa"],
        "CE":      m["audiobox"]["score"]["CE"],
        "CU":      m["audiobox"]["score"]["CU"],
        "PC":      m["audiobox"]["score"]["PC"],
        "PQ":      m["audiobox"]["score"]["PQ"],
    }
    g = gt[cid]
    rows.append((cid, flat, g))

print(f"joined {len(rows)} cuts (gt={len(gt)})\n")

METRICS = ["utmos","stoi","pesq","si_sdr","dnsmos","CE","CU","PC","PQ"]

# ---------- helpers ----------
def auc(pos, neg):
    """AUROC via Mann-Whitney U. pos = scores for 'good', neg = scores for 'bad'.
    Returns P(score_good > score_bad). 0.5 = no separation; 1.0 = perfect (good>bad)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v,1) for v in pos] + [(v,0) for v in neg])
    # rank with ties averaged
    ranks = [0.0]*len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j - 1)/2.0 + 1.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sum_pos = sum(r for r,(v,lab) in zip(ranks, allv) if lab==1)
    n1, n0 = len(pos), len(neg)
    u = sum_pos - n1*(n1+1)/2.0
    return u/(n1*n0)

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def std(xs):
    if len(xs)<2: return 0.0
    mu=mean(xs); return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1))
def pearson(xs, ys):
    n=len(xs); mx=mean(xs); my=mean(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs)); sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx>0 and sy>0 else float("nan")

# ---------- 1. per-aug metric means ----------
augs = sorted(set(g["aug"] for _,_,g in rows))
by_aug = defaultdict(list)
for cid,f,g in rows:
    by_aug[g["aug"]].append(f)

print("="*100)
print("1. MEAN METRIC VALUE PER AUGMENTATION (lower = the metric judges it worse)")
print("="*100)
hdr = "aug".ljust(11) + "n".rjust(5) + "".join(m.rjust(9) for m in METRICS) + "  lowQ"
print(hdr)
for a in ["original","exact","case","punct","speed","reverb","text_edit","noise","clip"]:
    if a not in by_aug: continue
    fs = by_aug[a]
    lowq = mean([1.0 if g["expect_low_quality"] else 0.0 for cid,f,g in rows if g["aug"]==a])
    line = a.ljust(11) + str(len(fs)).rjust(5)
    for m in METRICS:
        line += f"{mean([f[m] for f in fs]):9.2f}"
    line += f"  {lowq*100:4.0f}%"
    print(line)

# ---------- 2. discrimination: low-quality vs clean originals ----------
# "bad" = expect_low_quality True (clip + low-SNR noise). "good" = originals.
good = [f for cid,f,g in rows if g["aug"]=="original"]
bad  = [f for cid,f,g in rows if g["expect_low_quality"]]
print("\n"+"="*100)
print(f"2. AUROC — separate expect_low_quality ({len(bad)} cuts: clip + low-SNR noise) from originals ({len(good)})")
print("   1.00 = metric perfectly ranks clean above degraded; 0.50 = useless; <0.5 = inverted")
print("="*100)
aucs = {}
for m in METRICS:
    a = auc([f[m] for f in good], [f[m] for f in bad])
    aucs[m] = a
    print(f"  {m:8s}  AUROC = {a:.3f}")

# ---------- 2b. per-degradation-type AUROC ----------
print("\n  --- AUROC broken down by degradation type (vs originals) ---")
for deg in ["clip","noise_lowsnr","reverb","speed"]:
    if deg=="noise_lowsnr":
        bd=[f for cid,f,g in rows if g["aug"]=="noise" and g["expect_low_quality"]]
        label="noise(5/12dB)"
    elif deg=="noise_hi":
        continue
    else:
        bd=[f for cid,f,g in rows if g["aug"]==deg]
        label=deg
    if not bd: continue
    line=f"  {label:14s} (n={len(bd):4d})  "
    for m in METRICS:
        line+=f"{m}={auc([f[m] for f in good],[f[m] for f in bd]):.2f} "
    print(line)

# noise by SNR
print("\n  --- noise AUROC by SNR bucket (vs originals) ---")
snr_buckets=defaultdict(list)
for cid,f,g in rows:
    if g["aug"]=="noise":
        snr=g.get("params",{}).get("snr_db")
        snr_buckets[snr].append(f)
for snr in sorted(snr_buckets, key=lambda x:(x is None,x)):
    bd=snr_buckets[snr]
    line=f"  SNR={str(snr):6s} (n={len(bd):4d})  "
    for m in ["dnsmos","si_sdr","pesq","stoi","utmos","PQ","CE"]:
        line+=f"{m}={auc([f[m] for f in good],[f[m] for f in bd]):.2f} "
    print(line)

# ---------- 3. inter-metric correlation ----------
print("\n"+"="*100)
print("3. PEARSON CORRELATION BETWEEN METRICS (all cuts) — high |r| => redundant")
print("="*100)
allrows=[f for cid,f,g in rows]
print("        "+"".join(m.rjust(8) for m in METRICS))
for m1 in METRICS:
    line=m1.ljust(8)
    for m2 in METRICS:
        r=pearson([f[m1] for f in allrows],[f[m2] for f in allrows])
        line+=f"{r:8.2f}"
    print(line)

# ---------- 4. simple combination strategies ----------
print("\n"+"="*100)
print("4. COMBINATION STRATEGIES — AUROC for separating degraded (clip+lowSNR noise) from originals")
print("="*100)

# z-score normalize each metric over all rows
stats={m:(mean([f[m] for f in allrows]), std([f[m] for f in allrows])) for m in METRICS}
def z(f,m):
    mu,sd=stats[m]; return (f[m]-mu)/sd if sd>0 else 0.0

def combo_auc(fn):
    return auc([fn(f) for f in good],[fn(f) for f in bad])

combos = {
 "dnsmos only":            lambda f: f["dnsmos"],
 "si_sdr only":            lambda f: f["si_sdr"],
 "mean z(all 9)":          lambda f: sum(z(f,m) for m in METRICS),
 "mean z(dnsmos,sisdr,pesq,utmos)": lambda f: z(f,"dnsmos")+z(f,"si_sdr")+z(f,"pesq")+z(f,"utmos"),
 "mean z(dnsmos,sisdr)":   lambda f: z(f,"dnsmos")+z(f,"si_sdr"),
 "min z(dnsmos,sisdr,pesq,utmos)":  lambda f: min(z(f,"dnsmos"),z(f,"si_sdr"),z(f,"pesq"),z(f,"utmos")),
 "min z(all 9)":           lambda f: min(z(f,m) for m in METRICS),
 "min z(dnsmos,sisdr)":    lambda f: min(z(f,"dnsmos"),z(f,"si_sdr")),
}
for name,fn in combos.items():
    print(f"  {name:38s} AUROC = {combo_auc(fn):.3f}")
