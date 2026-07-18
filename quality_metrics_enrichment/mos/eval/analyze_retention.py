"""Family-level 'keep the best copy' test — the actual retention decision.
For each duplicate family (>=2 members), rank members by each scoring scheme and
check the top-1 ('kept') member: is it low-quality? is its audio clean?
"""
import json, math
from collections import defaultdict

GT  = "/capstor/scratch/cscs/sgodey/dedup_pipeline_test/synthetic_shar/cv22_synth/en/ground_truth.jsonl"
MOS = "/capstor/scratch/cscs/sgodey/dedup_pipeline_test/mos_synth/mos_rank_0000.jsonl"

gt = {json.loads(l)["cut_id"]: json.loads(l) for l in open(GT)}
rows = []
for l in open(MOS):
    d = json.loads(l); cid = d["cut_id"]
    if cid not in gt: continue
    m = d["metrics"]
    f = {"utmos":m["utmos"]["score"]["utmos"],
         "stoi":m["squim"]["score"]["stoi"],"pesq":m["squim"]["score"]["pesq"],
         "si_sdr":m["squim"]["score"]["si_sdr"],"dnsmos":m["dnsmos_nisqa"]["score"]["nisqa"],
         "CE":m["audiobox"]["score"]["CE"],"CU":m["audiobox"]["score"]["CU"],
         "PC":m["audiobox"]["score"]["PC"],"PQ":m["audiobox"]["score"]["PQ"]}
    rows.append((cid, f, gt[cid]))

METRICS=["utmos","stoi","pesq","si_sdr","dnsmos","CE","CU","PC","PQ"]
def mean(xs): return sum(xs)/len(xs)
def std(xs):
    mu=mean(xs); return math.sqrt(sum((x-mu)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else 1.0
allf=[f for _,f,_ in rows]
st={m:(mean([f[m] for f in allf]),std([f[m] for f in allf])) for m in METRICS}
def z(f,m): mu,sd=st[m]; return (f[m]-mu)/sd if sd else 0.0

# audiobox OVL exactly as retention.py computes it (includes PC, positive sign)
def aes_ovl(f): return (f["CE"]+f["CU"]+f["PC"]+f["PQ"])/4.0

schemes = {
 "current retention (0.4u+0.3d+0.2aesOVL, raw)": lambda f: 0.4*f["utmos"]+0.3*f["dnsmos"]+0.2*aes_ovl(f),
 "current but aesOVL w/o PC":                    lambda f: 0.4*f["utmos"]+0.3*f["dnsmos"]+0.2*((f["CE"]+f["CU"]+f["PQ"])/3.0),
 "dnsmos only":                                  lambda f: f["dnsmos"],
 "PQ only":                                      lambda f: f["PQ"],
 "z(dnsmos)+z(si_sdr)+z(utmos)+z(pesq)":         lambda f: z(f,"dnsmos")+z(f,"si_sdr")+z(f,"utmos")+z(f,"pesq"),
 "z(dnsmos)+z(si_sdr)":                          lambda f: z(f,"dnsmos")+z(f,"si_sdr"),
 "min z(dnsmos,si_sdr,utmos,pesq)":              lambda f: min(z(f,"dnsmos"),z(f,"si_sdr"),z(f,"utmos"),z(f,"pesq")),
}

fams=defaultdict(list)
for cid,f,g in rows: fams[g["family_id"]].append((cid,f,g))
multi={k:v for k,v in fams.items() if len(v)>=2}
print(f"{len(multi)} families with >=2 members "
      f"(avg size {mean([len(v) for v in multi.values()]):.1f})\n")

# A family member's audio is 'clean' if not expect_low_quality AND not reverb
def is_clean(g): return (not g["expect_low_quality"]) and g["aug"]!="reverb"
def is_lowq(g):  return g["expect_low_quality"]

print("Scheme: of the kept (top-1) copy per family —")
print(f"{'scheme':48s}{'%kept low-Q':>12s}{'%kept clean':>12s}{'%kept reverb':>13s}")
for name,fn in schemes.items():
    klow=kclean=krev=0; n=0
    for fid,mem in multi.items():
        # only families that actually contain a degraded option (else trivial)
        kept=max(mem, key=lambda t: fn(t[1]))
        g=kept[2]; n+=1
        klow  += is_lowq(g)
        kclean+= is_clean(g)
        krev  += (g["aug"]=="reverb")
    print(f"{name:48s}{100*klow/n:11.1f}%{100*kclean/n:11.1f}%{100*krev/n:12.1f}%")

# restrict to the HARD families: those that contain a clip or noise member
hard={k:v for k,v in multi.items() if any(m[2]["expect_low_quality"] for m in v)}
print(f"\nRestricted to {len(hard)} families that contain >=1 low-quality (clip/lowSNR-noise) member:")
print(f"{'scheme':48s}{'%kept low-Q':>12s}{'%kept clean':>12s}")
for name,fn in schemes.items():
    klow=kclean=0; n=0
    for fid,mem in hard.items():
        kept=max(mem, key=lambda t: fn(t[1])); g=kept[2]; n+=1
        klow+=is_lowq(g); kclean+=is_clean(g)
    print(f"{name:48s}{100*klow/n:11.1f}%{100*kclean/n:11.1f}%")
