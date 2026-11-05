"""Stage E: the floor scales as sqrt(d), so does cutting dimension help?

Stage D found the WT measurement floor sits at ~sqrt(d) and the entire
WT-to-double-mutant signal spans less than that floor. Since the floor is a
property of the dimension and the signal is a property of the data, a lower
dimensional embedding should widen the ratio -- unless the signal is spread
across many components, in which case truncation throws it away too. Only a
scan settles it.
"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(REPO, "src"))
from sawnergy.md import WTReference

z = np.load(f"{DATA}/stage_bc/embeddings.npz", allow_pickle=False)
systems = z["systems"].astype(str)
BASE = "N439K+K417V"
out = {}

for key, label in (("Z_pca", "PCA"), ("Z_ae", "nested AE")):
    Zfull = z[key]
    print(f"\n{'=' * 74}\n{label}: effect of embedding dimension\n{'=' * 74}")
    print(f"{'d':>4s} {'floor sqrt(d)':>14s} {'D_WT':>7s} {'span':>7s} "
          f"{'span/floor':>11s} {'AUC N439K':>10s} {'AUC K417V':>10s}")
    rows = []
    for d in (2, 4, 8, 12, 16, 24, 32):
        if d > Zfull.shape[1]:
            continue
        Z = Zfull[:, :d]
        rng = np.random.default_rng(0)
        wt = np.flatnonzero(systems == "WT"); rng.shuffle(wt)
        h = wt.size // 2
        ref = WTReference.fit(Z[wt[:h]])
        D = {s: np.asarray(ref.distance(
                Z[wt[h:] if s == "WT" else np.flatnonzero(systems == s)])).ravel()
             for s in ("WT", "N439K", "K417V", BASE)}
        d_wt, d_base = D["WT"].mean(), D[BASE].mean()
        span = d_base - d_wt
        aucs = {}
        for s in ("N439K", "K417V"):
            a, b = D["WT"], D[s]
            aucs[s] = float((a[:, None] < b[None, :]).mean())
        print(f"{d:4d} {np.sqrt(d):14.3f} {d_wt:7.3f} {span:7.3f} "
              f"{span/d_wt:11.3f} {aucs['N439K']:10.3f} {aucs['K417V']:10.3f}")
        rows.append({"d": d, "floor": float(np.sqrt(d)), "D_WT": float(d_wt),
                     "span": float(span), "ratio": float(span/d_wt),
                     "auc": aucs})
    out[key] = rows
    best = max(rows, key=lambda r: r["ratio"])
    print(f"  best span/floor at d={best['d']}: {best['ratio']:.3f} "
          f"(vs {rows[-1]['ratio']:.3f} at d={rows[-1]['d']})")

Path(f"{DATA}/stage_e").mkdir(exist_ok=True)
json.dump(out, open(f"{DATA}/stage_e/stage_e.json", "w"), indent=2)
print(f"\nwrote {DATA}/stage_e/stage_e.json")
