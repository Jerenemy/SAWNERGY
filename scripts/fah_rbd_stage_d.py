"""Stage D: fix the restoration metric's floor, and test ordering without
assuming mutations are additive.

Two problems surfaced in stage C.

**1. The restoration score has no zero.** ``rescue_score`` computes
``R = 1 - D/D_baseline``, which assumes a wild-type run sits at ``D = 0``. It
cannot. The Mahalanobis distance of a sample from its *own* d-dimensional
distribution follows a chi distribution with mean ~sqrt(d), so in 32 dimensions
wild type sits at ~5.66 by construction. Stage C measured 5.50 (PCA) and 4.84
(AE) for held-out WT runs -- the floor, not a deficit. Dividing by a baseline
that also contains that floor compresses the whole scale: WT scored +0.43
instead of +1.0. The fix is to rescale between the two measured anchors,

    R = (D_baseline - D) / (D_baseline - D_WT)

so wild type is 1 by construction and the untreated baseline is 0.

**2. The dose-ordering test assumed additivity.** It failed because N439K alone
sits *further* from WT than the double mutant carrying N439K and K417V
together. That is epistasis, not necessarily a broken metric: nothing requires
two substitutions to perturb a contact ensemble additively. So the ordering
test as written cannot distinguish "the metric is wrong" from "the assumption
is wrong". This stage replaces it with a test that does not need additivity:
project every run onto the WT->double discriminant axis and ask where the
single mutants fall. A metric that measures perturbation along a defined
deficit axis should place partial variants between the anchors even when their
total distances are non-additive.
"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, os.path.join(REPO, "src"))
from sawnergy.md import WTReference  # noqa: E402

IN = Path(f"{DATA}/stage_bc/embeddings.npz")
OUT = Path(f"{DATA}/stage_d")
SYSTEMS = ("WT", "N439K", "K417V", "N439K+K417V")
BASELINE = "N439K+K417V"
SEED = 0


def boot_ci(x, rng, n=4000):
    b = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyse(Z, systems, label, *, seed=SEED):
    rng = np.random.default_rng(seed)
    d = Z.shape[1]

    # --- split WT so its own distance is measured out of sample --------------
    wt_idx = np.flatnonzero(systems == "WT")
    rng.shuffle(wt_idx)
    half = wt_idx.size // 2
    ref = WTReference.fit(Z[wt_idx[:half]])
    idx_of = {s: (wt_idx[half:] if s == "WT" else np.flatnonzero(systems == s))
              for s in SYSTEMS}
    D = {s: np.asarray(ref.distance(Z[i])).ravel() for s, i in idx_of.items()}

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"\nexpected floor for d={d}: E[chi_{d}] ~ {np.sqrt(d):.3f}")
    print(f"measured WT floor (out of sample): {D['WT'].mean():.3f}")
    print(f"  -> {'consistent with the floor' if abs(D['WT'].mean() - np.sqrt(d)) < 1.5 else 'NOT the floor alone'}")

    # --- floor-corrected restoration ----------------------------------------
    d_wt, d_base = float(D["WT"].mean()), float(D[BASELINE].mean())
    span = d_base - d_wt
    print(f"\nanchors: D_WT={d_wt:.3f}  D_baseline({BASELINE})={d_base:.3f}  "
          f"span={span:.3f}")
    print(f"  dynamic range / floor = {span / d_wt:.2f}"
          + ("   (narrow: the floor dominates)" if span / d_wt < 1 else ""))

    print(f"\n{'system':14s} {'n':>4s} {'mean D':>8s} {'R_raw':>8s} "
          f"{'R_floor':>9s}  {'95% CI of R_floor':>22s}")
    rows = {}
    for s in SYSTEMS:
        x = D[s]
        r_raw = 1.0 - x / d_base
        r_floor = (d_base - x) / span
        lo, hi = boot_ci(r_floor, rng)
        rows[s] = {"n": int(x.size), "D": float(x.mean()),
                   "D_sd": float(x.std()), "R_raw": float(r_raw.mean()),
                   "R_floor": float(r_floor.mean()),
                   "R_floor_ci": [lo, hi]}
        print(f"{s:14s} {x.size:4d} {x.mean():8.3f} {r_raw.mean():+8.3f} "
              f"{r_floor.mean():+9.3f}  [{lo:+8.3f}, {hi:+8.3f}]")

    # --- ordering along the WT -> baseline axis, no additivity assumed ------
    mu_wt = Z[idx_of["WT"]].mean(0)
    mu_base = Z[idx_of[BASELINE]].mean(0)
    axis = mu_base - mu_wt
    axis /= np.linalg.norm(axis) + 1e-12
    print(f"\nprojection onto the WT -> {BASELINE} axis "
          f"(0 = WT centroid, 1 = baseline centroid)")
    scale = float((mu_base - mu_wt) @ axis)
    proj = {}
    for s in SYSTEMS:
        p = ((Z[idx_of[s]] - mu_wt) @ axis) / scale
        lo, hi = boot_ci(p, rng)
        proj[s] = {"mean": float(p.mean()), "ci": [lo, hi]}
        print(f"  {s:14s} {p.mean():+7.3f}  [{lo:+7.3f}, {hi:+7.3f}]")

    singles_between = all(
        0.0 < proj[s]["mean"] < 1.0 for s in ("N439K", "K417V"))
    print(f"  single mutants fall between the anchors: "
          f"{'YES' if singles_between else 'NO'}")

    # --- epistasis: is the double the sum of the singles? -------------------
    v1 = Z[idx_of["N439K"]].mean(0) - mu_wt
    v2 = Z[idx_of["K417V"]].mean(0) - mu_wt
    v12 = mu_base - mu_wt
    additive = v1 + v2
    cos = float(additive @ v12 / ((np.linalg.norm(additive) *
                                   np.linalg.norm(v12)) + 1e-12))
    ratio = float(np.linalg.norm(v12) / (np.linalg.norm(additive) + 1e-12))
    print(f"\nadditivity check  |v1+v2| vs |v12|: ratio {ratio:.3f}, "
          f"cos angle {cos:+.3f}")
    print("  " + ("additive" if abs(ratio - 1) < 0.25 and cos > 0.9
                  else "NOT additive -- the double mutant is not the sum of "
                       "its parts, so a dose-ordering test on total distance "
                       "is the wrong test"))

    # --- can a single run be classified as rescued or not? -------------------
    print("\nper-run discrimination along R_floor "
          "(can one run be called rescued?)")
    for s in ("N439K", "K417V"):
        a = (d_base - D["WT"]) / span
        b = (d_base - D[s]) / span
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        dprime = abs(a.mean() - b.mean()) / (pooled + 1e-12)
        # AUC of separating WT from this system on R alone
        auc = float((a[:, None] > b[None, :]).mean())
        print(f"  WT vs {s:14s} d'={dprime:5.2f}  AUC={auc:.3f}")

    return {"floor_expected": float(np.sqrt(d)), "rows": rows,
            "projection": proj, "singles_between": bool(singles_between),
            "additivity": {"ratio": ratio, "cos": cos},
            "anchors": {"D_WT": d_wt, "D_baseline": d_base, "span": span}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(IN, allow_pickle=False)
    systems = z["systems"].astype(str)
    out = {}
    for key, label in (("Z_pca", "PCA(32) over run-mean contacts"),
                       ("Z_ae", "Nested autoencoder z (d_traj=32)")):
        out[key] = analyse(z[key], systems, label)
    with open(OUT / "stage_d.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT / 'stage_d.json'}")


if __name__ == "__main__":
    main()
