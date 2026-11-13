"""Stage H, neural arm: learn the state decomposition instead of clustering.

The classical arm (`fah_rbd_stage_h.py`) builds the state model the standard
way -- TICA for the slow directions, k-means to partition them -- and both steps
are linear. If the slow coordinate is a nonlinear function of the dihedrals, TICA
has to approximate it with a linear combination and the clustering cannot
recover what was lost. A VAMPnet replaces both with one network trained to
maximise the VAMP-2 score, which is a variational bound on the slow subspace
rather than a proxy for it.

That matters here because of what MD_EMBEDDER.md section 8.2 records about the
nested autoencoder: between two configurations its *reconstruction* improved
while its ability to separate systems got worse. Reconstruction and variance
are the wrong objectives, and PCA keeps tying because it optimises the same
wrong thing. VAMP-2 is the first objective tried in this package that is aimed
at the dynamics directly.

The network is fed the **shared linear pre-reduction**, not TICA's output: being
handed TICA's coordinates would concede in advance the step it exists to
replace. Everything else is the frozen protocol -- stage G's seed-0 split, the
same masking, the same column selection, standardisation, PCA to d = 32, and
the same held-out scoring -- so these arms are directly comparable to every arm
in stages G and H.

Three disciplines, stated before the numbers:

- **Every fit is repeated over three seeds and reported as mean +- sd.** A
  neural result from a single seed is not a measurement, and the spread here is
  the thing a reader needs in order to judge whether a win is a win.
- **The grid is pre-registered**: ``n_states`` in {10, 50}, seeds {0, 1, 2}, one
  architecture. No search. Stage G's note stands -- tuning until the new method
  wins is how one manufactures a result.
- **Early stopping never touches the held-out runs.** `fit_vampnet` splits the
  *training* runs again for its own validation signal. The 111 held-out runs
  take part in no fitting decision of any kind, including when to stop.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sawnergy.md.states import MarkovStates  # noqa: E402
from sawnergy.md.vampnet import (VampConfig, fit_vampnet,  # noqa: E402
                                 soft_populations)

import fah_rbd_stage_g as g  # noqa: E402  -- the frozen protocol
import fah_rbd_stage_h as h  # noqa: E402  -- the classical arm

OUT = Path(g.DATA) / "stage_h"
N_STATES_GRID = (10, 50)
SEEDS = (0, 1, 2)


def embed(X, tr, d_embed):
    """Stage G's reduction, verbatim: select, standardise, PCA, all on train."""
    sel = g._select(X[tr])
    X = X[:, sel]
    Xs = g._standardise(X[tr], X)
    mu = Xs[tr].mean(0)
    Vt = np.linalg.svd(Xs[tr] - mu, full_matrices=False)[2][:d_embed]
    return (Xs - mu) @ Vt.T, X.shape[1]


def paired_bootstrap(Za, Zb, systems, *, a="WT", b="K417V", n_boot=2000,
                     seed=0):
    """Is ``Zb``'s separation ratio really above ``Za``'s, or is it the split?

    The arms in this stage are all scored on the **same** 111 held-out runs, so
    their separation ratios are paired and the uncertainty on the *difference*
    is much smaller than the uncertainty on either one alone. That matters: with
    about 28 runs per system the standard error of a single ``d`` is of order
    0.28, which is larger than every gain this stage reports. Quoting those
    marginal errors would make any result look like noise, and ignoring the
    question entirely would make noise look like a result.

    So the held-out runs are resampled with replacement, both arms are rescored
    on the identical resample, and the difference is accumulated. The returned
    fraction below zero is a one-sided bootstrap p-value for "the gain is zero
    or negative".
    """
    rng = np.random.default_rng(seed)
    idx_a = np.flatnonzero(systems == a)
    idx_b = np.flatnonzero(systems == b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(idx_a, size=idx_a.size, replace=True)
        rb = rng.choice(idx_b, size=idx_b.size, replace=True)
        take = np.concatenate([ra, rb])
        lab = systems[take]
        try:
            deltas[i] = (g.separation_ratio(Zb[take], lab, a, b)
                         - g.separation_ratio(Za[take], lab, a, b))
        except ValueError:  # a resample with <2 runs of a system
            deltas[i] = np.nan
    deltas = deltas[np.isfinite(deltas)]
    return {"n_boot": int(deltas.size),
            "mean_delta": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5))],
            "p_delta_le_0": float((deltas <= 0).mean())}


def summarise(per_seed, label):
    """Mean +- sd of each scalar metric across seeds."""
    keys = ("loro", "span_over_floor")
    out = {"n_seeds": len(per_seed)}
    for k in keys:
        v = np.array([r[k] for r in per_seed])
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    for contrast in ("WT|N439K", "WT|K417V", "WT|N439K+K417V"):
        v = np.array([r["separations"][contrast] for r in per_seed])
        out[contrast] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    for contrast in ("N439K", "K417V"):
        v = np.array([r["auc"][contrast] for r in per_seed])
        out[f"auc_{contrast}"] = {"mean": float(v.mean()),
                                  "sd": float(v.std(ddof=1))}
    print(f"  {label:22s} LORO {out['loro']['mean']:.4f}+-"
          f"{out['loro']['sd']:.4f}  "
          f"d(N439K) {out['WT|N439K']['mean']:.3f}+-"
          f"{out['WT|N439K']['sd']:.3f}  "
          f"d(K417V) {out['WT|K417V']['mean']:.3f}+-"
          f"{out['WT|K417V']['sd']:.3f}  "
          f"AUC_K417V {out['auc_K417V']['mean']:.3f}+-"
          f"{out['auc_K417V']['sd']:.3f}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 110)
    print("STAGE H (neural)  VAMPnet state decomposition vs TICA + k-means "
          "vs CA occupancy")
    print("=" * 110)

    blocks, systems, clones, res_seq = g.load()
    n = len(systems)
    print(f"\nloaded {n} runs: " + "  ".join(
        f"{s} {int((systems == s).sum())}" for s in g.SYSTEMS))

    pair_all, pair_masked, res_keep, n_dropped = g.masks(
        blocks, systems, res_seq)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(round(g.VAL_FRAC * n))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    print(f"split: {tr.size} train / {va.size} held out (stage G's split)")

    results: dict[str, dict] = {}
    reports: dict[str, list] = {}
    for arm_label, pair_keep, rkeep in (
            ("unmasked", pair_all, np.ones(g.N_RES, dtype=bool)),
            ("masked", pair_masked, res_keep)):
        print("\n" + "-" * 110)
        print(f"{arm_label.upper()}"
              + ("  <- the result" if arm_label == "masked"
                 else "  <- upper bound"))
        print("-" * 110)

        pre, _ = h.pre_coordinates(blocks["dih_frames"], rkeep, tr)
        tica_frames, tica = h.frame_coordinates(blocks["dih_frames"], rkeep, tr)

        # --- baselines, through the identical pipeline ----------------------
        held: dict[str, np.ndarray] = {}
        Xca = g.build_arm("ca", blocks, pair_keep, rkeep, tr,
                          np.stack([c.mean(0) for c in tica_frames]))
        # `ca` at d = 32 and again at d = 10. The second is not padding: a
        # 10-state arm yields 10 columns and is therefore scored at d = 10,
        # and stage H showed that the dimension alone moves CA a long way
        # (d(N439K) 0.552 -> 0.850 masked), because the measurement floor grows
        # as sqrt(d) while the signal does not. Comparing vamp10 against `ca`
        # at d = 32 would credit the network with a dimension change.
        for d_embed, name in ((g.D_EMBED, "ca"), (10, "ca@10")):
            Z, w = embed(Xca, tr, d_embed)
            held[name] = Z[va]
            results[f"{arm_label}/{name}"] = g.metrics(
                Z[va], systems[va], f"{name}  ({w}d -> {Z.shape[1]})")

        ms = MarkovStates.fit([tica_frames[i] for i in tr], k=50,
                              seed=h.KMEANS_SEED, n_init=h.KMEANS_INIT)
        Xmsm = np.stack([ms.populations(c, soft=True, hellinger=True)
                         for c in tica_frames])
        Z, w = embed(Xmsm, tr, g.D_EMBED)
        held["msm50"] = Z[va]
        results[f"{arm_label}/msm50"] = g.metrics(
            Z[va], systems[va], f"msm50  ({w}d -> {Z.shape[1]})")

        # --- the neural arms ------------------------------------------------
        for k in N_STATES_GRID:
            per_seed, fit_reports, seed_held = [], [], []
            for seed in SEEDS:
                cfg = VampConfig(n_states=k, lag=h.TICA_LAG, seed=seed)
                model, std, rep = fit_vampnet([pre[i] for i in tr], cfg)
                fit_reports.append({
                    "seed": seed, "train_score": rep.train_score,
                    "val_score": rep.val_score, "max_score": rep.max_score,
                    "collapsed": rep.collapsed, "restarts": rep.restarts,
                    "best_epoch": rep.best_epoch, "epochs_run": rep.epochs_run})
                print(f"    k={k} seed={seed}  {rep}")
                X = np.stack([soft_populations(model, std, c, hellinger=True)
                              for c in pre])
                Z, w = embed(X, tr, min(g.D_EMBED, X.shape[1]))
                seed_held.append(Z[va])
                per_seed.append(g.metrics(
                    Z[va], systems[va], f"vamp{k} s{seed}  ({w}d -> "
                                        f"{Z.shape[1]})"))
            # the dimension-matched baseline for this k
            ref = "ca@10" if min(g.D_EMBED, k) == 10 else "ca"
            boots = [paired_bootstrap(held[ref], Zv, systems[va])
                     for Zv in seed_held]
            print(f"  vamp{k} vs {ref} on d(WT|K417V), paired bootstrap over "
                  f"the held-out runs:")
            for seed, bs in zip(SEEDS, boots):
                print(f"    seed {seed}  delta {bs['mean_delta']:+.3f}  "
                      f"95% CI [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]  "
                      f"p(delta<=0) {bs['p_delta_le_0']:.4f}")
            results[f"{arm_label}/vamp{k}"] = {
                "per_seed": per_seed,
                "baseline": ref,
                "bootstrap_vs_baseline": boots,
                "summary": summarise(per_seed, f"vamp{k} (mean of "
                                               f"{len(SEEDS)} seeds)")}
            reports[f"{arm_label}/vamp{k}"] = fit_reports

    # --- the comparison that answers the question ---------------------------
    print("\n" + "=" * 110)
    print("DOES A LEARNED DECOMPOSITION BEAT A CLUSTERED ONE?  masked arm")
    print("=" * 110)
    ca = results["masked/ca"]
    msm = results["masked/msm50"]
    print(f"  {'ca (baseline)':22s} LORO {ca['loro']:.4f}          "
          f"d(N439K) {ca['separations']['WT|N439K']:.3f}        "
          f"d(K417V) {ca['separations']['WT|K417V']:.3f}        "
          f"AUC_K417V {ca['auc']['K417V']:.3f}")
    print(f"  {'msm50 (TICA+kmeans)':22s} LORO {msm['loro']:.4f}          "
          f"d(N439K) {msm['separations']['WT|N439K']:.3f}        "
          f"d(K417V) {msm['separations']['WT|K417V']:.3f}        "
          f"AUC_K417V {msm['auc']['K417V']:.3f}")
    print(f"  {'ca@10 (dim-matched)':22s} LORO "
          f"{results['masked/ca@10']['loro']:.4f}          "
          f"d(N439K) {results['masked/ca@10']['separations']['WT|N439K']:.3f}"
          f"        "
          f"d(K417V) {results['masked/ca@10']['separations']['WT|K417V']:.3f}"
          f"        "
          f"AUC_K417V {results['masked/ca@10']['auc']['K417V']:.3f}")
    rows = []
    for k in N_STATES_GRID:
        r = results[f"masked/vamp{k}"]
        s = r["summary"]
        print(f"  {'vamp' + str(k):22s} LORO {s['loro']['mean']:.4f}+-"
              f"{s['loro']['sd']:.4f}  "
              f"d(N439K) {s['WT|N439K']['mean']:.3f}+-{s['WT|N439K']['sd']:.3f}  "
              f"d(K417V) {s['WT|K417V']['mean']:.3f}+-{s['WT|K417V']['sd']:.3f}  "
              f"AUC_K417V {s['auc_K417V']['mean']:.3f}+-"
              f"{s['auc_K417V']['sd']:.3f}")
        ref = results[f"masked/{r['baseline']}"]
        boots = r["bootstrap_vs_baseline"]
        rows.append({
            "arm": f"vamp{k}", "baseline": r["baseline"],
            "d_K417V": s["WT|K417V"]["mean"], "d_K417V_sd": s["WT|K417V"]["sd"],
            "d_K417V_delta_vs_baseline": (s["WT|K417V"]["mean"]
                                          - ref["separations"]["WT|K417V"]),
            "d_K417V_delta_vs_ca32": (s["WT|K417V"]["mean"]
                                      - ca["separations"]["WT|K417V"]),
            "d_K417V_delta_vs_msm": (s["WT|K417V"]["mean"]
                                     - msm["separations"]["WT|K417V"]),
            "bootstrap_p_worst": max(b["p_delta_le_0"] for b in boots),
            "bootstrap_ci_low_worst": min(b["ci95"][0] for b in boots),
            "loro": s["loro"]["mean"], "loro_sd": s["loro"]["sd"],
            "loro_delta_vs_baseline": s["loro"]["mean"] - ref["loro"],
            "auc_K417V": s["auc_K417V"]["mean"],
            "auc_K417V_delta_vs_baseline": (s["auc_K417V"]["mean"]
                                            - ref["auc"]["K417V"])})
    print("\n  against the dimension-matched baseline, worst seed's bootstrap:")
    for r in rows:
        print(f"    {r['arm']:8s} vs {r['baseline']:6s}  "
              f"d(K417V) {r['d_K417V_delta_vs_baseline']:+.3f}  "
              f"LORO {r['loro_delta_vs_baseline']:+.4f}  "
              f"AUC {r['auc_K417V_delta_vs_baseline']:+.3f}  "
              f"p(no gain) <= {r['bootstrap_p_worst']:.4f}  "
              f"CI low >= {r['bootstrap_ci_low_worst']:+.3f}")

    payload = {"n_runs": int(n), "train": tr.tolist(), "val": va.tolist(),
               "n_states_grid": list(N_STATES_GRID), "seeds": list(SEEDS),
               "config": VampConfig().__dict__ | {"lag": h.TICA_LAG},
               "fit_reports": reports, "results": results,
               "comparison": rows}
    json.dump(payload, open(OUT / "stage_h_vamp.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'stage_h_vamp.json'}")


if __name__ == "__main__":
    main()
