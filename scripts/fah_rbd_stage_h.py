"""Stage H: does the *joint* structure of the ensemble carry what marginals miss?

Stage G varied what was measured and held everything else fixed, and the answer
was no: of eight feature sets, none beat time-averaged CA-CA contact occupancy
on K417V, the contrast that matters, and concatenating all of them was worse
than CA alone. But every one of those eight arms was a **marginal**: a per-pair
or per-residue average over frames. Marginals are blind to correlation. Two
contacts each formed half the time look identical whether the run alternates
between *both formed* and *neither formed*, or between *one* and *the other* --
and those are different conformational ensembles.

This stage swaps the descriptor for a Markov state model. Frames are projected
onto the slow modes, clustered once into a shared set of discrete states, and
each run is described by the fraction of its time spent in each state. Still
order-free, so it cannot latch onto the run-specific drift that distinguishes
replicates, but it is a description of the joint distribution rather than of
each coordinate separately.

The arm grid is **pre-registered here and reported in full**, including the arms
that lose. Stage G's section 8.1 note applies: searching hyperparameters until
the new method wins is how one manufactures a result.

    ca                    CA-CA contact occupancy      -- the standing baseline
    ca@d                  the same, reduced to d        -- dimension controls
    tica                  stage G's slow-mode arm      -- the nearest relative
    msm{k}                soft state populations, k states, Hellinger
    msm50-hard            hard assignment              -- does softening help?
    msm50-raw             populations without sqrt     -- does Hellinger help?
    msm10+trans           populations plus transition rates
    ca+msm50              does it add anything to CA, rather than beat it?

The protocol is not re-implemented: the loading, masking, train/held-out split,
column selection, standardisation, PCA and scoring are all imported from
`fah_rbd_stage_g`, so the arms here are directly comparable to the arms there
and cannot drift from them.

Two honest caveats, stated before the numbers:

- **The dimension is not free.** Stage E found that the measurement floor grows
  as sqrt(d) while the signal does not, so d = 12 beat d = 32 for PCA. A k-state
  population vector has k dimensions before reduction, so a small-k arm is
  confounded with a dimension change. The ``ca@d`` arms exist to break that
  confound.
- **90 frames is few.** A population is a multinomial proportion with standard
  error sqrt(p(1-p)/T). At T = 90 and k = 50 that error is the same order as
  the population itself, and it lands in the denominator of every separation
  ratio. A null result here is at least as likely to be a statement about the
  sampling depth as about the joint structure -- which is what stage G
  concluded by a different route.
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

from sawnergy.md.states import MarkovStates, multinomial_noise  # noqa: E402
from sawnergy.md.temporal import TICA  # noqa: E402

import fah_rbd_stage_g as g  # noqa: E402  -- the frozen protocol

OUT = Path(g.DATA) / "stage_h"
TICA_LAG = 5
TICA_PRE = 120
TICA_K = 32
K_GRID = (10, 25, 50, 100)
KMEANS_SEED = 0
KMEANS_INIT = 4


# ---------------------------------------------------------------------------
# per-frame slow-mode coordinates
# ---------------------------------------------------------------------------

def pre_coordinates(frames, res_keep, tr):
    """``[(T, TICA_PRE)]`` per-run frames after the shared linear pre-reduction.

    The masked dihedral columns reduced by an SVD fitted on **training frames
    only**. Stage G's `fit_tica` does this as an internal step because TICA has
    to invert a covariance in 1158 dimensions and the lagged covariance is
    rank-deficient there; it is factored out here because it is also the right
    input for anything that wants to learn its own coordinates rather than be
    handed TICA's.
    """
    cols = np.repeat(res_keep, 6)
    Ftr = [np.asarray(frames[i], dtype=np.float32)[:, cols] for i in tr]
    pool = np.concatenate(Ftr)
    mu = pool.mean(0)
    Vt = np.linalg.svd(pool - mu, full_matrices=False)[2][:TICA_PRE]
    out = [(np.asarray(F, dtype=np.float32)[:, cols] - mu) @ Vt.T
           for F in frames]
    return out, (mu, Vt)


def frame_coordinates(frames, res_keep, tr):
    """``[(T, TICA_K)]`` per-run slow-mode trajectories, fitted on train only.

    Identical preprocessing to stage G's `fit_tica` -- masked dihedral columns,
    an SVD down to `TICA_PRE` fitted on training frames, then TICA at
    `TICA_LAG` -- but it keeps the per-frame projection instead of collapsing
    each run to its mean. The mean is what stage G scored; the frames are what a
    state decomposition needs.
    """
    pre, _ = pre_coordinates(frames, res_keep, tr)
    tica = TICA.fit([pre[i] for i in tr], lag=TICA_LAG, n_components=TICA_K)
    return [tica.transform(P) for P in pre], tica


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def state_arms(coords, tr, *, verbose=True):
    """Build every MSM-derived design matrix. Clusterings use train frames only.

    Returns:
        ``(arms, diagnostics)`` where ``arms`` maps name -> ``(n_runs, cols)``.
    """
    arms, diag = {}, {}
    for k in K_GRID:
        ms = MarkovStates.fit([coords[i] for i in tr], k=k, seed=KMEANS_SEED,
                              n_init=KMEANS_INIT)
        soft_h = np.stack([ms.populations(c, soft=True, hellinger=True)
                           for c in coords])
        arms[f"msm{k}"] = soft_h
        pops = np.stack([ms.populations(c, soft=True) for c in coords])
        hard = np.stack([ms.populations(c, soft=False) for c in coords])
        # The multinomial standard error is the sampling noise of a *count*, so
        # it is only comparable with the hard-assignment scatter. Soft
        # assignment is a smoothed estimator of the same populations with a
        # smaller variance, which is why it is the default -- quoting the
        # closed-form count noise against the soft scatter would overstate the
        # problem by whatever the smoothing buys.
        se = multinomial_noise(hard[tr].mean(0), coords[0].shape[0])
        between_hard = hard[tr].std(0)
        between_soft = pops[tr].std(0)
        diag[f"msm{k}"] = {
            "k": k,
            "mean_population": float(hard[tr].mean(0).mean()),
            "between_run_sd_hard": float(between_hard.mean()),
            "between_run_sd_soft": float(between_soft.mean()),
            "mean_multinomial_se": float(se.mean()),
            # the share of the hard-assignment run-to-run scatter that is
            # counting noise rather than system difference
            "noise_fraction_hard": float(
                (se / np.maximum(between_hard, 1e-12)).mean()),
            "softening_variance_ratio": float(
                (between_soft / np.maximum(between_hard, 1e-12)).mean()),
            "occupied_states": int((hard[tr].mean(0) > 1e-4).sum()),
        }
        if verbose:
            d = diag[f"msm{k}"]
            print(f"  k={k:4d}  occupied {d['occupied_states']:4d}  "
                  f"between-run sd hard {d['between_run_sd_hard']:.5f} / "
                  f"soft {d['between_run_sd_soft']:.5f}  "
                  f"multinomial se {d['mean_multinomial_se']:.5f}  "
                  f"noise/scatter {d['noise_fraction_hard']:.2f}  "
                  f"softening {d['softening_variance_ratio']:.2f}x")
        if k == 50:
            arms["msm50-hard"] = np.stack(
                [ms.populations(c, soft=False, hellinger=True)
                 for c in coords])
            arms["msm50-raw"] = pops
        if k == 10:
            Tm = np.stack([ms.transition_matrix(c, lag=1).ravel()
                           for c in coords])
            arms["msm10+trans"] = np.hstack([soft_h, Tm])
    return arms, diag


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 110)
    print("STAGE H  joint state populations vs marginal time averages")
    print("=" * 110)

    blocks, systems, clones, res_seq = g.load()
    n = len(systems)
    counts = {s: int((systems == s).sum()) for s in g.SYSTEMS}
    print(f"\nloaded {n} runs: " + "  ".join(f"{k} {v}"
                                            for k, v in counts.items()))

    pair_all, pair_masked, res_keep, n_dropped = g.masks(
        blocks, systems, res_seq)
    print(f"pairs: {pair_all.sum()} unmasked, {pair_masked.sum()} masked "
          f"({n_dropped} of {g.N_RES} residues dropped)")

    # the same seed-0 split stage G used, so the held-out set is the same runs
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(round(g.VAL_FRAC * n))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    print(f"split: {tr.size} train / {va.size} held out (stage G's split)")

    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    for arm_label, pair_keep, rkeep in (
            ("unmasked", pair_all, np.ones(g.N_RES, dtype=bool)),
            ("masked", pair_masked, res_keep)):
        print("\n" + "-" * 110)
        print(f"{arm_label.upper()}"
              + ("  <- the result; the mutation's consequences, not its identity"
                 if arm_label == "masked" else "  <- upper bound"))
        print("-" * 110)

        coords, tica = frame_coordinates(blocks["dih_frames"], rkeep, tr)
        if arm_label == "masked":
            print(f"  TICA implied timescales (frames): "
                  f"{np.round(tica.timescales()[:5], 1)}")
        arms, diag = state_arms(coords, tr)
        diagnostics[arm_label] = diag

        # the marginal baselines, through the identical pipeline
        Ztica_run = np.stack([c.mean(axis=0) for c in coords])
        arms["ca"] = g.build_arm("ca", blocks, pair_keep, rkeep, tr, Ztica_run)
        arms["tica"] = Ztica_run
        arms["ca+msm50"] = np.hstack([arms["ca"], arms["msm50"]])

        order = (["ca", "tica"]
                 + [f"msm{k}" for k in K_GRID]
                 + ["msm50-hard", "msm50-raw", "msm10+trans", "ca+msm50"])
        # dimension controls: CA reduced to each width an MSM arm ends up with
        dims = sorted({min(g.D_EMBED, arms[a].shape[1]) for a in order
                       if a.startswith("msm")})
        for d in dims:
            if d < g.D_EMBED:
                order.insert(2, f"ca@{d}")

        for name in order:
            d_embed = g.D_EMBED
            if name.startswith("ca@"):
                d_embed, X = int(name[3:]), arms["ca"]
            else:
                X = arms[name]
                d_embed = min(g.D_EMBED, X.shape[1])
            sel = g._select(X[tr])
            X = X[:, sel]
            Xs = g._standardise(X[tr], X)
            mu = Xs[tr].mean(0)
            Vt = np.linalg.svd(Xs[tr] - mu, full_matrices=False)[2][:d_embed]
            Z = (Xs - mu) @ Vt.T
            results[f"{arm_label}/{name}"] = g.metrics(
                Z[va], systems[va], f"{name}  ({X.shape[1]}d -> {Z.shape[1]})")

    # --- the comparison that answers the question ---------------------------
    print("\n" + "=" * 110)
    print("DOES THE JOINT STRUCTURE HELP?  change against CA-only, masked arm")
    print("=" * 110)
    base = results["masked/ca"]
    rows = []
    for key in sorted(k for k in results if k.startswith("masked/")):
        name = key.split("/", 1)[1]
        if name == "ca":
            continue
        r = results[key]
        row = {"arm": name, "dim": r["dim"],
               "d_K417V": r["separations"]["WT|K417V"],
               "d_K417V_delta": (r["separations"]["WT|K417V"]
                                 - base["separations"]["WT|K417V"]),
               "d_N439K": r["separations"]["WT|N439K"],
               "loro": r["loro"],
               "loro_delta": r["loro"] - base["loro"],
               "need_K417V": r["replicates_required"]["WT|K417V"]}
        rows.append(row)
        print(f"  {name:14s} d(K417V) {row['d_K417V']:.3f} "
              f"({row['d_K417V_delta']:+.3f})   d(N439K) {row['d_N439K']:.3f}   "
              f"LORO {r['loro']:.4f} ({row['loro_delta']:+.4f})   "
              f"n {row['need_K417V']:6.0f}")
    print(f"\n  baseline CA: d(K417V) {base['separations']['WT|K417V']:.3f}   "
          f"d(N439K) {base['separations']['WT|N439K']:.3f}   "
          f"LORO {base['loro']:.4f}   "
          f"n {base['replicates_required']['WT|K417V']:.0f}")
    best = max(rows, key=lambda r: r["d_K417V"])
    print(f"  best on the undetectable contrast: {best['arm']} at "
          f"d = {best['d_K417V']:.3f} "
          f"({best['d_K417V_delta']:+.3f} vs CA)")

    payload = {"n_runs": int(n), "counts": counts,
               "train": tr.tolist(), "val": va.tolist(),
               "tica": {"lag": TICA_LAG, "pre": TICA_PRE, "k": TICA_K},
               "k_grid": list(K_GRID),
               "kmeans": {"seed": KMEANS_SEED, "n_init": KMEANS_INIT},
               "state_diagnostics": diagnostics,
               "results": results, "comparison": rows}
    json.dump(payload, open(OUT / "stage_h.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'stage_h.json'}")


if __name__ == "__main__":
    main()
