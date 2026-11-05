"""Stage B: nested autoencoder vs PCA on the RBD variant panel.
Stage C: can either one quantify restoration?

Stage B is the comparison the mdCATH work could not settle, because there the
autoencoder had to pool across proteins and lost residue identity in the
process. Here there is one protein and one numbering, so the autoencoder sees
full residue-pair contact vectors -- the same features PCA sees -- and the
comparison is finally apples to apples.

Stage C asks the question the whole p53 programme rests on. There is no rescue
compound in this dataset, but there is a dose axis: WT, two single mutants, and
the double mutant carrying both substitutions. If a contact-space embedding
measures sequence perturbation at all, Mahalanobis distance from the WT
reference must order

    WT  <  single mutants  <  double mutant

and the restoration score of a single mutant relative to the double must land
strictly between 0 and 1. Nothing about that ordering is guaranteed by the
training objective, so it is a real test rather than a restatement.
"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(REPO, "src"))

from sawnergy.md import (AEConfig, FitConfig, PCAEmbedder,  # noqa: E402
                         WTReference, build_nested_autoencoder, contact_map,
                         embed_runs, fit_nested, leave_one_replicate_out,
                         make_windows, rescue_score, separation_ratio,
                         seq_separation_mask, variance_components)

sys.path.insert(0, str(Path(__file__).parent))
from stage_a import KEEP_FRAMES, OUT as STAGE_A_OUT, load_runs  # noqa: E402

OUT = Path(f"{DATA}/stage_bc")
SYSTEMS = ("WT", "N439K", "K417V", "N439K+K417V")
FRAME_STRIDE = 2          # 180 -> 90 frames per run, 1 ns apart
WINDOW, WIN_STRIDE = 45, 23
VAL_FRAC = 0.25           # fraction of runs held out, split at RUN level
SEED = 0

#: Minimum between-run sd for a residue pair to be kept. Of the 18,145 pairs
#: with |i-j|>3, only ~2,000 ever vary between runs; the rest are pairs that
#: never come into contact in any replicate and sit at zero everywhere. They
#: carry no information and inflate the autoencoder's input width ninefold,
#: which is the same parameters-to-runs problem that sank the mdCATH attempt.
#: The threshold is applied to TRAINING runs only -- selecting features using
#: held-out runs would leak the very difference being measured.
MIN_PAIR_SD = 1e-3


# ---------------------------------------------------------------------------
# per-frame features
# ---------------------------------------------------------------------------

def frame_matrices(frames_list):
    """One ``(T, P)`` contact matrix per run, full residue-pair resolution."""
    n = frames_list[0].coords.shape[1]
    iu = np.triu_indices(n, k=1)
    keep = seq_separation_mask(n, 3)[iu]
    out, systems, reps = [], [], []
    for fr in frames_list:
        ca = fr.ca_coords()[::FRAME_STRIDE]
        c = contact_map(ca)
        out.append(c[:, iu[0], iu[1]][:, keep].astype(np.float32))
        systems.append(fr.run.system)
        reps.append(fr.run.replicate)
    return out, np.array(systems), np.array(reps)


def run_level_split(systems, reps, *, val_frac=VAL_FRAC, seed=SEED):
    """Hold out whole runs, stratified by system.

    A frame-level split here would report memorisation: consecutive frames of
    one trajectory are nearly identical, so a model that has seen frame 40 can
    reconstruct frame 41 without having learned anything transferable.
    """
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for s in np.unique(systems):
        idx = np.flatnonzero(systems == s)
        rng.shuffle(idx)
        k = max(2, int(round(val_frac * idx.size)))
        va.extend(idx[:k].tolist())
        tr.extend(idx[k:].tolist())
    return np.array(sorted(tr)), np.array(sorted(va))


# ---------------------------------------------------------------------------
# evaluation shared by both embedders
# ---------------------------------------------------------------------------

def evaluate(Z, systems, reps, label):
    """Separation, ICC and LORO for one run-level embedding."""
    vc = variance_components(Z, systems)
    disc = leave_one_replicate_out(Z, systems, n_permutations=0)
    seps = {}
    for a, b in (("WT", "N439K"), ("WT", "K417V"), ("WT", "N439K+K417V"),
                 ("N439K", "N439K+K417V"), ("K417V", "N439K+K417V")):
        if (systems == a).sum() >= 3 and (systems == b).sum() >= 3:
            seps[f"{a}|{b}"] = float(separation_ratio(Z, systems, a, b))
    print(f"\n  {label}")
    print(f"    dim              {Z.shape[1]}")
    print(f"    ICC(system)      {vc.icc:+.4f}  (p={vc.p_value:.4f})")
    print(f"    LORO accuracy    {disc.accuracy:.4f}  (chance {disc.chance:.4f})")
    for k, v in seps.items():
        print(f"    d({k:26s}) {v:6.3f}")
    return {"dim": int(Z.shape[1]), "icc": float(vc.icc),
            "p": float(vc.p_value), "loro": float(disc.accuracy),
            "chance": float(disc.chance), "separations": seps,
            "per_system_loro": {k: float(v) for k, v in disc.per_system.items()}}


# ---------------------------------------------------------------------------
# stage C: restoration
# ---------------------------------------------------------------------------

def restoration(Z, systems, label, *, n_boot=2000, seed=SEED):
    """Mahalanobis distance from the WT reference, and the dose ordering.

    The WT reference is fitted on a *disjoint half* of the WT runs so that the
    WT distances reported are out-of-sample. Fitting on all WT runs and then
    measuring WT's own distance would report shrinkage, not deviation.
    """
    rng = np.random.default_rng(seed)
    wt_idx = np.flatnonzero(systems == "WT")
    rng.shuffle(wt_idx)
    half = wt_idx.size // 2
    fit_idx, test_idx = wt_idx[:half], wt_idx[half:]

    ref = WTReference.fit(Z[fit_idx])
    dists, ci = {}, {}
    for s in SYSTEMS:
        idx = test_idx if s == "WT" else np.flatnonzero(systems == s)
        if idx.size == 0:
            continue
        d = np.asarray(ref.distance(Z[idx])).ravel()
        dists[s] = d
        boot = np.array([rng.choice(d, d.size, replace=True).mean()
                         for _ in range(n_boot)])
        ci[s] = (float(np.percentile(boot, 2.5)),
                 float(np.percentile(boot, 97.5)))

    print(f"\n  {label}: Mahalanobis distance from WT reference "
          f"(fitted on {fit_idx.size} WT runs)")
    print(f"    {'system':14s} {'n':>4s}  {'mean D':>8s}  {'95% CI':>18s}")
    for s in SYSTEMS:
        if s not in dists:
            continue
        d = dists[s]
        print(f"    {s:14s} {d.size:4d}  {d.mean():8.3f}  "
              f"[{ci[s][0]:7.3f}, {ci[s][1]:7.3f}]")

    # --- the ordering test ---------------------------------------------------
    order_ok, notes = True, []
    if {"WT", "N439K", "N439K+K417V"} <= dists.keys():
        for single in ("N439K", "K417V"):
            if single not in dists:
                continue
            a, b, c = (dists["WT"].mean(), dists[single].mean(),
                       dists["N439K+K417V"].mean())
            ok = a < b < c
            order_ok &= ok
            notes.append(f"WT {a:.2f} {'<' if a < b else '>='} {single} {b:.2f} "
                         f"{'<' if b < c else '>='} double {c:.2f}  "
                         f"{'ordered' if ok else 'NOT ordered'}")
    for n in notes:
        print(f"    {n}")

    # --- restoration score ---------------------------------------------------
    scores = {}
    if "N439K+K417V" in dists:
        d_base = float(dists["N439K+K417V"].mean())
        print(f"\n    restoration R = 1 - D/D_double, baseline D_double="
              f"{d_base:.3f}")
        for s in SYSTEMS:
            if s not in dists:
                continue
            idx = (np.flatnonzero(systems == "WT")[half:] if s == "WT"
                   else np.flatnonzero(systems == s))
            R = np.asarray(rescue_score(Z[idx], ref, d_base)).ravel()
            scores[s] = (float(R.mean()), float(R.std()))
            print(f"      {s:14s} R = {R.mean():+.3f} +- {R.std():.3f}")
    return {"distances": {k: [float(v.mean()), float(v.std()), int(v.size)]
                          for k, v in dists.items()},
            "ci": ci, "ordered": bool(order_ok), "notes": notes,
            "restoration": scores}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("STAGE B/C  nested autoencoder vs PCA on the RBD variant panel")
    print("=" * 72)

    frames_list, rejects = load_runs()
    mats, systems, reps = frame_matrices(frames_list)
    d_in = mats[0].shape[1]
    print(f"\n{len(mats)} runs, {mats[0].shape[0]} frames each, {d_in} pair "
          f"features, no pooling")
    for s in SYSTEMS:
        print(f"  {s:14s} {(systems == s).sum():4d} runs")

    tr, va = run_level_split(systems, reps)
    print(f"\nrun-level split: {tr.size} train / {va.size} held-out runs")
    assert not (set(tr.tolist()) & set(va.tolist()))

    # --- pair selection, fitted on training runs only ------------------------
    means_full = np.stack([m.mean(axis=0) for m in mats])
    sd_tr = means_full[tr].std(axis=0)
    sel = sd_tr > MIN_PAIR_SD
    print(f"\npair selection (training runs only): {sel.sum()} of {d_in} pairs "
          f"have between-run sd > {MIN_PAIR_SD:g}")
    print(f"  discarded {(~sel).sum()} pairs that never vary; "
          f"they would have been {100*(~sel).sum()/d_in:.1f}% of the AE input")
    mats = [m[:, sel] for m in mats]
    d_in = int(sel.sum())

    # ---------------- PCA baseline (fitted on training runs only) -----------
    print("\n" + "-" * 72)
    print("BASELINE: PCA over run-mean contact vectors")
    print("-" * 72)
    Xmean = means_full[:, sel]
    pca = PCAEmbedder.fit(Xmean[tr], n_components=32)
    Zp = pca.transform(Xmean)
    print(f"  fitted on {tr.size} runs; explained variance "
          f"{pca.explained[:5].round(3).tolist()} ... "
          f"total {pca.explained.sum():.3f}")
    res_pca = evaluate(Zp, systems, reps, "PCA(32) over run means")
    res_pca_held = evaluate(Zp[va], systems[va], reps[va],
                            "PCA(32), held-out runs only")

    # ---------------- nested autoencoder ------------------------------------
    print("\n" + "-" * 72)
    print("NESTED AUTOENCODER  (frame AE -> trajectory AE -> joint)")
    print("-" * 72)
    ids = [f"{systems[i]}|310|{reps[i]}" for i in range(len(mats))]
    Xtr, ttr, otr = make_windows([mats[i] for i in tr], WINDOW, WIN_STRIDE,
                                 [ids[i] for i in tr])
    Xva, tva, ova = make_windows([mats[i] for i in va], WINDOW, WIN_STRIDE,
                                 [ids[i] for i in va])
    print(f"  windows: {Xtr.shape[0]} train / {Xva.shape[0]} val "
          f"(window {WINDOW} frames, stride {WIN_STRIDE})")
    assert not (set(otr) & set(ova)), "window owners leaked across the split"

    cfg = AEConfig(d_in=d_in, d_frame=64, d_traj=32,
                   frame_hidden=(512, 256), traj_hidden=(256, 128),
                   dropout=0.1)
    model = build_nested_autoencoder(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  parameters {n_par/1e6:.2f}M for {len(tr)} training runs "
          f"({Xtr.shape[0]} windows)")
    reports = fit_nested(model, Xtr, ttr, Xva, tva, cfg,
                         FitConfig(epochs_frame=40, epochs_traj=40,
                                   epochs_joint=25, batch_size=16, lr=1e-3,
                                   device="cuda", seed=SEED))
    print("\n  stage reports (validation on held-out RUNS):")
    for r in reports:
        print(f"    {r.stage:18s} val_loss={r.val_loss:.5f} "
              f"val_r2={r.val_r2:+.4f}")

    Za = embed_runs(model, mats)
    res_ae = evaluate(Za, systems, reps, "nested AE z (d_traj=32)")
    res_ae_held = evaluate(Za[va], systems[va], reps[va],
                           "nested AE z, held-out runs only")

    # ---------------- stage C -----------------------------------------------
    print("\n" + "=" * 72)
    print("STAGE C  restoration metric")
    print("=" * 72)
    rest_pca = restoration(Zp, systems, "PCA(32)")
    rest_ae = restoration(Za, systems, "nested AE")

    # ---------------- verdict -----------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    better = []
    if res_ae["loro"] > res_pca["loro"]:
        better.append(f"LORO {res_ae['loro']:.3f} > {res_pca['loro']:.3f}")
    else:
        better.append(f"LORO {res_ae['loro']:.3f} <= {res_pca['loro']:.3f}")
    if res_ae["icc"] > res_pca["icc"]:
        better.append(f"ICC {res_ae['icc']:+.3f} > {res_pca['icc']:+.3f}")
    else:
        better.append(f"ICC {res_ae['icc']:+.3f} <= {res_pca['icc']:+.3f}")
    print("  AE vs PCA: " + "; ".join(better))
    print(f"  dose ordering  PCA: {'holds' if rest_pca['ordered'] else 'FAILS'}"
          f"   AE: {'holds' if rest_ae['ordered'] else 'FAILS'}")

    with open(OUT / "stage_bc.json", "w") as fh:
        json.dump({"n_runs": len(mats), "d_in": int(d_in),
                   "n_pairs_before_selection": int(sel.size),
                   "min_pair_sd": MIN_PAIR_SD,
                   "counts": {s: int((systems == s).sum()) for s in SYSTEMS},
                   "n_train_runs": int(tr.size), "n_val_runs": int(va.size),
                   "ae_parameters": int(n_par),
                   "stages": [{"stage": r.stage, "val_loss": float(r.val_loss),
                               "val_r2": float(r.val_r2)} for r in reports],
                   "pca": res_pca, "pca_heldout": res_pca_held,
                   "ae": res_ae, "ae_heldout": res_ae_held,
                   "restoration_pca": rest_pca, "restoration_ae": rest_ae},
                  fh, indent=2)
    np.savez_compressed(OUT / "embeddings.npz", Z_pca=Zp, Z_ae=Za,
                        systems=systems, replicates=reps, train=tr, val=va,
                        pair_selection=sel)
    print(f"\nwrote {OUT / 'stage_bc.json'}")


if __name__ == "__main__":
    main()
