"""Train the nested autoencoder on mdCATH and benchmark it against PCA.

The question this script answers is not "does it reconstruct" -- a big enough
autoencoder always reconstructs. It is:

    does z separate conditions better than a linear baseline,
    measured on runs the model never saw?

Separation ratio is the quantity that sets the replicate budget (see
docs/MD_EMBEDDER.md section 4.4), so it is the headline number. The benchmark
set earlier was: lift separation at the smallest temperature contrast from
0.78 to > 1.4 and the replicate requirement drops from ~26 to ~8.

Protocol, per domain:
  * split the 5 replicates per temperature into train / held-out
  * fit BOTH the autoencoder and the PCA baseline on training runs only
  * embed the held-out runs with each
  * compare separation ratio and leave-one-run-out accuracy on held-out runs
  * ablate time conditioning on / off

Usage:
    python scripts/train_nested_ae.py <dir-with-mdcath-h5> [--epochs 60]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sawnergy.md import (AEConfig, FitConfig, MdCathReader,  # noqa: E402
                         PCAEmbedder, build_nested_autoencoder, embed_runs,
                         fit_nested, frame_vectors, leave_one_replicate_out,
                         make_windows, replicates_required, run_descriptor,
                         separation_ratio)
from sawnergy.md.validate import gate_sham  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load(path: Path, stride: int):
    """Per-run frame matrices, run descriptors and labels."""
    r = MdCathReader(path)
    frames, descs, temps, reps = [], [], [], []
    for f in r.iter_runs(stride=stride):
        frames.append(frame_vectors(f))
        d = run_descriptor(f)
        descs.append(np.concatenate([d.mean_contact, d.contact_fluct]))
        temps.append(f.run.temperature_K)
        reps.append(f.run.replicate)
    return r.domain, frames, np.array(descs), np.array(temps), np.array(reps)


def evaluate(Z, temps, tag, adj_pair):
    """Separation on the hardest (adjacent) contrast, plus LORO."""
    labels = np.array([f"T{t:g}" for t in temps])
    a, b = adj_pair
    try:
        sep = separation_ratio(Z, labels, f"T{a:g}", f"T{b:g}")
        need = replicates_required(sep)
    except ValueError:
        sep, need = float("nan"), -1
    try:
        disc = leave_one_replicate_out(Z, labels, n_permutations=500)
        acc, chance, p = disc.accuracy, disc.chance, disc.p_value
    except ValueError:
        acc = chance = p = float("nan")
    print(f"   {tag:<28} sep({a:g}/{b:g}K)={sep:5.2f} -> {need:4d} reps | "
          f"LORO={acc:.3f} (chance {chance:.2f}, p={p:.3g})")
    return sep, acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--d-frame", type=int, default=64)
    ap.add_argument("--d-traj", type=int, default=16)
    ap.add_argument("--holdout", type=int, default=2,
                    help="replicates per temperature held out")
    ap.add_argument("--frame-hidden", type=int, nargs="*", default=[512, 256])
    ap.add_argument("--traj-hidden", type=int, nargs="*", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=1e-5)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("mdcath_dataset_*.h5"))
    summary = []

    for path in files:
        dom, frames, descs, temps, reps = load(path, args.stride)
        banner(f"DOMAIN {dom}  ({len(frames)} runs, "
               f"{frames[0].shape[1]} pair features, "
               f"{frames[0].shape[0]} frames/run after stride+burn-in)")

        # ---- split on RUNS: last `holdout` replicates of each temperature
        held = reps >= (reps.max() + 1 - args.holdout)
        tr_idx = np.where(~held)[0]
        te_idx = np.where(held)[0]
        print(f"   train runs {len(tr_idx)}   held-out runs {len(te_idx)}  "
              f"(split on replicate id, never on frames)")

        run_ids = [f"r{i}" for i in range(len(frames))]
        Xtr, ttr, _ = make_windows([frames[i] for i in tr_idx], args.window,
                                   max(args.window // 2, 1),
                                   [run_ids[i] for i in tr_idx])
        Xva, tva, _ = make_windows([frames[i] for i in te_idx], args.window,
                                   max(args.window // 2, 1),
                                   [run_ids[i] for i in te_idx])

        temps_sorted = np.unique(temps)
        adj = (temps_sorted[0], temps_sorted[1])

        # ---------------------------------------------------- PCA baseline
        print("\n-- BASELINE (PCA on run descriptors, fitted on train runs)")
        pca = PCAEmbedder.fit(descs[tr_idx], n_components=min(args.d_traj,
                                                              len(tr_idx) - 1))
        Z_pca_te = pca.transform(descs[te_idx])
        sep_pca, acc_pca = evaluate(Z_pca_te, temps[te_idx], "PCA held-out", adj)

        # ------------------------------------------------- nested AE x2
        print("\n-- NESTED AUTOENCODER (frame AE -> trajectory AE, symmetric)")
        for time_cond in (True, False):
            cfg = AEConfig(d_in=frames[0].shape[1], d_frame=args.d_frame,
                           d_traj=args.d_traj,
                           frame_hidden=tuple(args.frame_hidden),
                           traj_hidden=tuple(args.traj_hidden),
                           dropout=args.dropout,
                           time_conditioned=time_cond)
            model = build_nested_autoencoder(cfg)
            n_par = sum(p.numel() for p in model.parameters())
            reports = fit_nested(
                model, Xtr, ttr, Xva, tva, cfg,
                FitConfig(epochs_frame=args.epochs,
                          epochs_traj=args.epochs,
                          epochs_joint=max(args.epochs // 2, 1),
                          batch_size=32, lr=1e-3, weight_decay=args.wd,
                          seed=0))
            tag = "time-stamped" if time_cond else "no time stamp"
            print(f"   [{tag}] {n_par:,} params / {Xtr.shape[0]} training "
                  f"windows ({Xtr.shape[0]*Xtr.shape[1]:,} frames)")
            for r in reports:
                print(f"   [{tag}] {r}")
            Z_te = embed_runs(model, [frames[i] for i in te_idx])
            sep_ae, acc_ae = evaluate(Z_te, temps[te_idx],
                                      f"AE ({tag}) held-out", adj)
            if time_cond:
                summary.append((dom, sep_pca, acc_pca, sep_ae, acc_ae,
                                [r.val_r2 for r in reports]))

            # sham on the AE latent: fake conditions must NOT separate
            g = gate_sham(Z_te, np.array([f"T{t:g}" for t in temps[te_idx]]),
                          reps[te_idx], n_trials=8, n_permutations=400,
                          rng=np.random.default_rng(0))
            print(f"   [{tag}] {g}")

    banner("SUMMARY: does the nested AE beat PCA on held-out runs?")
    print(f"{'domain':<10} {'sep PCA':>8} {'sep AE':>8} {'LORO PCA':>9} "
          f"{'LORO AE':>8}  {'stage R2 (S1/S2/S3)':>24}")
    print("-" * 78)
    for dom, sp, ap_, sa, aa, r2 in summary:
        print(f"{dom:<10} {sp:8.2f} {sa:8.2f} {ap_:9.3f} {aa:8.3f}  "
              f"{r2[0]:+6.2f}/{r2[1]:+6.2f}/{r2[2]:+6.2f}")
    if summary:
        sp = np.mean([s[1] for s in summary])
        sa = np.mean([s[3] for s in summary])
        print(f"\nmean separation  PCA={sp:.2f} -> {replicates_required(sp)} reps"
              f"   AE={sa:.2f} -> {replicates_required(max(sa, 1e-3))} reps")
        print(f"benchmark to beat: separation > 1.4 (would need ~8 replicates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
