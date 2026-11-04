"""Does more data fix the autoencoder? Broad pretraining, then within-domain test.

The earlier benchmark (docs/MD_EMBEDDER.md section 8) found the nested
autoencoder losing to PCA, and diagnosed the cause as too few independent
runs -- 15 -- rather than too little capacity. This script tests that
diagnosis directly by raising the run count roughly twentyfold.

The obstacle was that pair features have a width set by the protein's residue
count, so domains could not be pooled. `features.patch_vectors` removes it:
fixed-size spatial neighbourhoods give every domain the same feature width, so
patches from 50 proteins form one corpus. That is the pipeline document's
section 8 Option C -- self-supervised pretraining on a large MD corpus before
the final trajectory model.

Protocol:
  1. Hold out several domains entirely.
  2. Pretrain the patch autoencoder on patches pooled from every other domain.
  3. On each held-out domain: freeze the pretrained encoder, build per-frame
     representations, train the trajectory autoencoder on training runs,
     embed held-out runs.
  4. Compare separation ratio against PCA and against the same model with no
     pretraining.

The comparison that decides the question is the third column against the
second, on domains that contributed nothing to pretraining.

Usage:
    python scripts/pretrain_patches.py <dir> [--eval-domains 5] [--epochs 40]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sawnergy.md import (AEConfig, FitConfig, MdCathReader,  # noqa: E402
                         PCAEmbedder, build_nested_autoencoder, embed_runs,
                         fit_nested, leave_one_replicate_out, make_windows,
                         patch_vectors, replicates_required, run_descriptor,
                         separation_ratio)
from sawnergy.md.encoder import build_frame_autoencoder  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def usable_files(data_dir: str) -> list[Path]:
    """Domain files that open cleanly.

    Skips truncated downloads rather than dying partway through a long run --
    an interrupted transfer leaves a valid-looking .h5 that raises only when
    opened.
    """
    out = []
    for p in sorted(Path(data_dir).glob("mdcath_dataset_*.h5")):
        try:
            MdCathReader(p).temperatures()
            out.append(p)
        except Exception as exc:
            print(f"   skipping {p.name}: {type(exc).__name__}", flush=True)
    return out


# ---------------------------------------------------------------------------
# pretraining corpus
# ---------------------------------------------------------------------------

def sample_patches(path: Path, *, stride: int, n_centres: int,
                   max_frames: int, rng) -> np.ndarray:
    """Subsample patches from every run of one domain."""
    r = MdCathReader(path)
    out = []
    for t in r.temperatures():
        for rep in r.replicates(t):
            f = r.read(t, rep, stride=stride, max_frames=max_frames)
            n_res = f.n_residues
            if n_res < 20:
                continue
            centres = rng.choice(n_res, size=min(n_centres, n_res),
                                 replace=False)
            p = patch_vectors(f, residues=centres)        # (T, C, P)
            out.append(p.reshape(-1, p.shape[-1]))
    return np.concatenate(out) if out else np.zeros((0, 120), np.float32)


def pretrain(files, *, d_latent: int, epochs: int, stride: int,
             n_centres: int, max_frames: int, device: str, seed: int = 0):
    """Train the patch autoencoder on patches pooled across domains."""
    import torch

    rng = np.random.default_rng(seed)
    chunks, t0 = [], time.time()
    for i, p in enumerate(files):
        chunks.append(sample_patches(p, stride=stride, n_centres=n_centres,
                                     max_frames=max_frames, rng=rng))
        if (i + 1) % 10 == 0:
            print(f"   loaded {i+1}/{len(files)} domains, "
                  f"{sum(c.shape[0] for c in chunks):,} patches "
                  f"({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(chunks).astype(np.float32)
    print(f"   pretraining corpus: {X.shape[0]:,} patches x {X.shape[1]} dims "
          f"from {len(files)} domains", flush=True)

    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd).to(device)

    enc, dec = build_frame_autoencoder(X.shape[1], d_latent, (256, 128))
    enc, dec = enc.to(device), dec.to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=1e-3, weight_decay=1e-5)
    n, bs = Xn.shape[0], 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            b = Xn[perm[i:i + bs]]
            loss = torch.nn.functional.mse_loss(dec(enc(b)), b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * b.shape[0]
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"   pretrain epoch {ep:3d}  mse={tot/n:.5f}", flush=True)
    enc.eval()
    return enc, (mu, sd)


# ---------------------------------------------------------------------------
# per-domain evaluation
# ---------------------------------------------------------------------------

def frame_reps(path: Path, encoder, norm, *, stride: int, device: str):
    """Per-run frame representations from the (frozen) patch encoder."""
    import torch

    mu, sd = norm
    r = MdCathReader(path)
    frames, descs, temps, reps = [], [], [], []
    for f in r.iter_runs(stride=stride):
        p = patch_vectors(f)                             # (T, N, P)
        T, N, P = p.shape
        with torch.no_grad():
            h = encoder(torch.from_numpy(
                ((p.reshape(-1, P) - mu) / sd).astype(np.float32)).to(device))
            h = h.reshape(T, N, -1)
            rep = torch.cat([h.mean(1), h.std(1)], dim=-1).cpu().numpy()
        frames.append(rep.astype(np.float32))
        d = run_descriptor(f)
        descs.append(np.concatenate([d.mean_contact, d.contact_fluct]))
        temps.append(f.run.temperature_K)
        reps.append(f.run.replicate)
    return r.domain, frames, np.array(descs), np.array(temps), np.array(reps)


def score(Z, temps, adj):
    labels = np.array([f"T{t:g}" for t in temps])
    try:
        sep = separation_ratio(Z, labels, f"T{adj[0]:g}", f"T{adj[1]:g}")
    except ValueError:
        sep = float("nan")
    try:
        acc = leave_one_replicate_out(Z, labels, n_permutations=0).accuracy
    except ValueError:
        acc = float("nan")
    return sep, acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--eval-domains", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--traj-epochs", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--pre-stride", type=int, default=20)
    ap.add_argument("--centres", type=int, default=20)
    ap.add_argument("--pre-max-frames", type=int, default=25)
    ap.add_argument("--d-patch", type=int, default=32)
    ap.add_argument("--d-traj", type=int, default=8)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--holdout", type=int, default=2)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    files = usable_files(args.data_dir)
    if len(files) < args.eval_domains + 5:
        print(f"need more domains; found {len(files)}", file=sys.stderr)
        return 1
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    eval_files = [files[i] for i in order[:args.eval_domains]]
    pre_files = [files[i] for i in order[args.eval_domains:]]

    banner(f"PRETRAIN on {len(pre_files)} domains "
           f"({len(pre_files)*25} runs) | EVALUATE on {len(eval_files)} "
           f"held-out domains")
    print("   eval domains:", [p.stem.split('_')[-1] for p in eval_files])

    enc, norm = pretrain(pre_files, d_latent=args.d_patch, epochs=args.epochs,
                         stride=args.pre_stride, n_centres=args.centres,
                         max_frames=args.pre_max_frames, device=dev)

    rows = []
    for path in eval_files:
        dom, frames, descs, temps, reps = frame_reps(
            path, enc, norm, stride=args.stride, device=dev)
        banner(f"EVAL DOMAIN {dom}  ({len(frames)} runs, "
               f"frame rep dim {frames[0].shape[1]})")

        held = reps >= (reps.max() + 1 - args.holdout)
        tr, te = np.where(~held)[0], np.where(held)[0]
        adj = tuple(np.unique(temps)[:2])
        ids = [f"r{i}" for i in range(len(frames))]

        sep_pca, acc_pca = score(
            PCAEmbedder.fit(descs[tr], n_components=min(args.d_traj, len(tr) - 1))
            .transform(descs[te]), temps[te], adj)

        Xtr, ttr, _ = make_windows([frames[i] for i in tr], args.window,
                                   args.window // 2, [ids[i] for i in tr])
        Xva, tva, _ = make_windows([frames[i] for i in te], args.window,
                                   args.window // 2, [ids[i] for i in te])
        cfg = AEConfig(d_in=frames[0].shape[1], d_frame=args.d_patch,
                       d_traj=args.d_traj, frame_hidden=(128,),
                       traj_hidden=(64,), dropout=0.1, time_conditioned=True)
        model = build_nested_autoencoder(cfg)
        reports = fit_nested(model, Xtr, ttr, Xva, tva, cfg,
                             FitConfig(epochs_frame=args.traj_epochs,
                                       epochs_traj=args.traj_epochs,
                                       epochs_joint=args.traj_epochs // 2,
                                       batch_size=32, lr=1e-3,
                                       weight_decay=1e-3, device=dev))
        for r in reports:
            print(f"   {r}", flush=True)
        sep_ae, acc_ae = score(embed_runs(model, [frames[i] for i in te]),
                               temps[te], adj)
        print(f"   PCA      sep={sep_pca:5.2f}  LORO={acc_pca:.3f}")
        print(f"   AE+pre   sep={sep_ae:5.2f}  LORO={acc_ae:.3f}")
        rows.append((dom, sep_pca, acc_pca, sep_ae, acc_ae,
                     [r.val_r2 for r in reports]))

    banner("SUMMARY -- pretrained on held-out domains")
    print(f"{'domain':<10} {'sep PCA':>8} {'sep AE':>8} {'LORO PCA':>9} "
          f"{'LORO AE':>8}   {'stage R2':>22}")
    print("-" * 78)
    for dom, sp, ap_, sa, aa, r2 in rows:
        print(f"{dom:<10} {sp:8.2f} {sa:8.2f} {ap_:9.3f} {aa:8.3f}   "
              f"{r2[0]:+5.2f}/{r2[1]:+5.2f}/{r2[2]:+5.2f}")
    sp = float(np.nanmean([r[1] for r in rows]))
    sa = float(np.nanmean([r[3] for r in rows]))
    print(f"\nmean separation  PCA={sp:.2f} ({replicates_required(max(sp,1e-3))} reps)"
          f"   AE+pretrain={sa:.2f} ({replicates_required(max(sa,1e-3))} reps)")
    print("benchmark to beat: separation > 1.4 (~8 replicates)")
    print(f"prior result without pretraining: AE=0.47, PCA=0.74")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
