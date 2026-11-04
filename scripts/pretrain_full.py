"""Pretrain BOTH levels of the nested autoencoder across domains.

`scripts/pretrain_patches.py` pretrained only the patch (frame) encoder across
domains and then trained the trajectory level separately on each evaluation
domain. That lifted the frame level (held-out R2 0.04-0.40 -> 0.48-0.64) and
roughly doubled separation (0.47 -> 1.09), but the trajectory stages came back
with **negative** held-out R2: the trajectory encoder still only ever saw the
15 training runs of one domain.

The shortage was therefore never "data" in the abstract -- it was data *at the
trajectory level*, and the architecture was not using what was available.
Because pooled patch embeddings have the same width for every protein, the
trajectory autoencoder can be shared across domains exactly as the patch
encoder is. That turns 15 training runs into ~1,050.

This script does that, and then evaluates **zero-shot**: held-out domains get
no per-domain training at all, so a separation measured there reflects a
representation that transfers, not one fitted to that protein.

Usage:
    python scripts/pretrain_full.py <dir> [--eval-domains 8] [--epochs 40]
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
                         separation_ratio, variance_components)
from sawnergy.md import build_triplets, nested_loss, triplet_loss  # noqa: E402
from sawnergy.md.encoder import build_frame_autoencoder  # noqa: E402
from sawnergy.md.validate import gate_sham  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def usable_files(data_dir: str) -> list[Path]:
    out = []
    for p in sorted(Path(data_dir).glob("mdcath_dataset_*.h5")):
        try:
            MdCathReader(p).temperatures()
            out.append(p)
        except Exception as exc:
            print(f"   skipping {p.name}: {type(exc).__name__}", flush=True)
    return out


# ---------------------------------------------------------------------------
# stage 1: patch encoder across domains
# ---------------------------------------------------------------------------

def train_patch_encoder(files, *, d_latent, epochs, stride, n_centres,
                        max_frames, device, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    chunks, t0 = [], time.time()
    for i, p in enumerate(files):
        r = MdCathReader(p)
        for t in r.temperatures():
            for rep in r.replicates(t):
                f = r.read(t, rep, stride=stride, max_frames=max_frames)
                if f.n_residues < 20:
                    continue
                c = rng.choice(f.n_residues,
                               size=min(n_centres, f.n_residues), replace=False)
                pv = patch_vectors(f, residues=c)
                chunks.append(pv.reshape(-1, pv.shape[-1]))
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(files)} domains, "
                  f"{sum(c.shape[0] for c in chunks):,} patches "
                  f"({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(chunks).astype(np.float32)
    print(f"   patch corpus: {X.shape[0]:,} x {X.shape[1]}", flush=True)

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
            print(f"   patch epoch {ep:3d} mse={tot/n:.5f}", flush=True)
    enc.eval()
    return enc, (mu, sd)


# ---------------------------------------------------------------------------
# frame representations (same width for every protein)
# ---------------------------------------------------------------------------

def frame_reps(path, encoder, norm, *, stride, device):
    import torch
    mu, sd = norm
    r = MdCathReader(path)
    frames, descs, temps, reps = [], [], [], []
    for f in r.iter_runs(stride=stride):
        pv = patch_vectors(f)
        T, N, P = pv.shape
        with torch.no_grad():
            h = encoder(torch.from_numpy(
                ((pv.reshape(-1, P) - mu) / sd).astype(np.float32)).to(device)
            ).reshape(T, N, -1)
            rep = torch.cat([h.mean(1), h.std(1)], -1).cpu().numpy()
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
    ap.add_argument("--eval-domains", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--traj-epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--pre-stride", type=int, default=20)
    ap.add_argument("--centres", type=int, default=20)
    ap.add_argument("--pre-max-frames", type=int, default=25)
    ap.add_argument("--d-patch", type=int, default=32)
    ap.add_argument("--d-traj", type=int, default=16)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--triplet-weight", type=float, default=0.0,
                    help="weight on the metric-learning term (0 = pure "
                         "reconstruction, the plan's section 7 Option A)")
    ap.add_argument("--triplet-epochs", type=int, default=40)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--finetune-epochs", type=int, default=0,
                    help="if >0, fine-tune on each eval domain's TRAINING runs "
                         "and score only its held-out runs -- the realistic "
                         "deployment setting, where the target protein is "
                         "available but the specific runs are not")
    ap.add_argument("--holdout", type=int, default=2)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    files = usable_files(args.data_dir)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    eval_files = [files[i] for i in order[:args.eval_domains]]
    pre_files = [files[i] for i in order[args.eval_domains:]]

    banner(f"STAGE 1 -- patch encoder on {len(pre_files)} domains "
           f"({len(pre_files)*25} runs)")
    enc, norm = train_patch_encoder(
        pre_files, d_latent=args.d_patch, epochs=args.epochs,
        stride=args.pre_stride, n_centres=args.centres,
        max_frames=args.pre_max_frames, device=dev)

    banner("STAGE 2 -- frame representations for every run")
    pre_frames, pre_ids = [], []
    for i, p in enumerate(pre_files):
        dom, fr, _, tmp, rp = frame_reps(p, enc, norm, stride=args.stride,
                                         device=dev)
        pre_frames.extend(fr)
        pre_ids.extend([f"{dom}|T{t:g}|r{r}" for t, r in zip(tmp, rp)])
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(pre_files)} domains, {len(pre_frames)} runs",
                  flush=True)
    print(f"   trajectory-level training set: {len(pre_frames)} runs "
          f"(was 15 per domain)", flush=True)

    eval_data = [frame_reps(p, enc, norm, stride=args.stride, device=dev)
                 for p in eval_files]

    banner("STAGE 3 -- nested autoencoder trained ACROSS domains")
    d_in = pre_frames[0].shape[1]
    Xtr, ttr, own_tr = make_windows(pre_frames, args.window, args.window,
                                    pre_ids)
    va_frames, va_ids = [], []
    for dom, fr, _, tmp, rp in eval_data:
        va_frames.extend(fr)
        va_ids.extend([f"{dom}|T{t:g}|r{r}" for t, r in zip(tmp, rp)])
    Xva, tva, _ = make_windows(va_frames, args.window, args.window, va_ids)
    print(f"   {Xtr.shape[0]:,} training windows from {len(pre_frames)} runs; "
          f"{Xva.shape[0]:,} validation windows from {len(va_frames)} "
          f"held-out-domain runs", flush=True)

    cfg = AEConfig(d_in=d_in, d_frame=args.d_patch, d_traj=args.d_traj,
                   frame_hidden=(128,), traj_hidden=(128, 64), dropout=0.1,
                   time_conditioned=True)
    model = build_nested_autoencoder(cfg)
    print(f"   {sum(p.numel() for p in model.parameters()):,} parameters",
          flush=True)
    for r in fit_nested(model, Xtr, ttr, Xva, tva, cfg,
                        FitConfig(epochs_frame=args.traj_epochs,
                                  epochs_traj=args.traj_epochs,
                                  epochs_joint=args.traj_epochs // 2,
                                  batch_size=128, lr=1e-3, weight_decay=1e-4,
                                  device=dev)):
        print(f"   {r}", flush=True)

    if args.triplet_weight > 0:
        banner(f"STAGE 3b -- fine-tune with triplet loss "
               f"(w={args.triplet_weight}, margin={args.margin})")
        rng = np.random.default_rng(0)
        mu_n, sd_n = model._sawnergy_norm
        Xt = torch.from_numpy(((Xtr - mu_n) / sd_n).astype(np.float32)).to(dev)
        Tt = torch.from_numpy(ttr.astype(np.float32)).to(dev)
        owners = np.asarray(own_tr)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                weight_decay=1e-4)
        n = Xt.shape[0]
        for ep in range(args.triplet_epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            tot_r = tot_t = 0.0
            nb = 0
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                out = model(Xt[idx], Tt[idx])
                l_rec, _ = nested_loss(out, Xt[idx], cfg)
                tri = build_triplets(owners[idx.cpu().numpy()], rng,
                                     max_triplets=512)
                l_tri = triplet_loss(out["z"],
                                     torch.from_numpy(tri).to(dev),
                                     margin=args.margin)
                loss = l_rec + args.triplet_weight * l_tri
                opt.zero_grad(); loss.backward(); opt.step()
                tot_r += float(l_rec); tot_t += float(l_tri); nb += 1
            if ep % 10 == 0 or ep == args.triplet_epochs - 1:
                print(f"   epoch {ep:3d}  recon={tot_r/nb:.4f}  "
                      f"triplet={tot_t/nb:.4f}", flush=True)
        model.eval()

    mode = ("FINE-TUNED per domain, scored on held-out runs"
            if args.finetune_epochs > 0 else "ZERO-SHOT on held-out domains")
    banner(f"STAGE 4 -- {mode}")
    import copy
    base_state = copy.deepcopy(model.state_dict())
    rows = []
    for dom, fr, descs, temps, reps in eval_data:
        adj = tuple(np.unique(temps)[:2])
        if args.finetune_epochs > 0:
            held = reps >= (reps.max() + 1 - args.holdout)
            tr_i, te_i = np.where(~held)[0], np.where(held)[0]
            model.load_state_dict(base_state)
            ids = [f"{dom}|T{temps[i]:g}|r{reps[i]}" for i in tr_i]
            Xf, tf, of = make_windows([fr[i] for i in tr_i], args.window,
                                      args.window // 2, ids)
            mu_n, sd_n = model._sawnergy_norm
            Xf_t = torch.from_numpy(((Xf - mu_n) / sd_n).astype(np.float32)).to(dev)
            tf_t = torch.from_numpy(tf.astype(np.float32)).to(dev)
            rng2 = np.random.default_rng(1)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                    weight_decay=1e-4)
            for ep in range(args.finetune_epochs):
                model.train()
                perm = torch.randperm(Xf_t.shape[0], device=dev)
                for i in range(0, Xf_t.shape[0], 64):
                    idx = perm[i:i + 64]
                    out = model(Xf_t[idx], tf_t[idx])
                    l_rec, _ = nested_loss(out, Xf_t[idx], cfg)
                    loss = l_rec
                    if args.triplet_weight > 0:
                        tri = build_triplets(np.asarray(of)[idx.cpu().numpy()],
                                             rng2, max_triplets=256)
                        loss = loss + args.triplet_weight * triplet_loss(
                            out["z"], torch.from_numpy(tri).to(dev),
                            margin=args.margin)
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            Z = embed_runs(model, [fr[i] for i in te_i])
            sep_ae, acc_ae = score(Z, temps[te_i], adj)
            pca = PCAEmbedder.fit(descs[tr_i],
                                  n_components=min(args.d_traj, len(tr_i) - 1))
            sep_pca, acc_pca = score(pca.transform(descs[te_i]), temps[te_i], adj)
            temps, reps = temps[te_i], reps[te_i]
        else:
            Z = embed_runs(model, fr)
            sep_ae, acc_ae = score(Z, temps, adj)
            pca = PCAEmbedder.fit(descs, n_components=args.d_traj)
            sep_pca, acc_pca = score(pca.transform(descs), temps, adj)
        g = gate_sham(Z, np.array([f"T{t:g}" for t in temps]), reps,
                      n_trials=8, n_permutations=400,
                      rng=np.random.default_rng(0))
        try:
            vc = variance_components(Z, np.array([f"T{t:g}" for t in temps]),
                                     n_permutations=1000)
            icc, pv = vc.icc, vc.p_value
        except ValueError:
            icc, pv = float("nan"), float("nan")
        print(f"\n{dom}: AE(zero-shot) sep={sep_ae:5.2f} LORO={acc_ae:.2f} "
              f"ICC={icc:+.3f} p={pv:.3g} | PCA(in-domain) sep={sep_pca:5.2f} "
              f"LORO={acc_pca:.2f}")
        print(f"   {g}", flush=True)
        rows.append((dom, sep_pca, acc_pca, sep_ae, acc_ae, icc, g.passed))

    banner("SUMMARY")
    print(f"{'domain':<10} {'sep PCA':>8} {'sep AE':>8} {'LORO PCA':>9} "
          f"{'LORO AE':>8} {'ICC AE':>8} {'sham':>6}")
    print("-" * 70)
    for dom, sp, ap_, sa, aa, icc, sham in rows:
        print(f"{dom:<10} {sp:8.2f} {sa:8.2f} {ap_:9.2f} {aa:8.2f} "
              f"{icc:+8.3f} {'ok' if sham else 'FAIL':>6}")
    sp = float(np.nanmean([r[1] for r in rows]))
    sa = float(np.nanmean([r[3] for r in rows]))
    wins = sum(1 for r in rows if r[3] > r[1])
    print(f"\nmean separation  PCA={sp:.2f} ({replicates_required(max(sp,1e-3))} reps)"
          f"   AE zero-shot={sa:.2f} ({replicates_required(max(sa,1e-3))} reps)")
    print(f"AE beats PCA on {wins}/{len(rows)} held-out domains")
    print("prior: AE no-pretrain=0.47 | AE frame-pretrain only=1.09")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
