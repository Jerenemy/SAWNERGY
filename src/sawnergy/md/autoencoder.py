"""Nested, symmetric frame/trajectory autoencoder with time-stamped units.

Architecture (pipeline document sections 3-5), nested and mirror-symmetric:

    inward                                                   outward
    X_t  --E_f-->  h_t                                 h_t  --D_f-->  X_t_hat
                   |                                    ^
                   +-- (+ TimeEncoding(t)) --E_tau--> z --+ (queried by t) D_tau

Written out:

    X_1..X_T  --E_f-->  h_1..h_T  --E_tau-->  z  --D_tau-->  h_1_hat..h_T_hat
                                                          --D_f-->  X_1_hat..X_T_hat

Every frame enters the trajectory encoder carrying its own timestep, and the
trajectory decoder is *queried by timestep*: ``D_tau(z, t) -> h_t_hat``. That is
what makes the two halves genuine mirrors -- the decoder reconstructs a
specific indexed unit rather than emitting an unlabelled bag -- and it is why
the time stamp has to be a first-class input rather than a positional
afterthought.

Layer widths are mirrored exactly: ``D_f`` reverses ``E_f``, ``D_tau`` reverses
``E_tau``.

Three losses, one per nesting level::

    L = w_frame * ||X - D_f(E_f(X))||^2          # frame AE alone
      + w_h     * ||h - D_tau(E_tau(h))||^2      # trajectory AE in h-space
      + w_full  * ||X - D_f(D_tau(E_tau(E_f(X))))||^2   # end to end

The frame term keeps ``h`` physically meaningful, the h term trains the
temporal compression, and the full term lets frame representations adapt to
what matters at trajectory level -- the joint fine-tuning the document asks
for in section 5.

A note on what this cannot fix
------------------------------
Reconstruction quality and latent usefulness are different things, as the
document itself flags in section 7. A trajectory autoencoder with enough
capacity will reconstruct its training runs perfectly by memorising run
identity, and its ``z`` can still be useless for telling systems apart. So the
validation here never scores reconstruction on frames held out from a *seen*
run; it holds out whole runs, and then puts ``z`` through the same replicate-
aware gates as any other representation (`sawnergy.md.validate`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_logger = logging.getLogger(__name__)


def _torch():
    try:
        import torch
        return torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for the nested autoencoder. "
                          "Install it with `pip install torch`.") from exc


# ---------------------------------------------------------------------------
# time encoding
# ---------------------------------------------------------------------------

def build_time_encoding(n_freq: int = 8):
    """Sinusoidal encoding of a normalised timestep in ``[0, 1]``.

    Returns a module mapping ``(B,)`` or ``(B, T)`` times to ``2 * n_freq``
    features. Fourier features rather than a learned table so that the encoder
    generalises to trajectories of a length it never saw, and so that the
    decoder can be queried at any ``t``.
    """
    torch = _torch()
    nn = torch.nn

    class TimeEncoding(nn.Module):
        def __init__(self, n_freq: int):
            super().__init__()
            self.n_freq = n_freq
            self.register_buffer(
                "freqs", 2.0 ** torch.arange(n_freq).float() * np.pi)

        @property
        def dim(self) -> int:
            return 2 * self.n_freq

        def forward(self, t):
            x = t.unsqueeze(-1) * self.freqs
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    return TimeEncoding(n_freq)


# ---------------------------------------------------------------------------
# the nested model
# ---------------------------------------------------------------------------

@dataclass
class AEConfig:
    """Shapes and training weights for `NestedAutoencoder`."""

    d_in: int
    d_frame: int = 64
    d_traj: int = 32
    frame_hidden: tuple[int, ...] = (512, 256)
    traj_hidden: tuple[int, ...] = (256, 128)
    n_time_freq: int = 8
    time_conditioned: bool = True
    w_frame: float = 1.0
    w_h: float = 1.0
    w_full: float = 1.0
    dropout: float = 0.0


def build_nested_autoencoder(cfg: AEConfig):
    """Construct the symmetric nested autoencoder.

    Returns:
        A ``torch.nn.Module`` with ``encode_frames``, ``encode_trajectory``,
        ``decode_trajectory``, ``decode_frames`` and ``forward``.
    """
    torch = _torch()
    nn = torch.nn

    def mlp(sizes, dropout=0.0):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers += [nn.GELU(), nn.LayerNorm(sizes[i + 1])]
                if dropout:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    class NestedAutoencoder(nn.Module):
        def __init__(self, cfg: AEConfig):
            super().__init__()
            self.cfg = cfg
            self.time = build_time_encoding(cfg.n_time_freq)
            t_dim = self.time.dim if cfg.time_conditioned else 0

            # --- level 1: frame, mirrored widths
            enc_sizes = [cfg.d_in, *cfg.frame_hidden, cfg.d_frame]
            self.E_f = mlp(enc_sizes, cfg.dropout)
            self.D_f = mlp(list(reversed(enc_sizes)), cfg.dropout)

            # --- level 2: trajectory, mirrored widths
            # encoder consumes a time-stamped frame embedding
            tenc_sizes = [cfg.d_frame + t_dim, *cfg.traj_hidden, cfg.d_traj]
            self.E_tau_token = mlp(tenc_sizes, cfg.dropout)
            # pooled over time, then projected to z
            self.E_tau_pool = nn.Sequential(
                nn.Linear(2 * cfg.d_traj, cfg.d_traj), nn.GELU(),
                nn.LayerNorm(cfg.d_traj), nn.Linear(cfg.d_traj, cfg.d_traj))
            # decoder: queried by timestep -- the mirror of the encoder token
            tdec_sizes = [cfg.d_traj + t_dim, *reversed(cfg.traj_hidden),
                          cfg.d_frame]
            self.D_tau = mlp(tdec_sizes, cfg.dropout)

        # -------------------------------------------------- level 1
        def encode_frames(self, X):
            """``(B, T, d_in)`` or ``(N, d_in)`` -> frame embeddings."""
            return self.E_f(X)

        def decode_frames(self, H):
            return self.D_f(H)

        # -------------------------------------------------- level 2
        def _stamp(self, H, t):
            if not self.cfg.time_conditioned:
                return H
            return torch.cat([H, self.time(t)], dim=-1)

        def encode_trajectory(self, H, t, mask=None):
            """``(B, T, d_frame)`` + ``(B, T)`` times -> ``(B, d_traj)``.

            Pooling is mean+sd over the time axis. The timestep is already
            inside each token, so the pooling itself does not need to be
            ordered; what carries the temporal information is the stamp.
            """
            tok = self.E_tau_token(self._stamp(H, t))          # (B, T, d_traj)
            if mask is None:
                mu, sd = tok.mean(1), tok.std(1)
            else:
                m = mask.unsqueeze(-1).float()
                n = m.sum(1).clamp(min=1.0)
                mu = (tok * m).sum(1) / n
                var = ((tok - mu.unsqueeze(1)) ** 2 * m).sum(1) / n
                sd = var.clamp(min=1e-8).sqrt()
            return self.E_tau_pool(torch.cat([mu, sd], dim=-1))

        def decode_trajectory(self, z, t):
            """``(B, d_traj)`` queried at ``(B, T)`` times -> ``(B, T, d_frame)``."""
            B, T = t.shape
            zz = z.unsqueeze(1).expand(B, T, z.shape[-1])
            return self.D_tau(self._stamp(zz, t))

        # -------------------------------------------------- full pass
        def forward(self, X, t, mask=None):
            """Returns a dict with every intermediate the losses need."""
            H = self.encode_frames(X)                 # (B, T, d_frame)
            X_hat_frame = self.decode_frames(H)       # frame AE path
            z = self.encode_trajectory(H, t, mask)    # (B, d_traj)
            H_hat = self.decode_trajectory(z, t)      # (B, T, d_frame)
            X_hat_full = self.decode_frames(H_hat)    # end-to-end path
            return {"H": H, "z": z, "H_hat": H_hat,
                    "X_hat_frame": X_hat_frame, "X_hat_full": X_hat_full}

    return NestedAutoencoder(cfg)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def nested_loss(out: dict, X, cfg: AEConfig, mask=None) -> tuple:
    """Three-level reconstruction loss. Returns ``(total, parts)``."""
    torch = _torch()

    def mse(a, b):
        if mask is None:
            return torch.nn.functional.mse_loss(a, b)
        m = mask.unsqueeze(-1).float()
        return (((a - b) ** 2) * m).sum() / m.sum().clamp(min=1.0) / a.shape[-1]

    l_frame = mse(out["X_hat_frame"], X)
    l_h = mse(out["H_hat"], out["H"].detach())
    l_full = mse(out["X_hat_full"], X)
    total = cfg.w_frame * l_frame + cfg.w_h * l_h + cfg.w_full * l_full
    return total, {"frame": float(l_frame), "h": float(l_h),
                   "full": float(l_full), "total": float(total)}


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------

def make_windows(frames_by_run: list[np.ndarray], window: int, stride: int,
                 run_ids: list) -> tuple:
    """Cut each run into fixed-length windows of time-stamped frames.

    Windows are training *augmentation*, not extra samples: every window
    inherits its parent run id so that splits and all downstream statistics
    stay at run level. Timestamps are normalised to ``[0, 1]`` **within the
    parent run**, so a window knows where in the trajectory it came from.

    Args:
        frames_by_run: One ``(T_i, d)`` array per run.
        window: Frames per window.
        stride: Step between window starts.
        run_ids: Parallel list of run identifiers.

    Returns:
        ``(X, t, owner)`` with shapes ``(W, window, d)``, ``(W, window)``,
        ``(W,)``.
    """
    Xs, ts, owners = [], [], []
    for arr, rid in zip(frames_by_run, run_ids):
        T = arr.shape[0]
        if T < window:
            continue
        times = np.linspace(0.0, 1.0, T, dtype=np.float32)
        for s in range(0, T - window + 1, stride):
            Xs.append(arr[s:s + window])
            ts.append(times[s:s + window])
            owners.append(rid)
    if not Xs:
        raise ValueError(f"no run is at least {window} frames long")
    return (np.stack(Xs).astype(np.float32),
            np.stack(ts).astype(np.float32),
            np.asarray(owners))


# ---------------------------------------------------------------------------
# staged training with a validation gate after every stage
# ---------------------------------------------------------------------------

@dataclass
class StageReport:
    """Outcome of one training stage."""

    stage: str
    train_loss: float
    val_loss: float
    val_r2: float
    passed: bool
    detail: str = ""
    history: list = field(default_factory=list)

    def __str__(self) -> str:
        return (f"[{'PASS' if self.passed else 'FAIL'}] {self.stage}: "
                f"train={self.train_loss:.5f} val={self.val_loss:.5f} "
                f"val_R2={self.val_r2:+.3f}  {self.detail}")


@dataclass
class FitConfig:
    """Training schedule for `fit_nested`."""

    epochs_frame: int = 30
    epochs_traj: int = 30
    epochs_joint: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 0
    min_r2_frame: float = 0.5
    min_r2_traj: float = 0.0
    min_r2_full: float = 0.0


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_nested(model, X_tr, t_tr, X_va, t_va, cfg: AEConfig,
               fit: FitConfig | None = None) -> list[StageReport]:
    """Three-stage training, validated on held-out **runs** after each stage.

    Stage 1 trains the frame autoencoder alone, stage 2 trains the trajectory
    autoencoder on frozen frame embeddings, stage 3 fine-tunes everything
    jointly -- the schedule the pipeline document specifies in section 5.

    Args:
        X_tr, t_tr: ``(W, T, d_in)`` windows and ``(W, T)`` timestamps from
            training runs.
        X_va, t_va: Same from held-out runs. These must come from runs the
            model has never seen; a frame-level split reports memorisation.

    Returns:
        One `StageReport` per stage. A stage that fails its R2 floor does not
        abort training, but it is recorded -- a trajectory level that cannot
        beat predicting the mean is not compressing anything.
    """
    torch = _torch()
    fit = fit or FitConfig()
    torch.manual_seed(fit.seed)
    dev = fit.device if (fit.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = model.to(dev)

    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(0, keepdims=True)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(0, keepdims=True) + 1e-6
    Xtr = torch.from_numpy(((X_tr - mu) / sd).astype(np.float32)).to(dev)
    Xva = torch.from_numpy(((X_va - mu) / sd).astype(np.float32)).to(dev)
    ttr = torch.from_numpy(t_tr.astype(np.float32)).to(dev)
    tva = torch.from_numpy(t_va.astype(np.float32)).to(dev)
    model._sawnergy_norm = (mu, sd)          # type: ignore[attr-defined]

    def _epoch(params, weights, train: bool):
        w_frame, w_h, w_full = weights
        local = AEConfig(**{**cfg.__dict__, "w_frame": w_frame,
                            "w_h": w_h, "w_full": w_full})
        n = Xtr.shape[0]
        opt = torch.optim.AdamW(params, lr=fit.lr, weight_decay=fit.weight_decay) \
            if train else None
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        model.train() if train else model.eval()
        for i in range(0, n, fit.batch_size):
            idx = perm[i:i + fit.batch_size]
            out = model(Xtr[idx], ttr[idx])
            loss, _ = nested_loss(out, Xtr[idx], local)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        return tot / n

    def _validate(path: str) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            out = model(Xva, tva)
            pred = out["X_hat_frame"] if path == "frame" else out["X_hat_full"]
            if path == "h":
                p, tgt = out["H_hat"], out["H"]
            else:
                p, tgt = pred, Xva
            mse = float(torch.nn.functional.mse_loss(p, tgt))
            r2 = _r2(p.cpu().numpy(), tgt.cpu().numpy())
        return mse, r2

    reports: list[StageReport] = []

    # ---- stage 1: frame autoencoder alone
    params = list(model.E_f.parameters()) + list(model.D_f.parameters())
    hist = []
    for ep in range(fit.epochs_frame):
        hist.append(_epoch(params, (1.0, 0.0, 0.0), True))
    mse, r2 = _validate("frame")
    reports.append(StageReport(
        "S1 frame AE", hist[-1], mse, r2, r2 >= fit.min_r2_frame,
        f"frame reconstruction on held-out runs (floor R2={fit.min_r2_frame})",
        hist))

    # ---- stage 2: trajectory autoencoder on frozen frame embeddings
    for p in params:
        p.requires_grad_(False)
    tparams = (list(model.E_tau_token.parameters())
               + list(model.E_tau_pool.parameters())
               + list(model.D_tau.parameters()))
    hist = []
    for ep in range(fit.epochs_traj):
        hist.append(_epoch(tparams, (0.0, 1.0, 0.0), True))
    mse_h, r2_h = _validate("h")
    reports.append(StageReport(
        "S2 trajectory AE", hist[-1], mse_h, r2_h, r2_h >= fit.min_r2_traj,
        "h-space reconstruction through the trajectory bottleneck", hist))

    # ---- stage 3: joint fine-tune
    for p in params:
        p.requires_grad_(True)
    allp = list(model.parameters())
    hist = []
    for ep in range(fit.epochs_joint):
        hist.append(_epoch(allp, (cfg.w_frame, cfg.w_h, cfg.w_full), True))
    mse_f, r2_f = _validate("full")
    reports.append(StageReport(
        "S3 joint", hist[-1], mse_f, r2_f, r2_f >= fit.min_r2_full,
        "end-to-end reconstruction through both bottlenecks", hist))

    return reports


def embed_runs(model, frames_by_run: list[np.ndarray], *,
               device: str | None = None) -> np.ndarray:
    """One ``z`` per run, using the whole run as a single time-stamped unit.

    Args:
        model: A fitted `NestedAutoencoder`.
        frames_by_run: One ``(T_i, d_in)`` array per run; lengths may differ.

    Returns:
        ``(n_runs, d_traj)``.
    """
    torch = _torch()
    mu, sd = getattr(model, "_sawnergy_norm", (0.0, 1.0))
    dev = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for arr in frames_by_run:
            x = torch.from_numpy(((arr - mu) / sd).astype(np.float32))[None].to(dev)
            t = torch.linspace(0, 1, arr.shape[0], device=dev)[None]
            H = model.encode_frames(x)
            out.append(model.encode_trajectory(H, t).cpu().numpy()[0])
    return np.stack(out)
