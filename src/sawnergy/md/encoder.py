"""Frame and trajectory encoders producing a *single shared* latent space.

Why an autoencoder is not merely an upgrade over the existing skip-gram
embedder, but a correctness fix
-------------------------------------------------------------------------
SAWNERGY's `sawnergy.embedding.Embedder` trains a skip-gram model per frame.
Skip-gram embeddings are identified only up to an arbitrary orthogonal
transform, and each frame is trained from its own initialisation with its own
negative samples, so **every frame lands in a different basis**. The repository
acknowledges this by shipping `align_frames`, an Orthogonal Procrustes
alignment. That is a reasonable patch for visualising one trajectory, but it
cannot support a wild-type-versus-mutant distance: Procrustes always finds
*some* alignment, and fitting it minimises exactly the quantity being measured.

The fix is structural, not statistical. Train **one** encoder over all frames
of all systems. Then every embedding lives in the same space by construction,
distances are defined without any alignment step, and the comparison is
well-posed.

Why set pooling instead of an LSTM
----------------------------------
The pipeline document proposes LSTM/GRU/Transformer temporal encoders. For an
*equilibrium* ensemble that is the wrong inductive bias and it is actively
dangerous here. A converged trajectory is a sample from a stationary
distribution: the frame ordering is close to arbitrary, and what characterises
the system is the distribution over frames, not its sequence. An order-aware
encoder has the capacity to latch onto run-specific drift -- which is precisely
the between-replicate variation the whole validation suite is trying to
suppress. `TrajectorySetEncoder` therefore pools order-invariantly (mean, sd
and quantiles) by default. Order-aware pooling is still available for
genuinely non-stationary processes such as forced unfolding, via
``order_aware=True``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# baseline: linear, deterministic, shared basis by construction
# ---------------------------------------------------------------------------

@dataclass
class PCAEmbedder:
    """PCA over run-level descriptors.

    This is the baseline every learned encoder must beat. It is deterministic,
    needs no GPU, and shares a basis across systems automatically. If a trained
    autoencoder does not improve on this under replicate-held-out evaluation,
    it is adding parameters and failure modes for nothing -- so `validate` runs
    this alongside whatever model is being tested.

    Attributes:
        mean: Feature mean used for centring.
        components: ``(k, d)`` principal directions.
        explained: Explained-variance ratio per component.
    """

    mean: np.ndarray
    components: np.ndarray
    explained: np.ndarray
    scale: np.ndarray | None = None

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int = 16, *,
            standardize: bool = True) -> "PCAEmbedder":
        """Fit on ``(n_runs, d)`` descriptors.

        Args:
            X: Run-level descriptors. **Fit this on training runs only.** Fitting
                on all runs and then evaluating leaks across the replicate split.
            n_components: Latent dimension, capped at ``min(n_runs, d)``.
            standardize: Scale features to unit variance first. Contact means and
                contact fluctuations live on different scales; without this the
                larger-variance block dominates the projection.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be (n_runs, d); got {X.shape}")
        mu = X.mean(axis=0)
        Xc = X - mu
        scale = None
        if standardize:
            scale = Xc.std(axis=0)
            scale[scale < 1e-8] = 1.0
            Xc = Xc / scale
        k = int(min(n_components, min(Xc.shape)))
        # economy SVD; rows are observations
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2
        explained = var[:k] / var.sum() if var.sum() > 0 else np.zeros(k)
        return cls(mean=mu, components=Vt[:k], explained=explained, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xc = X - self.mean
        if self.scale is not None:
            Xc = Xc / self.scale
        return Xc @ self.components.T


# ---------------------------------------------------------------------------
# learned encoders
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
        return torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for the learned encoders. "
                          "Install it with `pip install torch`.") from exc


def build_frame_autoencoder(d_in: int, d_latent: int = 64,
                            hidden: tuple[int, ...] = (512, 256)):
    """Symmetric MLP autoencoder over flattened upper-triangular frame features.

    Returns:
        ``(encoder, decoder)`` as ``torch.nn.Module``s.
    """
    torch = _torch()
    nn = torch.nn

    enc_layers, prev = [], d_in
    for h in hidden:
        enc_layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)]
        prev = h
    enc_layers += [nn.Linear(prev, d_latent)]

    dec_layers, prev = [], d_latent
    for h in reversed(hidden):
        dec_layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)]
        prev = h
    dec_layers += [nn.Linear(prev, d_in)]

    return nn.Sequential(*enc_layers), nn.Sequential(*dec_layers)


class TrajectorySetEncoder:
    """Pool frame embeddings into one trajectory embedding.

    Order-invariant by default -- see the module docstring for why that is the
    right default for equilibrium ensembles.
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                 order_aware: bool = False) -> None:
        self.quantiles = quantiles
        self.order_aware = order_aware
        if order_aware:
            _logger.warning(
                "order_aware pooling can encode run-specific drift, which is the "
                "between-replicate confound this pipeline tries to remove. Use it "
                "only for genuinely non-stationary processes.")

    def __call__(self, H: np.ndarray) -> np.ndarray:
        """Pool ``(T, d)`` frame embeddings into ``(d * n_stats,)``."""
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 2:
            raise ValueError(f"H must be (T, d); got {H.shape}")
        parts = [H.mean(0), H.std(0)]
        if self.quantiles:
            parts.append(np.quantile(H, self.quantiles, axis=0).ravel())
        if self.order_aware:
            half = H.shape[0] // 2
            parts.append(H[half:].mean(0) - H[:half].mean(0))   # net drift
        return np.concatenate(parts)


@dataclass
class TrainConfig:
    """Hyperparameters for `train_frame_autoencoder`."""

    d_latent: int = 64
    hidden: tuple[int, ...] = (512, 256)
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 0
    log_every: int = 10


def train_frame_autoencoder(X_train: np.ndarray, cfg: TrainConfig | None = None,
                            X_val: np.ndarray | None = None):
    """Train the shared frame autoencoder.

    Args:
        X_train: ``(n_frames, d)`` flattened frame features from **training
            runs only**. The caller is responsible for the replicate-level
            split; this function deliberately does not split for you, because a
            random split over frames is the single easiest way to produce a
            beautiful and completely invalid result.
        cfg: Hyperparameters.
        X_val: Optional validation frames from held-out *runs*.

    Returns:
        ``(encoder, decoder, history)``.
    """
    torch = _torch()
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if dev != cfg.device:
        _logger.warning("CUDA unavailable; training on CPU")

    X = np.asarray(X_train, dtype=np.float32)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    enc, dec = build_frame_autoencoder(X.shape[1], cfg.d_latent, cfg.hidden)
    enc, dec = enc.to(dev), dec.to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.from_numpy(Xn).to(dev)
    xv = None
    if X_val is not None:
        xv = torch.from_numpy(((np.asarray(X_val, np.float32) - mu) / sd)).to(dev)

    history: list[dict] = []
    n = xt.shape[0]
    for ep in range(cfg.epochs):
        enc.train(); dec.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            b = xt[perm[i:i + cfg.batch_size]]
            loss = torch.nn.functional.mse_loss(dec(enc(b)), b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * b.shape[0]
        rec = {"epoch": ep, "train_mse": tot / n}
        if xv is not None:
            enc.eval(); dec.eval()
            with torch.no_grad():
                rec["val_mse"] = float(torch.nn.functional.mse_loss(dec(enc(xv)), xv))
        history.append(rec)
        if cfg.log_every and ep % cfg.log_every == 0:
            _logger.info("epoch %d %s", ep, rec)

    enc.eval(); dec.eval()
    enc._sawnergy_norm = (mu, sd)          # type: ignore[attr-defined]
    return enc, dec, history


def encode_frames(encoder, X: np.ndarray, *, device: str | None = None,
                  batch_size: int = 4096) -> np.ndarray:
    """Apply a trained frame encoder, reusing its stored normalisation."""
    torch = _torch()
    mu, sd = getattr(encoder, "_sawnergy_norm", (0.0, 1.0))
    dev = device or next(encoder.parameters()).device
    Xn = (np.asarray(X, dtype=np.float32) - mu) / sd
    out = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch_size):
            b = torch.from_numpy(Xn[i:i + batch_size]).to(dev)
            out.append(encoder(b).cpu().numpy())
    return np.concatenate(out, axis=0)
