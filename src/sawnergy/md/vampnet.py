"""VAMPnets: learn the state decomposition instead of clustering into one.

`states` builds a Markov state model the classical way -- project onto slow
modes with `TICA`, then cluster the projections with k-means. Both halves are
linear and neither is trained on the quantity that matters. TICA finds
directions of maximal autocorrelation under a linear model; k-means then
partitions them under an isotropic-variance assumption. If the slow coordinate
is a nonlinear function of the dihedrals -- a torsion flip, a pocket opening --
TICA has to approximate it with a linear combination, and whatever it loses the
clustering cannot recover.

A VAMPnet (Mardt, Pasquali, Wu & Noe, *Nat. Commun.* 9:5 (2018)) replaces both
steps with one network. A single encoder maps a frame to a soft assignment over
``k`` states, and it is trained to maximise the **VAMP-2 score**: how much of
the dynamics at lag tau the assignment can account for. That is a variational
bound on the slow subspace, so unlike a reconstruction objective it is aimed at
exactly the property the descriptor is supposed to capture.

That distinction is the reason this module exists. The nested autoencoder in
`autoencoder` is trained to reconstruct, and section 8.2 of MD_EMBEDDER.md
records what that bought: between two configurations its reconstruction
improved while its ability to tell systems apart got *worse*. It was learning
the wrong thing harder. PCA and a reconstruction autoencoder are both chasing
variance; slow modes are not high-variance modes. VAMP-2 is the first objective
in this package that is not a proxy.

What is deliberately **not** here is a system-label objective. A contrastive
loss trained to separate WT from mutant would find the most reliable
discriminator available, and stage G measured what that is: masking the
substituted residues and their 8 A shell costs CA LORO 0.883 -> 0.631, so most
of the apparent signal *is* the substituted residue. A model trained on system
labels would learn residue 417's identity -- which is perfectly predictive,
trivially learnable, and the one feature a compound can never change. Every
such model would score every compound as unrescued. The VAMP-2 score never sees
a system label; it sees only pairs of frames from the same trajectory, so the
quantity it optimises is available for a compound-bound run that did not exist
when the model was trained. The descriptor stays open-set.

The run descriptor is the same shape as the classical one -- mean soft
assignment over frames, a population vector -- so it drops into the stage
protocol as one more arm and is scored against CA contact occupancy on the same
held-out runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

try:  # torch is optional for the rest of the package
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by import guard
    torch = None
    nn = None

_logger = logging.getLogger(__name__)

#: Ridge added to the covariances before inverting, **relative to their own
#: trace** -- the same form `TICA.fit` uses, and for the same reason.
#:
#: The score needs ``C00^{-1/2}``, and a softmax output has two ways of making
#: ``C00`` near-singular: a state the network has stopped using, and -- far more
#: dangerous -- the near-uniform output every softmax starts from at
#: initialisation, where every column is almost identical and the covariance is
#: almost rank one. An absolute ridge cannot cope with both, because the scale
#: of ``C00`` depends on ``k``: entries of a ``k``-state assignment covariance
#: are of order ``1/k**2``, so a fixed 1e-6 is negligible for ``k = 2`` and
#: comparable to the signal for ``k = 100``. Scaling by the trace makes the
#: regularisation mean the same thing at every ``k``, and is what stops the
#: first few gradient steps from being dominated by whichever direction the
#: untrained network happens to have collapsed along.
COV_EPS = 1e-4

#: A best-restart inner-validation score below this is reported as a collapse
#: rather than as a result. It is deliberately a hair above zero: a genuinely
#: uninformative assignment and a genuinely memoryless process both score here,
#: and the point of the flag is to stop the first being mistaken for the second.
COLLAPSE_SCORE = 0.01

#: Absolute lower bound on the ridge, for the degenerate case the relative one
#: cannot cover: a covariance that is identically zero has nothing to be
#: relative to. See `_inv_sqrt`.
RIDGE_FLOOR = 1e-12


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "VAMPnets need PyTorch; the rest of sawnergy.md does not. "
            "Install torch or use sawnergy.md.states for the classical "
            "TICA + k-means route.")


@dataclass
class VampConfig:
    """Training configuration.

    Attributes:
        n_states: Width of the softmax output -- the number of states.
        hidden: Hidden layer widths.
        lag: Lag in frames for the lagged pairs. Matched to the TICA lag used
            elsewhere so the two routes are compared at the same timescale.
        epochs: Maximum epochs; `fit_vampnet` early-stops on an inner
            validation split and keeps the best weights.
        patience: Epochs without inner-validation improvement before stopping.
        inner_val_frac: Fraction of the **training runs** held back to early-stop
            on. It must be runs and not frames: frames from the same trajectory
            are not independent, so a frame-level split would let the model
            select on data it has effectively seen, and it must come out of the
            training runs and never the stage's held-out runs, which no fitting
            decision may touch.
        dropout: Applied between hidden layers.
        n_init: Independent restarts, the best by **inner validation** score
            kept. Not a nicety: a softmax starts near-uniform, which is a dead
            local optimum of the VAMP-2 score -- the covariance is near rank
            one, the gradient through the whitening nearly vanishes, and a run
            that begins there never escapes it. Measured on a synthetic
            two-state process, roughly one fit in six collapsed that way even
            with the trace-relative ridge, and early stopping then ends it
            before it can recover. Restarts are the same remedy `kmeans`
            applies to the same class of problem, and selecting among them on
            the inner split is legitimate for the same reason early stopping on
            it is: the stage's held-out runs are not consulted.
    """

    n_states: int = 50
    hidden: tuple[int, ...] = (100, 100)
    lag: int = 5
    epochs: int = 300
    patience: int = 40
    n_init: int = 3
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    inner_val_frac: float = 0.2
    seed: int = 0
    device: str | None = None


@dataclass
class VampReport:
    """What a fit did, for the record rather than for the result.

    Attributes:
        train_score: Best VAMP-2 score on the inner training runs.
        val_score: VAMP-2 score on the inner validation runs at the kept epoch.
        best_epoch: Epoch whose weights were kept.
        epochs_run: Epochs actually run before early stopping.
        max_score: ``n_states - 1``, the ceiling for a centred VAMP-2 score.
            Reaching it means the assignment is as informative about the lag-tau
            dynamics as a ``k``-state partition can be. Note it is a ceiling and
            usually not an attainable one: for a two-state process with
            per-frame flip probability ``p``, the attainable score at lag ``tau``
            is ``(1 - 2p)**(2*tau)``, so a well-fitted model on genuinely
            stochastic dynamics sits far below ``k - 1``.
        collapsed: True when even the best restart scored near zero, i.e. the
            network found no slow process. Reported explicitly so that a failed
            fit cannot be quietly read as evidence that the dynamics carry no
            signal -- those are different claims and only one of them is about
            the data.
        restarts: Per-restart best inner-validation scores, including the ones
            discarded.
        history: Per-epoch ``(train, val)`` scores of the kept restart.
    """

    train_score: float
    val_score: float
    best_epoch: int
    epochs_run: int
    max_score: float
    collapsed: bool = False
    restarts: list[float] = field(default_factory=list)
    history: list[tuple[float, float]] = field(default_factory=list)

    def __str__(self) -> str:
        tail = "  COLLAPSED" if self.collapsed else ""
        restarts = (f" [restarts {', '.join(f'{s:.3f}' for s in self.restarts)}]"
                    if len(self.restarts) > 1 else "")
        return (f"VAMP-2 train {self.train_score:.3f} / val "
                f"{self.val_score:.3f} of at most {self.max_score:.0f} "
                f"(epoch {self.best_epoch}/{self.epochs_run}){restarts}{tail}")


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def _inv_sqrt(C, eps: float = COV_EPS):
    """``C^{-1/2}`` with a trace-relative ridge. See `COV_EPS`.

    The ridge is floored at `RIDGE_FLOOR` because a trace-relative one vanishes
    in exactly the case it is most needed: an assignment collapsed onto a single
    state has a centred covariance that is identically zero, so its trace is
    zero, and a purely relative ridge would leave ``rsqrt(0) = inf`` to
    propagate a NaN into the score. With the floor the whitening of a zero
    covariance is finite, the cross-covariance is zero too, and the score comes
    out as 0 -- which is the right answer for an assignment that resolves
    nothing.
    """
    k = C.shape[-1]
    ridge = eps * torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / k
    ridge = torch.clamp(ridge, min=RIDGE_FLOOR)
    C = C + ridge * torch.eye(k, dtype=C.dtype, device=C.device)
    w, V = torch.linalg.eigh(C)
    w = torch.clamp(w, min=ridge * 1e-3)
    return (V * w.rsqrt()) @ V.transpose(-1, -2)


def vamp2_score(chi0, chit, *, eps: float = COV_EPS):
    """VAMP-2 score of a soft assignment, on **centred** covariances.

    ``|| C00^{-1/2} C0t C tt^{-1/2} ||_F^2``, the sum of squared singular values
    of the whitened cross-covariance -- how much of the lag-tau dynamics the
    assignment resolves.

    Centring removes the trivial stationary singular value, which is exactly 1
    for any valid assignment and carries no information, so the ceiling here is
    ``k - 1`` rather than ``k``. It is removed because leaving it in adds a
    constant to the objective and a term to the gradient that pushes the network
    toward reproducing the stationary distribution it is already guaranteed to
    reproduce.

    Args:
        chi0: ``(N, k)`` assignments of the instantaneous frames.
        chit: ``(N, k)`` assignments of the frames one lag later.
    """
    _require_torch()
    if chi0.shape != chit.shape:
        raise ValueError(f"shapes differ: {tuple(chi0.shape)} vs "
                         f"{tuple(chit.shape)}")
    n = chi0.shape[0]
    if n < 2:
        raise ValueError("need at least two lagged pairs")
    a = chi0 - chi0.mean(0, keepdim=True)
    b = chit - chit.mean(0, keepdim=True)
    c00 = a.T @ a / (n - 1)
    ctt = b.T @ b / (n - 1)
    c0t = a.T @ b / (n - 1)
    K = _inv_sqrt(c00, eps) @ c0t @ _inv_sqrt(ctt, eps)
    return (K ** 2).sum()


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

def build_vampnet(n_features: int, config: VampConfig):
    """An MLP mapping one frame to a softmax over ``config.n_states``."""
    _require_torch()
    torch.manual_seed(config.seed)
    layers: list = []
    w = n_features
    for h in config.hidden:
        layers += [nn.Linear(w, h), nn.ELU()]
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        w = h
    layers += [nn.Linear(w, config.n_states), nn.Softmax(dim=-1)]
    return nn.Sequential(*layers)


def lagged_pairs(runs, lag: int):
    """``(X0, Xt)`` stacked lagged pairs, never crossing a run boundary.

    Frame ``T`` of one replicate does not precede frame 0 of the next, and
    treating it as though it did invents dynamics -- the same guard `TICA.fit`
    applies.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")
    a, b = [], []
    for r in runs:
        r = np.asarray(r, dtype=np.float32)
        if r.shape[0] <= lag:
            continue
        a.append(r[:-lag])
        b.append(r[lag:])
    if not a:
        raise ValueError(f"no run is longer than the lag ({lag} frames)")
    return np.concatenate(a), np.concatenate(b)


def fit_vampnet(runs, config: VampConfig | None = None):
    """Train a VAMPnet on ``runs``; returns ``(model, standardiser, report)``.

    Args:
        runs: One ``(T_i, P)`` array per run. Pass **training runs only**.

    Returns:
        ``model`` in eval mode with the best-inner-validation weights,
        ``standardiser`` as the ``(mean, sd)`` fitted on training frames, and a
        `VampReport`.
    """
    _require_torch()
    config = config or VampConfig()
    dev = torch.device(config.device or
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = [np.asarray(r, dtype=np.float32) for r in runs]
    if len(runs) < 2:
        raise ValueError("need at least two runs to split for early stopping")

    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(len(runs))
    n_val = max(1, int(round(config.inner_val_frac * len(runs))))
    val_runs = [runs[i] for i in perm[:n_val]]
    tr_runs = [runs[i] for i in perm[n_val:]]

    pool = np.concatenate(tr_runs)
    mu = pool.mean(0)
    sd = np.where(pool.std(0) > 1e-6, pool.std(0), 1.0)

    def prep(rs):
        X0, Xt = lagged_pairs([(r - mu) / sd for r in rs], config.lag)
        return (torch.as_tensor(X0, device=dev),
                torch.as_tensor(Xt, device=dev))

    X0, Xt = prep(tr_runs)
    V0, Vt = prep(val_runs)

    overall = None
    restart_scores: list[float] = []
    for init in range(max(1, config.n_init)):
        seed = config.seed + 1000 * init
        model = build_vampnet(
            X0.shape[1], replace_seed(config, seed)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
        gen = torch.Generator().manual_seed(seed)

        best = {"val": -np.inf, "epoch": -1, "state": None, "train": -np.inf}
        history: list[tuple[float, float]] = []
        epoch = 0
        for epoch in range(config.epochs):
            model.train()
            idx = torch.randperm(X0.shape[0], generator=gen).to(dev)
            for s in range(0, len(idx), config.batch_size):
                batch = idx[s:s + config.batch_size]
                if batch.numel() < 2:
                    continue
                opt.zero_grad()
                loss = -vamp2_score(model(X0[batch]), model(Xt[batch]))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                tr_s = float(vamp2_score(model(X0), model(Xt)))
                va_s = float(vamp2_score(model(V0), model(Vt)))
            history.append((tr_s, va_s))
            if va_s > best["val"]:
                best = {"val": va_s, "epoch": epoch, "train": tr_s,
                        "state": {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}}
            elif epoch - best["epoch"] >= config.patience:
                break

        restart_scores.append(float(best["val"]))
        if overall is None or best["val"] > overall[0]["val"]:
            overall = (best, model, epoch + 1, history)

    best, model, epochs_run, history = overall
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    report = VampReport(
        train_score=best["train"], val_score=best["val"],
        best_epoch=int(best["epoch"]), epochs_run=epochs_run,
        max_score=float(config.n_states - 1),
        collapsed=bool(best["val"] < COLLAPSE_SCORE),
        restarts=restart_scores, history=history)
    if report.collapsed:
        _logger.warning("every restart collapsed: %s", report)
    else:
        _logger.info("%s", report)
    return model, (mu, sd), report


def replace_seed(config: VampConfig, seed: int) -> VampConfig:
    """A copy of ``config`` with a different seed, for a restart."""
    return VampConfig(**{**config.__dict__, "seed": seed})


def soft_populations(model, standardiser, run, *, hellinger: bool = False
                     ) -> np.ndarray:
    """``(k,)`` mean soft state assignment over a run's frames.

    The same descriptor `MarkovStates.populations` returns, so the two routes
    are interchangeable downstream -- and, like it, an average over frames and
    therefore invariant to frame order.
    """
    _require_torch()
    mu, sd = standardiser
    X = (np.asarray(run, dtype=np.float32) - mu) / sd
    dev = next(model.parameters()).device
    with torch.no_grad():
        chi = model(torch.as_tensor(X, device=dev)).cpu().numpy()
    p = chi.mean(axis=0).astype(np.float64)
    return np.sqrt(p) if hellinger else p
