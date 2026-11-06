"""Features that use the time axis, instead of averaging it away.

Everything the statistics in this package consumed until now was a **time
average**: `run_descriptor` reduces a trajectory to the mean contact occupancy
per residue pair, one vector per run, and PCA sees only that. Ninety frames
collapse to one number per pair before any model is fitted. Every dynamical
quantity -- how fast a contact breaks, which motions are slow, whether two
regions move together -- is integrated out first.

The nested autoencoder is the one component that sees frames, with timestamps,
and on the RBD panel it scored *identically* to time-averaged PCA on held-out
runs (0.856 both). Two readings: the temporal structure carries no extra
discriminative information, or an architecture given time and a reconstruction
objective does not learn to use it. This module exists to distinguish them, by
extracting temporal structure explicitly rather than hoping a network finds it.

- `fluctuation` -- per-feature standard deviation over time. Already computed
  inside `run_descriptor` as ``contact_fluct`` and never used by anything.
- `autocorrelation` -- lag-tau autocorrelation per feature: how persistent each
  contact is, independent of how often it is formed.
- `lifetime` -- mean dwell time above a threshold, in frames. Two pairs can
  share an occupancy of 0.5 while one flickers every frame and the other sits
  formed for half the trajectory; the mean cannot tell them apart.
- `tica` -- time-lagged independent component analysis. PCA maximises variance;
  TICA maximises autocorrelation at lag tau, recovering the *slow* collective
  modes. These are not the same directions, and slow modes are the standard
  coordinates for distinguishing conformational ensembles. A reconstruction-
  trained autoencoder and PCA are both chasing variance, which is a candidate
  explanation for why they keep tying.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)


def fluctuation(X: np.ndarray) -> np.ndarray:
    """``(P,)`` standard deviation over time of a ``(T, P)`` frame matrix."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"expected (T, P); got {X.shape}")
    return X.std(axis=0).astype(np.float32)


def autocorrelation(X: np.ndarray, lag: int = 1) -> np.ndarray:
    """``(P,)`` lag-``lag`` autocorrelation of each column of ``(T, P)``.

    Features that never change have undefined autocorrelation; they are
    reported as 0 rather than NaN so the vector stays usable.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (T, P); got {X.shape}")
    if not 0 < lag < X.shape[0]:
        raise ValueError(f"lag must be in (0, T); got {lag} with T={X.shape[0]}")
    A, B = X[:-lag], X[lag:]
    a = A - A.mean(0)
    b = B - B.mean(0)
    num = (a * b).sum(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    out = np.zeros(X.shape[1])
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out.astype(np.float32)


def lifetime(X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """``(P,)`` mean number of consecutive frames each feature stays above
    ``threshold``.

    Returns 0 for features that never cross it. Occupancy and lifetime are
    independent: a pair formed 50% of the time in one long stretch and a pair
    formed 50% of the time in single-frame flickers have the same mean and very
    different kinetics.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"expected (T, P); got {X.shape}")
    on = X > threshold
    T, P = on.shape
    out = np.zeros(P, dtype=np.float32)
    for p in range(P):
        col = on[:, p]
        if not col.any():
            continue
        # run-length encode the True stretches
        d = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        out[p] = float((ends - starts).mean())
    return out


# ---------------------------------------------------------------------------
# TICA
# ---------------------------------------------------------------------------

@dataclass
class TICA:
    """Time-lagged independent component analysis.

    Solves the generalised eigenproblem ``C_tau v = lambda C_0 v`` on the
    symmetrised time-lagged covariance, giving directions ordered by
    autocorrelation rather than by variance.

    Attributes:
        mean: Feature mean used for centring.
        components: ``(k, P)`` projection directions.
        eigenvalues: Autocorrelation of each component at the fitted lag; these
            are the quantities being maximised, and they map to implied
            timescales ``-lag / ln(lambda)``.
        lag: Lag used, in frames.
    """

    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    lag: int

    @classmethod
    def fit(cls, runs, *, lag: int = 1, n_components: int = 16,
            reg: float = 1e-6) -> "TICA":
        """Fit on a list of ``(T_i, P)`` per-run frame matrices.

        Covariances are pooled across runs and lagged pairs are never taken
        across a run boundary -- frame T of one replicate does not precede frame
        0 of the next, and treating it as though it did invents dynamics.

        Args:
            runs: One ``(T_i, P)`` array per run. Pass **training runs only**.
            lag: Lag in frames. Too short and every mode looks slow; too long
                and the lagged covariance is pure noise.
            reg: Ridge added to ``C_0`` before inversion. With P >> number of
                frames this is load-bearing, not cosmetic.
        """
        runs = [np.asarray(r, dtype=np.float64) for r in runs]
        if not runs:
            raise ValueError("no runs supplied")
        P = runs[0].shape[1]
        if any(r.shape[1] != P for r in runs):
            raise ValueError("every run must have the same feature width")
        usable = [r for r in runs if r.shape[0] > lag]
        if not usable:
            raise ValueError(
                f"no run is longer than the lag ({lag} frames); TICA has "
                f"nothing to correlate")
        if len(usable) < len(runs):
            _logger.warning("%d of %d runs are shorter than the lag and were "
                            "dropped", len(runs) - len(usable), len(runs))

        mu = np.concatenate([r for r in usable]).mean(axis=0)
        C0 = np.zeros((P, P))
        Ct = np.zeros((P, P))
        n = 0
        for r in usable:
            A = r[:-lag] - mu
            B = r[lag:] - mu
            C0 += A.T @ A + B.T @ B
            Ct += A.T @ B + B.T @ A
            n += 2 * A.shape[0]
        C0 /= n
        Ct /= n
        C0 += reg * np.trace(C0) / P * np.eye(P)

        # whiten by C0, then diagonalise the lagged covariance
        w, V = np.linalg.eigh(C0)
        keep = w > w.max() * 1e-10
        Wh = V[:, keep] / np.sqrt(w[keep])
        M = Wh.T @ Ct @ Wh
        M = 0.5 * (M + M.T)
        lam, U = np.linalg.eigh(M)
        order = np.argsort(lam)[::-1][:n_components]
        comp = (Wh @ U[:, order]).T
        return cls(mean=mu, components=comp.astype(np.float64),
                   eigenvalues=lam[order], lag=int(lag))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``(T, P)`` frames or ``(P,)`` onto the slow modes."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return (X - self.mean) @ self.components.T

    def transform_run(self, X: np.ndarray) -> np.ndarray:
        """``(k,)`` run-level embedding: the mean projection over frames.

        The mean is used so the result is directly comparable with the existing
        run-level descriptors; the difference from PCA is which *directions* are
        averaged along, not that the average has been removed.
        """
        return self.transform(X).mean(axis=0)

    def timescales(self) -> np.ndarray:
        """Implied timescale of each component, in frames: ``-lag / ln(lambda)``.

        A mode with ``lambda <= 0`` anticorrelates across the lag and has no
        relaxation time in this sense. It is reported as **0**, not ``inf``:
        such a mode is fast, not infinitely slow, and reporting ``inf`` would
        both invert its physical meaning and break the ordering that sorting by
        eigenvalue is supposed to guarantee. ``lambda >= 1`` would be infinitely
        slow, but cannot occur for a correctly symmetrised lagged covariance.
        """
        lam = np.clip(self.eigenvalues, 1e-12, 1 - 1e-12)
        out = -self.lag / np.log(lam)
        out[self.eigenvalues <= 0] = 0.0
        return out
