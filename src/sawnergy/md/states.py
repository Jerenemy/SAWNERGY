"""Conformational state populations: *which* shapes a run visits, and how often.

Every run-level descriptor in this package so far has been a **marginal** time
average. `run_descriptor` reports, per residue pair, the fraction of frames in
which that pair was in contact. Stage G added more marginals -- side-chain
contacts, closest heavy-atom approach, dihedral means, burial -- and none of
them beat the original on the contrast that mattered.

Marginals throw away the *joint* structure of the ensemble, and the loss is not
subtle. Take two contacts, A and B, each formed half the time. One run flips
between **both formed** and **neither formed**; the other flips between
**A only** and **B only**. The occupancy vector is ``[0.5, 0.5]`` for both.
They are visiting entirely different sets of shapes. No amount of adding more
marginal features recovers the difference, because the difference is in the
correlation, not in any one feature's mean.

A Markov state model keeps it. Cluster the frames of every run into a shared
set of discrete conformational states, then describe each run by **how much
time it spent in each state**:

    [0.30, 0.50, 0.20, ...]

That is still order-free -- a converged equilibrium trajectory is a sample from
a stationary distribution, frame ordering is near-arbitrary, and an order-aware
descriptor would latch onto run-specific drift, which is the between-replicate
variance everything else here is trying to suppress (see
`temporal`'s module docstring and MD_EMBEDDER.md section 2.1). But unlike a
marginal mean, the population vector is a description of the joint
distribution: state 7 *is* the shape where A is formed and B is not.

It also has a property no other descriptor here has: it is **interpretable**.
"Wild type spends 60% of its time in the closed-pocket state and the mutant
20%" is a sentence a structural biologist can argue with. The current
embeddings can rank systems but cannot say what changed.

Two warnings, both load-bearing.

**One shared clustering, fitted on training runs only.** Population vectors are
comparable only if every run is scored against the *same* states. Clustering
each run separately produces k numbers per run that mean something different
for every run, which is not a feature space. `MarkovStates.fit` therefore takes
the training runs, pools their frames, and clusters once; `populations` then
scores any run -- including runs and systems the clustering never saw -- against
those fixed centres. That is what makes this an open-set descriptor, usable for
a compound-bound trajectory that did not exist when the states were defined.

**The frame budget is the binding constraint.** A population estimate from
``T`` frames is a multinomial proportion with standard error
``sqrt(p(1-p)/T)``. On this panel ``T = 90``. With ``k = 50`` states the mean
population is 0.02 and its standard error is 0.015 -- the noise on each
coordinate is the same order as the coordinate. That noise lands directly in
the denominator of the separation ratio, so a large ``k`` can easily lose to
plain contact occupancy for reasons that have nothing to do with whether the
joint structure carries signal. Hence `soft=True` by default, which averages a
kernel over all centres instead of counting one hard hit per frame and cuts the
variance substantially, and hence `multinomial_noise`, which reports the floor
so a null result can be attributed rather than merely reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)

#: Added to transition counts before row-normalising. With ``T - lag``
#: transitions observed per run and ``k**2`` matrix entries, most entries are
#: genuinely zero; dividing by a zero row sum is the common case, not an edge
#: case.
TRANSITION_PRIOR = 1e-3


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def kmeans(X: np.ndarray, k: int, *, seed: int = 0, n_init: int = 4,
           max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """``(k, P)`` centres by Lloyd's algorithm with k-means++ seeding.

    Hand-rolled rather than taken from scikit-learn, which is not a dependency
    of this package -- as with `TICA` and the PCA used throughout the stage
    scripts, the linear algebra here is short enough that adding a dependency
    costs more than it saves.

    Args:
        X: ``(N, P)`` frames, pooled across **training runs only**.
        k: Number of states. Must not exceed ``N``.
        n_init: Independent restarts; the lowest-inertia one is returned.
            Lloyd's algorithm is only locally optimal and k-means++ seeding is
            randomised, so a single run is a coin toss.

    Returns:
        ``(k, P)`` centres, ordered by descending population, so that state
        indices are stable enough to be quoted in prose.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (N, P); got {X.shape}")
    n = X.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, N={n}]; got {k}")

    best_centres, best_inertia = None, np.inf
    for init in range(n_init):
        rng = np.random.default_rng(seed + init)
        centres = _kmeanspp(X, k, rng)
        prev = np.inf
        for _ in range(max_iter):
            lab = _nearest(X, centres)
            for j in range(k):
                m = lab == j
                if m.any():
                    centres[j] = X[m].mean(axis=0)
                else:
                    # An empty cluster contributes nothing and leaves a wasted
                    # dimension in every population vector. Re-seed it on the
                    # worst-fitted frame instead of dropping it, which would
                    # make the feature width depend on the data.
                    d = ((X - centres[lab]) ** 2).sum(axis=1)
                    centres[j] = X[int(np.argmax(d))]
            inertia = float(((X - centres[_nearest(X, centres)]) ** 2).sum())
            if prev - inertia <= tol * max(prev, 1.0):
                break
            prev = inertia
        if inertia < best_inertia:
            best_centres, best_inertia = centres.copy(), inertia

    lab = _nearest(X, best_centres)
    order = np.argsort(np.bincount(lab, minlength=k))[::-1]
    return best_centres[order]


def _kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centres = np.empty((k, X.shape[1]))
    centres[0] = X[rng.integers(X.shape[0])]
    d2 = ((X - centres[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:
            centres[j] = X[rng.integers(X.shape[0])]
        else:
            centres[j] = X[rng.choice(X.shape[0], p=d2 / total)]
        d2 = np.minimum(d2, ((X - centres[j]) ** 2).sum(axis=1))
    return centres


def _nearest(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.argmin(_sqdist(X, centres), axis=1)


def _sqdist(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """``(N, k)`` squared euclidean distances via the gram identity.

    Clipped at zero: the identity is exact in real arithmetic but can return
    small negatives in floating point when a frame sits on a centre, and a
    negative squared distance propagates into ``sqrt`` as a NaN.
    """
    d2 = ((X ** 2).sum(1)[:, None] - 2 * X @ centres.T
          + (centres ** 2).sum(1)[None, :])
    return np.maximum(d2, 0.0)


# ---------------------------------------------------------------------------
# state populations
# ---------------------------------------------------------------------------

@dataclass
class MarkovStates:
    """A shared discrete state decomposition and the run descriptors it gives.

    Attributes:
        centres: ``(k, P)`` state centres in whatever coordinates were fitted
            (TICA components, here).
        bandwidth: Kernel width for soft assignment, set at fit time to the
            median frame-to-nearest-centre distance over the training frames.
            Fixing it from the training data keeps `populations` free of tuning
            knobs at scoring time.
        n_frames_fit: Number of training frames pooled, for the record.
    """

    centres: np.ndarray
    bandwidth: float
    n_frames_fit: int

    @property
    def k(self) -> int:
        return int(self.centres.shape[0])

    @classmethod
    def fit(cls, runs, k: int, *, seed: int = 0, n_init: int = 4
            ) -> "MarkovStates":
        """Cluster the pooled frames of ``runs`` into ``k`` shared states.

        Args:
            runs: One ``(T_i, P)`` array per run. Pass **training runs only**;
                frames from held-out runs must not influence where the states
                sit, or every number downstream is contaminated.
            k: Number of states. See the module docstring on why this trades
                resolution against multinomial noise at ``T = 90``.
        """
        runs = [np.asarray(r, dtype=np.float64) for r in runs]
        if not runs:
            raise ValueError("no runs supplied")
        P = runs[0].shape[1]
        if any(r.shape[1] != P for r in runs):
            raise ValueError("every run must have the same feature width")
        pool = np.concatenate(runs)
        centres = kmeans(pool, k, seed=seed, n_init=n_init)
        nearest = np.sqrt(_sqdist(pool, centres).min(axis=1))
        bw = float(np.median(nearest))
        if not bw > 0:
            # Degenerate only when every frame sits exactly on a centre, which
            # happens in tests with k == N. A zero bandwidth would make the
            # kernel a delta function and divide by zero.
            bw = 1.0
        return cls(centres=centres, bandwidth=bw, n_frames_fit=int(len(pool)))

    def assign(self, X: np.ndarray) -> np.ndarray:
        """``(T,)`` hard state index per frame of a ``(T, P)`` run."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return _nearest(X, self.centres)

    def memberships(self, X: np.ndarray, *, soft: bool = True) -> np.ndarray:
        """``(T, k)`` per-frame state memberships.

        With ``soft=False`` these are one-hot. With ``soft=True`` they are a
        Gaussian kernel over the centres, normalised per frame -- which is what
        a frame genuinely between two states deserves, and which reduces the
        variance of the population estimate at fixed ``T``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        d2 = _sqdist(X, self.centres)
        if not soft:
            out = np.zeros_like(d2)
            out[np.arange(len(d2)), np.argmin(d2, axis=1)] = 1.0
            return out
        # subtract the row minimum before exponentiating: the shift cancels in
        # the normalisation and keeps exp() away from underflow
        logw = -d2 / (2.0 * self.bandwidth ** 2)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        return w / w.sum(axis=1, keepdims=True)

    def populations(self, X: np.ndarray, *, soft: bool = True,
                    hellinger: bool = False) -> np.ndarray:
        """``(k,)`` fraction of the run's time spent in each state.

        Args:
            hellinger: Return ``sqrt(p)`` instead of ``p``. Populations are
                compositional, and the square root is the variance-stabilising
                transform for multinomial proportions: euclidean distance
                between root-populations is the Hellinger distance between the
                two ensembles, which is the natural distance between
                distributions, and it stops a handful of high-occupancy states
                from dominating the covariance.
        """
        p = self.memberships(X, soft=soft).mean(axis=0)
        return np.sqrt(p) if hellinger else p

    def transition_matrix(self, X: np.ndarray, *, lag: int = 1,
                          prior: float = TRANSITION_PRIOR) -> np.ndarray:
        """``(k, k)`` row-normalised transition probabilities for one run.

        This is the one descriptor here that reads the time axis. It is still
        replicate-safe: a transition *rate* is a property of the dynamics and is
        shared between replicates, whereas the specific sequence of visits is
        not. What it is **not** is well-estimated -- a 90-frame run at lag 1
        offers 89 transitions to fill ``k**2`` entries, so anything past
        ``k`` of about 10 is mostly prior. `populations` is the descriptor to
        reach for first.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if not 0 < lag < len(X):
            raise ValueError(f"lag must be in (0, T={len(X)}); got {lag}")
        lab = self.assign(X)
        C = np.full((self.k, self.k), float(prior))
        np.add.at(C, (lab[:-lag], lab[lag:]), 1.0)
        return C / C.sum(axis=1, keepdims=True)


def multinomial_noise(populations: np.ndarray, n_frames: int) -> np.ndarray:
    """``(k,)`` standard error of each population, ``sqrt(p(1-p)/T)``.

    The honest companion to any population vector: it says how much of the
    between-run scatter is sampling noise rather than system difference. With
    ``T = 90`` and ``k = 50`` this is the same order as the populations
    themselves, which is a fact about the trajectories and not about the method.
    """
    p = np.asarray(populations, dtype=np.float64)
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    return np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n_frames)
