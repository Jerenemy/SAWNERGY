"""Replicate-aware statistics.

Everything in this module exists to answer one question: **is the difference I
see between two systems bigger than the difference between two runs of the same
system?** If it is not, the embedding is measuring MD noise and every rescue
score built on it is meaningless.

Three rules are enforced throughout:

1. **The unit of analysis is the run, never the frame.** Frames within a
   trajectory are autocorrelated on the ns timescale, so treating them as
   independent samples inflates n by orders of magnitude and will report
   p < 1e-30 for two runs of the *same* system. Functions here take one vector
   per run.

2. **Permutations shuffle run labels.** A frame-level permutation null is
   broken for the same reason -- it tests "are frames exchangeable", which they
   are not, instead of "are runs exchangeable", which is the actual hypothesis.

3. **Effect size is reported next to significance.** With enough replicates a
   trivial difference becomes significant; what matters for a rescue metric is
   whether the system separation is large *relative to* replicate scatter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# variance decomposition
# ---------------------------------------------------------------------------

@dataclass
class VarianceComponents:
    """One-way (system) decomposition of run-level vectors.

    Attributes:
        ss_between: Between-system sum of squares.
        ss_within: Within-system (i.e. between-replicate) sum of squares.
        ms_between, ms_within: Corresponding mean squares.
        f_stat: Pseudo-F = ms_between / ms_within.
        icc: Intraclass correlation -- the fraction of total variance
            attributable to system identity. This is the headline number.
        n_groups, n_runs: Design sizes.
        p_value: Permutation p-value, if computed.
        n_permutations: Permutations used.
    """

    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    f_stat: float
    icc: float
    n_groups: int
    n_runs: int
    p_value: float | None = None
    n_permutations: int = 0

    def __str__(self) -> str:
        p = "n/a" if self.p_value is None else f"{self.p_value:.4g}"
        return (f"ICC={self.icc:.3f}  F={self.f_stat:.3f}  p={p}  "
                f"({self.n_groups} systems, {self.n_runs} runs)")


def _group_indices(labels: np.ndarray) -> dict:
    out: dict = {}
    for i, l in enumerate(labels):
        out.setdefault(l, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def _f_and_ss(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, int]:
    """Return (ss_between, ss_within, n_groups) for run-level vectors."""
    groups = _group_indices(labels)
    grand = X.mean(axis=0, keepdims=True)
    ss_b = 0.0
    ss_w = 0.0
    for idx in groups.values():
        Xg = X[idx]
        mu = Xg.mean(axis=0, keepdims=True)
        ss_b += len(idx) * float(((mu - grand) ** 2).sum())
        ss_w += float(((Xg - mu) ** 2).sum())
    return ss_b, ss_w, len(groups)


def variance_components(X: np.ndarray, system_labels, *,
                        n_permutations: int = 5000,
                        rng: np.random.Generator | None = None,
                        ) -> VarianceComponents:
    """Decompose run-level variance into between-system and between-replicate.

    Args:
        X: ``(n_runs, d)`` one feature vector per *run*. Passing frames here is
            a bug; see the module docstring.
        system_labels: ``(n_runs,)`` system identity per run.
        n_permutations: Permutations for the null. Set 0 to skip.
        rng: Optional generator for reproducibility.

    Returns:
        `VarianceComponents`. ``icc`` is the fraction of variance explained by
        system identity, corrected for finite replicate count; it is negative
        when replicates scatter more than systems do, which is the signature of
        a metric dominated by MD noise.

    Raises:
        ValueError: If fewer than two systems, or any system has one run (in
            which case within-system variance is unestimable).
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(system_labels)
    if X.ndim != 2:
        raise ValueError(f"X must be (n_runs, d); got {X.shape}")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and system_labels disagree on n_runs")

    groups = _group_indices(labels)
    if len(groups) < 2:
        raise ValueError("need >= 2 systems to decompose between-system variance")
    singles = [k for k, v in groups.items() if len(v) < 2]
    if singles:
        raise ValueError(
            f"systems {singles} have a single run: within-system (replicate) "
            "variance cannot be estimated, so no claim about a between-system "
            "difference is supportable. Add replicates.")

    n, k = X.shape[0], len(groups)
    ss_b, ss_w, _ = _f_and_ss(X, labels)
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n - k)
    f_stat = ms_b / ms_w if ms_w > 0 else np.inf

    # balanced-design ICC; n0 is the (harmonic) average group size
    sizes = np.array([len(v) for v in groups.values()], dtype=float)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    var_b = (ms_b - ms_w) / n0
    icc = var_b / (var_b + ms_w) if (var_b + ms_w) != 0 else 0.0

    p_value = None
    if n_permutations > 0 and np.isfinite(f_stat):
        rng = rng or np.random.default_rng(0)
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(labels)          # shuffle RUN labels
            b, w, kk = _f_and_ss(X, perm)
            f = (b / (kk - 1)) / (w / (n - kk)) if w > 0 else np.inf
            if f >= f_stat:
                count += 1
        p_value = (count + 1) / (n_permutations + 1)

    return VarianceComponents(ss_b, ss_w, ms_b, ms_w, f_stat, icc,
                              k, n, p_value, n_permutations)


# ---------------------------------------------------------------------------
# two-way: system and temperature
# ---------------------------------------------------------------------------

def two_way_components(X: np.ndarray, system_labels, temperature_labels, *,
                       n_permutations: int = 2000,
                       rng: np.random.Generator | None = None) -> dict:
    """Partition variance across system and temperature simultaneously.

    Needed because temperature is not a nuisance here: for Y220C the phenotype
    *is* thermal. A metric that responds only to temperature, or only to system,
    is diagnostic of a different failure, so both effects are reported with
    their own permutation null (each permuted within the levels of the other,
    which preserves the design).

    Returns:
        dict with ``icc_system``, ``icc_temperature``, ``f_system``,
        ``f_temperature``, ``p_system``, ``p_temperature``.
    """
    X = np.asarray(X, dtype=np.float64)
    sysl = np.asarray(system_labels)
    templ = np.asarray(temperature_labels)
    rng = rng or np.random.default_rng(0)

    vc_sys = variance_components(X, sysl, n_permutations=0)
    vc_tmp = variance_components(X, templ, n_permutations=0)

    def _restricted_p(observed: float, target, blocking) -> float:
        """Permute `target` labels *within* each level of `blocking`."""
        n = X.shape[0]
        count = 0
        for _ in range(n_permutations):
            perm = np.array(target, copy=True)
            for lvl in np.unique(blocking):
                m = blocking == lvl
                perm[m] = rng.permutation(perm[m])
            b, w, kk = _f_and_ss(X, perm)
            f = (b / (kk - 1)) / (w / (n - kk)) if w > 0 else np.inf
            if f >= observed:
                count += 1
        return (count + 1) / (n_permutations + 1)

    p_sys = _restricted_p(vc_sys.f_stat, sysl, templ) if n_permutations else None
    p_tmp = _restricted_p(vc_tmp.f_stat, templ, sysl) if n_permutations else None

    return {"icc_system": vc_sys.icc, "icc_temperature": vc_tmp.icc,
            "f_system": vc_sys.f_stat, "f_temperature": vc_tmp.f_stat,
            "p_system": p_sys, "p_temperature": p_tmp,
            "n_runs": int(X.shape[0])}


# ---------------------------------------------------------------------------
# held-out replicate discriminability
# ---------------------------------------------------------------------------

@dataclass
class Discriminability:
    """Leave-one-replicate-out nearest-centroid accuracy.

    This is the operational version of the question: given a *new* run of an
    unknown system, can the representation say which system it came from? It is
    strictly harder than an in-sample separation and it is the number to quote.
    """

    accuracy: float
    chance: float
    n_runs: int
    per_system: dict = field(default_factory=dict)
    p_value: float | None = None

    def __str__(self) -> str:
        p = "n/a" if self.p_value is None else f"{self.p_value:.4g}"
        return (f"LORO accuracy={self.accuracy:.3f} (chance={self.chance:.3f}) "
                f"p={p} over {self.n_runs} runs")


def leave_one_replicate_out(X: np.ndarray, system_labels, *,
                            n_permutations: int = 2000,
                            rng: np.random.Generator | None = None,
                            ) -> Discriminability:
    """Hold out one run at a time; classify it by nearest system centroid.

    The held-out run is excluded from its own centroid, which is the whole
    point: including it guarantees near-perfect accuracy and tells you nothing.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(system_labels)
    groups = _group_indices(labels)
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError("every system needs >= 2 runs for leave-one-run-out")

    def _accuracy(lab: np.ndarray) -> tuple[float, dict]:
        g = _group_indices(lab)
        correct = 0
        per: dict = {k: [0, 0] for k in g}
        for i in range(X.shape[0]):
            cents, keys = [], []
            for k, idx in g.items():
                idx2 = idx[idx != i]
                if len(idx2) == 0:
                    continue
                cents.append(X[idx2].mean(axis=0))
                keys.append(k)
            d = np.linalg.norm(np.stack(cents) - X[i], axis=1)
            pred = keys[int(np.argmin(d))]
            per[lab[i]][1] += 1
            if pred == lab[i]:
                correct += 1
                per[lab[i]][0] += 1
        return correct / X.shape[0], {k: v[0] / v[1] for k, v in per.items() if v[1]}

    acc, per_system = _accuracy(labels)
    sizes = np.array([len(v) for v in groups.values()], dtype=float)
    chance = float((sizes / sizes.sum()).max())

    p_value = None
    if n_permutations > 0:
        rng = rng or np.random.default_rng(0)
        count = sum(1 for _ in range(n_permutations)
                    if _accuracy(rng.permutation(labels))[0] >= acc)
        p_value = (count + 1) / (n_permutations + 1)

    return Discriminability(acc, chance, int(X.shape[0]), per_system, p_value)


# ---------------------------------------------------------------------------
# effect size and power
# ---------------------------------------------------------------------------

def separation_ratio(X: np.ndarray, labels, a, b) -> float:
    """Distance between two system means in units of replicate scatter.

    This is a multivariate Cohen's d: ``||mu_a - mu_b|| / sigma_within``. It
    answers "how many replicate-noise widths apart are these systems", which is
    what decides whether a rescue score can rank compounds.

    A value below ~1 means two runs of the same system routinely differ as much
    as the two systems do.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    Xa, Xb = X[labels == a], X[labels == b]
    if len(Xa) < 2 or len(Xb) < 2:
        raise ValueError("both systems need >= 2 runs to estimate scatter")
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    # pooled within-system RMS distance to own centroid
    ss = (((Xa - mu_a) ** 2).sum() + ((Xb - mu_b) ** 2).sum())
    dof = (len(Xa) - 1) + (len(Xb) - 1)
    sigma = np.sqrt(ss / dof)
    return float(np.linalg.norm(mu_a - mu_b) / sigma) if sigma > 0 else np.inf


def replicates_required(effect: float, *, power: float = 0.8,
                        alpha: float = 0.05, max_n: int = 200) -> int:
    """Replicates per system needed to detect a given separation ratio.

    Uses the normal approximation for a two-sample comparison of means, which
    is adequate for planning. ``effect`` is the `separation_ratio` measured on
    a pilot.

    Returns:
        Replicates per system, clipped at ``max_n``.
    """
    if effect <= 0:
        return max_n
    try:
        from scipy.stats import norm
        z_a = norm.ppf(1 - alpha / 2)
        z_b = norm.ppf(power)
    except Exception:                      # pragma: no cover - scipy optional
        z_a, z_b = 1.959963985, 0.8416212336
    n = 2 * ((z_a + z_b) / effect) ** 2
    return int(min(max_n, max(2, np.ceil(n))))


def autocorrelation_time(series: np.ndarray, *, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time of a scalar series, in frames.

    Used to justify the stride and to compute the *effective* number of
    independent frames inside one run. Note that even a large effective frame
    count does not substitute for replicates: frames sample within one basin,
    replicates sample across basins.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    x = x - x.mean()
    n = x.size
    if n < 4 or x.std() == 0:
        return 1.0
    max_lag = max_lag or min(n // 2, 500)
    var = (x @ x) / n
    tau = 0.5
    for lag in range(1, max_lag):
        c = (x[:-lag] @ x[lag:]) / (n - lag) / var
        if c <= 0:
            break
        tau += c
    return float(max(1.0, 2.0 * tau))
