"""Staged validation gates for the MD embedder.

Each gate is a falsifiable check with an explicit pass criterion, ordered so
that a failure stops the pipeline before the next stage wastes compute. Nothing
downstream of a failed gate is trustworthy, so `run_gates` reports the first
failure loudly rather than producing a score anyway.

    G0  corpus            -- enough independent replicates to say anything
    G1  features          -- the featuriser is finite, varying and informative
    G2  replicate noise   -- how much do runs of the SAME system differ?
    G3  sham comparison   -- split one system's replicates into two fake
                             "systems". The pipeline MUST fail to separate them.
    G4  separability      -- real systems separate, under replicate-held-out
                             evaluation, with a permutation null
    G5  effect size       -- the separation is large relative to replicate
                             scatter, and the design has the power to see it
    G6  baseline          -- the learned encoder beats a linear baseline
    G7  temperature       -- the representation responds to temperature
                             monotonically and the thermal axis is well fitted
    G8  disentangling     -- system signal is not merely a temperature proxy

G3 is the one that matters most for the question "is this reflecting structure
or just MD variability". A pipeline that reports a significant difference
between two halves of the *same* system's replicates is measuring noise, and no
amount of downstream modelling repairs that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from . import stats as _stats
from .manifest import Corpus

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# gate plumbing
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    """Outcome of a single validation gate."""

    name: str
    passed: bool
    detail: str = ""
    values: dict = field(default_factory=dict)
    fatal: bool = True

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.fatal else "WARN")
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class GateReport:
    gates: list[Gate] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(g.passed or not g.fatal for g in self.gates)

    @property
    def first_failure(self) -> Gate | None:
        for g in self.gates:
            if not g.passed and g.fatal:
                return g
        return None

    def __str__(self) -> str:
        lines = [str(g) for g in self.gates]
        lines.append("-" * 60)
        lines.append(f"OVERALL: {'PASS' if self.ok else 'FAIL'}")
        if f := self.first_failure:
            lines.append(f"Stopped at: {f.name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# replicate-aware splitting
# ---------------------------------------------------------------------------

def replicate_splits(system_labels, replicate_labels, *, n_folds: int = 5,
                     rng: np.random.Generator | None = None):
    """Yield ``(train_idx, test_idx)`` splitting on *runs*, never frames.

    Frames inside a run are autocorrelated. A random frame split puts
    near-duplicate frames on both sides and reports an accuracy that reflects
    memorisation, not generalisation. Every split in this pipeline groups by
    run.
    """
    sysl = np.asarray(system_labels)
    repl = np.asarray(replicate_labels)
    keys = np.array([f"{s}|{r}" for s, r in zip(sysl, repl)])
    uniq = np.unique(keys)
    rng = rng or np.random.default_rng(0)
    order = rng.permutation(uniq)
    folds = np.array_split(order, min(n_folds, len(order)))
    for f in folds:
        test = np.isin(keys, f)
        if test.all() or not test.any():
            continue
        yield np.where(~test)[0], np.where(test)[0]


# ---------------------------------------------------------------------------
# individual gates
# ---------------------------------------------------------------------------

def gate_corpus(corpus: Corpus, **kw) -> Gate:
    """G0: the design can support the comparison."""
    rep = corpus.check(**kw)
    detail = f"{len(corpus)} runs, {len(corpus.systems)} systems"
    if rep.errors:
        detail += " | " + "; ".join(rep.errors[:3])
    return Gate("G0 corpus", rep.ok, detail,
                {"errors": rep.errors, "warnings": rep.warnings, "cells": rep.cells})


def gate_features(X: np.ndarray, *, min_var_frac: float = 0.01) -> Gate:
    """G1: features are finite and carry variance."""
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X).all()
    var = X.var(axis=0)
    live = float((var > 1e-12).mean())
    ok = bool(finite and live >= min_var_frac)
    return Gate("G1 features", ok,
                f"finite={finite}, {live:.1%} of {X.shape[1]} dims vary, "
                f"n_runs={X.shape[0]}",
                {"finite": bool(finite), "live_fraction": live,
                 "n_features": int(X.shape[1])})


def gate_replicate_noise(X: np.ndarray, system_labels,
                         replicate_labels) -> Gate:
    """G2: quantify within-system replicate scatter. Informational, never fatal.

    This gate does not pass or fail on a threshold -- it reports the number that
    every later claim is measured against. If within-system scatter is large,
    that is not itself a bug; it means more replicates are needed.
    """
    X = np.asarray(X, dtype=np.float64)
    sysl = np.asarray(system_labels)
    per = {}
    for s in np.unique(sysl):
        Xs = X[sysl == s]
        if len(Xs) < 2:
            continue
        mu = Xs.mean(0)
        per[str(s)] = float(np.sqrt(((Xs - mu) ** 2).sum(1)).mean())
    overall = float(np.mean(list(per.values()))) if per else float("nan")
    return Gate("G2 replicate noise", True,
                f"mean within-system run-to-centroid distance = {overall:.4g}",
                {"per_system": per, "overall": overall}, fatal=False)


def gate_sham(X: np.ndarray, system_labels, replicate_labels, *,
              n_trials: int = 20, alpha: float = 0.05,
              n_permutations: int = 1000,
              rng: np.random.Generator | None = None) -> Gate:
    """G3: a sham comparison must NOT come out significant.

    Within each real system, replicates are randomly relabelled into two fake
    sub-systems and the full separability test is run. Those halves differ only
    by MD noise, so the false-positive rate across trials must sit near
    ``alpha``. A rate well above ``alpha`` means the statistics are treating
    correlated samples as independent somewhere, and every "significant"
    WT-versus-mutant result from the same code is suspect.

    This is the gate that answers "is the difference real or just MD".
    """
    X = np.asarray(X, dtype=np.float64)
    sysl = np.asarray(system_labels)
    rng = rng or np.random.default_rng(12345)

    pvals: list[float] = []
    for s in np.unique(sysl):
        idx = np.where(sysl == s)[0]
        if idx.size < 4:                      # need >=2 per fake half
            continue
        for _ in range(n_trials):
            perm = rng.permutation(idx)
            half = perm.size // 2
            fake = np.empty(perm.size, dtype=object)
            fake[:half] = "sham_A"
            fake[half:] = "sham_B"
            try:
                vc = _stats.variance_components(
                    X[perm], fake, n_permutations=n_permutations, rng=rng)
            except ValueError:
                continue
            if vc.p_value is not None:
                pvals.append(vc.p_value)

    if not pvals:
        return Gate("G3 sham comparison", False,
                    "could not run: need >= 4 replicates in at least one system",
                    {"n_trials": 0})

    fpr = float(np.mean(np.asarray(pvals) < alpha))
    # binomial upper tolerance: allow roughly 3 sigma above alpha
    tol = alpha + 3 * np.sqrt(alpha * (1 - alpha) / len(pvals))
    ok = fpr <= max(tol, 2 * alpha)
    return Gate("G3 sham comparison", ok,
                f"false-positive rate {fpr:.3f} over {len(pvals)} sham splits "
                f"(alpha={alpha}, tolerance={max(tol, 2*alpha):.3f})",
                {"fpr": fpr, "n_sham": len(pvals), "tolerance": float(tol)})


def gate_separability(X: np.ndarray, system_labels, *,
                      alpha: float = 0.05, n_permutations: int = 2000,
                      rng: np.random.Generator | None = None) -> Gate:
    """G4: real systems separate under replicate-held-out evaluation."""
    X = np.asarray(X, dtype=np.float64)
    vc = _stats.variance_components(X, system_labels,
                                    n_permutations=n_permutations, rng=rng)
    disc = _stats.leave_one_replicate_out(X, system_labels,
                                          n_permutations=n_permutations, rng=rng)
    ok = bool(vc.p_value is not None and vc.p_value < alpha
              and disc.accuracy > disc.chance and disc.p_value < alpha)
    return Gate("G4 separability", ok,
                f"{vc} | {disc}",
                {"icc": vc.icc, "f": vc.f_stat, "p_variance": vc.p_value,
                 "loro_accuracy": disc.accuracy, "loro_chance": disc.chance,
                 "p_loro": disc.p_value, "per_system": disc.per_system})


def gate_effect_size(X: np.ndarray, system_labels, pair: tuple, *,
                     min_ratio: float = 1.0) -> Gate:
    """G5: separation is large relative to replicate scatter, with power."""
    a, b = pair
    ratio = _stats.separation_ratio(X, system_labels, a, b)
    need = _stats.replicates_required(ratio)
    have = int(min((np.asarray(system_labels) == a).sum(),
                   (np.asarray(system_labels) == b).sum()))
    ok = bool(ratio >= min_ratio and have >= need)
    return Gate("G5 effect size", ok,
                f"{a} vs {b}: separation={ratio:.2f} replicate-sd "
                f"(need >= {min_ratio}); replicates have={have} need={need}",
                {"separation_ratio": ratio, "replicates_needed": need,
                 "replicates_have": have})


def gate_beats_baseline(acc_model: float, acc_baseline: float, *,
                        margin: float = 0.0) -> Gate:
    """G6: the learned encoder earns its complexity."""
    ok = acc_model >= acc_baseline + margin
    return Gate("G6 beats baseline", ok,
                f"model LORO={acc_model:.3f} vs PCA baseline={acc_baseline:.3f}",
                {"model": acc_model, "baseline": acc_baseline}, fatal=False)


def gate_temperature(X: np.ndarray, temperature_labels, *,
                     min_r2: float = 0.3,
                     min_spearman: float = 0.5) -> Gate:
    """G7: the representation tracks temperature monotonically.

    Checks two things: that a linear thermal axis explains a reasonable share
    of variance (needed for `metrics.ThermalAxis` to be invertible), and that
    the leading thermal coordinate is monotone in temperature (needed for the
    apparent-temperature reading to be interpretable rather than folded).
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(temperature_labels, dtype=np.float64)
    levels = np.unique(T)
    if levels.size < 3:
        return Gate("G7 temperature", False,
                    f"only {levels.size} temperature level(s); need >= 3 to fit "
                    "a thermal axis", {"n_levels": int(levels.size)})

    A = np.stack([np.ones_like(T), T], axis=1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    resid = X - A @ coef
    ss_tot = float(((X - X.mean(0)) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0

    b = coef[1]
    proj = (X - coef[0]) @ b / (b @ b) if (b @ b) > 0 else np.zeros(len(T))
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(T, proj).statistic)
    except Exception:                       # pragma: no cover
        rho = float(np.corrcoef(T, proj)[0, 1])

    ok = bool(r2 >= min_r2 and abs(rho) >= min_spearman)
    return Gate("G7 temperature", ok,
                f"thermal-axis R2={r2:.3f} (need >= {min_r2}), "
                f"Spearman(T, axis)={rho:+.3f} (need |rho| >= {min_spearman}), "
                f"{levels.size} levels",
                {"r2": r2, "spearman": rho, "n_levels": int(levels.size)})


def gate_disentangled(X: np.ndarray, system_labels, temperature_labels, *,
                      n_permutations: int = 1000,
                      rng: np.random.Generator | None = None) -> Gate:
    """G8: the system signal survives conditioning on temperature.

    A representation that merely reads off "how fast is this moving" will
    separate temperatures beautifully and may separate systems only because the
    systems were run at different temperatures. The permutation here shuffles
    system labels *within* each temperature, so a significant result cannot be
    explained by temperature.
    """
    res = _stats.two_way_components(np.asarray(X, dtype=np.float64),
                                    system_labels, temperature_labels,
                                    n_permutations=n_permutations, rng=rng)
    p = res["p_system"]
    ok = bool(p is not None and p < 0.05)
    return Gate("G8 disentangled", ok,
                f"system effect conditioned on temperature: "
                f"ICC_sys={res['icc_system']:.3f} p={p:.4g}; "
                f"ICC_temp={res['icc_temperature']:.3f} p={res['p_temperature']:.4g}",
                res)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_gates(X: np.ndarray, system_labels, replicate_labels,
              temperature_labels=None, *, corpus: Corpus | None = None,
              pair: tuple | None = None, baseline_accuracy: float | None = None,
              n_permutations: int = 1000, sham_trials: int = 10,
              rng: np.random.Generator | None = None,
              stop_on_failure: bool = True) -> GateReport:
    """Run the gate suite in order.

    Args:
        X: ``(n_runs, d)`` run-level embeddings or descriptors.
        system_labels, replicate_labels: ``(n_runs,)`` provenance.
        temperature_labels: ``(n_runs,)``; enables G7 and G8.
        corpus: Enables G0.
        pair: ``(system_a, system_b)`` for the effect-size gate.
        baseline_accuracy: PCA baseline LORO accuracy, enables G6.
        stop_on_failure: Stop after the first fatal failure.
    """
    rng = rng or np.random.default_rng(0)
    report = GateReport()

    def _add(g: Gate) -> bool:
        report.gates.append(g)
        _logger.info("%s", g)
        return g.passed or not g.fatal

    if corpus is not None and not _add(gate_corpus(corpus)) and stop_on_failure:
        return report
    if not _add(gate_features(X)) and stop_on_failure:
        return report
    _add(gate_replicate_noise(X, system_labels, replicate_labels))
    if not _add(gate_sham(X, system_labels, replicate_labels,
                          n_trials=sham_trials, n_permutations=n_permutations,
                          rng=rng)) and stop_on_failure:
        return report
    if not _add(gate_separability(X, system_labels,
                                  n_permutations=n_permutations, rng=rng)) \
            and stop_on_failure:
        return report
    if pair is not None and not _add(gate_effect_size(X, system_labels, pair)) \
            and stop_on_failure:
        return report
    if baseline_accuracy is not None:
        disc = _stats.leave_one_replicate_out(X, system_labels, n_permutations=0)
        _add(gate_beats_baseline(disc.accuracy, baseline_accuracy))
    if temperature_labels is not None:
        _add(gate_temperature(X, temperature_labels))
        _add(gate_disentangled(X, system_labels, temperature_labels,
                               n_permutations=n_permutations, rng=rng))
    return report
