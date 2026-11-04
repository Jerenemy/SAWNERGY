"""Rescue metrics: distance to the wild-type dynamical state, and thermal shift.

Two metrics are provided.

**`rescue_score`** is the pipeline document's ``R_i = 1 - D_WT(z_i)/D_WT(z_Y220C)``.
It is dimensionless, bounded above by 1, and easy to read. Its weaknesses are
that it depends on an arbitrary latent scale, it is undefined if the untreated
mutant happens to sit close to WT, and it cannot be compared against any
experiment.

**`thermal_shift`** is the addition this module argues for. The Y220C defect is
*thermal*: the DNA-binding domain melts at ~41.6 C wild-type versus ~33.2 C for
Y220C, and body temperature (37 C) sits between the two. That is the whole
disease mechanism -- nothing is broken at the DNA interface, the domain simply
unfolds at the temperature it has to work at. So the natural question is not
"how far is this trajectory from WT in latent units" but **"what temperature
does this system behave as if it were at?"**

Fit the wild-type embedding as a function of temperature, then invert it: a
trajectory's apparent temperature ``T_app`` is the temperature at which WT
would look the way this system looks. Then

    dT_apparent = T_app - T_actual

is positive for a destabilised system (it behaves hotter than it is), is
measured in kelvin, and is directly comparable to the ``dTm`` that a
differential scanning fluorimetry experiment reports. A compound's rescue is
``dT_apparent(mutant) - dT_apparent(mutant + compound)``, again in kelvin.

This also yields a falsifiable prediction that doubles as a validation gate:
Y220C at temperature T should embed near WT at roughly ``T + 8.4 K``. A
representation that cannot reproduce that is not tracking stability, whatever
its classification accuracy.

**Caveat, stated plainly.** Spontaneous unfolding is not accessible to
conventional MD at 310 K on any affordable timescale. `ThermalAxis` is
therefore meant to be fitted over an elevated temperature ladder (the mdCATH
design -- 320/348/379/413/450 K -- exists for exactly this reason) and
extrapolated down, or driven by enhanced sampling. Fitting it on a single
temperature, or on two temperatures 33 K apart at 20 ns each, will produce a
number with no support; `ThermalAxis.fit` refuses fewer than three
temperatures and reports its own extrapolation distance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_logger = logging.getLogger(__name__)

#: Experimental melting temperatures for the p53 DNA-binding domain.
#: Used only as the expected magnitude in validation gate G5.
TM_WT_C = 41.6
TM_Y220C_C = 33.2
DELTA_TM_K = TM_WT_C - TM_Y220C_C          # ~8.4 K


# ---------------------------------------------------------------------------
# wild-type reference distribution
# ---------------------------------------------------------------------------

@dataclass
class WTReference:
    """The wild-type dynamical distribution in embedding space.

    Built from *multiple* WT runs, never one. The covariance is what makes the
    distance meaningful: directions in which wild type naturally varies a lot
    should not count as deviation, which is precisely the Mahalanobis
    correction the pipeline document lists as option C.
    """

    mean: np.ndarray
    cov: np.ndarray
    n_runs: int
    shrinkage: float = 0.0
    _inv: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def fit(cls, Z: np.ndarray, *, shrinkage: float | None = None) -> "WTReference":
        """Estimate the WT mean and covariance from run-level embeddings.

        Args:
            Z: ``(n_runs, d)`` one embedding per WT run.
            shrinkage: Ledoit-Wolf style shrinkage toward a scaled identity. If
                ``None``, chosen as ``d / (d + n_runs)``, which is heavy when
                runs are few -- as they always are here. Without shrinkage the
                sample covariance of 5 runs in 50 dimensions is singular and
                the Mahalanobis distance is pure noise amplification.
        """
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError(f"Z must be (n_runs, d); got {Z.shape}")
        n, d = Z.shape
        if n < 2:
            raise ValueError(
                "the wild-type reference needs >= 2 runs. A single WT trajectory "
                "gives no estimate of how much WT varies on its own, so every "
                "downstream distance would be uncalibrated.")
        if n < 5:
            _logger.warning("WT reference built from only %d runs; the covariance "
                            "estimate is weak and distances will be unstable", n)
        mu = Z.mean(axis=0)
        S = np.cov(Z, rowvar=False, ddof=1)
        S = np.atleast_2d(S)
        lam = d / (d + n) if shrinkage is None else float(shrinkage)
        lam = min(max(lam, 0.0), 1.0)
        target = np.eye(d) * (np.trace(S) / d if d else 1.0)
        S = (1 - lam) * S + lam * target
        return cls(mean=mu, cov=S, n_runs=n, shrinkage=lam)

    def _inv_cov(self) -> np.ndarray:
        if self._inv is None:
            self._inv = np.linalg.pinv(self.cov)
        return self._inv

    def distance(self, z: np.ndarray, *, metric: str = "mahalanobis") -> np.ndarray:
        """Distance of one or more embeddings from the WT distribution.

        Args:
            z: ``(d,)`` or ``(n, d)``.
            metric: ``"mahalanobis"`` (default), ``"euclidean"`` or ``"cosine"``.
        """
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        delta = z - self.mean
        if metric == "euclidean":
            out = np.linalg.norm(delta, axis=1)
        elif metric == "cosine":
            a = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12)
            b = self.mean / (np.linalg.norm(self.mean) + 1e-12)
            out = 1.0 - a @ b
        elif metric == "mahalanobis":
            Vi = self._inv_cov()
            out = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, Vi, delta), 0.0))
        else:
            raise ValueError(f"unknown metric {metric!r}")
        return out if out.size > 1 else out


def rescue_score(z: np.ndarray, wt: WTReference, d_baseline: float, *,
                 metric: str = "mahalanobis") -> np.ndarray:
    """``R = 1 - D_WT(z) / D_WT(baseline)``.

    Args:
        z: Embedding(s) of the treated system.
        wt: Wild-type reference.
        d_baseline: ``D_WT`` of the *untreated* mutant, averaged over its
            replicates. Using a single baseline trajectory here propagates that
            run's idiosyncrasy into every compound's score.

    Returns:
        ``R`` -- 0 is no rescue, 1 is full restoration, negative means the
        compound pushed the dynamics further from wild type.
    """
    if d_baseline <= 0:
        raise ValueError("baseline distance must be positive; if the untreated "
                         "mutant sits on top of WT there is no deficit to rescue "
                         "and the ratio is undefined")
    return 1.0 - wt.distance(z, metric=metric) / d_baseline


# ---------------------------------------------------------------------------
# thermal axis
# ---------------------------------------------------------------------------

@dataclass
class ThermalAxis:
    """Wild-type embedding as a function of temperature, and its inverse.

    Model: ``z(T) ~ a + b T``. Given an embedding, the least-squares estimate
    of the temperature that produced it is the projection onto ``b``:

        ``T_app(z) = (z - a) . b / (b . b)``

    Attributes:
        a, b: Intercept and slope vectors in embedding space.
        temperatures: Temperatures used for the fit.
        r2: Fraction of WT embedding variance explained by temperature.
        residual_sd: RMS residual, in embedding units.
        t_sd_K: Standard error of a single ``T_app`` estimate, in kelvin.
    """

    a: np.ndarray
    b: np.ndarray
    temperatures: np.ndarray
    r2: float
    residual_sd: float
    t_sd_K: float

    @classmethod
    def fit(cls, Z: np.ndarray, temperatures, *, min_levels: int = 3) -> "ThermalAxis":
        """Fit the WT temperature response.

        Args:
            Z: ``(n_runs, d)`` wild-type run embeddings.
            temperatures: ``(n_runs,)`` matching temperatures.
            min_levels: Minimum distinct temperatures. Below three the fit is
                not identifiable as a trend and cannot be extrapolated.

        Raises:
            ValueError: If too few temperature levels are present.
        """
        Z = np.asarray(Z, dtype=np.float64)
        T = np.asarray(temperatures, dtype=np.float64).ravel()
        levels = np.unique(T)
        if levels.size < min_levels:
            raise ValueError(
                f"ThermalAxis needs >= {min_levels} distinct temperatures, got "
                f"{levels.size} ({levels}). A two-point fit cannot distinguish a "
                "thermal trend from the difference between two conditions, and "
                "extrapolating it to body temperature is unsupportable.")
        if Z.shape[0] != T.size:
            raise ValueError("Z and temperatures disagree on n_runs")

        X = np.stack([np.ones_like(T), T], axis=1)        # (n, 2)
        coef, *_ = np.linalg.lstsq(X, Z, rcond=None)      # (2, d)
        a, b = coef[0], coef[1]

        pred = X @ coef
        resid = Z - pred
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((Z - Z.mean(0)) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_sd = float(np.sqrt(ss_res / max(Z.shape[0] - 2, 1)))

        bb = float(b @ b)
        if bb <= 0:
            raise ValueError("wild-type embedding does not vary with temperature; "
                             "the thermal axis is undefined")
        # error in T_app from residual scatter projected onto the axis
        t_sd = residual_sd / np.sqrt(bb)
        return cls(a=a, b=b, temperatures=levels, r2=r2,
                   residual_sd=residual_sd, t_sd_K=float(t_sd))

    def apparent_temperature(self, z: np.ndarray) -> np.ndarray:
        """Temperature at which wild type would look like ``z``."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return ((z - self.a) @ self.b) / float(self.b @ self.b)

    def extrapolation_K(self, t_app) -> np.ndarray:
        """How far outside the fitted range an apparent temperature falls.

        Zero inside the ladder. Report it: a ``dT_apparent`` derived by
        extrapolating 80 K below the lowest simulated temperature is a model
        artefact, not a measurement.
        """
        t = np.atleast_1d(np.asarray(t_app, dtype=np.float64))
        lo, hi = self.temperatures.min(), self.temperatures.max()
        return np.maximum(np.maximum(lo - t, t - hi), 0.0)


@dataclass
class ThermalShift:
    """Result of `thermal_shift`."""

    delta_T_K: float
    apparent_T_K: float
    actual_T_K: float
    sem_K: float
    extrapolation_K: float
    n_runs: int

    def __str__(self) -> str:
        flag = "" if self.extrapolation_K <= 0 else \
            f"  [EXTRAPOLATED {self.extrapolation_K:.1f} K beyond the ladder]"
        return (f"dT_apparent = {self.delta_T_K:+.2f} +/- {self.sem_K:.2f} K "
                f"(looks like {self.apparent_T_K:.1f} K at {self.actual_T_K:.1f} K)"
                f"{flag}")


def thermal_shift(Z: np.ndarray, axis: ThermalAxis, actual_T_K: float) -> ThermalShift:
    """Apparent destabilisation of a system, in kelvin.

    Args:
        Z: ``(n_runs, d)`` embeddings of the system's replicates at one
            temperature. Multiple runs are required: the standard error is what
            tells you whether a 2 K shift is real.
        axis: Wild-type thermal axis.
        actual_T_K: The temperature those runs were actually simulated at.

    Returns:
        `ThermalShift`. Positive ``delta_T_K`` means the system behaves as if it
        were hotter than it is, i.e. destabilised.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    if Z.shape[0] < 2:
        raise ValueError("thermal_shift needs >= 2 replicates to report an error "
                         "bar; a single run gives a point estimate with no way "
                         "to tell it from noise")
    t_app = axis.apparent_temperature(Z)
    mean_app = float(t_app.mean())
    sem = float(t_app.std(ddof=1) / np.sqrt(t_app.size))
    extrap = float(axis.extrapolation_K(mean_app)[0])
    return ThermalShift(delta_T_K=mean_app - actual_T_K, apparent_T_K=mean_app,
                        actual_T_K=float(actual_T_K), sem_K=sem,
                        extrapolation_K=extrap, n_runs=int(Z.shape[0]))


def thermal_rescue_K(treated: ThermalShift, baseline: ThermalShift) -> tuple[float, float]:
    """Rescue in kelvin: how much of the mutant's thermal deficit is recovered.

    Returns:
        ``(delta_delta_T, sem)`` -- positive means the compound restored
        stability. Comparable in magnitude and units to an experimental
        ``dTm`` from differential scanning fluorimetry, which is how this field
        actually measures Y220C rescue.
    """
    ddt = baseline.delta_T_K - treated.delta_T_K
    sem = float(np.hypot(baseline.sem_K, treated.sem_K))
    return float(ddt), sem
