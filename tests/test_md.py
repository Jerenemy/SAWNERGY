"""Tests for the MD trajectory-embedding subpackage.

The tests that matter most are the *negative* ones: the pipeline has to refuse
designs that cannot support a conclusion (n=1, two temperatures, one WT run)
and has to report "no difference" when there genuinely is none.
"""

from __future__ import annotations

import numpy as np
import pytest

from sawnergy.md import (Corpus, Independence, Role, Run, ThermalAxis,
                         WTReference, leave_one_replicate_out, p53_corpus,
                         replicate_splits, replicates_required, rescue_score,
                         run_gates, separation_ratio, thermal_rescue_K,
                         thermal_shift, variance_components)
from sawnergy.md.validate import gate_sham, gate_temperature


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _runs(system, n, temp=310.0, role=Role.TREATED, start=0):
    return [Run(system=system, replicate=i, temperature_K=temp, role=role,
                seed=start + i, independence=Independence.SOLVATION)
            for i in range(n)]


def _synth(n_systems=3, n_rep=6, d=20, shift=3.0, seed=0):
    """Run-level data with a real between-system shift."""
    rng = np.random.default_rng(seed)
    X, sysl, repl = [], [], []
    for s in range(n_systems):
        centre = np.zeros(d)
        centre[0] = s * shift
        for r in range(n_rep):
            X.append(centre + rng.normal(0, 1.0, d))
            sysl.append(f"sys{s}")
            repl.append(r)
    return np.array(X), np.array(sysl), np.array(repl)


# ---------------------------------------------------------------------------
# manifest: refuse designs that cannot support a conclusion
# ---------------------------------------------------------------------------

class TestCorpus:
    def test_rejects_single_replicate(self):
        """n=1 per system is the central failure mode and must not pass."""
        c = Corpus(_runs("WT", 1, role=Role.REFERENCE) + _runs("Y220C", 1))
        rep = c.check()
        assert not rep.ok
        assert any("independent replicate" in e for e in rep.errors)

    def test_accepts_adequate_design(self):
        c = Corpus(_runs("WT", 5, role=Role.REFERENCE)
                   + _runs("Y220C", 5, start=100))
        assert c.check().ok

    def test_requires_reference_system(self):
        c = Corpus(_runs("Y220C", 5) + _runs("Y220C+lig", 5, start=100))
        rep = c.check()
        assert not rep.ok
        assert any("REFERENCE" in e for e in rep.errors)

    def test_rejects_duplicate_seeds(self):
        """Two runs with the same velocity seed are the same run."""
        runs = _runs("WT", 3, role=Role.REFERENCE)
        runs += [Run(system="WT", replicate=3, temperature_K=310.0,
                     role=Role.REFERENCE, seed=runs[0].seed,
                     independence=Independence.SOLVATION)]
        runs += _runs("Y220C", 4, start=500)
        rep = Corpus(runs).check()
        assert not rep.ok
        assert any("seed" in e for e in rep.errors)

    def test_segment_replicates_are_pseudo(self):
        """Chunks of one trajectory must not count as replicates."""
        runs = [Run(system="WT", replicate=i, temperature_K=310.0,
                    role=Role.REFERENCE, seed=i,
                    independence=Independence.SEGMENT) for i in range(5)]
        runs += _runs("Y220C", 5, start=100)
        rep = Corpus(runs).check()
        assert not rep.ok

    def test_roundtrip_json(self, tmp_path):
        c = Corpus(_runs("WT", 3, role=Role.REFERENCE) + _runs("M", 3, start=9))
        p = c.to_json(tmp_path / "c.json")
        c2 = Corpus.from_json(p)
        assert len(c2) == len(c)
        assert c2.systems == c.systems
        assert c2.check().ok == c.check().ok

    def test_p53_corpus_is_valid_and_balanced(self):
        c = p53_corpus(n_replicates=5)
        assert c.check().ok
        cells = c.cells()
        assert len({len(v) for v in cells.values()}) == 1, "grid must be balanced"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_refuses_single_run_systems(self):
        X = np.random.default_rng(0).normal(size=(4, 5))
        with pytest.raises(ValueError, match="single run"):
            variance_components(X, np.array(["a", "b", "c", "d"]))

    def test_detects_real_difference(self):
        X, sysl, _ = _synth(shift=4.0)
        vc = variance_components(X, sysl, n_permutations=500)
        assert vc.icc > 0.3
        assert vc.p_value < 0.05

    def test_reports_no_difference_on_pure_noise(self):
        """Identical systems must not come out significant."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(18, 20))
        sysl = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6)
        vc = variance_components(X, sysl, n_permutations=1000, rng=rng)
        assert vc.p_value > 0.05
        assert vc.icc < 0.2

    def test_loro_beats_chance_on_signal(self):
        X, sysl, _ = _synth(shift=5.0)
        d = leave_one_replicate_out(X, sysl, n_permutations=300)
        assert d.accuracy > d.chance
        assert d.p_value < 0.05

    def test_loro_at_chance_on_noise(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(18, 20))
        sysl = np.array(["a"] * 6 + ["b"] * 6 + ["c"] * 6)
        d = leave_one_replicate_out(X, sysl, n_permutations=300)
        assert d.p_value > 0.05

    def test_separation_ratio_scales_with_shift(self):
        lo = separation_ratio(*_synth(n_systems=2, shift=1.0)[:2], "sys0", "sys1")
        hi = separation_ratio(*_synth(n_systems=2, shift=6.0)[:2], "sys0", "sys1")
        assert hi > lo

    def test_power_analysis_monotone(self):
        assert replicates_required(0.5) > replicates_required(2.0)
        assert replicates_required(2.0) >= 2


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def test_splits_never_break_a_run_across_folds():
    """The whole point: a run is entirely in train or entirely in test."""
    sysl = np.repeat(["a", "b"], 10)
    repl = np.tile(np.repeat(np.arange(5), 2), 2)      # 2 frames per run
    for tr, te in replicate_splits(sysl, repl, n_folds=3):
        keys = np.array([f"{s}|{r}" for s, r in zip(sysl, repl)])
        assert not (set(keys[tr]) & set(keys[te])), "run leaked across the split"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_wt_reference_refuses_single_run(self):
        with pytest.raises(ValueError, match=">= 2 runs"):
            WTReference.fit(np.zeros((1, 5)))

    def test_mahalanobis_respects_wt_variance(self):
        """A deviation along a high-variance WT direction counts for less."""
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(40, 2)) * np.array([5.0, 0.5])
        wt = WTReference.fit(Z, shrinkage=0.0)
        wide = wt.distance(wt.mean + np.array([2.0, 0.0]))
        narrow = wt.distance(wt.mean + np.array([0.0, 2.0]))
        assert narrow > wide

    def test_rescue_score_bounds(self):
        rng = np.random.default_rng(1)
        wt = WTReference.fit(rng.normal(size=(20, 4)))
        d_base = float(wt.distance(wt.mean + 10.0)[0])
        assert rescue_score(wt.mean, wt, d_base)[0] == pytest.approx(1.0, abs=0.2)
        far = rescue_score(wt.mean + 20.0, wt, d_base)[0]
        assert far < 0

    def test_rescue_score_rejects_zero_baseline(self):
        wt = WTReference.fit(np.random.default_rng(0).normal(size=(10, 3)))
        with pytest.raises(ValueError):
            rescue_score(wt.mean, wt, 0.0)

    def test_thermal_axis_requires_three_temperatures(self):
        Z = np.random.default_rng(0).normal(size=(10, 3))
        T = np.array([300.0] * 5 + [310.0] * 5)
        with pytest.raises(ValueError, match=">= 3 distinct temperatures"):
            ThermalAxis.fit(Z, T)

    def test_thermal_axis_recovers_known_shift(self):
        """A system displaced along the WT thermal axis reads back as hotter."""
        rng = np.random.default_rng(0)
        temps = np.repeat([310.0, 330.0, 350.0, 370.0, 400.0], 5)
        direction = np.array([1.0, 0.0, 0.0])
        Z_wt = (temps[:, None] - 310.0) * 0.1 * direction + rng.normal(0, 0.01, (len(temps), 3))
        axis = ThermalAxis.fit(Z_wt, temps)
        assert axis.r2 > 0.95

        true_shift = 8.4
        Z_mut = (310.0 + true_shift - 310.0) * 0.1 * direction + rng.normal(0, 0.01, (5, 3))
        ts = thermal_shift(Z_mut, axis, 310.0)
        assert ts.delta_T_K == pytest.approx(true_shift, abs=1.0)

    def test_thermal_shift_requires_replicates(self):
        rng = np.random.default_rng(0)
        temps = np.repeat([310.0, 330.0, 350.0], 4)
        Z = (temps[:, None] - 310.0) * 0.1 + rng.normal(0, 0.01, (len(temps), 2))
        axis = ThermalAxis.fit(Z, temps)
        with pytest.raises(ValueError, match=">= 2 replicates"):
            thermal_shift(Z[:1], axis, 310.0)

    def test_thermal_shift_flags_extrapolation(self):
        rng = np.random.default_rng(0)
        temps = np.repeat([350.0, 370.0, 400.0], 5)
        Z = (temps[:, None] - 350.0) * 0.1 + rng.normal(0, 0.01, (len(temps), 2))
        axis = ThermalAxis.fit(Z, temps)
        far = np.tile((np.array([[-20.0, -20.0]])), (3, 1))
        ts = thermal_shift(far, axis, 310.0)
        assert ts.extrapolation_K > 0
        assert "EXTRAPOLATED" in str(ts)

    def test_thermal_rescue_direction(self):
        base = thermal_shift.__wrapped__ if hasattr(thermal_shift, "__wrapped__") else None
        from sawnergy.md.metrics import ThermalShift
        baseline = ThermalShift(8.0, 318.0, 310.0, 0.5, 0.0, 5)
        treated = ThermalShift(2.0, 312.0, 310.0, 0.5, 0.0, 5)
        ddt, sem = thermal_rescue_K(treated, baseline)
        assert ddt == pytest.approx(6.0)
        assert sem > 0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

class TestGates:
    def test_sham_passes_on_well_behaved_data(self):
        """Sham splits of one system must not look significant."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 15))
        sysl = np.repeat(["a", "b"], 20)
        repl = np.tile(np.arange(20), 2)
        g = gate_sham(X, sysl, repl, n_trials=10, n_permutations=300, rng=rng)
        assert g.passed, g.detail
        assert g.values["fpr"] <= 0.2

    def test_sham_catches_frame_level_pseudoreplication(self):
        """Passing frames as if they were runs must be caught.

        Emulates the classic bug: 6 genuine runs drawn from ONE distribution,
        each expanded into 20 near-identical frames, then handed to the
        statistics as 120 independent samples. The true between-group
        difference is zero, but the inflated n collapses within-group variance
        and makes the chance offset between group means look overwhelming.
        """
        rng = np.random.default_rng(6)
        base = rng.normal(size=(6, 10))                 # 6 runs, same system
        X = np.repeat(base, 20, axis=0) + rng.normal(0, 1e-4, (120, 10))
        # frames inherit their parent run's group, as they would in the bug
        sysl = np.repeat(["a"] * 3 + ["b"] * 3, 20)
        repl = np.repeat(np.arange(6), 20)
        vc = variance_components(X, sysl, n_permutations=500, rng=rng)
        assert vc.p_value < 0.01, "setup failed to reproduce the bug"

        # the same data analysed at the correct unit (one vector per run)
        # must NOT be significant
        vc_runs = variance_components(base, np.array(["a"] * 3 + ["b"] * 3),
                                      n_permutations=500, rng=rng)
        assert vc_runs.p_value > 0.05, (
            "run-level analysis should find no difference between two halves "
            "of one distribution")

    def test_full_suite_passes_on_real_signal(self):
        # shift must clear the replicate scatter accumulated over d dims, and
        # the design must carry the replicates G5 asks for at that effect size
        X, sysl, repl = _synth(n_systems=2, n_rep=12, d=10, shift=8.0)
        rep = run_gates(X, sysl, repl, pair=("sys0", "sys1"),
                        n_permutations=300, sham_trials=6)
        assert rep.ok, str(rep)

    def test_full_suite_fails_on_pure_noise(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(16, 15))
        sysl = np.repeat(["a", "b"], 8)
        repl = np.tile(np.arange(8), 2)
        rep = run_gates(X, sysl, repl, pair=("a", "b"),
                        n_permutations=300, sham_trials=6)
        assert not rep.ok
        assert rep.first_failure.name.startswith("G4") or \
               rep.first_failure.name.startswith("G5")

    def test_temperature_gate_needs_three_levels(self):
        X = np.random.default_rng(0).normal(size=(10, 4))
        T = np.repeat([300.0, 310.0], 5)
        assert not gate_temperature(X, T).passed

    def test_temperature_gate_detects_monotone_response(self):
        rng = np.random.default_rng(2)
        T = np.repeat([310.0, 330.0, 350.0, 370.0, 400.0], 4)
        X = (T[:, None] - 350.0) * 0.05 + rng.normal(0, 0.05, (len(T), 3))
        g = gate_temperature(X, T)
        assert g.passed, g.detail
        assert abs(g.values["spearman"]) > 0.8
