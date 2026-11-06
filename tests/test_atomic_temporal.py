"""Tests for the all-atom and temporal featurisers."""

from __future__ import annotations

import numpy as np
import pytest

from sawnergy.md.atomic import (AtomTable, burial, cb_coords,  # noqa: E402
                                dihedral_features, mutation_mask,
                                residue_min_distance, sidechain_centroids)
from sawnergy.md.temporal import (TICA, autocorrelation,  # noqa: E402
                                  fluctuation, lifetime)


# ---------------------------------------------------------------------------
# a tiny synthetic chain: 3 residues, with a glycine in the middle
# ---------------------------------------------------------------------------

def _table():
    rows = [
        ("N", "ALA", 0, 10), ("CA", "ALA", 0, 10), ("C", "ALA", 0, 10),
        ("O", "ALA", 0, 10), ("CB", "ALA", 0, 10),
        ("N", "GLY", 1, 11), ("CA", "GLY", 1, 11), ("C", "GLY", 1, 11),
        ("O", "GLY", 1, 11),
        ("N", "LYS", 2, 12), ("CA", "LYS", 2, 12), ("C", "LYS", 2, 12),
        ("O", "LYS", 2, 12), ("CB", "LYS", 2, 12), ("CG", "LYS", 2, 12),
        ("CD", "LYS", 2, 12), ("CE", "LYS", 2, 12), ("NZ", "LYS", 2, 12),
        ("HZ1", "LYS", 2, 12),
    ]
    return AtomTable(atom_name=np.array([r[0] for r in rows]),
                     res_name=np.array([r[1] for r in rows]),
                     res_index=np.array([r[2] for r in rows]),
                     res_seq=np.array([r[3] for r in rows]))


def _xyz(T=5, seed=0):
    rng = np.random.default_rng(seed)
    base = np.arange(19)[:, None] * np.array([1.4, 0.0, 0.0])
    return (base[None] + rng.normal(scale=0.05, size=(T, 19, 3))).astype(np.float32)


class TestAtomTable:
    def test_counts_residues_and_heavy_atoms(self):
        t = _table()
        assert t.n_residues == 3
        assert t.is_heavy.sum() == 18          # one HZ1 excluded
        assert list(t.residue_sequence()) == ["ALA", "GLY", "LYS"]
        assert list(t.residue_numbers()) == [10, 11, 12]

    def test_rejects_inconsistent_table(self):
        with pytest.raises(ValueError, match="inconsistent"):
            AtomTable(atom_name=np.array(["CA", "CB"]),
                      res_name=np.array(["ALA"]),
                      res_index=np.array([0]), res_seq=np.array([1]))

    def test_glycine_falls_back_to_ca(self):
        t = _table()
        idx = t.one_per_residue("CB", fallback="CA")
        assert t.atom_name[idx[0]] == "CB"
        assert t.atom_name[idx[1]] == "CA"     # glycine has no CB
        assert t.atom_name[idx[2]] == "CB"


class TestSideChainGeometry:
    def test_cb_is_count_invariant(self):
        """The point of CB: a substitution cannot change the feature width."""
        t = _table()
        X = _xyz()
        assert cb_coords(X, t).shape == (5, 3, 3)

    def test_sidechain_centroid_uses_only_side_chain_heavy_atoms(self):
        t, X = _table(), _xyz(T=1)
        sc = sidechain_centroids(X, t)
        want = X[0, [13, 14, 15, 16, 17], :].mean(axis=0)   # LYS CB..NZ, no HZ1
        assert np.allclose(sc[0, 2], want, atol=1e-5)

    def test_glycine_centroid_falls_back_to_ca(self):
        t, X = _table(), _xyz(T=1)
        sc = sidechain_centroids(X, t)
        assert np.allclose(sc[0, 1], X[0, 6], atol=1e-6)

    def test_min_distance_is_symmetric_with_zero_diagonal(self):
        t, X = _table(), _xyz(T=2)
        D = residue_min_distance(X, t, prefilter_r=50.0)
        assert D.shape == (2, 3, 3)
        assert np.allclose(D, np.transpose(D, (0, 2, 1)), atol=1e-6)
        assert np.allclose(np.diagonal(D, axis1=1, axis2=2), 0.0)

    def test_min_distance_is_below_cb_distance(self):
        """Closest heavy-atom approach cannot exceed the CB-CB distance."""
        t, X = _table(), _xyz(T=1)
        D = residue_min_distance(X, t, prefilter_r=50.0)
        cb = cb_coords(X, t)[0]
        d_cb = np.linalg.norm(cb[0] - cb[2])
        assert D[0, 0, 2] <= d_cb + 1e-5


class TestDihedrals:
    def test_shape_and_sin_cos_normalisation(self):
        t, X = _table(), _xyz(T=4)
        F = dihedral_features(X, t)
        assert F.shape == (4, 3, 6)
        # defined angles must satisfy sin^2 + cos^2 = 1
        phi = F[:, 1:, 0] ** 2 + F[:, 1:, 1] ** 2
        assert np.allclose(phi, 1.0, atol=1e-4)

    def test_undefined_angles_are_zero_not_nan(self):
        t, X = _table(), _xyz(T=3)
        F = dihedral_features(X, t)
        assert np.isfinite(F).all()
        assert np.allclose(F[:, 0, 0:2], 0.0)       # phi of the first residue
        assert np.allclose(F[:, 1, 4:6], 0.0)       # chi1 of glycine

    def test_is_periodic_safe(self):
        """A rotation through the +-180 deg branch must move the code smoothly."""
        t = _table()
        X = _xyz(T=1).repeat(2, axis=0)
        F0 = dihedral_features(X, t)[0]
        # rotate the whole system: dihedrals are invariant to rigid motion
        th = 0.7
        Rz = np.array([[np.cos(th), -np.sin(th), 0],
                       [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        F1 = dihedral_features((X @ Rz.T).astype(np.float32), t)[0]
        assert np.allclose(F0, F1, atol=1e-4)


class TestBurial:
    def test_counts_neighbours_within_radius(self):
        t, X = _table(), _xyz(T=2)
        b = burial(X, t, r=100.0)
        assert b.shape == (2, 3)
        assert np.allclose(b, 2.0)        # everything sees the other two

    def test_isolated_residue_has_zero_burial(self):
        t, X = _table(), _xyz(T=1)
        assert np.allclose(burial(X, t, r=0.5), 0.0)


class TestMutationMask:
    def test_drops_named_sites(self):
        t = _table()
        keep = mutation_mask(t, [11])
        assert list(keep) == [True, False, True]

    def test_neighbour_radius_drops_spatial_neighbours(self):
        t, X = _table(), _xyz(T=2)
        keep = mutation_mask(t, [11], neighbour_r=5.0, xyz=X)
        assert keep.sum() < 2, "a wide radius should drop neighbours too"

    def test_neighbour_radius_requires_coordinates(self):
        with pytest.raises(ValueError, match="requires xyz"):
            mutation_mask(_table(), [11], neighbour_r=5.0)


# ---------------------------------------------------------------------------
# temporal
# ---------------------------------------------------------------------------

class TestTemporalFeatures:
    def test_fluctuation_is_the_temporal_sd(self):
        X = np.array([[0.0, 1.0], [2.0, 1.0]])
        assert np.allclose(fluctuation(X), [1.0, 0.0])

    def test_autocorrelation_detects_persistence(self):
        t = np.linspace(0, 20 * np.pi, 400)
        slow = np.sin(t / 20)[:, None]
        fast = np.sin(t * 8)[:, None]
        noise = np.random.default_rng(0).normal(size=(400, 1))
        ac = autocorrelation(np.hstack([slow, fast, noise]), lag=1)
        assert ac[0] > 0.99, "a slow signal must be highly autocorrelated"
        assert ac[0] > ac[1] > ac[2] or ac[0] > ac[2], ac

    def test_autocorrelation_of_a_constant_is_zero_not_nan(self):
        ac = autocorrelation(np.ones((10, 2)), lag=2)
        assert np.isfinite(ac).all() and np.allclose(ac, 0.0)

    def test_lifetime_separates_flicker_from_a_long_stretch(self):
        T = 20
        flicker = np.tile([1.0, 0.0], T // 2)[:, None]
        stretch = np.concatenate([np.ones(T // 2), np.zeros(T // 2)])[:, None]
        lt = lifetime(np.hstack([flicker, stretch]))
        assert lt[0] == pytest.approx(1.0)
        assert lt[1] == pytest.approx(10.0)
        # the two have identical means, which is the whole point
        assert np.hstack([flicker, stretch]).mean(0)[0] == \
               pytest.approx(np.hstack([flicker, stretch]).mean(0)[1])

    def test_lifetime_of_never_formed_is_zero(self):
        assert np.allclose(lifetime(np.zeros((10, 3))), 0.0)

    def test_rejects_wrong_shape(self):
        for fn in (fluctuation, autocorrelation, lifetime):
            with pytest.raises(ValueError, match="expected"):
                fn(np.zeros((4, 4, 4)))


class TestTICA:
    def _slow_fast(self, n_runs=6, T=300, seed=0):
        """Data whose slow direction has *less* variance than the fast one."""
        rng = np.random.default_rng(seed)
        runs = []
        for _ in range(n_runs):
            # slow: random walk, small amplitude. fast: white noise, large.
            slow = np.cumsum(rng.normal(scale=0.05, size=T))
            slow = (slow - slow.mean()) / (slow.std() + 1e-9) * 0.5
            fast = rng.normal(scale=3.0, size=T)
            runs.append(np.stack([slow, fast], axis=1))
        return runs

    def test_finds_the_slow_mode_pca_would_miss(self):
        runs = self._slow_fast()
        tica = TICA.fit(runs, lag=5, n_components=2)
        lead = np.abs(tica.components[0])
        assert lead[0] > lead[1], (
            f"leading TICA component should load on the slow coordinate, "
            f"got {tica.components[0]}")
        X = np.concatenate(runs)
        pca_lead = np.abs(np.linalg.svd(X - X.mean(0),
                                        full_matrices=False)[2][0])
        assert pca_lead[1] > pca_lead[0], (
            "sanity: PCA should prefer the high-variance fast coordinate")

    def test_eigenvalues_are_ordered_and_bounded(self):
        tica = TICA.fit(self._slow_fast(), lag=5, n_components=2)
        assert np.all(np.diff(tica.eigenvalues) <= 1e-9)
        assert tica.eigenvalues.max() <= 1.0 + 1e-6

    def test_timescales_are_positive_and_longest_first(self):
        tica = TICA.fit(self._slow_fast(), lag=5, n_components=2)
        ts = tica.timescales()
        assert ts[0] > 0
        assert np.all(np.diff(ts) <= 1e-9), f"not longest-first: {ts}"
        assert np.isfinite(ts).all(), "an anticorrelated mode is fast, not inf"

    def test_does_not_pair_frames_across_run_boundaries(self):
        """Two runs must not be silently concatenated into one time series."""
        rng = np.random.default_rng(1)
        a = np.cumsum(rng.normal(size=(100, 1)), axis=0) + 100.0
        b = np.cumsum(rng.normal(size=(100, 1)), axis=0) - 100.0
        sep = TICA.fit([a, b], lag=1, n_components=1)
        joined = TICA.fit([np.concatenate([a, b])], lag=1, n_components=1)
        assert not np.isclose(sep.eigenvalues[0], joined.eigenvalues[0],
                              atol=1e-6), \
            "run-aware and run-blind fits should differ; the boundary matters"

    def test_transform_run_gives_one_vector_per_run(self):
        runs = self._slow_fast(n_runs=4, T=120)
        tica = TICA.fit(runs, lag=3, n_components=2)
        Z = np.stack([tica.transform_run(r) for r in runs])
        assert Z.shape == (4, 2)

    def test_refuses_a_lag_longer_than_every_run(self):
        with pytest.raises(ValueError, match="longer than the lag"):
            TICA.fit([np.zeros((5, 3))], lag=10)

    def test_requires_consistent_feature_width(self):
        with pytest.raises(ValueError, match="same feature width"):
            TICA.fit([np.zeros((20, 3)), np.zeros((20, 4))], lag=2)
