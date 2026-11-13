"""Tests for `sawnergy.md.states` -- shared conformational state populations.

The properties worth testing here are the ones that make the descriptor valid
rather than merely runnable: that the clustering is shared and reusable on runs
it never saw, that populations are a probability vector, that order is
discarded, and that the joint structure a marginal mean destroys actually
survives.
"""

import numpy as np
import pytest

from sawnergy.md.states import (MarkovStates, TRANSITION_PRIOR, kmeans,
                                multinomial_noise)


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------

def _three_blobs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    cen = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return np.concatenate([c + 0.2 * rng.standard_normal((n, 2)) for c in cen])


def test_kmeans_recovers_well_separated_blobs():
    X = _three_blobs()
    centres = kmeans(X, 3, seed=0)
    assert centres.shape == (3, 2)
    truth = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    # every true centre is matched by some fitted centre
    for t in truth:
        assert np.min(np.linalg.norm(centres - t, axis=1)) < 0.5


def test_kmeans_is_deterministic_for_a_fixed_seed():
    X = _three_blobs()
    assert np.allclose(kmeans(X, 4, seed=3), kmeans(X, 4, seed=3))


def test_kmeans_orders_centres_by_descending_population():
    rng = np.random.default_rng(1)
    X = np.concatenate([
        np.zeros((50, 2)) + 0.01 * rng.standard_normal((50, 2)),
        np.array([20.0, 0.0]) + 0.01 * rng.standard_normal((5, 2)),
    ])
    centres = kmeans(X, 2, seed=0)
    # the 50-frame blob must come first
    assert np.linalg.norm(centres[0]) < np.linalg.norm(centres[1])


def test_kmeans_returns_k_centres_even_with_fewer_distinct_points():
    # k empty clusters must be re-seeded, not dropped: a feature width that
    # depends on the data is not a feature space
    X = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0]]), 10, axis=0)
    assert kmeans(X, 5, seed=0).shape == (5, 2)


def test_kmeans_rejects_k_larger_than_n():
    with pytest.raises(ValueError, match="k must be in"):
        kmeans(np.zeros((3, 2)), 4)


def test_kmeans_rejects_non_matrix_input():
    with pytest.raises(ValueError, match=r"expected \(N, P\)"):
        kmeans(np.zeros(5), 2)


# ---------------------------------------------------------------------------
# MarkovStates: fitting
# ---------------------------------------------------------------------------

def _runs(n_runs=6, T=40, seed=0):
    rng = np.random.default_rng(seed)
    cen = np.array([[0.0, 0.0], [6.0, 0.0]])
    out = []
    for _ in range(n_runs):
        pick = rng.integers(0, 2, size=T)
        out.append(cen[pick] + 0.3 * rng.standard_normal((T, 2)))
    return out


def test_fit_pools_frames_and_records_the_count():
    runs = _runs(n_runs=5, T=40)
    ms = MarkovStates.fit(runs, k=2, seed=0)
    assert ms.k == 2
    assert ms.n_frames_fit == 200
    assert ms.bandwidth > 0


def test_fit_rejects_empty_and_ragged_input():
    with pytest.raises(ValueError, match="no runs supplied"):
        MarkovStates.fit([], k=2)
    with pytest.raises(ValueError, match="same feature width"):
        MarkovStates.fit([np.zeros((5, 2)), np.zeros((5, 3))], k=2)


def test_states_fitted_on_training_runs_score_unseen_runs():
    """The open-set property: the whole point of a shared clustering."""
    train, held_out = _runs(n_runs=5, seed=0), _runs(n_runs=2, seed=99)
    ms = MarkovStates.fit(train, k=3, seed=0)
    for r in held_out:
        p = ms.populations(r)
        assert p.shape == (3,)
        assert np.isclose(p.sum(), 1.0)


# ---------------------------------------------------------------------------
# memberships and populations
# ---------------------------------------------------------------------------

def test_hard_memberships_are_one_hot():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=3, seed=0)
    M = ms.memberships(runs[0], soft=False)
    assert M.shape == (len(runs[0]), 3)
    assert set(np.unique(M)) <= {0.0, 1.0}
    assert np.allclose(M.sum(axis=1), 1.0)


def test_soft_memberships_are_normalised_and_not_one_hot():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=3, seed=0)
    M = ms.memberships(runs[0], soft=True)
    assert np.allclose(M.sum(axis=1), 1.0)
    assert np.isfinite(M).all() and (M >= 0).all()
    # genuinely soft: at least some frames are shared between states. Frames
    # sitting deep inside one basin may still underflow to a hard assignment,
    # which is correct behaviour, so this is asserted over the run and not
    # frame by frame.
    assert M.max(axis=1).min() < 0.99
    assert not np.allclose(M, ms.memberships(runs[0], soft=False))


def test_hard_memberships_agree_with_assign():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=4, seed=0)
    M = ms.memberships(runs[0], soft=False)
    assert np.array_equal(np.argmax(M, axis=1), ms.assign(runs[0]))


def test_populations_are_a_probability_vector():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=4, seed=0)
    for soft in (False, True):
        p = ms.populations(runs[0], soft=soft)
        assert p.shape == (4,)
        assert (p >= 0).all()
        assert np.isclose(p.sum(), 1.0)


def test_populations_are_invariant_to_frame_order():
    """Equilibrium frames are an unordered sample; shuffling must change
    nothing. This is the property that makes the descriptor seed-independent."""
    runs = _runs()
    ms = MarkovStates.fit(runs, k=4, seed=0)
    r = runs[0]
    shuffled = r[np.random.default_rng(7).permutation(len(r))]
    assert np.allclose(ms.populations(r), ms.populations(shuffled))


def test_hellinger_is_the_square_root_and_preserves_ordering():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=4, seed=0)
    p = ms.populations(runs[0])
    h = ms.populations(runs[0], hellinger=True)
    assert np.allclose(h, np.sqrt(p))
    assert np.array_equal(np.argsort(p), np.argsort(h))


def test_soft_populations_have_lower_variance_across_replicates():
    """The reason `soft=True` is the default: at fixed T it is a lower-variance
    estimator of the same quantity, and that variance is the denominator of
    every separation ratio in this package."""
    runs = _runs(n_runs=12, T=40, seed=5)
    ms = MarkovStates.fit(runs, k=6, seed=0)
    hard = np.stack([ms.populations(r, soft=False) for r in runs])
    soft = np.stack([ms.populations(r, soft=True) for r in runs])
    assert soft.std(axis=0).mean() < hard.std(axis=0).mean()


def test_populations_recover_a_known_occupancy_split():
    rng = np.random.default_rng(0)
    cen = np.array([[0.0, 0.0], [8.0, 0.0]])
    train = [cen[rng.integers(0, 2, 200)] + 0.1 * rng.standard_normal((200, 2))
             for _ in range(3)]
    ms = MarkovStates.fit(train, k=2, seed=0)
    # a run that is 75% in one basin
    pick = np.array([0] * 150 + [1] * 50)
    run = cen[pick] + 0.1 * rng.standard_normal((200, 2))
    p = ms.populations(run, soft=False)
    assert np.isclose(max(p), 0.75, atol=0.02)


# ---------------------------------------------------------------------------
# the thing marginals cannot do
# ---------------------------------------------------------------------------

def test_state_populations_separate_ensembles_that_share_every_marginal():
    """The motivating case from the module docstring.

    Two features, each formed half the time in both systems. System A visits
    (on, on) and (off, off); system B visits (on, off) and (off, on). Every
    marginal mean is 0.5 for both. The joint distributions are disjoint.
    """
    rng = np.random.default_rng(0)

    def make(system, n_runs, T=80):
        out = []
        for _ in range(n_runs):
            # exactly balanced, so both marginals are 0.5 by construction and
            # the test is about the joint structure rather than sampling luck
            pick = rng.permutation(np.arange(T) % 2)
            if system == "A":
                mu = np.where(pick[:, None] == 0, [0.0, 0.0], [1.0, 1.0])
            else:
                mu = np.where(pick[:, None] == 0, [1.0, 0.0], [0.0, 1.0])
            out.append(mu + 0.05 * rng.standard_normal((T, 2)))
        return out

    A, B = make("A", 6), make("B", 6)

    # marginals are indistinguishable
    mA = np.stack([r.mean(axis=0) for r in A])
    mB = np.stack([r.mean(axis=0) for r in B])
    assert np.abs(mA.mean(axis=0) - mB.mean(axis=0)).max() < 0.05

    # populations over shared states are not
    ms = MarkovStates.fit(A[:3] + B[:3], k=4, seed=0)
    pA = np.stack([ms.populations(r, soft=False) for r in A[3:]])
    pB = np.stack([ms.populations(r, soft=False) for r in B[3:]])
    assert np.abs(pA.mean(axis=0) - pB.mean(axis=0)).max() > 0.4


# ---------------------------------------------------------------------------
# transitions
# ---------------------------------------------------------------------------

def test_transition_matrix_rows_are_normalised():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=3, seed=0)
    T = ms.transition_matrix(runs[0])
    assert T.shape == (3, 3)
    assert np.allclose(T.sum(axis=1), 1.0)


def test_transition_matrix_recovers_a_known_two_state_rate():
    """A run that alternates every frame has off-diagonal probability 1."""
    cen = np.array([[0.0, 0.0], [9.0, 0.0]])
    train = [cen[np.arange(60) % 2] for _ in range(3)]
    ms = MarkovStates.fit(train, k=2, seed=0)
    T = ms.transition_matrix(cen[np.arange(60) % 2], lag=1)
    assert T[0, 1] > 0.99 and T[1, 0] > 0.99
    assert T[0, 0] < 0.01


def test_transition_matrix_of_a_frozen_run_is_the_identity():
    cen = np.array([[0.0, 0.0], [9.0, 0.0]])
    train = [cen[np.arange(60) % 2] for _ in range(3)]
    ms = MarkovStates.fit(train, k=2, seed=0)
    T = ms.transition_matrix(np.zeros((40, 2)), lag=1)
    assert T[0, 0] > 0.99


def test_transition_matrix_is_order_dependent_unlike_populations():
    """Documenting the asymmetry: this descriptor reads the time axis, which is
    why `populations` is the one recommended for equilibrium ensembles."""
    runs = _runs()
    ms = MarkovStates.fit(runs, k=3, seed=0)
    r = runs[0]
    shuffled = r[np.random.default_rng(3).permutation(len(r))]
    assert not np.allclose(ms.transition_matrix(r),
                           ms.transition_matrix(shuffled))


def test_transition_matrix_rejects_an_out_of_range_lag():
    runs = _runs()
    ms = MarkovStates.fit(runs, k=2, seed=0)
    with pytest.raises(ValueError, match="lag must be in"):
        ms.transition_matrix(runs[0], lag=len(runs[0]))


def test_transition_prior_keeps_unvisited_rows_finite():
    cen = np.array([[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]])
    train = [cen[np.arange(60) % 3] for _ in range(3)]
    ms = MarkovStates.fit(train, k=3, seed=0)
    T = ms.transition_matrix(np.zeros((30, 2)))  # never visits two states
    assert np.isfinite(T).all()
    assert np.allclose(T.sum(axis=1), 1.0)
    assert TRANSITION_PRIOR > 0


# ---------------------------------------------------------------------------
# the noise floor
# ---------------------------------------------------------------------------

def test_multinomial_noise_matches_the_closed_form():
    p = np.array([0.5, 0.25, 0.25])
    assert np.allclose(multinomial_noise(p, 100), np.sqrt(p * (1 - p) / 100))


def test_multinomial_noise_is_the_order_of_the_populations_at_this_panels_budget():
    """The honest warning in the module docstring, as an assertion: with 90
    frames and 50 states the per-coordinate noise is within a factor of two of
    the coordinate itself."""
    p = np.full(50, 1 / 50)
    se = multinomial_noise(p, 90)
    assert se.mean() > 0.5 * p.mean()


def test_multinomial_noise_rejects_a_nonpositive_frame_count():
    with pytest.raises(ValueError, match="n_frames must be positive"):
        multinomial_noise(np.array([0.5, 0.5]), 0)
