"""Tests for `sawnergy.md.vampnet`.

The score is a mathematical object with known bounds and known invariances, so
most of this is property-based rather than a regression on trained weights:
a VAMP-2 score has a ceiling of ``k - 1``, is invariant to relabelling the
states, collapses to 1 when the assignment carries no dynamical information, and
must never be computed across a run boundary.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sawnergy.md.vampnet import (COLLAPSE_SCORE, COV_EPS, VampConfig,  # noqa
                                 build_vampnet, fit_vampnet, lagged_pairs,
                                 soft_populations, vamp2_score)


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------
# lagged pairs
# ---------------------------------------------------------------------------

def test_lagged_pairs_never_cross_a_run_boundary():
    runs = [np.arange(10).reshape(10, 1), 100 + np.arange(10).reshape(10, 1)]
    X0, Xt = lagged_pairs(runs, lag=2)
    assert X0.shape == Xt.shape == (16, 1)
    # within a run the offset is exactly the lag; a crossed boundary would
    # produce a difference of about 100
    assert np.all(Xt - X0 == 2)


def test_lagged_pairs_drops_runs_shorter_than_the_lag():
    runs = [np.zeros((3, 2)), np.ones((20, 2))]
    X0, _ = lagged_pairs(runs, lag=5)
    assert len(X0) == 15


def test_lagged_pairs_rejects_an_impossible_lag():
    with pytest.raises(ValueError, match="lag must be >= 1"):
        lagged_pairs([np.zeros((5, 1))], lag=0)
    with pytest.raises(ValueError, match="no run is longer than the lag"):
        lagged_pairs([np.zeros((4, 1))], lag=9)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def test_perfectly_predictable_two_state_dynamics_hits_the_ceiling():
    """An assignment that determines the next state resolves all of the
    dynamics a two-state partition can: the centred ceiling is k - 1 = 1."""
    chi0 = _t([[1.0, 0.0]] * 50 + [[0.0, 1.0]] * 50)
    chit = _t([[0.0, 1.0]] * 50 + [[1.0, 0.0]] * 50)
    assert float(vamp2_score(chi0, chit)) == pytest.approx(1.0, abs=1e-3)


def test_score_of_an_uninformative_assignment_is_near_zero():
    rng = np.random.default_rng(0)
    chi = rng.random((400, 3))
    chi /= chi.sum(1, keepdims=True)
    chi2 = rng.random((400, 3))
    chi2 /= chi2.sum(1, keepdims=True)
    # independent assignments: the whitened cross-covariance is noise
    assert float(vamp2_score(_t(chi), _t(chi2))) < 0.2


def test_score_is_invariant_to_permuting_the_state_labels():
    rng = np.random.default_rng(1)
    chi0 = rng.random((200, 4)); chi0 /= chi0.sum(1, keepdims=True)
    chit = np.roll(chi0, 1, axis=0)
    p = [2, 0, 3, 1]
    a = float(vamp2_score(_t(chi0), _t(chit)))
    b = float(vamp2_score(_t(chi0[:, p]), _t(chit[:, p])))
    assert a == pytest.approx(b, rel=1e-4)


def test_score_is_invariant_to_an_invertible_linear_reparametrisation():
    """VAMP-2 whitens both sides, so it measures the subspace, not the basis --
    which is why it can be compared across differently parametrised models."""
    rng = np.random.default_rng(2)
    chi0 = rng.random((300, 3)); chi0 /= chi0.sum(1, keepdims=True)
    chit = 0.7 * chi0 + 0.3 * np.roll(chi0, 1, axis=0)
    A = rng.standard_normal((3, 3))
    base = float(vamp2_score(_t(chi0), _t(chit)))
    mixed = float(vamp2_score(_t(chi0 @ A), _t(chit @ A)))
    assert base == pytest.approx(mixed, rel=1e-3)


def test_score_of_a_collapsed_assignment_is_zero_not_a_nan():
    """Every frame in one state: C00 is singular, and the ridge has to keep the
    gradient finite rather than letting it blow up."""
    chi = _t(np.tile([1.0, 0.0, 0.0], (100, 1)))
    s = float(vamp2_score(chi, chi))
    assert np.isfinite(s)
    assert s < 1e-3


def test_score_rises_with_the_resolved_dynamics():
    rng = np.random.default_rng(3)
    lab = (rng.random(600) < 0.5).astype(int)
    hard = np.eye(2)[lab]
    # a noisier version of the same assignment must score lower
    noisy = 0.5 * hard + 0.5 * 0.5
    nxt = np.roll(lab, -1)
    assert (float(vamp2_score(_t(hard), _t(np.eye(2)[nxt])))
            >= float(vamp2_score(_t(noisy), _t(np.eye(2)[nxt]))))


def test_score_rejects_mismatched_or_degenerate_input():
    with pytest.raises(ValueError, match="shapes differ"):
        vamp2_score(_t(np.zeros((4, 2))), _t(np.zeros((4, 3))))
    with pytest.raises(ValueError, match="at least two lagged pairs"):
        vamp2_score(_t(np.zeros((1, 2))), _t(np.zeros((1, 2))))


def test_cov_eps_is_positive():
    assert COV_EPS > 0


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

def test_build_vampnet_outputs_a_normalised_simplex():
    net = build_vampnet(7, VampConfig(n_states=5, hidden=(8,)))
    out = net(torch.randn(13, 7))
    assert out.shape == (13, 5)
    assert torch.allclose(out.sum(-1), torch.ones(13), atol=1e-5)
    assert (out >= 0).all()


def test_build_vampnet_is_deterministic_for_a_fixed_seed():
    cfg = VampConfig(n_states=4, hidden=(8,), seed=11)
    x = torch.randn(5, 6)
    a = build_vampnet(6, cfg).eval()(x)
    b = build_vampnet(6, cfg).eval()(x)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _two_basin_runs(n_runs=8, T=120, seed=0, flip=0.05):
    """Runs hopping between two basins with a nonlinear boundary.

    The observable is a ring: the basins sit at opposite points of a circle, so
    no single linear coordinate separates them cleanly. This is the regime where
    a VAMPnet is supposed to have something over TICA plus k-means.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_runs):
        s, states = 0, []
        for _ in range(T):
            if rng.random() < flip:
                s = 1 - s
            states.append(s)
        states = np.array(states)
        theta = np.where(states == 0, 0.0, np.pi) + 0.25 * rng.standard_normal(T)
        out.append(np.c_[np.cos(theta), np.sin(theta),
                         0.1 * rng.standard_normal(T)].astype(np.float32))
    return out, out


def test_fit_vampnet_approaches_the_analytic_vamp2_limit():
    """The score has a known ceiling for this process, so the test is against
    theory rather than against a guessed threshold.

    For a two-state chain with per-frame flip probability ``p``, the slow
    eigenvalue at lag ``tau`` is ``(1 - 2p)**tau``, and a centred two-state
    VAMP-2 score is that eigenvalue squared. At ``p = 0.05`` and ``tau = 3``
    that is ``0.9**6 = 0.531`` -- well below the ``k - 1 = 1`` ceiling, which is
    reachable only by deterministic dynamics. A fit that lands near 0.53 has
    extracted everything there is; one that lands near 0 has collapsed.
    """
    p, lag = 0.05, 3
    limit = (1 - 2 * p) ** (2 * lag)
    runs, _ = _two_basin_runs(n_runs=10, T=150, seed=0, flip=p)
    cfg = VampConfig(n_states=2, hidden=(16, 16), lag=lag, epochs=60,
                     patience=20, dropout=0.0, seed=0, device="cpu")
    model, std, report = fit_vampnet(runs, cfg)
    assert report.max_score == 1.0
    assert 0.0 <= report.val_score <= report.max_score + 1e-6
    # within reach of the analytic limit, and not past it by more than noise
    assert report.train_score > 0.75 * limit
    assert report.train_score < limit + 0.1
    assert 0 <= report.best_epoch < report.epochs_run
    assert len(report.history) == report.epochs_run
    assert "VAMP-2" in str(report)


def test_fit_vampnet_scores_a_memoryless_process_near_zero():
    """The companion to the limit test: when there is no slow process to find,
    the score must report that rather than fitting noise. At ``p = 0.45`` the
    analytic limit is ``0.1**6``, i.e. nothing."""
    runs, _ = _two_basin_runs(n_runs=10, T=150, seed=1, flip=0.45)
    cfg = VampConfig(n_states=2, hidden=(16, 16), lag=3, epochs=60,
                     patience=20, dropout=0.0, seed=0, device="cpu")
    _, _, report = fit_vampnet(runs, cfg)
    assert report.val_score < 0.15


def test_fit_vampnet_early_stops_before_the_epoch_cap():
    runs, _ = _two_basin_runs(n_runs=8, T=100, seed=1)
    cfg = VampConfig(n_states=2, hidden=(8,), lag=2, epochs=500, patience=3,
                     dropout=0.0, seed=0, device="cpu")
    _, _, report = fit_vampnet(runs, cfg)
    assert report.epochs_run < 500


def test_fit_vampnet_holds_out_runs_not_frames():
    """A frame-level split would leak: frames of one trajectory are not
    independent, so the early-stopping signal would be contaminated."""
    runs, _ = _two_basin_runs(n_runs=10, T=60, seed=2)
    cfg = VampConfig(n_states=2, hidden=(8,), lag=2, epochs=5, dropout=0.0,
                     seed=0, device="cpu", inner_val_frac=0.3)
    _, _, r1 = fit_vampnet(runs, cfg)
    # the same configuration with a different seed picks different *runs* for
    # the inner split, so the validation score is expected to move
    _, _, r2 = fit_vampnet(runs, VampConfig(**{**cfg.__dict__, "seed": 7}))
    assert r1.val_score != r2.val_score


def test_fit_vampnet_rejects_a_single_run():
    with pytest.raises(ValueError, match="at least two runs"):
        fit_vampnet([np.zeros((20, 3), dtype=np.float32)],
                    VampConfig(device="cpu"))


# ---------------------------------------------------------------------------
# the run descriptor
# ---------------------------------------------------------------------------

def test_soft_populations_are_a_probability_vector():
    runs, _ = _two_basin_runs(n_runs=6, T=80, seed=3)
    cfg = VampConfig(n_states=3, hidden=(8,), lag=2, epochs=5, dropout=0.0,
                     seed=0, device="cpu")
    model, std, _ = fit_vampnet(runs, cfg)
    p = soft_populations(model, std, runs[0])
    assert p.shape == (3,)
    assert (p >= 0).all() and np.isclose(p.sum(), 1.0)


def test_soft_populations_are_invariant_to_frame_order():
    runs, _ = _two_basin_runs(n_runs=6, T=80, seed=4)
    cfg = VampConfig(n_states=3, hidden=(8,), lag=2, epochs=5, dropout=0.0,
                     seed=0, device="cpu")
    model, std, _ = fit_vampnet(runs, cfg)
    r = runs[0]
    shuffled = r[np.random.default_rng(0).permutation(len(r))]
    assert np.allclose(soft_populations(model, std, r),
                       soft_populations(model, std, shuffled), atol=1e-6)


def test_soft_populations_hellinger_is_the_square_root():
    runs, _ = _two_basin_runs(n_runs=6, T=80, seed=5)
    cfg = VampConfig(n_states=3, hidden=(8,), lag=2, epochs=5, dropout=0.0,
                     seed=0, device="cpu")
    model, std, _ = fit_vampnet(runs, cfg)
    p = soft_populations(model, std, runs[0])
    h = soft_populations(model, std, runs[0], hellinger=True)
    assert np.allclose(h, np.sqrt(p))


def test_soft_populations_separate_two_processes_with_different_occupancies():
    """The descriptor has to respond to the quantity it claims to measure.

    Trained on a **slow** process -- a fast one has no slow mode for VAMP-2 to
    find, so the network would have nothing to learn and the test would be
    measuring an untrained network.
    """
    slow, _ = _two_basin_runs(n_runs=8, T=200, seed=6, flip=0.05)
    cfg = VampConfig(n_states=2, hidden=(16, 16), lag=3, epochs=60,
                     patience=20, dropout=0.0, seed=0, device="cpu")
    model, std, _ = fit_vampnet(slow, cfg)

    rng = np.random.default_rng(0)

    def biased(p_second, n=6, T=200):
        out = []
        for _ in range(n):
            s = (rng.random(T) < p_second).astype(int)
            theta = np.where(s == 0, 0.0, np.pi) + 0.25 * rng.standard_normal(T)
            out.append(np.c_[np.cos(theta), np.sin(theta),
                             0.1 * rng.standard_normal(T)].astype(np.float32))
        return out

    lo = np.stack([soft_populations(model, std, r) for r in biased(0.2)])
    hi = np.stack([soft_populations(model, std, r) for r in biased(0.8)])

    # The comparison is a standardised effect size, not an absolute difference
    # in population units, and that is a statement about VAMP-2 rather than a
    # weakened test. The score is invariant to any invertible linear map of the
    # assignment (`test_score_is_invariant_to_an_invertible_linear_
    # reparametrisation`), so it pins down the slow *subspace* and says nothing
    # about scale: a near-uniform softmax whose small variations point the right
    # way scores exactly as well as a sharp partition. The descriptor is
    # therefore informative only up to an affine transform, which the
    # standardisation in the stage pipeline removes -- and which is also why the
    # interpretability a clustered state model offers ("60% of the time in the
    # closed state") does not transfer to this one.
    pooled = np.sqrt((lo.var(0, ddof=1) + hi.var(0, ddof=1)) / 2)
    d = np.abs(lo.mean(0) - hi.mean(0)) / pooled
    assert d.max() > 0.8


def test_report_flags_a_collapse_instead_of_reporting_it_as_a_null():
    """A collapsed fit and a signal-free process both score near zero. The flag
    is what keeps the first from being read as evidence for the second."""
    runs, _ = _two_basin_runs(n_runs=8, T=120, seed=8, flip=0.45)
    cfg = VampConfig(n_states=2, hidden=(8,), lag=3, epochs=30, patience=10,
                     dropout=0.0, seed=0, device="cpu", n_init=1)
    _, _, report = fit_vampnet(runs, cfg)
    assert report.collapsed == (report.val_score < COLLAPSE_SCORE)
    if report.collapsed:
        assert "COLLAPSED" in str(report)


def test_restarts_are_all_recorded_and_the_best_is_kept():
    runs, _ = _two_basin_runs(n_runs=10, T=120, seed=9, flip=0.05)
    cfg = VampConfig(n_states=2, hidden=(16, 16), lag=3, epochs=40,
                     patience=15, dropout=0.0, seed=0, device="cpu", n_init=3)
    _, _, report = fit_vampnet(runs, cfg)
    assert len(report.restarts) == 3
    assert report.val_score == pytest.approx(max(report.restarts))
    assert "restarts" in str(report)


def test_restarts_rescue_the_collapse_that_a_single_fit_falls_into():
    """Dataset seed 6 collapses deterministically at fit seed 0 and succeeds at
    fit seed 1; the restart loop has to find the second one."""
    runs, _ = _two_basin_runs(n_runs=8, T=200, seed=6, flip=0.05)
    base = dict(n_states=2, hidden=(16, 16), lag=3, epochs=120, patience=30,
                dropout=0.0, device="cpu")
    _, _, single = fit_vampnet(runs, VampConfig(**base, seed=0, n_init=1))
    _, _, multi = fit_vampnet(runs, VampConfig(**base, seed=0, n_init=3))
    assert single.collapsed
    assert not multi.collapsed
    assert multi.val_score > 0.3
