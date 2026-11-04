"""Tests for the nested symmetric frame/trajectory autoencoder."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sawnergy.md import (AEConfig, FitConfig, build_nested_autoencoder,  # noqa: E402
                         build_time_encoding, embed_runs, fit_nested,
                         make_windows, nested_loss)


def _cfg(**kw):
    base = dict(d_in=40, d_frame=16, d_traj=8, frame_hidden=(64, 32),
                traj_hidden=(32, 16))
    base.update(kw)
    return AEConfig(**base)


def _runs(n=8, T=40, d=40, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(T, d)).astype(np.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_shapes_through_the_nest(self):
        cfg = _cfg()
        m = build_nested_autoencoder(cfg)
        X = torch.randn(5, 20, cfg.d_in)
        t = torch.linspace(0, 1, 20).expand(5, 20)
        out = m(X, t)
        assert out["H"].shape == (5, 20, cfg.d_frame)
        assert out["z"].shape == (5, cfg.d_traj)
        assert out["H_hat"].shape == (5, 20, cfg.d_frame)
        assert out["X_hat_frame"].shape == X.shape
        assert out["X_hat_full"].shape == X.shape

    def test_decoder_mirrors_encoder_widths(self):
        """The 'symmetrical' requirement, checked rather than asserted."""
        cfg = _cfg()
        m = build_nested_autoencoder(cfg)
        enc = [l.out_features for l in m.E_f if hasattr(l, "out_features")]
        dec = [l.out_features for l in m.D_f if hasattr(l, "out_features")]
        assert enc[:-1][::-1] == dec[:-1]
        assert dec[-1] == cfg.d_in

    def test_trajectory_decoder_is_queried_by_timestep(self):
        """D_tau(z, t) must actually depend on t, or the mirror is cosmetic."""
        cfg = _cfg(time_conditioned=True)
        m = build_nested_autoencoder(cfg)
        z = torch.randn(1, cfg.d_traj)
        h_early = m.decode_trajectory(z, torch.zeros(1, 1))
        h_late = m.decode_trajectory(z, torch.ones(1, 1))
        assert not torch.allclose(h_early, h_late), \
            "decoder output is identical at t=0 and t=1; it ignores the stamp"

    def test_time_conditioning_can_be_disabled(self):
        cfg = _cfg(time_conditioned=False)
        m = build_nested_autoencoder(cfg)
        z = torch.randn(1, cfg.d_traj)
        a = m.decode_trajectory(z, torch.zeros(1, 1))
        b = m.decode_trajectory(z, torch.ones(1, 1))
        assert torch.allclose(a, b)

    def test_variable_length_runs_share_one_latent_space(self):
        """Different T must still give a z of the same shape."""
        cfg = _cfg()
        m = build_nested_autoencoder(cfg)
        m._sawnergy_norm = (0.0, 1.0)
        Z = embed_runs(m, [np.random.randn(30, cfg.d_in).astype(np.float32),
                           np.random.randn(55, cfg.d_in).astype(np.float32)])
        assert Z.shape == (2, cfg.d_traj)


class TestTimeEncoding:
    def test_distinct_times_give_distinct_codes(self):
        te = build_time_encoding(6)
        a = te(torch.tensor([0.0]))
        b = te(torch.tensor([0.37]))
        assert a.shape[-1] == te.dim == 12
        assert not torch.allclose(a, b)

    def test_is_continuous_in_time(self):
        te = build_time_encoding(4)
        a = te(torch.tensor([0.50]))
        b = te(torch.tensor([0.5001]))
        assert torch.allclose(a, b, atol=1e-2)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

class TestLoss:
    def test_reports_all_three_levels(self):
        cfg = _cfg()
        m = build_nested_autoencoder(cfg)
        X = torch.randn(4, 16, cfg.d_in)
        t = torch.linspace(0, 1, 16).expand(4, 16)
        _, parts = nested_loss(m(X, t), X, cfg)
        assert set(parts) == {"frame", "h", "full", "total"}
        assert all(v >= 0 for v in parts.values())

    def test_weights_select_levels(self):
        cfg = _cfg(w_frame=1.0, w_h=0.0, w_full=0.0)
        m = build_nested_autoencoder(cfg)
        X = torch.randn(4, 16, cfg.d_in)
        t = torch.linspace(0, 1, 16).expand(4, 16)
        total, parts = nested_loss(m(X, t), X, cfg)
        assert float(total) == pytest.approx(parts["frame"], rel=1e-5)


# ---------------------------------------------------------------------------
# windowing keeps run provenance
# ---------------------------------------------------------------------------

class TestWindows:
    def test_every_window_keeps_its_parent_run(self):
        runs = _runs(n=3, T=40)
        X, t, owner = make_windows(runs, 20, 10, ["a", "b", "c"])
        assert X.shape[1] == 20 and t.shape == X.shape[:2]
        assert set(owner) == {"a", "b", "c"}
        assert len(owner) == X.shape[0]

    def test_timestamps_are_run_relative(self):
        """A late window must carry larger timestamps than an early one."""
        X, t, owner = make_windows(_runs(n=1, T=100), 20, 20, ["a"])
        assert t[0].max() < t[-1].min()
        assert 0.0 <= t.min() and t.max() <= 1.0

    def test_refuses_runs_shorter_than_the_window(self):
        with pytest.raises(ValueError, match="at least"):
            make_windows(_runs(n=2, T=10), 50, 10, ["a", "b"])


# ---------------------------------------------------------------------------
# staged training
# ---------------------------------------------------------------------------

class TestFit:
    def test_three_stages_reported(self):
        cfg = _cfg()
        m = build_nested_autoencoder(cfg)
        Xtr, ttr, _ = make_windows(_runs(n=6, seed=1), 20, 10, list("abcdef"))
        Xva, tva, _ = make_windows(_runs(n=2, seed=99), 20, 10, list("gh"))
        reps = fit_nested(m, Xtr, ttr, Xva, tva, cfg,
                          FitConfig(epochs_frame=3, epochs_traj=3,
                                    epochs_joint=2, device="cpu"))
        assert [r.stage for r in reps] == ["S1 frame AE", "S2 trajectory AE",
                                           "S3 joint"]
        assert all(np.isfinite(r.val_loss) for r in reps)

    def test_learns_recoverable_structure(self):
        """On low-rank data the frame AE must beat predicting the mean."""
        rng = np.random.default_rng(0)
        basis = rng.normal(size=(4, 40))
        def mk(n, seed):
            r = np.random.default_rng(seed)
            return [(r.normal(size=(40, 4)) @ basis).astype(np.float32)
                    for _ in range(n)]
        cfg = _cfg(d_frame=8)
        m = build_nested_autoencoder(cfg)
        Xtr, ttr, _ = make_windows(mk(10, 1), 20, 10, [f"r{i}" for i in range(10)])
        Xva, tva, _ = make_windows(mk(3, 42), 20, 10, [f"v{i}" for i in range(3)])
        reps = fit_nested(m, Xtr, ttr, Xva, tva, cfg,
                          FitConfig(epochs_frame=120, epochs_traj=20,
                                    epochs_joint=10, device="cpu", lr=3e-3))
        assert reps[0].val_r2 > 0.5, f"frame AE did not learn: {reps[0]}"

    def test_validation_uses_held_out_runs_not_frames(self):
        """Sanity: validation windows must not share owners with training."""
        tr = [f"r{i}" for i in range(5)]
        va = [f"v{i}" for i in range(2)]
        _, _, o_tr = make_windows(_runs(n=5, seed=2), 20, 10, tr)
        _, _, o_va = make_windows(_runs(n=2, seed=3), 20, 10, va)
        assert not (set(o_tr) & set(o_va))
