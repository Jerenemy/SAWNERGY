"""Validate the MD embedder on real trajectories, before any p53 compute.

mdCATH contains no p53. It is used here because it is the only large public MD
set whose replicate and temperature structure is known by construction: 5
replicates x 5 temperatures per domain. Within one domain, **temperature plays
exactly the role that the mutation plays in the p53 problem** -- a condition
that changes the dynamics, sampled with several independent replicates -- so
every claim the pipeline will later make about WT vs Y220C can be rehearsed
here against ground truth.

Experiments:

  A. Sham (G3)         Within ONE temperature, split the 5 replicates into two
                       fake conditions. They differ only by MD noise, so the
                       pipeline must report no difference.
  B. Separability (G4) Across temperatures, under leave-one-run-out.
  C. Power (G5)        Measured effect size -> replicates actually required.
  D. Thermal axis (G7) Fit the temperature response, then hold out one
                       temperature entirely and try to recover it. This is the
                       direct test of the apparent-temperature metric that the
                       p53 dT_apparent readout depends on.
  E. Baseline          Does anything beat the trivial RMSF descriptor?

Usage:
    python scripts/validate_on_mdcath.py <dir-with-mdcath-h5> [--stride 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sawnergy.md import (MdCathReader, PCAEmbedder, ThermalAxis,  # noqa: E402
                         leave_one_replicate_out, replicates_required,
                         run_descriptor, separation_ratio, thermal_shift,
                         two_way_components, variance_components)
from sawnergy.md.validate import (gate_separability, gate_sham,  # noqa: E402
                                  gate_temperature)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load_domain(path: Path, stride: int, burn_in: float):
    """Return descriptors + labels for every run of one domain."""
    r = MdCathReader(path)
    vecs_mean, vecs_fluct, vecs_rmsf, temps, reps = [], [], [], [], []
    for frames in r.iter_runs(stride=stride):
        d = run_descriptor(frames, burn_in=burn_in)
        vecs_mean.append(d.mean_contact)
        vecs_fluct.append(d.contact_fluct)
        vecs_rmsf.append(d.rmsf)
        temps.append(frames.run.temperature_K)
        reps.append(frames.run.replicate)
    return (r.domain, np.array(vecs_mean), np.array(vecs_fluct),
            np.array(vecs_rmsf), np.array(temps), np.array(reps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--burn-in", type=float, default=0.1)
    ap.add_argument("--permutations", type=int, default=2000)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("mdcath_dataset_*.h5"))
    if not files:
        print(f"no mdCATH files in {args.data_dir}", file=sys.stderr)
        return 1

    print(f"domains: {len(files)}  stride={args.stride}  burn_in={args.burn_in}")

    for path in files:
        dom, M, F, R, T, REP = load_domain(path, args.stride, args.burn_in)
        banner(f"DOMAIN {dom}   ({len(T)} runs, "
               f"{len(np.unique(T))} temperatures x {len(np.unique(REP))} replicates)")

        blocks = {"mean_contact": M, "contact_fluct": F, "rmsf": R,
                  "contact_both": np.hstack([M, F])}

        # ---------------------------------------------------------- A: sham
        banner_done = False
        for name, X in blocks.items():
            g = gate_sham(X, np.array([f"T{t:g}" for t in T]), REP,
                          n_trials=8, n_permutations=500,
                          rng=np.random.default_rng(0))
            if not banner_done:
                print("\n-- A. SHAM: fake conditions inside one temperature "
                      "(must find NO difference)")
                banner_done = True
            print(f"   {name:16s} {g}")

        # ------------------------------------------------- B: separability
        print("\n-- B. SEPARABILITY across temperature (leave-one-run-out)")
        tl = np.array([f"T{t:g}" for t in T])
        for name, X in blocks.items():
            g = gate_separability(X, tl, n_permutations=args.permutations,
                                  rng=np.random.default_rng(1))
            print(f"   {name:16s} {g.detail}")

        # -------------------------------------------------------- C: power
        print("\n-- C. EFFECT SIZE and replicates required")
        temps_sorted = np.unique(T)
        lo, hi = temps_sorted[0], temps_sorted[1]       # ADJACENT rungs: hardest
        for name, X in blocks.items():
            try:
                ratio = separation_ratio(X, tl, f"T{lo:g}", f"T{hi:g}")
                need = replicates_required(ratio)
                print(f"   {name:16s} {lo:g}K vs {hi:g}K: separation="
                      f"{ratio:5.2f} replicate-sd -> need {need} replicates "
                      f"(have {int((tl == f'T{lo:g}').sum())})")
            except ValueError as exc:
                print(f"   {name:16s} {exc}")

        # ------------------------------------------------- D: thermal axis
        print("\n-- D. THERMAL AXIS (fit, then recover a held-out temperature)")
        X = blocks["contact_both"]
        emb = PCAEmbedder.fit(X, n_components=8)
        Z = emb.transform(X)

        g = gate_temperature(Z, T)
        print(f"   {g}")

        for held in temps_sorted:
            train = T != held
            if len(np.unique(T[train])) < 3:
                continue
            # refit the projection on training runs only -- no leakage
            emb_tr = PCAEmbedder.fit(X[train], n_components=8)
            Z_tr, Z_te = emb_tr.transform(X[train]), emb_tr.transform(X[~train])
            try:
                axis = ThermalAxis.fit(Z_tr, T[train])
                ts = thermal_shift(Z_te, axis, float(held))
                err = ts.apparent_T_K - held
                flag = "" if ts.extrapolation_K <= 0 else " [extrapolated]"
                print(f"   held out {held:3.0f} K -> predicted "
                      f"{ts.apparent_T_K:6.1f} K  (error {err:+6.1f} K, "
                      f"sem {ts.sem_K:4.1f}, axis R2={axis.r2:.3f}){flag}")
            except ValueError as exc:
                print(f"   held out {held:3.0f} K -> {exc}")

        # ------------------------------------------------ E: disentangling
        print("\n-- E. TWO-WAY: temperature vs replicate")
        res = two_way_components(blocks["contact_both"], tl,
                                 np.array([f"r{r}" for r in REP]),
                                 n_permutations=1000,
                                 rng=np.random.default_rng(2))
        print(f"   ICC(temperature)={res['icc_system']:+.3f} p={res['p_system']:.4g}")
        print(f"   ICC(replicate)  ={res['icc_temperature']:+.3f} "
              f"p={res['p_temperature']:.4g}   <- should be ~0 and non-significant")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
