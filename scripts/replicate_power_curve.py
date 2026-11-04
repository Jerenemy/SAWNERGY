"""How many replicates does a given effect size need? Measured on real MD.

Within one mdCATH domain every pair of temperature rungs is a condition
contrast of known size, from 28 K (adjacent) to 130 K (extreme). Measuring the
separation ratio for each pair gives an empirical curve of
"perturbation size -> replicates required", on real trajectories rather than
from a rule of thumb.

That curve is what sizes the p53 campaign. The Y220C melting-point deficit is
~8.4 K, which is *smaller than the smallest contrast available here*, so the
curve has to be read as a lower bound on the replicate count -- the honest
reading is "at least this many, probably more".

Usage:
    python scripts/replicate_power_curve.py <dir-with-mdcath-h5>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sawnergy.md import (MdCathReader, replicates_required,  # noqa: E402
                         run_descriptor, separation_ratio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("mdcath_dataset_*.h5"))
    rows: list[tuple] = []

    for path in files:
        r = MdCathReader(path)
        X, T = [], []
        for frames in r.iter_runs(stride=args.stride):
            d = run_descriptor(frames)
            X.append(np.concatenate([d.mean_contact, d.contact_fluct]))
            T.append(frames.run.temperature_K)
        X = np.array(X)
        T = np.array(T)
        labels = np.array([f"T{t:g}" for t in T])
        temps = np.unique(T)
        for i in range(len(temps)):
            for j in range(i + 1, len(temps)):
                a, b = temps[i], temps[j]
                ratio = separation_ratio(X, labels, f"T{a:g}", f"T{b:g}")
                rows.append((r.domain, b - a, ratio, replicates_required(ratio)))

    print(f"{'domain':<10} {'dT (K)':>7} {'separation':>11} {'n needed':>9}")
    print("-" * 42)
    for dom, dt, ratio, need in sorted(rows, key=lambda x: x[1]):
        print(f"{dom:<10} {dt:7.0f} {ratio:11.2f} {need:9d}")

    arr = np.array([(dt, ratio) for _, dt, ratio, _ in rows])
    print("\nPooled across domains:")
    for lo, hi in [(0, 40), (40, 70), (70, 100), (100, 200)]:
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.sum():
            med = float(np.median(arr[m, 1]))
            print(f"  dT {lo:3d}-{hi:3d} K : median separation {med:5.2f} "
                  f"replicate-sd -> {replicates_required(med):4d} replicates "
                  f"({int(m.sum())} pairs)")

    # linear fit of separation vs dT, extrapolated to the Y220C-sized effect
    if len(arr) >= 3:
        slope, intercept = np.polyfit(arr[:, 0], arr[:, 1], 1)
        for target in (8.4, 15.0, 28.0):
            pred = slope * target + intercept
            print(f"\n  extrapolated to dT = {target:4.1f} K: separation "
                  f"~{pred:.2f} replicate-sd -> ~{replicates_required(max(pred, 1e-3))} "
                  f"replicates per system")
        print("\n  NOTE: this is an extrapolation below the smallest measured "
              "contrast (28 K) and assumes the p53 mutation perturbs contacts "
              "as much as an equivalent temperature change. Treat it as a lower "
              "bound on the replicate count, not an estimate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
