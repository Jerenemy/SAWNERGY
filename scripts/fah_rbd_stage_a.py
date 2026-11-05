"""Stage A: run-level statistics and validation gates on FAH PROJ17311.

The analogue system. Same protein (SARS-CoV-2 RBD, 193 residues, close in size
to the p53 DBD's 195), four sequence variants, hundreds of velocity-independent
replicates each:

    run3  WT                 reference
    run0  N439K              single mutant
    run1  K417V              single mutant
    run2  N439K/K417V        double mutant

This is the WT-vs-mutant structure the p53 programme needs and cannot get, with
a replicate count well past the ~87 the power extrapolation demands. The double
mutant supplies a dose axis: if a contact-space embedding measures sequence
perturbation at all, distance from WT should order WT < single < double.

Features are full residue-pair contact descriptors with NO patch pooling. The
cross-protein pooling that the mdCATH work required is exactly what destroyed
the within-protein signal, and here it is unnecessary: one protein, one
numbering, so pair i-j means the same thing in every run.
"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(REPO, "src"))

from sawnergy.md import (Frames, Independence, Role, Run,  # noqa: E402
                         contact_map, leave_one_replicate_out,
                         replicate_splits, replicates_required, run_gates,
                         separation_ratio, seq_separation_mask,
                         variance_components)

CA_DIR = Path(f"{DATA}/ca")
OUT = Path(f"{DATA}/stage_a")

#: run id -> (system label, expected residue 417, expected residue 439, role)
RUNS = {
    3: ("WT", "LYS", "ASN", Role.REFERENCE),
    0: ("N439K", "LYS", "LYS", Role.BASELINE),
    1: ("K417V", "VAL", "ASN", Role.BASELINE),
    2: ("N439K+K417V", "VAL", "LYS", Role.BASELINE),
}

#: Frames kept per run after burn-in. Fixing this removes trajectory length as a
#: confounder: FAH trajectory lengths vary with donor hardware, not with system,
#: but a length difference that correlated with system by chance would look
#: exactly like a real effect.
KEEP_FRAMES = 180
BURN_IN = 0.10
TEMP_K = 310.0


# ---------------------------------------------------------------------------
# loading, with provenance validation
# ---------------------------------------------------------------------------

def load_runs(ca_dir: Path = CA_DIR):
    """Load extracted CA trajectories, validating every label.

    Returns:
        (frames_list, rejects) where rejects maps file name -> reason.
    """
    frames, rejects = [], {}
    counter = {k: 0 for k in RUNS}
    for p in sorted(ca_dir.glob("run*-clone*.npz")):
        run_id = int(p.name.split("-")[0][3:])
        clone = int(p.name.split("clone")[1].split(".")[0])
        if run_id not in RUNS:
            rejects[p.name] = f"unknown run {run_id}"
            continue
        system, exp417, exp439, role = RUNS[run_id]
        with np.load(p, allow_pickle=False) as z:
            probe = json.loads(str(z["probe"]))
            # G-identity: the sequence must match the label, independently of
            # the deposit's README.
            if probe.get("417") != exp417 or probe.get("439") != exp439:
                rejects[p.name] = (f"sequence mismatch: 417={probe.get('417')} "
                                   f"439={probe.get('439')}, expected "
                                   f"{exp417}/{exp439}")
                continue
            xyz = z["xyz"]
            resseq = z["resseq"]
        start = int(round(BURN_IN * xyz.shape[0]))
        usable = xyz[start:]
        if usable.shape[0] < KEEP_FRAMES:
            rejects[p.name] = f"too short: {usable.shape[0]} < {KEEP_FRAMES}"
            continue
        ca = usable[:KEEP_FRAMES]
        n = ca.shape[1]
        run = Run(system=system, replicate=counter[run_id],
                  temperature_K=TEMP_K, independence=Independence.VELOCITY,
                  seed=clone, role=role, n_frames=KEEP_FRAMES,
                  length_ns=KEEP_FRAMES * 0.5,
                  meta={"fah_run": run_id, "fah_clone": clone,
                        "resseq_min": int(resseq.min()),
                        "resseq_max": int(resseq.max()),
                        "source": "FAH PROJ17311 RBD:ACE2",
                        "n_frames_available": int(xyz.shape[0])})
        counter[run_id] += 1
        frames.append(Frames(coords=ca.reshape(KEEP_FRAMES, n, 3),
                             resid=np.arange(n), run=run,
                             names=np.array(["CA"] * n)))
    return frames, rejects


# ---------------------------------------------------------------------------
# features: full residue-pair contact descriptors, no pooling
# ---------------------------------------------------------------------------

def pair_features(frames_list, *, stride: int = 1):
    """Per-run mean contact vector over all kept residue pairs.

    Returns:
        (X, systems, replicates, pair_index) with X of shape (n_runs, P).
    """
    n = frames_list[0].coords.shape[1]
    iu = np.triu_indices(n, k=1)
    keep = seq_separation_mask(n, 3)[iu]
    pair_index = np.stack([iu[0][keep], iu[1][keep]], axis=1)

    X, systems, reps = [], [], []
    for fr in frames_list:
        ca = fr.ca_coords()[::stride]
        c = contact_map(ca)
        v = c[:, iu[0], iu[1]][:, keep]
        X.append(v.mean(axis=0).astype(np.float32))
        systems.append(fr.run.system)
        reps.append(fr.run.replicate)
    return (np.stack(X), np.array(systems), np.array(reps), pair_index)


# ---------------------------------------------------------------------------
# length-confound check
# ---------------------------------------------------------------------------

def length_confound(frames_list) -> dict:
    """Is available trajectory length associated with system?

    FAH work-unit lengths depend on which volunteer's GPU picked up the job, so
    they should be independent of system. If they are not, truncation to a
    common length is doing real work rather than being cosmetic.
    """
    by_sys: dict[str, list[int]] = {}
    for fr in frames_list:
        by_sys.setdefault(fr.run.system, []).append(
            fr.run.meta["n_frames_available"])
    groups = [np.asarray(v, dtype=float) for v in by_sys.values()]
    grand = np.concatenate(groups)
    k = len(groups)
    ss_b = sum(g.size * (g.mean() - grand.mean()) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, grand.size - k
    F = (ss_b / df_b) / (ss_w / df_w) if ss_w > 0 and df_w > 0 else np.inf

    rng = np.random.default_rng(0)
    sizes = [g.size for g in groups]
    worse = 0
    for _ in range(2000):
        perm = rng.permutation(grand)
        cuts = np.cumsum(sizes)[:-1]
        gs = np.split(perm, cuts)
        sb = sum(g.size * (g.mean() - grand.mean()) ** 2 for g in gs)
        sw = sum(((g - g.mean()) ** 2).sum() for g in gs)
        f = (sb / df_b) / (sw / df_w) if sw > 0 else np.inf
        worse += f >= F
    return {"F": float(F), "p": float((worse + 1) / 2001),
            "means": {k2: float(np.mean(v)) for k2, v in by_sys.items()}}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames_list, rejects = load_runs()

    counts: dict[str, int] = {}
    for fr in frames_list:
        counts[fr.run.system] = counts.get(fr.run.system, 0) + 1

    print("=" * 72)
    print("STAGE A  FAH PROJ17311 RBD:ACE2 -- run-level statistics")
    print("=" * 72)
    print(f"\nloaded {len(frames_list)} runs, rejected {len(rejects)}")
    for name, why in list(rejects.items())[:8]:
        print(f"  reject {name}: {why}")
    if len(rejects) > 8:
        print(f"  ... and {len(rejects) - 8} more")
    print("\nreplicates per system:")
    for s in ("WT", "N439K", "K417V", "N439K+K417V"):
        n = counts.get(s, 0)
        print(f"  {s:14s} {n:4d}" + ("" if n >= 10 else "   << thin"))

    # --- validation: trajectory length must not track system -----------------
    lc = length_confound(frames_list)
    print(f"\nG-length  available frames by system: "
          + ", ".join(f"{k}={v:.0f}" for k, v in lc["means"].items()))
    print(f"          F={lc['F']:.2f} p={lc['p']:.4f} -> "
          + ("length tracks system; truncation is load-bearing"
             if lc["p"] < 0.05 else "length independent of system"))

    # --- features ------------------------------------------------------------
    X, systems, reps, pair_index = pair_features(frames_list)
    print(f"\nfeatures  {X.shape[0]} runs x {X.shape[1]} residue pairs "
          f"(193 residues, |i-j|>3, no pooling)")
    print(f"          mean contact occupancy {X.mean():.4f}, "
          f"per-pair sd in [{X.std(axis=0).min():.5f}, "
          f"{X.std(axis=0).max():.5f}]")

    # Save features before the gates: the gate ladder takes ~35 min and a
    # crash after it should never cost the cheap part.
    np.savez_compressed(OUT / "stage_a.npz", X=X, systems=systems,
                        replicates=reps, pair_index=pair_index)

    # --- gate ladder ---------------------------------------------------------
    print("\n" + "-" * 72)
    print("GATE LADDER")
    print("-" * 72)
    report = run_gates(X, systems, reps,
                       pair=("WT", "N439K+K417V"))
    for g in report.gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] {g.name:22s} {g.detail}")
    print(f"\n  overall: {'PASS' if report.ok else 'FAIL'}")
    if report.first_failure:
        print(f"  stopped at: {report.first_failure.name}")

    # --- variance components -------------------------------------------------
    vc = variance_components(X, systems)
    print("\n" + "-" * 72)
    print("VARIANCE COMPONENTS (is the signal structure or run-to-run noise?)")
    print("-" * 72)
    print(f"  ICC(system)      {vc.icc:+.4f}")
    print(f"  pseudo-F         {vc.pseudo_f:.3f}")
    print(f"  permutation p    {vc.p_value:.5f}   ({vc.n_permutations} shuffles "
          f"of run labels)")

    # --- pairwise separation and the power question --------------------------
    print("\n" + "-" * 72)
    print("PAIRWISE SEPARATION (multivariate d, in replicate-sd units)")
    print("-" * 72)
    pairs = [("WT", "N439K"), ("WT", "K417V"), ("WT", "N439K+K417V"),
             ("N439K", "N439K+K417V"), ("K417V", "N439K+K417V"),
             ("N439K", "K417V")]
    seps = {}
    for a, b in pairs:
        if counts.get(a, 0) < 3 or counts.get(b, 0) < 3:
            continue
        d = separation_ratio(X, systems, a, b)
        n_req = replicates_required(d)
        seps[f"{a} vs {b}"] = d
        print(f"  {a:12s} vs {b:12s}  d={d:6.3f}   n for 80% power: {n_req:4d}")

    # --- discriminability ----------------------------------------------------
    print("\n" + "-" * 72)
    print("LEAVE-ONE-REPLICATE-OUT CLASSIFICATION")
    print("-" * 72)
    disc = leave_one_replicate_out(X, systems, n_permutations=0)
    print(f"  accuracy {disc.accuracy:.4f}   chance {disc.chance:.4f}   "
          f"(each of {X.shape[0]} runs held out in turn)")
    for s, acc in sorted(disc.per_system.items()):
        print(f"    {s:14s} {acc:.4f}")

    np.savez_compressed(OUT / "stage_a.npz", X=X, systems=systems,
                        replicates=reps, pair_index=pair_index)
    with open(OUT / "stage_a.json", "w") as fh:
        json.dump({"counts": counts, "n_rejected": len(rejects),
                   "rejects": rejects, "length_confound": lc,
                   "icc_system": vc.icc, "pseudo_f": vc.pseudo_f,
                   "p_value": vc.p_value, "separations": seps,
                   "loro_accuracy": disc.accuracy, "loro_chance": disc.chance,
                   "gates": {g.name: [g.passed, g.detail]
                             for g in report.gates}}, fh, indent=2)
    print(f"\nwrote {OUT / 'stage_a.npz'} and {OUT / 'stage_a.json'}")


if __name__ == "__main__":
    main()
