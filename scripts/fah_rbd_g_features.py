"""Stage G, part 1: build every candidate feature block from the all-atom pull.

One cached npz per run, so the evaluation in stage_g.py can be re-run without
recomputing 444 trajectories. Everything stored here is indexed by **residue**,
not by atom, so the blocks are directly comparable across variants whose atom
counts differ (3009 / 2995 / 3003 / 3001). That alignment is the point: an
atom-indexed feature would encode the substitution itself rather than its
consequences, which is the trap documented in sawnergy.md.atomic.

Blocks, all means over the 90 kept frames unless noted:

    ca_mean    soft contact occupancy, CA-CA          -- the existing baseline
    cb_mean    soft contact occupancy, CB-CB          -- side-chain direction
    sc_mean    soft contact occupancy, side-chain centroid
    md_mean    closest heavy-atom approach, in A
    ca_fluct   temporal sd of the CA contact series
    ca_ac      lag-1 autocorrelation of the CA contact series
    ca_life    mean dwell time of the CA contact series, in frames
    dih_mean   resultant of phi/psi/chi1 as sin/cos
    bur_mean   CB neighbour count within 10 A
    dih_frames per-frame dihedrals, float16, kept for TICA

Usage: g_features.py <run,run,...> <clone_lo> <clone_hi> <outdir>
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")

sys.path.insert(0, os.path.join(REPO, "src"))

from sawnergy.md import contact_map  # noqa: E402
from sawnergy.md.atomic import (AtomTable, burial, cb_coords,  # noqa: E402
                                dihedral_features, residue_min_distance,
                                sidechain_centroids)
from sawnergy.md.temporal import (autocorrelation, fluctuation,  # noqa: E402
                                  lifetime)

ALLATOM = os.path.join(DATA, "allatom")

#: Chain 0 as deposited holds 196 residues: 194 numbered 333-526, plus an ACE
#: and an NME terminal cap. The CA-only pass kept 193 of them, because it
#: filtered to the standard twenty and so dropped NLN -- the N-glycosylated
#: asparagine at the single sequon in this construct. Stage G keeps the same
#: 193 so its CA arm reproduces stage A exactly and every block, from contacts
#: to dihedrals, shares one residue index. NLN is present in all four variants,
#: so this costs comparability nothing; it is alignment, not a filter on signal.
N_RES = 193
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _protein_only(xyz, table):
    """Restrict to the 193 standard residues, re-indexing contiguously."""
    keep_atom = np.isin(table.res_name, list(AA3))
    old = table.res_index[keep_atom]
    _, new = np.unique(old, return_inverse=True)
    sub = AtomTable(atom_name=table.atom_name[keep_atom],
                    res_name=table.res_name[keep_atom],
                    res_index=new.astype(np.int32),
                    res_seq=table.res_seq[keep_atom])
    if sub.n_residues != N_RES:
        raise ValueError(f"{sub.n_residues} standard residues, expected {N_RES}")
    return xyz[:, keep_atom], sub


def blocks(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float32)          # (T, A, 3) in angstrom
        table = AtomTable(atom_name=z["atom_name"].astype(str),
                          res_name=z["res_name"].astype(str),
                          res_index=z["res_index"], res_seq=z["res_seq"])
        probe = str(z["probe"])
    xyz, table = _protein_only(xyz, table)

    iu = np.triu_indices(N_RES, k=1)
    out: dict[str, np.ndarray] = {}

    # --- three geometries, same contact kernel, same pair index -------------
    for name, pts in (("ca", xyz[:, table.one_per_residue("CA", fallback=None)]),
                      ("cb", cb_coords(xyz, table)),
                      ("sc", sidechain_centroids(xyz, table))):
        c = contact_map(pts)[:, iu[0], iu[1]]      # (T, P)
        out[f"{name}_mean"] = c.mean(axis=0).astype(np.float32)
        if name == "ca":
            out["ca_fluct"] = fluctuation(c)
            out["ca_ac"] = autocorrelation(c, lag=1)
            out["ca_life"] = lifetime(c)

    # --- closest heavy-atom approach ---------------------------------------
    md = residue_min_distance(xyz, table, prefilter_r=20.0)
    out["md_mean"] = md[:, iu[0], iu[1]].mean(axis=0).astype(np.float32)

    # --- per-residue, count-invariant --------------------------------------
    dih = dihedral_features(xyz, table)            # (T, R, 6)
    out["dih_mean"] = dih.mean(axis=0).ravel().astype(np.float32)
    out["dih_frames"] = dih.reshape(dih.shape[0], -1).astype(np.float16)
    out["bur_mean"] = burial(xyz, table).mean(axis=0).astype(np.float32)

    out["res_seq"] = table.residue_numbers().astype(np.int32)
    out["res_name"] = table.residue_sequence().astype("U4")
    out["probe"] = np.array(probe)
    return out


if __name__ == "__main__":
    runs = [int(x) for x in sys.argv[1].split(",")]
    lo, hi, outdir = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(outdir, exist_ok=True)
    for c in range(lo, hi):
        for r in runs:
            src = os.path.join(ALLATOM, f"run{r}-clone{c}.npz")
            dst = os.path.join(outdir, f"run{r}-clone{c}.npz")
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            try:
                np.savez_compressed(dst, **blocks(src))
                print(f"run{r}-clone{c}: ok", flush=True)
            except Exception as e:
                print(f"run{r}-clone{c}: FAIL {type(e).__name__}: {e}",
                      flush=True)
