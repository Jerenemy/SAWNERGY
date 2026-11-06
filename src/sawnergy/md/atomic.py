"""Residue-level geometry from **all-atom** coordinates.

`sawnergy.md.features` describes a trajectory entirely through soft contacts
between Cα atoms. That is the wrong representation for the question this package
exists to answer. Every mutation of interest is a side-chain substitution:
p53's Y220C deletes a bulky aromatic and opens a surface crevice, and the
SARS-CoV-2 validation panel's N439K and K417V change charge and reach. The Cα
barely moves in any of them. A Cα-only featuriser measures the part of the
protein the mutation does not change.

Measured consequence on the RBD panel: K417V (Lys->Val, removing a long charged
side chain) was the one contrast nothing could detect -- separation 0.330,
leave-one-run-out 0.739, and the only contrast where a supervised discriminant
*lost* to an unsupervised autoencoder. Lys->Val is nearly invisible in Cα
contacts and obvious in side-chain geometry.

This module adds the observables that see it, all reduced to one value per
residue or per residue pair so that they stay comparable across runs:

- `sidechain_centroids` -- side-chain centre of geometry per residue
- `cb_coords` -- Cβ positions, the cheapest proxy for side-chain direction
- `residue_min_distance` -- closest approach between two residues' heavy atoms
- `dihedral_features` -- phi/psi/chi1 as sin/cos pairs
- `burial` -- Cβ neighbour count, a cheap solvent-exposure proxy

**The atom-count trap.** A mutation changes a residue's atom count: on the RBD
panel, Asn->Lys adds 8 atoms and Lys->Val removes 6. Any feature that counts
atoms therefore identifies the variant *without reference to its structure*,
which is leakage rather than signal. Two defences, both provided:

1. Prefer count-invariant features. `cb_coords`, `dihedral_features` and
   `burial` (which counts Cβ atoms, exactly one per non-glycine residue) do not
   change their dimension or their arithmetic when a side chain gains atoms.
2. Use `mutation_mask` to drop the substituted sites and, optionally, their
   spatial neighbours. The honest question is not whether a representation can
   see the mutation -- a sequence lookup can do that -- but whether it can see
   the mutation's *propagated effect on the ensemble*. That requires excluding
   the mutated residues themselves.

`residue_min_distance` is count-dependent in a benign way: a longer side chain
genuinely reaches further, so the change is physical rather than bookkeeping.
It is still subject to defence 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)

#: Backbone atom names, excluded when locating side chains.
BACKBONE = ("N", "CA", "C", "O", "OXT")

#: Atoms whose presence marks a hydrogen; excluded from heavy-atom geometry.
_H_PREFIX = ("H", "1H", "2H", "3H")

#: First side-chain atom used for the chi1 dihedral, by residue.
CHI1_THIRD = {
    "ARG": "CG", "ASN": "CG", "ASP": "CG", "CYS": "SG", "CYX": "SG",
    "GLN": "CG", "GLU": "CG", "GLH": "CG", "HIS": "CG", "HID": "CG",
    "HIE": "CG", "HIP": "CG", "ILE": "CG1", "LEU": "CG", "LYS": "CG",
    "LYN": "CG", "MET": "CG", "PHE": "CG", "PRO": "CG", "SER": "OG",
    "THR": "OG1", "TRP": "CG", "TYR": "CG", "VAL": "CG1",
}

#: Cβ neighbour-count radius for `burial`, in angstrom.
BURIAL_R = 10.0


@dataclass
class AtomTable:
    """Per-atom metadata for one chain, shared by every frame.

    Attributes:
        atom_name: ``(A,)`` atom names.
        res_name: ``(A,)`` residue name per atom.
        res_index: ``(A,)`` 0-based residue index per atom.
        res_seq: ``(A,)`` author residue number per atom.
    """

    atom_name: np.ndarray
    res_name: np.ndarray
    res_index: np.ndarray
    res_seq: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.atom_name)
        for f in ("res_name", "res_index", "res_seq"):
            if len(getattr(self, f)) != n:
                raise ValueError(
                    f"{f} has {len(getattr(self, f))} entries but atom_name has "
                    f"{n}; the atom table is inconsistent")

    @property
    def n_residues(self) -> int:
        return int(self.res_index.max()) + 1 if len(self.res_index) else 0

    @property
    def is_heavy(self) -> np.ndarray:
        """Non-hydrogen atoms."""
        return ~np.array([a.strip().startswith(_H_PREFIX)
                          for a in self.atom_name])

    def residue_sequence(self) -> np.ndarray:
        """``(R,)`` residue names, one per residue."""
        out = np.empty(self.n_residues, dtype=object)
        for a in range(len(self.atom_name)):
            out[self.res_index[a]] = self.res_name[a]
        return out.astype("U4")

    def residue_numbers(self) -> np.ndarray:
        """``(R,)`` author residue numbers, one per residue."""
        out = np.zeros(self.n_residues, dtype=np.int32)
        for a in range(len(self.atom_name)):
            out[self.res_index[a]] = self.res_seq[a]
        return out

    def select(self, names) -> np.ndarray:
        """Atom indices whose name is in ``names``."""
        want = {n.strip() for n in names}
        return np.flatnonzero([a.strip() in want for a in self.atom_name])

    def one_per_residue(self, name: str, *, fallback: str | None = "CA"
                        ) -> np.ndarray:
        """``(R,)`` atom index of ``name`` in each residue.

        Glycine has no Cβ, so ``fallback`` supplies a substitute. Returns ``-1``
        where neither is present.
        """
        out = np.full(self.n_residues, -1, dtype=np.int64)
        for a in range(len(self.atom_name)):
            nm = self.atom_name[a].strip()
            r = self.res_index[a]
            if nm == name:
                out[r] = a
            elif fallback and nm == fallback and out[r] == -1:
                out[r] = a
        return out


# ---------------------------------------------------------------------------
# side-chain geometry
# ---------------------------------------------------------------------------

def sidechain_centroids(xyz: np.ndarray, table: AtomTable) -> np.ndarray:
    """``(T, R, 3)`` centre of geometry of each residue's heavy side-chain atoms.

    Glycine, which has no side chain, falls back to its Cα.
    """
    heavy = table.is_heavy
    is_bb = np.array([a.strip() in BACKBONE for a in table.atom_name])
    use = heavy & ~is_bb
    R, T = table.n_residues, xyz.shape[0]
    out = np.zeros((T, R, 3), dtype=np.float32)
    counts = np.bincount(table.res_index[use], minlength=R)
    for d in range(3):
        summed = np.zeros((T, R), dtype=np.float64)
        np.add.at(summed.T, table.res_index[use], xyz[:, use, d].T)
        out[:, :, d] = (summed / np.maximum(counts, 1)).astype(np.float32)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        ca = table.one_per_residue("CA", fallback=None)
        for r in empty:
            if ca[r] >= 0:
                out[:, r, :] = xyz[:, ca[r], :]
    return out


def cb_coords(xyz: np.ndarray, table: AtomTable) -> np.ndarray:
    """``(T, R, 3)`` Cβ positions, with Cα substituted for glycine.

    Count-invariant by construction: exactly one atom per residue whatever the
    residue is, so a substitution cannot change this feature's dimension.
    """
    idx = table.one_per_residue("CB", fallback="CA")
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} residues have neither CB nor CA")
    return xyz[:, idx, :].astype(np.float32)


def residue_min_distance(xyz: np.ndarray, table: AtomTable, *,
                         prefilter_r: float = 20.0) -> np.ndarray:
    """``(T, R, R)`` closest approach between two residues' heavy atoms.

    The full atom-by-atom calculation is ``A^2`` per frame, so pairs whose Cβ
    atoms are further apart than ``prefilter_r`` are skipped and left at
    ``prefilter_r``. Contact features care about short distances, and the switch
    function is saturated well before 20 Å.
    """
    heavy = np.flatnonzero(table.is_heavy)
    rid = table.res_index[heavy]
    R, T = table.n_residues, xyz.shape[0]
    cb = cb_coords(xyz, table)
    by_res = [heavy[rid == r] for r in range(R)]

    out = np.full((T, R, R), prefilter_r, dtype=np.float32)
    for t in range(T):
        d_cb = np.linalg.norm(cb[t][:, None, :] - cb[t][None, :, :], axis=-1)
        near = np.argwhere((d_cb < prefilter_r) & (np.arange(R)[:, None]
                                                   < np.arange(R)[None, :]))
        P = xyz[t]
        for i, j in near:
            ai, aj = by_res[i], by_res[j]
            if ai.size == 0 or aj.size == 0:
                continue
            dd = np.linalg.norm(P[ai][:, None, :] - P[aj][None, :, :], axis=-1)
            m = float(dd.min())
            out[t, i, j] = out[t, j, i] = m
        out[t][np.diag_indices(R)] = 0.0
    return out


# ---------------------------------------------------------------------------
# dihedrals
# ---------------------------------------------------------------------------

def _dihedral(p0, p1, p2, p3):
    """Signed dihedral in radians for stacks of 4 points, shape ``(..., 3)``."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def dihedral_features(xyz: np.ndarray, table: AtomTable) -> np.ndarray:
    """``(T, R, 6)`` sin/cos of phi, psi and chi1 per residue.

    Angles are encoded as sin/cos pairs rather than radians because they are
    periodic: a residue flipping between -179 deg and +179 deg has barely moved,
    but the raw difference is 358 deg, which would dominate any variance-based
    method. Undefined angles (phi of the first residue, chi1 of glycine and
    alanine) are left at zero.

    Count-invariant: six numbers per residue regardless of residue identity.
    """
    R, T = table.n_residues, xyz.shape[0]
    N = table.one_per_residue("N", fallback=None)
    CA = table.one_per_residue("CA", fallback=None)
    C = table.one_per_residue("C", fallback=None)
    seq = table.residue_sequence()
    third = np.full(R, -1, dtype=np.int64)
    for a in range(len(table.atom_name)):
        r = table.res_index[a]
        want = CHI1_THIRD.get(str(seq[r]).strip())
        if want and table.atom_name[a].strip() == want:
            third[r] = a
    CB = table.one_per_residue("CB", fallback=None)

    out = np.zeros((T, R, 6), dtype=np.float32)

    def put(col, ang, rows):
        out[:, rows, 2 * col] = np.sin(ang).astype(np.float32)
        out[:, rows, 2 * col + 1] = np.cos(ang).astype(np.float32)

    # phi: C(i-1) N(i) CA(i) C(i)
    rows = np.array([r for r in range(1, R)
                     if min(C[r - 1], N[r], CA[r], C[r]) >= 0])
    if rows.size:
        put(0, _dihedral(xyz[:, C[rows - 1]], xyz[:, N[rows]],
                         xyz[:, CA[rows]], xyz[:, C[rows]]), rows)
    # psi: N(i) CA(i) C(i) N(i+1)
    rows = np.array([r for r in range(R - 1)
                     if min(N[r], CA[r], C[r], N[r + 1]) >= 0])
    if rows.size:
        put(1, _dihedral(xyz[:, N[rows]], xyz[:, CA[rows]],
                         xyz[:, C[rows]], xyz[:, N[rows + 1]]), rows)
    # chi1: N(i) CA(i) CB(i) X(i)
    rows = np.array([r for r in range(R)
                     if min(N[r], CA[r], CB[r], third[r]) >= 0])
    if rows.size:
        put(2, _dihedral(xyz[:, N[rows]], xyz[:, CA[rows]],
                         xyz[:, CB[rows]], xyz[:, third[rows]]), rows)
    return out


# ---------------------------------------------------------------------------
# burial
# ---------------------------------------------------------------------------

def burial(xyz: np.ndarray, table: AtomTable, *, r: float = BURIAL_R
           ) -> np.ndarray:
    """``(T, R)`` number of other residues' Cβ atoms within ``r`` angstrom.

    A cheap stand-in for solvent exposure, which matters here because the Y220C
    defect *is* a surface crevice opening. Counting Cβ atoms rather than all
    atoms keeps it invariant to the atom-count change a substitution causes.
    """
    cb = cb_coords(xyz, table)
    T, R = cb.shape[0], cb.shape[1]
    out = np.zeros((T, R), dtype=np.float32)
    for t in range(T):
        d = np.linalg.norm(cb[t][:, None, :] - cb[t][None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        out[t] = (d < r).sum(1)
    return out


# ---------------------------------------------------------------------------
# excluding the substituted sites
# ---------------------------------------------------------------------------

def mutation_mask(table: AtomTable, sites, *, neighbour_r: float = 0.0,
                  xyz: np.ndarray | None = None) -> np.ndarray:
    """``(R,)`` boolean, True for residues to **keep**.

    Args:
        sites: Author residue numbers of the substituted positions.
        neighbour_r: If > 0, also drop residues whose mean Cβ distance to any
            site is below this, which removes the mutation's direct steric
            footprint as well as the site itself.
        xyz: Required when ``neighbour_r > 0``.

    Why this exists: a representation that separates wild type from mutant
    *using the mutated residue itself* has rediscovered the sequence, which was
    never in doubt. The claim worth testing is that the substitution changes the
    ensemble elsewhere. Excluding the sites is what makes that claim falsifiable.
    """
    nums = table.residue_numbers()
    keep = ~np.isin(nums, np.asarray(list(sites)))
    if neighbour_r > 0:
        if xyz is None:
            raise ValueError("neighbour_r > 0 requires xyz")
        cb = cb_coords(xyz, table).mean(axis=0)
        site_rows = np.flatnonzero(np.isin(nums, np.asarray(list(sites))))
        if site_rows.size:
            d = np.linalg.norm(cb[:, None, :] - cb[None, site_rows, :],
                               axis=-1).min(axis=1)
            keep &= d >= neighbour_r
    return keep
