"""Trajectory readers that yield coordinates plus replicate/temperature labels.

A reader's job is to hand the rest of the pipeline frames *together with the
provenance needed to group them*. Grouping is not cosmetic here: every
statistic in `sawnergy.md.stats` is computed over replicates, so a reader that
loses track of which frames came from which run silently destroys the only
thing that makes the comparison valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .manifest import Corpus, Independence, Role, Run

_logger = logging.getLogger(__name__)


@dataclass
class Frames:
    """Coordinates for one run, with its labels attached.

    Attributes:
        coords: ``(T, N, 3)`` coordinates in angstrom.
        resid: ``(N,)`` 0-based residue index per atom.
        names: ``(N,)`` atom names, when available.
        run: The `Run` these frames belong to.
        rmsf: Optional ``(n_residues,)`` reference RMSF from the source.
    """

    coords: np.ndarray
    resid: np.ndarray
    run: Run
    names: np.ndarray | None = None
    rmsf: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_residues(self) -> int:
        return int(self.resid.max()) + 1 if self.resid.size else 0

    def ca_coords(self) -> np.ndarray:
        """``(T, n_residues, 3)`` residue representative coordinates.

        Uses CA where atom names are known, otherwise the residue centroid.
        Falling back to a centroid keeps the featurizer usable on sources that
        do not ship atom names, at the cost of a slightly softer contact map.
        """
        if self.names is not None:
            is_ca = np.asarray([n.strip() == "CA" for n in self.names])
            if is_ca.sum() == self.n_residues:
                return self.coords[:, is_ca, :]
            _logger.debug("CA count %d != residue count %d; using centroids",
                          int(is_ca.sum()), self.n_residues)
        nres = self.n_residues
        T = self.coords.shape[0]
        out = np.zeros((T, nres, 3), dtype=np.float32)
        counts = np.bincount(self.resid, minlength=nres).astype(np.float32)
        for d in range(3):
            summed = np.zeros((T, nres), dtype=np.float32)
            np.add.at(summed.T, self.resid, self.coords[:, :, d].T)
            out[:, :, d] = summed / np.maximum(counts, 1)
        return out


class MdCathReader:
    """Reader for mdCATH per-domain HDF5 files.

    mdCATH stores ``<domain>/<temperature>/<replicate>/coords``, i.e. exactly
    the 5-temperature x 5-replicate design this pipeline needs to validate
    against. It is used here as the *method* validation corpus: it contains no
    p53, but it is the only large public MD set where the replicate and
    temperature structure is known by construction, so it is where the
    embedder's claims can be falsified before any p53 compute is spent.

    Reference:
        Mirarchi et al., mdCATH, Scientific Data (2024). CC-BY-4.0.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    # ------------------------------------------------------------------ meta
    def _open(self):
        import h5py  # local import: h5py is an optional dependency
        return h5py.File(self.path, "r")

    @property
    def domain(self) -> str:
        with self._open() as f:
            keys = [k for k in f.keys()]
        if len(keys) != 1:
            raise ValueError(f"expected one domain group in {self.path}, got {keys}")
        return keys[0]

    def temperatures(self) -> list[float]:
        """Temperature groups, i.e. every child key that is a number.

        mdCATH files carry non-trajectory members alongside the temperature
        groups (``chain``, ``element``, ``pdb``, ``pdbProteinAtoms``, ...), and
        the exact set varies between releases, so filter positively on "parses
        as a number" rather than against a fixed deny-list.
        """
        out = []
        with self._open() as f:
            for k in f[self.domain].keys():
                try:
                    out.append(float(k))
                except (TypeError, ValueError):
                    continue
        if not out:
            raise ValueError(f"no temperature groups found in {self.path}")
        return sorted(out)

    def replicates(self, temperature: float) -> list[int]:
        with self._open() as f:
            return sorted(int(r) for r in f[self.domain][f"{temperature:g}"].keys())

    @staticmethod
    def _atom_names_from_pdb(text: str) -> np.ndarray:
        return np.asarray([l[12:16] for l in text.splitlines()
                           if l.startswith(("ATOM", "HETATM"))])

    def _resid(self, f) -> tuple[np.ndarray, np.ndarray | None]:
        """Per-atom residue index and atom names.

        Prefers the file's own ``resid`` dataset. Falling back to parsing the
        embedded PDB is only a last resort, and must use ``pdbProteinAtoms``
        rather than ``pdb``: the latter is the full solvated system (8,266
        atoms for a 639-atom protein in the reference file), so parsing it
        yields an atom count that does not match ``coords`` at all.
        """
        g = f[self.domain]

        names = None
        if "pdbProteinAtoms" in g:
            raw = g["pdbProteinAtoms"][()]
            text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            names = self._atom_names_from_pdb(text)

        if "resid" in g:
            raw = np.asarray(g["resid"][()]).ravel()
            # compact to contiguous 0-based indices, preserving order
            _, inv = np.unique(raw, return_inverse=True)
            order = np.argsort(np.argsort([np.where(raw == v)[0][0]
                                           for v in np.unique(raw)]))
            resid = order[inv].astype(np.int32)
            return resid, names

        if names is None:
            raise ValueError(
                f"{self.path}: no 'resid' dataset and no 'pdbProteinAtoms' "
                "record; cannot map atoms to residues")

        raw = g["pdbProteinAtoms"][()]
        text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
        order_map: dict[tuple, int] = {}
        seq: list[int] = []
        for line in text.splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            key = (line[21], line[22:27])          # chain + resseq/icode
            if key not in order_map:
                order_map[key] = len(order_map)
            seq.append(order_map[key])
        if not seq:
            raise ValueError("no ATOM records parsed from pdbProteinAtoms")
        return np.asarray(seq, dtype=np.int32), names

    # ------------------------------------------------------------------ read
    def read(self, temperature: float, replicate: int, *,
             stride: int = 1, max_frames: int | None = None,
             role: Role = Role.CONTROL, system: str | None = None) -> Frames:
        """Load one replicate.

        Args:
            temperature: Thermostat setpoint, as stored (e.g. 320).
            replicate: Replicate index.
            stride: Keep every ``stride``-th frame. Striding reduces temporal
                autocorrelation; it does **not** create independent samples.
            max_frames: Optional cap after striding.
            role: Role to tag the synthesised `Run` with.
            system: System label; defaults to the domain id.
        """
        dom = self.domain
        with self._open() as f:
            grp = f[dom][f"{temperature:g}"][str(replicate)]
            coords = np.asarray(grp["coords"][()], dtype=np.float32)
            rmsf = np.asarray(grp["rmsf"][()], dtype=np.float32) if "rmsf" in grp else None
            resid, names = self._resid(f)

        coords = coords[::stride]
        if max_frames is not None:
            coords = coords[:max_frames]

        run = Run(system=system or dom, replicate=int(replicate),
                  temperature_K=float(temperature),
                  # mdCATH replicates are spawned from one equilibrated system
                  # with fresh velocities -- real replicates, but not
                  # independently solvated.
                  independence=Independence.VELOCITY,
                  seed=None, path=self.path, role=role,
                  n_frames=int(coords.shape[0]),
                  meta={"source": "mdCATH", "domain": dom,
                        "forcefield": "CHARMM22*", "stride": stride})
        return Frames(coords=coords, resid=resid, names=names, run=run, rmsf=rmsf)

    def iter_runs(self, *, temperatures: list[float] | None = None,
                  stride: int = 1, max_frames: int | None = None,
                  role: Role = Role.CONTROL,
                  system: str | None = None) -> Iterator[Frames]:
        for t in (temperatures or self.temperatures()):
            for r in self.replicates(t):
                yield self.read(t, r, stride=stride, max_frames=max_frames,
                                role=role, system=system)

    def corpus(self, *, temperatures: list[float] | None = None,
               role: Role = Role.CONTROL, system: str | None = None) -> Corpus:
        """Build the `Corpus` describing this file without loading coordinates."""
        runs = []
        dom = self.domain
        for t in (temperatures or self.temperatures()):
            for r in self.replicates(t):
                runs.append(Run(system=system or dom, replicate=int(r),
                                temperature_K=float(t),
                                independence=Independence.VELOCITY,
                                path=self.path, role=role,
                                meta={"source": "mdCATH", "domain": dom}))
        return Corpus(runs)
