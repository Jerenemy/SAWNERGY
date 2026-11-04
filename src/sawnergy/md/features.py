"""Frame and run featurisation.

Design constraints that matter for a WT-vs-mutant distance:

**Basis-free.** SAWNERGY's existing skip-gram embeddings are defined only up to
an arbitrary orthogonal transform -- which is why the codebase ships
`align_frames` (Orthogonal Procrustes). That is fine for visualising one
trajectory, but it makes a cross-system distance ill-posed: Procrustes will
always find *some* alignment, and fitting it minimises the very quantity you
are trying to measure. Every feature here is instead indexed by residue pair,
so two systems with the same residue numbering are directly comparable with no
alignment step and no learned basis.

**Superposition-free.** All features are internal coordinates (pairwise
distances), so they are invariant to global rotation and translation. No
reference frame has to be chosen, and no RMSD-fitting artefact can leak in.

**Structure separated from fluctuation.** Raw coordinate spread scales with
temperature for trivial kinetic reasons. If a model is handed only fluctuation
magnitude it can "detect temperature" without learning anything about folding.
`run_descriptor` therefore emits the time-averaged contact map and the contact
*fluctuation* as separate blocks so `sawnergy.md.validate` can test which block
actually carries the system signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .readers import Frames

_logger = logging.getLogger(__name__)

#: Midpoint of the logistic contact switch, in angstrom. 8 A is the
#: conventional CA-CA contact cutoff for residue interaction networks.
CONTACT_R0 = 8.0
#: Steepness of the switch. Softness matters: a hard cutoff makes the feature
#: discontinuous in the coordinates, which turns thermal jitter near the cutoff
#: into spurious high-variance channels.
CONTACT_BETA = 1.5
#: Residue pairs closer than this in sequence are dominated by backbone
#: connectivity and carry no tertiary information.
SEQ_SEPARATION = 3


def pair_distances(ca: np.ndarray) -> np.ndarray:
    """Pairwise residue distances.

    Args:
        ca: ``(T, N, 3)`` residue coordinates.

    Returns:
        ``(T, N, N)`` Euclidean distances.
    """
    diff = ca[:, :, None, :] - ca[:, None, :, :]
    return np.sqrt((diff ** 2).sum(-1) + 1e-12).astype(np.float32)


def contact_map(ca: np.ndarray, *, r0: float = CONTACT_R0,
                beta: float = CONTACT_BETA) -> np.ndarray:
    """Smooth contact probability in ``[0, 1]``.

    Uses a logistic switch ``1 / (1 + exp(beta * (d - r0)))`` rather than a hard
    cutoff so the feature is differentiable in the coordinates.
    """
    d = pair_distances(ca)
    # clip the exponent: far-apart residues otherwise overflow float32 exp and
    # emit a warning, though the result saturates to 0 either way
    return (1.0 / (1.0 + np.exp(np.clip(beta * (d - r0), -60.0, 60.0)))
            ).astype(np.float32)


def seq_separation_mask(n: int, min_sep: int = SEQ_SEPARATION) -> np.ndarray:
    """``(N, N)`` boolean mask keeping pairs at least ``min_sep`` apart."""
    idx = np.arange(n)
    return (np.abs(idx[:, None] - idx[None, :]) >= min_sep)


@dataclass
class RunDescriptor:
    """Fixed-length, residue-indexed summary of one run.

    Attributes:
        mean_contact: ``(P,)`` time-averaged contact probability per kept pair.
        contact_fluct: ``(P,)`` standard deviation of contact over time.
        rmsf: ``(N,)`` per-residue fluctuation about the run's own mean.
        pair_index: ``(P, 2)`` residue indices for each kept pair.
        run: Provenance.
    """

    mean_contact: np.ndarray
    contact_fluct: np.ndarray
    rmsf: np.ndarray
    pair_index: np.ndarray
    run: object

    def vector(self, blocks: tuple[str, ...] = ("mean_contact", "contact_fluct")) -> np.ndarray:
        """Concatenate the requested blocks into one feature vector."""
        parts = []
        for b in blocks:
            if b == "mean_contact":
                parts.append(self.mean_contact)
            elif b == "contact_fluct":
                parts.append(self.contact_fluct)
            elif b == "rmsf":
                parts.append(self.rmsf)
            else:
                raise ValueError(f"unknown block {b!r}")
        return np.concatenate(parts).astype(np.float32)


def run_descriptor(frames: Frames, *, r0: float = CONTACT_R0,
                   beta: float = CONTACT_BETA,
                   min_sep: int = SEQ_SEPARATION,
                   burn_in: float = 0.1) -> RunDescriptor:
    """Summarise one run into a fixed-length descriptor.

    Args:
        frames: One run's coordinates.
        r0, beta: Contact switch parameters.
        min_sep: Minimum sequence separation for a pair to be kept.
        burn_in: Fraction of leading frames to discard. Production trajectories
            relax out of their equilibrated state over the first few ns; those
            frames are correlated with the *starting structure*, which is shared
            across replicates and therefore suppresses exactly the
            between-replicate variance the statistics need to see.

    Returns:
        A `RunDescriptor`.
    """
    ca = frames.ca_coords()
    if not 0.0 <= burn_in < 1.0:
        raise ValueError("burn_in must be in [0, 1)")
    start = int(round(burn_in * ca.shape[0]))
    ca = ca[start:]
    if ca.shape[0] < 2:
        raise ValueError(f"{frames.run.uid}: fewer than 2 frames after burn-in")

    n = ca.shape[1]
    mask = seq_separation_mask(n, min_sep)
    iu = np.triu_indices(n, k=1)
    keep = mask[iu]
    pair_index = np.stack(iu, axis=1)[keep]

    c = contact_map(ca, r0=r0, beta=beta)          # (T, N, N)
    c_flat = c[:, iu[0], iu[1]][:, keep]           # (T, P)

    mean_contact = c_flat.mean(axis=0)
    contact_fluct = c_flat.std(axis=0)

    centroid = ca.mean(axis=0, keepdims=True)
    rmsf = np.sqrt(((ca - centroid) ** 2).sum(-1).mean(axis=0)).astype(np.float32)

    return RunDescriptor(mean_contact=mean_contact.astype(np.float32),
                         contact_fluct=contact_fluct.astype(np.float32),
                         rmsf=rmsf, pair_index=pair_index, run=frames.run)


def frame_tensor(frames: Frames, *, r0: float = CONTACT_R0,
                 beta: float = CONTACT_BETA, burn_in: float = 0.1,
                 channels: tuple[str, ...] = ("contact", "inv_distance")) -> np.ndarray:
    """Per-frame multi-channel representation ``X_t``.

    This is the ``X_t in R^{N x N x C}`` of the pipeline document, for the
    frame-level autoencoder.

    Returns:
        ``(T, N, N, C)`` float32.
    """
    ca = frames.ca_coords()
    start = int(round(burn_in * ca.shape[0]))
    ca = ca[start:]
    out = []
    d = pair_distances(ca)
    for ch in channels:
        if ch == "contact":
            out.append(1.0 / (1.0 + np.exp(beta * (d - r0))))
        elif ch == "distance":
            out.append(d)
        elif ch == "inv_distance":
            # bounded and emphasises the near field, where packing lives
            out.append(1.0 / (1.0 + d / r0))
        else:
            raise ValueError(f"unknown channel {ch!r}")
    return np.stack(out, axis=-1).astype(np.float32)
