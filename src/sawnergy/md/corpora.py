"""Concrete MD corpora: what to simulate, at what temperatures, how many times.

Two corpora are defined.

`method_validation_corpus` uses mdCATH, which contains no p53 at all. That is
deliberate. mdCATH is the only large public MD set whose replicate and
temperature structure is known by construction (5 replicates x 5 temperatures
per domain), so it is where the embedder's claims can be *falsified* before any
p53 compute is spent. In particular the sham gate (G3) needs many replicates of
a system known to be identical to itself -- mdCATH supplies that for 5,398
domains at zero simulation cost.

`p53_corpus` is the p53 collection proper. It has to be generated: a survey of
the public resources found none of these trajectories available.

    - ATLAS (1,938 chains, 3 replicates, 100 ns, 300 K): no p53 DNA-binding
      domain. Closest entries are p73 tetramerisation fragments (50 residues).
    - mdCATH (5,398 CATH domains): no p53 domain.
    - The one directly relevant study (Lee et al., "P53 Mutant Structures
      Predicted by AlphaFold and Molecular Dynamics Simulations of Candidate
      Stabilizing Compounds") simulated exactly the six systems below, at 277 K
      and 310 K -- but publishes inputs only, 20 ns per run, and states: "Each
      system was simulated once." n = 1 per cell is the failure this package
      exists to prevent, so those runs could not be used as a reference even if
      the trajectories were released.

So the p53 corpus is a simulation plan, not a download. `p53_corpus()` returns
the manifest; `estimate_cost` says what it costs.
"""

from __future__ import annotations

from pathlib import Path

from .manifest import Corpus, Independence, Role, Run

# ---------------------------------------------------------------------------
# temperature ladder
# ---------------------------------------------------------------------------

#: Physiological temperature. The Y220C deficit is defined here: wild-type p53
#: DBD melts at ~41.6 C (314.8 K) and Y220C at ~33.2 C (306.4 K), so 310 K sits
#: *between* the two melting points. This is the temperature the phenotype
#: lives at and it must be in the ladder.
T_PHYSIOLOGICAL = 310.0

#: Below both melting points. Included as a negative control: if WT and Y220C
#: are indistinguishable at 277 K but separate at 310 K, that is evidence the
#: representation is tracking thermal stability rather than a static structural
#: difference near the mutation site.
T_COLD = 277.0

#: Elevated rungs. Spontaneous unfolding is not reachable at 310 K on any
#: affordable timescale, so the thermal axis has to be anchored where unfolding
#: actually happens and then read back down. This is why mdCATH runs 320-450 K.
#: Three rungs is the minimum for an identifiable (and invertible) axis; five
#: is the minimum for a credible one.
T_LADDER = (310.0, 330.0, 350.0, 370.0, 400.0)

#: Full ladder used by `p53_corpus` when `thermal=True`.
T_FULL = (T_COLD,) + T_LADDER


# ---------------------------------------------------------------------------
# the p53 systems
# ---------------------------------------------------------------------------

#: (label, role, note). Built around the structures already prepared for the
#: generative run, so the same conformers feed both halves of the program.
P53_SYSTEMS: tuple[tuple[str, Role, str], ...] = (
    ("WT",               Role.REFERENCE,
     "Wild-type p53 DBD. Defines the target dynamical state. Multiple runs are "
     "mandatory: the WT distribution, not a WT trajectory, is the reference."),
    ("Y220C",            Role.BASELINE,
     "Untreated Y220C. Defines the deficit; every rescue score is a ratio "
     "against this, so it needs as many replicates as WT."),
    ("Y220C+rezatapopt", Role.TREATED,
     "Clinical positive control (PDB 9S9O conformer). The pipeline must rank "
     "this as rescuing, or the metric is wrong."),
    ("Y220C+PK9320",     Role.TREATED,
     "Carbazole stabiliser, PDB 6GGC. Second positive control, weaker."),
    ("Y220C+PhiKan083",  Role.TREATED,
     "Original fragment hit, PDB 2VUK. Weak stabiliser: tests gradation, not "
     "just binary rescue."),
    ("Y220C+KG13",       Role.TREATED,
     "Covalent Cys220 adduct, PDB 8DC8. The only chemotype reported to restore "
     "Tm to near wild-type, so it should score highest of the treated set."),
    ("Y220C+decoy",      Role.CONTROL,
     "A Y220C-cleft binder with no reported stabilisation. The critical "
     "negative: binding without rescue must score ~0, or the metric has "
     "collapsed back into an affinity proxy."),
    ("Y220C+inert",      Role.CONTROL,
     "A non-binder at the same concentration. Must score ~0."),
    ("R175H",            Role.CONTROL,
     "A structural p53 mutant that is NOT Y220C. Tests that the metric "
     "measures the Y220C deficit specifically rather than 'any mutation'."),
)


#: Replicates per cell. NOT a guess -- measured on real MD.
#: `scripts/replicate_power_curve.py` fits separation ratio against contrast
#: size across 30 temperature-pair contrasts in 3 mdCATH domains. A 28 K
#: contrast (the smallest available) gives a median separation of 0.78
#: replicate-sd, needing ~26 replicates for 80% power. Extrapolated to the
#: Y220C-sized effect (dTm ~ 8.4 K) the requirement is ~87 replicates per
#: system. 20 is set here as a deliberately uncomfortable default: it is far
#: above common practice (the relevant published study used n = 1) and still
#: below what the extrapolation demands, so any result obtained with it must
#: be reported with its measured effect size and power, not just a p-value.
DEFAULT_REPLICATES = 20


def p53_corpus(*, n_replicates: int = DEFAULT_REPLICATES, thermal: bool = True,
               temperatures=None, root: str | Path | None = None,
               independence: Independence = Independence.SOLVATION) -> Corpus:
    """The p53 simulation plan.

    Args:
        n_replicates: Independent replicates per (system, temperature) cell.
            See `DEFAULT_REPLICATES` -- the measured power curve says a
            Y220C-sized effect needs far more than the 3-5 that is customary.
            Three is only the hard floor the corpus check enforces, not a
            defensible design.
        thermal: Use the full temperature ladder (needed for `ThermalAxis`).
            With ``False`` only 310 K is simulated, which supports the plain
            `rescue_score` but **not** the thermal-shift metric.
        temperatures: Override the ladder.
        root: Directory trajectories will be written to.
        independence: How replicates are decorrelated. SOLVATION (independent
            solvation, ions and equilibration per replicate) is strongly
            preferred over VELOCITY: velocity-only replicates share an
            equilibrated box and understate between-replicate variance, which
            inflates every effect size computed later.

    Returns:
        A `Corpus` whose runs have ``path=None`` until generated.
    """
    temps = tuple(temperatures) if temperatures is not None else (
        T_FULL if thermal else (T_PHYSIOLOGICAL,))
    root = Path(root) if root else None

    runs: list[Run] = []
    seed = 20260101
    for label, role, note in P53_SYSTEMS:
        for t in temps:
            for rep in range(n_replicates):
                path = (root / label / f"{t:g}K" / f"rep{rep}" / "prod.xtc"
                        if root else None)
                runs.append(Run(
                    system=label, replicate=rep, temperature_K=float(t),
                    role=role, independence=independence, seed=seed, path=path,
                    meta={"note": note,
                          "engine": "GROMACS",
                          "forcefield": "AMBER ff19SB / GAFF2 for ligands",
                          "water": "OPC",
                          "target_ns": 200},
                ))
                seed += 1
    return Corpus(runs)


def method_validation_corpus(h5_files, *, role: Role = Role.CONTROL) -> Corpus:
    """Corpus over downloaded mdCATH domain files.

    Each domain becomes one "system"; its 25 runs (5 temperatures x 5
    replicates) supply the replicate structure the gates need.
    """
    from .readers import MdCathReader

    runs: list[Run] = []
    for p in h5_files:
        runs.extend(MdCathReader(p).corpus(role=role).runs)
    return Corpus(runs)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

def estimate_cost(corpus: Corpus, *, ns_per_run: float = 200.0,
                  ns_per_day: float = 150.0, n_gpus: int = 1) -> dict:
    """Rough wall-clock estimate for generating a corpus.

    Args:
        ns_per_run: Production length per replicate.
        ns_per_day: Throughput for a ~50k-atom solvated DBD on one modern GPU.
            An RTX 3090 with GROMACS sits around 100-200 ns/day for a system
            this size; 150 is a working midpoint.
        n_gpus: Parallel GPUs.

    Returns:
        dict with run count, aggregate microseconds and GPU-days.
    """
    n = len(corpus)
    total_ns = n * ns_per_run
    gpu_days = total_ns / ns_per_day
    return {"n_runs": n, "total_us": total_ns / 1000.0,
            "gpu_days": gpu_days, "wall_days": gpu_days / max(n_gpus, 1),
            "ns_per_run": ns_per_run, "ns_per_day": ns_per_day}
