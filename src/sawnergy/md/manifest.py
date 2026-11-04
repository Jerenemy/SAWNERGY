"""Corpus registry for MD trajectories.

The point of this module is to make the two mistakes that invalidate a
WT-vs-mutant comparison *structurally impossible to make silently*:

1. **n = 1 per condition.** A single trajectory per system cannot distinguish
   "these systems differ" from "these two runs differ". `Corpus.check()` refuses
   a corpus whose systems do not carry enough independent replicates, and
   nothing downstream will run without a passing check.

2. **Replicates that are not independent.** Frames from one trajectory are
   autocorrelated; two runs continued from the same equilibrated state with the
   same seed are the same run. A `Run` therefore has to declare its
   `independence` level, and the statistics treat anything weaker than
   `SOLVATION` as pseudo-replication.

The published study this program is chasing (Lee et al., p53-Y220C-MD-inputs)
simulated exactly our six systems at 277 K and 310 K and states plainly: "Each
system was simulated once." That is the failure mode this module exists to
prevent.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence


class Role(str, Enum):
    """What a system is *for* in the rescue comparison."""

    REFERENCE = "reference"   # wild type: defines the target dynamical state
    BASELINE = "baseline"     # untreated mutant: defines the deficit
    TREATED = "treated"       # mutant + compound: the thing being scored
    CONTROL = "control"       # negative/decoy compounds, unrelated mutants


class Independence(str, Enum):
    """How independent one replicate is from its siblings.

    Ordered weakest to strongest. Anything below SOLVATION shares structure with
    its siblings and must not be counted as an independent sample.
    """

    SEGMENT = "segment"        # chunks of ONE trajectory. NOT a replicate.
    VELOCITY = "velocity"      # same equilibrated box, fresh velocity seed.
    SOLVATION = "solvation"    # independent solvation + ions + equilibration.
    CONFORMER = "conformer"    # independent starting conformer as well.

    @property
    def rank(self) -> int:
        return {"segment": 0, "velocity": 1, "solvation": 2, "conformer": 3}[self.value]


#: Replicates weaker than this are pseudo-replication: they share a parent state
#: and understate between-replicate variance, which inflates every effect size.
MIN_INDEPENDENCE = Independence.VELOCITY

#: Below this many independent replicates per (system, temperature) cell, a
#: between-system difference cannot be separated from between-run noise with any
#: useful power. See `sawnergy.md.stats.replicates_required`.
MIN_REPLICATES = 3


@dataclass(frozen=True)
class Run:
    """One production trajectory.

    Attributes:
        system: System label, e.g. ``"Y220C"`` or ``"Y220C+rezatapopt"``.
        replicate: Index of this run within its (system, temperature) cell.
        temperature_K: Production thermostat setpoint.
        independence: How this replicate was decorrelated from its siblings.
        seed: RNG seed used to generate initial velocities, when known.
        path: Location of the trajectory, if materialised.
        topology: Location of the matching topology/parameter file.
        role: Role of the parent system in the comparison.
        n_frames: Frame count, when known.
        length_ns: Production length in nanoseconds, when known.
        meta: Free-form provenance (force field, water model, engine, ...).
    """

    system: str
    replicate: int
    temperature_K: float
    independence: Independence = Independence.VELOCITY
    seed: int | None = None
    path: Path | None = None
    topology: Path | None = None
    role: Role = Role.TREATED
    n_frames: int | None = None
    length_ns: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def cell(self) -> tuple[str, float]:
        """The (system, temperature) cell this run belongs to."""
        return (self.system, self.temperature_K)

    @property
    def uid(self) -> str:
        return f"{self.system}@{self.temperature_K:g}K#r{self.replicate}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["independence"] = self.independence.value
        d["role"] = self.role.value
        d["path"] = str(self.path) if self.path else None
        d["topology"] = str(self.topology) if self.topology else None
        return d


class CorpusError(ValueError):
    """Raised when a corpus cannot support the comparison being asked of it."""


@dataclass
class CorpusReport:
    """Result of `Corpus.check()`."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cells: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Corpus check: {'PASS' if self.ok else 'FAIL'}"]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        return "\n".join(lines)

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise CorpusError("\n".join(self.errors))


class Corpus:
    """A set of runs, grouped by system and temperature.

    Example:
        >>> c = Corpus([
        ...     Run("WT",    r, 310.0, role=Role.REFERENCE) for r in range(5)
        ... ] + [
        ...     Run("Y220C", r, 310.0, role=Role.BASELINE)  for r in range(5)
        ... ])
        >>> c.check().ok
        True
    """

    def __init__(self, runs: Iterable[Run] = ()) -> None:
        self._runs: list[Run] = list(runs)

    # ---------------------------------------------------------------- access
    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def __repr__(self) -> str:
        return (f"Corpus(runs={len(self._runs)}, systems={len(self.systems)}, "
                f"temperatures={sorted(self.temperatures)})")

    def add(self, run: Run) -> "Corpus":
        self._runs.append(run)
        return self

    @property
    def runs(self) -> list[Run]:
        return list(self._runs)

    @property
    def systems(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self._runs:
            seen.setdefault(r.system, None)
        return list(seen)

    @property
    def temperatures(self) -> list[float]:
        return sorted({r.temperature_K for r in self._runs})

    def role_of(self, system: str) -> Role:
        for r in self._runs:
            if r.system == system:
                return r.role
        raise KeyError(system)

    def by_role(self, role: Role) -> list[Run]:
        return [r for r in self._runs if r.role is role]

    def select(self, *, system: str | None = None,
               temperature_K: float | None = None,
               role: Role | None = None) -> list[Run]:
        out = self._runs
        if system is not None:
            out = [r for r in out if r.system == system]
        if temperature_K is not None:
            out = [r for r in out if r.temperature_K == temperature_K]
        if role is not None:
            out = [r for r in out if r.role is role]
        return list(out)

    def cells(self) -> dict[tuple[str, float], list[Run]]:
        d: dict[tuple[str, float], list[Run]] = defaultdict(list)
        for r in self._runs:
            d[r.cell].append(r)
        return dict(d)

    # ------------------------------------------------------------ validation
    def check(self, *, min_replicates: int = MIN_REPLICATES,
              min_independence: Independence = MIN_INDEPENDENCE,
              require_reference: bool = True) -> CorpusReport:
        """Verify the corpus can support a system-level comparison.

        This is gate G0. Everything downstream calls it.
        """
        errors: list[str] = []
        warnings: list[str] = []
        cells = self.cells()

        if not self._runs:
            return CorpusReport(False, ["corpus is empty"])

        # -- a reference system must exist, or "distance to WT" is undefined
        if require_reference:
            refs = {r.system for r in self.by_role(Role.REFERENCE)}
            if not refs:
                errors.append(
                    "no system has role=REFERENCE; the WT dynamical reference is "
                    "what every distance is measured against")
            elif len(refs) > 1:
                warnings.append(
                    f"{len(refs)} systems marked REFERENCE ({sorted(refs)}); the WT "
                    "distribution will pool them, which is only valid if they are "
                    "genuinely the same system")

        # -- replicate count per cell
        for (system, temp), runs in sorted(cells.items()):
            usable = [r for r in runs
                      if r.independence.rank >= min_independence.rank]
            n_pseudo = len(runs) - len(usable)
            if n_pseudo:
                warnings.append(
                    f"{system}@{temp:g}K: {n_pseudo} run(s) below "
                    f"independence={min_independence.value} are pseudo-replicates "
                    "and are excluded from replicate counts")
            if len(usable) < min_replicates:
                errors.append(
                    f"{system}@{temp:g}K: {len(usable)} independent replicate(s), "
                    f"need >= {min_replicates}. With n=1 a system difference is "
                    "not separable from run-to-run variance.")
            dup = [k for k, v in Counter(r.replicate for r in runs).items() if v > 1]
            if dup:
                errors.append(f"{system}@{temp:g}K: duplicate replicate ids {sorted(dup)}")

            seeds = [r.seed for r in usable if r.seed is not None]
            if len(seeds) != len(set(seeds)):
                errors.append(
                    f"{system}@{temp:g}K: repeated velocity seeds {sorted(seeds)}; "
                    "those runs are not independent")

        # -- the comparison must be balanced across temperature
        per_temp = defaultdict(set)
        for r in self._runs:
            per_temp[r.temperature_K].add(r.system)
        all_systems = set(self.systems)
        for temp, present in sorted(per_temp.items()):
            missing = all_systems - present
            if missing:
                warnings.append(
                    f"T={temp:g}K is missing {sorted(missing)}; temperature and "
                    "system are partially confounded, so a difference at this "
                    "temperature cannot be attributed cleanly")

        summary = {f"{s}@{t:g}K": len(v) for (s, t), v in sorted(cells.items())}
        return CorpusReport(not errors, errors, warnings, summary)

    # ------------------------------------------------------------------- io
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps([r.to_dict() for r in self._runs], indent=2))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "Corpus":
        raw = json.loads(Path(path).read_text())
        runs = []
        for d in raw:
            d = dict(d)
            d["independence"] = Independence(d.get("independence", "velocity"))
            d["role"] = Role(d.get("role", "treated"))
            d["path"] = Path(d["path"]) if d.get("path") else None
            d["topology"] = Path(d["topology"]) if d.get("topology") else None
            runs.append(Run(**d))
        return cls(runs)

    # -------------------------------------------------------------- builders
    @classmethod
    def grid(cls, systems: Sequence[tuple[str, Role]],
             temperatures: Sequence[float],
             n_replicates: int,
             *, independence: Independence = Independence.SOLVATION,
             seed0: int = 1000) -> "Corpus":
        """Build a full system x temperature x replicate grid.

        A balanced grid is what makes the two-way decomposition (system effect
        vs temperature effect vs replicate noise) identifiable.
        """
        runs, seed = [], seed0
        for sys_name, role in systems:
            for t in temperatures:
                for rep in range(n_replicates):
                    runs.append(Run(system=sys_name, replicate=rep,
                                    temperature_K=float(t), role=role,
                                    independence=independence, seed=seed))
                    seed += 1
        return cls(runs)
