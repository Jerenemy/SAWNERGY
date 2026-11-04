# Trajectory embedding and WT-restoration metrics

Implementation of sections 1–13 of *SAWNERGY: Refined End-to-End Research
Pipeline*, with three substantive departures from the plan and one measured
result that changes what the programme can claim.

Everything here lives in `sawnergy.md`. Tests: `tests/test_md.py` (30, all
passing). Validation scripts: `scripts/validate_on_mdcath.py`,
`scripts/replicate_power_curve.py`.

---

## 1. Why the existing embedder cannot support a WT-vs-mutant distance

`sawnergy.embedding.Embedder` trains a skip-gram model **per frame**.
Skip-gram embeddings are identified only up to an arbitrary orthogonal
transform, and each frame is trained from its own initialisation with its own
negative samples, so every frame lands in a different basis. The repository
already acknowledges this by shipping `align_frames`, an Orthogonal Procrustes
alignment.

That is a reasonable patch for visualising one trajectory. It cannot support a
cross-system distance, because Procrustes always finds *some* alignment and
fitting it minimises the very quantity being measured.

So the plan's frame/trajectory autoencoder is not a refinement — **it is what
makes the distance well-defined at all.** One encoder is trained across all
frames of all systems, every embedding lands in one space by construction, and
no alignment step is involved. `sawnergy.md.encoder` implements this.

## 2. Three departures from the plan

### 2.1 Set pooling, not LSTM/GRU/Transformer (plan §4)

The plan proposes recurrent or attention temporal encoders. For an
**equilibrium** ensemble that is the wrong inductive bias and is actively
dangerous here: a converged trajectory is a sample from a stationary
distribution, so frame ordering is near-arbitrary, and what characterises the
system is the distribution over frames. An order-aware encoder has the capacity
to latch onto run-specific drift — which is precisely the between-replicate
variation everything else is trying to suppress.

`TrajectorySetEncoder` pools order-invariantly (mean, sd, quantiles) by
default. `order_aware=True` remains available for genuinely non-stationary
processes such as forced unfolding, and warns when used.

### 2.2 Thermal shift in kelvin, not a dimensionless ratio (plan §12)

The plan's rescue score is `R = 1 − D_WT(z) / D_WT(z_Y220C)`. It depends on an
arbitrary latent scale, is undefined if the untreated mutant happens to sit near
WT, and cannot be compared against any experiment. It is implemented
(`rescue_score`), but it should not be the headline metric.

The Y220C defect is **thermal**. Wild-type p53 DBD melts at ~41.6 °C and Y220C
at ~33.2 °C; body temperature (37 °C) sits *between* the two. Nothing is broken
at the DNA interface — Tyr220 is not a DNA-contacting residue — the domain
simply unfolds at the temperature it has to work at.

So the natural question is not "how far is this trajectory from WT in latent
units" but **"what temperature does this system behave as if it were at?"** Fit
the WT embedding against temperature, invert it, and report

```
dT_apparent = T_apparent − T_actual
```

positive for a destabilised system, measured in kelvin, directly comparable to
the `dTm` that differential scanning fluorimetry reports — which is how this
field actually measures Y220C rescue. Compound rescue is
`dT_apparent(mutant) − dT_apparent(mutant + compound)`, also in kelvin.

This also yields a falsifiable prediction that doubles as a validation gate:
Y220C at temperature *T* should embed near WT at *T* + 8.4 K.

**Status: implemented, and it fails its own validation gate — see §4.3.**

### 2.3 Temperature is a design axis, not a nuisance variable

The relevant published p53 study ran 277 K and 310 K. Two levels cannot
distinguish a thermal trend from the difference between two conditions, so
`ThermalAxis.fit` refuses fewer than three temperatures. `corpora.T_LADDER`
uses five (310/330/350/370/400 K) plus 277 K as a negative control: if WT and
Y220C are indistinguishable at 277 K but separate at 310 K, that is evidence
the representation tracks thermal stability rather than a static structural
difference near the mutation site.

Spontaneous unfolding is not reachable at 310 K on any affordable timescale,
which is why the ladder extends upward and why mdCATH runs 320–450 K.
`ThermalAxis.extrapolation_K` reports how far outside the fitted range any
reading falls, and `ThermalShift.__str__` flags extrapolated values.

## 3. Not confusing a system difference with MD noise

This is enforced structurally, not by convention.

**The unit of analysis is the run, never the frame.** Frames within a
trajectory are autocorrelated on the ns timescale; treating them as independent
inflates *n* by orders of magnitude and will report p < 1e-30 for two runs of
the *same* system. Every function in `sawnergy.md.stats` takes one vector per
run, and `replicate_splits` groups by run.

**n = 1 is refused.** `Corpus.check()` (gate G0) rejects a corpus whose systems
lack replicates, `variance_components` raises on single-run systems,
`WTReference.fit` raises on a single WT run, and `thermal_shift` raises on a
single replicate. `Independence` distinguishes real replicates from segments of
one trajectory, and duplicate velocity seeds are an error.

**Permutations shuffle run labels**, and in `gate_disentangled` they shuffle
system labels *within* each temperature, so a significant system effect cannot
be explained by temperature.

**The sham gate (G3).** Within each real system, replicates are randomly
relabelled into two fake sub-systems and the full separability test is run.
Those halves differ only by MD noise, so the false-positive rate must sit near
alpha. This is the gate that answers "is the difference real or just MD".

### The gate ladder

| Gate | Checks | Fatal |
|---|---|---|
| G0 corpus | enough independent replicates | yes |
| G1 features | finite, varying | yes |
| G2 replicate noise | within-system scatter (reported, not thresholded) | no |
| G3 sham | fake conditions inside one system must NOT separate | yes |
| G4 separability | real systems separate, leave-one-run-out + permutation | yes |
| G5 effect size | separation large vs replicate scatter; design has power | yes |
| G6 baseline | learned encoder beats PCA | no |
| G7 temperature | monotone thermal response, axis fittable | yes |
| G8 disentangled | system signal survives conditioning on temperature | yes |

---

## 4. Validation on real MD, before any p53 compute

No public p53 DBD trajectories exist. Checked: **ATLAS** (1,938 chains) has no
p53 DBD — the nearest entries are 50-residue p73 tetramerisation fragments.
**mdCATH** (5,398 CATH domains) has no p53 domain. The one directly relevant
study publishes inputs only, 20 ns per run, and states *"Each system was
simulated once."*

So the embedder was validated on **mdCATH**, where within one domain
**temperature plays exactly the role the mutation plays in p53**: a condition
that changes the dynamics, sampled with several independent replicates, with
ground truth known. Three domains (1balA00, 1e8rA00, 1ha8A00), 5 temperatures ×
5 replicates each, 75 runs.

### 4.1 The statistics are correctly calibrated — PASS

| Check | Result |
|---|---|
| G3 sham false-positive rate | **0.000** over 40 splits, all 4 feature blocks, all 3 domains |
| ICC(replicate) | −0.07 to −0.13, p = 0.42–0.60 |

Two halves of one condition's replicates never came out significant, and
replicate identity carries no signal. The machinery does not manufacture
differences.

### 4.2 Real condition differences are detected — PASS

| Domain | ICC(temperature) | p | LORO accuracy (chance 0.20) |
|---|---|---|---|
| 1balA00 | 0.600 | 0.0005 | 0.64 |
| 1e8rA00 | 0.599 | 0.0005 | 0.56 |
| 1ha8A00 | 0.341 | 0.0005 | 0.24 |

**RMSF fails as a feature** (ICC −0.19 to 0.23, LORO at chance). The signal is
in contact *structure*, not fluctuation magnitude — which is reassuring, since
a representation that merely read off "how much is it moving" would separate
temperatures for trivial kinetic reasons.

### 4.3 The thermal-shift metric does NOT yet work — FAIL

Holding out one temperature entirely, refitting on the rest, and trying to
recover it:

| Held out | 1balA00 | 1e8rA00 | 1ha8A00 |
|---|---|---|---|
| 320 K | +3.7 K | +29.5 K | +36.4 K |
| 348 K | −15.1 K | −12.4 K | −11.5 K |
| 379 K | +57.1 K | −7.9 K | −21.9 K |
| 413 K | +17.5 K | −0.9 K | +20.8 K |
| 450 K | −47.2 K | +81.9 K | +677.8 K |

The deficit to be measured is **8.4 K**. The method's own held-out error is
**10–80 K**. `dT_apparent` is therefore implemented but **not validated**, and
must not be used to rank compounds in its current form.

Likely causes, in order of suspicion: contact loss versus temperature near a
melting transition is **sigmoidal, not linear**, so a linear axis is simply the
wrong model; five widely-spaced rungs with five replicates is too thin to fit
anything better; and PCA over contact descriptors is a weak embedder. The fix
is a monotone/sigmoidal axis with a denser ladder — not more parameters
elsewhere.

### 4.4 The replicate requirement is far higher than anyone's practice

30 temperature-pair contrasts across 3 domains, separation ratio measured for
each (`scripts/replicate_power_curve.py`):

| Contrast | Median separation | Replicates for 80 % power |
|---|---|---|
| 28–40 K | 0.78 replicate-sd | **26** |
| 40–70 K | 1.44 | 8 |
| 70–100 K | 1.63 | 6 |
| 100–200 K | 2.06 | 4 |

Extrapolating the fitted line to a Y220C-sized effect:

| Effect | Separation | Replicates per system |
|---|---|---|
| **8.4 K (Y220C dTm)** | ~0.43 | **~87** |
| 15 K | ~0.54 | ~54 |
| 28 K (smallest measured) | ~0.76 | ~27 |

This is an extrapolation below the smallest measured contrast and assumes the
mutation perturbs contacts as much as an equivalent temperature change, so read
it as a **lower bound**. Even so: the published n = 1 design is roughly two
orders of magnitude short, and the customary 3–5 replicates is one order short.

`DEFAULT_REPLICATES = 20` is set as a deliberately uncomfortable default — well
above common practice, still below what the extrapolation demands — so that any
result obtained with it has to be reported with its measured effect size and
power rather than a bare p-value.

---

## 5. What this costs, and the problem that creates

`estimate_cost` on the 9-system corpus, 200 ns per run, ~150 ns/day for a
solvated DBD on one GPU:

| Design | Runs | Aggregate | GPU-days |
|---|---|---|---|
| Full ladder (6 T × 20 replicates) | 1,080 | 216 µs | 1,440 |
| 310 K only, 20 replicates | 180 | 36 µs | 240 |
| 310 K only, 5 replicates | 45 | 9 µs | 60 |

**The design the evidence demands does not fit the available compute.** That is
a finding, not a failure of the plan. The options, honestly:

1. **Raise separation instead of n.** The replicate count is set by the
   separation ratio, which depends on the *descriptor*. Lifting separation at
   the 28 K contrast from 0.78 to 1.4 cuts the requirement from 26 to 8. This
   is now a concrete, measurable benchmark for the learned encoder — and the
   first thing it should be asked to do.
2. **Cut the system list before cutting replicates.** Four well-powered systems
   beat nine underpowered ones; the latter cannot support any claim.
3. **Enhanced sampling** (REST2, replica exchange) buys the thermal axis what
   brute-force MD cannot at 310 K.
4. **Accept a bounded claim.** With 5 replicates the honest output is a ranking
   with wide error bars, not a `dTm` estimate.

---

## 6. The p53 corpus

`corpora.p53_corpus()` returns the manifest. Nine systems:

| System | Role | Why |
|---|---|---|
| WT | reference | defines the target dynamical state |
| Y220C | baseline | defines the deficit |
| Y220C+rezatapopt | treated | clinical positive control (9S9O) |
| Y220C+PK9320 | treated | carbazole stabiliser (6GGC) |
| Y220C+PhiKan083 | treated | weak fragment (2VUK) — tests gradation |
| Y220C+KG13 | treated | covalent Cys220 adduct (8DC8); should score highest |
| Y220C+decoy | control | binds, does not stabilise — **the critical negative** |
| Y220C+inert | control | non-binder |
| R175H | control | a different structural mutant |

The decoy is the one that matters most. If a cleft binder with no reported
stabilisation scores as rescuing, the metric has collapsed back into an
affinity proxy — which is the failure the whole dynamical programme exists to
escape.

Conformers come from the structures already prepared for the generative work,
so both halves of the programme share inputs.

---

## 7. Status

| Component | State |
|---|---|
| Corpus manifest + design checks | done, tested |
| mdCATH reader | done, verified on 3 domains |
| Contact featuriser | done, tested |
| Replicate-aware statistics | done, tested, **validated on real MD** |
| Validation gate suite | done, tested, **validated on real MD** |
| `WTReference`, `rescue_score` | done, tested |
| `ThermalAxis`, `thermal_shift` | implemented, **fails validation (§4.3)** |
| PCA baseline embedder | done |
| Frame autoencoder | implemented, **not yet benchmarked against baseline** |
| p53 trajectories | **not generated** — no MD engine installed |

Next, in order: fix the thermal axis (monotone fit, denser ladder); benchmark
the autoencoder against PCA on the mdCATH 28 K contrast, targeting separation
> 1.4; then size the p53 campaign from the separation the winning encoder
achieves, rather than from the extrapolation in §4.4.
