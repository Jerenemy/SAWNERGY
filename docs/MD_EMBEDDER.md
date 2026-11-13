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

## 8. The nested symmetric autoencoder (plan §3–§5)

`sawnergy.md.autoencoder` implements the nested architecture with time-stamped
units:

```
X_1..X_T --E_f--> h_1..h_T --E_tau--> z --D_tau--> h_hat --D_f--> X_hat
```

Every frame enters the trajectory encoder carrying its own normalised
timestep, and the trajectory decoder is **queried by timestep**:
`D_tau(z, t) -> h_t_hat`. That is what makes the halves genuine mirrors — the
decoder reconstructs a specific indexed unit rather than emitting an
unlabelled bag — and it is why the stamp is a first-class input rather than a
positional afterthought. Layer widths are mirrored exactly (`D_f` reverses
`E_f`, `D_tau` reverses `E_tau`), and `tests/test_autoencoder.py` checks both
the mirroring and that the decoder's output actually changes with `t`.

Three losses, one per nesting level: frame-only, h-space, and end-to-end.
Training is the plan's three-stage schedule (frame AE → trajectory AE on frozen
embeddings → joint fine-tune), with validation on **held-out runs** after each
stage.

Note it is a plain autoencoder, not variational. The plan specifies "VAE" exactly
once in 15 pages, in §16's list of alternative *molecular generators* — never
for the MD embedder, which §3 and §4 call an autoencoder throughout. A KL term
would also work against this particular use: it shrinks latent distances toward
the prior, and distance between systems is the quantity being measured.

### 8.1 Benchmark result: it does not beat PCA — FAIL

`scripts/train_nested_ae.py`, per domain, fitting both the AE and the PCA
baseline on training runs only and scoring on held-out runs:

| Domain | separation PCA | separation AE | stage R² (S1/S2/S3) |
|---|---|---|---|
| 1balA00 | 1.62 | 0.65 | +0.35 / +0.31 / +0.02 |
| 1e8rA00 | 0.44 | 0.43 | +0.40 / +0.58 / +0.09 |
| 1ha8A00 | 0.16 | 0.32 | +0.29 / +0.58 / +0.02 |
| **mean** | **0.74** → 29 replicates | **0.47** → 72 replicates | |

Two configurations were tried: a large one (629k–700k parameters, stride 5) and
a smaller, regularised one (stride 1, dropout 0.1, weight decay 1e-3). Neither
beat the linear baseline, and tuning was stopped there rather than continued —
searching hyperparameters until the architecture wins is how one manufactures a
result.

### 8.2 The informative part: reconstruction improved while separation got worse

Between the two configurations the trajectory-level reconstruction **improved**
(S2 R² 0.02–0.31 → 0.31–0.58) while the separation ratio **degraded**
(0.72 → 0.47). The model got better at the objective it was trained on and
worse at the job it exists to do.

This is the plan's own §7 caveat — "reconstruction alone does not guarantee
that Euclidean or cosine distance in trajectory latent space corresponds to
biological similarity" — turned from a caution into a measurement. §7 Option A
says pure unsupervised reconstruction "should be tested first". It has now been
tested, and meaningful dynamical states did **not** emerge naturally.

So §7 Options B and C (contrastive learning, triplet loss) are no longer
optional refinements to try if reconstruction disappoints. They are the next
step, and the triplet form the plan already writes down is the right one:

```
d(z_WT1, z_WT2) < d(z_WT1, z_mutant)
```

with triplets built across **replicates** — two runs of the same system pulled
together, runs of different systems pushed apart. That objective optimises the
separation ratio directly, which is also the quantity that sets the replicate
budget.

### 8.3 Why it could not be rescued by tuning

15 training runs against 629k parameters. Train loss 0.43 versus validation
0.87 in the regularised configuration, and 0.29 versus 0.91–7.57 in the large
one. The frame autoencoder never cleared its R² ≥ 0.5 floor on held-out runs
(best 0.40).

The binding constraint is the number of independent **runs**, not frames and
not capacity. Windowing multiplies frames, but every window from one trajectory
carries that trajectory's idiosyncrasy, so it adds no information about
between-run variation — which is exactly what the trajectory latent has to
generalise over. This is the same shortage that §4.4 quantified from the other
direction, and it is the single thing most worth fixing before more modelling.

One consequence for the gate suite: with 2 held-out replicates per condition
the sham gate (G3) cannot run at all — it needs ≥ 4. Held-out evaluation of a
learned embedder therefore needs ≥ 6 replicates per cell just to stay testable.

---

## 9. Scaling the data: five experiments, one conclusion

Section 8 found the nested autoencoder losing to PCA and blamed too few
independent runs. That diagnosis was tested by raising the run count 16-fold
and then, when that was not enough, by changing the objective. Five regimes,
all scored by separation ratio on the hardest (adjacent-temperature) contrast,
against a PCA baseline fitted on the same runs.

### 9.1 More data

mdCATH holds 5,398 domains x 25 runs ~ 135,000 runs. Three domains had been
used. Fifty were downloaded (12 GB, **1,250 runs**).

The obstacle to using them was that pair-contact features have a width set by
the protein's residue count, so domains could not be pooled.
`features.patch_vectors` removes it: for each residue, the contact submatrix
over its *k* nearest neighbours in space (not in sequence -- the Y220C cleft
is lined by residues 145-157 and 220-230, adjacent in space and far apart in
sequence). Width depends only on *k*, so every protein shares a feature space.
Verified constant at 120 dims across domains of 50-57 residues.

### 9.2 Results

| # | Regime | Runs for trajectory encoder | S1 / S2 / S3 held-out R² | sep AE | sep PCA |
|---|---|---|---|---|---|
| 1 | no pretraining | 15 | +0.29 / +0.31 / +0.02 | 0.47 | 0.74 |
| 2 | frame encoder pretrained across 42 domains | 15 | +0.48…0.64 / **neg** / **neg** | 1.09 | 1.52 |
| 3 | both levels trained across 42 domains | **1,050** | **+0.76 / +0.54 / +0.42** | 0.63 | 0.73 |
| 4 | regime 3 + triplet loss | 1,050 | +0.78 / +0.53 / +0.44 | 0.66 | 0.73 |
| 5 | regime 4 + per-domain fine-tuning | 1,050 + 15 | — | 0.89 | 1.45 |

(PCA differs between rows because the evaluation sets differ — zero-shot rows
score all 25 runs, fine-tuned rows only held-out runs. Only the AE-vs-PCA
comparison *within* a row is meaningful. The AE loses every row.)

**Data scaling decisively fixed reconstruction.** Regime 2 -> 3 took the
trajectory stages from negative held-out R² to +0.54 and +0.42, measured on
200 runs from 8 domains the model never saw. The nested autoencoder genuinely
works as an autoencoder now, and generalises across proteins.

**It did not fix discrimination.** Separation stayed at or below the linear
baseline in all five regimes.

**The triplet objective helped, measurably, but not enough.** Regime 3 -> 4
left reconstruction unchanged while mean ICC on the latent rose from **0.116
to 0.299** and leave-one-run-out accuracy from 0.32 to 0.39. The metric
objective does what it claims; it simply does not close a gap this size.

**Fine-tuning made it worse** relative to the baseline (beats PCA on 1/8
domains rather than 4/8): with 15 runs, per-domain adaptation overfits while
PCA on the same 15 does not.

### 9.3 Why — and what it means for p53

The pooling that made cross-protein data usable is the same step that costs
the discriminative signal. `patch_frame_vectors` averages patch embeddings
over residues, so the frame representation knows *what kinds of local
environments are present* but not *which residue is where*. PCA on full pair
descriptors keeps residue identity throughout. For telling conditions apart in
one protein, residue identity is most of the signal.

So the generality that unlocked a 16-fold data increase is precisely what
makes the representation worse at the job. That is a real tension, not a
tuning failure, and it has a direct consequence:

**For p53, do not use cross-protein pretraining.** WT and Y220C are the same
protein with the same residue numbering, so the constraint that forced pooling
does not apply. Use full residue-pair descriptors directly, where every
contact keeps its identity. On everything measured here, **PCA over
mean-contact + contact-fluctuation descriptors is the best trajectory
representation available**, and the nested autoencoder should not be put in
front of it on the strength of architecture alone.

The autoencoder keeps its value where cross-protein transfer is the point --
reconstruction R² 0.42-0.76 on unseen proteins is a real result, and a
pretrained patch encoder is a reasonable featuriser. It is not currently the
right scoring stack for Y220C rescue.

### 9.4 Caveats

- All 50 domains are 50-57 residues, a consequence of selecting files by size.
  p53 DBD is ~200 residues; transfer to that scale is untested.
- The fine-tuned evaluation holds out 2 replicates per condition, so the sham
  gate **could not run** there (it needs >= 4). It passed at FPR 0.000 in the
  zero-shot evaluation, which keeps 5. This is the >= 6 replicates-per-cell
  floor from section 8.3 showing up again.
- Separation is measured on the single hardest contrast. The triplet gain is
  clearer in ICC, which uses all five conditions.

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
| Nested symmetric AE (time-stamped) | done, tested (15 tests); reconstructs well cross-protein, **still loses to PCA on discrimination — §8, §9** |
| Cross-protein patch features + pretraining | done; 50 domains / 1,250 runs |
| Triplet metric learning | done; ICC 0.116 → 0.299, insufficient alone |
| p53 trajectories | **not generated** — no MD engine installed |

Next, in order:

1. **Use PCA over full residue-pair descriptors as the p53 representation.**
   It wins every comparison run so far (§9.2) and needs no pretraining. Drop
   cross-protein pooling for p53, where residue numbering is shared (§9.3).
2. **Fix the thermal axis** — monotone/sigmoidal rather than linear, denser
   ladder (§4.3).
3. **Raise the replicate floor to ≥ 6 per cell**, below which held-out
   evaluation of a learned embedder is not even testable (§8.3).
4. Only then size the p53 campaign, from the separation the winning encoder
   actually achieves rather than from the extrapolation in §4.4.

---

## 10. Validation on a real WT-vs-mutant panel (SARS-CoV-2 RBD)

Sections 4 and 8 validated on mdCATH, where **temperature** stood in for the
mutation because no public MD dataset paired a wild type with point mutants at
replicate scale. One does: Folding@home project **17311**, the RBD:ACE2 complex
with glycosylation, AMBER14SB/TIP3P, 310 K NPT, frames every 0.5 ns, on the AWS
Open Data Registry under
`fah-public-data-covid19-antibodies/vir-collaboration/SARS-CoV-2-ACE2-RBD/`.

| FAH RUN | System | Role | Sequence check |
|---|---|---|---|
| run3 | WT | reference | LYS417 / ASN439 |
| run0 | N439K | baseline | LYS417 / **LYS**439 |
| run1 | K417V | baseline | **VAL**417 / ASN439 |
| run2 | N439K+K417V | baseline | **VAL**417 / **LYS**439 |

The RBD chain is **193 residues** against the p53 DBD's 195, and there are
thousands of velocity-independent clones per variant. 473 clones were pulled,
473 extracted cleanly, **444 usable** after requiring 180 frames past a 10%
burn-in: 113 WT, 107 N439K, 111 K417V, 113 double. Residue 417 and 439 identity
is read from each trajectory's own topology, so grouping does not rest on the
deposit's README.

Trajectory length is **independent of system** (F = 0.38, p = 0.77), as it must
be: FAH work-unit length depends on which volunteer's GPU took the job.

### 10.1 Representation: no patch pooling

The §9.3 diagnosis said patch pooling — needed to compare *different* proteins —
discards residue identity, which is most of the within-protein signal. With one
protein and one numbering there is no reason to pool, so features here are
**full residue-pair contact descriptors**, 18,145 pairs with |i−j| > 3.

Of those, only **2,049 vary between runs at all**; the other 16,096 are pairs
that never form a contact in any of 444 trajectories. Selecting on training runs
only (never on held-out runs, which would leak the difference being measured)
cuts the autoencoder's input width by 88.7% and its parameter count from 19M to
2.5M.

### 10.2 The gate ladder — PASS except effect size

| Gate | Result | |
|---|---|---|
| G1 features | 27.1% of 18,145 dims vary, 444 runs | PASS |
| G2 replicate noise | within-system run-to-centroid distance 1.187 | PASS |
| G3 sham | **FPR 0.025** over 40 sham splits (α = 0.05) | PASS |
| G4 separability | ICC 0.115, F = 15.4, p = 0.000999; LORO 0.815 vs chance 0.255 | PASS |
| G5 effect size | WT vs double **d = 0.36**; need 119 replicates, have 113 | FAIL |

G3 is the important one: the sham test now passes on a **real mutation** rather
than a temperature contrast.

### 10.3 The replicate requirement, confirmed without extrapolation

§4.4 extrapolated a temperature curve down to the Y220C-sized effect and got
*d ≈ 0.43 → ~87 replicates*, flagged as a lower bound resting on an assumption.
Measured here on two real point mutations: **d = 0.36 → ~121 replicates**.

Different protein, real substitutions, no sigmoid extrapolation anywhere — and
the same regime, slightly worse. Two point mutations move a protein's contact
ensemble by about **a third of its own run-to-run noise**. The replicate budget
in §4.4 was not an artifact of the temperature proxy.

### 10.4 The autoencoder finally does not lose — PASS (narrowly)

Evaluated on 111 held-out runs, split at run level:

| | LORO (held-out runs) | ICC(system) | AUC WT vs N439K | AUC WT vs K417V |
|---|---|---|---|---|
| PCA(32) | 0.856 | +0.204 | 0.956 (0.987 at d=12) | 0.870 (0.884 at d=16) |
| Nested AE(32) | 0.856 | +0.198 | **0.995** | **0.899** |

A dead tie on four-way classification; the AE ahead on per-run discrimination
along the rescue axis (d′ 3.83 vs 2.11). This is the first benchmark in which
the autoencoder is not beaten by PCA (contrast §8.1 and §9.2), and the only
thing that changed is dropping the cross-protein pooling.

Reconstruction and discrimination stayed **decoupled**, as in §8.2: validation
R² fell +0.46 → +0.37 → **+0.079** across the three training stages while
discrimination held. Reconstruction remains the wrong objective.

### 10.5 A calibration bug in `rescue_score` — FIXED

`rescue_score` computed `R = 1 − D_WT(z)/D_baseline`, which assumes a wild-type
run sits at `D = 0`. **It cannot.** Whitened residuals are standard normal, so a
Mahalanobis distance follows a chi distribution with `d` degrees of freedom,
mean `sqrt(2)·Γ((d+1)/2)/Γ(d/2) ≈ sqrt(d − 0.5)`.

Measured: in 32 dimensions the analytic floor is **5.613**; held-out WT runs sat
at **5.446** (PCA) and **4.928** (AE). The old formula therefore scored *wild
type itself* at **+0.39**, on a scale where 1.0 means fully restored.

The error is affine in `D`, so it never changed the **ranking** of two
compounds. It made every absolute statement wrong, and made scores from
embeddings of different dimension incomparable, because the compression factor
is `sqrt(d)/D_baseline`.

Fixed by anchoring both ends:

    R = (D_baseline − D_WT(z)) / (D_baseline − D_WT_floor)

- `expected_floor(d)` gives the analytic floor.
- `WTReference.fit_split` fits the covariance on half the WT runs and measures
  the floor on the other half. The floor **must** be out of sample: a run inside
  the fit sits closer to its own mean than a fresh run does, so an in-sample
  floor is biased low and inflates every score.
- `rescue_score` now takes `d_wt`, falls back to a calibrated `wt.floor`, then
  to the analytic floor with a warning, and **refuses** a baseline that does not
  exceed the floor — the case where there is no measurable deficit to rescue.

How it escaped: `test_rescue_score_bounds` asserted that `wt.mean` scores ≈ 1.
The mean is the single point where `D = 0`, so the test encoded the same wrong
assumption as the code. It now asserts that a *held-out WT run* scores ≈ 1, and
a regression guard checks the old form visibly undershoots.

### 10.6 The floor is the binding constraint

span / floor = **0.64** (PCA) and 0.47 (AE): the entire WT-to-double-mutant
signal is *smaller than the measurement floor*. Since the floor goes as
`sqrt(d)` while the signal does not, dimension is a tunable:

| d | 2 | 4 | 8 | **12** | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| PCA span/floor | 0.331 | 0.460 | 0.657 | **0.726** | 0.715 | 0.678 | 0.640 |
| PCA AUC (N439K) | 0.965 | 0.978 | 0.970 | **0.987** | 0.982 | 0.967 | 0.956 |

**Use d ≈ 12 for the PCA embedding, not 32.** For the autoencoder, truncation is
meaningless — its latent dimensions have no canonical ordering, and cutting to
d = 2 maximises span/floor (0.792) while destroying AUC (0.712). Use all 32.

### 10.7 Dose ordering fails, because the premise is wrong

The test "WT < single < double" **fails**: N439K alone sits further from WT
(12.54) than the double mutant carrying both substitutions (9.03). Additivity
check: `‖v₁+v₂‖` vs `‖v₁₂‖` ratio **0.40**, cosine **+0.49**. The double mutant
is not the vector sum of its parts — K417V partially compensates N439K. That is
epistasis, and nothing requires two substitutions to perturb a contact ensemble
additively, so the test cannot separate "the metric is wrong" from "the
assumption is wrong".

Replaced with a test that needs no additivity: project each run onto the
WT→double axis. PCA gives WT 0.000, **K417V +0.264**, N439K +0.969, double
1.000 — the partial variant lands cleanly between the anchors. For p53 this is
the better-posed test anyway: a compound either moves the mutant along the
measured deficit axis or it does not.

### 10.8 What this changes for p53

1. **Drop patch pooling.** Build the p53 representation from full residue-pair
   descriptors. §9.3 argued it; this measures it.
2. **Use d ≈ 12**, not 32, for a PCA embedding.
3. **Score with the two-anchor `R`**, and calibrate the floor on held-out WT
   replicates.
4. **Test rescue by projection onto the WT→Y220C axis**, not by total distance,
   and never assume additivity across perturbations.
5. **Budget ~120 replicates per system**, not 87. The extrapolation was
   optimistic by ~40%.
6. Still untested, and still the thing the programme rests on: whether any of
   these geometric quantities correlates with a measured ΔTm.

---

## 11. Is the ceiling the *marginals*? (stage H)

Stage G varied which quantity was measured — side-chain contacts, closest
heavy-atom approach, dihedrals, burial, contact kinetics, TICA — and none of
the eight arms beat time-averaged Cα–Cα contact occupancy on K417V, the one
contrast nothing in this programme has been able to see. It concluded that the
limit is the sampling depth rather than the representation.

That conclusion has a gap. Every one of those eight arms was a **marginal**: a
per-pair or per-residue average over frames. Marginals are blind to
correlation, and the blindness is not subtle. Take two contacts, A and B, each
formed half the time. One run flips between *both formed* and *neither
formed*; the other flips between *A only* and *B only*. The occupancy vector is
`[0.5, 0.5]` for both. They are visiting entirely different sets of shapes, and
no quantity of additional marginal features recovers the difference, because
the difference lives in the joint distribution and not in any one feature's
mean.

Stage H replaces the descriptor with a conformational **state population**
vector. Frames are clustered once into a shared set of discrete states and each
run is described by the fraction of its time spent in each. That is still
order-free — §2.1's argument stands, an order-aware descriptor would latch onto
the run-specific drift that distinguishes replicates — but it is a description
of the joint distribution rather than of each coordinate separately. It is also
the only descriptor in this package that is *interpretable*: "wild type spends
60% of its time in the closed-pocket state and the mutant 20%" is a sentence a
structural biologist can argue with, which the 18,000 anonymous contact
occupancies are not.

Two routes are tested, in `states` and in `vampnet`.

### 11.1 The classical route: TICA + k-means — FAIL

`sawnergy.md.states` does it the standard way: project onto the slow modes with
`TICA`, cluster the projections with k-means fitted on **training frames
only**, then score each run against those fixed centres. The clustering being
shared and fixed is what makes the descriptor open-set, and therefore usable
for a compound-bound trajectory that did not exist when the states were
defined. k-means is hand-rolled, as `TICA` and the stage scripts' PCA already
are, so this adds no dependency.

`scripts/fah_rbd_stage_h.py` **imports** stage G's loader, masking, train/held-out
split, column selection, standardisation, PCA and scoring rather than
re-implementing any of it, so the arms cannot drift from stage G's. It
reproduces stage G's masked Cα baseline exactly — d(K417V) 0.523, d(N439K)
0.552, LORO 0.6306 — which is the check that the protocol really is frozen. The
arm grid is pre-registered in the script's docstring and reported in full,
including the arms that lose; §8.3's note applies, and searching until the new
method wins is how one manufactures a result.

| Arm (masked) | d(WT vs K417V) | vs Cα | d(WT vs N439K) | LORO |
|---|---|---|---|---|
| **ca** — the baseline | 0.523 | — | 0.552 | 0.631 |
| msm100 | 0.517 | −0.007 | 0.621 | 0.595 |
| msm50 | 0.513 | −0.010 | 0.608 | 0.559 |
| msm50-raw (no Hellinger) | 0.512 | −0.012 | 0.607 | 0.559 |
| msm25 | 0.483 | −0.040 | 0.574 | 0.505 |
| msm50-hard | 0.431 | −0.093 | 0.434 | 0.523 |
| msm10 | 0.451 | −0.073 | 0.412 | 0.369 |
| msm10+trans | 0.334 | −0.189 | 0.320 | 0.306 |
| ca+msm50 | 0.525 | +0.001 | 0.559 | 0.631 |

**No state arm beats Cα occupancy.** The best, msm100, is 0.007 *below* it and
costs 0.036 of LORO. Concatenating the populations onto Cα moves d(K417V) by
+0.001 and LORO not at all, which is the informative version of the result: the
joint descriptor is not trading one kind of information for another, it is
adding nothing Cα did not already have.

Transition matrices are actively harmful — LORO 0.306, span/floor −0.015 — and
should be, since a 90-frame run at lag 1 offers 89 transitions to fill `k²`
entries, so anything past `k ≈ 10` is mostly prior. `populations` is the
descriptor to reach for; `transition_matrix` is in the package for completeness
and for non-equilibrium use, not for this.

Two findings that are not negative:

- **Soft assignment beats hard** by a wide margin: 0.513 against 0.431, LORO
  0.559 against 0.523. At fixed `T` a kernel-weighted membership is a
  lower-variance estimator of the same populations than a hard count, and that
  variance is the denominator of every separation ratio here. Hence
  `soft=True` is the default. The Hellinger transform, by contrast, makes
  essentially no difference (0.513 vs 0.512) and is left available rather than
  recommended.
- **The best arm in the stage is a dimension change, not a feature change.**
  Cα reduced to d = 25 gives d(K417V) 0.535 at identical LORO, and Cα at d = 10
  lifts d(N439K) from 0.552 to 0.850. This is §10.6's √d floor again — the
  floor grows with dimension while the signal does not — and it is free. It
  also means a k-state arm with k < 32 is confounded with a dimension change,
  which is why the `ca@d` controls exist and why any comparison against a
  10-state arm must use `ca@10` and not `ca`.

### 11.2 Why this null is different from stage G's

Stage G attributed its null to the sampling depth: 90 frames at 1 ns spacing.
The obvious reading of stage H is the same, since a population is a multinomial
proportion with standard error `sqrt(p(1-p)/T)`, and at `T = 90` with `k = 50`
that error is the same order as the population itself. `multinomial_noise`
exists to report exactly that.

Measured, it does not hold. The multinomial standard error is **0.22–0.27 of
the hard-assignment run-to-run scatter** across every `k` tried, in both the
masked and unmasked arms. Counting noise is a fifth of the between-run spread,
not the bulk of it. (The comparison has to be made against the *hard*
scatter: the closed form is the sampling noise of a count, and soft assignment
is a smoothed estimator of the same populations whose scatter is 25–30× smaller,
so quoting the count noise against the soft scatter overstates the problem by
whatever the smoothing buys. An earlier version of this diagnostic did exactly
that and read 5.7–8.4 instead of 0.22–0.27.)

So the denominator that defeats the state descriptor is **real
replicate-to-replicate ensemble heterogeneity**, not finite-sample counting
error. Two runs of the same system genuinely populate the conformational states
differently, by much more than their frame counts explain. That is a different
problem from stage G's, and a harder one: longer trajectories fix counting
noise, and do not obviously fix genuine heterogeneity between independent
replicates. What addresses it is a representation that is *built* to be
invariant to it — which is what §11.3 tests.

### 11.3 The learned route: VAMPnets — the first gain on K417V, and it does not quite clear the bar

The classical route is linear twice over: TICA finds directions of maximal
autocorrelation under a linear model, and k-means partitions them under an
isotropic-variance assumption. If the slow coordinate is a nonlinear function
of the dihedrals — a torsion flip, a pocket opening — TICA must approximate it
with a linear combination and the clustering cannot recover what was lost.

`sawnergy.md.vampnet` replaces both with one network mapping a frame to a soft
assignment over `k` states, trained to maximise the **VAMP-2 score** (Mardt et
al., *Nat. Commun.* 9:5, 2018): a variational bound on how much of the lag-τ
dynamics the assignment resolves. This is the first objective tried in this
package that is not a proxy for the quantity of interest. §8.2 records what a
reconstruction objective bought — reconstruction improved while separation got
*worse* — and PCA keeps tying because variance is the same wrong target. Slow
modes are not high-variance modes.

The network is fed the shared linear pre-reduction, not TICA's output: handing
it TICA's coordinates would concede in advance the step it exists to replace.
Everything downstream is the frozen protocol.

**What is deliberately absent is a system-label objective.** Stage G measured
why: masking the substituted residues and their 8 Å shell costs Cα LORO 0.883 →
0.631, so most of the apparent WT-vs-mutant signal *is* the substituted
residue. A contrastive loss trained on system labels would find that feature,
because it is perfectly predictive and trivially learnable — and it is the one
feature a compound can never change, so every such model would score every
compound as unrescued. VAMP-2 never sees a label; it sees pairs of frames from
one trajectory, so its objective is available for a compound-bound run that did
not exist at training time. The descriptor stays open-set.

Three disciplines, fixed before the numbers: a pre-registered grid (`k` ∈
{10, 50}, three seeds, one architecture, no search); every fit reported as
mean ± sd across seeds, since a neural result from one seed is not a
measurement; and early stopping on an inner split of the **training** runs, so
the 111 held-out runs take part in no fitting decision, including when to stop.

| Arm (masked) | d(K417V) | d(N439K) | LORO | AUC_K417V |
|---|---|---|---|---|
| **ca** (d = 32) | 0.523 | 0.552 | 0.631 | **0.891** |
| ca@10 (d = 10) | 0.518 | 0.850 | 0.478 | 0.864 |
| msm50 (TICA + k-means) | 0.513 | 0.608 | 0.559 | 0.773 |
| vamp10 (d = 10) | **0.770** ± 0.051 | 0.862 ± 0.258 | 0.577 ± 0.117 | 0.848 ± 0.035 |
| vamp50 (d = 32) | 0.710 ± 0.041 | 0.819 ± 0.042 | **0.649** ± 0.024 | 0.846 ± 0.009 |

**The K417V gain is real and is not a dimension artifact.** Against the
dimension-matched baseline, vamp10 is +0.252 over `ca@10` and vamp50 is +0.186
over `ca`. The separation is the quantity that sets the budget — `n ≈ 16/d²` —
so 0.523 → 0.710 is 58 replicates down to 32, and → 0.770 is 27.

The dimension control earns its place here, because it kills half the result.
**The N439K gain is almost entirely a dimension effect**: `ca@10` reaches 0.850
for free, above both VAMPnet arms, so nothing about the learned decomposition
is responsible for it. K417V is the opposite — `ca` at d = 10 gives 0.518 and
at d = 25 gives 0.535, so the dimension does nothing there, and the VAMPnet's
0.71–0.77 is genuinely beyond what any reduction of Cα occupancy achieves. This
is the first arm in the programme to move the contrast that nothing could move.

**It does not clear a conventional bar.** All arms are scored on the same 111
held-out runs, so the comparison is paired and the uncertainty on the
*difference* is far smaller than the ≈ 0.28 standard error on a single `d`; a
paired bootstrap over held-out runs (2,000 resamples, both arms rescored on each
resample) is the honest version of the question:

| Arm | seed | Δd(K417V) | 95% CI | p(Δ ≤ 0) |
|---|---|---|---|---|
| vamp10 vs ca@10 | 0 | +0.285 | [+0.087, +0.498] | 0.0040 |
| | 1 | +0.213 | [+0.002, +0.456] | 0.0250 |
| | 2 | +0.187 | [−0.020, +0.404] | 0.0405 |
| vamp50 vs ca | 0 | +0.140 | [−0.064, +0.349] | 0.0865 |
| | 1 | +0.209 | [−0.005, +0.433] | 0.0300 |
| | 2 | +0.144 | [−0.050, +0.350] | 0.0760 |

Every seed of both arms is positive, which is itself evidence, but only two of
six have a confidence interval excluding zero and vamp50's worst seed sits at
p = 0.087. The status is **suggestive, not established**. Reporting it as a
win would be the same error §8.1 declined to make in the other direction.

There is also a cost that does not go away: **AUC falls** — 0.846–0.848 against
Cα's 0.891 — while `d` rises. This is the third appearance of that pattern, after
stage F's within-class whitening, and the consistent reading is that these
methods tighten replicate scatter, which is `d`'s denominator, without adding
discriminative information. Cheaper replicates, not better ranking. Compound
scoring is a ranking problem, so the thing being improved is not quite the thing
that is needed.

Two practical notes from building it. The VAMP-2 score is invariant to any
invertible linear map of the assignment, so it pins down the slow *subspace* and
says nothing about scale: a near-uniform softmax whose small variations point
the right way scores exactly as well as a sharp partition. The standardisation
in the stage pipeline removes the ambiguity, but it means the interpretability
§11 claims for a clustered state model — "60% of the time in the closed state" —
does **not** transfer to the learned one. And a softmax starts near-uniform,
which is a dead local optimum of the score: the covariance is near rank-one and
the gradient through the whitening nearly vanishes. Roughly one fit in six
collapsed there even with a trace-relative ridge, so `fit_vampnet` restarts
`n_init` times and keeps the best by inner-validation score, the same remedy
`kmeans` applies to the same class of problem, and `VampReport.collapsed`
flags the case explicitly — a collapsed fit and a signal-free process both score
zero, and only one of those is a statement about the data.

### 11.4 What stage H changes

1. **Drop the state-population descriptor as a replacement for Cα occupancy.**
   Clustered populations lose on every contrast, and adding them to Cα adds
   nothing (+0.001 on K417V, LORO unchanged).
2. **Take the dimension for free.** Cα at d = 25 beats Cα at d = 32 on K417V at
   identical LORO, and d = 10 lifts N439K from 0.552 to 0.850. Any future arm
   must be compared against a dimension-matched baseline or the comparison is
   meaningless.
3. **Keep the VAMPnet, and settle it.** It is the first thing to move K417V
   beyond what dimension explains, at `p` between 0.004 and 0.087 depending on
   seed. More seeds and a held-out set larger than 111 runs would settle it;
   both are cheap compared with what is being decided.
4. **Do not train on system labels.** The open-set requirement and stage G's
   masking result both forbid it, independently.
5. **The denominator remains the target.** Three methods have now raised `d` by
   tightening replicate scatter without raising AUC. The quantity that
   compound ranking needs has not moved since stage B.
