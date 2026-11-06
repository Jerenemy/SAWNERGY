"""Stage G: does the MD the package was throwing away actually help?

Pass one fed the pipeline time-averaged CA-CA contact occupancy and nothing
else. Two mutations in this panel are side-chain substitutions, and one of them,
K417V, was effectively undetectable: separation d = 0.330, 145 replicates
required, the single contrast where supervised whitening *lost* to the
unsupervised autoencoder. Lys->Val deletes a long charged side chain while
leaving the backbone where it was, which is precisely the change a CA-only
representation cannot see. That is a hypothesis about the features, not the
model, and this stage tests it by holding the protocol fixed and varying only
what is measured.

Arms, each reduced to d = 32 and scored on runs held out of every fit:

    ca            CA-CA contact occupancy             -- the pass-one baseline
    cb            CB-CB contact occupancy
    sc            side-chain centroid contacts
    md            closest heavy-atom approach
    dih+bur       phi/psi/chi1 and CB burial          -- count-invariant
    ca+temporal   occupancy plus its fluctuation, autocorrelation and lifetime
    tica          slow modes of the per-frame dihedrals
    all           everything concatenated

Each arm runs twice. The **masked** arm removes residues 417 and 439 and their
spatial neighbours before any fitting. This is the control that decides what the
result means: a representation that tells wild type from mutant by looking at
the mutated residue has rediscovered the sequence, which was never in doubt,
and side-chain features are exactly the ones that can do this trivially -- the
substitutions change atom composition at those sites. The claim worth testing is
that the substitution reorganises the ensemble *elsewhere*, so the masked
numbers are the real result and the unmasked ones are the upper bound.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")

sys.path.insert(0, os.path.join(REPO, "src"))

from sawnergy.md import (leave_one_replicate_out,  # noqa: E402
                         replicates_required, separation_ratio,
                         seq_separation_mask)
from sawnergy.md.metrics import WTReference  # noqa: E402
from sawnergy.md.temporal import TICA  # noqa: E402

GFEAT = Path(DATA) / "gfeat"
OUT = Path(DATA) / "stage_g"
N_RES = 193
SEQ_SEP = 3
SITES = (417, 439)
#: A residue counts as a neighbour of a mutation site if its closest heavy-atom
#: approach to that site, averaged over wild-type runs, is under this.
NEIGHBOUR_A = 8.0
D_EMBED = 32
VAL_FRAC = 0.25
BASE = "N439K+K417V"
SYSTEMS = ("WT", "N439K", "K417V", BASE)

#: run id -> (label, expected residue 417, expected residue 439)
RUNS = {3: ("WT", "LYS", "ASN"), 0: ("N439K", "LYS", "LYS"),
        1: ("K417V", "VAL", "ASN"), 2: (BASE, "VAL", "LYS")}

PAIR_BLOCKS = ("ca_mean", "ca_fluct", "ca_ac", "ca_life", "cb_mean",
               "sc_mean", "md_mean")
RES_BLOCKS = ("bur_mean",)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load every cached run, validating the label against its own topology."""
    paths = sorted(GFEAT.glob("run*-clone*.npz"))
    cols: dict[str, list] = {}
    systems, clones, rejects = [], [], {}
    res_seq = None
    for p in paths:
        run_id = int(p.name.split("-")[0][3:])
        clone = int(p.name.split("clone")[1].split(".")[0])
        label, e417, e439 = RUNS[run_id]
        with np.load(p, allow_pickle=False) as z:
            probe = json.loads(str(z["probe"]))
            if probe.get("417") != e417 or probe.get("439") != e439:
                rejects[p.name] = (f"sequence mismatch 417={probe.get('417')} "
                                   f"439={probe.get('439')}")
                continue
            if res_seq is None:
                res_seq = z["res_seq"]
            elif not np.array_equal(res_seq, z["res_seq"]):
                rejects[p.name] = "residue numbering differs"
                continue
            for k in (*PAIR_BLOCKS, *RES_BLOCKS, "dih_mean"):
                cols.setdefault(k, []).append(z[k])
            cols.setdefault("dih_frames", []).append(z["dih_frames"])
        systems.append(label)
        clones.append(clone)
    blocks = {k: (np.stack(v) if k != "dih_frames" else v)
              for k, v in cols.items()}
    if rejects:
        print(f"  rejected {len(rejects)}: "
              f"{list(rejects.items())[:3]}")
    return blocks, np.array(systems), np.array(clones), res_seq


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------

def masks(blocks, systems, res_seq):
    """Residue keep-mask and the pair keep-masks it implies.

    Returns:
        (pair_keep_all, pair_keep_masked, res_keep, n_dropped)
    """
    iu = np.triu_indices(N_RES, k=1)
    seq_ok = seq_separation_mask(N_RES, SEQ_SEP)[iu]

    site_idx = [int(np.flatnonzero(res_seq == s)[0]) for s in SITES]
    # mean closest approach over wild-type runs, as a full matrix
    wt = systems == "WT"
    md = blocks["md_mean"][wt].mean(axis=0)
    M = np.zeros((N_RES, N_RES), dtype=np.float32)
    M[iu] = md
    M = M + M.T
    res_keep = np.ones(N_RES, dtype=bool)
    for s in site_idx:
        res_keep &= ~(M[s] < NEIGHBOUR_A)
        res_keep[s] = False
    pair_keep_masked = seq_ok & res_keep[iu[0]] & res_keep[iu[1]]
    return seq_ok, pair_keep_masked, res_keep, int((~res_keep).sum())


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def _standardise(Xtr, X):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (X - mu) / sd


def _select(Xtr, k=4000):
    """Keep the columns that vary most across training runs.

    A column with no between-run variance carries no information about system
    and only inflates the dimension the covariance has to be estimated in.
    """
    v = Xtr.std(0)
    if v.size <= k:
        return np.ones(v.size, dtype=bool)
    thr = np.partition(v, -k)[-k]
    return v >= thr


def build_arm(name, blocks, pair_keep, res_keep, tr, dih_tica):
    """Return the raw design matrix for one arm."""
    def pairs(key):
        return blocks[key][:, pair_keep]

    def res(key):
        return blocks[key][:, res_keep]

    def dih():
        # dih_mean is (n_runs, R*6); keep the surviving residues' six columns
        D = blocks["dih_mean"].reshape(len(blocks["dih_mean"]), N_RES, 6)
        return D[:, res_keep, :].reshape(len(D), -1)

    if name == "ca":
        return pairs("ca_mean")
    if name == "cb":
        return pairs("cb_mean")
    if name == "sc":
        return pairs("sc_mean")
    if name == "md":
        return pairs("md_mean")
    if name == "dih+bur":
        return np.hstack([dih(), res("bur_mean")])
    if name == "ca+temporal":
        return np.hstack([pairs("ca_mean"), pairs("ca_fluct"),
                          pairs("ca_ac"), pairs("ca_life")])
    if name == "tica":
        return dih_tica
    if name == "all":
        return np.hstack([pairs("ca_mean"), pairs("ca_fluct"),
                          pairs("ca_ac"), pairs("ca_life"),
                          pairs("cb_mean"), pairs("sc_mean"),
                          pairs("md_mean"), dih(), res("bur_mean"),
                          dih_tica])
    raise ValueError(name)


def fit_tica(frames, res_keep, tr, *, lag=5, pre=120, k=D_EMBED, seed=0):
    """Slow modes of the per-frame dihedrals, fitted on training runs only.

    The frame matrices are 1158-dimensional and TICA has to invert a covariance
    in that space, so the frames are first reduced by a PCA fitted on training
    frames. Without it the lagged covariance is rank-deficient and the whitening
    amplifies noise.
    """
    cols = np.repeat(res_keep, 6)
    Ftr = [np.asarray(frames[i], dtype=np.float32)[:, cols] for i in tr]
    pool = np.concatenate(Ftr)
    mu = pool.mean(0)
    Vt = np.linalg.svd(pool - mu, full_matrices=False)[2][:pre]
    runs_tr = [(F - mu) @ Vt.T for F in Ftr]
    tica = TICA.fit(runs_tr, lag=lag, n_components=k)
    Z = np.stack([tica.transform_run(
        (np.asarray(F, dtype=np.float32)[:, cols] - mu) @ Vt.T)
        for F in frames])
    return Z.astype(np.float32), tica


# ---------------------------------------------------------------------------
# scoring: identical to stage F, so the numbers are comparable
# ---------------------------------------------------------------------------

def metrics(Z, systems, label):
    disc = leave_one_replicate_out(Z, systems, n_permutations=0)
    seps = {f"WT|{b}": float(separation_ratio(Z, systems, "WT", b))
            for b in ("N439K", "K417V", BASE)}
    need = {k: float(replicates_required(v)) for k, v in seps.items()}
    rng = np.random.default_rng(0)
    wt = np.flatnonzero(systems == "WT")
    rng.shuffle(wt)
    h = wt.size // 2
    ref = WTReference.fit(Z[wt[:h]])
    D = {s: np.asarray(ref.distance(
            Z[wt[h:] if s == "WT" else np.flatnonzero(systems == s)])).ravel()
         for s in SYSTEMS}
    floor = float(D["WT"].mean())
    span = float(D[BASE].mean() - D["WT"].mean())
    auc = {s: float((D["WT"][:, None] < D[s][None, :]).mean())
           for s in ("N439K", "K417V", BASE)}
    print(f"  {label:26s} LORO {disc.accuracy:.4f}  "
          f"d(N439K) {seps['WT|N439K']:.3f}  d(K417V) {seps['WT|K417V']:.3f}  "
          f"d(double) {seps['WT|' + BASE]:.3f}  span/floor {span/floor:6.3f}  "
          f"AUC_K417V {auc['K417V']:.3f}")
    return {"dim": int(Z.shape[1]), "loro": float(disc.accuracy),
            "chance": float(disc.chance), "separations": seps,
            "replicates_required": need, "floor": floor, "span": span,
            "span_over_floor": span / floor, "auc": auc}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 110)
    print("STAGE G  all-atom and temporal features vs the CA-only baseline")
    print("=" * 110)
    blocks, systems, clones, res_seq = load()
    n = len(systems)
    counts = {s: int((systems == s).sum()) for s in SYSTEMS}
    print(f"\nloaded {n} runs: " + "  ".join(f"{k} {v}" for k, v in counts.items()))

    pair_all, pair_masked, res_keep, n_dropped = masks(blocks, systems, res_seq)
    print(f"pairs: {pair_all.sum()} unmasked, {pair_masked.sum()} with the "
          f"two mutation sites and their <{NEIGHBOUR_A:.0f} A neighbours "
          f"removed ({n_dropped} of {N_RES} residues dropped)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(round(VAL_FRAC * n))
    va, tr = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    print(f"split: {tr.size} train / {va.size} held out "
          f"(every fit below uses train only)")

    results: dict[str, dict] = {}
    for arm_label, pair_keep, rkeep in (
            ("unmasked", pair_all, np.ones(N_RES, dtype=bool)),
            ("masked", pair_masked, res_keep)):
        print("\n" + "-" * 110)
        print(f"{arm_label.upper()}"
              + ("  <- the result that is about the mutation's consequences"
                 if arm_label == "masked" else
                 "  <- upper bound; includes the substituted residues themselves"))
        print("-" * 110)
        Ztica, tica = fit_tica(blocks["dih_frames"], rkeep, tr)
        if arm_label == "masked":
            ts = tica.timescales()
            print(f"  TICA slowest implied timescales (frames): "
                  f"{np.round(ts[:5], 1)}")
        for name in ("ca", "cb", "sc", "md", "dih+bur", "ca+temporal",
                     "tica", "all"):
            X = build_arm(name, blocks, pair_keep, rkeep, tr, Ztica)
            sel = _select(X[tr])
            X = X[:, sel]
            Xs = _standardise(X[tr], X)
            mu = Xs[tr].mean(0)
            Vt = np.linalg.svd(Xs[tr] - mu, full_matrices=False)[2][:D_EMBED]
            Z = (Xs - mu) @ Vt.T
            results[f"{arm_label}/{name}"] = metrics(
                Z[va], systems[va], f"{name}  ({X.shape[1]}d -> {D_EMBED})")

    # --- the comparison that answers the question ---------------------------
    print("\n" + "=" * 110)
    print("DOES IT HELP?  change against the CA-only baseline, masked arm")
    print("=" * 110)
    base = results["masked/ca"]
    rows = []
    for name in ("cb", "sc", "md", "dih+bur", "ca+temporal", "tica", "all"):
        r = results[f"masked/{name}"]
        row = {"arm": name,
               "d_K417V": r["separations"]["WT|K417V"],
               "d_K417V_delta": r["separations"]["WT|K417V"]
               - base["separations"]["WT|K417V"],
               "loro_delta": r["loro"] - base["loro"],
               "need_K417V": r["replicates_required"]["WT|K417V"]}
        rows.append(row)
        print(f"  {name:14s} d(K417V) {row['d_K417V']:.3f} "
              f"({row['d_K417V_delta']:+.3f})   LORO {r['loro']:.4f} "
              f"({row['loro_delta']:+.4f})   replicates needed "
              f"{row['need_K417V']:6.0f}  (baseline "
              f"{base['replicates_required']['WT|K417V']:.0f})")
    best = max(rows, key=lambda r: r["d_K417V"])
    print(f"\n  best on the undetectable contrast: {best['arm']} at "
          f"d = {best['d_K417V']:.3f} vs {base['separations']['WT|K417V']:.3f} "
          f"for CA only")

    payload = {"n_runs": int(n), "counts": counts,
               "train": tr.tolist(), "val": va.tolist(),
               "neighbour_A": NEIGHBOUR_A, "n_residues_dropped": n_dropped,
               "n_pairs_unmasked": int(pair_all.sum()),
               "n_pairs_masked": int(pair_masked.sum()),
               "results": results, "comparison": rows}
    json.dump(payload, open(OUT / "stage_g.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'stage_g.json'}")


if __name__ == "__main__":
    main()
