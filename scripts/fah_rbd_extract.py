"""Stream FAH PROJ17311 clones: download -> extract RBD CA coords -> delete.

Keeps disk use to one trajectory at a time. Writes one .npz per clone with
CA coordinates in angstrom, the residue numbering, the time axis, and the
identity of the two mutated positions so provenance can be validated later.
"""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FAH_RBD_DIR", "/home/jzay/fah_rbd")
import json, os, sys, subprocess, h5py, numpy as np

BUCKET = ("https://fah-public-data-covid19-antibodies.s3.amazonaws.com/"
          "vir-collaboration/SARS-CoV-2-ACE2-RBD/munged/solute/PROJ17311")
AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
          "THR TRP TYR VAL HID HIE HIP CYX ASH GLH LYN".split())
RBD_CHAIN = 0
PROBE = (417, 439, 501)


def rbd_ca_indices(topology_json):
    t = json.loads(topology_json)
    chain = next(c for c in t["chains"] if c["index"] == RBD_CHAIN)
    idx, resseq, names = [], [], []
    for r in chain["residues"]:
        if r["name"] not in AA3:
            continue
        ca = [a for a in r["atoms"] if a["name"] == "CA"]
        if not ca:
            continue
        idx.append(ca[0]["index"])
        resseq.append(r["resSeq"])
        names.append(r["name"])
    return np.array(idx), np.array(resseq), names


def one(run, clone, outdir, tmpdir, stride):
    out = os.path.join(outdir, f"run{run}-clone{clone}.npz")
    if os.path.exists(out):
        return "skip"
    tmp = os.path.join(tmpdir, f"r{run}c{clone}.h5")
    url = f"{BUCKET}/run{run}-clone{clone}.h5"
    rc = subprocess.run(["curl", "-sfL", "--max-time", "900", "-o", tmp, url]).returncode
    if rc != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "missing"
    try:
        with h5py.File(tmp, "r") as f:
            idx, resseq, names = rbd_ca_indices(f["topology"][0].decode())
            n_frames = f["coordinates"].shape[0]
            sl = slice(None, None, stride)
            xyz = f["coordinates"][sl][:, idx, :] * 10.0      # nm -> angstrom
            time = f["time"][sl]
        probe = {str(p): (names[list(resseq).index(p)] if p in resseq else "?")
                 for p in PROBE}
        np.savez_compressed(out, xyz=xyz.astype(np.float32), resseq=resseq,
                            time=time, n_frames_full=n_frames,
                            probe=json.dumps(probe))
        return f"ok {xyz.shape[0]}f {probe['417']}417/{probe['439']}439"
    except Exception as e:
        return f"bad {type(e).__name__}: {e}"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    runs = [int(x) for x in sys.argv[1].split(",")]
    clones = range(int(sys.argv[2]), int(sys.argv[3]))
    outdir, tmpdir = sys.argv[4], sys.argv[5]
    stride = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    os.makedirs(outdir, exist_ok=True); os.makedirs(tmpdir, exist_ok=True)
    for c in clones:
        for r in runs:
            print(f"run{r}-clone{c}: {one(r, c, outdir, tmpdir, stride)}", flush=True)
