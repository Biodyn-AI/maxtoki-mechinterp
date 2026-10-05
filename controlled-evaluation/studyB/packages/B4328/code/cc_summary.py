"""Collect the three cc_geometry results into one table, and score decodability on the cells all three share.

Part 1 reads outputs/cc_geometry_{scgpt,geneformer,expr}.json (no new computation).
Part 2 re-scores the linear and kNN circ-R2 of all three representations on the 2,000 cells that are in every
file (the Geneformer cells), with the Geneformer file's phase label as the target. scGPT and expression rows are
taken from their own files for those cells. This takes a few seconds; no null draws are run.

Run from the package root:  python code/cc_summary.py [--out DIR]
Writes DIR/cc_summary.json and DIR/cc_summary.log (default DIR = outputs/).
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import ROOT, prep, emb_path, DIM  # noqa: E402
from cc_geometry import circ_r2, MODELS  # noqa: E402

LABEL = {"scgpt": "scGPT layer 11", "geneformer": "Geneformer layer 11", "expr": "expression (no model)"}


def main(out_dir):
    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    rows = []
    for m in MODELS:
        p = os.path.join(out_dir, f"cc_geometry_{m}.json")
        if not os.path.exists(p):
            log(f"missing {p}; run code/cc_geometry.py {m} first"); return
        g = json.load(open(p))
        rows.append(dict(model=m, label=LABEL[m], n=g["n"], input_dim=g["input_dim"],
                         lin_r2=g["linear_circ_r2"], knn_r2=g["knn_circ_r2"], gap=g["decodability_gap"],
                         out_of_plane_deg=g["real"]["out_of_plane_deg"],
                         out_of_plane_null=g["flat_null"]["out_of_plane_mean"],
                         out_of_plane_null_sd=g["flat_null"]["out_of_plane_sd"],
                         p_out_of_plane=g["p_out_of_plane"], Hflat_rejected=g["H_flat_rejected"]))

    log("Part 1. Each representation on its own cells (from cc_geometry_<model>.json)")
    log(f"  {'representation':<24}{'cells':>6}{'dim':>6}{'linear':>9}{'kNN':>8}{'kNN-lin':>9}"
        f"{'out-of-plane':>14}{'flat null':>14}{'p':>7}  flat circle rejected?")
    for r in rows:
        log(f"  {r['label']:<24}{r['n']:>6}{r['input_dim']:>6}{r['lin_r2']:>9.3f}{r['knn_r2']:>8.3f}"
            f"{r['gap']:>+9.3f}{r['out_of_plane_deg']:>12.1f} d"
            f"{r['out_of_plane_null']:>8.1f} +-{r['out_of_plane_null_sd']:.1f}{r['p_out_of_plane']:>7.3f}  "
            f"{'yes' if r['Hflat_rejected'] else 'no'}")

    gf = np.load(emb_path("geneformer"), allow_pickle=False)
    phi = gf["phi"].astype(np.float64)
    shared = []
    for m in MODELS:
        z = np.load(emb_path(m), allow_pickle=False)
        pos = {c: i for i, c in enumerate(z["cell_idx"])}
        idx = np.array([pos[c] for c in gf["cell_idx"]])
        Xz = prep(z["emb"][idx].astype(np.float64), DIM)
        shared.append(dict(model=m, label=LABEL[m], n=int(len(idx)),
                           lin_r2=circ_r2(Xz, phi, "linear"), knn_r2=circ_r2(Xz, phi, "knn")))
    log("")
    log(f"Part 2. Same {len(phi)} cells for all three (the Geneformer cells), Geneformer file's phase label")
    log(f"  {'representation':<24}{'linear':>9}{'kNN':>8}")
    for r in shared:
        log(f"  {r['label']:<24}{r['lin_r2']:>9.3f}{r['knn_r2']:>8.3f}")

    out = dict(own_cells=rows, shared_cells=shared)
    json.dump(out, open(os.path.join(out_dir, "cc_summary.json"), "w"), indent=1)
    open(os.path.join(out_dir, "cc_summary.log"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    main(ap.parse_args().out)
