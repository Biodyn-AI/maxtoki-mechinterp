"""compare_lines - does the RPE1 cell-cycle phase label agree with the K562 one?

For each cell line (3,000 non-targeting control cells each):
  1. phase angle per cell from cc_phase.phase_angle,
  2. for every gene in the Whitfield et al. 2002 peak-phase list, its peak phase in that line: the
     expression-weighted circular mean of the cell phases, with weight = expression above the gene's own
     mean (clipped at 0).
The two lines are then compared gene by gene on the genes present in both panels: circular correlation
(Jammalamadaka & SenGupta) and the median absolute angular difference.

Run from the package root:
    python code/compare_lines.py            # writes outputs/
    python code/compare_lines.py --out DIR  # writes to another folder
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle, line_checks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = {"k562": "data/k562_controls.h5ad", "rpe1": "data/rpe1_controls.h5ad"}

# Whitfield et al. 2002 (synchronised HeLa) peak phase, degrees, G1/S boundary at 0.
WHITFIELD_PEAK = {
    "CCNE1": 0, "CCNE2": 0, "E2F1": 10, "PCNA": 20, "CDC6": 15, "CDT1": 15, "MCM2": 10, "MCM3": 10,
    "MCM4": 10, "MCM5": 10, "MCM6": 10, "MCM7": 10, "SLBP": 30, "RRM2": 45, "TYMS": 45, "DHFR": 40,
    "CDC45": 35, "GINS2": 40, "CLSPN": 35, "FEN1": 50, "RFC4": 45, "POLA1": 40,
    "CCNA2": 120, "TOP2A": 170, "NUSAP1": 175, "UBE2C": 200, "BIRC5": 205, "TPX2": 190,
    "CENPF": 185, "CENPE": 200, "CDK1": 180, "CCNB1": 210, "CCNB2": 210, "PLK1": 205,
    "BUB1": 195, "BUB1B": 195, "AURKA": 205, "AURKB": 215, "KIF11": 185, "KIF23": 210,
    "ANLN": 200, "ECT2": 195, "CDC20": 220, "PTTG1": 215, "CKS2": 190, "AURKAIP1": 200,
}


def circ_corr_deg(a, b):
    a, b = np.radians(a), np.radians(b)
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    return float(np.sum(sa * sb) / np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)))


def peak_phase(ad, theta, genes):
    import scipy.sparse as sp
    vn = np.char.upper(np.asarray(ad.var_names).astype(str))
    gi = {g: i for i, g in enumerate(vn)}
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    z = np.exp(1j * theta)
    out = {}
    for g in genes:
        if g not in gi:
            continue
        x = X[:, gi[g]].astype(float)
        x = np.clip(x - x.mean(), 0, None)
        if x.sum() <= 0:
            continue
        out[g] = float(np.degrees(np.angle((x * z).sum() / x.sum())) % 360)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()
    import anndata
    os.makedirs(args.out, exist_ok=True)
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    res = {"lines": {}}
    peaks = {}
    for tag, rel in LINES.items():
        ad = anndata.read_h5ad(os.path.join(ROOT, rel))
        theta, s, g2m, info = phase_angle(ad)
        chk = line_checks(theta, s, g2m)
        res["lines"][tag] = dict(n_cells=int(ad.n_obs), n_genes=int(ad.n_vars), **info, **chk)
        peaks[tag] = peak_phase(ad, theta, WHITFIELD_PEAK)
        say(f"{tag}: {ad.n_obs} cells x {ad.n_vars} genes | cc genes {info['n_cc_genes']} | "
            f"PC1+PC2 var {info['pc12_var_ratio']:.3f} | phase R {chk['phase_R']:.3f} | "
            f"S peak {chk['s_peak_deg']:.1f} deg, G2M peak {chk['g2m_peak_deg']:.1f} deg "
            f"(separation {chk['s_g2m_separation_deg']:.1f}) | Whitfield genes with a peak {len(peaks[tag])}")

    genes = sorted(set(peaks["k562"]) & set(peaks["rpe1"]))
    a = np.array([peaks["k562"][g] for g in genes])
    b = np.array([peaks["rpe1"][g] for g in genes])
    d = np.abs(a - b) % 360
    d = np.minimum(d, 360 - d)
    res["n_genes_compared"] = len(genes)
    res["genes_not_compared"] = sorted(set(WHITFIELD_PEAK) - set(genes))
    res["circ_corr_k562_vs_rpe1"] = round(circ_corr_deg(a, b), 4)
    res["median_abs_diff_deg"] = round(float(np.median(d)), 1)
    res["mean_abs_diff_deg"] = round(float(np.mean(d)), 1)
    say(f"compared {len(genes)} genes (not compared: {', '.join(res['genes_not_compared'])})")
    say(f"circ_corr K562 vs RPE1 per-gene peak phase {res['circ_corr_k562_vs_rpe1']:+.3f} | "
        f"median |diff| {res['median_abs_diff_deg']:.0f} deg | mean |diff| {res['mean_abs_diff_deg']:.0f} deg")

    with open(os.path.join(args.out, "phase_agreement.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, "gene_peak_phase.csv"), "w") as f:
        f.write("gene,k562_peak_deg,rpe1_peak_deg,abs_diff_deg\n")
        for g, x, y, dd in zip(genes, a, b, d):
            f.write(f"{g},{x:.1f},{y:.1f},{dd:.1f}\n")
    with open(os.path.join(args.out, "run_log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
