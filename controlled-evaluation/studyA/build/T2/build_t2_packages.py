"""Build the Study A task T2 packages (CRISPRi direction from MaxToki-217M SAE circuit edges).

Writes:
  studyA/tasks/T2-paper/     BRIEF.md, data/, methods/source_method_paper.pdf
  studyA/tasks/T2-contract/  the same, plus contract/SPEC.md
The two data/ and methods/ folders are byte-identical (checked at the end). The two BRIEF.md
files differ only by the one contract sentence (checked at the end).

Sources (read only, never modified):
  runs/circuit-tracing-217M/outputs/v2_circuit/task_data/   edges, top genes, records, silenced genes
  runs/circuit-tracing-217M/outputs/v2_circuit/lfc/         LFC matrix, var symbols, control mean, sources
  references/2603.01752_Kendiukhov_causal_circuit_tracing.pdf
  pipelines/sparse-autoencoders/02-causal-circuit-tracing.md (deployed spec, file dated 2026-04-15), verbatim
CPU only. No model is run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

EVAL = Path("<EVAL_ROOT>")
BIOMI = Path("<REPO_ROOT>")
V2 = BIOMI / "projects/maxtoki/runs/circuit-tracing-217M/outputs/v2_circuit"
TD = V2 / "task_data"
LFC = V2 / "lfc"
PDF = BIOMI / "references/2603.01752_Kendiukhov_causal_circuit_tracing.pdf"
SPEC = BIOMI / "pipelines/sparse-autoencoders/02-causal-circuit-tracing.md"
BUILD = EVAL / "studyA/build/T2"
PKG = {"paper": EVAL / "studyA/tasks/T2-paper", "contract": EVAL / "studyA/tasks/T2-contract"}
CONTRACT_SENTENCE = "A method specification for this kind of analysis is provided in contract/SPEC.md; follow it.\n\n"
MARK = "<<CONTRACT_SENTENCE>>\n"

COPY = {  # package name <- task_data name (byte copies)
    "circuit_edges.csv": "circuit_edges.csv",
    "source_features.tsv": "source_features.tsv",
    "feature_top_genes.tsv": "edge_feature_genes.tsv",
    "silenced_genes.tsv": "silenced_genes.tsv",
    "gene_pairs.parquet": "records.parquet",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def build_data(out: Path, log: dict):
    out.mkdir(parents=True, exist_ok=True)
    for dst, src in COPY.items():
        shutil.copyfile(TD / src, out / dst)
    # LFC matrix restricted to the silenced genes of the package, in silenced_genes.tsv order
    sg = pd.read_csv(TD / "silenced_genes.tsv", sep="\t")
    src = pd.read_csv(LFC / "sources.csv")
    src = src[src.has_data].set_index("source")
    lfc = np.load(LFC / "lfc_matrix.npy")
    var = (LFC / "var_symbols.txt").read_text().split("\n")
    assert len(var) == lfc.shape[1] == len(set(var))
    cmean = np.load(LFC / "control_mean.npy")
    assert cmean.shape == (len(var),)
    rows = src.loc[sg.silenced_gene, "matrix_row"].to_numpy().astype(int)
    ncell = src.loc[sg.silenced_gene, "n_cells"].to_numpy().astype(np.int32)
    assert (ncell == sg.n_crispri_cells.to_numpy()).all()
    L = lfc[rows].astype(np.float32)
    np.savez(out / "knockdown_lfc.npz",
             lfc=L,
             silenced_genes=np.array(sg.silenced_gene.tolist(), dtype="U"),
             measured_genes=np.array(var, dtype="U"),
             n_crispri_cells=ncell,
             control_mean=cmean.astype(np.float32))
    # consistency: gene_pairs lfc == matrix value
    gp = pd.read_parquet(out / "gene_pairs.parquet")
    r = dict(zip(sg.silenced_gene, range(len(sg))))
    c = {g: i for i, g in enumerate(var)}
    x = L[gp.silenced_gene.map(r).to_numpy(), gp.target_gene.map(c).to_numpy()].astype(np.float64)
    log["checks"]["gene_pairs_lfc_equals_matrix"] = bool(np.array_equal(x, gp.lfc.to_numpy()))
    assert log["checks"]["gene_pairs_lfc_equals_matrix"]
    log["checks"]["n_gene_pairs"] = int(len(gp))
    log["checks"]["n_silenced"] = int(gp.silenced_gene.nunique())
    log["checks"]["n_targets"] = int(gp.target_gene.nunique())
    log["checks"]["lfc_shape"] = list(L.shape)
    # README + manifest
    shutil.copyfile(BUILD / "data_README.md", out / "README.md")
    names = ["circuit_edges.csv", "source_features.tsv", "feature_top_genes.tsv", "silenced_genes.tsv",
             "gene_pairs.parquet", "knockdown_lfc.npz", "README.md"]
    with open(out / "MANIFEST.sha256", "w") as f:
        for n in names:
            f.write(f"{sha(out / n)}  {n}\n")


def main():
    log = {"sources": {}, "checks": {}, "outputs": {}}
    for p in [*(TD / s for s in COPY.values()), LFC / "lfc_matrix.npy", LFC / "var_symbols.txt",
              LFC / "control_mean.npy", LFC / "sources.csv", PDF, SPEC, BUILD / "data_README.md",
              BUILD / "BRIEF_template.md"]:
        log["sources"][str(p)] = sha(p)
    tmpl = (BUILD / "BRIEF_template.md").read_text()
    assert tmpl.count(MARK) == 1
    for arm, root in PKG.items():
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        build_data(root / "data", log)
        (root / "methods").mkdir()
        shutil.copyfile(PDF, root / "methods/source_method_paper.pdf")
        brief = tmpl.replace(MARK, CONTRACT_SENTENCE if arm == "contract" else "")
        (root / "BRIEF.md").write_text(brief)
        if arm == "contract":
            (root / "contract").mkdir()
            shutil.copyfile(SPEC, root / "contract/SPEC.md")
    # checks across arms
    a, b = PKG["paper"], PKG["contract"]
    for sub in ["data", "methods"]:
        fa = sorted(p.relative_to(a) for p in (a / sub).rglob("*") if p.is_file() and not p.name.startswith("._"))
        fb = sorted(p.relative_to(b) for p in (b / sub).rglob("*") if p.is_file() and not p.name.startswith("._"))
        assert fa == fb, (fa, fb)
        for p in fa:
            assert sha(a / p) == sha(b / p), p
    ba = (a / "BRIEF.md").read_text(); bb = (b / "BRIEF.md").read_text()
    assert bb.replace(CONTRACT_SENTENCE, "") == ba and CONTRACT_SENTENCE in bb and CONTRACT_SENTENCE not in ba
    assert sha(b / "contract/SPEC.md") == sha(SPEC)
    log["checks"]["data_methods_identical_across_arms"] = True
    log["checks"]["brief_differs_only_by_contract_sentence"] = True
    log["checks"]["spec_verbatim"] = True
    for arm, root in PKG.items():
        log["outputs"][arm] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                               if p.is_file() and not p.name.startswith("._")}
    (BUILD / "build_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log["checks"], indent=1))


if __name__ == "__main__":
    main()
