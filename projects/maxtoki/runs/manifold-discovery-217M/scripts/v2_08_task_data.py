"""v2 step 8: build outputs/v2_intervals/task_data/ — anchor-level data for a later controlled experiment.

Per panel (internal, external, zeroshot, lung_nonhema, lung_control = the first, replaced lung control):
  centroids.npy                 (n, 12, 1232) float32  anchor centroids (copy of artifacts/anchors/centroids_<panel>.npy)
  pooled_drift_raw.npy          (n, 2464) float32      the head input BEFORE standardisation (pooled drift)
  pooled_drift_standardised.npy (n, 2464) float32      the head input AS USED (internal mean / SD applied)
  z_frozen_head.npy             (n, 10) float32        output of the deployed Phase-5 head
  d_target_H65.npy              (n, n) float32         H65 ruler (copy)
  d_target_null.npy             (n, n) float32         within-branch-shuffled null ruler (copy, where it exists)
  anchors.csv                   anchor_id, donor_id, tissue, cell_type, hema_stage, branch, stage_depth,
                                n_cells_in_atlas_group, n_cells_centroided, row
shared/: stage DAG, 34x34 stage distance table, internal standardisation mean/SD, pooled-drift operator components,
         L10H6 operator, deployed head weights, frozen gate spec.
manifest.json: sha256 and source path of every file.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, shutil
import numpy as np, pandas as pd, torch
from v2_common import *

TD = OUT / "task_data"; TD.mkdir(parents=True, exist_ok=True)
manifest = {}


def record(dst, src=None):
    manifest[str(dst.relative_to(TD))] = {"sha256": sha256(dst), "bytes": dst.stat().st_size,
                                          "source": str(src) if src else "computed by v2_08_task_data.py"}


def copy(src, dst):
    shutil.copyfile(src, dst); record(dst, src)


FZ = Frozen()
head = LETHead(FZ.F_int.shape[1], LATENT_DIM)
head.load_state_dict(torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")); head.eval()
dag = stage_dag(); s2b = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
s2d = {k: v["depth"] for k, v in dag["stage_to_branch"].items()}

for p in ["internal", "external", "zeroshot", "lung_nonhema", "lung_control"]:
    pd_dir = TD / p; pd_dir.mkdir(exist_ok=True)
    c, d, m = load_panel(p)
    copy(ART / f"anchors/centroids_{p}.npy", pd_dir / "centroids.npy")
    raw = build_pooled_drift(np.asarray(c), FZ.A_e, FZ.A_m, FZ.A_l, FZ.part)
    std = ((raw - FZ.mu) / FZ.sd).astype(np.float32)
    np.save(pd_dir / "pooled_drift_raw.npy", raw); record(pd_dir / "pooled_drift_raw.npy")
    np.save(pd_dir / "pooled_drift_standardised.npy", std); record(pd_dir / "pooled_drift_standardised.npy")
    z = head_z(head, std)
    np.save(pd_dir / "z_frozen_head.npy", z); record(pd_dir / "z_frozen_head.npy")
    copy(ART / f"anchors/d_target_{p}.npy", pd_dir / "d_target_H65.npy")
    nul = ART / f"anchors/d_target_{p}_null_shuffled.npy"
    if nul.exists():
        copy(nul, pd_dir / "d_target_null.npy")
    a = pd.DataFrame({"row": np.arange(len(m)), "anchor_id": m["anchor_id"], "donor_id": m["donor_id"], "tissue": m["tissue"],
                      "cell_type": m["cell_type"], "hema_stage": m["hema_stage"],
                      "branch": m["hema_stage"].map(s2b).fillna("_unk"), "stage_depth": m["hema_stage"].map(s2d),
                      "n_cells_in_atlas_group": m["n"], "n_cells_centroided": m["n_cells_centroided"]})
    if p == "lung_nonhema":
        a["note"] = "non-hematopoietic lung cells; hema_stage drawn uniformly at random from the 34 DAG stages"
    if p == "lung_control":
        a["note"] = ("FIRST lung control (replaced during the run): lung-resident immune cells with their real stage labels; "
                     "it passed all four gates under the frozen head")
    a.to_csv(pd_dir / "anchors.csv", index=False); record(pd_dir / "anchors.csv")

sh = TD / "shared"; sh.mkdir(exist_ok=True)
copy(PLAN / "h65_stage_dag.json", sh / "h65_stage_dag.json")
nodes, idx, T = stage_distance_table()
pd.DataFrame(T, index=nodes, columns=nodes).to_csv(sh / "stage_distance_table.csv"); record(sh / "stage_distance_table.csv")
np.savez(sh / "internal_standardisation.npz", mean=FZ.mu, sd=FZ.sd); record(sh / "internal_standardisation.npz")
copy(ART / "operators/pooled_drift_components.npz", sh / "pooled_drift_components.npz")
copy(ART / "operators/operator_index.json", sh / "operator_index.json")
copy(ART / "operators/layer10_head6.npy", sh / "operator_L10H6.npy")
copy(ART / "heads/let_anchor_internal.pt", sh / "let_anchor_internal_head.pt")
copy(REP / "quality_gates_spec.json", sh / "quality_gates_spec.json")
total = sum(v["bytes"] for v in manifest.values())
(TD / "manifest.json").write_text(json.dumps({"total_bytes": total, "files": manifest}, indent=2))
write_run_config("v2_08_task_data", CORE_INPUTS, {"total_bytes": total})
print("task_data bytes", total)
