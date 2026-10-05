"""Phase 1B+C — MaxToki forward → per-cell hidden states → anchor centroids → ruler.

For each cell we extract per-layer hidden states (mean-pooled across gene-token
positions), then aggregate cells to anchor centroids, then build the H65
biological-ruler distance matrix from the curated stage DAG.

Per-cell feature shape: (n_layers + 1, hidden_size) = (12, 1232) for MaxToki-217M.
Per-anchor centroid: mean of per-cell features across cells in the anchor.

Run one panel at a time:
  python phase1bc_hidden_states_and_centroids.py internal
  python phase1bc_hidden_states_and_centroids.py external
  python phase1bc_hidden_states_and_centroids.py zeroshot
  python phase1bc_hidden_states_and_centroids.py lung_control

Outputs (under runs/manifold-discovery-217M/artifacts/anchors/):
  centroids_<panel>.npy        (n_anchors, n_layers+1, hidden) float32
  anchor_meta_<panel>.csv      anchor_id, donor, tissue, cell_type, hema_stage, n_cells
  d_target_<panel>.npy         (n_anchors, n_anchors) — only for hematopoietic-stage panels
  pos_pool_meta.json           pool method + dims metadata
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiAttentionExtractor

OUT_PHASE1 = RUN / "outputs/phase1"
ART = RUN / "artifacts/anchors"
ART.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32
BATCH = 1  # single-cell forward; 217M is small but seq length varies

# Panel-specific cell-cap for memory-pressured machines. Subsamples cells_<panel>.npz
# down to MAX_CELLS_PER_PANEL.get(panel) before forward pass. Centroids are computed on
# the subsample. Keeps anchor counts unchanged.
MAX_CELLS_PER_PANEL = {
    "zeroshot": 5000,
    "lung_control": 2124,   # already small
    "external": 12000,
}


def load_stage_graph():
    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    nodes = list(dag["stage_to_branch"].keys())
    edges = dag["edges"]
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return nodes, adj


def shortest_path_distance(src: str, adj: dict) -> dict[str, int]:
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def build_ruler(stage_labels: list[str]) -> np.ndarray:
    """Build (n, n) shortest-path distance matrix on the H65 stage DAG.
    Stages not in the DAG raise an error."""
    nodes, adj = load_stage_graph()
    cache: dict[str, dict[str, int]] = {}
    n = len(stage_labels)
    D = np.zeros((n, n), dtype=np.float32)
    for i, si in enumerate(stage_labels):
        if si not in cache:
            cache[si] = shortest_path_distance(si, adj)
        for j in range(i + 1, n):
            sj = stage_labels[j]
            if sj not in cache[si]:
                # disconnected components → use a large sentinel distance (max DAG diameter + 1)
                D[i, j] = D[j, i] = 99
            else:
                D[i, j] = D[j, i] = float(cache[si][sj])
    return D


def main():
    if len(sys.argv) != 2:
        print("Usage: phase1bc_hidden_states_and_centroids.py <panel>")
        sys.exit(1)
    panel = sys.argv[1]
    assert panel in {"internal", "external", "zeroshot", "lung_control", "lung_nonhema"}

    print("=" * 70)
    print(f"PHASE 1B+C — panel={panel}  device={DEVICE}")
    print("=" * 70)
    t_phase = time.time()

    # ----------- Load tokenized cells + cell metadata
    npz = np.load(OUT_PHASE1 / f"cells_{panel}.npz")
    token_ids = npz["token_ids"]      # (n_cells, L_cap) int32
    attn_mask = npz["attn_mask"]      # (n_cells, L_cap) int8
    seq_lens = npz["seq_lens"]        # (n_cells,) int32
    cells_df = pd.read_csv(OUT_PHASE1 / f"cells_{panel}_obs.csv")
    anchors_df = pd.read_csv(OUT_PHASE1 / f"anchors_{panel}.csv")
    n_cells_full = token_ids.shape[0]
    print(f"  cells (full): {n_cells_full}  anchors: {len(anchors_df)}  L_cap: {token_ids.shape[1]}")

    # Memory-pressure-aware subsampling: stratify by anchor_id so every anchor still
    # has cells contributing to its centroid.
    cap = MAX_CELLS_PER_PANEL.get(panel, n_cells_full)
    if n_cells_full > cap:
        rng = np.random.default_rng(42)
        keep_idx_per_anchor = []
        # target cells per anchor
        target_per_anchor = max(3, int(np.ceil(cap / max(1, len(anchors_df)))))
        for aid in anchors_df["anchor_id"].astype(str):
            mask = (cells_df["anchor_id"].astype(str) == aid).to_numpy()
            idx = np.where(mask)[0]
            if len(idx) > target_per_anchor:
                idx = rng.choice(idx, size=target_per_anchor, replace=False)
            keep_idx_per_anchor.append(idx)
        keep_idx = np.sort(np.concatenate(keep_idx_per_anchor))
        token_ids = token_ids[keep_idx]
        attn_mask = attn_mask[keep_idx]
        seq_lens = seq_lens[keep_idx]
        cells_df = cells_df.iloc[keep_idx].reset_index(drop=True)
        n_cells = token_ids.shape[0]
        print(f"  subsampled to {n_cells} cells (cap={cap}, target/anchor={target_per_anchor})")
    else:
        n_cells = n_cells_full
    print(f"  seq_len  median: {int(np.median(seq_lens))}  mean: {seq_lens.mean():.1f}  max: {seq_lens.max()}")

    # ----------- Load MaxToki
    print(f"\n[1] Loading MaxToki-217M on {DEVICE}...")
    t0 = time.time()
    xt = MaxTokiAttentionExtractor(device=DEVICE, dtype=DTYPE)
    model = xt.model
    model.eval()
    n_layers = xt.n_layers
    hidden_size = xt.hidden_size
    n_layer_states = n_layers + 1  # embed + per-layer
    print(f"  n_layers={n_layers} hidden={hidden_size}  loaded in {time.time()-t0:.1f}s")

    # ----------- Forward each cell, mean-pool hidden states across gene-token positions
    per_cell = np.zeros((n_cells, n_layer_states, hidden_size), dtype=np.float32)
    print(f"\n[2] Running forward passes (mean-pool hidden states across gene-token positions)...")
    t0 = time.time()
    for i in range(n_cells):
        L = int(seq_lens[i])
        if L < 3:
            continue  # only BOS/EOS
        ids = torch.from_numpy(token_ids[i, :L].astype(np.int64))[None, :].to(DEVICE)
        mask = torch.from_numpy(attn_mask[i, :L].astype(np.int64))[None, :].to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        # out.hidden_states is a tuple of (n_layers + 1) tensors, each (1, L, hidden)
        hs = torch.stack([h.squeeze(0) for h in out.hidden_states], dim=0)  # (n_layer_states, L, hidden)
        # Pool across gene-token positions (skip BOS at pos 0 and EOS at pos L-1)
        if L >= 4:
            pooled = hs[:, 1 : L - 1, :].mean(dim=1)  # (n_layer_states, hidden)
        else:
            pooled = hs[:, 1:L, :].mean(dim=1)
        per_cell[i] = pooled.cpu().numpy().astype(np.float32)
        if (i + 1) % max(1, n_cells // 50) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_cells - (i + 1))
            print(f"    cell {i+1}/{n_cells}  elapsed {elapsed:.1f}s  eta {eta:.1f}s")
    print(f"  done in {time.time()-t0:.1f}s  per_cell shape={per_cell.shape}")

    # Free model memory before aggregation
    del model, xt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ----------- Aggregate to anchor centroids
    print(f"\n[3] Aggregating to anchor centroids...")
    anchor_ids = anchors_df["anchor_id"].astype(str).to_numpy()
    cells_anchor = cells_df["anchor_id"].astype(str).to_numpy()
    centroids = np.zeros((len(anchor_ids), n_layer_states, hidden_size), dtype=np.float32)
    n_per_anchor = np.zeros(len(anchor_ids), dtype=np.int32)
    for ai, aid in enumerate(anchor_ids):
        mask = cells_anchor == aid
        n = int(mask.sum())
        n_per_anchor[ai] = n
        if n == 0:
            continue
        centroids[ai] = per_cell[mask].mean(axis=0)
    np.save(ART / f"centroids_{panel}.npy", centroids)
    anchor_meta = anchors_df.copy()
    anchor_meta["n_cells_centroided"] = n_per_anchor
    anchor_meta.to_csv(ART / f"anchor_meta_{panel}.csv", index=False)
    print(f"  centroids saved → {(ART/f'centroids_{panel}.npy').relative_to(RUN)}")
    print(f"  anchor cell counts: min {n_per_anchor.min()}, median {int(np.median(n_per_anchor))}, max {n_per_anchor.max()}")

    # ----------- Build H65 ruler (for all panels with hema_stage labels — incl. random ones)
    if panel in {"internal", "external", "zeroshot", "lung_nonhema"}:
        print(f"\n[4] Building H65 ruler (graph distance on stage DAG)...")
        stage_labels = anchor_meta["hema_stage"].astype(str).tolist()
        D = build_ruler(stage_labels)
        np.save(ART / f"d_target_{panel}.npy", D)
        print(f"  d_target shape={D.shape}  min={D.min()} median={np.median(D):.1f} max={D.max()}")
        # Also build a null-shuffled ruler (within-branch stage permutation) for the H65_null branch
        nodes, adj = load_stage_graph()
        # Cache stage_to_branch for the within-branch shuffle
        stage_to_branch = {k: v["branch"] for k, v in json.loads(
            (RUN / "planning/h65_stage_dag.json").read_text()
        )["stage_to_branch"].items()}
        rng = np.random.default_rng(42)
        labels_array = np.array(stage_labels)
        shuffled = labels_array.copy()
        # Group anchor positions by branch, then permute stage labels within each group
        branches_per_anchor = np.array([stage_to_branch.get(s, "_unknown") for s in stage_labels])
        for branch in np.unique(branches_per_anchor):
            idx = np.where(branches_per_anchor == branch)[0]
            shuffled[idx] = rng.permutation(labels_array[idx])
        D_null = build_ruler(shuffled.tolist())
        np.save(ART / f"d_target_{panel}_null_shuffled.npy", D_null)
        print(f"  d_target_null_shuffled saved (within-branch shuffle)")

    meta_path = ART / "pos_pool_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta[panel] = {
        "pool_method": "mean across gene-token positions (excluding BOS at 0 and EOS at L-1)",
        "n_cells": int(n_cells),
        "n_anchors": int(len(anchor_ids)),
        "centroid_shape": list(centroids.shape),
        "device": DEVICE,
        "elapsed_seconds": float(time.time() - t_phase),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nPhase 1B+C ({panel}) complete in {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
