# Task data: anchor-level H65 panels from the MaxToki-217M manifold run

Built by `scripts/v2_08_task_data.py`. Every file's sha256 and source path is in `manifest.json`.
Nothing here needs a model forward pass. Total size about 108 MB.

## What an anchor is

An anchor is one group of cells that share **donor × tissue × cell type × stage label**, with at least 5 cells
in Tabula Sapiens (immune file for the three hematopoietic panels, lung file for the two lung controls).
The run kept the **largest** groups first. Each cell was passed through MaxToki-217M once. The cell's
hidden state is the mean over its gene-token positions (start and end tokens left out), at each of the
12 hidden-state entries (entry 0 = token embeddings, entries 1-11 = after each of the 11 layers; in the
Hugging Face Llama code the last entry is taken after the final RMSNorm). The anchor centroid is the mean
over its cells. Cells per anchor: internal 13-50, external 20, zero-shot 32, lung_nonhema 30, lung_control 6-50.
Per-cell hidden states were not saved, so nothing below the anchor level can be redone without the model.

## How H65 is defined

H65 is the hematopoietic stage ordering. The name and the idea come from the source paper; it was fixed before
the run (`planning/research_plan.md`, section 0.2). It was not picked from a sweep.

Each anchor gets a stage label (`hema_stage`, one of 34 nodes) from a hand-made map of cell-type names
(`shared/h65_stage_dag.json`, key `cell_type_to_stage`, 62 entries; written for this run on 2026-05-03 from
Laurenti & Göttgens 2018, Paul et al. 2015 and Velten et al. 2017). The ruler between two anchors is the number
of edges on the shortest path between their stages on the 34-node, 35-edge stage graph
(`shared/stage_distance_table.csv`; graph treated as undirected). Values are 0-9 in the hematopoietic panels
and 0-10 in lung_nonhema. Each stage also has a `branch` (12 branches in the file: T_lineage, B_lineage,
monocyte, granulocyte, erythroid, stem, macrophage, NK, dendritic, lymphoid, megakaryocyte, myeloid; the
research plan listed 7) and a `depth`.

The **null ruler** (`d_target_null.npy`) permutes the stage labels among anchors of the same branch
(numpy `default_rng(42)`), then rebuilds the ruler. It keeps branch structure and breaks within-branch order.
There is no null ruler for `lung_control`.

## The head and its input

- Head input = **pooled drift**, 2,464 numbers per anchor. `A_early`, `A_mid`, `A_late` are the mean over
  layers {0-3}, {4-7}, {8-10} of the transposed attention output matrix (`o_proj`, 1232 × 1232).
  `x_early`, `x_mid`, `x_late` are the mean of the anchor centroid over hidden-state entries {1-4}, {5-8}, {9-11}.
  Feature = `concat(x_early A_early − x_mid A_mid, x_mid A_mid − x_late A_late)`.
  `pooled_drift_raw.npy` is this before scaling. `pooled_drift_standardised.npy` is after subtracting the
  internal-panel mean and dividing by the internal-panel SD (+1e-6) — `shared/internal_standardisation.npz`.
  All panels use the internal mean and SD.
- Head = linear map to 10 numbers, `z = W (x − b)`, fitted on the 290 internal anchors so that
  `beta · arccos(cos(z_i, z_j))` matches the ruler (mean squared error, plus 0.1 × reconstruction error);
  Adam, lr 5e-3, 1,500 full-batch steps, torch seed 42. `shared/let_anchor_internal_head.pt` is the deployed head;
  `z_frozen_head.npy` is its output for each panel. Re-fitting with the same seed reproduces the saved weights
  exactly (`outputs/v2_intervals/v2_00_reproduce.json`).

## Panels

| folder | anchors | donors | notes |
|---|---:|---:|---|
| internal | 290 | 11 | head is fitted here; donor TSP2 holds 187 anchors (64%) |
| external | 600 | 13 | no donor or cell shared with internal; five donors hold 496 anchors (83%) |
| zeroshot | 160 | 12 | **all 12 donors are also external donors**; no anchor or cell shared with external; 41 of its 45 tissues are in external |
| lung_nonhema | 50 | 4 | non-hematopoietic lung cells; **stage labels drawn at random**; donor TSP2 is also a training donor; the paper's lung control |
| lung_control | 51 | 4 | the FIRST lung control, replaced during the run: lung-resident immune cells with their real stage labels; it passed all four gates |

## Columns of anchors.csv

`row` (row in the .npy files), `anchor_id`, `donor_id`, `tissue`, `cell_type`, `hema_stage`, `branch`,
`stage_depth`, `n_cells_in_atlas_group` (group size in the atlas), `n_cells_centroided` (cells averaged),
`note` (lung panels only).

## Cautions for a controlled experiment

- Trustworthiness in the run compares the 2,464-d input with the 10-d head output (k = 15 anchors). It never uses
  the ruler. It also depends on the number of anchors, so values from panels of different size are not directly
  comparable.
- Branches whose anchors all share one stage (macrophage, NK, lymphoid in most panels) give a constant ruler
  and cannot be scored by a within-branch correlation. In the internal panel, three of the six scored branches
  (monocyte, erythroid, stem) have more than one stage only because of donor TSP2.
- Re-fitting the head is sensitive to floating-point order: the torch thread count changed one null branch value
  (0.195 vs 0.180), and torch seeds 1-10 give internal trust 0.817-0.835 against 0.811 for seed 42
  (see `V2_INTERVALS_AND_FACTS_REPORT.md`).
- The unit of independent sampling is the donor (11 internal, 13 external). Anchors from one donor are not
  independent.
