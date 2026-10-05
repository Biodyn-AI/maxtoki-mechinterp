"""v3_attention_k562 -- attention-GRN, MaxToki-217M, Replogle K562, CORRECT input encoding (item V3-6).

Why this exists
---------------
checks/INPUT_ENCODING_AUDIT.md: the deployed Phase-0 K562 run (scripts/phase0_extract.py) fed
MaxToki genes ranked by log1p(CP10k) / median, because the stored X of replogle_concat.h5ad holds
log1p(CP10k), not counts. Its gene set, gene statistics, co-expression and knockdown labels were also
computed after a SECOND log (log1p(X / rowsum * 1e4) on X that was already log1p). This script redoes
Phase 0 with:
  * the same 2,000 K562 non-targeting cells (numpy default_rng(42).choice, sorted; checked against
    outputs/phase0/control_cells.csv),
  * correct model input: counts from setup/inputs_v3.py (round(expm1(X) / unit), integer counts up to
    one per-cell factor), counts / total * 1e4 / Geneformer gene median, ranked, first 2,046 genes,
    <bos>/<eos>; every cell passes inputs_v3.check_encoding at tokenisation AND again
    (assert_encoding_batch) right before its forward pass,
  * the same 1,500-gene rule (top variance among genes in the MaxToki vocabulary) computed on a SINGLE
    log: log1p(counts / rowsum * 1e4), rowsum over the file's 6,546 genes (exactly the deployed RPE1
    path, which started from raw counts),
  * gene features (mean, variance, dropout) and |Spearman| co-expression on the same single-log values.

Edge (deployed definition, unchanged): attention weights of model.model.layers[l].self_attn
(= output_attentions[l]), mean over the 8 heads, row = query gene, column = key gene, summed over the
cells that contain both genes and divided by that pair count (cells where the causal mask forces 0 are
included, as deployed). All 11 layers are accumulated; layer 8 is the pre-specified one.
Attention is accumulated over the UNION of the new and the deployed 1,500-gene sets, so the new-encoding
edges can also be read on the deployed gene set (to separate the encoding effect from the gene-set
effect).

Subcommands (run in this order)
  prep                       CPU: cells, counts, single-log gene set + features + co-expression, tokens
  extract [max_minutes]      MPS: forward passes, resumable (state saved every 100 cells and at the end
                             of each call); stops cleanly if available memory < 3 GB
  finalize                   edges = sum / pair count; zero-pattern and count checks
Outputs: outputs/v3_attention_k562/{prep,extract,edges}/
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True       # never write .pyc files into existing folders

import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/attention-grn-217M"
P0 = RUN / "outputs/phase0"                       # deployed Phase 0 (read only)
OUT3 = RUN / "outputs/v3_attention_k562"
SETUP = PROJ / "setup"
sys.path.insert(0, str(SETUP))

import inputs_v3 as I  # noqa: E402
from dataset_loader import K562_H5, SYM2ENS_PKL  # noqa: E402

MODEL_DIR = SETUP / "MaxToki-217M-HF"
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/"
                  "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
N_CTRL = 2000
N_HVG = 1500
MAX_LEN = 2048
SEED = 42                     # deployed Phase-0 cell draw
PRIMARY_LAYER = 8
BLOCK = 25                    # cells per encoding-check / GPU flush block
SAVE_EVERY_BLOCKS = 4         # state saved every 100 cells
MIN_FREE_GB = 3.0

D_PREP, D_EXT, D_EDGE = OUT3 / "prep", OUT3 / "extract", OUT3 / "edges"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(obj, path: Path) -> None:
    def _d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_d)
    os.replace(tmp, path)


def _dec(a):
    return [s.decode() if isinstance(s, bytes) else s for s in a]


# =============================================================================================== prep
def cmd_prep() -> None:
    import h5py
    from scipy.stats import rankdata, spearmanr
    t0 = time.time()
    D_PREP.mkdir(parents=True, exist_ok=True)
    chk: dict = {}

    # ---- 1. the deployed 2,000 control cells, re-drawn with the deployed rule (phase0_extract.py:85-120)
    with h5py.File(K562_H5, "r") as f:
        n_cells_total, n_vars = f["X"].shape
        var_symbols = _dec(f["var"]["gene_name_index"][:])
        cl_cats = [s.lower() for s in _dec(f["obs"]["cell_line"]["categories"][:])]
        cl_codes = f["obs"]["cell_line"]["codes"][:]
        pg_cats = _dec(f["obs"]["gene"]["categories"][:])
        pg_codes = f["obs"]["gene"]["codes"][:]
    k562 = cl_codes == cl_cats.index("k562")
    nt = np.isin(pg_codes, np.array([i for i, g in enumerate(pg_cats)
                                     if "non-targeting" in g.lower() or "nontargeting" in g.lower()], np.int64))
    ctrl_all = np.where(k562 & nt)[0]
    rows = np.sort(np.random.default_rng(SEED).choice(ctrl_all, size=N_CTRL, replace=False))
    dep_cells = pd.read_csv(P0 / "control_cells.csv")
    chk["n_k562_nontargeting_cells"] = int(len(ctrl_all))
    chk["rows_equal_deployed_control_cells_csv"] = bool(np.array_equal(rows, dep_cells["cell_idx_global"].to_numpy()))
    assert chk["rows_equal_deployed_control_cells_csv"], "cell draw differs from the deployed Phase 0"

    # ---- 2. counts (inputs_v3) and the stored X rows (for the deployed comparison only)
    vm = I.get_var_map("k562")
    counts = np.zeros((N_CTRL, n_vars), dtype=np.float64)
    Xs = np.zeros((N_CTRL, n_vars), dtype=np.float32)
    row_checks = []
    with I.CountReader("k562") as r:
        for i, row in enumerate(rows):
            c, cc = I.counts_accept_unit_one(r, int(row), return_check=True)
            counts[i] = c
            Xs[i] = r.stored_X(int(row)).astype(np.float32)
            row_checks.append({"row": int(row), "kind": cc["kind"], "unit": cc.get("unit"),
                               "max_dev": cc.get("max_dev"), "q_max": cc.get("q_max"),
                               "accepted_as": cc.get("accepted_as", "integer_multiple")})
    assert np.all(counts == np.round(counts)) and counts.max() < 2 ** 31
    rc = pd.DataFrame(row_checks)
    chk["count_rows"] = {"n": int(len(rc)), "kinds": rc["accepted_as"].value_counts().to_dict(),
                         "max_dev_from_whole_multiple": float(rc["max_dev"].max()),
                         "max_multiple": float(rc["q_max"].max()),
                         "stored_X_whole_number_fraction": float(np.mean(Xs[Xs > 0] == np.round(Xs[Xs > 0])))}
    # second way: implied total count of each cell = 1e4 / unit is a whole number close to UMI_count
    implied = 1e4 / rc["unit"].to_numpy(np.float64)
    umi = dep_cells["UMI_count"].to_numpy(np.float64)
    chk["implied_total_over_UMI_count"] = {"min": float(np.min(implied / umi)), "median": float(np.median(implied / umi)),
                                           "max": float(np.max(implied / umi))}
    chk["sum_counts_over_implied_total"] = {"median": float(np.median(counts.sum(1) / implied))}
    np.save(D_PREP / "counts_int32.npy", counts.astype(np.int32))
    rc.to_csv(D_PREP / "count_row_checks.csv", index=False)

    # ---- 3. single-log values and the 1,500-gene set (deployed rule, phase0_extract.py:142-187)
    rs = counts.sum(1, keepdims=True)
    L = np.log1p(counts / rs * 1e4)
    var_all = L.var(0)
    with open(SYM2ENS_PKL, "rb") as fh:
        sym2ens = pickle.load(fh)
    tok = I.get_tokenizer()
    mt_vocab = set(tok.gene_token_dict.keys())
    var_ens = [sym2ens.get(s) for s in var_symbols]
    eligible = np.array([e is not None and e in mt_vocab for e in var_ens], dtype=bool)

    def top_eligible(v):
        out = []
        for gi in np.argsort(-v):
            if eligible[gi]:
                out.append(int(gi))
                if len(out) == N_HVG:
                    break
        return np.array(sorted(out), dtype=np.int64)

    hvg = top_eligible(var_all)
    # second way: float32 arithmetic exactly as the deployed code would do it on counts
    c32 = counts.astype(np.float32)
    rs32 = c32.sum(1, keepdims=True)
    hvg32 = top_eligible(np.log1p(c32 / rs32 * 1e4).var(0))
    v_el = np.sort(var_all[eligible])[::-1]
    chk["gene_set"] = {"n_eligible_vocab_genes": int(eligible.sum()), "n_selected": int(len(hvg)),
                       "float32_route_same_set": bool(np.array_equal(hvg, hvg32)),
                       "variance_at_rank_1500": float(v_el[N_HVG - 1]), "variance_at_rank_1501": float(v_el[N_HVG]),
                       "relative_gap_1500_1501": float((v_el[N_HVG - 1] - v_el[N_HVG]) / v_el[N_HVG - 1])}
    dep_tab = pd.read_csv(P0 / "hvg_gene_table.csv")
    dep_var = dep_tab["var_idx"].to_numpy(np.int64)
    # deployed rule re-run on the stored X (double log) must give the deployed set back
    rsX = Xs.sum(1, keepdims=True); rsX[rsX == 0] = 1.0
    dep_rebuilt = top_eligible(np.log1p(Xs / rsX * 1e4).var(0))
    shared = np.intersect1d(hvg, dep_var)
    chk["gene_set"].update({
        "deployed_rule_on_stored_X_rebuilds_deployed_set": bool(np.array_equal(dep_rebuilt, dep_var)),
        "n_shared_with_deployed": int(len(shared)), "n_new_only": int(len(np.setdiff1d(hvg, dep_var))),
        "n_deployed_only": int(len(np.setdiff1d(dep_var, hvg))),
        "jaccard_with_deployed": float(len(shared) / len(np.union1d(hvg, dep_var)))})
    # how the single-log variance relates to the deployed (double-log) variance on the deployed genes
    dep_gf = pd.read_csv(P0 / "gene_features.csv")
    chk["gene_set"]["spearman_single_log_var_vs_deployed_var_on_deployed_genes"] = float(
        spearmanr(var_all[dep_var], dep_gf["variance"]).correlation)
    chk["gene_set"]["spearman_single_log_var_vs_storedX_var_on_new_genes"] = float(
        spearmanr(var_all[hvg], Xs[:, hvg].astype(np.float64).var(0)).correlation)

    union = np.union1d(hvg, dep_var)
    sym = [var_symbols[i] for i in hvg]
    ens = [var_ens[i] for i in hvg]
    tid = np.array([tok.gene_token_dict[e] for e in ens], dtype=np.int64)
    assert len(set(tid.tolist())) == N_HVG
    trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    tf_out = trrust.assign(tf=trrust["tf"].str.upper(), target=trrust["target"].str.upper()) \
        .groupby("tf")["target"].nunique().to_dict()
    Lh = L[:, hvg]
    gf = pd.DataFrame({
        "hvg_idx": np.arange(N_HVG), "symbol": sym, "ensembl_id": ens, "maxtoki_token_id": tid,
        "var_idx": hvg, "mean_expr": Lh.mean(0), "variance": Lh.var(0),
        "dropout_rate": (counts[:, hvg] == 0).mean(0),
        "tf_out_degree": [float(tf_out.get(s.upper(), 0)) for s in sym],
        "variance_storedX_log1p_cp10k": Xs[:, hvg].astype(np.float64).var(0),   # sensitivity only
        "in_deployed_gene_set": np.isin(hvg, dep_var)})
    gf.to_csv(D_PREP / "gene_features.csv", index=False)
    gf[["hvg_idx", "symbol", "ensembl_id", "maxtoki_token_id", "var_idx"]].to_csv(D_PREP / "hvg_gene_table.csv", index=False)
    pd.DataFrame({"var_idx": union, "symbol": [var_symbols[i] for i in union],
                  "ensembl_id": [var_ens[i] for i in union],
                  "maxtoki_token_id": [tok.gene_token_dict[var_ens[i]] for i in union],
                  "in_new_set": np.isin(union, hvg), "in_deployed_set": np.isin(union, dep_var)}
                 ).to_csv(D_PREP / "union_gene_table.csv", index=False)

    # ---- 4. |Spearman| co-expression on single-log values (deployed formula, phase0_extract.py:226-233)
    R = np.apply_along_axis(rankdata, 0, Lh)
    R -= R.mean(0, keepdims=True)
    R /= (R.std(0, keepdims=True) + 1e-12)
    sp = (R.T @ R) / Lh.shape[0]
    np.save(D_PREP / "spearman_edges.npy", sp.astype(np.float32))
    j = np.random.default_rng(0).choice(N_HVG, 2, replace=False)
    chk["spearman_spot_check_vs_scipy"] = float(abs(sp[j[0], j[1]] - spearmanr(Lh[:, j[0]], Lh[:, j[1]]).correlation))

    # ---- 5. tokens: correct (checked) and deployed (for comparison only; never fed to the model)
    toks = np.zeros((N_CTRL, MAX_LEN), dtype=np.int32)
    lens = np.zeros(N_CTRL, dtype=np.int32)
    dep_toks = np.zeros((N_CTRL, MAX_LEN), dtype=np.int32)
    dep_lens = np.zeros(N_CTRL, dtype=np.int32)
    agree, n_tie, max_viol, dep_fail = [], 0, 0, 0
    for i in range(N_CTRL):
        cell = I.tokenize_counts(counts[i], vm, MAX_LEN, check=True)          # raises if the check fails
        res = I.check_encoding(counts[i], cell, vm, MAX_LEN)
        assert res["pass"]
        n_tie += res["n_positions_tie_reordered"]; max_viol = max(max_viol, res["n_order_violations"])
        toks[i, :len(cell.token_ids)] = cell.token_ids; lens[i] = len(cell.token_ids)
        dc = tok.tokenize_cell(Xs[i], vm.var_indices, vm.token_ids, vm.medians, max_len=MAX_LEN)
        dep_toks[i, :len(dc.token_ids)] = dc.token_ids; dep_lens[i] = len(dc.token_ids)
        dep_fail += int(not I.check_encoding(counts[i], dc, vm, MAX_LEN)["pass"])
        fo_new = I.full_order(counts[i], vm)
        fo_dep = tok.tokenize_cell(Xs[i], vm.var_indices, vm.token_ids, vm.medians, max_len=10 ** 9).token_ids[1:-1]
        a = I.order_agreement(fo_dep, fo_new, MAX_LEN)
        tn, td = toks[i, 1:lens[i] - 1], dep_toks[i, 1:dep_lens[i] - 1]
        a.update({"cell": i, "row": int(rows[i]), "len_new": int(lens[i]), "len_deployed": int(dep_lens[i]),
                  "n_new_genes_in_seq": int(np.isin(tn, tid).sum()),
                  "n_deployed_genes_in_seq_deployed_tokens": int(np.isin(td, dep_tab["maxtoki_token_id"]).sum())})
        agree.append(a)
    ag = pd.DataFrame(agree)
    ag.to_csv(D_PREP / "order_agreement_per_cell.csv", index=False)
    chk["tokens"] = {"n_cells": N_CTRL, "all_pass_check_encoding": True, "max_order_violations": int(max_viol),
                     "n_positions_tie_reordered_total": int(n_tie), "mean_len": float(lens.mean()),
                     "min_len": int(lens.min()), "n_cells_cut_at_max_len": int((lens == MAX_LEN).sum()),
                     "deployed_tokens_failing_check_encoding": int(dep_fail),
                     "deployed_mean_len": float(dep_lens.mean())}
    q = lambda c: [float(ag[c].mean()), float(ag[c].min()), float(ag[c].max())]  # noqa: E731
    chk["order_agreement_deployed_vs_new_mean_min_max"] = {c: q(c) for c in
                                                           ["spearman_full", "spearman_fed", "top200_overlap",
                                                            "kept_set_overlap", "same_position_share"]}
    # proof that these are the deployed cells AND the deployed tokens: deployed tokens rebuild the saved pair counts
    dep_tid = dep_tab["maxtoki_token_id"].to_numpy(np.int64)
    Pm = np.zeros((N_CTRL, len(dep_tid)), dtype=np.float32)
    lut = np.full(20275, -1, np.int64); lut[dep_tid] = np.arange(len(dep_tid))
    for i in range(N_CTRL):
        u = lut[dep_toks[i, :dep_lens[i]]]
        Pm[i, u[u >= 0]] = 1.0
    pc_dep = np.load(P0 / "attention_pair_counts.npy").astype(np.int64)
    chk["deployed_tokens_rebuild_deployed_pair_counts"] = bool(np.array_equal(np.rint(Pm.T @ Pm).astype(np.int64), pc_dep))
    assert chk["deployed_tokens_rebuild_deployed_pair_counts"]

    np.savez(D_PREP / "cells_tokens.npz", rows=rows, tokens=toks, lens=lens,
             deployed_tokens=dep_toks, deployed_lens=dep_lens)
    np.savez(D_PREP / "gene_sets.npz", new_var_idx=hvg, deployed_var_idx=dep_var, union_var_idx=union)
    pd.DataFrame({"ctrl_idx_local": np.arange(N_CTRL), "cell_idx_global": rows,
                  "cell_barcode": dep_cells["cell_barcode"], "UMI_count": dep_cells["UMI_count"]}
                 ).to_csv(D_PREP / "control_cells.csv", index=False)
    chk["counts_sha256"] = I.array_sha256(counts.astype(np.int32))
    chk["tokens_sha256"] = I.array_sha256(toks)
    chk["seconds"] = time.time() - t0
    save_json(chk, D_PREP / "prep_check.json")
    log(json.dumps(chk, indent=1, default=str))


# =============================================================================================== extract
class Accumulator:
    """Per-layer head-mean attention summed over cells, on the union gene set (float32 on the device
    per block, flushed to float64 on the CPU)."""

    def __init__(self, model, n_layers: int, Gu: int, device: str):
        import torch
        self.torch = torch
        self.dev = device
        self.Gu = Gu
        self.n_layers = n_layers
        self.acc = torch.zeros((n_layers, Gu, Gu), dtype=torch.float32, device=device)
        self.pos = None
        self.ir = None
        self.ic = None
        self.upper_mass = []          # per cell, max over layers of the attention mass above the diagonal
        self._cell_upper = 0.0
        self.handles = []
        for li, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li):
        def hook(mod, inp, out):
            w = out[1]
            if w is None:
                raise RuntimeError("attention weights not returned; eager attention required")
            hm = w[0].mean(0)                                   # (T, T) head mean, row = query
            if li == PRIMARY_LAYER:
                self._cell_upper = float(self.torch.triu(hm, diagonal=1).sum())
            sub = hm.index_select(0, self.pos).index_select(1, self.pos)
            self.acc[li][self.ir, self.ic] += sub               # unique indices within a cell
            return None
        return hook

    def set_cell(self, pos: np.ndarray, ids: np.ndarray):
        t = self.torch
        self.pos = t.from_numpy(pos.astype(np.int64)).to(self.dev)
        idt = t.from_numpy(ids.astype(np.int64)).to(self.dev)
        self.ir, self.ic = idt[:, None], idt[None, :]

    def flush(self) -> np.ndarray:
        a = self.acc.detach().to("cpu").numpy().astype(np.float64)
        self.acc.zero_()
        return a

    def remove(self):
        for h in self.handles:
            h.remove()


def load_prep():
    z = np.load(D_PREP / "cells_tokens.npz")
    gs = np.load(D_PREP / "gene_sets.npz")
    ut = pd.read_csv(D_PREP / "union_gene_table.csv")
    return z, gs, ut


def _state_paths():
    return D_EXT / "state.npz", D_EXT / "state_meta.json"


def cmd_extract(max_minutes: float = 7.0) -> None:
    import torch
    import hooks_v2 as hv2
    from transformers import LlamaForCausalLM
    t_start = time.time()
    D_EXT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    z, gs, ut = load_prep()
    toks, lens, rows = z["tokens"], z["lens"], z["rows"]
    counts = np.load(D_PREP / "counts_int32.npy", mmap_mode="r")
    Gu = len(ut)
    tok2u = np.full(20275, -1, dtype=np.int64)
    tok2u[ut["maxtoki_token_id"].to_numpy(np.int64)] = np.arange(Gu)
    sp, mp = _state_paths()
    if mp.exists():
        meta = json.load(open(mp))
        st = np.load(sp)
        attn_sum, pair_counts, order_counts = st["attn_sum"], st["pair_counts"], st["order_counts"]
        assert meta["next_cell"] == int(st["next_cell"]), "state files out of sync"
    else:
        meta = {"next_cell": 0, "chunks": [], "per_cell": [], "encoding_checks": []}
        attn_sum = np.zeros((11, Gu, Gu), dtype=np.float64)
        pair_counts = np.zeros((Gu, Gu), dtype=np.int32)
        order_counts = np.zeros((Gu, Gu), dtype=np.int32)    # C[a, b] = #cells with both, b before a
    start_cell = meta["next_cell"]
    if start_cell >= N_CTRL:
        log("extraction already complete"); return
    gb = hv2.check_memory(MIN_FREE_GB)
    log(f"start at cell {start_cell}; available memory {gb:.1f} GB")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
    n_layers = model.config.num_hidden_layers
    acc = Accumulator(model, n_layers, Gu, dev)
    log(f"model loaded on {dev} in {time.time() - t_start:.1f}s")

    def save_state(next_cell):
        tmp = D_EXT / "state_tmp.npz"
        np.savez(tmp, attn_sum=attn_sum, pair_counts=pair_counts, order_counts=order_counts,
                 next_cell=np.array(next_cell))
        os.replace(tmp, sp)
        meta["next_cell"] = int(next_cell)
        save_json(meta, mp)

    ci = start_cell
    n_blocks_done = 0
    block_times = []
    stop_reason = "done"
    pending = []
    with torch.no_grad():
        while ci < N_CTRL:
            est = (np.median(block_times) if block_times else 30.0) + 10.0
            if time.time() - t_start + est > max_minutes * 60:
                stop_reason = "time budget"; break
            try:
                gb = hv2.check_memory(MIN_FREE_GB)
            except hv2.MemoryGuardError as e:
                stop_reason = f"memory guard: {e}"; break
            tb = time.time()
            blk = list(range(ci, min(ci + BLOCK, N_CTRL)))
            # encoding check on every cell of the block, right before the forward passes
            ec = I.assert_encoding_batch([counts[i].astype(np.float64) for i in blk],
                                         [toks[i, :lens[i]] for i in blk], "k562", MAX_LEN,
                                         label=f"cells {blk[0]}-{blk[-1]}")
            meta["encoding_checks"].append({k: ec[k] for k in ("label", "n_checked", "all_pass",
                                                                 "max_order_violations",
                                                                 "n_positions_tie_reordered")})
            for i in blk:
                t1 = time.time()
                seq = toks[i, :lens[i]].astype(np.int64)
                u = tok2u[seq]
                pos = np.flatnonzero(u >= 0)
                ids = u[pos]
                acc.set_cell(pos, ids)
                out = model(input_ids=torch.from_numpy(seq[None]).to(dev), use_cache=False)
                lg = out.logits[0, :-2].float()
                nll = float(torch.nn.functional.cross_entropy(lg, torch.from_numpy(seq[1:-1]).to(dev)))
                top1 = float((lg.argmax(-1).cpu().numpy() == seq[1:-1]).mean())
                del out, lg
                ix = np.ix_(ids, ids)
                pair_counts[ix] += 1
                order_counts[ix] += (pos[None, :] < pos[:, None]).astype(np.int32)
                pending.append({"cell": i, "row": int(rows[i]), "seq_len": int(lens[i]), "n_union_genes": int(len(ids)),
                                "nll_next_gene": nll, "top1_next_gene": top1,
                                "layer8_mass_above_diagonal": acc._cell_upper, "seconds": time.time() - t1})
            attn_sum += acc.flush()
            meta["per_cell"].extend(pending); pending = []
            ci = blk[-1] + 1
            n_blocks_done += 1
            block_times.append(time.time() - tb)
            if n_blocks_done % SAVE_EVERY_BLOCKS == 0 or ci >= N_CTRL:
                save_state(ci)
                log(f"cells done {ci}/{N_CTRL}; block {block_times[-1]:.1f}s; free {gb:.1f} GB; saved")
    acc.remove()
    if meta["next_cell"] != ci:
        save_state(ci)
    meta["chunks"].append({"start_cell": start_cell, "end_cell": ci, "seconds": time.time() - t_start,
                           "stop_reason": stop_reason, "device": dev,
                           "available_memory_gb_at_end": hv2.available_memory_gb(),
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    save_json(meta, mp)
    log(f"chunk end: cells {start_cell}->{ci} in {time.time() - t_start:.0f}s ({stop_reason})")


# =============================================================================================== finalize
def cmd_finalize() -> None:
    t0 = time.time()
    D_EDGE.mkdir(parents=True, exist_ok=True)
    sp, mp = _state_paths()
    meta = json.load(open(mp))
    assert meta["next_cell"] == N_CTRL, "extraction not finished"
    st = np.load(sp)
    S, PC, C = st["attn_sum"], st["pair_counts"].astype(np.int64), st["order_counts"].astype(np.int64)
    z, gs, ut = load_prep()
    Gu = len(ut)
    E = np.where(PC[None] > 0, S / np.maximum(PC, 1)[None], 0.0)
    off = ~np.eye(Gu, dtype=bool)
    m = off & (PC > 0)
    chk = {"n_cells": N_CTRL, "union_genes": Gu,
           "C_plus_CT_equals_pair_counts_offdiag": bool(np.array_equal((C + C.T)[off], PC[off])),
           "diag_C_zero": bool(np.all(np.diag(C) == 0))}
    # second way for the counts: rebuild presence from the saved tokens; diagonal = cells containing the gene,
    # off-diagonal pair counts = P^T P
    tok2u = np.full(20275, -1, dtype=np.int64)
    tok2u[ut["maxtoki_token_id"].to_numpy(np.int64)] = np.arange(Gu)
    Pm = np.zeros((N_CTRL, Gu), dtype=np.float32)
    for i in range(N_CTRL):
        u = tok2u[z["tokens"][i, :z["lens"][i]]]
        Pm[i, u[u >= 0]] = 1.0
    chk["pair_counts_equal_tokens_PtP"] = bool(np.array_equal(np.rint(Pm.T @ Pm).astype(np.int64), PC))
    # zero pattern at every layer: E[P,T] > 0 exactly when C[P,T] > 0 (causal mask)
    chk["zero_pattern_by_layer"] = []
    for li in range(E.shape[0]):
        chk["zero_pattern_by_layer"].append({"layer": li,
                                             "C0_but_E_pos": int(((C == 0) & (E[li] > 0) & m).sum()),
                                             "Cpos_but_E0": int(((C > 0) & (E[li] == 0) & m).sum())})
    chk["n_offdiag_pairs_n_pair_gt0"] = int(m.sum())
    pcd = meta["per_cell"]
    up = np.array([c["layer8_mass_above_diagonal"] for c in pcd])
    chk["max_layer8_attention_mass_above_diagonal_any_cell"] = float(up.max())
    nll = np.array([c["nll_next_gene"] for c in pcd])
    t1 = np.array([c["top1_next_gene"] for c in pcd])
    rng = np.random.default_rng(20261002)
    bm = nll[rng.integers(0, len(nll), (10000, len(nll)))].mean(1)
    chk["nll_next_gene_mean"] = float(nll.mean())
    chk["nll_next_gene_ci95_cell_bootstrap"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    chk["top1_next_gene_mean"] = float(t1.mean())
    chk["n_encoding_check_blocks"] = len(meta["encoding_checks"])
    chk["n_cells_encoding_checked_before_forward"] = int(sum(e["n_checked"] for e in meta["encoding_checks"]))
    chk["all_encoding_checks_pass"] = bool(all(e["all_pass"] for e in meta["encoding_checks"]))
    # slices: new gene set (main) and deployed gene set (encoding-only comparison)
    for tag, col in [("new", "in_new_set"), ("deployed_genes", "in_deployed_set")]:
        sel = np.flatnonzero(ut[col].to_numpy(bool))
        assert len(sel) == N_HVG
        ix = np.ix_(sel, sel)
        suffix = "" if tag == "new" else "_deployed_genes"
        np.save(D_EDGE / f"attention_edges_layer_mean{suffix}.npy", E[:, sel][:, :, sel].astype(np.float32))
        np.save(D_EDGE / f"attention_pair_counts{suffix}.npy", PC[ix].astype(np.int32))
        np.save(D_EDGE / f"order_counts{suffix}.npy", C[ix].astype(np.int32))
    # the union-level arrays are kept so that every slice can be rebuilt
    np.save(D_EDGE / "union_attention_edges_layer8.npy", E[PRIMARY_LAYER].astype(np.float32))
    np.save(D_EDGE / "union_pair_counts.npy", PC.astype(np.int32))
    np.save(D_EDGE / "union_order_counts.npy", C.astype(np.int32))
    # check the new-set slice is in the same gene order as prep/gene_features.csv
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    sel = np.flatnonzero(ut["in_new_set"].to_numpy(bool))
    chk["new_slice_gene_order_matches_gene_features"] = bool(np.array_equal(ut["var_idx"].to_numpy()[sel], gf["var_idx"].to_numpy()))
    dep_tab = pd.read_csv(P0 / "hvg_gene_table.csv")
    seld = np.flatnonzero(ut["in_deployed_set"].to_numpy(bool))
    chk["deployed_slice_gene_order_matches_deployed_table"] = bool(np.array_equal(ut["var_idx"].to_numpy()[seld], dep_tab["var_idx"].to_numpy()))
    chk["pair_counts_median_new"] = float(np.median(PC[np.ix_(sel, sel)][PC[np.ix_(sel, sel)] > 0]))
    chk["seconds"] = time.time() - t0
    save_json(chk, D_EDGE / "finalize_check.json")
    log(json.dumps({k: v for k, v in chk.items() if k != "zero_pattern_by_layer"}, indent=1))
    log("zero pattern:", chk["zero_pattern_by_layer"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "prep":
        cmd_prep()
    elif a[0] == "extract":
        cmd_extract(float(a[1]) if len(a) > 1 else 7.0)
    elif a[0] == "finalize":
        cmd_finalize()
    else:
        raise SystemExit(__doc__)
