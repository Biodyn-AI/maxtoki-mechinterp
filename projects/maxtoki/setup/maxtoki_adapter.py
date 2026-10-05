"""MaxToki attention-extraction adapter.

Loads MaxToki HF safetensors (LlamaForCausalLM), exposes per-layer per-head
attention over rank-value-encoded gene-token sequences, and projects
token-space attention back into gene-space for GRN analyses.

Compatible with the attention-GRN pipeline's Phase 0a/0b adapter interface
(see pipelines/attention-grn-extraction-and-evaluation.md).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_DIR = PROJECT_ROOT / "setup"

DEFAULT_MODEL_DIR = SETUP_DIR / "MaxToki-217M-HF"
DEFAULT_TOKEN_DICT = SETUP_DIR / "token_dictionary.json"
DEFAULT_GENE_MEDIAN_PKL = Path(
    "<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/"
    "crispri_validation/data/gene_median_dictionary_gc104M.pkl"
)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TokenizedCell:
    token_ids: np.ndarray      # int64, shape (L,), includes <bos> and <eos>
    gene_positions: np.ndarray # int64, shape (L,); -1 for <bos>/<eos>, else index into n_genes
    n_genes_in_cell: int


class MaxTokiTokenizer:
    """Rank-value encoder for MaxToki.

    Mirrors the Geneformer/scGPT procedure used in biodyn-nmi-paper/src/shared/
    01_unified_extraction.py: normalize by per-gene median, rank in descending
    order, clip to model_input_size - 2, wrap with <bos>/<eos>.
    """

    BOS = 2
    EOS = 3
    PAD = 0
    MASK = 1

    def __init__(
        self,
        token_dict_path: Path = DEFAULT_TOKEN_DICT,
        gene_median_path: Path = DEFAULT_GENE_MEDIAN_PKL,
        model_input_size: int = 4096,
    ):
        with open(token_dict_path) as f:
            full = json.load(f)
        self.full_token_dict: dict[str, int] = full
        # Keep only entries with id < 20275 (matches HF vocab_size for 217M-HF / 1B-HF).
        self.gene_token_dict: dict[str, int] = {
            k: int(v) for k, v in full.items() if k.startswith("ENSG") and v < 20275
        }
        assert all(v < 20275 for v in self.gene_token_dict.values())

        with open(gene_median_path, "rb") as f:
            full_median = pickle.load(f)
        # Filter to MaxToki vocab.
        self.gene_median: dict[str, float] = {
            k: float(v) for k, v in full_median.items() if k in self.gene_token_dict
        }
        missing = set(self.gene_token_dict) - set(self.gene_median)
        if missing:
            raise ValueError(
                f"{len(missing)} MaxToki genes missing from gene_median dict"
            )
        self.model_input_size = int(model_input_size)

    def make_var_mapping(self, var_ensembl_ids: Iterable[str]):
        """For a given AnnData.var ordering, return parallel arrays:
        (var_indices_in_token, token_ids, medians) — one row per gene kept.
        """
        kept_var_idx: list[int] = []
        token_ids: list[int] = []
        medians: list[float] = []
        for i, eid in enumerate(var_ensembl_ids):
            if not isinstance(eid, str):
                continue
            tid = self.gene_token_dict.get(eid)
            if tid is None:
                continue
            kept_var_idx.append(i)
            token_ids.append(tid)
            medians.append(self.gene_median[eid])
        return (
            np.array(kept_var_idx, dtype=np.int64),
            np.array(token_ids, dtype=np.int64),
            np.array(medians, dtype=np.float32),
        )

    def tokenize_cell(
        self,
        expression_row: np.ndarray,
        var_indices: np.ndarray,
        token_ids: np.ndarray,
        medians: np.ndarray,
        max_len: int | None = None,
    ) -> TokenizedCell | None:
        """Rank-value encode one cell's expression row into a token sequence.

        Returns a TokenizedCell whose gene_positions is a parallel array that
        maps each non-special token to its *position within the kept HVG set*
        (== index into `token_ids`). This is what Phase 0 uses to project
        attention back to a gene-space matrix.
        """
        max_len = int(max_len or self.model_input_size)
        if max_len < 3:
            raise ValueError("max_len must be >= 3 to hold <bos> + 1 gene + <eos>")

        expr = expression_row[var_indices]
        nonzero = expr > 0
        if nonzero.sum() == 0:
            return None

        expr_nz = expr[nonzero]
        med_nz = medians[nonzero]
        hvg_positions_nz = np.nonzero(nonzero)[0]  # index into token_ids / HVG set
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = expr_nz / med_nz
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0)
        order = np.argsort(-normalized)
        order = order[: max_len - 2]

        ranked_token_ids = token_ids[nonzero][order]
        ranked_gene_pos = hvg_positions_nz[order]

        seq_tokens = np.concatenate([[self.BOS], ranked_token_ids, [self.EOS]]).astype(
            np.int64
        )
        seq_positions = np.concatenate([[-1], ranked_gene_pos, [-1]]).astype(np.int64)
        return TokenizedCell(
            token_ids=seq_tokens,
            gene_positions=seq_positions,
            n_genes_in_cell=int(len(ranked_token_ids)),
        )


class MaxTokiAttentionExtractor:
    """HF LlamaForCausalLM wrapper with attention extraction and head masking."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        dtype: torch.dtype = torch.float32,
        device: str | None = None,
        attn_implementation: str = "eager",
    ):
        self.model_dir = Path(model_dir)
        self.device = device or _auto_device()
        config = LlamaConfig.from_pretrained(str(self.model_dir))
        self.config = config
        self.n_layers = int(config.num_hidden_layers)
        self.n_heads = int(config.num_attention_heads)
        self.hidden_size = int(config.hidden_size)

        # Load model in requested dtype, on the chosen device, with eager attention
        # so output_attentions works (SDPA/Flash silently drop attentions).
        self.model = LlamaForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.model.to(self.device)
        self.dtype = dtype

    @torch.no_grad()
    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Single-cell forward returning (logits, per-layer attentions).

        Attentions: tuple of length n_layers, each tensor shape
        (batch=1, n_heads, seq_len, seq_len), float32 on CPU.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        # Move attentions off-device promptly so VRAM/unified mem stays stable.
        attn = tuple(a.detach().to("cpu", dtype=torch.float32) for a in out.attentions)
        logits = out.logits.detach().to("cpu", dtype=torch.float32)
        return logits, attn

    # --- Phase 4 head-mask support --------------------------------------------
    # LlamaForCausalLM.forward does not accept a `head_mask` argument.  We
    # implement head masking by zeroing the contiguous head slice in the
    # *input* of each layer's `o_proj` via a forward-pre-hook.  The o_proj
    # input has shape (batch, seq, n_heads * head_dim) and concatenates heads
    # in layout [h=0,...,n_heads-1], so we zero out `[h*head_dim:(h+1)*head_dim]`
    # for every head we want to mask.
    def set_head_mask(self, mask_by_layer: "dict[int, list[int]]") -> None:
        """Install o_proj pre-hooks that zero out the specified heads.

        mask_by_layer: {layer_index: [head_idx, ...]} — any heads listed are
        zeroed out during the forward pass.  Call `clear_head_mask()` to
        remove them.
        """
        self.clear_head_mask()
        head_dim = self.hidden_size // self.n_heads
        self._head_hook_handles: list = []
        for li, layer in enumerate(self.model.model.layers):
            heads_to_mask = mask_by_layer.get(li, [])
            if not heads_to_mask:
                continue
            slices = [(h * head_dim, (h + 1) * head_dim) for h in heads_to_mask]

            def _pre_hook(module, inputs, slices=slices):
                x = inputs[0]
                for a, b in slices:
                    x[..., a:b] = 0.0
                return (x,)

            h = layer.self_attn.o_proj.register_forward_pre_hook(_pre_hook)
            self._head_hook_handles.append(h)

    def clear_head_mask(self) -> None:
        handles = getattr(self, "_head_hook_handles", None)
        if handles:
            for h in handles:
                h.remove()
        self._head_hook_handles = []

    # --- Residual-stream hidden state extraction (spectral-geometry pipeline) -
    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Single-cell forward returning (logits, per-layer hidden states).

        Hidden states: tuple of length (n_layers + 1) — first is the post-
        embedding residual stream, then one per transformer block. Each tensor
        has shape (batch=1, seq_len, hidden_size), float32 on CPU.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
        hidden = tuple(h.detach().to("cpu", dtype=torch.float32) for h in out.hidden_states)
        logits = out.logits.detach().to("cpu", dtype=torch.float32)
        return logits, hidden

    # --- Phase 0b value-weighted edge support --------------------------------
    # Capture the per-head context layer A_vh = softmax(QK^T / sqrt(d)) @ V
    # by hooking each layer's o_proj forward_pre (which receives the
    # reshape-flattened (batch, seq, n_heads*head_dim) tensor — the exact
    # concatenated context vector before output projection).
    @torch.no_grad()
    def forward_with_value_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Single-cell forward returning (logits, per-layer value-context).

        Per-layer tensor shape: (batch=1, n_heads, seq_len, head_dim), float32
        on CPU. This is the A_vh = attention_weights @ V tensor (post-softmax,
        post-value, pre-o_proj), used by Phase 0b to test whether regulatory
        information is encoded in the value pathway.
        """
        head_dim = self.hidden_size // self.n_heads
        captured: list[torch.Tensor | None] = [None] * self.n_layers
        handles = []

        def _make_pre_hook(layer_idx: int):
            def _pre_hook(module, inputs):
                x = inputs[0]  # (batch, seq, hidden_size)
                b, s, _ = x.shape
                # reshape to (batch, n_heads, seq, head_dim) using the layout
                # LlamaAttention emits (each head is a contiguous slice of the
                # last dim).
                captured[layer_idx] = (
                    x.detach()
                     .view(b, s, self.n_heads, head_dim)
                     .permute(0, 2, 1, 3)
                     .contiguous()
                     .to("cpu", dtype=torch.float32)
                )
            return _pre_hook

        for li, layer in enumerate(self.model.model.layers):
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(_make_pre_hook(li))
            )

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for h in handles:
                h.remove()

        logits = out.logits.detach().to("cpu", dtype=torch.float32)
        assert all(c is not None for c in captured), "missing hook capture"
        return logits, tuple(captured)


def smoke_test(
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str | None = None,
) -> dict:
    """Load MaxToki-217M, tokenize a synthetic cell, run one forward pass,
    verify attention tensor shapes, and print a summary.
    """
    import time

    tok = MaxTokiTokenizer()
    print(f"[tok] gene_token_dict entries: {len(tok.gene_token_dict)}")
    print(f"[tok] gene_median entries:     {len(tok.gene_median)}")
    print(f"[tok] model_input_size:        {tok.model_input_size}")

    # Synthetic: pretend we have 1500 HVG genes with expression sampled from
    # a heavy-tailed distribution.
    rng = np.random.default_rng(42)
    n_hvg = 1500
    ens_pool = list(tok.gene_token_dict.keys())
    var_ensembl = rng.choice(ens_pool, size=n_hvg, replace=False).tolist()
    var_idx, tok_ids, medians = tok.make_var_mapping(var_ensembl)
    print(f"[tok] HVG mapped to MaxToki vocab: {len(var_idx)}/{n_hvg}")

    expr = rng.lognormal(mean=0.0, sigma=1.0, size=n_hvg).astype(np.float32)
    expr[rng.random(n_hvg) < 0.5] = 0.0  # simulate dropout
    cell = tok.tokenize_cell(expr, var_idx, tok_ids, medians, max_len=1024)
    assert cell is not None
    print(f"[tok] seq_len: {len(cell.token_ids)} (n_genes_in_cell={cell.n_genes_in_cell})")
    print(f"[tok] first 10 ids: {cell.token_ids[:10].tolist()}")
    print(f"[tok] last  5 ids: {cell.token_ids[-5:].tolist()}")

    t0 = time.time()
    xt = MaxTokiAttentionExtractor(model_dir=model_dir, device=device)
    print(
        f"[model] loaded in {time.time()-t0:.1f}s  "
        f"device={xt.device}  L={xt.n_layers}  H={xt.n_heads}  D={xt.hidden_size}"
    )

    input_ids = torch.from_numpy(cell.token_ids[None, :])
    t1 = time.time()
    logits, attn = xt.forward_with_attention(input_ids)
    t2 = time.time()
    print(f"[fwd]   forward pass {t2-t1:.2f}s  logits {tuple(logits.shape)}")
    print(f"[fwd]   attn layers={len(attn)}  per-layer shape {tuple(attn[0].shape)}")
    assert len(attn) == xt.n_layers
    assert attn[0].shape == (1, xt.n_heads, input_ids.shape[1], input_ids.shape[1])

    # Basic numerical sanity on the attention row sums (causal attn rows sum to 1).
    row_sums = attn[0][0, 0].sum(dim=-1)
    print(
        f"[sanity] layer0 head0 row-sum: mean={row_sums.mean():.4f}  "
        f"min={row_sums.min():.4f}  max={row_sums.max():.4f}"
    )

    return {
        "model": str(model_dir),
        "device": xt.device,
        "n_layers": xt.n_layers,
        "n_heads": xt.n_heads,
        "hidden_size": xt.hidden_size,
        "smoke_seq_len": int(input_ids.shape[1]),
        "smoke_fwd_seconds": float(t2 - t1),
    }


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = p.parse_args()
    try:
        info = smoke_test(model_dir=args.model_dir, device=args.device)
        print("\n[smoke_test PASSED]")
        print(json.dumps(info, indent=2))
    except Exception as e:
        print(f"\n[smoke_test FAILED] {type(e).__name__}: {e}", file=sys.stderr)
        raise
