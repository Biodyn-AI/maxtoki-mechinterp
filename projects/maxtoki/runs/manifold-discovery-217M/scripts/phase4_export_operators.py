"""Phase 4 — export per-head operator library from MaxToki-217M checkpoint.

For Llama, the per-head value/output operator we use as the manifold-discovery
operator is the slice of `o_proj` (output projection) corresponding to head h.
That projection maps the head's contribution `value_h ∈ ℝ^{head_dim}` back into
the residual hidden space `ℝ^{hidden_size}`, so each per-head operator has shape
`(head_dim, hidden_size) = (154, 1232)` for MaxToki-217M.

Equivalently we materialise the *head-specific* output map:
    A_{ℓ,h} ∈ ℝ^{head_dim × hidden_size}
        = o_proj_ℓ.weight[:, h*head_dim : (h+1)*head_dim]^T
We also materialise per-block pooled means:
    A_early = mean_{ℓ ∈ early} A_{ℓ,*pooled across heads*}
where 'pooled across heads' means concatenation of per-head outputs back into
one (n_heads * head_dim, hidden_size) matrix and averaged across layers in the
block. Block partition for an 11-layer stack: early {0,1,2,3}, mid {4,5,6,7},
late {8,9,10}.

No forward pass through the model is needed; this is a checkpoint-export step.

Outputs (under runs/manifold-discovery-217M/artifacts/operators/):
  layer{L}_head{H}.npy        per-head operator A ∈ R^(154, 1232)
  pooled_drift_components.npz A_early, A_mid, A_late ∈ R^(8*154, 1232)  (for 217M)
  operator_index.json         metadata (block partition, shapes, file paths)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "setup"))

import torch
from transformers import LlamaForCausalLM

OUT = RUN / "artifacts/operators"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJ / "setup/MaxToki-217M-HF"

# Block partition for 11-layer MaxToki-217M (analogous to scGPT's 1-4/5-8/9-12
# but adjusted to 11 layers). early {0..3}, mid {4..7}, late {8..10}.
EARLY = list(range(0, 4))
MID = list(range(4, 8))
LATE = list(range(8, 11))


def main():
    print("=" * 70)
    print("PHASE 4 — operator library export from MaxToki-217M-HF")
    print("=" * 70)
    t_phase = time.time()

    print(f"\n[1] Loading MaxToki-217M-HF (CPU; only needed for o_proj weights)...")
    t0 = time.time()
    model = LlamaForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads  # 1232 / 8 = 154
    hidden_size = model.config.hidden_size
    print(f"  loaded in {time.time()-t0:.1f}s. n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}, hidden={hidden_size}")
    assert n_layers == 11 and n_heads == 8 and head_dim == 154 and hidden_size == 1232

    print(f"\n[2] Extracting per-head operators A_{{l,h}} ∈ R^({head_dim}, {hidden_size})...")
    per_head: dict[tuple[int, int], np.ndarray] = {}
    for li in range(n_layers):
        attn = model.model.layers[li].self_attn
        # In transformers, o_proj.weight is (hidden_size, hidden_size). The
        # row-block columns correspond to head outputs concatenated along d_v.
        W = attn.o_proj.weight.detach().cpu().numpy()  # (hidden, hidden)
        # For each head h: head_input = (head_dim,), head_contribution = head_input @ W[:, h*head_dim:(h+1)*head_dim].T
        # so the operator A_{l,h} that maps head value -> residual is W[:, h*head_dim:(h+1)*head_dim].T
        # of shape (head_dim, hidden_size). NB: torch's o_proj is `out = x @ W.T` for x = concat(heads).
        # The slice of W along the OUTPUT dimension (rows) is wrong; the slice we want is along the
        # INPUT dimension (columns of W.T = rows of W along axis-0)? Let's spell it out:
        #   o_proj computes: y = concat_h(z_h) @ W.T  where W has shape (hidden, hidden)
        #   = sum_h z_h @ W.T[h*head_dim:(h+1)*head_dim, :]
        #   = sum_h z_h @ W[:, h*head_dim:(h+1)*head_dim].T
        # So per-head map = W[:, h*head_dim:(h+1)*head_dim].T, shape (head_dim, hidden_size).
        for hi in range(n_heads):
            slc = W[:, hi * head_dim : (hi + 1) * head_dim].T.astype(np.float32)
            assert slc.shape == (head_dim, hidden_size), f"got {slc.shape}"
            per_head[(li, hi)] = slc
            np.save(OUT / f"layer{li:02d}_head{hi}.npy", slc)
    print(f"  exported {len(per_head)} per-head operators")

    print(f"\n[3] Building pooled drift components (A_early, A_mid, A_late)...")
    def pool_block(layer_indices: list[int]) -> np.ndarray:
        """Concatenate heads for each layer into (n_heads*head_dim, hidden_size),
        then mean across layers in this block."""
        per_layer_concat = []
        for li in layer_indices:
            blocks = [per_head[(li, hi)] for hi in range(n_heads)]  # each (head_dim, hidden)
            per_layer_concat.append(np.concatenate(blocks, axis=0))  # (n_heads*head_dim, hidden)
        return np.mean(np.stack(per_layer_concat, axis=0), axis=0).astype(np.float32)

    A_early = pool_block(EARLY)
    A_mid = pool_block(MID)
    A_late = pool_block(LATE)
    print(f"  shapes: early {A_early.shape}, mid {A_mid.shape}, late {A_late.shape}")
    np.savez_compressed(
        OUT / "pooled_drift_components.npz",
        A_early=A_early,
        A_mid=A_mid,
        A_late=A_late,
        early_layers=np.array(EARLY, dtype=np.int32),
        mid_layers=np.array(MID, dtype=np.int32),
        late_layers=np.array(LATE, dtype=np.int32),
    )

    print(f"\n[4] Writing operator_index.json...")
    index = {
        "model": "MaxToki-217M-HF",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "operator_definition": (
            "A_{l,h} = o_proj_l.weight[:, h*head_dim:(h+1)*head_dim].T  "
            "(maps head value vector -> residual hidden contribution)"
        ),
        "per_head_files": {
            f"L{li}H{hi}": f"layer{li:02d}_head{hi}.npy"
            for li in range(n_layers)
            for hi in range(n_heads)
        },
        "pooled_drift_components_file": "pooled_drift_components.npz",
        "block_partition": {"early": EARLY, "mid": MID, "late": LATE},
        "total_per_head_operators": len(per_head),
    }
    (OUT / "operator_index.json").write_text(json.dumps(index, indent=2))

    print(f"\nPhase 4 complete in {time.time()-t_phase:.1f}s")
    print(f"Operator library at {OUT.relative_to(RUN)}/")


if __name__ == "__main__":
    main()
