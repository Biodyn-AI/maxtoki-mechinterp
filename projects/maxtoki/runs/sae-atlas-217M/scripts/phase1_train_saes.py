"""Phase 1 — Train TopK SAEs on MaxToki-217M activations (3 layers)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
from topk_sae import train_sae

PHASE0 = PROJ / "runs/sae-atlas-217M/outputs/phase0"
OUT = PROJ / "runs/sae-atlas-217M/outputs/phase1"

import json, os, torch

DEVICE = os.environ.get("SAE_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
cfg = json.loads((PHASE0 / "run_config.json").read_text())
HIDDEN = cfg["hidden_size"]
LAYERS = cfg["target_layers"]

print("=" * 70)
print(f"SAE Phase 1 — TopK SAE training  d_model={HIDDEN}  layers={LAYERS}")
print("=" * 70)

for li in LAYERS:
    act_path = PHASE0 / f"layer_{li:02d}_activations.npy"
    if not act_path.exists():
        print(f"  SKIP L{li} (no activations)")
        continue
    out_dir = OUT / f"layer_{li:02d}"
    print(f"\n--- Training SAE for layer {li} ---")
    results = train_sae(
        activations_path=act_path,
        output_dir=out_dir,
        d_model=HIDDEN,
        d_sae=4 * HIDDEN,       # 4928
        k=32,
        lr=3e-4,
        batch_size=4096,
        n_epochs=4,
        n_train=1_000_000,
        n_eval=100_000,
        device=DEVICE,
    )

print(f"\nSAE PHASE 1 COMPLETE")
