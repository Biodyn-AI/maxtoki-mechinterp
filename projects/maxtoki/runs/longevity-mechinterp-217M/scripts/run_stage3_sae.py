"""Stage 3 — SAE pilot with donor-aware feature scoring on MaxToki-217M L5.

Trains a minimal ReLU SAE (latent_dim = 4 × input_dim) on the MaxToki contextual
representation cached by Stage 1, then scores each latent feature for:

  - mean_activation, std_activation, frac_active
  - cell_age_spearman, donor_age_spearman
  - donor_age_perm_p (NaN when n_donors < 20 — pipeline upstream skips
    permutation by design; documented at run_stage3_sae_pilot.py:280)
  - age_eta2, celltype_eta2, donor_eta2

**Important calibration caveat for this run.** Tabula Sapiens immune subset 20k
yields n_donors = 17 after balanced subsampling, which is below the upstream
n_donors >= 20 threshold for donor-level permutation calibration. The
`is_robust_feature` filter in pipeline §6 step 5 therefore cannot fire — this
matches pipeline §10.6 ("Permutation calibration fails at low donor counts").
We still report the descriptive donor-aware scores so the run is informative,
but flag that no features here meet the formal mechanism-grade robust-feature
bar. AIDA-scale execution (≥30–50 donors) is required for that.

Outputs:
    outputs/stage3_<date>/
      ├── sae_artifacts.npz       encoder/decoder weights + normalisation
      ├── feature_scores.csv      per-feature donor-aware scores
      ├── train_history.csv       per-epoch loss / mse / l1
      └── report.md
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))
sys.path.insert(
    0,
    "<REPO_ROOT>/repos/longevity-mechinterp/"
    "implementation/scripts",
)

from run_stage3_sae_pilot import (  # type: ignore  # noqa: E402
    _annotate_decoder_dimensions,
    _donor_aware_feature_scores,
    _train_sae,
)


LATENT_RATIO = 4
EPOCHS = 30
BATCH_SIZE = 1024
LR = 1e-3
L1_COEF = 1e-3
SEED = 42

DONOR_CORR_MIN = 0.3
DONOR_P_MAX = 0.05
MAX_CELLTYPE_ETA2 = 0.5
PERM_ITERS = 500
TOP_N_DECODER = 16


def _resolve_stage1_dir() -> Path:
    candidates = sorted((RUN_ROOT / "outputs").glob("stage1_*"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError("No Stage 1 output dirs")
    return candidates[-1]


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    today = datetime.now().strftime("%Y%m%d")
    stage1_dir = _resolve_stage1_dir()
    stage3_dir = RUN_ROOT / "outputs" / f"stage3_{today}"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage3] reading Stage 1 from {stage1_dir}")
    print(f"[stage3] writing Stage 3 to {stage3_dir}")

    meta = pd.read_csv(stage1_dir / "sampled_obs.csv")
    layer_arr = np.load(stage1_dir / "maxtoki_layer_reps.npy")
    # Shape: (n_cells, n_layer_indices, hidden_size). Stage 1 used layer_indices=(5,)
    if layer_arr.ndim == 3 and layer_arr.shape[1] == 1:
        X_full = layer_arr[:, 0]
    else:
        raise ValueError(f"unexpected layer_arr shape {layer_arr.shape}")

    finite_mask = np.isfinite(X_full).all(axis=1)
    X = X_full[finite_mask].astype(np.float32)
    meta = meta.loc[finite_mask].reset_index(drop=True)
    print(f"[stage3] X shape {X.shape}; n_cells {len(meta)}; "
          f"n_donors {meta['donor_id'].nunique()}; n_celltypes {meta['cell_type'].nunique()}")

    n_donors = int(meta["donor_id"].nunique())
    perm_iters = PERM_ITERS if n_donors >= 20 else 0
    if perm_iters == 0:
        print(f"[stage3] WARNING: n_donors={n_donors} < 20 → donor-permutation calibration "
              f"disabled (pipeline §10.6 / run_stage3_sae_pilot.py:280)")

    input_dim = X.shape[1]
    latent_dim = LATENT_RATIO * input_dim
    print(f"[stage3] training SAE: input_dim={input_dim}, latent_dim={latent_dim}, "
          f"epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, l1_coef={L1_COEF}")
    t0 = time.time()
    res = _train_sae(
        X=X,
        latent_dim=latent_dim,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        l1_coef=L1_COEF,
        device=_device(),
        seed=SEED,
    )
    print(f"[stage3] SAE trained in {time.time()-t0:.1f}s; recon_mse={res.recon_mse:.4f}")

    # Save SAE artefacts.
    np.savez(
        stage3_dir / "sae_artifacts.npz",
        encoder_weight=res.encoder_weight,
        encoder_bias=res.encoder_bias,
        decoder_weight=res.decoder_weight,
        decoder_bias=res.decoder_bias,
        input_mean=res.input_mean,
        input_std=res.input_std,
        latent=res.latent,
        recon_mse=np.array([res.recon_mse]),
    )
    pd.DataFrame(res.train_history).to_csv(stage3_dir / "train_history.csv", index=False)

    # Donor-aware feature scoring.
    print(f"[stage3] scoring {res.latent.shape[1]} latent features...")
    t0 = time.time()
    scores = _donor_aware_feature_scores(
        latent=res.latent, meta=meta, perm_iters=perm_iters, seed=SEED,
    )
    print(f"[stage3] scoring done in {time.time()-t0:.1f}s")

    # Robust-feature filter (will be all-False if n_donors < 20).
    is_robust = (
        (scores["abs_donor_age_spearman"] >= DONOR_CORR_MIN)
        & (scores["donor_age_perm_p"] <= DONOR_P_MAX)
        & (scores["celltype_eta2"] <= MAX_CELLTYPE_ETA2)
        & (scores["frac_active"] >= 0.05)
        & (scores["frac_active"] <= 0.95)
    )
    scores["is_robust_feature"] = is_robust.fillna(False)
    n_robust = int(scores["is_robust_feature"].sum())

    # Decoder annotation for top-N features.
    top_for_anno = scores.nlargest(20, "abs_donor_age_spearman").reset_index(drop=True)
    decoder_anno = _annotate_decoder_dimensions(
        feature_scores=top_for_anno, decoder_weight=res.decoder_weight, top_n=TOP_N_DECODER
    )
    scores = scores.merge(decoder_anno, on="feature_id", how="left")
    scores.to_csv(stage3_dir / "feature_scores.csv", index=False)

    summary = {
        "stage1_dir": str(stage1_dir),
        "n_cells": int(X.shape[0]),
        "n_donors": n_donors,
        "n_celltypes": int(meta["cell_type"].nunique()),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "epochs": EPOCHS,
        "recon_mse": float(res.recon_mse),
        "perm_iters_used": perm_iters,
        "perm_calibration_disabled_low_donor": perm_iters == 0,
        "n_robust_features": n_robust,
        "n_features_total": int(scores.shape[0]),
        "frac_features_active": float((scores["frac_active"] > 0.01).mean()),
        "max_abs_donor_age_spearman": float(scores["abs_donor_age_spearman"].max()),
        "median_abs_donor_age_spearman": float(scores["abs_donor_age_spearman"].median()),
    }
    (stage3_dir / "stage3_summary.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append(f"# Stage 3 — SAE pilot + donor-aware feature scoring ({today})")
    md.append("")
    md.append(f"- **Representation:** MaxToki-217M layer 5 (cached from Stage 1)")
    md.append(f"- **n_cells:** {X.shape[0]}, **n_donors:** {n_donors}, **n_celltypes:** {meta['cell_type'].nunique()}")
    md.append(f"- **SAE:** input_dim={input_dim}, latent_dim={latent_dim} (4× ratio), "
              f"epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, l1_coef={L1_COEF}")
    md.append(f"- **Recon MSE:** {res.recon_mse:.4f}")
    if perm_iters == 0:
        md.append("")
        md.append(f"> **⚠️ Donor-permutation calibration disabled.** n_donors={n_donors} < 20, "
                  f"matching the upstream guard at `run_stage3_sae_pilot.py:280`. The robust-"
                  f"feature filter relies on `donor_age_perm_p ≤ {DONOR_P_MAX}`, which is NaN "
                  f"under this regime — so `is_robust_feature` evaluates to False for every "
                  f"feature here, and no formal robust-feature claim is made on this run. The "
                  f"descriptive `donor_age_spearman` and η² metrics below remain informative.")
    md.append("")
    md.append(f"## Feature score summary")
    md.append("")
    md.append(f"- features with frac_active > 0.01: {summary['frac_features_active']:.2%}")
    md.append(f"- max |donor_age_spearman|: {summary['max_abs_donor_age_spearman']:.4f}")
    md.append(f"- median |donor_age_spearman|: {summary['median_abs_donor_age_spearman']:.4f}")
    md.append(f"- robust features (formal filter): **{n_robust}/{summary['n_features_total']}**")
    md.append("")
    md.append(f"## Top 15 features by |donor_age_spearman|")
    md.append("")
    top_view = scores.nlargest(15, "abs_donor_age_spearman")[
        ["feature_id", "frac_active", "mean_activation",
         "cell_age_spearman", "donor_age_spearman", "donor_age_perm_p",
         "age_eta2", "celltype_eta2", "donor_eta2"]
    ]
    md.append(top_view.to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append(f"## Training history (last 5 epochs)")
    md.append("")
    md.append(pd.DataFrame(res.train_history).tail(5).to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    (stage3_dir / "report.md").write_text("\n".join(md))
    print(f"[stage3] wrote report → {stage3_dir / 'report.md'}")
    print(f"[stage3] n_robust_features = {n_robust} (formal filter)")


if __name__ == "__main__":
    main()
