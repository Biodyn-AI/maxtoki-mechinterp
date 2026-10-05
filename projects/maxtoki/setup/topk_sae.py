"""TopK Sparse Autoencoder (Gao et al. 2024 variant).

Architecture:
  Encoder: h = TopK_k(W_enc · (x − μ) + b_enc)
  Decoder: x̂ = W_dec · h + μ

W_dec columns are unit-normalised after every optimizer step.
Loss: MSE(x, x̂) — no L1 penalty; TopK enforces exact sparsity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Linear(d_model, d_sae, bias=True)
        self.W_dec = nn.Linear(d_sae, d_model, bias=False)
        # Init decoder from encoder transpose, then unit-normalise
        with torch.no_grad():
            self.W_dec.weight.copy_(self.W_enc.weight.T)
            self._normalise_decoder()

    def _normalise_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_dec.weight.div_(norms)

    def encode(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Returns sparse latent h of shape (batch, d_sae)."""
        pre = self.W_enc(x - mu)
        # TopK: keep only top-k activations per sample
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        h = torch.zeros_like(pre)
        h.scatter_(1, topk_idx, topk_vals.clamp(min=0))  # ReLU on kept values
        return h

    def decode(self, h: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return self.W_dec(h) + mu

    def forward(self, x: torch.Tensor, mu: torch.Tensor):
        h = self.encode(x, mu)
        x_hat = self.decode(h, mu)
        return x_hat, h


def train_sae(
    activations_path: Path,
    output_dir: Path,
    d_model: int,
    d_sae: int = None,
    k: int = 32,
    lr: float = 3e-4,
    batch_size: int = 4096,
    n_epochs: int = 4,
    n_train: int = 1_000_000,
    n_eval: int = 100_000,
    device: str = "mps",
    seed: int = 42,
) -> dict:
    """Train a TopK SAE on stored activation tensor. Returns results dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if d_sae is None:
        d_sae = 4 * d_model

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"  Loading activations from {activations_path}...")
    acts = np.load(str(activations_path))  # (n_positions, d_model)
    n_total = acts.shape[0]
    print(f"  total positions: {n_total:,}  d_model={d_model}")

    # Subsample
    if n_total > n_train + n_eval:
        idx = rng.permutation(n_total)
        train_idx = idx[:n_train]
        eval_idx = idx[n_train:n_train + n_eval]
    else:
        split = int(0.9 * n_total)
        train_idx = np.arange(split)
        eval_idx = np.arange(split, n_total)
        n_train = split
        n_eval = n_total - split

    X_train = torch.from_numpy(acts[train_idx].astype(np.float32))
    X_eval = torch.from_numpy(acts[eval_idx].astype(np.float32))
    del acts  # free RAM

    # Pre-compute mean (on train split)
    mu = X_train.mean(dim=0)
    print(f"  train={len(X_train):,}  eval={len(X_eval):,}  d_sae={d_sae}  k={k}")

    # Model
    model = TopKSAE(d_model, d_sae, k).to(device)
    mu_dev = mu.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True,
                        drop_last=False)

    # Training
    log = []
    t0 = time.time()
    step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device)
            x_hat, h = model(xb, mu_dev)
            loss = ((xb - x_hat) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model._normalise_decoder()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
        log.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1),
                    "steps": step, "seconds": time.time() - t0})
        print(f"    epoch {epoch}: loss={log[-1]['loss']:.6f}  "
              f"steps={step}  time={log[-1]['seconds']:.1f}s")

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_ev = X_eval.to(device)
        x_hat_ev, h_ev = model(X_ev, mu_dev)
        resid_var = ((X_ev - x_hat_ev) ** 2).mean().item()
        input_var = ((X_ev - mu_dev) ** 2).mean().item()
        var_explained = 1.0 - resid_var / max(input_var, 1e-12)

        # Dead features (zero activation on eval set)
        h_ev_cpu = h_ev.cpu().numpy()
        alive = (h_ev_cpu > 0).any(axis=0)
        n_alive = int(alive.sum())
        n_dead = d_sae - n_alive

        # Mean absolute pairwise decoder cosine (sample 500 pairs)
        W = model.W_dec.weight.detach().cpu().numpy()  # (d_model, d_sae)
        n_sample_pairs = min(500, d_sae * (d_sae - 1) // 2)
        pair_cos = []
        rng_p = np.random.default_rng(seed + 999)
        for _ in range(n_sample_pairs):
            i, j = rng_p.choice(d_sae, 2, replace=False)
            cos = float(np.abs((W[:, i] * W[:, j]).sum()))
            pair_cos.append(cos)
        mean_dec_cos = float(np.mean(pair_cos))

    results = {
        "d_model": d_model, "d_sae": d_sae, "k": k,
        "n_train": int(len(X_train)), "n_eval": int(len(X_eval)),
        "n_epochs": n_epochs, "lr": lr, "batch_size": batch_size,
        "variance_explained": float(var_explained),
        "n_alive": n_alive, "n_dead": n_dead,
        "dead_rate": float(n_dead / d_sae),
        "mean_abs_decoder_cosine": mean_dec_cos,
        "total_train_seconds": float(time.time() - t0),
        "training_log": log,
    }

    # Save
    torch.save({
        "W_enc_weight": model.W_enc.weight.detach().cpu(),
        "W_enc_bias": model.W_enc.bias.detach().cpu(),
        "W_dec_weight": model.W_dec.weight.detach().cpu(),
        "mu": mu.cpu(),
        "config": {"d_model": d_model, "d_sae": d_sae, "k": k},
    }, output_dir / "sae_final.pt")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  var_explained={var_explained:.4f}  alive={n_alive}/{d_sae}  "
          f"dead={n_dead}  dec_cos={mean_dec_cos:.4f}")
    return results
