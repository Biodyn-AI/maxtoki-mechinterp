"""hooks_v2 -- correct residual-stream edits for MaxToki-217M SAE features.

Why this file exists
--------------------
The deployed intervention scripts (exhaustive-mapping, circuit-tracing) had two
hook bugs:
  1. Block offset. They read ``hidden_states[l]`` (the INPUT of block l) and
     wrote it back as the OUTPUT of block l with a forward hook. That deletes
     block l (its attention and MLP never reach the residual stream).
  2. Overwrite. Each hook returned a fixed tensor taken from the clean run.
     A later hook therefore erased every earlier edit (AB == B, ABC == C).
  3. (L11 only) ``hidden_states[11]`` is already final-normed. Writing it back
     as the output of block 10 made the model apply the final RMSNorm twice.

This module edits the residual stream exactly where each SAE reads it, and
computes the SAE code from the LIVE tensor, so edits at several layers stack.

Layer convention (transformers 5.5.4, MaxToki-217M, 11 decoder blocks)
---------------------------------------------------------------------
``model(..., output_hidden_states=True).hidden_states`` has 12 entries:
  * ``hidden_states[0]``        = token embeddings = input of block 0.
  * ``hidden_states[l]``, 1..10 = output of block l-1 = input of block l.
  * ``hidden_states[11]``       = model.model.norm(output of block 10)
                                  = input of ``model.lm_head``.
  (utils/output_capturing.py:108-113 and :259-263; models/llama/modeling_llama.py
  :405-421 and :480-482. Checked numerically by test_hooks_v2.py.)

The SAE in runs/sae-atlas-217M/outputs/phase1/layer_XX/ was trained on
``hidden_states[XX]`` (full_12layer_pipeline.py:140-141). So the SAE "site" is:
  * l in 0..10 -> forward PRE-hook on ``model.model.layers[l]`` (its input);
  * l == 11    -> forward PRE-hook on ``model.lm_head`` (after the final norm;
                  nothing re-normalises the edited tensor).

SAE convention (setup/topk_sae.py)
----------------------------------
  z     = TopK_k( W_enc (x - mu) + b_enc ), negatives clamped to 0, k = 32
  x_hat = W_dec z + mu,   W_dec columns have unit L2 norm (renormalised each step)
There is no separate pre-bias besides mu. Because decode is linear, the
residual-stream contribution of feature f at a position is z_f * W_dec[:, f].

Edit semantics (all deltas are ADDED to the live tensor x)
----------------------------------------------------------
  * Ablate(layer, feats):  delta = - sum_f z_f(x) * W_dec[:, f]
        (same delta as the deployed "zero the code and take the difference of
        two decodes"; the SAE reconstruction error is kept, not substituted.)
  * Steer(layer, f, alpha, mode="scale")  [default; matches experiment3]:
        delta = (alpha - 1) * z_f(x) * W_dec[:, f]
        Deployed code did z_f -> alpha * z_f and took decode(z') - decode(z),
        which is exactly this. alpha = 1 is an exact no-op. It only acts where
        the feature is active (z_f > 0).
    Steer(..., mode="add"): delta = alpha * W_dec[:, f] at every edited
        position (a fixed direction, independent of z). NOT what experiment3
        did; offered for other designs.
  * AddVector(layer, v):   delta = v (broadcast); used in tests.
  * ZeroDelta(layer):      delta = 0 exactly; used in tests.
Deliberate differences from the deployed code: (a) the delta is added to the
input of block l (the tensor the SAE was trained on) instead of replacing the
output of block l; (b) z is computed from the live tensor inside the forward
pass, not from a cached clean run. For a single edit at one layer, (b) gives
the same z as the clean run, so the only change is (a). (c) at L11 the edit is
applied after the final norm, so the norm is applied once.

If several edits target the SAME layer, z is computed ONCE from that layer's
live input and all their deltas are summed (order does not matter).

Read-outs
---------
``ResidualEditor(capture=[...])`` records the live input of each listed layer:
``captured[l]["pre"]`` (before any edit at l) and ``captured[l]["post"]``
(after the edits at l; the same tensor if there is no edit at l).
Do NOT use ``output_hidden_states`` under edits: torch passes the EDITED args
to forward hooks, so transformers records the post-edit tensor for index 0
but the pre-edit tensor for indices 1..10 (and a pre-edit, final-normed tensor
for index 11). Use the captures instead.

Positions: by default every token position is edited, including <bos>/<eos>
(as in the deployed scripts). Pass ``positions=`` (bool mask or index list over
the sequence) to restrict an edit. Batches > 1 work, but padded positions are
edited too unless masked.

Rules for later agents: import from here; you may ADD clearly separated new
functions at the end of this file; never change the existing ones.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

SETUP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SETUP_DIR.parent
SAE_DIR = PROJECT_DIR / "runs/sae-atlas-217M/outputs/phase1"
N_BLOCKS_217M = 11

if str(SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(SETUP_DIR))
from topk_sae import TopKSAE  # noqa: E402


# =============================================================================
# SAE loading and encoding
# =============================================================================
@dataclass
class LoadedSAE:
    """One TopK SAE on a device. ``W_dec`` is (d_model, d_sae)."""
    layer: int
    sae: TopKSAE
    mu: torch.Tensor
    path: Path

    @property
    def W_dec(self) -> torch.Tensor:
        return self.sae.W_dec.weight

    @torch.no_grad()
    def encode(self, x2d: torch.Tensor) -> torch.Tensor:
        """(N, d_model) -> (N, d_sae). Same maths as TopKSAE.encode."""
        return self.sae.encode(x2d.float(), self.mu)

    @torch.no_grad()
    def contribution(self, z: torch.Tensor, feats: Sequence[int]) -> torch.Tensor:
        """Residual-stream contribution of ``feats``: z[:, F] @ W_dec[:, F].T."""
        idx = torch.as_tensor(list(feats), dtype=torch.long, device=z.device)
        return z.index_select(1, idx) @ self.W_dec.index_select(1, idx).T


def load_sae(layer: int, device: str = "cpu", sae_dir: Path = SAE_DIR) -> LoadedSAE:
    """Load runs/sae-atlas-217M/outputs/phase1/layer_XX/sae_final.pt."""
    path = Path(sae_dir) / f"layer_{layer:02d}" / "sae_final.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    with torch.no_grad():
        sae.W_enc.weight.copy_(ckpt["W_enc_weight"])
        sae.W_enc.bias.copy_(ckpt["W_enc_bias"])
        sae.W_dec.weight.copy_(ckpt["W_dec_weight"])
    sae.eval().to(device)
    for p in sae.parameters():
        p.requires_grad_(False)
    return LoadedSAE(layer=layer, sae=sae, mu=ckpt["mu"].float().to(device), path=path)


def load_saes(layers: Iterable[int], device: str = "cpu", sae_dir: Path = SAE_DIR) -> dict:
    return {int(l): load_sae(int(l), device, sae_dir) for l in sorted(set(layers))}


# =============================================================================
# Edit specifications
# =============================================================================
@dataclass
class Edit:
    """Base class. ``layer`` is the SAE layer index 0..n_blocks (11 = post-norm)."""
    layer: int
    # keyword-only, so Ablate(5, [12]) means layer 5, features [12]
    positions: object = field(default=None, kw_only=True)  # None = all; bool mask (T,) or index list

    needs_code = True

    def delta(self, z: torch.Tensor | None, sae: LoadedSAE | None, x2d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class Ablate(Edit):
    """Remove each feature's decoded contribution: delta = -z_F @ W_dec[:, F].T."""
    features: Sequence[int] = field(default_factory=list)

    def delta(self, z, sae, x2d):
        return -sae.contribution(z, self.features)


@dataclass
class Steer(Edit):
    """mode="scale" (default, = experiment3): delta = (alpha-1) z_f W_dec[:, f].
    mode="add": delta = alpha * W_dec[:, f] at every edited position."""
    feature: int = 0
    alpha: float = 1.0
    mode: str = "scale"

    def delta(self, z, sae, x2d):
        d_f = sae.W_dec[:, self.feature]
        if self.mode == "scale":
            return (self.alpha - 1.0) * z[:, self.feature:self.feature + 1] * d_f[None, :]
        if self.mode == "add":
            return self.alpha * d_f[None, :].expand(x2d.shape[0], -1)
        raise ValueError(f"unknown steer mode {self.mode!r}")


@dataclass
class AddVector(Edit):
    """Add a fixed vector (d_model,) or tensor (T, d_model) to the live input."""
    vector: torch.Tensor = None
    needs_code = False

    def delta(self, z, sae, x2d):
        v = self.vector.to(device=x2d.device, dtype=x2d.dtype)
        if v.dim() == 1:
            return v[None, :].expand(x2d.shape[0], -1)
        return v.reshape(x2d.shape)


@dataclass
class ZeroDelta(Edit):
    """Exact no-op edit (delta = 0). For tests."""
    needs_code = False

    def delta(self, z, sae, x2d):
        return torch.zeros_like(x2d)


# =============================================================================
# Hook sites
# =============================================================================
def n_blocks(model) -> int:
    return int(model.config.num_hidden_layers)


def site_module(model, layer: int):
    """Module whose forward INPUT equals hidden_states[layer]."""
    nb = n_blocks(model)
    if 0 <= layer < nb:
        return model.model.layers[layer]
    if layer == nb:
        return model.lm_head
    raise ValueError(f"layer {layer} outside 0..{nb}")


def _position_mask(positions, T: int, B: int, device) -> torch.Tensor | None:
    if positions is None:
        return None
    if isinstance(positions, torch.Tensor) and positions.dtype == torch.bool:
        m = positions.to(device)
    else:
        arr = np.asarray(positions)
        if arr.dtype == bool:
            m = torch.from_numpy(arr).to(device)
        else:
            m = torch.zeros(T, dtype=torch.bool, device=device)
            m[torch.as_tensor(arr, dtype=torch.long, device=device)] = True
    if m.numel() != T:
        raise ValueError(f"position mask has {m.numel()} entries, sequence has {T}")
    return m.reshape(1, T).expand(B, T).reshape(B * T, 1)


class ResidualEditor:
    """Context manager: register pre-hooks for edits and read-outs.

    Example::
        saes = load_saes([0, 5, 11], device="mps")
        with ResidualEditor(model, saes, edits=[Ablate(0, features=[12]),
                                                 Ablate(5, features=[33])],
                            capture=[11]) as ed:
            out = model(input_ids, use_cache=False)
        z11 = ed.codes(11)          # SAE codes of the live L11 input (CPU)

    Attributes after the forward pass:
      captured[l] = {"pre": Tensor, "post": Tensor}  (float32, CPU unless
                    capture_device is set); batch dim kept: (B, T, d_model).
      edit_log[l] = {"n_active": {feature: #positions with z_f > 0},
                     "delta_norm": mean L2 norm of the summed delta per position}
    Each ``with`` block supports one forward pass (captures are overwritten
    by a second pass).
    """

    def __init__(self, model, saes: dict | None = None, edits: Sequence[Edit] = (),
                 capture: Iterable[int] = (), capture_device: str = "cpu"):
        self.model = model
        self.saes = saes or {}
        self.edits_by_layer: dict[int, list[Edit]] = {}
        for e in edits:
            self.edits_by_layer.setdefault(int(e.layer), []).append(e)
            if e.needs_code and int(e.layer) not in self.saes:
                raise KeyError(f"no SAE loaded for layer {e.layer}")
        self.capture = sorted(set(int(l) for l in capture))
        self.capture_device = capture_device
        self.captured: dict[int, dict] = {}
        self.edit_log: dict[int, dict] = {}
        self._handles = []

    # -- hook body -------------------------------------------------------
    def _make_hook(self, layer: int):
        edits = self.edits_by_layer.get(layer, [])
        want_capture = layer in self.capture

        def pre_hook(module, args):
            x = args[0]
            if want_capture:
                pre = x.detach().to(self.capture_device, dtype=torch.float32, copy=True)
                self.captured[layer] = {"pre": pre, "post": pre}
            if not edits:
                return None
            B, T, D = x.shape
            x2d = x.reshape(B * T, D)
            z = None
            if any(e.needs_code for e in edits):
                z = self.saes[layer].encode(x2d)
            total = torch.zeros(B * T, D, device=x.device, dtype=torch.float32)
            n_active = {}
            for e in edits:
                d = e.delta(z, self.saes.get(layer), x2d.float())
                m = _position_mask(e.positions, T, B, x.device)
                if m is not None:
                    d = d * m.to(d.dtype)
                total = total + d
                feats = getattr(e, "features", None) or (
                    [e.feature] if hasattr(e, "feature") else [])
                for f in feats:
                    n_active[int(f)] = int((z[:, int(f)] > 0).sum().item()) if z is not None else -1
            self.edit_log[layer] = {
                "n_active": n_active,
                "delta_norm": float(total.norm(dim=1).mean().item()),
            }
            x_new = x + total.reshape(B, T, D).to(x.dtype)
            if want_capture:
                self.captured[layer]["post"] = x_new.detach().to(
                    self.capture_device, dtype=torch.float32, copy=True)
            return (x_new,) + tuple(args[1:])

        return pre_hook

    def __enter__(self):
        layers = sorted(set(self.edits_by_layer) | set(self.capture))
        for l in layers:
            h = site_module(self.model, l).register_forward_pre_hook(self._make_hook(l))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # -- read-out helpers ------------------------------------------------
    def hidden(self, layer: int, which: str = "post", batch_index: int = 0) -> torch.Tensor:
        """Captured live input of ``layer`` as (T, d_model)."""
        return self.captured[layer][which][batch_index]

    def codes(self, layer: int, which: str = "post", batch_index: int = 0,
              sae: LoadedSAE | None = None) -> torch.Tensor:
        """SAE codes (T, d_sae) of the captured live input of ``layer``."""
        s = sae or self.saes[layer]
        h = self.hidden(layer, which, batch_index)
        dev = s.mu.device
        return s.encode(h.to(dev)).to("cpu")


@torch.no_grad()
def run_with_edits(model, input_ids: torch.Tensor, saes: dict | None = None,
                   edits: Sequence[Edit] = (), capture: Iterable[int] = (),
                   return_logits: bool = True, logits_device: str = "cpu"):
    """One forward pass with edits. Returns (logits or None, editor)."""
    dev = next(model.parameters()).device
    with ResidualEditor(model, saes, edits=edits, capture=capture) as ed:
        out = model(input_ids.to(dev), use_cache=False, return_dict=True)
    logits = out.logits.detach().to(logits_device, dtype=torch.float32) if return_logits else None
    del out
    return logits, ed


def downstream_codes(model, input_ids, saes: dict, edits: Sequence[Edit],
                     read_layers: Iterable[int], which: str = "pre"):
    """SAE codes at ``read_layers`` under ``edits``. Returns {layer: (T, d_sae) CPU}.

    ``which="pre"`` reads the live input BEFORE any edit placed at that same layer
    (the natural choice for measuring an upstream effect)."""
    read_layers = list(read_layers)
    _, ed = run_with_edits(model, input_ids, saes, edits, capture=read_layers, return_logits=False)
    return {l: ed.codes(l, which=which) for l in read_layers}


# =============================================================================
# Utilities (memory guard, cells, provenance)
# =============================================================================
class MemoryGuardError(RuntimeError):
    pass


def available_memory_gb() -> float:
    """macOS free + inactive pages (same definition as psutil 'available')."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = 16384
        first = out.splitlines()[0]
        if "page size of" in first:
            page = int(first.split("page size of")[1].split()[0])
        vals = {}
        for line in out.splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                vals[k.strip()] = int(v.strip().rstrip("."))
        return (vals.get("Pages free", 0) + vals.get("Pages inactive", 0)) * page / 1e9
    except Exception:
        return float("nan")


def check_memory(min_gb: float = 3.0) -> float:
    gb = available_memory_gb()
    if gb == gb and gb < min_gb:  # skip check if nan
        raise MemoryGuardError(f"available memory {gb:.2f} GB < {min_gb} GB; stopping cleanly")
    return gb


def free_device_cache():
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def load_model(device: str = "mps", dtype=torch.float32):
    """MaxToki-217M via the project adapter (eager attention). Returns extractor."""
    from maxtoki_adapter import MaxTokiAttentionExtractor
    return MaxTokiAttentionExtractor(device=device, dtype=dtype)


def load_k562_control_cells(n_cells: int, pool: int = 100, seed: int = 42, max_len: int = 2048):
    """Same K562 non-targeting cells as exhaustive-mapping experiment2 (seed 42,
    100-cell pool, sorted row indices, first ``n_cells`` that tokenize).
    Returns (list[TokenizedCell], list[int] dataset row indices, h5 path)."""
    import pickle
    import h5py
    from dataset_loader import resolve as load_ds, SYM2ENS_PKL
    from maxtoki_adapter import MaxTokiTokenizer
    ds = load_ds("k562")
    rng = np.random.default_rng(seed)
    ctrl_all = np.where(ds.cell_of_interest_mask &
                        np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
    ctrl_idx = np.sort(rng.choice(ctrl_all, size=pool, replace=False))
    need = ctrl_idx[: min(pool, max(n_cells * 2, n_cells + 5))]
    with h5py.File(ds.h5_path, "r") as f:
        X = np.stack([f["X"][int(i), :] for i in need]).astype(np.float32)
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    with open(SYM2ENS_PKL, "rb") as fh:
        sym2ens = pickle.load(fh)
    tok = MaxTokiTokenizer()
    vi, vt, vm = tok.make_var_mapping([sym2ens.get(s) for s in var_symbols])
    cells, rows = [], []
    for row, x in zip(need, X):
        c = tok.tokenize_cell(x, vi, vt, vm, max_len=max_len)
        if c is not None:
            cells.append(c)
            rows.append(int(row))
        if len(cells) == n_cells:
            break
    return cells, rows, str(ds.h5_path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_info(device: str) -> dict:
    import transformers
    return {
        "device": device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hooks_v2_sha256": sha256_file(__file__),
        "model_dir": str(SETUP_DIR / "MaxToki-217M-HF"),
        "model_dtype": "float32",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    tmp.replace(path)
