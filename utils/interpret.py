"""
utils/interpret.py
------------------
Interpretability tools:
  1. SHAP feature importance  — which baseline signals drive BEP sensitivity
  2. Sensitivity index        — per-BEP mark ranking
  3. Dose-response curves     — dCas9 intensity vs mark response
  4. BEP comparison matrix    — pairwise similarity of BEP effects
  5. In-silico perturbation   — set one mark to 0 and measure prediction change
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 1. SHAP feature importance
# ────────────────────────────────────────────────────────────────────────────

def compute_shap_importance(
    model: nn.Module,
    loader,
    device: torch.device,
    mark_names: List[str],
    n_background: int = 100,
    n_test: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Gradient × Input approximation of SHAP for the 25 histone baseline marks.
    Returns dict: {"histone_marks": (n_marks,), "atac": scalar, "meth": scalar}

    For full SHAP, install the shap library and replace with:
        import shap
        explainer = shap.DeepExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_inputs)
    """
    model.eval()
    # Gradient × Input: ∂L/∂input_j × input_j
    hist_grad_x_input = []
    atac_grad_x_input = []
    meth_grad_x_input = []

    count = 0
    for batch in loader:
        if count >= n_test:
            break
        b = {k: v.to(device) for k, v in batch.items()}

        # Enable gradients for input features
        h = b["hist_ctrl"].clone().detach().requires_grad_(True)
        a = b["atac_bins"].clone().detach().requires_grad_(True)
        m = b["meth_bins"].clone().detach().requires_grad_(True)

        out = model(
            None,
            b["dcas9_signal"], b["dcas9_scalar"],
            a,                 b["atac_scalar"],
            m,                 b["glob_meth"],
            h,
            b["bep_id"],       b["role_id"],
        )

        # Gradient of summed hist_log2fc w.r.t. inputs
        loss = out["hist_log2fc"].abs().mean()
        loss.backward()

        with torch.no_grad():
            # Gradient × Input (|GxI| averaged across batch)
            gi_h = (h.grad * h).abs().mean(0).cpu().numpy()     # (n_marks,)
            gi_a = (a.grad * a).abs().mean().item()              # scalar
            gi_m = (m.grad * m).abs().mean().item()              # scalar

        hist_grad_x_input.append(gi_h)
        atac_grad_x_input.append(gi_a)
        meth_grad_x_input.append(gi_m)
        count += b["hist_ctrl"].size(0)

    return {
        "histone_marks": np.stack(hist_grad_x_input).mean(0),   # (n_marks,)
        "atac":          float(np.mean(atac_grad_x_input)),
        "methylation":   float(np.mean(meth_grad_x_input)),
    }


# ────────────────────────────────────────────────────────────────────────────
# 2. Sensitivity index per BEP
# ────────────────────────────────────────────────────────────────────────────

def sensitivity_index(fc_arr: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """
    SI = mean(|log2FC|) × fraction(|log2FC| > thr)
    fc_arr: (N, n_marks)
    """
    abs_fc = np.abs(fc_arr)
    return abs_fc.mean(0) * (abs_fc > thr).mean(0)


def rank_marks_per_bep(
    model: nn.Module,
    loader,
    device: torch.device,
    mark_names: List[str],
    bep_names: List[str],
    bep_to_idx: Dict[str, int],
    treat_beps: List[str],
    thr: float = 0.5,
) -> Dict[str, List[Tuple[str, float]]]:
    """Returns {bep_name: [(mark_name, SI), ...] sorted descending}."""
    model.eval()
    bep_preds: Dict[int, List[np.ndarray]] = {
        bep_to_idx[b]: [] for b in treat_beps
    }

    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            out  = model(None,
                         b["dcas9_signal"], b["dcas9_scalar"],
                         b["atac_bins"],    b["atac_scalar"],
                         b["meth_bins"],    b["glob_meth"],
                         b["hist_ctrl"],
                         b["bep_id"],       b["role_id"])
            pred = out["hist_log2fc"].cpu().numpy()
            bids = b["bep_id"].cpu().numpy()
            for i, bid in enumerate(bids):
                if bid in bep_preds:
                    bep_preds[bid].append(pred[i])

    idx_to_name = {v: k for k, v in bep_to_idx.items()}
    results = {}
    for bid, preds in bep_preds.items():
        if not preds:
            continue
        arr  = np.stack(preds)
        si   = sensitivity_index(arr, thr)
        ranked = sorted(zip(mark_names, si.tolist()),
                        key=lambda x: x[1], reverse=True)
        results[idx_to_name[bid]] = ranked
    return results


# ────────────────────────────────────────────────────────────────────────────
# 3. Dose-response curves
# ────────────────────────────────────────────────────────────────────────────

def dose_response_curves(
    model: nn.Module,
    device: torch.device,
    ref_batch: dict,
    bep_id_int: int,
    role_id_int: int,
    n_doses: int = 25,
    dose_range: Tuple[float, float] = (-2.0, 2.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep dCas9 signal intensity (log2 scale factors) and record predictions.
    Returns:
      doses          : (n_doses,)
      hist_curves    : (n_doses, n_marks)
      atac_curves    : (n_doses,)
      rna_curves     : (n_doses,)
    """
    model.eval()
    doses = np.linspace(*dose_range, n_doses)

    ref_ds = ref_batch["dcas9_signal"][[0]].to(device)
    ref_ab = ref_batch["atac_bins"][[0]].to(device)
    ref_as = ref_batch["atac_scalar"][[0]].to(device)
    ref_mb = ref_batch["meth_bins"][[0]].to(device)
    ref_gm = ref_batch["glob_meth"][[0]].to(device)
    ref_h  = ref_batch["hist_ctrl"][[0]].to(device)
    bep_t  = torch.tensor([bep_id_int],  device=device)
    role_t = torch.tensor([role_id_int], device=device)

    h_curves, a_curves, r_curves = [], [], []
    with torch.no_grad():
        for log2_s in doses:
            scale    = 2.0 ** log2_s
            scaled   = ref_ds * scale
            dc_sc    = torch.tensor([[scaled.mean().item(), scaled.max().item()]],
                                    device=device)
            out = model(None, scaled, dc_sc, ref_ab, ref_as,
                        ref_mb, ref_gm, ref_h, bep_t, role_t)
            h_curves.append(out["hist_log2fc"].cpu().numpy()[0])
            a_curves.append(out["atac_log2fc"].cpu().numpy()[0, 0])
            r_curves.append(out["rna_log2fc"].cpu().numpy()[0, 0])

    return doses, np.stack(h_curves), np.array(a_curves), np.array(r_curves)


# ────────────────────────────────────────────────────────────────────────────
# 4. BEP comparison matrix
# ────────────────────────────────────────────────────────────────────────────

def bep_similarity_matrix(
    model: nn.Module,
    loader,
    device: torch.device,
    treat_beps: List[str],
    bep_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Compute pairwise cosine similarity of predicted histone log2FC vectors
    across BEPs (averaged over loci).
    Returns: (n_treat, n_treat) cosine similarity matrix.
    """
    model.eval()
    bep_mean: Dict[int, List[np.ndarray]] = {
        bep_to_idx[b]: [] for b in treat_beps
    }
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            out = model(None,
                        b["dcas9_signal"], b["dcas9_scalar"],
                        b["atac_bins"],    b["atac_scalar"],
                        b["meth_bins"],    b["glob_meth"],
                        b["hist_ctrl"],
                        b["bep_id"],       b["role_id"])
            pred = out["hist_log2fc"].cpu().numpy()
            bids = b["bep_id"].cpu().numpy()
            for i, bid in enumerate(bids):
                if bid in bep_mean:
                    bep_mean[bid].append(pred[i])

    vecs = []
    for b in treat_beps:
        arr = np.stack(bep_mean[bep_to_idx[b]])  # (N, n_marks)
        vecs.append(arr.mean(0))                  # (n_marks,)

    vecs = np.stack(vecs)  # (n_treat, n_marks)
    # Cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vn    = vecs / norms
    return vn @ vn.T   # (n_treat, n_treat)


# ────────────────────────────────────────────────────────────────────────────
# 5. In-silico mark ablation
# ────────────────────────────────────────────────────────────────────────────

def insilico_mark_ablation(
    model: nn.Module,
    device: torch.device,
    ref_batch: dict,
    mark_names: List[str],
    bep_id_int: int,
    role_id_int: int,
) -> Dict[str, float]:
    """
    For each histone mark, zero it out and measure the change in
    predicted |hist_log2fc| sum. Larger drop → mark is more causal.
    Returns dict: {mark_name: importance_score}
    """
    model.eval()
    b = ref_batch
    device_kw = dict(device=device)

    def run(hist):
        out = model(
            None,
            b["dcas9_signal"].to(device), b["dcas9_scalar"].to(device),
            b["atac_bins"].to(device),    b["atac_scalar"].to(device),
            b["meth_bins"].to(device),    b["glob_meth"].to(device),
            hist,
            torch.tensor([bep_id_int],  **device_kw),
            torch.tensor([role_id_int], **device_kw),
        )
        return out["hist_log2fc"].abs().sum().item()

    h_base  = b["hist_ctrl"][:1].to(device)
    baseline_score = run(h_base)

    scores = {}
    with torch.no_grad():
        for j, mark in enumerate(mark_names):
            h_ablated = h_base.clone()
            h_ablated[:, j] = 0.0
            ablated_score = run(h_ablated)
            scores[mark] = float(baseline_score - ablated_score)

    return scores
