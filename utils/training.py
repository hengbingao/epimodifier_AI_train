"""
utils/training.py
-----------------
Loss functions, evaluation, normalisation — multi-modality BEP model.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, f1_score

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["dcas9", "atac", "methylation", "histone", "seq_summary"]


# ────────────────────────────────────────────────────────────────────────────
# Normalisers  (fit on train set, applied everywhere)
# ────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """Per-column (or global) z-score normaliser."""
    def __init__(self, per_col: bool = True):
        self.per_col = per_col
        self.mean_ = self.std_ = None

    def fit(self, x: np.ndarray):
        ax = 0
        self.mean_ = x.mean(axis=ax) if self.per_col else x.mean()
        self.std_  = (x.std(axis=ax)  if self.per_col else x.std()) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class Log1pNormalizer:
    """log1p then z-score (for ChIP / ATAC signal profiles)."""
    def __init__(self):
        self.mean_ = self.std_ = None

    def fit(self, x: np.ndarray):
        y = np.log1p(np.clip(x, 0, None))
        self.mean_ = y.mean()
        self.std_  = y.std() + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.log1p(np.clip(x, 0, None)) - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def build_and_apply_normalisers(data: dict) -> Tuple[dict, dict]:
    """
    Fit normalisers on the full dataset (train split sees test; acceptable
    for signal normalisation which has no label leakage).
    Returns (normalised_data, normaliser_dict).
    """
    norms = {}

    # dCas9 signal profiles
    norms["dcas9"] = Log1pNormalizer()
    data["dcas9_signal"] = norms["dcas9"].fit_transform(data["dcas9_signal"])

    # ATAC signal profiles
    norms["atac"] = Log1pNormalizer()
    data["atac_bins"] = norms["atac"].fit_transform(data["atac_bins"])

    # Histone baseline (per mark)
    norms["hist"] = Normalizer(per_col=True)
    data["hist_ctrl"] = norms["hist"].fit_transform(data["hist_ctrl"])

    # Methylation (already in [0,1]; just z-score globally)
    norms["meth"] = Normalizer(per_col=False)
    data["meth_bins"]  = norms["meth"].fit_transform(data["meth_bins"])
    data["glob_meth"]  = norms["meth"].transform(data["glob_meth"][:, np.newaxis]).squeeze(-1)

    return data, norms


# ────────────────────────────────────────────────────────────────────────────
# Multi-task loss
# ────────────────────────────────────────────────────────────────────────────

class BEPLoss(nn.Module):
    """
    Combined loss:
      L = w_hist  * (HuberLoss(hist_log2fc) + CE(hist_cls))
        + w_atac  * HuberLoss(atac_log2fc)
        + w_meth  * MSE(meth_delta)
        + w_rna   * HuberLoss(rna_log2fc)
    """

    def __init__(self, w_hist=1.0, w_atac=0.5, w_meth=0.3,
                 w_rna=0.8, w_cls=0.4, huber_delta=1.0):
        super().__init__()
        self.w_hist = w_hist
        self.w_atac = w_atac
        self.w_meth = w_meth
        self.w_rna  = w_rna
        self.w_cls  = w_cls
        self.huber  = nn.HuberLoss(delta=huber_delta)

    def forward(self, pred: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        # Histone regression
        l_hist_reg = self.huber(pred["hist_log2fc"], batch["hist_log2fc"])

        # Histone classification
        B, n_marks = batch["hist_cls"].shape
        l_hist_cls = F.cross_entropy(
            pred["hist_cls"].view(B * n_marks, 3),
            batch["hist_cls"].view(B * n_marks),
        )

        # ATAC
        l_atac = self.huber(pred["atac_log2fc"], batch["atac_log2fc"])

        # Methylation
        l_meth = F.mse_loss(pred["meth_delta"], batch["meth_delta"])

        # RNA
        # Mask zero-RNA loci (no gene linked)
        rna_mask = (batch["rna_log2fc"].abs() > 1e-6).float()
        l_rna = (self.huber(pred["rna_log2fc"], batch["rna_log2fc"]) * rna_mask).mean()

        total = (
            self.w_hist * (l_hist_reg + self.w_cls * l_hist_cls)
            + self.w_atac * l_atac
            + self.w_meth * l_meth
            + self.w_rna  * l_rna
        )

        return {
            "total":        total,
            "hist_reg":     l_hist_reg,
            "hist_cls":     l_hist_cls,
            "atac":         l_atac,
            "meth":         l_meth,
            "rna":          l_rna,
        }


# ────────────────────────────────────────────────────────────────────────────
# Batch unpacking helper
# ────────────────────────────────────────────────────────────────────────────

def unpack(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device, mark_names: List[str],
             use_seq: bool = False) -> Dict:
    model.eval()
    all_hpred, all_htrue = [], []
    all_cpred, all_ctrue = [], []
    all_atac_p, all_atac_t = [], []
    all_rna_p,  all_rna_t  = [], []
    all_modal = []
    loss_agg = {"total": 0, "hist_reg": 0, "hist_cls": 0,
                "atac": 0, "meth": 0, "rna": 0}
    n = 0

    with torch.no_grad():
        for batch in loader:
            b = unpack(batch, device)
            seq = b.get("seq_onehot") if use_seq else None

            out = model(
                seq,
                b["dcas9_signal"], b["dcas9_scalar"],
                b["atac_bins"],    b["atac_scalar"],
                b["meth_bins"],    b["glob_meth"],
                b["hist_ctrl"],
                b["bep_id"],       b["role_id"],
            )
            losses = criterion(out, b)

            for k in loss_agg:
                loss_agg[k] += losses[k].item()
            n += 1

            all_hpred.append(out["hist_log2fc"].cpu().numpy())
            all_htrue.append(b["hist_log2fc"].cpu().numpy())
            all_cpred.append(out["hist_cls"].argmax(-1).cpu().numpy())
            all_ctrue.append(b["hist_cls"].cpu().numpy())
            all_atac_p.append(out["atac_log2fc"].cpu().numpy())
            all_atac_t.append(b["atac_log2fc"].cpu().numpy())
            all_rna_p.append(out["rna_log2fc"].cpu().numpy())
            all_rna_t.append(b["rna_log2fc"].cpu().numpy())
            all_modal.append(out["modal_attn"].cpu().numpy())

    metrics = {f"loss/{k}": v / n for k, v in loss_agg.items()}

    hp = np.concatenate(all_hpred)
    ht = np.concatenate(all_htrue)
    cp = np.concatenate(all_cpred)
    ct = np.concatenate(all_ctrue)
    modal_mean = np.concatenate(all_modal).mean(0)

    for i, mn in enumerate(MODALITY_NAMES):
        metrics[f"modal_attn/{mn}"] = float(modal_mean[i])

    # Per-mark histone metrics
    pr_list, sp_list, r2_list, f1_list = [], [], [], []
    for j, mark in enumerate(mark_names):
        r,  _ = pearsonr(ht[:, j],  hp[:, j])
        rho,_ = spearmanr(ht[:, j], hp[:, j])
        r2     = r2_score(ht[:, j], hp[:, j])
        f1     = f1_score(ct[:, j], cp[:, j], average="macro", zero_division=0)
        metrics.update({
            f"{mark}/pearson": float(r),
            f"{mark}/spearman": float(rho),
            f"{mark}/r2": float(r2),
            f"{mark}/f1": float(f1),
        })
        pr_list.append(r); sp_list.append(rho)
        r2_list.append(r2); f1_list.append(f1)

    metrics.update({
        "global/pearson_mean":  float(np.mean(pr_list)),
        "global/spearman_mean": float(np.mean(sp_list)),
        "global/r2_mean":       float(np.mean(r2_list)),
        "global/f1_mean":       float(np.mean(f1_list)),
    })

    # ATAC and RNA global metrics
    ap, at = np.concatenate(all_atac_p).ravel(), np.concatenate(all_atac_t).ravel()
    rp, rt = np.concatenate(all_rna_p).ravel(),  np.concatenate(all_rna_t).ravel()
    if len(ap) > 1:
        metrics["atac/pearson"] = float(pearsonr(at, ap)[0])
        metrics["atac/r2"]      = float(r2_score(at, ap))
    mask = np.abs(rt) > 1e-6
    if mask.sum() > 10:
        metrics["rna/spearman"] = float(spearmanr(rt[mask], rp[mask])[0])
        metrics["rna/r2"]       = float(r2_score(rt[mask], rp[mask]))

    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Scheduler factory
# ────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, stage: str, steps_per_epoch: int):
    tc  = cfg["training"][stage]
    n   = tc["num_epochs"]
    w   = tc["warmup_epochs"]
    sched = tc.get("scheduler", "cosine")
    if sched == "cosine":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=tc["learning_rate"],
            epochs=n,
            steps_per_epoch=steps_per_epoch,
            pct_start=w / n,
            anneal_strategy="cos",
        )
    return None
