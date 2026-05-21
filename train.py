#!/usr/bin/env python3
"""
train.py — BEP Epigenome Perturbation Model Training
=====================================================
Stage 2: Fine-tune BEP perturbation head on HEK293T data
Stage 3: Cross-cell transfer to K562 (adapt ATAC cell embedding)

Usage
-----
  # Stage 2 (HEK293T training)
  python train.py --config configs/config.yaml --stage 2

  # Stage 3 (K562 transfer, requires Stage 2 checkpoint)
  python train.py --config configs/config.yaml --stage 3 \
      --resume checkpoints/stage2_best.pt

  # Both stages sequentially
  python train.py --config configs/config.yaml --stage all

  # Resume mid-training
  python train.py --config configs/config.yaml --stage 2 \
      --resume checkpoints/stage2_epoch_0030.pt

  # Skip data re-parsing (use cached data.pkl)
  python train.py --config configs/config.yaml --stage 2 --skip_data
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset   import BEPDatasetBuilder, build_dataloaders, build_bep_role_idx
from models.model   import BEPPerturbationModel
from utils.training import (
    build_and_apply_normalisers, BEPLoss, evaluate,
    build_scheduler, unpack,
)
from utils.interpret import (
    compute_shap_importance, rank_marks_per_bep,
    dose_response_curves, bep_similarity_matrix,
    insilico_mark_ablation,
)


# ────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "train.log"),
        ],
    )
    return logging.getLogger("bep_train")


def resolve_device(cfg: dict) -> torch.device:
    d = cfg["training"]["device"]
    if d == "auto":
        if torch.cuda.is_available():             return torch.device("cuda")
        if torch.backends.mps.is_available():     return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ────────────────────────────────────────────────────────────────────────────

def get_data(cfg: dict, cell_type: str, ckpt_dir: Path,
             logger: logging.Logger, skip_cache: bool = False) -> dict:
    cache_path = ckpt_dir / f"data_{cell_type}.pkl"

    if skip_cache and cache_path.exists():
        logger.info(f"Loading cached {cell_type} data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Building {cell_type} dataset from raw files …")
    builder = BEPDatasetBuilder(cfg, cell_type=cell_type)
    data = builder.build()
    data, norms = build_and_apply_normalisers(data)

    # Save normalizers alongside data
    norm_path = ckpt_dir / f"normalizers_{cell_type}.pkl"
    with open(norm_path, "wb") as f:
        pickle.dump(norms, f)
    logger.info(f"Normalizers saved → {norm_path}")

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Data cached → {cache_path}")
    return data


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion,
                device, grad_clip, log_int, epoch, logger,
                use_seq: bool = False) -> float:
    model.train()
    total = 0.0
    t0    = time.time()

    for step, batch in enumerate(loader):
        b   = unpack(batch, device)
        seq = b.get("seq_onehot") if use_seq else None

        optimizer.zero_grad()
        out    = model(seq,
                       b["dcas9_signal"], b["dcas9_scalar"],
                       b["atac_bins"],    b["atac_scalar"],
                       b["meth_bins"],    b["glob_meth"],
                       b["hist_ctrl"],
                       b["bep_id"],       b["role_id"])
        losses = criterion(out, b)
        losses["total"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        total += losses["total"].item()

        if (step + 1) % log_int == 0:
            avg = total / (step + 1)
            logger.info(
                f"  E{epoch:03d} step {step+1:04d}/{len(loader):04d}"
                f"  loss={avg:.4f}"
                f"  hist={losses['hist_reg'].item():.3f}"
                f"  atac={losses['atac'].item():.3f}"
                f"  rna={losses['rna'].item():.3f}"
                f"  {int(time.time()-t0)}s"
            )

    return total / len(loader)


# ────────────────────────────────────────────────────────────────────────────
# Interpretability report
# ────────────────────────────────────────────────────────────────────────────

def run_interpret(model, test_loader, device, cfg, out_dir: Path, logger):
    marks      = cfg["histone_marks"]
    bep_to_idx, bep_role_idx = build_bep_role_idx(cfg)
    treat_beps = [b["name"] for b in cfg["beps"] if b["role"] != "control"]

    # 1. SHAP importance
    logger.info("Computing gradient×input feature importance …")
    shap_res = compute_shap_importance(model, test_loader, device, marks)
    with open(out_dir / "shap_importance.json", "w") as f:
        json.dump({
            "histone_marks": {marks[i]: float(v) for i, v in
                              enumerate(shap_res["histone_marks"])},
            "atac": shap_res["atac"],
            "methylation": shap_res["methylation"],
        }, f, indent=2)
    logger.info("Top 5 marks by GxI importance:")
    idx_sorted = np.argsort(shap_res["histone_marks"])[::-1]
    for i in idx_sorted[:5]:
        logger.info(f"  {marks[i]}: {shap_res['histone_marks'][i]:.4f}")

    # 2. Sensitivity ranking
    logger.info("Computing sensitivity index per BEP …")
    rankings = rank_marks_per_bep(
        model, test_loader, device, marks, [b["name"] for b in cfg["beps"]],
        bep_to_idx, treat_beps,
    )
    with open(out_dir / "sensitivity_rankings.json", "w") as f:
        json.dump({bep: [{"mark": m, "SI": round(v, 4)} for m, v in ranked]
                   for bep, ranked in rankings.items()}, f, indent=2)
    for bep, ranked in rankings.items():
        top3 = ", ".join(f"{m}({v:.3f})" for m, v in ranked[:3])
        logger.info(f"  {bep}: {top3}")

    # 3. Dose-response curves
    logger.info("Computing dose-response curves …")
    ref_batch = next(iter(test_loader))
    dose_results = {}
    for bep in treat_beps:
        bid  = bep_to_idx[bep]
        rid  = [b["role"] for b in cfg["beps"] if b["name"] == bep][0]
        role_map = {"control": 0, "repressor": 1, "activator": 2}
        doses, h_curves, a_curves, r_curves = dose_response_curves(
            model, device, ref_batch, bid, role_map[rid]
        )
        dose_results[bep] = {
            "doses_log2": doses.tolist(),
            "hist_log2fc": h_curves.tolist(),  # (n_doses, n_marks)
            "atac_log2fc": a_curves.tolist(),
            "rna_log2fc":  r_curves.tolist(),
            "mark_names":  marks,
        }
    with open(out_dir / "dose_response.json", "w") as f:
        json.dump(dose_results, f, indent=2)

    # 4. BEP similarity matrix
    logger.info("Computing BEP similarity matrix …")
    sim = bep_similarity_matrix(model, test_loader, device, treat_beps, bep_to_idx)
    with open(out_dir / "bep_similarity.json", "w") as f:
        json.dump({
            "bep_names": treat_beps,
            "cosine_similarity": sim.tolist(),
        }, f, indent=2)

    # 5. In-silico ablation
    logger.info("Running in-silico mark ablation …")
    ablation_results = {}
    for bep in treat_beps[:3]:   # top 3 BEPs for speed
        bid  = bep_to_idx[bep]
        rid  = role_map[[b["role"] for b in cfg["beps"] if b["name"]==bep][0]]
        scores = insilico_mark_ablation(
            model, device, {k: v[:1] for k, v in ref_batch.items()
                            if hasattr(v, "__len__")},
            marks, bid, rid
        )
        ablation_results[bep] = scores
    with open(out_dir / "ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info(f"Interpretability outputs → {out_dir}")


# ────────────────────────────────────────────────────────────────────────────
# Stage 2: HEK293T training
# ────────────────────────────────────────────────────────────────────────────

def run_stage2(cfg: dict, args, logger, device, ckpt_dir, out_dir):
    logger.info("=" * 60)
    logger.info("STAGE 2: BEP perturbation head training (HEK293T)")
    logger.info("=" * 60)

    # Data
    data = get_data(cfg, "hek293t", ckpt_dir, logger,
                    skip_cache=args.skip_data)
    tr_loader, va_loader, te_loader = build_dataloaders(cfg, data)

    # Model
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    model = BEPPerturbationModel(cfg, use_pretrained_backbone=not args.no_pretrain)
    model.freeze_backbone()
    model = model.to(device)
    counts = model.get_param_count()
    logger.info(
        f"Parameters: total={counts['total']:,} "
        f"trainable={counts['trainable']:,} frozen={counts['frozen']:,}"
    )

    tc      = cfg["training"]["stage2"]
    crit    = BEPLoss(
        w_hist=tc["loss_weights"]["histone"],
        w_atac=tc["loss_weights"]["atac"],
        w_meth=tc["loss_weights"]["methylation"],
        w_rna =tc["loss_weights"]["rna"],
    )
    opt     = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
    )
    sched   = build_scheduler(opt, cfg, "stage2", len(tr_loader))

    start_epoch = 1
    best_r2     = -np.inf
    patience_ctr = 0
    patience    = tc["early_stopping_patience"]
    mark_names  = cfg["histone_marks"]
    history     = []

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_r2     = ckpt.get("best_r2", -np.inf)
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, tc["num_epochs"] + 1):
        tr_loss = train_epoch(model, tr_loader, opt, sched, crit, device,
                              tc["gradient_clip"], cfg["logging"]["log_interval"],
                              epoch, logger)
        if sched and not isinstance(sched, torch.optim.lr_scheduler.OneCycleLR):
            sched.step()

        val_m = evaluate(model, va_loader, crit, device, mark_names)
        val_r2 = val_m["global/r2_mean"]

        logger.info(
            f"Epoch {epoch:03d}/{tc['num_epochs']}  "
            f"train={tr_loss:.4f}  val_loss={val_m['loss/total']:.4f}  "
            f"R²={val_r2:.4f}  Pearson={val_m['global/pearson_mean']:.4f}  "
            f"ATAC_r2={val_m.get('atac/r2', 0):.3f}  "
            f"RNA_ρ={val_m.get('rna/spearman', 0):.3f}"
        )
        attn_str = "  ".join(
            f"{n}={val_m.get(f'modal_attn/{n}', 0):.3f}"
            for n in ["dcas9", "atac", "methylation", "histone", "seq_summary"]
        )
        logger.info(f"  Modality attention: {attn_str}")

        history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})
        state = dict(epoch=epoch, model=model.state_dict(),
                     optimizer=opt.state_dict(), best_r2=best_r2, cfg=cfg)

        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_ctr = 0
            torch.save(state, ckpt_dir / "stage2_best.pt")
            logger.info(f"  ✓ New best R² = {val_r2:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info("Early stopping triggered.")
                break

        if epoch % cfg["logging"]["save_every_n_epochs"] == 0:
            torch.save(state, ckpt_dir / f"stage2_epoch_{epoch:04d}.pt")

    # Test evaluation
    logger.info("Loading best checkpoint for test evaluation …")
    best = torch.load(ckpt_dir / "stage2_best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_m = evaluate(model, te_loader, crit, device, mark_names)
    logger.info(f"Test R²={test_m['global/r2_mean']:.4f}  "
                f"Pearson={test_m['global/pearson_mean']:.4f}")

    (out_dir / "stage2").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage2" / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(out_dir / "stage2" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    run_interpret(model, te_loader, device, cfg,
                  out_dir / "stage2", logger)
    return model


# ────────────────────────────────────────────────────────────────────────────
# Stage 3: K562 cross-cell transfer
# ────────────────────────────────────────────────────────────────────────────

def run_stage3(cfg: dict, args, logger, device, ckpt_dir, out_dir,
               hek_model: BEPPerturbationModel = None):
    logger.info("=" * 60)
    logger.info("STAGE 3: Cross-cell transfer (K562)")
    logger.info("=" * 60)

    # Load model
    if hek_model is None:
        resume = args.resume or str(ckpt_dir / "stage2_best.pt")
        if not os.path.exists(resume):
            raise FileNotFoundError(
                f"Stage 2 checkpoint not found: {resume}. "
                "Run stage 2 first, or pass --resume path/to/stage2_best.pt"
            )
        import warnings; warnings.filterwarnings("ignore", category=UserWarning)
        model = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
        ckpt  = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Stage 2 weights loaded from {resume}")
    else:
        model = hek_model

    # Freeze everything except cell_adapter + output heads
    model.freeze_for_transfer()
    model = model.to(device)
    counts = model.get_param_count()
    logger.info(f"Transfer trainable parameters: {counts['trainable']:,}")

    # K562 data
    data_k562 = get_data(cfg, "k562", ckpt_dir, logger, skip_cache=args.skip_data)
    tr_loader, va_loader, te_loader = build_dataloaders(cfg, data_k562)

    tc   = cfg["training"]["stage3"]
    crit = BEPLoss(
        w_hist=cfg["training"]["stage2"]["loss_weights"]["histone"],
        w_atac=cfg["training"]["stage2"]["loss_weights"]["atac"],
        w_meth=cfg["training"]["stage2"]["loss_weights"]["methylation"],
        w_rna =cfg["training"]["stage2"]["loss_weights"]["rna"],
    )
    opt  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tc["learning_rate"], weight_decay=tc["weight_decay"],
    )
    sched = build_scheduler(opt, cfg, "stage3", len(tr_loader))

    best_r2 = -np.inf
    mark_names = cfg["histone_marks"]
    history    = []

    for epoch in range(1, tc["num_epochs"] + 1):
        tr_loss = train_epoch(model, tr_loader, opt, sched, crit, device,
                              tc["gradient_clip"], cfg["logging"]["log_interval"],
                              epoch, logger)
        val_m  = evaluate(model, va_loader, crit, device, mark_names)
        val_r2 = val_m["global/r2_mean"]

        logger.info(
            f"[K562] Epoch {epoch:03d}/{tc['num_epochs']}  "
            f"train={tr_loss:.4f}  R²={val_r2:.4f}  "
            f"Pearson={val_m['global/pearson_mean']:.4f}"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})

        state = dict(epoch=epoch, model=model.state_dict(),
                     optimizer=opt.state_dict(), best_r2=best_r2, cfg=cfg)
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(state, ckpt_dir / "stage3_best.pt")
            logger.info(f"  ✓ K562 best R² = {val_r2:.4f}")

    # Test
    best = torch.load(ckpt_dir / "stage3_best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_m = evaluate(model, te_loader, crit, device, mark_names)
    logger.info(f"[K562 Test] R²={test_m['global/r2_mean']:.4f}")

    (out_dir / "stage3").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage3" / "test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(out_dir / "stage3" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    run_interpret(model, te_loader, device, cfg,
                  out_dir / "stage3", logger)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BEP Epigenome Editing Model")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--stage",        default="2",
                        choices=["2", "3", "all"],
                        help="Training stage: 2=HEK293T, 3=K562, all=both")
    parser.add_argument("--resume",       default=None)
    parser.add_argument("--skip_data",    action="store_true",
                        help="Load cached data.pkl instead of re-parsing")
    parser.add_argument("--no_pretrain",  action="store_true",
                        help="Use stub backbone instead of pretrained Enformer")
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--device",       type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.epochs: cfg["training"]["stage2"]["num_epochs"] = args.epochs
    if args.lr:     cfg["training"]["stage2"]["learning_rate"] = args.lr
    if args.device: cfg["training"]["device"] = args.device

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    out_dir  = Path(cfg["logging"]["output_dir"])
    log_dir  = cfg["logging"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    device = resolve_device(cfg)
    logger.info(f"Device: {device}")
    logger.info(f"Stage: {args.stage}")

    hek_model = None
    if args.stage in ("2", "all"):
        hek_model = run_stage2(cfg, args, logger, device, ckpt_dir, out_dir)

    if args.stage in ("3", "all"):
        run_stage3(cfg, args, logger, device, ckpt_dir, out_dir, hek_model)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
