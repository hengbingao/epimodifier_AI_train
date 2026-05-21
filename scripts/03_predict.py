#!/usr/bin/env python3
"""
scripts/03_predict.py
=====================
Load a trained model and predict BEP effects on any loci.

Use cases
---------
  A. Predict all 9 treatment BEPs at K562 dCas9 peak loci
  B. Predict a specific BEP at user-supplied BED loci
  C. Simulate: what if we use BEP100_ZIM3 in K562?

Usage
-----
  # Predict all BEPs at K562 training loci (Stage 3 model)
  python scripts/03_predict.py \
      --config  configs/config.yaml \
      --ckpt    checkpoints/stage3_best.pt \
      --cell    K562 \
      --all_beps \
      --out     outputs/K562_predictions.tsv

  # Predict one BEP at custom BED loci
  python scripts/03_predict.py \
      --config  configs/config.yaml \
      --ckpt    checkpoints/stage2_best.pt \
      --cell    HEK293T \
      --bep     BEP100_ZIM3 \
      --peaks   my_target_sites.bed \
      --out     outputs/ZIM3_at_target_sites.tsv
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset   import BedgraphCache, query_signal, load_allc, query_meth
from models.model   import BEPPerturbationModel, MODALITY_NAMES


def load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def extract_features(
    chrom: str, center: int, cfg: dict, cell: str,
    bg_cache: BedgraphCache, norms: dict,
) -> dict:
    dc     = cfg["data"][cell.lower()]
    window = cfg["data"]["peak_window"]
    n_bins = cfg["data"]["n_signal_bins"]
    marks  = cfg["histone_marks"]
    ctrl   = cfg["control_bep"]

    # dCas9
    dcas9_df  = bg_cache.get(dc["dcas9_bg"].format(root=dc["root"]))
    dcas9_raw = query_signal(dcas9_df, chrom, center, window, n_bins)
    dcas9_n   = norms["dcas9"].transform(dcas9_raw[np.newaxis])[0]

    # ATAC baseline (GFP)
    atac_bins_list = []
    for rep in (1, 2):
        p  = dc["atac_template"].format(root=dc["root"], bep=ctrl, rep=rep)
        df = bg_cache.get(p)
        atac_bins_list.append(query_signal(df, chrom, center, window, n_bins))
    atac_raw = np.mean(atac_bins_list, axis=0)
    atac_n   = norms["atac"].transform(atac_raw[np.newaxis])[0]

    # Methylation
    meth_df    = load_allc(dc["methyl_template"].format(root=dc["root"], bep=ctrl))
    meth_b, gm = query_meth(meth_df, chrom, center, window, n_bins)
    meth_n, gm_n = norms["meth"].transform(
        meth_b[np.newaxis], np.array([[gm]])
    )

    # Histone baseline (25 marks, GFP)
    hist_means = np.zeros(len(marks), dtype=np.float32)
    for j, mark in enumerate(marks):
        sigs = []
        for rep in (1, 2):
            p  = dc["histone_template"].format(
                root=dc["root"], bep=ctrl, mark=mark, rep=rep)
            df = bg_cache.get(p)
            sigs.append(query_signal(df, chrom, center, window, n_bins).mean())
        hist_means[j] = float(np.mean(sigs))
    hist_n = norms["hist"].transform(hist_means[np.newaxis])[0]

    return {
        "dcas9_signal": torch.tensor(dcas9_n,          dtype=torch.float32),
        "dcas9_scalar": torch.tensor([dcas9_n.mean(), dcas9_n.max()], dtype=torch.float32),
        "atac_bins":    torch.tensor(atac_n,            dtype=torch.float32),
        "atac_scalar":  torch.tensor([atac_n.mean()],   dtype=torch.float32),
        "meth_bins":    torch.tensor(meth_n[0],         dtype=torch.float32),
        "glob_meth":    torch.tensor([float(gm_n[0,0])],dtype=torch.float32),
        "hist_ctrl":    torch.tensor(hist_n,            dtype=torch.float32),
    }


def predict_batch(model, feats_list, bep_id, role_id, device):
    def stack(key):
        return torch.stack([f[key] for f in feats_list]).to(device)
    bep_t  = torch.tensor([bep_id]  * len(feats_list), dtype=torch.long, device=device)
    role_t = torch.tensor([role_id] * len(feats_list), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(None,
                    stack("dcas9_signal"), stack("dcas9_scalar"),
                    stack("atac_bins"),    stack("atac_scalar"),
                    stack("meth_bins"),    stack("glob_meth"),
                    stack("hist_ctrl"),
                    bep_t, role_t)
    return {k: v.cpu().numpy() for k, v in out.items() if hasattr(v, "numpy")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--ckpt",     required=True)
    parser.add_argument("--cell",     default="K562",
                        help="Cell type for input features (HEK293T | K562)")
    parser.add_argument("--peaks",    default=None,
                        help="BED file of loci (default: dCas9 peaks from config)")
    parser.add_argument("--bep",      default=None,
                        help="Predict single BEP (e.g. BEP100_ZIM3)")
    parser.add_argument("--all_beps", action="store_true",
                        help="Predict all treatment BEPs")
    parser.add_argument("--out",      default="outputs/predictions.tsv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, args.ckpt, device)
    print(f"Model loaded  ({args.cell}, device={device})")

    # Load normalizers
    ckpt_dir  = Path(cfg["logging"]["checkpoint_dir"])
    norm_path = ckpt_dir / f"normalizers_{args.cell.lower()}.pkl"
    if not norm_path.exists():
        norm_path = ckpt_dir / "normalizers_hek293t.pkl"
    with open(norm_path, "rb") as f:
        norms = pickle.load(f)
    print(f"Normalizers: {norm_path}")

    # Peaks
    dc    = cfg["data"][args.cell.lower()]
    peaks_file = args.peaks or dc["dcas9_peaks"].format(root=dc["root"])
    peaks = pd.read_csv(
        peaks_file, sep="\t", header=None, usecols=[0,1,2],
        names=["chrom","start","end"],
        dtype={"chrom":str,"start":int,"end":int}, comment="#",
    )
    peaks["center"] = (peaks["start"] + peaks["end"]) // 2
    print(f"Loci: {len(peaks):,} from {peaks_file}")

    # BEPs
    role_map  = {"control": 0, "repressor": 1, "activator": 2}
    bep_meta  = {b["name"]: b for b in cfg["beps"]}
    ctrl_name = cfg["control_bep"]
    treat_beps = [b["name"] for b in cfg["beps"] if b["role"] != "control"]

    if args.all_beps:
        target_beps = treat_beps
    elif args.bep:
        target_beps = [args.bep]
    else:
        target_beps = treat_beps

    bg_cache = BedgraphCache()
    marks    = cfg["histone_marks"]

    # Extract features
    print("Extracting features …")
    all_feats = []
    for _, row in peaks.iterrows():
        f = extract_features(row["chrom"], int(row["center"]),
                             cfg, args.cell, bg_cache, norms)
        all_feats.append(f)

    # Predict
    rows = []
    bs   = args.batch_size
    for bep in target_beps:
        meta   = bep_meta[bep]
        bid    = meta["index"]
        rid    = role_map[meta["role"]]
        print(f"  Predicting {bep} …")
        h_all, cls_all, a_all, r_all, attn_all = [], [], [], [], []

        for i in range(0, len(all_feats), bs):
            batch = all_feats[i:i+bs]
            out   = predict_batch(model, batch, bid, rid, device)
            h_all.append(out["hist_log2fc"])
            cls_all.append(np.argmax(out["hist_cls"], -1))
            a_all.append(out["atac_log2fc"])
            r_all.append(out["rna_log2fc"])
            attn_all.append(out["modal_attn"])

        h_mat   = np.concatenate(h_all)
        cls_mat = np.concatenate(cls_all)
        a_vec   = np.concatenate(a_all).ravel()
        r_vec   = np.concatenate(r_all).ravel()
        attn_mat = np.concatenate(attn_all)
        cls_labels = ["down", "nc", "up"]

        for li, (_, row) in enumerate(peaks.iterrows()):
            rec = {
                "chrom": row["chrom"], "start": row["start"],
                "end": row["end"], "center": row["center"],
                "BEP": bep, "role": meta["role"],
                "atac_log2fc": round(float(a_vec[li]), 4),
                "rna_log2fc":  round(float(r_vec[li]), 4),
            }
            for ji, mark in enumerate(marks):
                rec[f"log2fc_{mark}"] = round(float(h_mat[li, ji]), 4)
                rec[f"cls_{mark}"]    = cls_labels[int(cls_mat[li, ji])]
            for mi, mn in enumerate(MODALITY_NAMES):
                rec[f"attn_{mn}"] = round(float(attn_mat[li, mi]), 4)
            rows.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\nPredictions saved → {args.out}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
