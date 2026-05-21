#!/usr/bin/env python3
"""
scripts/02_check_and_visualize.py
==================================
A. Validate all input files before training.
B. Generate result figures after training.

Usage
-----
  # Validate inputs
  python scripts/02_check_and_visualize.py --mode check \
      --config configs/config.yaml

  # Generate all result figures
  python scripts/02_check_and_visualize.py --mode figures \
      --config configs/config.yaml \
      --results outputs/results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# ── ANSI ────────────────────────────────────────────────────────────────────
OK   = "\033[92m[OK]  \033[0m"
ERR  = "\033[91m[ERR] \033[0m"
WARN = "\033[93m[WARN]\033[0m"
BOLD = "\033[1m";  RST = "\033[0m"


# ────────────────────────────────────────────────────────────────────────────
# A. Input Validation
# ────────────────────────────────────────────────────────────────────────────

def check_file(path: str, label: str, required: bool = True) -> bool:
    p = Path(path)
    if p.exists():
        size = p.stat().st_size / 1e6
        print(f"  {OK} {label}: {size:.1f} MB → {p}")
        return True
    sym = ERR if required else WARN
    print(f"  {sym} {label}: NOT FOUND → {path}")
    return not required


def check_bedgraph_format(path: str) -> bool:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.split("\t")
                if len(cols) < 4:
                    print(f"      {ERR} < 4 columns: {line[:60]!r}")
                    return False
                float(cols[3])
                return True
    except Exception as e:
        print(f"      {ERR} read error: {e}")
        return False
    return True


def run_check(cfg: dict):
    marks = cfg["histone_marks"]
    beps  = [b["name"] for b in cfg["beps"]]
    errors = 0

    print(f"\n{BOLD}{'='*60}{RST}")
    print(f"{BOLD}BEP Pipeline — Input Validation{RST}")
    print(f"{BOLD}{'='*60}{RST}\n")

    # Genome
    print(f"{BOLD}[1] Genome files{RST}")
    genome = cfg["data"]["genome"]
    for label, key in [("FASTA", "fasta"), ("GTF", "annotation_gtf"),
                        ("Blacklist", "blacklist"), ("Chrom sizes", "chrom_sizes")]:
        ok = check_file(genome[key], label, required=(key != "blacklist"))
        if not ok and key != "blacklist":
            errors += 1
    fai = genome["fasta"] + ".fai"
    if not Path(fai).exists():
        print(f"  {WARN} .fai index missing → run: samtools faidx {genome['fasta']}")

    # HEK293T
    print(f"\n{BOLD}[2] HEK293T histone bedgraphs ({len(beps)} BEPs × {len(marks)} marks × 2 reps){RST}")
    hek = cfg["data"]["hek293t"]
    missing = []
    for bep in beps:
        bep_ok = True
        for mark in marks:
            for rep in (1, 2):
                p = hek["histone_template"].format(root=hek["root"], bep=bep, mark=mark, rep=rep)
                if not Path(p).exists():
                    missing.append(p); bep_ok = False
        n_found = sum(
            Path(hek["histone_template"].format(
                root=hek["root"], bep=bep, mark=m, rep=r)
            ).exists()
            for m in marks for r in (1, 2)
        )
        sym = OK if bep_ok else WARN
        print(f"  {sym} {bep}: {n_found}/{len(marks)*2} histone files")
    if missing:
        print(f"\n  {WARN} {len(missing)} missing files. First 5:")
        for p in missing[:5]:
            print(f"       {p}")
    if len(missing) > len(marks) * 2:
        errors += 1

    # ATAC
    print(f"\n{BOLD}[3] ATAC-seq bedgraphs{RST}")
    for cell_key, cell_beps in [("hek293t", beps), ("k562", [beps[0]])]:
        dc = cfg["data"][cell_key]
        cell_name = cell_key.upper()
        for bep in cell_beps:
            for rep in (1, 2):
                p = dc["atac_template"].format(root=dc["root"], bep=bep, rep=rep)
                check_file(p, f"{cell_name}/{bep}/ATAC_rep{rep}", required=False)

    # Methylation
    print(f"\n{BOLD}[4] CG methylation (allc){RST}")
    for cell_key in ("hek293t", "k562"):
        dc = cfg["data"][cell_key]
        p  = dc["methyl_template"].format(root=dc["root"], bep=beps[0])
        check_file(p, f"{cell_key.upper()} GFP methylation")

    # dCas9
    print(f"\n{BOLD}[5] dCas9 ChIP-seq{RST}")
    for cell_key in ("hek293t", "k562"):
        dc = cfg["data"][cell_key]
        for key, label in [("dcas9_bg", "bedgraph"), ("dcas9_peaks", "peaks BED")]:
            p = dc[key].format(root=dc["root"])
            ok = check_file(p, f"{cell_key.upper()} dCas9 {label}")
            if not ok and cell_key == "hek293t":
                errors += 1

    # RNA
    print(f"\n{BOLD}[6] RNA-seq (HEK293T only){RST}")
    hek = cfg["data"]["hek293t"]
    for bep in beps:
        for rep in (1, 2):
            p = hek["rna_template"].format(root=hek["root"], bep=bep, rep=rep)
            check_file(p, f"{bep}_rep{rep}", required=False)

    # Summary
    print(f"\n{BOLD}{'='*60}{RST}")
    if errors == 0:
        print(f"{OK}{BOLD}All required files found. Ready to train.{RST}")
        print(f"\n  Next: python train.py --config {args.config} --stage 2\n")
    else:
        print(f"{ERR}{BOLD}{errors} error(s). Fix before training.{RST}\n")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────
# B. Figures
# ────────────────────────────────────────────────────────────────────────────

def run_figures(cfg: dict, results_root: str):
    """Generate all post-training figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.stats import spearmanr
    except ImportError:
        print("matplotlib/scipy required: pip install matplotlib scipy")
        sys.exit(1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "pdf.fonttype": 42,
    })

    marks   = cfg["histone_marks"]
    results = Path(results_root)
    figs    = results / "figures"
    figs.mkdir(exist_ok=True)

    # ── Per-stage figures ──────────────────────────────────────────────────
    for stage in ("stage2", "stage3"):
        stage_dir = results / stage
        if not stage_dir.exists():
            continue
        label = "HEK293T" if stage == "stage2" else "K562"
        print(f"\nGenerating {label} ({stage}) figures …")

        # Fig 1: Training history
        hist_path = stage_dir / "training_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            epochs     = [h["epoch"] for h in history]
            train_loss = [h["train_loss"] for h in history]
            val_r2     = [h.get("global/r2_mean", 0) for h in history]
            val_pearson= [h.get("global/pearson_mean", 0) for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(epochs, train_loss, label="Train loss", color="#4878CF")
            ax1.set(xlabel="Epoch", ylabel="Loss", title=f"{label} training loss")
            ax1.legend()
            ax2.plot(epochs, val_r2, label="Val R²", color="#6ACC65")
            ax2.plot(epochs, val_pearson, label="Val Pearson", color="#B47CC7")
            ax2.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1),
                    title=f"{label} regression quality")
            ax2.legend()
            fig.tight_layout()
            fig.savefig(figs / f"{stage}_training.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  → {stage}_training.pdf")

        # Fig 2: Per-mark R² bar chart
        test_path = stage_dir / "test_metrics.json"
        if test_path.exists():
            with open(test_path) as f:
                tm = json.load(f)
            r2s    = [tm.get(f"{m}/r2", 0) for m in marks]
            colors = ["#4878CF" if v>=0.5 else "#F8A538" if v>=0.2 else "#D65F5F"
                      for v in r2s]
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.bar(range(len(marks)), r2s, color=colors)
            ax.set(xticks=range(len(marks)), ylabel="R² (test)",
                   title=f"{label} per-mark prediction accuracy")
            ax.set_xticklabels(marks, rotation=45, ha="right", fontsize=8)
            ax.axhline(np.mean(r2s), ls="--", color="gray", lw=0.8,
                       label=f"Mean={np.mean(r2s):.3f}")
            ax.axhline(0, ls="-", color="black", lw=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figs / f"{stage}_per_mark_r2.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  → {stage}_per_mark_r2.pdf")

        # Fig 3: Sensitivity heatmap
        sens_path = stage_dir / "sensitivity_rankings.json"
        if sens_path.exists():
            with open(sens_path) as f:
                sens = json.load(f)
            bep_list  = sorted(sens.keys())
            mat = np.zeros((len(bep_list), len(marks)))
            for i, bep in enumerate(bep_list):
                si_d = {d["mark"]: d["SI"] for d in sens[bep]}
                for j, m in enumerate(marks):
                    mat[i, j] = si_d.get(m, 0)
            fig, ax = plt.subplots(figsize=(15, max(3, len(bep_list) * 0.8)))
            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                           vmin=0, vmax=np.percentile(mat, 95))
            plt.colorbar(im, ax=ax, label="Sensitivity Index", shrink=0.6)
            ax.set(xticks=range(len(marks)), yticks=range(len(bep_list)),
                   title=f"{label}: mark sensitivity per BEP")
            ax.set_xticklabels(marks, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([b.split("_", 1)[1] for b in bep_list])
            fig.tight_layout()
            fig.savefig(figs / f"{stage}_sensitivity_heatmap.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  → {stage}_sensitivity_heatmap.pdf")

        # Fig 4: SHAP importance
        shap_path = stage_dir / "shap_importance.json"
        if shap_path.exists():
            with open(shap_path) as f:
                shap = json.load(f)
            mark_shap  = [shap["histone_marks"].get(m, 0) for m in marks]
            extra_shap = {"ATAC": shap["atac"], "Methylation": shap["methylation"]}
            all_vals   = mark_shap + list(extra_shap.values())
            all_labels = marks + list(extra_shap.keys())
            colors = ["#4878CF"] * len(marks) + ["#D65F5F", "#6ACC65"]
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.bar(range(len(all_labels)), all_vals, color=colors)
            ax.set(xticks=range(len(all_labels)),
                   ylabel="|Gradient × Input|",
                   title=f"{label}: Feature importance (baseline signal → BEP effect)")
            ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color="#4878CF", label="Histone marks"),
                                Patch(color="#D65F5F", label="ATAC"),
                                Patch(color="#6ACC65", label="Methylation")])
            fig.tight_layout()
            fig.savefig(figs / f"{stage}_shap_importance.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  → {stage}_shap_importance.pdf")

        # Fig 5: Dose-response curves (top 4 marks per BEP)
        dose_path = stage_dir / "dose_response.json"
        if dose_path.exists():
            with open(dose_path) as f:
                dose = json.load(f)
            beps_list = list(dose.keys())
            n_bep = len(beps_list)
            fig, axes = plt.subplots(1, n_bep, figsize=(n_bep * 4, 4), sharey=False)
            if n_bep == 1:
                axes = [axes]
            for ax, bep in zip(axes, beps_list):
                doses    = np.array(dose[bep]["doses_log2"])
                h_curves = np.array(dose[bep]["hist_log2fc"])  # (n_doses, n_marks)
                # Plot top 4 marks by amplitude
                amplitudes = h_curves.max(0) - h_curves.min(0)
                top4 = np.argsort(amplitudes)[-4:][::-1]
                for j in top4:
                    ax.plot(doses, h_curves[:, j], label=marks[j], lw=1.5)
                ax.axhline(0, ls="--", color="gray", lw=0.5)
                ax.set(xlabel="log₂(dCas9 intensity scale)",
                       ylabel="Predicted log₂FC",
                       title=bep.split("_", 1)[1])
                ax.legend(fontsize=7)
            fig.suptitle(f"{label}: Dose-response curves", y=1.02)
            fig.tight_layout()
            fig.savefig(figs / f"{stage}_dose_response.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  → {stage}_dose_response.pdf")

        # Fig 6: BEP similarity matrix
        sim_path = stage_dir / "bep_similarity.json"
        if sim_path.exists():
            with open(sim_path) as f:
                sim_data = json.load(f)
            bep_names = sim_data["bep_names"]
            sim_mat   = np.array(sim_data["cosine_similarity"])
            short     = [b.split("_", 1)[1] for b in bep_names]
            fig, ax = plt.subplots(figsize=(max(5, len(bep_names)), max(5, len(bep_names))))
            im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label="Cosine similarity")
            ax.set(xticks=range(len(short)), yticks=range(len(short)),
                   title=f"{label}: BEP effect similarity")
            ax.set_xticklabels(short, rotation=45, ha="right")
            ax.set_yticklabels(short)
            fig.tight_layout()
            fig.savefig(figs / f"{stage}_bep_similarity.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  → {stage}_bep_similarity.pdf")

    print(f"\nAll figures saved → {figs}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["check", "figures"], default="check")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--results", default="outputs/results",
                        help="Results directory (for --mode figures)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "check":
        run_check(cfg)
    else:
        run_figures(cfg, args.results)


if __name__ == "__main__":
    main()
