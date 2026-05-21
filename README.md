# BEP-EpiPredict

**Predicting epigenome editing outcomes across cell types using multi-modal deep learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

BEP-EpiPredict is a three-stage transfer learning framework for predicting the epigenome editing effects of dCas9-recruited Biepigenetic Effector Proteins (BEPs). The model is trained on HEK293T multi-omics data and transfers to predict outcomes in K562 cells from baseline measurements alone.

**Key capabilities:**
- Predict changes in 25 histone modifications, chromatin accessibility (ATAC-seq), DNA methylation, and gene expression for 9 BEPs
- Transfer predictions from HEK293T (training) to K562 (inference) using only GFP baseline data
- Quantify which epigenomic signals (histone marks, ATAC, methylation) are most predictive of BEP sensitivity
- Model dCas9 binding intensity–response relationships

<p align="center">
  <img src="[https://github.com/hengbingao/epimodifier_AI_train/blob/main/model_architecture/architecture_1.png]" alt="Model Architecture" width="800"/>
</p>

---

## Experimental Background

dCas9 recruits BEPs to defined genomic loci via guide RNAs. Different BEPs alter the local epigenome through distinct mechanisms:

| BEP | Role | Primary mechanism |
|-----|------|-------------------|
| BEP073_GFP | Control (baseline) | None |
| BEP100_ZIM3 | Repressor | Polycomb-related silencing |
| BEP396_FOG1 | Repressor | Transcriptional repression |
| BEP486_SREBF2ddr | Activator | Transcriptional activation |
| BEP137_p65HSF1 | Activator | NF-κB/HSF1 activation |
| BEP217_HDAC4 | Repressor | Histone deacetylation |
| BEP304_REST | Repressor | Neuronal gene silencing |
| BEP447_RCOR1 | Repressor | CoREST complex |
| BEP450_SAP30 | Repressor | Sin3A complex |
| BEP491_KLF11ddr | Repressor | KLF11-mediated repression |

All experiments performed in HEK293T cells. K562 prediction uses only GFP control measurements.

---

## Model Architecture

```
Stage 1 — Frozen Enformer backbone
  DNA sequence (196 kbp) ─→ CNN + Transformer ─→ seq embedding (3072-dim)

Stage 2 — BEP perturbation head (trained on HEK293T)
  dCas9 ChIP  → CNN1D encoder ─┐
  ATAC-seq    → CNN1D encoder  ├-→ Cross-modal attention ─→ FiLM(BEP+role) ─→ Δ heads
  Methylation → Transformer   ─┤      (dCas9 as Query)
  Histone×25  → Transformer   ─┘
  Seq summary ───────────────────┘

Stage 3 — Cell-type adapter (K562 transfer)
  Freeze backbone + BEP head; train only ATAC cell adapter + output projections
```

**Output heads (all Δ relative to GFP control):**
- `hist_log2fc`: log₂FC for 25 histone marks (regression + 3-class)
- `atac_log2fc`: chromatin accessibility change
- `meth_delta`:  CG methylation level change
- `rna_log2fc`:  nearest-gene expression log₂FC

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/bep-epipredict.git
cd bep-epipredict
conda create -n bep python=3.10
conda activate bep
pip install -r requirements.txt

# Strongly recommended: pretrained Enformer backbone
pip install enformer-pytorch
```

**Bioinformatics tools** (for upstream data preparation):
```bash
conda install -c bioconda samtools deeptools macs3 star rsem bismark trim-galore
```

---

## Quick Start

### 1. Prepare your data

Organise files as follows:
```
data/
├── genome/
│   ├── hg38.fa             # Reference genome (with .fai index)
│   ├── hg38.chrom.sizes
│   ├── hg38-blacklist.v2.bed
│   └── gencode.v38.annotation.gtf
├── HEK293T/
│   ├── BEP073_GFP/
│   │   ├── histone/        # {MARK}_rep{1,2}.bedgraph  (25 marks × 2 reps)
│   │   ├── atac/           # ATAC_rep{1,2}.bedgraph
│   │   └── methylation/    # CG_allc.txt
│   ├── BEP100_ZIM3/        # Same structure for all 9 treatment BEPs
│   │   └── ...
│   ├── dcas9/
│   │   ├── dCas9_ChIP.bedgraph
│   │   └── dCas9_peaks.bed
│   └── rna/
│       └── RNA_{BEP}_rep{1,2}.tsv   # gene_id, TPM
└── K562/
    ├── BEP073_GFP/          # GFP baseline only
    └── dcas9/
```

Run the upstream preprocessing pipeline:
```bash
bash scripts/01_prepare_data.sh
```

### 2. Validate inputs

```bash
python scripts/02_check_and_visualize.py --mode check
```

### 3. Train

```bash
# Stage 2: HEK293T training (~hours on GPU)
python train.py --stage 2

# Stage 3: K562 transfer (~30 min)
python train.py --stage 3 --resume checkpoints/stage2_best.pt

# Or run everything at once
bash run_pipeline.sh
```

### 4. Predict

```bash
# Predict all BEPs at K562 loci
python scripts/03_predict.py \
    --ckpt   checkpoints/stage3_best.pt \
    --cell   K562 \
    --all_beps \
    --out    outputs/K562_predictions.tsv

# Predict a specific BEP at custom loci
python scripts/03_predict.py \
    --ckpt   checkpoints/stage3_best.pt \
    --cell   K562 \
    --peaks  my_target_sites.bed \
    --bep    BEP100_ZIM3 \
    --out    outputs/ZIM3_K562.tsv
```

### 5. Visualize

```bash
python scripts/02_check_and_visualize.py --mode figures --results outputs/results
```

---

## Output Files

After training, `outputs/results/` contains:

| File | Description |
|------|-------------|
| `stage2/test_metrics.json` | Per-mark Pearson, Spearman, R², F1 on HEK293T test set |
| `stage2/sensitivity_rankings.json` | Per-BEP ranked histone mark sensitivity (SI score) |
| `stage2/shap_importance.json` | Gradient×Input feature importance: marks vs ATAC vs methylation |
| `stage2/dose_response.json` | Predicted Δ signal at varying dCas9 intensities |
| `stage2/bep_similarity.json` | Pairwise cosine similarity of BEP effect profiles |
| `stage2/ablation.json` | In-silico mark ablation importance scores |
| `stage3/` | Same outputs for K562 transfer model |
| `figures/` | PDF + SVG publication figures |

**Prediction TSV columns:**

| Column | Description |
|--------|-------------|
| `chrom/start/end/center` | Locus coordinates |
| `BEP`, `role` | BEP identity and functional class |
| `log2fc_{MARK}` | Predicted log₂FC vs GFP for each of 25 marks |
| `cls_{MARK}` | Classification: `up` / `nc` / `down` |
| `atac_log2fc` | ATAC-seq accessibility change |
| `rna_log2fc` | Nearest-gene expression change |
| `attn_{modality}` | Cross-modal attention weight (5 modalities) |

---

## Interpreting Results

### Which histone mark is most sensitive to each BEP?

`sensitivity_rankings.json`:
```json
"BEP100_ZIM3": [
  {"mark": "H3K27me3", "SI": 0.847},
  {"mark": "H2AK119ub1", "SI": 0.631},
  ...
]
```
Sensitivity Index = `mean(|log₂FC|) × fraction(|log₂FC| > 0.5)`

### Which input modality drives predictions?

`shap_importance.json` and `modal_attention_per_bep.json` — higher value means that modality is more predictive of the BEP's effect. For example, a repressor BEP may be most sensitive to baseline H3K27me3 levels, while an activator BEP may be more ATAC-dependent.

### Are BEPs mechanistically similar?

`bep_similarity.json` provides a cosine similarity matrix of predicted effect profiles. BEPs with score > 0.8 likely act through overlapping mechanisms.

---

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

```yaml
data:
  peak_window: 4000       # bp window around dCas9 peak
  n_signal_bins: 100      # signal binning resolution

training:
  stage2:
    num_epochs: 100
    learning_rate: 5e-4
    loss_weights:
      histone: 1.0        # weight for Δhistone loss
      rna: 0.8            # weight for ΔRNA loss
      atac: 0.5
      methylation: 0.3
  stage3:
    num_epochs: 30
    learning_rate: 1e-4   # lower LR for transfer
```

---

## Resume Training

```bash
# Resume Stage 2 mid-training
python train.py --stage 2 --resume checkpoints/stage2_epoch_0050.pt

# Re-use cached parsed data (skip slow bedgraph loading)
python train.py --stage 2 --skip_data

# Use stub backbone (no enformer-pytorch, faster for debugging)
python train.py --stage 2 --no_pretrain
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` in config.yaml (try 16 or 8) |
| `enformer-pytorch` import error | Use `--no_pretrain` flag (stub backbone) |
| Very slow bedgraph loading | Use `--skip_data` after first run |
| Few loci pass QC filter | Lower `min_dcas9_signal` in config.yaml |
| Missing RNA-seq files | RNA is optional; set `w_rna: 0.0` in loss_weights |
| K562 norm file not found | Run Stage 2 fully before Stage 3 |

---

## Citation

If you use BEP-EpiPredict in your research, please cite:

```bibtex
@article{bep_epipredict_2025,
  title   = {Predicting epigenome editing outcomes across cell types 
             using multi-modal deep learning},
  author  = {Your Name et al.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {}
}
```

This work builds on:
- [Enformer](https://www.nature.com/articles/s41592-021-01252-x) (Avsec et al., 2021)
- [Enformer Celltyping](https://www.nature.com/articles/s41467-024-54441-5) (Szaruga et al., 2024)
- [Predicting epigenome editing effects](https://elifesciences.org/articles/92991) (Batra et al., 2024)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
