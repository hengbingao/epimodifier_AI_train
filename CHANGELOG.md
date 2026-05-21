# Changelog

All notable changes to BEP-EpiPredict are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Support for scATAC-seq cell-type embedding
- Integration with full SHAP library for exact Shapley values
- Pre-trained model weights download script
- Support for additional reference genomes (mm10, TAIR10)

---

## [1.0.0] — 2025-XX-XX

### Added
- Three-stage transfer learning architecture
  - Stage 1: Frozen Enformer backbone (sequence → epigenome grammar)
  - Stage 2: BEP perturbation head trained on HEK293T multi-omics
  - Stage 3: K562 cross-cell-type transfer via ATAC cell adapter
- Multi-modal inputs: 25 histone marks (CUT&Tag), ATAC-seq, CG methylation (allc), dCas9 ChIP-seq, DNA sequence (Enformer), RNA-seq
- Multi-task output: Δhistone (25 marks), ΔATAC, Δmethylation, ΔRNA
- BEP identity conditioning via FiLM (Feature-wise Linear Modulation) with role embedding (activator/repressor/control)
- Gated cross-modal attention with dCas9 signal as Query
- Five interpretability analyses:
  - Gradient × Input feature importance
  - Per-BEP mark sensitivity index ranking
  - dCas9 dose-response curves
  - BEP cosine similarity matrix
  - In-silico mark ablation
- Upstream data preparation pipeline (bash)
- Input validation checker
- Inference script for arbitrary loci and cell types
- Publication-quality figure generation (PDF + SVG)
- Jupyter notebook tutorial
- Unit test suite (pytest)
- GitHub Actions CI

### Data support
- 10 BEPs: GFP (control), ZIM3, FOG1, SREBF2ddr, p65HSF1, HDAC4, REST, RCOR1, SAP30, KLF11ddr
- 25 histone marks: H3K4me3/2/1, H3K27ac, H3K9ac/14ac/18ac, H4K16ac/5ac/8ac/20ac, H2BK20ac, H3R17me2a/R8me2a, H3K27me3/9me3/9me2, H4K20me3, H2AK119ub1, H3K36me3/36me2, Pol_II_Ser2P, Total_RNA_Pol_II, H3K27me1, H4K20me1
