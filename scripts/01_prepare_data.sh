#!/usr/bin/env bash
# ================================================================
# scripts/01_prepare_data.sh
# Upstream processing: BAM → bedgraph, RNA FASTQ → TPM, allc prep
# ================================================================
# Requirements (conda):
#   conda install -c bioconda samtools deeptools macs3 star rsem
#   conda install -c bioconda bismark trim-galore
#   pip install pyfaidx
#
# Run: bash scripts/01_prepare_data.sh
# ================================================================

set -euo pipefail

# ── User settings ───────────────────────────────────────────────────────────
GENOME_FA="./data/genome/hg38.fa"
BLACKLIST="./data/genome/hg38-blacklist.v2.bed"
RAW_BAM_DIR="./raw_data/bams"
RAW_FASTQ_RNA="./raw_data/rna_fastq"
OUT_DIR="./data"
THREADS=16
GENOME_SIZE=2700000000   # hg38 effective genome size

CELL_TYPES=(HEK293T K562)

BEPS=(BEP073_GFP BEP100_ZIM3 BEP396_FOG1 BEP486_SREBF2ddr
      BEP137_p65HSF1 BEP217_HDAC4 BEP304_REST BEP447_RCOR1
      BEP450_SAP30 BEP491_KLF11ddr)

MARKS=(H3K4me3 H3K4me2 H3K4me1 H3K27ac H3K9ac H3K14ac H3K18ac
       H4K16ac H4K5ac H4K8ac H4K20ac H2BK20ac H3R17me2a H3R8me2a
       H3K27me3 H3K9me3 H3K9me2 H4K20me3 H2AK119ub1 H3K36me3
       H3K36me2 Pol_II_Ser2P Total_RNA_Pol_II H3K27me1 H4K20me1)

# ================================================================
# STEP A: Index genome
# ================================================================
echo "[A] Indexing genome ..."
samtools faidx "$GENOME_FA"
echo "    → ${GENOME_FA}.fai"

# ================================================================
# STEP B: CUT&Tag histone bedgraphs (RPKM normalised, 10 bp bins)
# ================================================================
echo "[B] CUT&Tag histone bedgraphs ..."

for CELL in "${CELL_TYPES[@]}"; do
  # K562 only has GFP control
  if [ "$CELL" == "K562" ]; then
    BEPS_FOR_CELL=(BEP073_GFP)
  else
    BEPS_FOR_CELL=("${BEPS[@]}")
  fi

  for BEP in "${BEPS_FOR_CELL[@]}"; do
    OUT_BG="${OUT_DIR}/${CELL}/${BEP}/histone"
    mkdir -p "$OUT_BG"

    for MARK in "${MARKS[@]}"; do
      for REP in 1 2; do
        BAM="${RAW_BAM_DIR}/${CELL}/${BEP}/${MARK}_rep${REP}.bam"
        OUT="${OUT_BG}/${MARK}_rep${REP}.bedgraph"

        [ -f "$BAM" ] || { echo "  WARN: missing $BAM"; continue; }

        # Sort & index if needed
        if [ ! -f "${BAM%.bam}.sorted.bam" ]; then
          samtools sort -@ "$THREADS" -o "${BAM%.bam}.sorted.bam" "$BAM"
          samtools index "${BAM%.bam}.sorted.bam"
        fi
        SORTED="${BAM%.bam}.sorted.bam"

        bamCoverage \
          --bam "$SORTED" \
          --outFileName "$OUT" \
          --outFileFormat bedgraph \
          --normalizeUsing RPKM \
          --binSize 10 \
          --numberOfProcessors "$THREADS" \
          --ignoreDuplicates \
          --minMappingQuality 10 \
          --blackListFileName "$BLACKLIST" \
          2>/dev/null

        echo "    → $OUT"
      done
    done
  done
done

# ================================================================
# STEP C: ATAC-seq bedgraphs
# ================================================================
echo "[C] ATAC-seq bedgraphs ..."

for CELL in "${CELL_TYPES[@]}"; do
  if [ "$CELL" == "K562" ]; then
    BEPS_FOR_CELL=(BEP073_GFP)
  else
    BEPS_FOR_CELL=("${BEPS[@]}")
  fi

  for BEP in "${BEPS_FOR_CELL[@]}"; do
    OUT_ATAC="${OUT_DIR}/${CELL}/${BEP}/atac"
    mkdir -p "$OUT_ATAC"

    for REP in 1 2; do
      BAM="${RAW_BAM_DIR}/${CELL}/${BEP}/ATAC_rep${REP}.bam"
      OUT="${OUT_ATAC}/ATAC_rep${REP}.bedgraph"

      [ -f "$BAM" ] || { echo "  WARN: missing $BAM"; continue; }

      if [ ! -f "${BAM%.bam}.sorted.bam" ]; then
        samtools sort -@ "$THREADS" -o "${BAM%.bam}.sorted.bam" "$BAM"
        samtools index "${BAM%.bam}.sorted.bam"
      fi

      bamCoverage \
        --bam "${BAM%.bam}.sorted.bam" \
        --outFileName "$OUT" \
        --outFileFormat bedgraph \
        --normalizeUsing RPKM \
        --binSize 10 \
        --numberOfProcessors "$THREADS" \
        --ignoreDuplicates \
        --blackListFileName "$BLACKLIST" \
        2>/dev/null
      echo "    → $OUT"
    done
  done
done

# ================================================================
# STEP D: dCas9 ChIP-seq bedgraph + peak calling
# ================================================================
echo "[D] dCas9 ChIP-seq ..."

for CELL in "${CELL_TYPES[@]}"; do
  OUT_DCAS9="${OUT_DIR}/${CELL}/dcas9"
  mkdir -p "$OUT_DCAS9"

  # Merge reps for peak calling
  BAM1="${RAW_BAM_DIR}/${CELL}/dCas9/dCas9_rep1.bam"
  BAM2="${RAW_BAM_DIR}/${CELL}/dCas9/dCas9_rep2.bam"
  MERGED="${OUT_DCAS9}/dCas9_merged.sorted.bam"

  samtools merge -f -@ "$THREADS" - "$BAM1" "$BAM2" \
    | samtools sort -@ "$THREADS" -o "$MERGED" -
  samtools index "$MERGED"

  # Bedgraph
  bamCoverage \
    --bam "$MERGED" \
    --outFileName "${OUT_DCAS9}/dCas9_ChIP.bedgraph" \
    --outFileFormat bedgraph \
    --normalizeUsing RPKM \
    --binSize 10 \
    --numberOfProcessors "$THREADS" \
    2>/dev/null
  echo "    → ${OUT_DCAS9}/dCas9_ChIP.bedgraph"

  # Peak calling
  macs3 callpeak \
    -t "$MERGED" \
    --format BAM \
    --gsize "$GENOME_SIZE" \
    --name dCas9 \
    --outdir "$OUT_DCAS9" \
    --nomodel --extsize 200 \
    --keep-dup all -q 0.01 \
    2>/dev/null

  awk 'BEGIN{OFS="\t"} NR>1 {print $1,$2,$3}' \
    "${OUT_DCAS9}/dCas9_peaks.narrowPeak" \
    > "${OUT_DCAS9}/dCas9_peaks.bed"
  echo "    → ${OUT_DCAS9}/dCas9_peaks.bed"
done

# ================================================================
# STEP E: DNA methylation (Bismark → allc)
# Only GFP condition needed (control baseline)
# ================================================================
echo "[E] CG methylation (Bismark → allc) ..."

for CELL in "${CELL_TYPES[@]}"; do
  OUT_METH="${OUT_DIR}/${CELL}/BEP073_GFP/methylation"
  mkdir -p "$OUT_METH"

  # Assume Bismark CX report already generated; convert to allc
  CX_REPORT="${RAW_BAM_DIR}/${CELL}/GFP_methylation/GFP.CX_report.txt"

  if [ -f "$CX_REPORT" ]; then
    echo "  Converting CX report → allc for $CELL ..."
    awk 'BEGIN{OFS="\t"}
      $6=="CG" && ($4+$5)>=1 {
        mc=$4; cov=$4+$5
        print $1, $2, $3, "CG"$7, mc, cov, (mc/cov>=0.5?1:0)
      }' "$CX_REPORT" > "${OUT_METH}/CG_allc.txt"
    echo "    → ${OUT_METH}/CG_allc.txt"
  else
    echo "  WARN: $CX_REPORT not found. Run Bismark first."
    echo "  Expected command:"
    echo "    bismark_methylation_extractor --CX_context --comprehensive \\"
    echo "      --genome_folder ./data/genome --cytosine_report *.bam"
  fi
done

# ================================================================
# STEP F: RNA-seq → gene-level TPM (STAR + RSEM)
# ================================================================
echo "[F] RNA-seq quantification (STAR + RSEM) ..."

# Build STAR index (run once)
STAR_INDEX="${OUT_DIR}/genome/star_index"
if [ ! -d "$STAR_INDEX" ]; then
  mkdir -p "$STAR_INDEX"
  STAR \
    --runMode genomeGenerate \
    --genomeDir "$STAR_INDEX" \
    --genomeFastaFiles "$GENOME_FA" \
    --sjdbGTFfile "./data/genome/gencode.v38.annotation.gtf" \
    --runThreadN "$THREADS" \
    --genomeSAindexNbases 14
fi

# Process each BEP × replicate (HEK293T only; K562 has no RNA in this setup)
for BEP in "${BEPS[@]}"; do
  OUT_RNA="${OUT_DIR}/HEK293T/rna"
  mkdir -p "$OUT_RNA"

  for REP in 1 2; do
    R1="${RAW_FASTQ_RNA}/${BEP}_rep${REP}_R1.fastq.gz"
    R2="${RAW_FASTQ_RNA}/${BEP}_rep${REP}_R2.fastq.gz"

    [ -f "$R1" ] || { echo "  WARN: missing $R1"; continue; }

    STAR_OUT="${OUT_RNA}/${BEP}_rep${REP}_star"
    mkdir -p "$STAR_OUT"

    STAR \
      --genomeDir "$STAR_INDEX" \
      --readFilesIn "$R1" "$R2" \
      --readFilesCommand zcat \
      --outSAMtype BAM SortedByCoordinate \
      --outFileNamePrefix "${STAR_OUT}/" \
      --runThreadN "$THREADS" \
      --outSAMattributes NH HI AS NM \
      2>/dev/null

    samtools index "${STAR_OUT}/Aligned.sortedByCoord.out.bam"

    # RSEM for TPM
    rsem-calculate-expression \
      --bam --no-bam-output \
      --paired-end \
      --num-threads "$THREADS" \
      "${STAR_OUT}/Aligned.sortedByCoord.out.bam" \
      "${STAR_INDEX}/rsem_ref" \
      "${STAR_OUT}/rsem" \
      2>/dev/null

    # Extract gene_id + TPM → model-ready TSV
    awk 'BEGIN{OFS="\t"; print "gene_id","TPM"}
         NR>1 {print $1,$6}' \
      "${STAR_OUT}/rsem.genes.results" \
      > "${OUT_RNA}/RNA_${BEP}_rep${REP}.tsv"
    echo "    → ${OUT_RNA}/RNA_${BEP}_rep${REP}.tsv"
  done
done

# ================================================================
# Summary
# ================================================================
echo ""
echo "============================================================"
echo "Data preparation complete!"
echo ""
echo "Expected structure:"
echo "  data/"
echo "    genome/ hg38.fa + hg38.fa.fai + gencode.v38.annotation.gtf"
echo "    HEK293T/"
echo "      BEP073_GFP/histone/{MARK}_rep{1,2}.bedgraph  (25 marks × 10 BEPs)"
echo "      BEP073_GFP/atac/ATAC_rep{1,2}.bedgraph"
echo "      BEP073_GFP/methylation/CG_allc.txt"
echo "      dcas9/dCas9_ChIP.bedgraph + dCas9_peaks.bed"
echo "      rna/RNA_{BEP}_rep{1,2}.tsv"
echo "    K562/"
echo "      BEP073_GFP/  (baseline only)"
echo "      dcas9/ ..."
echo "============================================================"
