#!/usr/bin/env bash
# ================================================================
# run_pipeline.sh — Full end-to-end pipeline runner
# ================================================================
# Usage:
#   bash run_pipeline.sh                  # full pipeline
#   bash run_pipeline.sh --from 3         # start from step 3
#   bash run_pipeline.sh --only 5         # only step 5
# ================================================================

set -euo pipefail

GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
BOLD="\033[1m";   RESET="\033[0m"
ok()   { echo -e "${GREEN}${BOLD}[✓]${RESET} $* ($(( SECONDS - T0 ))s)"; }
info() { echo -e "${BOLD}[→]${RESET} $*"; }
err()  { echo -e "${RED}${BOLD}[✗]${RESET} $*"; exit 1; }

CONFIG="./configs/config.yaml"
FROM_STEP=1; ONLY_STEP=""
T0=$SECONDS

while [[ $# -gt 0 ]]; do
    case $1 in
        --from) FROM_STEP="$2"; shift 2 ;;
        --only) ONLY_STEP="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) err "Unknown arg: $1" ;;
    esac
done

run() {
    local s=$1
    [[ -n "$ONLY_STEP" ]] && [[ "$ONLY_STEP" == "$s" ]] && return 0
    [[ -z "$ONLY_STEP" ]] && [[ "$s" -ge "$FROM_STEP" ]] && return 0
    return 1
}

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  BEP Epigenome Editing Prediction — Full Pipeline${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

# ── Step 1: Upstream data preparation ────────────────────────────────────────
if run 1; then
    info "STEP 1: Upstream data preparation (BAM → bedgraph, allc, RNA TPM)"
    bash scripts/01_prepare_data.sh
    ok "Step 1: Data preparation"
fi

# ── Step 2: Install Python dependencies ───────────────────────────────────────
if run 2; then
    info "STEP 2: Installing Python dependencies"
    pip install -r requirements.txt -q
    # Try to install enformer-pytorch (optional but recommended)
    pip install enformer-pytorch -q 2>/dev/null \
        && echo "  enformer-pytorch installed" \
        || echo "  enformer-pytorch not available — will use stub backbone"
    ok "Step 2: Dependencies installed"
fi

# ── Step 3: Validate all input files ─────────────────────────────────────────
if run 3; then
    info "STEP 3: Input validation"
    python scripts/02_check_and_visualize.py \
        --mode check --config "$CONFIG" \
        || err "Validation failed. Fix errors above."
    ok "Step 3: All inputs valid"
fi

# ── Step 4: Stage 2 — HEK293T training ───────────────────────────────────────
if run 4; then
    info "STEP 4: Stage 2 — HEK293T BEP perturbation head training"
    DATA_FLAG=""
    [ -f "checkpoints/data_hek293t.pkl" ] && DATA_FLAG="--skip_data"
    python train.py --config "$CONFIG" --stage 2 $DATA_FLAG
    ok "Step 4: Stage 2 training complete"
fi

# ── Step 5: Stage 3 — K562 transfer ──────────────────────────────────────────
if run 5; then
    info "STEP 5: Stage 3 — K562 cross-cell-type transfer"
    DATA_FLAG=""
    [ -f "checkpoints/data_k562.pkl" ] && DATA_FLAG="--skip_data"
    python train.py --config "$CONFIG" --stage 3 \
        --resume checkpoints/stage2_best.pt $DATA_FLAG
    ok "Step 5: Stage 3 (K562 transfer) complete"
fi

# ── Step 6: K562 prediction ───────────────────────────────────────────────────
if run 6; then
    info "STEP 6: Predicting all BEPs at K562 loci"
    python scripts/03_predict.py \
        --config "$CONFIG" \
        --ckpt   checkpoints/stage3_best.pt \
        --cell   K562 \
        --all_beps \
        --out    outputs/K562_all_bep_predictions.tsv
    ok "Step 6: K562 predictions saved"
fi

# ── Step 7: Generate figures ──────────────────────────────────────────────────
if run 7; then
    info "STEP 7: Generating all result figures"
    python scripts/02_check_and_visualize.py \
        --mode figures \
        --config "$CONFIG" \
        --results outputs/results
    ok "Step 7: Figures generated"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${GREEN}${BOLD}Pipeline complete!  Total: $(( SECONDS - T0 ))s${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "Key outputs:"
echo "  checkpoints/stage2_best.pt          ← HEK293T trained model"
echo "  checkpoints/stage3_best.pt          ← K562 transfer model"
echo "  outputs/results/stage2/             ← HEK293T metrics + interpretability"
echo "  outputs/results/stage3/             ← K562 metrics + interpretability"
echo "  outputs/K562_all_bep_predictions.tsv ← per-locus predictions"
echo "  outputs/results/figures/            ← PDF figures"
echo ""
echo "Re-run inference on custom loci:"
echo "  python scripts/03_predict.py \\"
echo "    --ckpt checkpoints/stage3_best.pt \\"
echo "    --cell K562 \\"
echo "    --peaks my_sites.bed \\"
echo "    --bep  BEP100_ZIM3 \\"
echo "    --out  outputs/custom_pred.tsv"
