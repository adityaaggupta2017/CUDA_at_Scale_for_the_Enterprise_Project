#!/usr/bin/env bash
# =============================================================================
# run.sh  –  End-to-end build and run for the Iris GPU Classifier
# CUDA at Scale for the Enterprise – Independent Project
# =============================================================================
# This script:
#   1. Verifies that a CUDA-capable GPU is present
#   2. Builds the CUDA binary with make
#   3. Runs the classifier with k=5 (default) on the Iris dataset
#   4. Re-runs with k=3 and k=7 to sweep k values
#   5. Runs once without feature normalisation for comparison
#   6. Prints a final summary
# =============================================================================
set -euo pipefail

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
ARCH="${ARCH:-sm_86}"
BINARY="./iris_gpu_classifier"
DATASET="iris/iris.data"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m';   RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
error()   { echo -e "${RED}[ERR]${RESET}  $*" >&2; exit 1; }

echo -e "${BOLD}============================================${RESET}"
echo -e "${BOLD}  Iris GPU Classifier – run.sh              ${RESET}"
echo -e "${BOLD}============================================${RESET}"
echo ""

# ---------------------------------------------------------------------------
# 1. GPU check
# ---------------------------------------------------------------------------
info "Checking for a CUDA-capable GPU …"
if ! command -v nvidia-smi &>/dev/null; then
  error "nvidia-smi not found. Please run on a machine with an NVIDIA GPU."
fi
nvidia-smi --query-gpu=name,driver_version,memory.total \
           --format=csv,noheader | \
    awk '{print "  GPU detected: " $0}'
success "GPU check passed."
echo ""

# ---------------------------------------------------------------------------
# 2. Dataset check
# ---------------------------------------------------------------------------
info "Checking dataset '${DATASET}' …"
if [ ! -f "${DATASET}" ]; then
  error "Dataset not found at '${DATASET}'. Ensure the iris/ directory is present."
fi
num_samples=$(grep -cE "Iris-" "${DATASET}" || true)
success "Dataset ready: ${num_samples} samples in '${DATASET}'."
echo ""

# ---------------------------------------------------------------------------
# 3. Build
# ---------------------------------------------------------------------------
info "Building with CUDA_PATH=${CUDA_PATH}  ARCH=${ARCH} …"
make --jobs=4 CUDA_PATH="${CUDA_PATH}" ARCH="${ARCH}" all
success "Build complete."
echo ""
mkdir -p results

# ---------------------------------------------------------------------------
# 4. Run: k=5, normalised (default)
# ---------------------------------------------------------------------------
echo -e "${BOLD}--- Run 1: k=5, z-score normalised (default) ---${RESET}"
"${BINARY}"                          \
    --input       "${DATASET}"       \
    --k-neighbors 5                  \
    --predictions results/predictions_k5.csv    \
    --stats       results/feature_stats.csv     \
    --log         results/processing.log
echo ""

# ---------------------------------------------------------------------------
# 5. Run: k=3
# ---------------------------------------------------------------------------
echo -e "${BOLD}--- Run 2: k=3, z-score normalised ---${RESET}"
"${BINARY}"                          \
    --input       "${DATASET}"       \
    --k-neighbors 3                  \
    --predictions results/predictions_k3.csv    \
    --log         results/processing_k3.log
echo ""

# ---------------------------------------------------------------------------
# 6. Run: k=7
# ---------------------------------------------------------------------------
echo -e "${BOLD}--- Run 3: k=7, z-score normalised ---${RESET}"
"${BINARY}"                          \
    --input       "${DATASET}"       \
    --k-neighbors 7                  \
    --predictions results/predictions_k7.csv    \
    --log         results/processing_k7.log
echo ""

# ---------------------------------------------------------------------------
# 7. Run: k=5, NO normalisation
# ---------------------------------------------------------------------------
echo -e "${BOLD}--- Run 4: k=5, NO normalisation ---${RESET}"
"${BINARY}"                          \
    --input         "${DATASET}"     \
    --k-neighbors   5                \
    --no-normalize                   \
    --predictions results/predictions_k5_raw.csv  \
    --log         results/processing_k5_raw.log
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
success "All runs complete."
echo ""
echo -e "${BOLD}Output files:${RESET}"
ls -lh results/*.csv results/*.log 2>/dev/null | awk '{print "  " $0}'
echo ""
echo -e "${BOLD}Done.${RESET}  Commit the 'results/' directory as proof of execution."
