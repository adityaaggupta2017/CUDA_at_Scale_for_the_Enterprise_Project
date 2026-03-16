#!/usr/bin/env bash
# =============================================================================
# run.sh  –  End-to-end build and run script
# CUDA at Scale for the Enterprise – Independent Project
# =============================================================================
# This script:
#   1. Checks that a CUDA-capable GPU is present
#   2. Generates 200 synthetic test PGM images (if the input dir is empty)
#   3. Builds the CUDA binary with make
#   4. Runs all four processing operations on the full image set
#   5. Prints a summary and saves the log to results/processing.log
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration – override with environment variables if desired.
# ---------------------------------------------------------------------------
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
ARCH="${ARCH:-sm_86}"
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
LOG_FILE="results/processing.log"
BINARY="./cuda_image_processor"
NUM_IMAGES=200

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
error()   { echo -e "${RED}[ERR]${RESET}  $*" >&2; exit 1; }

echo -e "${BOLD}============================================${RESET}"
echo -e "${BOLD}  CUDA Batch Image Processor – run.sh       ${RESET}"
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
# 2. Generate synthetic test data if the input directory is empty
# ---------------------------------------------------------------------------
info "Checking test data in '${INPUT_DIR}' …"
pgm_count=$(find "${INPUT_DIR}" -name '*.pgm' 2>/dev/null | wc -l)
if [ "${pgm_count}" -lt "${NUM_IMAGES}" ]; then
  info "Found ${pgm_count} images – generating ${NUM_IMAGES} synthetic images …"
  mkdir -p "${INPUT_DIR}"
  python3 scripts/generate_test_data.py \
      --output "${INPUT_DIR}" \
      --count  "${NUM_IMAGES}"
  success "Test data ready."
else
  success "Found ${pgm_count} PGM images – skipping generation."
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Build
# ---------------------------------------------------------------------------
info "Building with CUDA_PATH=${CUDA_PATH}  ARCH=${ARCH} …"
make --jobs=4 CUDA_PATH="${CUDA_PATH}" ARCH="${ARCH}" all
success "Build complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Create output directory
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
mkdir -p results

# ---------------------------------------------------------------------------
# 5. Run – all four operations on the full image set
# ---------------------------------------------------------------------------
info "Running: all operations on ${pgm_count:-${NUM_IMAGES}} images …"
echo ""

time "${BINARY}"            \
    --input   "${INPUT_DIR}" \
    --output  "${OUTPUT_DIR}"\
    --operation all          \
    --batch-size 50          \
    --log "${LOG_FILE}"      \
    --verbose

echo ""
success "Processing complete."
echo ""

# ---------------------------------------------------------------------------
# 6. Summary of output files
# ---------------------------------------------------------------------------
out_count=$(find "${OUTPUT_DIR}" -name '*.pgm' 2>/dev/null | wc -l)
info "Output images saved in '${OUTPUT_DIR}': ${out_count} files"
info "Processing log: ${LOG_FILE}"
echo ""
echo -e "${BOLD}Done.${RESET}  Commit the 'results/' and 'data/output/' directories as proof of execution."
