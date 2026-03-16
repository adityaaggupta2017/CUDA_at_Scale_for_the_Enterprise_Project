#!/usr/bin/env bash
# =============================================================================
# download_data.sh  –  Download real-world test images (optional)
# CUDA at Scale for the Enterprise – Independent Project
#
# Downloads the Kodak Lossless True Color Image Suite (24 images, 768×512 or
# 512×768 pixels) and converts them to grayscale PGM format using ImageMagick.
#
# The synthetic generator (scripts/generate_test_data.py) is sufficient to
# run the project; this script is provided for completeness.
#
# Requirements:
#   wget  – for downloading
#   convert (ImageMagick) – for PNG → grayscale PGM conversion
# =============================================================================
set -euo pipefail

OUTPUT_DIR="${1:-data/input}"
TMP_DIR="/tmp/kodak_download"
BASE_URL="http://r0k.us/graphics/kodak/kodak"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
error()   { echo -e "${RED}[ERR]${RESET}  $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
for cmd in wget convert; do
  if ! command -v "${cmd}" &>/dev/null; then
    error "'${cmd}' is not installed. Install it with:  sudo apt-get install ${cmd}"
  fi
done

mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

# ---------------------------------------------------------------------------
# Download and convert each image
# ---------------------------------------------------------------------------
info "Downloading Kodak image suite (24 images) to '${OUTPUT_DIR}' …"

for i in $(seq -f "%02g" 1 24); do
  url="${BASE_URL}/kodim${i}.png"
  png_file="${TMP_DIR}/kodim${i}.png"
  pgm_file="${OUTPUT_DIR}/kodim${i}.pgm"

  if [ -f "${pgm_file}" ]; then
    info "  kodim${i}.pgm already exists, skipping."
    continue
  fi

  info "  Downloading kodim${i}.png …"
  wget -q --show-progress -O "${png_file}" "${url}" || {
    echo "  Warning: failed to download ${url}, skipping."
    continue
  }

  # Convert to 8-bit grayscale PGM.
  convert "${png_file}" -colorspace Gray -depth 8 "${pgm_file}"
  success "  Saved ${pgm_file}"
done

rm -rf "${TMP_DIR}"
success "Download complete.  Images saved to '${OUTPUT_DIR}'."
echo ""
echo "You can now run the processor on these images:"
echo "  ./cuda_image_processor --input ${OUTPUT_DIR} --operation all"
