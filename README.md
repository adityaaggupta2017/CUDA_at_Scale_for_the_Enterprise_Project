# CUDA Batch Image Processor

**CUDA at Scale for the Enterprise – Independent Project**
*Aditya Gupta*

---

## Overview

This project implements a **GPU-accelerated batch image processing pipeline** that applies four distinct image processing operations to hundreds of grayscale images simultaneously. It demonstrates both the CUDA NPP library and custom CUDA kernels, processes 200 synthetic 256×256 images in a single execution, and produces comprehensive timing statistics.

### GPU Operations

| # | Operation | Implementation | API |
|---|-----------|---------------|-----|
| 1 | **Gaussian Blur** | 5×5 Gaussian filter | CUDA NPP `nppiFilterGaussBorder_8u_C1R` |
| 2 | **Sobel Edge Detection** | 3×3 Sobel X + Y, gradient magnitude | Custom CUDA kernel `SobelKernel` |
| 3 | **Histogram Equalization** | Contrast-limited intensity stretching | CUDA NPP `nppiEqualizeHist_8u_C1R` |
| 4 | **Unsharp-Mask Sharpening** | NPP Gaussian blur + custom kernel | NPP + custom CUDA kernel `UnsharpMaskKernel` |

---

## Dataset

The project ships with a **Python synthetic-data generator** that requires no external downloads:

```
python3 scripts/generate_test_data.py --count 200 --size 256
```

This produces **200 grayscale 256×256 PGM images** in five pattern categories:

| Category | Pattern | Highlights |
|----------|---------|-----------|
| 0 | Linear gradient (random angle) | Tests blur smoothing |
| 1 | Concentric rings + noise | Clear Sobel edge response |
| 2 | Low-contrast Gaussian blobs | Dramatic histogram equalization improvement |
| 3 | Checkerboard + Gaussian noise | Sharpening reveals crisp edges |
| 4 | Geometric shapes + salt-and-pepper noise | Combined stress test |

An optional download script (`scripts/download_data.sh`) retrieves the **Kodak Lossless True Color Image Suite** (24 images at 768×512) if real-world data is preferred.

---

## Project Structure

```
CUDA_at_Scale_for_the_Enterprise_Project/
├── src/
│   ├── main.cpp              – CLI entry-point (getopt_long argument parsing)
│   ├── image_processor.cu    – CUDA kernels + NPP wrappers + ProcessBatch()
│   ├── image_processor.cuh   – Public API: Config, ProcessingStats, ProcessBatch
│   ├── pnm_utils.cpp         – PGM load/save, directory listing
│   └── pnm_utils.h           – GrayscaleImage struct + I/O declarations
├── scripts/
│   ├── generate_test_data.py – Synthetic PGM image generator (NumPy only)
│   └── download_data.sh      – Optional: download Kodak image suite
├── data/
│   ├── input/                – Input PGM images (populated by generate script)
│   └── output/               – Processed output images
├── results/
│   └── processing.log        – Timing log (written after execution)
├── Makefile                  – Build system
├── run.sh                    – One-command end-to-end script
└── README.md
```

---

## Requirements

| Dependency | Version tested | Notes |
|-----------|---------------|-------|
| CUDA Toolkit | 12.0 | `/usr/local/cuda` |
| NVIDIA GPU | Compute 8.6 (RTX A6000/A5000) | Adjust `ARCH` for other GPUs |
| NPP libraries | 12.0 | `libnppc`, `libnppif`, `libnppist` |
| g++ | 9+ | C++17 |
| Python 3 | 3.8+ | Only for data generation |
| NumPy | 1.20+ | Only for data generation |

---

## Quick Start

### Option A — One command (recommended)

```bash
chmod +x run.sh
./run.sh
```

This will:
1. Verify that an NVIDIA GPU is available
2. Generate 200 synthetic PGM test images
3. Compile the CUDA binary
4. Run all four operations on all 200 images
5. Save processed images to `data/output/`
6. Write a timing log to `results/processing.log`

---

### Option B — Step by step

```bash
# 1. Generate test data
python3 scripts/generate_test_data.py --output data/input --count 200

# 2. Build
make CUDA_PATH=/usr/local/cuda ARCH=sm_86

# 3. Run (all four operations)
./cuda_image_processor --input data/input --output data/output --operation all --verbose

# 4. Run a single operation
./cuda_image_processor --input data/input --output data/output --operation edges

# 5. Run with custom sharpening strength
./cuda_image_processor --input data/input --output data/output \
    --operation sharpen --sharpen-amount 2.5
```

---

## CLI Reference

```
cuda_image_processor [OPTIONS]

Options:
  -i, --input <dir>        Input directory containing .pgm files
                           (default: data/input)
  -o, --output <dir>       Output directory for processed images
                           (default: data/output)
  -p, --operation <name>   Processing operation:
                             blur      – Gaussian blur        (NPP)
                             edges     – Sobel edge detection (CUDA kernel)
                             equalize  – Histogram equalization (NPP)
                             sharpen   – Unsharp-mask          (CUDA kernel)
                             all       – All four operations   (default)
  -a, --sharpen-amount <f> Sharpening multiplier (default: 1.5)
  -b, --batch-size <n>     Images per processing round (default: 50)
  -l, --log <file>         Log file path (default: results/processing.log)
  -v, --verbose            Print per-image progress
  -h, --help               Show help and exit
```

### Examples

```bash
# Process all images with all operations, verbose output
./cuda_image_processor -v

# Edge detection only
./cuda_image_processor -p edges

# Use custom directories
./cuda_image_processor -i /data/real_images -o /data/results -p all

# Strong sharpening
./cuda_image_processor -p sharpen -a 3.0
```

---

## Build Customization

```bash
# Different GPU architecture
make ARCH=sm_75   # Turing (RTX 20xx)
make ARCH=sm_80   # Ampere A100
make ARCH=sm_89   # Ada Lovelace (RTX 40xx)

# Custom CUDA installation
make CUDA_PATH=/opt/cuda-12.0

# Clean build artefacts
make clean
```

---

## Output Files

When `--operation all` is used, four output files are generated per input image:

```
data/output/
├── image_0000_blur.pgm
├── image_0000_edges.pgm
├── image_0000_equalized.pgm
├── image_0000_sharpened.pgm
├── image_0001_blur.pgm
...
```

For a single operation:
```
data/output/
├── image_0000_blur.pgm
├── image_0001_blur.pgm
...
```

---

## Implementation Details

### Custom CUDA Kernels

**`SobelKernel`** (2-D grid, 16×16 thread blocks):
- Loads a 3×3 neighbourhood for each pixel
- Computes Sobel X (vertical edge response) and Sobel Y (horizontal edge response)
- Returns approximated gradient magnitude `|Gx| + |Gy|` clamped to [0, 255]
- Border pixels are set to zero

**`UnsharpMaskKernel`** (1-D grid, 256 threads/block):
- Implements the classical unsharp-mask formula:
  `output = clamp(original + amount × (original − blurred), 0, 255)`
- The `blurred` input is produced by a preceding NPP Gaussian blur call

### NPP Library Calls

| Function | Purpose |
|----------|---------|
| `nppiFilterGaussBorder_8u_C1R` | 5×5 Gaussian blur with border replication |
| `nppiEqualizeHist_8u_C1R` | Full histogram equalisation with scratch buffer |
| `nppiEqualizeHistGetBufferSize_8u_C1R` | Query scratch-buffer size for equalization |

### Memory Management
- Host-side images loaded into `std::vector<uint8_t>` (row-major)
- Three persistent device buffers (`d_src`, `d_dst`, `d_tmp`) allocated once per batch and reused across all images to minimise `cudaMalloc` overhead
- Image step (pitch) equals image width (no padding), so plain `cudaMemcpy` is sufficient

### Timing
- **GPU time**: measured with `cudaEvent_t` pairs surrounding all kernel/NPP calls
- **Wall-clock time**: measured with `std::chrono::high_resolution_clock`

---

## Lessons Learned

1. **NPP scratch buffers**: `nppiEqualizeHist_8u_C1R` requires a device-side scratch buffer whose size must be queried at runtime. Forgetting to allocate it is a common source of NPP errors.

2. **Border handling**: The non-border variants of NPP filter functions (`nppiFilterGauss_8u_C1R`) expect the ROI to exclude the image border. Using `nppiFilterGaussBorder_8u_C1R` with `NPP_BORDER_REPLICATE` avoids having to shrink the ROI.

3. **Buffer reuse**: Preallocating device buffers for the largest image in the batch and reusing them eliminates per-image `cudaMalloc`/`cudaFree` overhead, which can dominate runtime for small images.

4. **Custom vs. library kernels**: For standard operations (blur, histogram equalization) NPP is faster to implement and likely more optimised; for operations like Sobel (where you need the combined magnitude) or unsharp masking (where you need access to both original and blurred data), custom CUDA kernels offer more flexibility and clarity.

5. **Image format choice**: Binary PGM (P5) is the simplest possible image format—no external library required, trivial to write a reader/writer, and universally viewable. This eliminates one class of dependency failures in lab environments.

---

## Proof of Execution

After running `./run.sh`, the following artefacts are produced:

- `results/processing.log` – per-image log with timing statistics
- `data/output/*.pgm` – 800 processed images (200 input × 4 operations)

Sample log excerpt:
```
=== CUDA Batch Image Processor v1.0.0 ===

Input directory  : data/input
Output directory : data/output
Operation        : all
Sharpen amount   : 1.5
Batch size       : 50

--- Timing ---
  Images processed : 200
  GPU time         : ~XX ms
  Wall-clock time  : ~XX ms
  Throughput (GPU) : ~XXXX images/s
```

---

## References

- [NVIDIA NPP Documentation](https://docs.nvidia.com/cuda/npp/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Sobel Operator – Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator)
- [Unsharp Masking – Wikipedia](https://en.wikipedia.org/wiki/Unsharp_masking)
- [Kodak Image Suite](http://r0k.us/graphics/kodak/) (optional real-world dataset)
