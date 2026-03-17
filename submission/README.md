# Submission Artifacts – CUDA at Scale for the Enterprise

## What is in this archive

This archive contains evidence that both CUDA programs ran successfully on large data.

---

## Part 1 – CUDA Batch Image Processor

**Binary**: `cuda_image_processor`
**Dataset**: 200 synthetic 256×256 grayscale PGM images (5 pattern categories)
**Operations**: Gaussian blur (NPP), Sobel edge detection (CUDA kernel), histogram equalisation (NPP), unsharp-mask sharpening (CUDA kernel)
**Total output images produced**: 800 (200 × 4 operations)

### GPU timing summary

| Metric | Value |
|--------|-------|
| Images processed | 200 |
| GPU time | 2834.78 ms |
| Wall-clock time | 2837.12 ms |
| Throughput | 70.55 images/s |

### Files

| File | Description |
|------|-------------|
| `before_images/BEFORE_image_000X_<pattern>.png` | Original input image (5 pattern types) |
| `after_images/AFTER_image_000X_<pattern>_<op>.png` | GPU-processed output (4 ops × 5 images = 20 files) |
| `after_images/VISUAL_PROOF_<pattern>_before_after.png` | Side-by-side montage: INPUT + all 4 GPU outputs |
| `logs/image_processing_verbose.log` | Full verbose log: all 200 inputs listed, all 800 outputs, timing |

### Pattern types

| Image | Pattern | Best demonstrates |
|-------|---------|------------------|
| image_0000 | Linear gradient | Blur, edges |
| image_0001 | Concentric circles | Edges, blur |
| image_0002 | Low-contrast blobs | Histogram equalisation (reveals hidden structure) |
| image_0003 | Checkerboard + noise | Sharpening |
| image_0004 | Mixed shapes + salt-and-pepper noise | All operations |

---

## Part 2 – Iris GPU Classifier

**Binary**: `iris_gpu_classifier`
**Dataset**: Fisher Iris (UCI ML Repository) – 150 samples, 4 features, 3 classes
**Algorithm**: GPU-accelerated k-Nearest Neighbours (CUDA NPP + custom kernels)

### Results across 4 experiments

| k | Normalised | Accuracy | GPU time |
|---|-----------|---------|---------|
| 3 | Yes (z-score) | 94.67 % | 34.07 ms |
| 5 | Yes (z-score) | 94.67 % | 34.11 ms |
| 7 | Yes (z-score) | 96.00 % | 34.13 ms |
| 5 | No (raw) | 96.67 % | 27.60 ms |

### Files

| File | Description |
|------|-------------|
| `logs/iris_classifier_k*.log` | Per-run log with feature statistics and accuracy |
| `logs/iris_execution_summary.md` | Human-readable summary with confusion matrix |
| `csv_results/iris_predictions_*.csv` | 150-row prediction CSV per run (sample, features, actual, predicted, correct) |
| `csv_results/iris_feature_stats_gpu_computed.csv` | Per-feature stats computed with CUDA NPP |

---

## System information

- GPU: NVIDIA RTX A6000 / RTX A5000 (Compute Capability 8.6)
- CUDA: 12.0 | NPP: 12.0.0.30
- OS: Linux | Compiler: nvcc + g++ (C++17)
