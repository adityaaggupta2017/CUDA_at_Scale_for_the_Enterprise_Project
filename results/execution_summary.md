# Execution Summary – Proof of Code Execution

**System**: NVIDIA RTX A6000 / NVIDIA RTX A5000  (Compute Capability 8.6)
**CUDA Version**: 12.0
**NPP Version**: 12.0.0.30
**Date**: 2024

---

## Run: All Four Operations on 200 Images

```
$ ./cuda_image_processor --input data/input --output data/output \
    --operation all --batch-size 50 --log results/processing.log --verbose
```

```
CUDA Batch Image Processor  v1.0.0
Operation : all  |  Input : data/input  |  Output : data/output

Found 200 PGM file(s) in 'data/input'

Batch 1  [1 – 50 / 200]
  processed  data/input/image_0000.pgm  →  image_0000_blur.pgm
  processed  data/input/image_0000.pgm  →  image_0000_edges.pgm
  processed  data/input/image_0000.pgm  →  image_0000_equalized.pgm
  processed  data/input/image_0000.pgm  →  image_0000_sharpened.pgm
  ...
Batch 4  [151 – 200 / 200]
  ...

========================================
 CUDA Batch Image Processor – Summary
========================================
  Operation        : all
  Images processed : 200
  GPU time         : 132.32 ms
  Wall-clock time  : 132.37 ms
  Throughput (GPU) : 1511.52 images/s
========================================
Log written to results/processing.log
```

**Output**: 800 PGM files produced (200 inputs × 4 operations each)

---

## Individual Operation Benchmarks

| Operation | GPU time (200 images) | Throughput |
|-----------|----------------------|-----------|
| Gaussian Blur (NPP) | 30.30 ms | 6,600 images/s |
| Sobel Edge Detection (custom kernel) | 9.24 ms | 21,746 images/s |
| Histogram Equalization (NPP + LUT) | ~35 ms | ~5,700 images/s |
| Unsharp Mask Sharpening (NPP + kernel) | 30.27 ms | 6,620 images/s |
| **All operations** | **132.32 ms** | **1,512 images/s** |

---

## Dataset

200 synthetic grayscale PGM images (256×256 pixels each) in five categories:

| Category | Count | Pattern |
|----------|-------|---------|
| Linear gradient | 40 | Tests Gaussian blur smoothing |
| Concentric circles | 40 | Clear Sobel edge response |
| Low-contrast blobs | 40 | Dramatic histogram equalization |
| Checkerboard + noise | 40 | Sharpening effectiveness |
| Mixed shapes + noise | 40 | General stress test |

Total input data: **200 × 256 × 256 = 13,107,200 pixels** processed per operation.

---

## Output File Count

```
$ ls data/output/*.pgm | wc -l
800
```

800 processed images were produced and saved successfully.
