# Iris GPU Classifier

**CUDA at Scale for the Enterprise – Independent Project**
*Aditya Gupta*

---

## Overview

This project implements a **GPU-accelerated k-Nearest Neighbours (k-NN) classifier and statistical analysis pipeline** for the classic **Fisher Iris dataset** (UCI ML Repository, 1936).  It demonstrates both the **CUDA NPP signal-processing library** and **custom CUDA kernels**, classifying all 150 samples in a single GPU execution and producing a full confusion matrix, per-feature statistics, and prediction CSV files.

---

## Dataset

**Fisher Iris Dataset** (UCI Machine Learning Repository)
- **Source**: `iris/iris.data` — included in this repository
- **Samples**: 150 instances (50 per class)
- **Features**: 4 real-valued measurements per sample
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (all in cm)
- **Classes**: 3
  - `Iris-setosa` (label 0) — linearly separable from the others
  - `Iris-versicolor` (label 1) — not linearly separable from virginica
  - `Iris-virginica` (label 2) — not linearly separable from versicolor
- **Note**: Two corrections from the original Fisher article have been applied
  (sample 35: petal_width 0.2; sample 38: sepal_width 3.6, petal_length 1.4)

---

## GPU Pipeline

Seven stages, all executed on the GPU:

| Stage | Operation | GPU API |
|-------|-----------|---------|
| 1 | Transpose feature matrix row-major → column-major | Custom CUDA kernel |
| 2 | Per-feature column sum | **CUDA NPP** `nppsSum_32f` |
| 3 | Per-feature standard deviation (parallel reduction) | Custom CUDA kernel |
| 4 | Per-feature min / max | **CUDA NPP** `nppsMinMax_32f` |
| 5 | Z-score feature normalisation | **CUDA NPP** `nppsNormalize_32f` |
| 6 | 150×150 pairwise Euclidean distance matrix | Custom CUDA kernel |
| 7 | k-NN classification (partial insertion sort + majority vote) | Custom CUDA kernel |

### Custom CUDA Kernels

**`TransposeToColumnMajorKernel`** / **`TransposeToRowMajorKernel`**
- 2-D thread grid (32×4 blocks)
- Rearranges the feature matrix so each feature column is contiguous in memory, enabling NPP signal functions which require 1-D contiguous arrays

**`SumSquaredDevKernel`** + **`FinaliseStdDevKernel`**
- Accumulates per-feature sum-of-squared-deviations via `atomicAdd`
- Second kernel finalises σ = √(Σ(x−μ)² / n)

**`DistanceMatrixKernel`**
- 16×16 thread blocks; each thread (i, j) computes one cell of the 150×150 distance matrix independently
- Uses Euclidean distance: d(i,j) = √Σ(x_if − x_jf)²

**`KNNClassifyKernel`**
- One thread per query sample
- Maintains a sorted top-k list (partial insertion sort, O(k·n) per thread)
- Majority vote over k neighbours → predicted class

### CUDA NPP Signal Functions

| Function | Purpose |
|---------|---------|
| `nppsSum_32f` | Compute sum of a 1-D float array (→ mean) |
| `nppsMinMax_32f` | Compute min and max of a 1-D float array |
| `nppsNormalize_32f` | Apply z-score: `(x − mean) / std_dev` element-wise |

---

## Results (on NVIDIA RTX A6000, SM 8.6)

| k | Normalised | Accuracy | GPU time |
|---|-----------|---------|---------|
| 3 | Yes (z-score) | 94.67 % | 34.07 ms |
| **5** | **Yes (z-score)** | **94.67 %** | **34.11 ms** |
| 7 | Yes (z-score) | 96.00 % | 34.13 ms |
| 5 | No (raw) | 96.67 % | 27.60 ms |

Confusion matrix for k=5, normalised:

|                      | Pred. Setosa | Pred. Versicolor | Pred. Virginica |
|---------------------|:---:|:---:|:---:|
| **Actual Setosa**    | **50** | 0 | 0 |
| **Actual Versicolor**|   0 | **46** | 4 |
| **Actual Virginica** |   0 |  4 | **46** |

*Setosa is perfectly classified (linearly separable). The 8 misclassifications are all between the two non-linearly-separable classes.*

---

## Project Structure

```
CUDA_at_Scale_for_the_Enterprise_Project/
├── iris/
│   ├── iris.data             – Fisher Iris dataset (150 samples)
│   ├── bezdekIris.data       – Corrected version
│   ├── iris.names            – Dataset description
│   └── Index                 – UCI index file
├── src/
│   ├── main.cpp              – CLI entry-point (getopt_long)
│   ├── iris_processor.cu     – Custom CUDA kernels + NPP wrappers
│   ├── iris_processor.cuh    – PipelineResult, ClassifierConfig, RunGPUPipeline
│   ├── csv_utils.cpp         – CSV parser and result writer
│   └── csv_utils.h           – IrisSample struct + I/O declarations
├── results/
│   ├── predictions_k5.csv    – Per-sample predictions (k=5, normalised)
│   ├── predictions_k3.csv
│   ├── predictions_k7.csv
│   ├── predictions_k5_raw.csv
│   ├── feature_stats.csv     – GPU-computed per-feature statistics
│   ├── processing.log        – Timing and accuracy log
│   └── execution_summary.md  – Human-readable proof of execution
├── Makefile                  – Build system
├── run.sh                    – End-to-end run script (4 experiments)
└── README.md
```

---

## Requirements

| Dependency | Version tested | Notes |
|-----------|---------------|-------|
| CUDA Toolkit | 12.0 | Default path: `/usr/local/cuda` |
| NVIDIA GPU | Compute 8.6 (RTX A6000/A5000) | Adjust `ARCH` for other GPUs |
| NPP signal library | 12.0 (`libnpps`) | Included in CUDA Toolkit |
| g++ | 9+ | C++17 required |
| Python 3 | — | Not required for this project |

---

## Quick Start

### Option A — One command (recommended)

```bash
chmod +x run.sh
./run.sh
```

This will:
1. Verify that an NVIDIA GPU is present
2. Compile the CUDA binary
3. Run four experiments (k=3, 5, 7 with normalisation; k=5 without)
4. Save predictions and statistics to `results/`

---

### Option B — Step by step

```bash
# 1. Build
make CUDA_PATH=/usr/local/cuda ARCH=sm_86

# 2. Run with defaults (k=5, z-score normalised)
./iris_gpu_classifier

# 3. Run with k=3, verbose per-sample output
./iris_gpu_classifier --k-neighbors 3 --verbose

# 4. Run with k=7
./iris_gpu_classifier --k-neighbors 7

# 5. Run without normalisation
./iris_gpu_classifier --k-neighbors 5 --no-normalize

# 6. Show help
./iris_gpu_classifier --help
```

---

## CLI Reference

```
iris_gpu_classifier [OPTIONS]

Options:
  -i, --input <file>      Iris CSV file          (default: iris/iris.data)
  -o, --predictions <f>   Output predictions CSV (default: results/predictions.csv)
  -s, --stats <file>      Output feature-stats CSV (default: results/feature_stats.csv)
  -l, --log <file>        Log file               (default: results/processing.log)
  -k, --k-neighbors <n>   Neighbours for k-NN   (default: 5)
  -n, --no-normalize      Skip z-score normalisation
  -v, --verbose           Print per-sample prediction result
  -h, --help              Show help and exit
```

---

## Build Customization

```bash
# Different GPU architecture
make ARCH=sm_75   # Turing  (RTX 20xx / Tesla T4)
make ARCH=sm_80   # Ampere  (A100)
make ARCH=sm_89   # Ada Lovelace (RTX 40xx)

# Custom CUDA installation
make CUDA_PATH=/opt/cuda-12.0

# Clean build artefacts
make clean
```

---

## Output Files

```
results/
├── predictions_k5.csv       – 150 rows × 8 columns
│                              (sample_id, 4 features, actual, predicted, correct)
├── feature_stats.csv        – 4 rows × 5 columns
│                              (feature, mean, std_dev, min, max)
└── processing.log           – timing, accuracy, feature statistics
```

---

## Implementation Details

### Memory Layout
- Host: `std::vector<IrisSample>` with `float features[4]` per sample
- Device row-major: `d_features[sample * 4 + feature]` (used by distance kernel)
- Device column-major: `d_col_arrays[feature * 150 + sample]` (used by NPP signal functions)
- Transpose between the two layouts is done on the GPU with dedicated kernels

### Why a Transpose?
NPP signal functions (`nppsSum_32f`, `nppsMinMax_32f`, `nppsNormalize_32f`) require their input as a **contiguous 1-D array**. Storing features column-major puts all 150 values for one feature contiguously, enabling direct NPP calls without strided access.

### Distance Matrix
Each of the 150×150 = 22,500 cells is computed by an independent GPU thread, making this stage perfectly parallel. The normalised feature vectors ensure that all four features contribute equally to the distance regardless of their original scale.

### k-NN Kernel
Each thread maintains a sorted array of the k smallest distances found so far (partial insertion sort), avoiding a full sort of 150 values. For k≤15 and n=150 this is `O(k·n)` per thread—extremely fast on GPU.

---

## Lessons Learned

1. **Column-major vs row-major**: NPP signal functions require contiguous 1-D arrays. Rearranging the feature matrix on the GPU with a transpose kernel was essential to use NPP effectively without copying data to the CPU.

2. **atomicAdd for parallel reductions**: The std-dev kernel uses `atomicAdd` to accumulate squared deviations from multiple threads. While a tree-reduction with shared memory is faster for large arrays, `atomicAdd` is simpler and correct for n=150.

3. **Normalisation affects accuracy**: For this dataset, raw Euclidean distances (without normalisation) perform slightly better because the scales of the four features already encode useful information. This highlights that normalisation is a design choice, not always beneficial.

4. **NPP on 1-D signal data**: The NPP signal library (`libnpps`) is equally applicable to tabular/CSV data as it is to audio signals—any 1-D float array is a valid input. This demonstrates that NPP is not limited to image processing.

5. **GPU overhead vs dataset size**: For n=150, the GPU kernel execution is sub-millisecond; most of the 34 ms is CUDA context initialisation. For the same pipeline on a dataset with 150,000 samples, the GPU advantage would be substantial.

---

## References

- Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.
- UCI ML Repository: Iris Dataset – https://archive.ics.uci.edu/dataset/53/iris
- [CUDA NPP Signal Processing Documentation](https://docs.nvidia.com/cuda/npp/group__signal__statistics.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [k-Nearest Neighbours Algorithm – Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
