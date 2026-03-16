# Execution Summary – Proof of Code Execution

**System**: NVIDIA RTX A6000 / NVIDIA RTX A5000 (Compute Capability 8.6)
**CUDA Version**: 12.0 | **NPP Version**: 12.0.0.30
**Dataset**: Fisher Iris (UCI ML Repository) – 150 samples, 4 features, 3 classes

---

## Run 1 – k=5, z-score normalised (default)

```
$ ./iris_gpu_classifier --input iris/iris.data --k-neighbors 5
```

```
========================================
  Iris GPU Classifier – Summary
========================================
  k-neighbors   : 5
  Normalise     : yes (z-score)
  Samples       : 150
  Correct       : 142
  Accuracy      : 94.67 %
  GPU time      : 34.11 ms
  Wall-clock    : 36.22 ms
========================================
```

**Confusion matrix**

|                  | Predicted Setosa | Predicted Versicolor | Predicted Virginica |
|-----------------|:---:|:---:|:---:|
| **Actual Setosa**      | 50 | 0 | 0 |
| **Actual Versicolor**  |  0 | 46 | 4 |
| **Actual Virginica**   |  0 |  4 | 46 |

---

## Run 2 – k=3, z-score normalised

```
$ ./iris_gpu_classifier --k-neighbors 3
```
```
  Accuracy : 94.67 %  |  GPU time : 34.07 ms
```

---

## Run 3 – k=7, z-score normalised

```
$ ./iris_gpu_classifier --k-neighbors 7
```
```
  Accuracy : 96.00 %  |  GPU time : 34.13 ms
```

---

## Run 4 – k=5, NO normalisation

```
$ ./iris_gpu_classifier --k-neighbors 5 --no-normalize
```
```
  Accuracy : 96.67 %  |  GPU time : 27.60 ms
```

---

## GPU-Computed Feature Statistics (all runs)

| Feature       | Mean   | Std Dev | Min | Max |
|--------------|--------|---------|-----|-----|
| sepal_length | 5.8433 | 0.8253  | 4.3 | 7.9 |
| sepal_width  | 3.0540 | 0.4321  | 2.0 | 4.4 |
| petal_length | 3.7587 | 1.7585  | 1.0 | 6.9 |
| petal_width  | 1.1987 | 0.7606  | 0.1 | 2.5 |

Statistics computed with CUDA NPP (`nppsSum_32f`, `nppsMinMax_32f`) and custom CUDA kernels.

---

## k-Value Accuracy Sweep

| k | Normalised | Accuracy |
|---|-----------|---------|
| 3 | Yes | 94.67 % |
| 5 | Yes | 94.67 % |
| 7 | Yes | 96.00 % |
| 5 | No  | 96.67 % |

---

## Output Files Produced

```
results/
├── predictions_k5.csv       (150 rows: sample, features, actual, predicted, correct)
├── predictions_k3.csv
├── predictions_k7.csv
├── predictions_k5_raw.csv
├── feature_stats.csv        (per-feature GPU-computed statistics)
├── processing.log
├── processing_k3.log
├── processing_k7.log
└── processing_k5_raw.log
```

All 150 samples classified in a single GPU execution per run.
