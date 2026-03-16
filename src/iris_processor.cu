// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// iris_processor.cu  –  GPU pipeline implementation for Iris dataset analysis.
//
// Pipeline stages
// ---------------
//  1. Pack host feature matrix (150×4 row-major float)
//  2. Upload to device
//  3. [CUDA kernel]  Transpose to column-major (4 × 150 arrays)
//  4. [CUDA NPP]     nppsSum_32f   → sum of each feature column
//  5. [CPU]          Divide by n_samples → means
//  6. [CUDA kernel]  StdDevKernel  → std dev of each feature column
//  7. [CUDA NPP]     nppsMinMax_32f → min/max of each feature column
//  8. [CUDA NPP]     nppsNormalize_32f → z-score normalise each column
//  9. [CUDA kernel]  TransposeToRowMajorKernel → back to row-major
// 10. [CUDA kernel]  DistanceMatrixKernel → 150×150 Euclidean distance matrix
// 11. [CUDA kernel]  KNNClassifyKernel → k-NN prediction for every sample
// 12. Download predictions and return

#include "iris_processor.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <npps_arithmetic_and_logical_operations.h>
#include <npps_statistics_functions.h>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error at %s:%d – %s\n",                      \
                   __FILE__, __LINE__, cudaGetErrorString(_e));                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NPP_CHECK(call)                                                        \
  do {                                                                         \
    NppStatus _s = (call);                                                     \
    if (_s != NPP_SUCCESS) {                                                   \
      std::fprintf(stderr, "NPP error at %s:%d – status %d\n",                \
                   __FILE__, __LINE__, static_cast<int>(_s));                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Custom CUDA kernels
// ---------------------------------------------------------------------------

// Transpose a row-major matrix (n_samples × n_features) into n_features
// separate column arrays, each of length n_samples.
// Layout: col_arrays[f * n_samples + s] = row_data[s * n_features + f]
__global__ void TransposeToColumnMajorKernel(const float* __restrict__ row_data,
                                              float*       __restrict__ col_arrays,
                                              int n_samples, int n_features) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int f = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (s >= n_samples || f >= n_features) return;
  col_arrays[f * n_samples + s] = row_data[s * n_features + f];
}

// Transpose back from column-major to row-major.
__global__ void TransposeToRowMajorKernel(const float* __restrict__ col_arrays,
                                           float*       __restrict__ row_data,
                                           int n_samples, int n_features) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int f = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (s >= n_samples || f >= n_features) return;
  row_data[s * n_features + f] = col_arrays[f * n_samples + s];
}

// Compute the standard deviation of each feature column.
// Each thread processes one element; atomicAdd accumulates the sum of
// squared deviations, then the first thread in each column divides.
// Uses a two-pass approach: pass 1 = sum of squares, pass 2 = sqrt(var).
__global__ void SumSquaredDevKernel(const float* __restrict__ col_arrays,
                                     const float* __restrict__ means,
                                     float*       __restrict__ sq_sum,
                                     int n_samples, int n_features) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int f = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (s >= n_samples || f >= n_features) return;

  const float diff = col_arrays[f * n_samples + s] - means[f];
  atomicAdd(&sq_sum[f], diff * diff);
}

__global__ void FinaliseStdDevKernel(float*       std_dev,
                                      const float* sq_sum,
                                      int          n_samples,
                                      int          n_features) {
  const int f = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (f >= n_features) return;
  std_dev[f] = sqrtf(sq_sum[f] / static_cast<float>(n_samples));
}

// Compute the pairwise Euclidean distance matrix.
// Each thread computes one cell D[i, j] where i = row, j = col.
// features is row-major: features[s * n_features + f].
__global__ void DistanceMatrixKernel(const float* __restrict__ features,
                                      float*       __restrict__ dist_matrix,
                                      int n_samples, int n_features) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= n_samples || j >= n_samples) return;

  float sum_sq = 0.f;
  for (int f = 0; f < n_features; ++f) {
    const float diff = features[i * n_features + f]
                     - features[j * n_features + f];
    sum_sq += diff * diff;
  }
  dist_matrix[i * n_samples + j] = sqrtf(sum_sq);
}

// k-NN classification kernel.
// Each thread handles one query sample i: scans its row in dist_matrix,
// picks k nearest neighbours (excluding self), votes by label, returns
// the majority class as predictions[i].
//
// Supports up to kMaxK = 15 neighbours and kMaxSamples = 200 samples.
static constexpr int kMaxK = 15;

__global__ void KNNClassifyKernel(const float* __restrict__ dist_matrix,
                                   const int*   __restrict__ labels,
                                   int*         __restrict__ predictions,
                                   int n_samples, int k,
                                   int n_classes) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_samples) return;

  // Partial selection: find k smallest distances (excluding self, dist==0).
  float  min_dist[kMaxK];
  int    min_idx[kMaxK];
  for (int m = 0; m < k; ++m) {
    min_dist[m] = 1e30f;
    min_idx[m]  = -1;
  }

  for (int j = 0; j < n_samples; ++j) {
    if (j == i) continue;
    const float d = dist_matrix[i * n_samples + j];

    // Insert d into the sorted top-k list if it is smaller than the current max.
    if (d < min_dist[k - 1]) {
      // Find insertion position.
      int pos = k - 1;
      while (pos > 0 && d < min_dist[pos - 1]) --pos;
      // Shift right.
      for (int m = k - 1; m > pos; --m) {
        min_dist[m] = min_dist[m - 1];
        min_idx[m]  = min_idx[m - 1];
      }
      min_dist[pos] = d;
      min_idx[pos]  = j;
    }
  }

  // Majority vote among k neighbours.
  int votes[3] = {0, 0, 0};  // supports up to 3 classes
  for (int m = 0; m < k; ++m) {
    if (min_idx[m] >= 0) {
      const int lbl = labels[min_idx[m]];
      if (lbl >= 0 && lbl < n_classes) ++votes[lbl];
    }
  }

  int best = 0;
  for (int c = 1; c < n_classes; ++c) {
    if (votes[c] > votes[best]) best = c;
  }
  predictions[i] = best;
}

// ---------------------------------------------------------------------------
// NPP helper wrappers
// ---------------------------------------------------------------------------

// Compute sum of a 1-D float array on device using NPP.
// Returns the sum copied back to the host.
static float NppSum(const Npp32f* d_array, int length) {
  int buf_size = 0;
  NPP_CHECK(nppsSumGetBufferSize_32f(length, &buf_size));

  Npp8u*  d_buf = nullptr;
  Npp32f* d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buf), buf_size));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sum), sizeof(Npp32f)));

  NPP_CHECK(nppsSum_32f(d_array, length, d_sum, d_buf));

  float h_sum = 0.f;
  CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_sum));
  return h_sum;
}

// Compute min and max of a 1-D float array on device using NPP.
static void NppMinMax(const Npp32f* d_array, int length,
                      float* h_min, float* h_max) {
  int buf_size = 0;
  NPP_CHECK(nppsMinMaxGetBufferSize_32f(length, &buf_size));

  Npp8u*  d_buf = nullptr;
  Npp32f* d_min = nullptr;
  Npp32f* d_max = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buf), buf_size));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_min), sizeof(Npp32f)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_max), sizeof(Npp32f)));

  NPP_CHECK(nppsMinMax_32f(d_array, length, d_min, d_max, d_buf));

  CUDA_CHECK(cudaMemcpy(h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_min));
  CUDA_CHECK(cudaFree(d_max));
}

// Apply z-score normalisation to a 1-D float array using NPP:
//   pDst[i] = (pSrc[i] - mean) / std_dev
static void NppZScoreNormalize(const Npp32f* d_src, Npp32f* d_dst,
                                int length, float mean, float std_dev) {
  if (std_dev < 1e-9f) {
    // Degenerate: constant feature – copy unchanged.
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, length * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    return;
  }
  NPP_CHECK(nppsNormalize_32f(d_src, d_dst, length, mean, std_dev));
}

// ---------------------------------------------------------------------------
// Public pipeline implementation
// ---------------------------------------------------------------------------

PipelineResult RunGPUPipeline(const std::vector<IrisSample>& samples,
                               const ClassifierConfig&         config) {
  PipelineResult result;
  if (samples.empty()) return result;

  const int n_samples  = static_cast<int>(samples.size());
  const int n_features = kNumFeatures;
  const int n_classes  = kNumClasses;

  // -- Sanity-check k --
  const int k = std::min(config.k_neighbors, std::min(kMaxK, n_samples - 1));

  // -- Verify CUDA device --
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "RunGPUPipeline: no CUDA-capable device found.\n";
    return result;
  }
  CUDA_CHECK(cudaSetDevice(0));

  // -----------------------------------------------------------------------
  // Host → flat feature matrix (row-major, 32-bit float) + label array
  // -----------------------------------------------------------------------
  std::vector<float> h_features(static_cast<std::size_t>(n_samples * n_features));
  std::vector<int>   h_labels(static_cast<std::size_t>(n_samples));
  for (int s = 0; s < n_samples; ++s) {
    for (int f = 0; f < n_features; ++f) {
      h_features[static_cast<std::size_t>(s * n_features + f)] =
          samples[static_cast<std::size_t>(s)].features[f];
    }
    h_labels[static_cast<std::size_t>(s)] =
        samples[static_cast<std::size_t>(s)].label;
  }

  // -----------------------------------------------------------------------
  // Allocate device buffers
  // -----------------------------------------------------------------------
  float *d_features    = nullptr;  // row-major features   [n_samples × n_features]
  float *d_col_arrays  = nullptr;  // column-major layouts [n_features × n_samples]
  float *d_col_norm    = nullptr;  // normalised column-major layout
  float *d_means       = nullptr;  // per-feature means        [n_features]
  float *d_sq_sum      = nullptr;  // per-feature sum(sq-dev)  [n_features]
  float *d_std_dev     = nullptr;  // per-feature std devs     [n_features]
  float *d_dist_matrix = nullptr;  // distance matrix  [n_samples × n_samples]
  int   *d_labels      = nullptr;  // ground-truth labels      [n_samples]
  int   *d_predictions = nullptr;  // predicted labels         [n_samples]

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_features),
                        n_samples * n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_col_arrays),
                        n_features * n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_col_norm),
                        n_features * n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_means),
                        n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sq_sum),
                        n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_std_dev),
                        n_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dist_matrix),
                        n_samples * n_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_labels),
                        n_samples * sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_predictions),
                        n_samples * sizeof(int)));

  // -----------------------------------------------------------------------
  // Upload data to device
  // -----------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(d_features, h_features.data(),
                        n_samples * n_features * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(),
                        n_samples * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sq_sum, 0, n_features * sizeof(float)));

  // -----------------------------------------------------------------------
  // CUDA event timers
  // -----------------------------------------------------------------------
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const auto wall_t0 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaEventRecord(ev_start));

  // -----------------------------------------------------------------------
  // Stage 1: Transpose row-major → column-major  [custom CUDA kernel]
  // -----------------------------------------------------------------------
  {
    const dim3 block(32, 4);
    const dim3 grid(
        (static_cast<unsigned>(n_samples)  + block.x - 1) / block.x,
        (static_cast<unsigned>(n_features) + block.y - 1) / block.y);
    TransposeToColumnMajorKernel<<<grid, block>>>(
        d_features, d_col_arrays, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());
  }

  // -----------------------------------------------------------------------
  // Stage 2: Per-feature sum via NPP → mean on host  [CUDA NPP signal]
  // -----------------------------------------------------------------------
  float h_means[kNumFeatures]   = {};
  float h_std_devs[kNumFeatures] = {};
  float h_mins[kNumFeatures]    = {};
  float h_maxs[kNumFeatures]    = {};

  for (int f = 0; f < n_features; ++f) {
    const Npp32f* col_ptr = d_col_arrays + f * n_samples;
    const float sum = NppSum(col_ptr, n_samples);
    h_means[f] = sum / static_cast<float>(n_samples);
  }

  // Upload means to device for the std-dev kernel.
  CUDA_CHECK(cudaMemcpy(d_means, h_means, n_features * sizeof(float),
                        cudaMemcpyHostToDevice));

  // -----------------------------------------------------------------------
  // Stage 3: Per-feature std dev  [custom CUDA kernel]
  // -----------------------------------------------------------------------
  {
    const dim3 block(64, 4);
    const dim3 grid(
        (static_cast<unsigned>(n_samples)  + block.x - 1) / block.x,
        (static_cast<unsigned>(n_features) + block.y - 1) / block.y);
    SumSquaredDevKernel<<<grid, block>>>(
        d_col_arrays, d_means, d_sq_sum, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());
  }
  {
    const dim3 block(kNumFeatures);
    const dim3 grid(1);
    FinaliseStdDevKernel<<<grid, block>>>(
        d_std_dev, d_sq_sum, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaMemcpy(h_std_devs, d_std_dev, n_features * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // -----------------------------------------------------------------------
  // Stage 4: Per-feature min/max via NPP  [CUDA NPP signal]
  // -----------------------------------------------------------------------
  for (int f = 0; f < n_features; ++f) {
    const Npp32f* col_ptr = d_col_arrays + f * n_samples;
    NppMinMax(col_ptr, n_samples, &h_mins[f], &h_maxs[f]);
  }

  // Store stats in result.
  for (int f = 0; f < n_features; ++f) {
    result.stats.mean[f]    = h_means[f];
    result.stats.std_dev[f] = h_std_devs[f];
    result.stats.min_val[f] = h_mins[f];
    result.stats.max_val[f] = h_maxs[f];
  }

  // -----------------------------------------------------------------------
  // Stage 5: Z-score normalisation via NPP  [CUDA NPP signal]
  // -----------------------------------------------------------------------
  if (config.normalize) {
    for (int f = 0; f < n_features; ++f) {
      const Npp32f* col_src = d_col_arrays + f * n_samples;
      Npp32f*       col_dst = d_col_norm   + f * n_samples;
      NppZScoreNormalize(col_src, col_dst, n_samples,
                         h_means[f], h_std_devs[f]);
    }
    // Transpose normalised column-major back to row-major for distance kernel.
    const dim3 block(32, 4);
    const dim3 grid(
        (static_cast<unsigned>(n_samples)  + block.x - 1) / block.x,
        (static_cast<unsigned>(n_features) + block.y - 1) / block.y);
    TransposeToRowMajorKernel<<<grid, block>>>(
        d_col_norm, d_features, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());
  }

  // -----------------------------------------------------------------------
  // Stage 6: Pairwise Euclidean distance matrix  [custom CUDA kernel]
  // -----------------------------------------------------------------------
  {
    const dim3 block(16, 16);
    const dim3 grid(
        (static_cast<unsigned>(n_samples) + block.x - 1) / block.x,
        (static_cast<unsigned>(n_samples) + block.y - 1) / block.y);
    DistanceMatrixKernel<<<grid, block>>>(
        d_features, d_dist_matrix, n_samples, n_features);
    CUDA_CHECK(cudaGetLastError());
  }

  // -----------------------------------------------------------------------
  // Stage 7: k-NN classification  [custom CUDA kernel]
  // -----------------------------------------------------------------------
  {
    const dim3 block(32);
    const dim3 grid(
        (static_cast<unsigned>(n_samples) + block.x - 1) / block.x);
    KNNClassifyKernel<<<grid, block>>>(
        d_dist_matrix, d_labels, d_predictions, n_samples, k, n_classes);
    CUDA_CHECK(cudaGetLastError());
  }

  // -----------------------------------------------------------------------
  // GPU timing
  // -----------------------------------------------------------------------
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float gpu_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
  result.gpu_time_ms = static_cast<double>(gpu_ms);

  const auto wall_t1 = std::chrono::high_resolution_clock::now();
  result.wall_time_ms =
      std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

  // -----------------------------------------------------------------------
  // Download predictions
  // -----------------------------------------------------------------------
  std::vector<int> h_predictions(static_cast<std::size_t>(n_samples));
  CUDA_CHECK(cudaMemcpy(h_predictions.data(), d_predictions,
                        n_samples * sizeof(int), cudaMemcpyDeviceToHost));
  result.predictions = h_predictions;

  // -----------------------------------------------------------------------
  // Compute accuracy on host
  // -----------------------------------------------------------------------
  int correct = 0;
  for (int s = 0; s < n_samples; ++s) {
    if (h_predictions[static_cast<std::size_t>(s)] ==
        h_labels[static_cast<std::size_t>(s)]) {
      ++correct;
    }
    if (config.verbose) {
      std::cout << "  sample " << (s + 1)
                << "  actual=" << kClassNames[h_labels[static_cast<std::size_t>(s)]]
                << "  predicted=" << kClassNames[h_predictions[static_cast<std::size_t>(s)]]
                << (h_predictions[static_cast<std::size_t>(s)] ==
                    h_labels[static_cast<std::size_t>(s)] ? "" : "  *** WRONG ***")
                << "\n";
    }
  }
  result.correct  = correct;
  result.accuracy = static_cast<float>(correct) /
                    static_cast<float>(n_samples) * 100.f;

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------
  CUDA_CHECK(cudaFree(d_features));
  CUDA_CHECK(cudaFree(d_col_arrays));
  CUDA_CHECK(cudaFree(d_col_norm));
  CUDA_CHECK(cudaFree(d_means));
  CUDA_CHECK(cudaFree(d_sq_sum));
  CUDA_CHECK(cudaFree(d_std_dev));
  CUDA_CHECK(cudaFree(d_dist_matrix));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_predictions));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  return result;
}
