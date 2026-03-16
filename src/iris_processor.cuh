// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// iris_processor.cuh  –  GPU pipeline interface for Iris dataset analysis.
//
// GPU computation overview:
//   1. Feature extraction     – TransposeToColumnMajorKernel  (custom CUDA)
//   2. Column sums            – nppsSum_32f                   (CUDA NPP signal)
//   3. Std-dev computation    – StdDevKernel                  (custom CUDA)
//   4. Min/Max per feature    – nppsMinMax_32f                (CUDA NPP signal)
//   5. Feature normalization  – nppsNormalize_32f             (CUDA NPP signal)
//   6. Pairwise distances     – DistanceMatrixKernel          (custom CUDA)
//   7. k-NN classification    – KNNClassifyKernel             (custom CUDA)

#pragma once

#include <string>
#include <vector>

#include "csv_utils.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct ClassifierConfig {
  int   k_neighbors  = 5;      // k for k-nearest-neighbours
  bool  normalize    = true;   // z-score normalise features before distance
  bool  verbose      = false;  // print per-sample progress
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

// Per-feature statistics computed on the GPU.
struct FeatureStats {
  float mean[kNumFeatures]    = {};
  float std_dev[kNumFeatures] = {};
  float min_val[kNumFeatures] = {};
  float max_val[kNumFeatures] = {};
};

// Aggregated output from the full GPU pipeline.
struct PipelineResult {
  std::vector<int> predictions;  // Predicted label per sample
  FeatureStats     stats;        // Per-feature statistics (GPU-computed)
  int              correct       = 0;
  float            accuracy      = 0.f;
  double           gpu_time_ms   = 0.0;  // CUDA event timing
  double           wall_time_ms  = 0.0;  // Wall-clock timing
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Run the full GPU pipeline on |samples| and return classification results.
PipelineResult RunGPUPipeline(const std::vector<IrisSample>& samples,
                               const ClassifierConfig&         config);
