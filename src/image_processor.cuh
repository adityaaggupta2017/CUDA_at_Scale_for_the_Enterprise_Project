// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// image_processor.cuh  –  GPU image processing interface.
//
// Four operations are supported:
//   kGaussianBlur          – 5×5 Gaussian blur via CUDA NPP
//   kEdgeDetection         – Sobel edge magnitude via a custom CUDA kernel
//   kHistogramEqualization – Adaptive contrast stretch via CUDA NPP
//   kSharpen               – Unsharp-mask sharpening via custom CUDA kernels
//   kAll                   – All four operations, each saved to a separate file

#pragma once

#include <string>
#include <vector>

#include "pnm_utils.h"

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

enum class Operation {
  kGaussianBlur          = 0,
  kEdgeDetection         = 1,
  kHistogramEqualization = 2,
  kSharpen               = 3,
  kAll                   = 4,
};

// Configuration passed to ProcessBatch.
struct Config {
  Operation operation     = Operation::kAll;
  float     sharpen_amount = 1.5f;  // Multiplier for the detail layer (≥ 0)
  bool      verbose        = false;
};

// Aggregate timing / throughput statistics returned by ProcessBatch.
struct ProcessingStats {
  int    images_processed   = 0;
  int    images_failed      = 0;
  double wall_time_ms       = 0.0;  // Total wall-clock time (host + device)
  double gpu_time_ms        = 0.0;  // Pure GPU kernel + NPP time (CUDA events)
};

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

// Process all images in |inputs|, append results to |outputs|, and return
// aggregate statistics.  Output filenames encode the applied operation,
// e.g.  "image_001_blur.pgm".
//
// When config.operation == kAll, four output images are produced per input.
ProcessingStats ProcessBatch(const std::vector<GrayscaleImage>& inputs,
                             std::vector<GrayscaleImage>*       outputs,
                             const Config&                       config);

// Return a human-readable label for the given operation.
const char* OperationLabel(Operation op);

// Parse an operation name string ("blur", "edges", "equalize", "sharpen",
// "all") into an Operation enum.  Returns false if the string is unrecognised.
bool ParseOperation(const std::string& name, Operation* op);
