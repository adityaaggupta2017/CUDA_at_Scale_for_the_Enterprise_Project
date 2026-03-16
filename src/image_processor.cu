// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// image_processor.cu  –  GPU image processing implementation.
//
// GPU operations:
//   1. Gaussian Blur        – nppiFilterGaussBorder_8u_C1R      (CUDA NPP)
//   2. Edge Detection       – SobelKernel                       (custom CUDA)
//   3. Histogram Equalise   – nppiEqualizeHist_8u_C1R           (CUDA NPP)
//   4. Unsharp-mask Sharpen – GaussianBlur + UnsharpMaskKernel  (NPP + custom)

#include "image_processor.cuh"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>
#include <nppi_statistics_functions.h>

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d – %s\n",                      \
                   __FILE__, __LINE__, cudaGetErrorString(_err));              \
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

// Sobel edge-detection kernel.
// Computes gradient magnitude |Gx| + |Gy| using 3×3 Sobel operators and
// clamps the result to [0, 255].  Border pixels are set to zero.
__global__ void SobelKernel(const uint8_t* __restrict__ src,
                             uint8_t*       __restrict__ dst,
                             int width, int height) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  if (x >= width || y >= height) return;

  // Border pixels → zero (no neighbours available).
  if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
    dst[y * width + x] = 0;
    return;
  }

  // Load 3×3 neighbourhood.
  const int tl = src[(y-1)*width + (x-1)];
  const int tc = src[(y-1)*width +  x   ];
  const int tr = src[(y-1)*width + (x+1)];
  const int ml = src[ y   *width + (x-1)];
  const int mr = src[ y   *width + (x+1)];
  const int bl = src[(y+1)*width + (x-1)];
  const int bc = src[(y+1)*width +  x   ];
  const int br = src[(y+1)*width + (x+1)];

  // Sobel X  (detects vertical edges)
  const int gx = -tl - 2*ml - bl + tr + 2*mr + br;
  // Sobel Y  (detects horizontal edges)
  const int gy = -tl - 2*tc - tr + bl + 2*bc + br;

  // Approximate magnitude: |Gx| + |Gy| (faster than sqrt, same visual result)
  const int mag = min(abs(gx) + abs(gy), 255);
  dst[y * width + x] = static_cast<uint8_t>(mag);
}

// Unsharp-mask sharpening kernel.
// sharpened = clamp(original + amount * (original – blurred), 0, 255)
__global__ void UnsharpMaskKernel(const uint8_t* __restrict__ original,
                                   const uint8_t* __restrict__ blurred,
                                   uint8_t*       __restrict__ dst,
                                   int num_pixels,
                                   float amount) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_pixels) return;

  const float orig   = static_cast<float>(original[idx]);
  const float blur   = static_cast<float>(blurred[idx]);
  const float result = orig + amount * (orig - blur);
  dst[idx] = static_cast<uint8_t>(fmaxf(0.f, fminf(255.f, result)));
}

// ---------------------------------------------------------------------------
// NPP wrappers (operate on already-allocated device buffers)
// ---------------------------------------------------------------------------

// Apply a 5×5 Gaussian blur using NPP.  src and dst may point to the same
// memory only if the NPP implementation permits in-place operation (it does
// not for filter functions, so always pass distinct buffers).
static void NppGaussianBlur(const Npp8u* d_src, int src_step,
                             Npp8u*       d_dst, int dst_step,
                             int width,          int height) {
  const NppiSize src_size    = {width, height};
  const NppiSize roi_size    = {width, height};
  const NppiPoint src_offset = {0, 0};

  NPP_CHECK(nppiFilterGaussBorder_8u_C1R(
      d_src, src_step,
      src_size, src_offset,
      d_dst, dst_step,
      roi_size,
      NPP_MASK_SIZE_5_X_5,
      NPP_BORDER_REPLICATE));
}

// Apply histogram equalisation using NPP histogram + LUT pipeline.
//
// Steps:
//   1. nppiHistogramEven_8u_C1R  – compute 256-bin histogram on device
//   2. Compute CDF on host       – only 256 values, negligible cost
//   3. nppiLUT_8u_C1R            – apply equalisation mapping as a LUT
static void NppHistogramEqualization(const Npp8u* d_src, int src_step,
                                      Npp8u*       d_dst, int dst_step,
                                      int width,          int height) {
  const NppiSize roi_size = {width, height};
  constexpr int kBins   = 256;
  constexpr int kLevels = kBins + 1;  // nppiHistogramEven uses level count

  // --- Step 1: Query scratch buffer and compute histogram on device ---
  int scratch_size = 0;
  NPP_CHECK(nppiHistogramEvenGetBufferSize_8u_C1R(roi_size, kLevels,
                                                   &scratch_size));
  Npp8u*  d_scratch = nullptr;
  Npp32s* d_hist    = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scratch), scratch_size));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hist),
                        kBins * sizeof(Npp32s)));

  NPP_CHECK(nppiHistogramEven_8u_C1R(
      d_src, src_step, roi_size,
      d_hist, kLevels,
      /*nLowerLevel=*/0, /*nUpperLevel=*/256,
      d_scratch));

  CUDA_CHECK(cudaFree(d_scratch));

  // --- Step 2: Copy histogram to host and compute CDF ---
  Npp32s h_hist[kBins];
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, kBins * sizeof(Npp32s),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_hist));

  // Build CDF and find the minimum non-zero CDF value.
  long long cdf[kBins];
  cdf[0] = h_hist[0];
  for (int i = 1; i < kBins; ++i) {
    cdf[i] = cdf[i - 1] + h_hist[i];
  }
  const long long total   = static_cast<long long>(width) * height;
  long long       cdf_min = 0;
  for (int i = 0; i < kBins; ++i) {
    if (cdf[i] > 0) { cdf_min = cdf[i]; break; }
  }

  // LUT values: equalised output for each input intensity.
  Npp32s h_lut_values[kBins];
  for (int v = 0; v < kBins; ++v) {
    if (total == cdf_min) {
      h_lut_values[v] = static_cast<Npp32s>(v);  // degenerate: identity
    } else {
      h_lut_values[v] = static_cast<Npp32s>(
          std::lround(static_cast<double>(cdf[v] - cdf_min) * 255.0 /
                      static_cast<double>(total - cdf_min)));
    }
  }

  // LUT input levels: 0, 1, …, 255  (nppiLUT needs explicit level array)
  Npp32s h_lut_levels[kBins];
  for (int v = 0; v < kBins; ++v) h_lut_levels[v] = v;

  // --- Step 3: Apply LUT on device ---
  Npp32s* d_lut_values = nullptr;
  Npp32s* d_lut_levels = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lut_values),
                        kBins * sizeof(Npp32s)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lut_levels),
                        kBins * sizeof(Npp32s)));
  CUDA_CHECK(cudaMemcpy(d_lut_values, h_lut_values, kBins * sizeof(Npp32s),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lut_levels, h_lut_levels, kBins * sizeof(Npp32s),
                        cudaMemcpyHostToDevice));

  NPP_CHECK(nppiLUT_8u_C1R(
      d_src, src_step,
      d_dst, dst_step,
      roi_size,
      d_lut_values, d_lut_levels, kBins));

  CUDA_CHECK(cudaFree(d_lut_values));
  CUDA_CHECK(cudaFree(d_lut_levels));
}

// ---------------------------------------------------------------------------
// Single-image dispatch
// ---------------------------------------------------------------------------

// Apply one specific operation (not kAll) to a single image stored on the
// device.  |d_tmp| is an already-allocated temporary buffer of at least
// width*height bytes (used by the sharpen path for the blur intermediate).
static void ApplyOperation(Operation    op,
                            const Npp8u* d_src,  int src_step,
                            Npp8u*       d_dst,  int dst_step,
                            Npp8u*       d_tmp,
                            int          width,  int height,
                            float        sharpen_amount) {
  const int num_pixels = width * height;
  const dim3 block2d(16, 16);
  const dim3 grid2d(
      (static_cast<unsigned>(width)  + block2d.x - 1) / block2d.x,
      (static_cast<unsigned>(height) + block2d.y - 1) / block2d.y);
  const dim3 block1d(256);
  const dim3 grid1d((static_cast<unsigned>(num_pixels) + block1d.x - 1) /
                    block1d.x);

  switch (op) {
    case Operation::kGaussianBlur:
      NppGaussianBlur(d_src, src_step, d_dst, dst_step, width, height);
      break;

    case Operation::kEdgeDetection:
      SobelKernel<<<grid2d, block2d>>>(
          reinterpret_cast<const uint8_t*>(d_src),
          reinterpret_cast<uint8_t*>(d_dst),
          width, height);
      CUDA_CHECK(cudaGetLastError());
      break;

    case Operation::kHistogramEqualization:
      NppHistogramEqualization(d_src, src_step, d_dst, dst_step, width, height);
      break;

    case Operation::kSharpen:
      // Step 1: blur the source into the temp buffer.
      NppGaussianBlur(d_src, src_step, d_tmp, src_step, width, height);
      // Step 2: compute unsharp mask from original + blurred.
      UnsharpMaskKernel<<<grid1d, block1d>>>(
          reinterpret_cast<const uint8_t*>(d_src),
          reinterpret_cast<const uint8_t*>(d_tmp),
          reinterpret_cast<uint8_t*>(d_dst),
          num_pixels, sharpen_amount);
      CUDA_CHECK(cudaGetLastError());
      break;

    default:
      break;
  }
}

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

const char* OperationLabel(Operation op) {
  switch (op) {
    case Operation::kGaussianBlur:          return "blur";
    case Operation::kEdgeDetection:         return "edges";
    case Operation::kHistogramEqualization: return "equalized";
    case Operation::kSharpen:               return "sharpened";
    case Operation::kAll:                   return "all";
    default:                                return "unknown";
  }
}

bool ParseOperation(const std::string& name, Operation* op) {
  if (name == "blur")      { *op = Operation::kGaussianBlur;          return true; }
  if (name == "edges")     { *op = Operation::kEdgeDetection;         return true; }
  if (name == "equalize")  { *op = Operation::kHistogramEqualization; return true; }
  if (name == "sharpen")   { *op = Operation::kSharpen;               return true; }
  if (name == "all")       { *op = Operation::kAll;                   return true; }
  return false;
}

// Build the output filename by injecting "_<label>" before the extension.
static std::string MakeOutputFilename(const std::string& base,
                                      const char*         label) {
  // Strip directory path.
  const std::size_t slash = base.find_last_of("/\\");
  std::string stem = (slash == std::string::npos) ? base : base.substr(slash + 1);

  // Strip ".pgm" extension if present.
  const std::size_t dot = stem.rfind(".pgm");
  if (dot != std::string::npos) stem = stem.substr(0, dot);

  return stem + "_" + label + ".pgm";
}

ProcessingStats ProcessBatch(const std::vector<GrayscaleImage>& inputs,
                             std::vector<GrayscaleImage>*       outputs,
                             const Config&                       config) {
  ProcessingStats stats;
  if (inputs.empty()) return stats;

  // -- CUDA device check --
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "ProcessBatch: no CUDA-capable device found.\n";
    return stats;
  }
  CUDA_CHECK(cudaSetDevice(0));

  // -- CUDA events for GPU timing --
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // -- Determine which single operations to run --
  std::vector<Operation> ops_to_run;
  if (config.operation == Operation::kAll) {
    ops_to_run = {
        Operation::kGaussianBlur,
        Operation::kEdgeDetection,
        Operation::kHistogramEqualization,
        Operation::kSharpen,
    };
  } else {
    ops_to_run.push_back(config.operation);
  }

  // -- Wall-clock timer start --
  const auto wall_start = std::chrono::high_resolution_clock::now();

  // -- GPU event start --
  CUDA_CHECK(cudaEventRecord(ev_start));

  // Allocate device buffers once (reused across all images).
  // We allocate for the maximum image size in the batch.
  int max_pixels = 0;
  for (const auto& img : inputs) {
    max_pixels = std::max(max_pixels, img.width * img.height);
  }

  Npp8u* d_src = nullptr;
  Npp8u* d_dst = nullptr;
  Npp8u* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src), max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dst), max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp), max_pixels));

  for (const auto& img : inputs) {
    const int num_pixels = img.width * img.height;
    // step = width for tightly packed (no padding) images.
    const int step = img.width;

    // Upload image to device.
    CUDA_CHECK(cudaMemcpy(d_src, img.data.data(), num_pixels,
                          cudaMemcpyHostToDevice));

    for (const Operation op : ops_to_run) {
      // Apply the GPU operation.
      ApplyOperation(op,
                     d_src, step,
                     d_dst, step,
                     d_tmp,
                     img.width, img.height,
                     config.sharpen_amount);
      CUDA_CHECK(cudaGetLastError());

      // Download result.
      GrayscaleImage result;
      result.width    = img.width;
      result.height   = img.height;
      result.filename = MakeOutputFilename(img.filename, OperationLabel(op));
      result.data.resize(static_cast<std::size_t>(num_pixels));
      CUDA_CHECK(cudaMemcpy(result.data.data(), d_dst, num_pixels,
                            cudaMemcpyDeviceToHost));

      if (config.verbose) {
        std::cout << "  processed  " << img.filename
                  << "  →  " << result.filename << "\n";
      }

      outputs->push_back(std::move(result));
    }
    ++stats.images_processed;
  }

  // -- GPU event stop and synchronise --
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float gpu_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
  stats.gpu_time_ms = static_cast<double>(gpu_ms);

  // -- Wall-clock timer stop --
  const auto wall_stop = std::chrono::high_resolution_clock::now();
  stats.wall_time_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();

  // -- Cleanup --
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  return stats;
}
