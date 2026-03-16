// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// main.cpp  –  CLI entry-point for the CUDA batch image processor.
//
// Usage:
//   cuda_image_processor [OPTIONS]
//
// Run with --help for full usage information.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "image_processor.cuh"
#include "pnm_utils.h"

static constexpr const char* kVersion        = "1.0.0";
static constexpr const char* kDefaultInputDir  = "data/input";
static constexpr const char* kDefaultOutputDir = "data/output";
static constexpr const char* kDefaultLogFile   = "results/processing.log";
static constexpr const char* kDefaultOperation = "all";
static constexpr int          kDefaultBatchSize  = 50;

// ---------------------------------------------------------------------------
// Usage / help
// ---------------------------------------------------------------------------

static void PrintUsage(const char* program_name) {
  std::cout
      << "CUDA Batch Image Processor  v" << kVersion << "\n"
         "GPU-accelerated processing of grayscale PGM images using\n"
         "custom CUDA kernels and the NVIDIA NPP library.\n\n"
         "Usage:\n"
         "  " << program_name << " [OPTIONS]\n\n"
         "Options:\n"
         "  -i, --input <dir>        Input directory containing .pgm files\n"
         "                           (default: " << kDefaultInputDir << ")\n"
         "  -o, --output <dir>       Output directory for processed images\n"
         "                           (default: " << kDefaultOutputDir << ")\n"
         "  -p, --operation <name>   Processing operation to apply:\n"
         "                             blur      – 5×5 Gaussian blur  (NPP)\n"
         "                             edges     – Sobel edge detection (CUDA kernel)\n"
         "                             equalize  – Histogram equalization (NPP)\n"
         "                             sharpen   – Unsharp-mask sharpening (CUDA kernel)\n"
         "                             all       – All four operations (default)\n"
         "  -a, --sharpen-amount <f> Sharpening multiplier  (default: 1.5)\n"
         "  -b, --batch-size <n>     Images loaded per processing round\n"
         "                           (default: " << kDefaultBatchSize << ")\n"
         "  -l, --log <file>         Path for the processing log\n"
         "                           (default: " << kDefaultLogFile << ")\n"
         "  -v, --verbose            Print per-image progress\n"
         "  -h, --help               Show this message and exit\n\n"
         "Examples:\n"
         "  " << program_name << "                            # process all images with all ops\n"
         "  " << program_name << " -p edges -v               # edge detection, verbose\n"
         "  " << program_name << " -i /tmp/imgs -o /tmp/out  # custom directories\n"
         "  " << program_name << " -p sharpen -a 2.0         # stronger sharpening\n";
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
  std::string input_dir     = kDefaultInputDir;
  std::string output_dir    = kDefaultOutputDir;
  std::string log_file      = kDefaultLogFile;
  std::string operation_str = kDefaultOperation;
  Operation   operation     = Operation::kAll;
  float       sharpen_amount = 1.5f;
  int         batch_size    = kDefaultBatchSize;
  bool        verbose       = false;
};

static bool ParseArgs(int argc, char* argv[], Args* args) {
  static const option kLongOpts[] = {
      {"input",          required_argument, nullptr, 'i'},
      {"output",         required_argument, nullptr, 'o'},
      {"operation",      required_argument, nullptr, 'p'},
      {"sharpen-amount", required_argument, nullptr, 'a'},
      {"batch-size",     required_argument, nullptr, 'b'},
      {"log",            required_argument, nullptr, 'l'},
      {"verbose",        no_argument,       nullptr, 'v'},
      {"help",           no_argument,       nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  int opt = 0;
  while ((opt = getopt_long(argc, argv, "i:o:p:a:b:l:vh", kLongOpts,
                            nullptr)) != -1) {
    switch (opt) {
      case 'i': args->input_dir      = optarg; break;
      case 'o': args->output_dir     = optarg; break;
      case 'p': args->operation_str  = optarg; break;
      case 'a': args->sharpen_amount = std::stof(optarg); break;
      case 'b': args->batch_size     = std::stoi(optarg); break;
      case 'l': args->log_file       = optarg; break;
      case 'v': args->verbose        = true;   break;
      case 'h': PrintUsage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  return false;
    }
  }

  if (!ParseOperation(args->operation_str, &args->operation)) {
    std::cerr << "Unknown operation '" << args->operation_str
              << "'. Valid: blur, edges, equalize, sharpen, all\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

static void WriteLog(const std::string&       log_path,
                     const Args&              args,
                     const std::vector<GrayscaleImage>& inputs,
                     const std::vector<GrayscaleImage>& outputs,
                     const ProcessingStats&   stats) {
  // Ensure the results directory exists.
  const std::size_t last_slash = log_path.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    EnsureDirectory(log_path.substr(0, last_slash));
  }

  std::ofstream log(log_path);
  if (!log.is_open()) {
    std::cerr << "Warning: could not write log to '" << log_path << "'\n";
    return;
  }

  log << "=== CUDA Batch Image Processor  v" << kVersion << " ===\n\n";
  log << "Input directory  : " << args.input_dir     << "\n";
  log << "Output directory : " << args.output_dir    << "\n";
  log << "Operation        : " << args.operation_str << "\n";
  log << "Sharpen amount   : " << args.sharpen_amount << "\n";
  log << "Batch size       : " << args.batch_size    << "\n\n";

  log << "--- Input images (" << inputs.size() << ") ---\n";
  for (const auto& img : inputs) {
    log << "  " << img.filename
        << "  (" << img.width << "×" << img.height << ")\n";
  }

  log << "\n--- Output images (" << outputs.size() << ") ---\n";
  for (const auto& img : outputs) {
    log << "  " << img.filename
        << "  (" << img.width << "×" << img.height << ")\n";
  }

  log << "\n--- Timing ---\n";
  log << std::fixed << std::setprecision(2);
  log << "  Images processed : " << stats.images_processed << "\n";
  log << "  GPU time         : " << stats.gpu_time_ms  << " ms\n";
  log << "  Wall-clock time  : " << stats.wall_time_ms << " ms\n";
  if (stats.images_processed > 0 && stats.gpu_time_ms > 0) {
    log << "  Throughput (GPU) : "
        << (1000.0 * stats.images_processed / stats.gpu_time_ms)
        << " images/s\n";
  }
  log.close();
  std::cout << "Log written to " << log_path << "\n";
}

static void PrintSummary(const ProcessingStats& stats,
                         const std::string&     operation_str) {
  std::cout << "\n========================================\n";
  std::cout << " CUDA Batch Image Processor – Summary\n";
  std::cout << "========================================\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  Operation        : " << operation_str      << "\n";
  std::cout << "  Images processed : " << stats.images_processed << "\n";
  std::cout << "  GPU time         : " << stats.gpu_time_ms  << " ms\n";
  std::cout << "  Wall-clock time  : " << stats.wall_time_ms << " ms\n";
  if (stats.images_processed > 0 && stats.gpu_time_ms > 0.0) {
    std::cout << "  Throughput (GPU) : "
              << (1000.0 * stats.images_processed / stats.gpu_time_ms)
              << " images/s\n";
  }
  std::cout << "========================================\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    std::cerr << "Run with --help for usage information.\n";
    return EXIT_FAILURE;
  }

  // Print startup banner.
  std::cout << "CUDA Batch Image Processor  v" << kVersion << "\n";
  std::cout << "Operation : " << args.operation_str
            << "  |  Input : " << args.input_dir
            << "  |  Output : " << args.output_dir << "\n\n";

  // Ensure output directory exists.
  if (!EnsureDirectory(args.output_dir)) {
    std::cerr << "Failed to create output directory '" << args.output_dir
              << "'\n";
    return EXIT_FAILURE;
  }

  // Scan for PGM files in the input directory.
  const std::vector<std::string> pgm_paths =
      ListFiles(args.input_dir, ".pgm");
  if (pgm_paths.empty()) {
    std::cerr << "No .pgm files found in '" << args.input_dir << "'\n"
                 "Run  python3 scripts/generate_test_data.py  to create "
                 "synthetic test images.\n";
    return EXIT_FAILURE;
  }
  std::cout << "Found " << pgm_paths.size() << " PGM file(s) in '"
            << args.input_dir << "'\n";

  // Accumulators across all batches.
  ProcessingStats total_stats;
  std::vector<GrayscaleImage> all_inputs;
  std::vector<GrayscaleImage> all_outputs;

  // Configure GPU processing.
  Config config;
  config.operation      = args.operation;
  config.sharpen_amount = args.sharpen_amount;
  config.verbose        = args.verbose;

  // Process in batches to keep host-memory pressure bounded.
  const int total_images = static_cast<int>(pgm_paths.size());
  int batch_start = 0;
  int batch_num   = 1;

  while (batch_start < total_images) {
    const int batch_end =
        std::min(batch_start + args.batch_size, total_images);
    std::cout << "\nBatch " << batch_num
              << "  [" << (batch_start + 1) << " – " << batch_end
              << " / " << total_images << "]\n";

    // Load images for this batch.
    std::vector<GrayscaleImage> batch_inputs;
    batch_inputs.reserve(static_cast<std::size_t>(batch_end - batch_start));
    for (int i = batch_start; i < batch_end; ++i) {
      GrayscaleImage img;
      if (LoadPGM(pgm_paths[static_cast<std::size_t>(i)], &img)) {
        batch_inputs.push_back(std::move(img));
      } else {
        std::cerr << "  Warning: skipping unreadable file '"
                  << pgm_paths[static_cast<std::size_t>(i)] << "'\n";
        ++total_stats.images_failed;
      }
    }

    // Run GPU processing.
    std::vector<GrayscaleImage> batch_outputs;
    const ProcessingStats batch_stats =
        ProcessBatch(batch_inputs, &batch_outputs, config);

    // Accumulate statistics.
    total_stats.images_processed += batch_stats.images_processed;
    total_stats.gpu_time_ms      += batch_stats.gpu_time_ms;
    total_stats.wall_time_ms     += batch_stats.wall_time_ms;

    // Save each output image.
    for (const auto& out : batch_outputs) {
      const std::string out_path = args.output_dir + "/" + out.filename;
      if (!SavePGM(out_path, out)) {
        std::cerr << "  Warning: could not save '" << out_path << "'\n";
        ++total_stats.images_failed;
      }
    }

    // Keep a record for the log file.
    for (auto& img : batch_inputs)   all_inputs.push_back(std::move(img));
    for (auto& img : batch_outputs)  all_outputs.push_back(std::move(img));

    batch_start = batch_end;
    ++batch_num;
  }

  // Print summary and write log.
  PrintSummary(total_stats, args.operation_str);
  WriteLog(args.log_file, args, all_inputs, all_outputs, total_stats);

  return (total_stats.images_processed > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
