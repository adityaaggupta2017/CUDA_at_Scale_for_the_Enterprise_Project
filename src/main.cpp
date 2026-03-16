// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// main.cpp  –  CLI entry-point for the GPU-accelerated Iris classifier.
//
// Usage:
//   iris_gpu_classifier [OPTIONS]
//
// Run with --help for full usage information.

#include <getopt.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "csv_utils.h"
#include "iris_processor.cuh"

static constexpr const char* kVersion        = "1.0.0";
static constexpr const char* kDefaultInput   = "iris/iris.data";
static constexpr const char* kDefaultPredOut = "results/predictions.csv";
static constexpr const char* kDefaultStatOut = "results/feature_stats.csv";
static constexpr const char* kDefaultLog     = "results/processing.log";
static constexpr int          kDefaultK       = 5;

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void PrintUsage(const char* name) {
  std::cout
      << "Iris GPU Classifier  v" << kVersion << "\n"
         "GPU-accelerated k-NN classification of the Fisher Iris dataset\n"
         "using custom CUDA kernels and CUDA NPP signal-processing functions.\n\n"
         "Usage:\n"
         "  " << name << " [OPTIONS]\n\n"
         "Options:\n"
         "  -i, --input <file>      Iris CSV file  (default: " << kDefaultInput << ")\n"
         "  -o, --predictions <f>   Output predictions CSV  (default: " << kDefaultPredOut << ")\n"
         "  -s, --stats <file>      Output feature-stats CSV  (default: " << kDefaultStatOut << ")\n"
         "  -l, --log <file>        Log file  (default: " << kDefaultLog << ")\n"
         "  -k, --k-neighbors <n>   Neighbours for k-NN  (default: " << kDefaultK << ")\n"
         "  -n, --no-normalize      Skip z-score feature normalisation\n"
         "  -v, --verbose           Print per-sample classification result\n"
         "  -h, --help              Show this message and exit\n\n"
         "Examples:\n"
         "  " << name << "                        # run with defaults\n"
         "  " << name << " -k 3 -v               # k=3, verbose output\n"
         "  " << name << " -k 7 --no-normalize   # k=7, raw features\n"
         "  " << name << " -i custom.csv -k 5    # custom dataset\n";
}

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

struct Args {
  std::string input        = kDefaultInput;
  std::string pred_output  = kDefaultPredOut;
  std::string stats_output = kDefaultStatOut;
  std::string log_file     = kDefaultLog;
  int         k_neighbors  = kDefaultK;
  bool        normalize    = true;
  bool        verbose      = false;
};

static bool ParseArgs(int argc, char* argv[], Args* args) {
  static const option kLongOpts[] = {
      {"input",        required_argument, nullptr, 'i'},
      {"predictions",  required_argument, nullptr, 'o'},
      {"stats",        required_argument, nullptr, 's'},
      {"log",          required_argument, nullptr, 'l'},
      {"k-neighbors",  required_argument, nullptr, 'k'},
      {"no-normalize", no_argument,       nullptr, 'n'},
      {"verbose",      no_argument,       nullptr, 'v'},
      {"help",         no_argument,       nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  int opt = 0;
  while ((opt = getopt_long(argc, argv, "i:o:s:l:k:nvh",
                            kLongOpts, nullptr)) != -1) {
    switch (opt) {
      case 'i': args->input        = optarg; break;
      case 'o': args->pred_output  = optarg; break;
      case 's': args->stats_output = optarg; break;
      case 'l': args->log_file     = optarg; break;
      case 'k': args->k_neighbors  = std::stoi(optarg); break;
      case 'n': args->normalize    = false;  break;
      case 'v': args->verbose      = true;   break;
      case 'h': PrintUsage(argv[0]); std::exit(EXIT_SUCCESS);
      default:  return false;
    }
  }
  if (args->k_neighbors < 1) {
    std::cerr << "Error: k must be ≥ 1\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

static void PrintFeatureStats(const FeatureStats& stats) {
  std::cout << "\n--- GPU-computed feature statistics ---\n";
  std::cout << std::left << std::setw(16) << "Feature"
            << std::right
            << std::setw(10) << "Mean"
            << std::setw(10) << "Std Dev"
            << std::setw(10) << "Min"
            << std::setw(10) << "Max"
            << "\n";
  std::cout << std::string(56, '-') << "\n";
  for (int f = 0; f < kNumFeatures; ++f) {
    std::cout << std::left  << std::setw(16) << kFeatureNames[f]
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(10) << stats.mean[f]
              << std::setw(10) << stats.std_dev[f]
              << std::setw(10) << stats.min_val[f]
              << std::setw(10) << stats.max_val[f]
              << "\n";
  }
}

static void PrintConfusionMatrix(const std::vector<IrisSample>& samples,
                                  const std::vector<int>&        predictions) {
  int cm[kNumClasses][kNumClasses] = {};
  const std::size_t n = std::min(samples.size(), predictions.size());
  for (std::size_t i = 0; i < n; ++i) {
    ++cm[samples[i].label][predictions[i]];
  }

  std::cout << "\n--- Confusion matrix ---\n";
  std::cout << std::string(52, ' ') << "Predicted\n";
  std::cout << std::string(20, ' ');
  for (int c = 0; c < kNumClasses; ++c) {
    std::cout << std::setw(17) << kClassNames[c];
  }
  std::cout << "\n";
  for (int actual = 0; actual < kNumClasses; ++actual) {
    std::cout << std::left << std::setw(20) << kClassNames[actual];
    for (int pred = 0; pred < kNumClasses; ++pred) {
      std::cout << std::right << std::setw(17) << cm[actual][pred];
    }
    std::cout << "\n";
  }
}

static void PrintSummary(const PipelineResult& res, const Args& args) {
  std::cout << "\n========================================\n";
  std::cout << "  Iris GPU Classifier – Summary\n";
  std::cout << "========================================\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  k-neighbors   : " << args.k_neighbors << "\n";
  std::cout << "  Normalise     : " << (args.normalize ? "yes (z-score)" : "no") << "\n";
  std::cout << "  Samples       : " << res.predictions.size() << "\n";
  std::cout << "  Correct       : " << res.correct << "\n";
  std::cout << "  Accuracy      : " << res.accuracy << " %\n";
  std::cout << "  GPU time      : " << res.gpu_time_ms  << " ms\n";
  std::cout << "  Wall-clock    : " << res.wall_time_ms << " ms\n";
  std::cout << "========================================\n";
}

static void WriteLog(const std::string& log_path,
                     const Args& args,
                     const PipelineResult& res) {
  EnsureParentDirectory(log_path);
  std::ofstream log(log_path);
  if (!log.is_open()) {
    std::cerr << "Warning: could not write log to '" << log_path << "'\n";
    return;
  }
  log << "=== Iris GPU Classifier  v" << kVersion << " ===\n\n";
  log << "Input dataset   : " << args.input       << "\n";
  log << "k-neighbors     : " << args.k_neighbors << "\n";
  log << "Normalise       : " << (args.normalize ? "yes" : "no") << "\n\n";
  log << "--- Feature statistics ---\n";
  log << std::left << std::setw(16) << "Feature"
      << std::right
      << std::setw(10) << "Mean"
      << std::setw(10) << "StdDev"
      << std::setw(10) << "Min"
      << std::setw(10) << "Max"
      << "\n";
  for (int f = 0; f < kNumFeatures; ++f) {
    log << std::left  << std::setw(16) << kFeatureNames[f]
        << std::right << std::fixed << std::setprecision(4)
        << std::setw(10) << res.stats.mean[f]
        << std::setw(10) << res.stats.std_dev[f]
        << std::setw(10) << res.stats.min_val[f]
        << std::setw(10) << res.stats.max_val[f]
        << "\n";
  }
  log << "\n--- Results ---\n";
  log << "Samples    : " << res.predictions.size() << "\n";
  log << "Correct    : " << res.correct    << "\n";
  log << "Accuracy   : " << std::fixed << std::setprecision(2)
      << res.accuracy << " %\n";
  log << "GPU time   : " << res.gpu_time_ms  << " ms\n";
  log << "Wall-clock : " << res.wall_time_ms << " ms\n";
  std::cout << "Log written to " << log_path << "\n";
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

  std::cout << "Iris GPU Classifier  v" << kVersion << "\n";
  std::cout << "Input: " << args.input
            << "  |  k=" << args.k_neighbors
            << "  |  normalize=" << (args.normalize ? "yes" : "no") << "\n\n";

  // Load dataset.
  std::vector<IrisSample> samples;
  if (!LoadIrisCSV(args.input, &samples)) {
    std::cerr << "Failed to load dataset from '" << args.input << "'\n";
    return EXIT_FAILURE;
  }
  std::cout << "Loaded " << samples.size() << " samples from '"
            << args.input << "'\n";

  // Class distribution.
  int class_count[kNumClasses] = {};
  for (const auto& s : samples) {
    if (s.label >= 0 && s.label < kNumClasses) ++class_count[s.label];
  }
  for (int c = 0; c < kNumClasses; ++c) {
    std::cout << "  " << kClassNames[c] << ": " << class_count[c] << " samples\n";
  }
  std::cout << "\n";

  // Configure and run GPU pipeline.
  ClassifierConfig config;
  config.k_neighbors = args.k_neighbors;
  config.normalize   = args.normalize;
  config.verbose     = args.verbose;

  const PipelineResult result = RunGPUPipeline(samples, config);

  // Print results.
  PrintFeatureStats(result.stats);
  PrintConfusionMatrix(samples, result.predictions);
  PrintSummary(result, args);

  // Save output files.
  EnsureParentDirectory(args.pred_output);
  EnsureParentDirectory(args.stats_output);
  if (SavePredictions(args.pred_output, samples, result.predictions)) {
    std::cout << "Predictions saved to " << args.pred_output << "\n";
  }
  if (SaveFeatureStats(args.stats_output,
                       result.stats.mean,    result.stats.std_dev,
                       result.stats.min_val, result.stats.max_val)) {
    std::cout << "Feature stats saved to " << args.stats_output << "\n";
  }
  WriteLog(args.log_file, args, result);

  return EXIT_SUCCESS;
}
