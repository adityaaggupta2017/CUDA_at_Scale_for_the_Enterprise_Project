// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// csv_utils.h  –  CSV I/O utilities for the Fisher Iris dataset.
//
// The Iris dataset (Fisher, 1936) contains 150 samples with four real-valued
// features and a class label:
//   sepal length (cm), sepal width (cm), petal length (cm), petal width (cm),
//   class  { Iris-setosa = 0, Iris-versicolor = 1, Iris-virginica = 2 }

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

// Number of measured features per sample.
static constexpr int kNumFeatures = 4;

// Number of class labels.
static constexpr int kNumClasses = 3;

// Human-readable feature names (column order in the CSV).
static const char* const kFeatureNames[kNumFeatures] = {
    "sepal_length", "sepal_width", "petal_length", "petal_width",
};

// Human-readable class names.
static const char* const kClassNames[kNumClasses] = {
    "Iris-setosa", "Iris-versicolor", "Iris-virginica",
};

// A single sample from the Iris dataset.
struct IrisSample {
  float features[kNumFeatures];  // sepal_l, sepal_w, petal_l, petal_w
  int   label;                   // 0 / 1 / 2
  std::string class_name;        // original string from CSV
};

// ---------------------------------------------------------------------------
// I/O functions
// ---------------------------------------------------------------------------

// Load the Iris CSV file at |filename| into |samples|.
// Lines that are empty or contain only whitespace are silently skipped.
// Returns true on success; false (with a message to stderr) on any error.
bool LoadIrisCSV(const std::string& filename,
                 std::vector<IrisSample>* samples);

// Write per-sample classification results to |filename|.
// Columns: sample_id, sepal_length, sepal_width, petal_length, petal_width,
//          actual_class, predicted_class, correct(0/1)
bool SavePredictions(const std::string&            filename,
                     const std::vector<IrisSample>& samples,
                     const std::vector<int>&        predictions);

// Write per-feature statistics to |filename|.
// Columns: feature, mean, std_dev, min, max
bool SaveFeatureStats(const std::string& filename,
                      const float means[kNumFeatures],
                      const float stds[kNumFeatures],
                      const float mins[kNumFeatures],
                      const float maxs[kNumFeatures]);

// Create the directory component of |filepath| if it does not exist.
bool EnsureParentDirectory(const std::string& filepath);
