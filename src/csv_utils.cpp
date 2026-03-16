// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// csv_utils.cpp  –  CSV parsing / writing for the Iris dataset.

#include "csv_utils.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Trim leading and trailing whitespace from a string.
static std::string Trim(const std::string& s) {
  const std::size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  const std::size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// Map a class-name string to an integer label.  Returns -1 if unknown.
static int ClassNameToLabel(const std::string& name) {
  for (int i = 0; i < kNumClasses; ++i) {
    if (name == kClassNames[i]) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// LoadIrisCSV
// ---------------------------------------------------------------------------
bool LoadIrisCSV(const std::string& filename,
                 std::vector<IrisSample>* samples) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "LoadIrisCSV: cannot open '" << filename << "': "
              << std::strerror(errno) << "\n";
    return false;
  }

  std::string line;
  int line_no = 0;
  while (std::getline(file, line)) {
    ++line_no;
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) continue;  // skip blank lines

    std::stringstream ss(trimmed);
    std::string token;
    IrisSample sample{};

    bool parse_ok = true;
    for (int f = 0; f < kNumFeatures; ++f) {
      if (!std::getline(ss, token, ',')) {
        std::cerr << "LoadIrisCSV: line " << line_no
                  << ": expected " << kNumFeatures << " features\n";
        parse_ok = false;
        break;
      }
      try {
        sample.features[f] = std::stof(Trim(token));
      } catch (...) {
        std::cerr << "LoadIrisCSV: line " << line_no
                  << ": cannot parse feature " << f
                  << " value '" << token << "'\n";
        parse_ok = false;
        break;
      }
    }
    if (!parse_ok) continue;

    if (!std::getline(ss, token, ',')) {
      std::cerr << "LoadIrisCSV: line " << line_no << ": missing class label\n";
      continue;
    }
    sample.class_name = Trim(token);
    sample.label      = ClassNameToLabel(sample.class_name);
    if (sample.label < 0) {
      std::cerr << "LoadIrisCSV: line " << line_no
                << ": unknown class '" << sample.class_name << "'\n";
      continue;
    }

    samples->push_back(sample);
  }

  if (samples->empty()) {
    std::cerr << "LoadIrisCSV: no valid samples loaded from '" << filename << "'\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// SavePredictions
// ---------------------------------------------------------------------------
bool SavePredictions(const std::string&            filename,
                     const std::vector<IrisSample>& samples,
                     const std::vector<int>&        predictions) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "SavePredictions: cannot open '" << filename << "' for writing\n";
    return false;
  }

  file << "sample_id,sepal_length,sepal_width,petal_length,petal_width,"
          "actual_class,predicted_class,correct\n";

  const std::size_t n = std::min(samples.size(), predictions.size());
  for (std::size_t i = 0; i < n; ++i) {
    const auto& s = samples[i];
    const int   pred = predictions[i];
    file << (i + 1) << ","
         << s.features[0] << ","
         << s.features[1] << ","
         << s.features[2] << ","
         << s.features[3] << ","
         << s.class_name  << ","
         << kClassNames[pred] << ","
         << (pred == s.label ? 1 : 0) << "\n";
  }
  return file.good();
}

// ---------------------------------------------------------------------------
// SaveFeatureStats
// ---------------------------------------------------------------------------
bool SaveFeatureStats(const std::string& filename,
                      const float means[kNumFeatures],
                      const float stds[kNumFeatures],
                      const float mins[kNumFeatures],
                      const float maxs[kNumFeatures]) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "SaveFeatureStats: cannot open '" << filename
              << "' for writing\n";
    return false;
  }

  file << "feature,mean,std_dev,min,max\n";
  for (int f = 0; f < kNumFeatures; ++f) {
    file << kFeatureNames[f] << ","
         << means[f] << ","
         << stds[f]  << ","
         << mins[f]  << ","
         << maxs[f]  << "\n";
  }
  return file.good();
}

// ---------------------------------------------------------------------------
// EnsureParentDirectory
// ---------------------------------------------------------------------------
bool EnsureParentDirectory(const std::string& filepath) {
  const std::size_t sep = filepath.find_last_of("/\\");
  if (sep == std::string::npos) return true;  // no directory component
  const std::string dir = filepath.substr(0, sep);
  struct stat st{};
  if (stat(dir.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
  if (mkdir(dir.c_str(), 0755) != 0) {
    std::cerr << "EnsureParentDirectory: cannot create '" << dir << "': "
              << std::strerror(errno) << "\n";
    return false;
  }
  return true;
}
