// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// pnm_utils.h  –  Portable GrayMap (PGM) image I/O utilities.
//
// Supports binary PGM (P5) with 8-bit depth.  All images loaded through
// this header are stored as plain host-side byte arrays in row-major order.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Holds a single grayscale 8-bit image in host memory.
struct GrayscaleImage {
  std::vector<uint8_t> data;  // Row-major pixel data (width * height bytes)
  int width  = 0;
  int height = 0;
  std::string filename;       // Original file path (informational)
};

// ---------------------------------------------------------------------------
// I/O functions
// ---------------------------------------------------------------------------

// Load a binary PGM (P5) file into |image|.
// Returns true on success; |image| is left unchanged on failure.
bool LoadPGM(const std::string& filename, GrayscaleImage* image);

// Save |image| as a binary PGM (P5) file at |filename|.
// Returns true on success.
bool SavePGM(const std::string& filename, const GrayscaleImage& image);

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------

// Return all files inside |directory| whose names end with |extension|
// (e.g. ".pgm"), sorted lexicographically.
std::vector<std::string> ListFiles(const std::string& directory,
                                   const std::string& extension);

// Create |path| and any missing parent directories.
// Returns true if the directory already existed or was just created.
bool EnsureDirectory(const std::string& path);
