// Copyright 2024 Aditya Gupta. All rights reserved.
// CUDA at Scale for the Enterprise – Independent Project
//
// pnm_utils.cpp  –  Implementation of PGM I/O and directory utilities.

#include "pnm_utils.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// LoadPGM
// ---------------------------------------------------------------------------
bool LoadPGM(const std::string& filename, GrayscaleImage* image) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "LoadPGM: cannot open '" << filename << "': "
              << std::strerror(errno) << "\n";
    return false;
  }

  // Read magic number.
  std::string magic;
  file >> magic;
  if (magic != "P5") {
    std::cerr << "LoadPGM: '" << filename
              << "' is not a binary PGM (P5) file (got '" << magic << "')\n";
    return false;
  }

  // Skip comments (lines beginning with '#').
  char ch = '\0';
  file.get(ch);
  while (ch == '#') {
    std::string comment;
    std::getline(file, comment);
    file.get(ch);
  }
  file.unget();

  // Read dimensions and maximum value.
  int width = 0, height = 0, maxval = 0;
  file >> width >> height >> maxval;
  if (width <= 0 || height <= 0) {
    std::cerr << "LoadPGM: invalid dimensions " << width << "x" << height
              << " in '" << filename << "'\n";
    return false;
  }
  if (maxval != 255) {
    std::cerr << "LoadPGM: only 8-bit PGM (maxval=255) is supported, got "
              << maxval << " in '" << filename << "'\n";
    return false;
  }

  // Consume the single whitespace byte that separates the header from data.
  file.get(ch);

  // Read raw pixel data.
  const std::size_t num_pixels = static_cast<std::size_t>(width * height);
  std::vector<uint8_t> pixels(num_pixels);
  file.read(reinterpret_cast<char*>(pixels.data()),
            static_cast<std::streamsize>(num_pixels));
  if (!file) {
    std::cerr << "LoadPGM: short read on '" << filename << "'\n";
    return false;
  }

  image->data     = std::move(pixels);
  image->width    = width;
  image->height   = height;
  image->filename = filename;
  return true;
}

// ---------------------------------------------------------------------------
// SavePGM
// ---------------------------------------------------------------------------
bool SavePGM(const std::string& filename, const GrayscaleImage& image) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "SavePGM: cannot open '" << filename << "' for writing: "
              << std::strerror(errno) << "\n";
    return false;
  }

  file << "P5\n" << image.width << " " << image.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(image.data.data()),
             static_cast<std::streamsize>(image.data.size()));
  if (!file) {
    std::cerr << "SavePGM: write error on '" << filename << "'\n";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// ListFiles
// ---------------------------------------------------------------------------
std::vector<std::string> ListFiles(const std::string& directory,
                                   const std::string& extension) {
  std::vector<std::string> files;
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) {
    std::cerr << "ListFiles: cannot open directory '" << directory << "': "
              << std::strerror(errno) << "\n";
    return files;
  }

  struct dirent* entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    const std::string name(entry->d_name);
    if (name.size() >= extension.size() &&
        name.compare(name.size() - extension.size(),
                     extension.size(), extension) == 0) {
      files.push_back(directory + "/" + name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

// ---------------------------------------------------------------------------
// EnsureDirectory
// ---------------------------------------------------------------------------
bool EnsureDirectory(const std::string& path) {
  struct stat st{};
  if (stat(path.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }
  if (mkdir(path.c_str(), 0755) != 0) {
    std::cerr << "EnsureDirectory: cannot create '" << path << "': "
              << std::strerror(errno) << "\n";
    return false;
  }
  return true;
}
