# =============================================================================
# Makefile  –  CUDA Batch Image Processor
# CUDA at Scale for the Enterprise – Independent Project
# =============================================================================
#
# Targets
#   make            – build the binary (default: "all")
#   make run        – generate test data, build, and run with default settings
#   make generate   – generate synthetic test images via Python
#   make clean      – remove build artefacts
#   make help       – print this message
#
# Variables you may override on the command line:
#   CUDA_PATH   – path to CUDA toolkit  (default: /usr/local/cuda)
#   ARCH        – GPU compute capability (default: sm_86)
#   JOBS        – parallel make jobs    (default: 4)
# =============================================================================

CUDA_PATH   ?= /usr/local/cuda
ARCH        ?= sm_86
JOBS        ?= 4

NVCC        := $(CUDA_PATH)/bin/nvcc
CXX         := g++

# ---------------------------------------------------------------------------
# Compiler / linker flags
# ---------------------------------------------------------------------------
NVCC_FLAGS  := -std=c++17 -O2 -arch=$(ARCH)                  \
               -Xcompiler "-Wall -Wextra -Wpedantic"          \
               --compiler-options -fPIC

CXX_FLAGS   := -std=c++17 -O2 -Wall -Wextra -Wpedantic

INCLUDES    := -I$(CUDA_PATH)/include -Isrc

LIBS        := -L$(CUDA_PATH)/lib64                           \
               -lnppc                                          \
               -lnppif                                         \
               -lnppist                                        \
               -lnppicc                                        \
               -lcudart                                        \
               -lm

# ---------------------------------------------------------------------------
# Sources and objects
# ---------------------------------------------------------------------------
TARGET      := cuda_image_processor

SRCS_CU     := src/image_processor.cu
SRCS_CPP    := src/main.cpp src/pnm_utils.cpp

OBJS_CU     := $(SRCS_CU:.cu=.o)
OBJS_CPP    := $(SRCS_CPP:.cpp=.o)
OBJS        := $(OBJS_CU) $(OBJS_CPP)

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all run generate clean help

all: $(TARGET)

# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)
	@echo ""
	@echo "Build successful → ./$(TARGET)"
	@echo "Run  ./run.sh  or  ./$(TARGET) --help  to get started."

# ---------------------------------------------------------------------------
# Compile CUDA translation units
# ---------------------------------------------------------------------------
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# ---------------------------------------------------------------------------
# Compile C++ translation units
# ---------------------------------------------------------------------------
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c -o $@ $<

# ---------------------------------------------------------------------------
# Generate synthetic test data
# ---------------------------------------------------------------------------
generate:
	@echo "Generating synthetic test images …"
	python3 scripts/generate_test_data.py --output data/input --count 200

# ---------------------------------------------------------------------------
# One-command end-to-end run
# ---------------------------------------------------------------------------
run: generate all
	@bash run.sh

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Cleaned build artefacts."

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Usage: make [TARGET] [VARIABLE=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  all       – build the binary (default)"
	@echo "  run       – generate data, build, and run"
	@echo "  generate  – generate synthetic PGM test images"
	@echo "  clean     – remove build artefacts"
	@echo "  help      – print this message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  ARCH=$(ARCH)   (e.g. sm_75, sm_80, sm_86, sm_89)"
	@echo "  JOBS=$(JOBS)"
