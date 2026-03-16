# =============================================================================
# Makefile  –  Iris GPU Classifier
# CUDA at Scale for the Enterprise – Independent Project
# =============================================================================
#
# Targets
#   make            – build the binary (default: "all")
#   make run        – build and run with default settings
#   make clean      – remove build artefacts
#   make help       – print available targets
#
# Override variables on the command line, e.g.:
#   make ARCH=sm_75
#   make CUDA_PATH=/opt/cuda-12
# =============================================================================

CUDA_PATH ?= /usr/local/cuda
ARCH      ?= sm_86
JOBS      ?= 4

NVCC := $(CUDA_PATH)/bin/nvcc
CXX  := g++

# ---------------------------------------------------------------------------
# Compiler / linker flags
# ---------------------------------------------------------------------------
NVCC_FLAGS := -std=c++17 -O2 -arch=$(ARCH)                   \
              -Xcompiler "-Wall -Wextra"                       \
              --compiler-options -fPIC

CXX_FLAGS  := -std=c++17 -O2 -Wall -Wextra

INCLUDES   := -I$(CUDA_PATH)/include -Isrc

# NPP libraries used:
#   libnppc   – NPP core types
#   libnpps   – NPP signal processing (nppsSum, nppsMinMax, nppsNormalize)
LIBS := -L$(CUDA_PATH)/lib64 \
        -lnppc               \
        -lnpps               \
        -lcudart             \
        -lm

# ---------------------------------------------------------------------------
# Sources and objects
# ---------------------------------------------------------------------------
TARGET   := iris_gpu_classifier

SRCS_CU  := src/iris_processor.cu
SRCS_CPP := src/main.cpp src/csv_utils.cpp

OBJS_CU  := $(SRCS_CU:.cu=.o)
OBJS_CPP := $(SRCS_CPP:.cpp=.o)
OBJS     := $(OBJS_CU) $(OBJS_CPP)

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all run clean help

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
# One-command run
# ---------------------------------------------------------------------------
run: all
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
	@echo "  all    – build the binary (default)"
	@echo "  run    – build and run end-to-end"
	@echo "  clean  – remove build artefacts"
	@echo "  help   – print this message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  ARCH=$(ARCH)   (e.g. sm_75, sm_80, sm_86, sm_89)"
