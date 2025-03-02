# Project name
PROJECT = dummy_kernel

# CUDA compiler
NVCC = nvcc

# Source files
SRCS = src/main.cu src/dummy_kernel.cu

# Include directories
INCLUDES = -I./include

# Default target
all: $(PROJECT)

# Build target
$(PROJECT): $(SRCS)
	$(NVCC) $(INCLUDES) -o $(PROJECT) $(SRCS)

# Clean target
clean:
	rm -f $(PROJECT) *.o