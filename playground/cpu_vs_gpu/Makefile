CC = gcc
CFLAGS = -I./include
NVCC = nvcc
NVCCFLAGS = -I./include

# Build both CPU and GPU programs
all: cpu gpu

# Build the CPU program
cpu: main.o vector_add_cpu.o
	$(CC) -o cpu_program main.o vector_add_cpu.o

# Build the GPU program
gpu: main_gpu.o vector_add_gpu.o
	$(NVCC) -o gpu_program main_gpu.o vector_add_gpu.o

# Compile the CPU main file
main.o: src/main.c
	$(CC) $(CFLAGS) -c src/main.c -o main.o

# Compile the CPU vector addition file
vector_add_cpu.o: src/vector_add_cpu.c
	$(CC) $(CFLAGS) -c src/vector_add_cpu.c -o vector_add_cpu.o

# Compile the GPU main file
main_gpu.o: src/main.cu
	$(NVCC) $(NVCCFLAGS) -c src/main.cu -o main_gpu.o

# Compile the GPU vector addition file
vector_add_gpu.o: src/vector_add_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c src/vector_add_gpu.cu -o vector_add_gpu.o

# Clean up the build artifacts
clean:
	rm -f *.o cpu_program gpu_program