# dummy-kernel

First of all, the questions is: **"What is CUDA ?"**. CUDA is the tool that opens the door to GPU programming.

Modern GPUs thrive on parallelism, letting thousands of lightweight threads handle tasks in tandem. CUDA exposes these capabilities through an extension of C/C++ that includes special keywords and memory models for GPU programming.

## But why do we program on GPU ?

The answer is simple, GPUs are designed to handle parallel tasks. They have thousands of cores, while CPUs have only a few. This makes GPUs ideal for tasks that can be parallelized, such as image processing, physics simulations, and machine learning.

For example:

- **CPU Cores** are optimized for fast, complex operations on fewer threads at once.
- **GPU Cores** are designed for handling thousands of smaller, simpler threads, excelling at data-parallel operations.

So when you have huge arrays/marices/tensors to process, or repetitive tasks to perform, you can use GPU to speed up the process.

## What are some core CUDA concepts ?

The core concepts of CUDA programming are:

### a) Execution Model

- **Threads**: The smallest unit of parallel work. Each thread executes the same kernel (function), but on different data (Single Instruction, Multiple Thread approach).
- **Blocks**: Group of threads that can share fast on-chip memory (called **shared memory**).
- **Grid**: Collection of all blocks needed to solve your problem.

### b) Memory Hierarchy

- **Global Memory**: Large but relatively slower; accessible by all threads.
- **Shared Memory**: A small, fast region local to each block.
- **Local/Private Memory**: Registers or thread-local storage.
- **Constant & Texture Memory**: Specialized for read-only data or caching.

### c) Kernel Functions

- Marked with `__global__`, these are the functions you launch on the GPU.
- Kernel launches use the triple-angle-bracket syntax: `kernelName<<<blocks, threads>>>(...)`

## Setting up the Environment

Setting up a clean, propper environment is crucial for a smooth CUDA development experience. Here's why:

1. **Scalability**: As soon as you add multiple kernels or external libraries (cuBLAS, cuFFT, etc.), a solid environment helps maintain order.
2. **Maintainability**: Isolating host vs. device code, using consistent naming conventions, and adopting a build system (e.g., Make or CMake) all reduce technical debt.
3. **Performance Insights**: Quick access to profiling tools (Nsight Systems, Nsight Compute) and debug builds fosters iterative optimization.
4. **Cross-Platform Consistency**: With a reliable setup, you can develop on Windows, Linux, or macOS while minimizing environment-related quirks.

To get started, you need to have some basic tools and libraries installed on your system. Here's a quick checklist:

1. **Install NVIDIA Driver**

   - Must match or exceed your CUDA Toolkit version; mismatches lead to compilation or runtime errors.
   - Check with `nvidia-smi` (Linux) or in Windows' Device Manager → Display Adapters → NVIDIA driver version.

2. **Install CUDA Toolkit**

   - [Download here](https://developer.nvidia.com/cuda-downloads) for your OS (Windows, Linux, macOS).
   - Includes `nvcc`, header files, libraries (e.g., `libcudart`, `libcublas`).
   - Make sure it's on your `PATH` (Windows) or in your `LD_LIBRARY_PATH` (Linux/macOS) if needed.

3. **Verify the Install**

   - Check `nvcc --version` in your terminal or Command Prompt.
   - If you see a valid version (e.g., `Cuda compilation tools, release 11.x`), you’re set.
   - Run `nvidia-smi` (Linux) or open the NVIDIA Control Panel (Windows) to see your GPU model and compute capability.
   - This helps when using advanced CUDA features or specifying compilation flags like `-arch=sm_75`.

4. **Host Compiler (e.g., gcc, clang, MSVC)**

   - `nvcc` invokes a host compiler for CPU portions of your CUDA code.
   - Keep your host compiler updated, and ensure it's compatible with your CUDA version (check [NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#supported-compilers)).

5. **IDE or Editor**
   - Many developers use **Visual Studio** (Windows) or **Nsight Eclipse Edition** (Linux).
   - Others prefer **VS Code**, **CLion**, or a text editor (Vim, Emacs) with a custom build workflow.

## Organizing Your Project

A clear folder structure prevents confusion once you start splitting logic into multiple files:

```
my-cuda-project/
├── src/
│   ├── main.cu          # Host code & kernel invocations
│   ├── kernels.cu       # Device kernels (can be multiple .cu files)
│   └── utils.cu         # Additional device or host utilities
├── include/
│   ├── kernels.h        # Declarations for kernels
│   └── utils.h
├── Makefile (or CMakeLists.txt)
├── docs/                # Optional: documentation, design notes
└── README.md
```

**Tips**:

- Keep separate `.h` files for device/host function prototypes.
- If you plan to use advanced features like **separate compilation** or **device linking**, structure accordingly (e.g., `-dc` and `-dlink` flags in Makefiles).

## Compiler & Build System Essentials

### Basic `nvcc` Usage

- **`-o <output>`**: Set output file name (e.g., `-o my_app`).
- **`-arch=sm_XX`**: Compile for a specific GPU architecture, like `sm_75` (Turing) or `sm_86` (Ampere).
- **`-G`**: Enable debug info for device code (disables certain optimizations).
- **`-lineinfo`**: Include source line info for better profiling and debugging.
- **`-Xcompiler "<flags>"`**: Pass additional flags to the host compiler (e.g., `-Wall`, `-O2`).

### Build Systems

1. **Makefiles**
   - Quick, widely used. A simple example can compile all `.cu` files in `src/`.
2. **CMake**
   - More robust for cross-platform builds.
   - Add lines like `enable_language(CUDA)` and `project(MyProject LANGUAGES CXX CUDA)` in your `CMakeLists.txt`.
3. **Visual Studio or Nsight Eclipse**
   - Create a CUDA project using the provided wizards; IDE auto-manages compilation and linking.

### Separate Compilation & Device Linking

For very large projects with multiple CUDA files, you might use:

- **`-dc`**: Compile device code but don't link.
- **`-dlink`**: Perform device linking at a later stage to combine multiple compiled `.o` or `.obj` files into a single executable.

```bash
nvcc -dc kernels.cu -o kernels.o
nvcc -dc utils.cu -o utils.o
nvcc -dlink kernels.o utils.o -o dlink.o
nvcc main.cu kernels.o utils.o dlink.o -o final_app
```

### Debugging Approaches

1. **cuda-gdb**

   - GDB-based debugger for CUDA. Set device breakpoints, inspect thread-local variables, and step through kernel instructions.
   - Usage: `cuda-gdb ./my_app`

2. **Nsight Eclipse (Linux) / Nsight Visual Studio (Windows)**

   - Integrates source-level debugging, breakpoints, variable inspection, and GPU kernel stepping within an IDE.

3. **`printf` in Kernels**

   - Quick for small-scale debugging. But watch out for performance overhead if you're printing a lot.

4. **`cuda-memcheck`**
   - Checks for out-of-bounds access, misaligned memory usage, and other GPU memory errors.
   - Usage: `cuda-memcheck ./my_app`

### Profiling & Performance Tools

1. **Nsight Systems**

   - A timeline-based profiler. It shows how CPU functions and GPU kernels overlap, helps identify concurrency or synchronization bottlenecks.

2. **Nsight Compute**

   - Offers deep kernel-level metrics: occupancy, instruction throughput, warp divergence, memory transactions.
   - Vital for performance tuning.

3. **CLI Tools (Legacy)**
   - `nvprof` and `nvvp` (Visual Profiler) are older, now mostly replaced by Nsight. Some legacy workflows might still rely on them.


## Practical Example

In this example, we have created a simple CUDA kernel that prints a message from each block and thread. The project structure is as follows:

```
dummy-kernel
├── include
│   └── dummy_kernel.h
├── src
│   ├── dummy_kernel.cu
│   └── main.cu
├── Makefile
└── README.md
```

To check the code, you can open each of the files in the `src` and `include` directories. The `Makefile` contains the necessary commands to compile the project. You can run the following command to compile the project:

```bash
make
```

After compiling, you can run the executable with the following command:

```bash
./dummy_kernel
```

You should see output similar to:

```
Hello from CPU!
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
Hello from block 0, thread 3
Hello from block 1, thread 0
Hello from block 1, thread 1
Hello from block 1, thread 2
Hello from block 1, thread 3
```

## Conclusion

You now have the **tools and knowledge** to structure a multi-file CUDA project, compile it with the right flags, and debug or profile your kernels. Investing time to set up a **clean, scalable environment** early on will pay off as you write more complex kernels, integrate libraries, and strive for maximum performance.

**Key Points**:

- Keep your project organized with separate files for kernels, headers, and host logic.
- Use Makefiles or CMake for consistent, repeatable builds.
- Familiarize yourself with cuda-gdb and Nsight tools to tackle debugging and performance tuning head-on.

## References & Additional Information

1. **[CUDA C Programming Guide – Chapters 1, 2 & 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)**  
   Introduction and basic concepts of CUDA, including threads, blocks, and the compilation process.

2. **[CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)**  
   Step-by-step instructions for installing the toolkit and verifying your setup.

3. **[GPU Gems – Chapter 1](https://developer.nvidia.com/gpugems/gpugems)**  
   Provides a solid overview of GPU architecture and parallel computing fundamentals.

4. **[NVIDIA Developer Blog: Introduction to CUDA](https://developer.nvidia.com/blog/tag/cuda)**  
   Articles that dive deeper into the “why” of GPU computing and feature beginner-friendly examples.

5. **[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)**  
   System-wide profiling and concurrency visualization.

6. **[Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)**  
   Kernel-level performance metrics and optimization insights.

7. **[CMake for CUDA](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)**  
   Official docs on enabling CUDA in CMake-based workflows.

8. **[cuda-memcheck Documentation](https://docs.nvidia.com/cuda/cuda-memcheck/index.html)**  
   Detecting GPU memory errors.
