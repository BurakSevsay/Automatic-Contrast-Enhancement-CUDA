# Automatic-Contrast-Enhancement-CUDA
CUDA implementation of automatic contrast enhancement for 8-bit grayscale images. Results are compared with CPU implementation.

The algorithm uses all parallel reduction techniques in [Nvidia, Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)

There are three kernels for the GPU process. <br />
- Kernel 1: Finding minimum and maximum pixel values in the source image.
- Kernel 2: Subtracting minimum pixel value from all pixels in the source image.
- Kernel 3: Scaling the pixels of the source image by a factor of 255 / (maxValue-minValue).

Some of the applied techniques:
- To prevent cores from waiting in the idle state, unrolling and using templates are implemented.
- To decrease the global memory access, shared memory is used effectively.
- To increase parallelization and decrease kernel instances, minimum and maximum value searching is applied on the same kernel grid.

## Benchmarking
Benchmarking is done by repeating each method 1000 times and averaging the
total elapsed time. For comparison, the time elapsed on CPU, the time elapsed on GPU, and
time elapsed on GPU without data transfer between GPU and CPU are recorded.

|   | GPU time wo loading data (ms) | GPU time including loading data (ms) | CPU time (ms) |
| ------------- | ------------- | ------------- | ------------- |
| Sample 1: 640x426  | 0.101239 | 0.292942 | 1.920919 |
| Sample 2: 1280x843  | 0.260451 | 0.874949 | 7.859998 |
| Sample 3: 1920x1280  | 0.447119 | 1.541617 | 17.384371 |
| Sample 4: 5184x3456  | 2.737699 | 10.010386 | 126.918327 |

## Environment 
A desktop having Intel® Xeon(R) W-2223 CPU @ 3.60GHz × 8, and NVIDIA GeForce GTX 1060 6GB GPU is utilized for this assignment. A 64-bit Ubuntu-Linux operating system runs on the desktop. Nvidia CUDA Compiler (nvcc) is used to compile and run the code on the desktop with Nvidia-cuda-toolkit. Version information is as following; NVIDIA-SMI 510.47.03, Driver Version: 510.47.03, CUDA Version: 11.6.
