/*
    MMI713 - Assignment
    Burak SEVSAY
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>
#include <iostream>
#include <ctime>

#define NUM_CHANNELS 1

//Scale pixels with constant
__global__ void scale_with_constant(uint8_t *output, int const_val, int scale_factor, int width, int height)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

    // Excess threads should not do calculation
	if (ix >= width)
		return;
	if (iy >= height)
		return;

	int idx = iy * width + ix;
	uint8_t value = output[idx];

	output[idx] = static_cast<uint8_t>((int(value)*const_val) >> (scale_factor-1)); // scaling 
}

// Extract minimum pixel value from all pixels
__global__ void substract_min(uint8_t * input, uint8_t * output, uint8_t min_val, int width, int height)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

    // Excess threads should not do calculation
	if (ix >= width)
		return;
	if (iy >= height)
		return;

	int idx = iy * width + ix;

	output[idx] = input[idx] - min_val; //substraction
}

template <unsigned int block_size, bool _final_>
__global__ void min_max_calcualtion(uint8_t *input_array, uint8_t * output_array, int input_size, int input_number_for_last_block)
{
    extern __shared__ uint8_t sdata[];
    uint8_t* sdata_min = sdata;
    uint8_t* sdata_max = (uint8_t*)&sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

    uint8_t val1; 
    uint8_t val2; 

    /*
    - Each thread access corresponding indeces of input: i and i+blockDim.x
    - Threads compare these values, and write the minimum one to shared memory sdata_min
    and wrtie the maximum one to shared memory sdata_max for further comparison
    - Then, there is a search in shared memories to find min and max value. Parallel reduction (v7 in lecture notes) is used. 
    - Threads in the last block may have not enough corresponding input. Following logic solve this problem.
    For the last block: 
        - If thread can access two input indeces, do the above (comparison) procedure.
        - If thread can access only one input index, write the input to the shared memory without comparison
        - If thread can't access any input indeces, write the possible max. number to the shared memory 

    Since input size is bigger than max. block size, block-wise comparison will be done. 
    So, the kernel will have an ouput of min. and max. values from each block. 
    This output will be the next input of another kernel instance of this kernel.
    If the _final_ variable is set true, it means input size is less than max. block size.
    */

   if (_final_) { // For this case; image is scanned by previous kernel(s) and grid size is 1.
       if (tid < input_size/2) 
       {
        sdata_min[tid] = input_array[i]; // half of the input belongs min. array from previous kernel computation
        sdata_max[tid] = input_array[i+input_size/2]; // other half of the input belongs max. array  from previous kernel computation
       }  else { // fill the gap between block_size-input_size
           sdata_min[tid] = 255;
           sdata_max[tid] = 0;
       }
   } else { // Due to input size, this case can not find the exact min/max values
    bool condition_1 = (input_number_for_last_block > blockDim.x); // There are some inputs that can be compared in the last block
    bool condition_2 = (blockIdx.x == (gridDim.x - 1)); // Last block    

        if (condition_2 && condition_1) { // There are some inputs that can be compared in the last block
            if (tid < (input_number_for_last_block - blockDim.x)) // Do comparison for possible input indices
            {
                    val1 = input_array[i];
                    val2 = input_array[i + blockDim.x];
                    sdata_min[tid] = (val1 < val2) ? val1 : val2; //Write smaller value to sdata_min
                    sdata_max[tid] = (val1 > val2) ? val1 : val2; //Write bigger value to sdata_min         
            } else { // Not enough input to compare, so write the input directly
                sdata_min[tid] = input_array[i];
                sdata_max[tid] = input_array[i];
            }
        } else if (condition_2) { //Not enough input to compare for the last block
            if (tid < input_number_for_last_block) // For threads that can access corresponding input
            {
                sdata_min[tid] = input_array[i];
                sdata_max[tid] = input_array[i];
            } else { // For threads that cant access corresponding input
                sdata_min[tid] = 255;
                sdata_max[tid] = 0;
            }
        } else {
            val1 = input_array[i];
            val2 = input_array[i + blockDim.x];
            sdata_min[tid] = (val1 < val2) ? val1 : val2; //Write smaller value to sdata_min
            sdata_max[tid] = (val1 > val2) ? val1 : val2; //Write bigger value to sdata_min 
        }
    }

	__syncthreads();  // Synchronize threads to load data to shared memory

    // Find min&max values for each block. Do unrolled comparison for efficiency
	if (block_size >= 512)
	{
		if (tid < 256)
			sdata_min[tid] = (sdata_min[tid + 256] < sdata_min[tid]) ? sdata_min[tid + 256] : sdata_min[tid];
		else if(tid < 512)
			sdata_max[tid-256] = (sdata_max[tid - 256] > sdata_max[tid]) ? sdata_max[tid - 256] : sdata_max[tid];
	}
	__syncthreads();
	if (block_size >= 256)
	{
		if (tid < 128)
			sdata_min[tid] = (sdata_min[tid + 128] < sdata_min[tid]) ? sdata_min[tid + 128] : sdata_min[tid];
		else if(tid < 256)
			sdata_max[tid - 128] = (sdata_max[tid - 128] > sdata_max[tid]) ? sdata_max[tid - 128] : sdata_max[tid];
	}
	__syncthreads();
	if (block_size >= 128)
	{
		if (tid < 64)
			sdata_min[tid] = (sdata_min[tid + 64] < sdata_min[tid]) ? sdata_min[tid + 64] : sdata_min[tid];
		else if(tid < 128)
			sdata_max[tid - 64] = (sdata_max[tid - 64] > sdata_max[tid]) ? sdata_max[tid - 64] : sdata_max[tid];
	}	
	__syncthreads();
	if (block_size >= 64)
	{
		if (tid < 32)
			sdata_min[tid] = (sdata_min[tid + 32] < sdata_min[tid]) ? sdata_min[tid + 32] : sdata_min[tid];
		else if(tid < 64)
			sdata_max[tid - 32] = (sdata_max[tid - 32] > sdata_max[tid]) ? sdata_max[tid - 32] : sdata_max[tid];
	}
	__syncthreads();
	if (block_size >= 32)
	{
		if (tid < 16)
			sdata_min[tid] = (sdata_min[tid + 16] < sdata_min[tid]) ? sdata_min[tid + 16] : sdata_min[tid];
		else if(tid < 32)
			sdata_max[tid - 16] = (sdata_max[tid - 16] > sdata_max[tid]) ? sdata_max[tid - 16] : sdata_max[tid];
	}
    __syncthreads();
	if (block_size >= 16)
	{
		if (tid < 8)
			sdata_min[tid] = (sdata_min[tid + 8] < sdata_min[tid]) ? sdata_min[tid + 8] : sdata_min[tid];
		else if(tid < 16)
			sdata_max[tid - 8] = (sdata_max[tid - 8] > sdata_max[tid]) ? sdata_max[tid - 8] : sdata_max[tid];
	}
    __syncthreads();
	if (block_size >= 8)
	{
		if (tid < 4)
			sdata_min[tid] = (sdata_min[tid + 4] < sdata_min[tid]) ? sdata_min[tid + 4] : sdata_min[tid];
		else if(tid < 8)
			sdata_max[tid - 4] = (sdata_max[tid - 4] > sdata_max[tid]) ? sdata_max[tid - 4] : sdata_max[tid];
	}
    __syncthreads();
	if (block_size >= 4)
	{
		if (tid < 2)
			sdata_min[tid] = (sdata_min[tid + 2] < sdata_min[tid]) ? sdata_min[tid + 2] : sdata_min[tid];
		else if(tid < 4)
			sdata_max[tid - 2] = (sdata_max[tid - 2] > sdata_max[tid]) ? sdata_max[tid - 2] : sdata_max[tid];
	}
    __syncthreads();
	if (block_size >= 2)
	{
		if (tid < 1)
			sdata_min[tid] = (sdata_min[tid + 1] < sdata_min[tid]) ? sdata_min[tid + 1] : sdata_min[tid];
		else if(tid < 2)
			sdata_max[tid - 1] = (sdata_max[tid - 1] > sdata_max[tid]) ? sdata_max[tid - 1] : sdata_max[tid];
	}
    __syncthreads();

    // the first data in the shared memory contain the smallest value, write it to the output
    if (tid == 0)
    {
        if (_final_) { // Write final min-max values to the output
            output_array[0] = sdata_min[0];
            output_array[1] = sdata_max[0];
        } else { // Write block-wise min-max results to ouput
            output_array[blockIdx.x] = sdata_min[tid];
            output_array[blockIdx.x + gridDim.x] = sdata_max[tid];         
        }
    }
}

// CPU baseline for finding minimum and maximum pixel values in an image
void min_max_of_img_host(uint8_t *img, uint8_t *min, uint8_t *max, int image_size)
{
    int max_tmp = 0;
    int min_tmp = 255;
    for (int n=0; n < image_size; n++){
        max_tmp = (img[n] > max_tmp) ? img[n] : max_tmp;
        min_tmp = (img[n] < min_tmp) ? img[n] : min_tmp;
    }

    *max = max_tmp;
    *min = min_tmp;
}

// CPU baseline for subtracting a value from all pixels in an image
void sub_host(uint8_t* img, uint8_t sub_value, int image_size) {
    for (int n=0; n < image_size; n++){
        img[n] -= sub_value;
    }
}


// CPU baseline for scaling pixel values in an image avoiding finite precision
// integer arithmetic given "power" and "constant" values.
void scale_host(uint8_t* img, float constant, int image_size) {
    for (int n=0; n < image_size; n++){
        img[n] = img[n] * constant; //note the implicit type conversion
    }
}

void RunOnCPU(uint8_t* image, int image_size) {
    uint8_t min_host, max_host;
      
    std::clock_t c_start = std::clock(); // Start timing

    // Calculate the value of minimum and maximum pixels (to be replaced with GPU kernel)
    min_max_of_img_host(image, &min_host, &max_host, image_size);

    float scale_constant = 255.0f / (max_host - min_host);

    // Subtract minimum pixel value from all pixels (to be replaced with GPU kernel)
    sub_host(image, min_host, image_size);

    // Scale pixel values between 0 and 255 (to be replaced with GPU kernel)
    scale_host(image, scale_constant, image_size);

    std::clock_t c_end = std::clock(); // Finish timing

    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";
}

/*
It is for the final kernel of finding min-max.
Grid size is the output length of previous kernel. It means input of the final kernel. 
Decide block size of the final kernel according to input size.
*/
int CalculateNewBlockSize(int grid_size) {  
    int new_block_size = 1024;
    if (grid_size <= 4)
        new_block_size = 4;
    else if (grid_size <= 8)
        new_block_size = 8;
    else if (grid_size <= 16)
        new_block_size = 16;
    else if (grid_size <= 32)
        new_block_size = 32;
    else if (grid_size <= 64)
        new_block_size = 64;
    else if (grid_size <= 128)
        new_block_size = 128;
    else if (grid_size <= 256)
        new_block_size = 256;
    else if (grid_size <= 512)
        new_block_size = 512;
    else
        new_block_size = 1024;

    return new_block_size;
}

// calcualte grid size according to input size and block size for finding min-max problem
int CalculateGridSize(int input_size, int block_size) {
    int excess_block_number = (input_size % (block_size*block_size*2) == 0) ? 0 : 1; 
    int grid_size = (input_size) / (block_size*2) + excess_block_number; 
    return grid_size;
}

void RunOnGPU(uint8_t* image, uint8_t* out_image, int image_size, int nWidth, int nHeight) {
    // Device parameter declarations.
    uint8_t *input_d, *output_d;
    uint8_t *min_max_result_device, *min_max_result_host;
    uint8_t min_val_host, max_val_host;

    // Variable definitions
    const int block_size = 1024;
    int mem_image_size = sizeof(uint8_t) * image_size;

    // START : ----------------- FINDING MIN-MAX -----------------

    /*
    ----------------- FINDING MIN-MAX -----------------
    Parallel Reduction v7 (in the lecture notes) is used.
    Firstly, a thread with index i will access both ith and (i+blocksize)th element in global memory. 
    Compare two value, write minimum one to shared_mem_min, write maximum one to shared_mem_max
    Then, parallel reduction technique is used on shared memories. 
    Unrolled comparison and templates are used for efficiency.
    The minimum value in a block will be in the first element of the shared_mem_min. 
    The maximum value in a block will be in the first element of the shared_mem_max. 
    */

    int grid_size = CalculateGridSize(image_size, block_size);
    int mem_grid_size = sizeof(uint8_t) * grid_size; // Memory size of the grid size
    int smem_size = sizeof(uint8_t) * block_size *2; // Shared memory size
    int input_number_for_last_block = image_size - (2 * block_size * (grid_size - 1)); // last block may not access input that twice the block size because of input size

    // Start timer for benchmarking
    cudaEvent_t start_0, stop_0;
    cudaEvent_t start_1, stop_1;
    cudaEventCreate(&start_0); cudaEventCreate(&stop_0);
    cudaEventCreate(&start_1); cudaEventCreate(&stop_1);

    // Start timer_0
    cudaEventRecord(start_0, 0);

    // Allocate device memory
    cudaMalloc((void **)&input_d, mem_image_size); // Allocation for input on device
    cudaMalloc((void **)&output_d, mem_image_size); // Allocation for image output on device

    cudaMalloc((void **)&min_max_result_device, mem_grid_size * 2); // Allocation for min-max output of the kernel on device. Each block will have result, so we will have number of result consistent with number of block (grid size). Multiplication of 2 is for one value of min and one value of max

    // Allocate host memory 
    uint8_t *input_h = (uint8_t*)malloc(mem_image_size); // Allocation for input on host
    min_max_result_host = (uint8_t*)malloc(mem_grid_size * 2); // Allocation for output of the kernel on host.  Multiplication of 2 is for one value of min and one value of max
    
    cudaMemcpy(input_d, image, mem_image_size, cudaMemcpyHostToDevice); // copy image array to gpu input 

    // Calculate min max values of the image. Save it to min_val_host, max_val_host
    dim3 grid, block;
    grid.x = grid_size;
    block.x = block_size;
    std::cout << "Process the image on GPU." << std::endl;

    // Since same memory region in the GPU is used by different kernels, they should be in the same stream
    cudaStream_t stream1;
	cudaStreamCreate(&stream1);

    // Start timer_1
    cudaEventRecord(start_1, 0);

    //GPU calculation. Block-wise results will be acquires
    min_max_calcualtion<block_size, false> <<<grid, block, smem_size, stream1>>>(input_d, min_max_result_device, image_size, input_number_for_last_block);
    cudaDeviceSynchronize();
    /*
    Kernel above created an output array with size of grid_size. The final kernel will have grid_size = 1
    If grid size of the above kernel is bigger than (twice) maximum block size, do finding min-max on output array 
    */
    if (grid_size/block_size > 1) { 
        int new_input_size = grid_size;
        int grid_size_ = CalculateGridSize(new_input_size, block_size);
        grid.x = grid_size_;
        input_number_for_last_block = new_input_size - (2 * block_size * (grid_size_ - 1)); // last block may not access input that twice the block size because of input size
        min_max_calcualtion<block_size, false> <<<grid, block, smem_size, stream1>>>(min_max_result_device, min_max_result_device, new_input_size, input_number_for_last_block);
        cudaDeviceSynchronize();
    }

    int new_block_size; 
    new_block_size = CalculateNewBlockSize(grid_size * 2); // calculate block size of the final kernel 
    grid.x = 1;
    block.x = new_block_size;

    // Final searching. Grid size is one, so there will be one block that finds absolute min-max from an array with searched/processed values
    switch (new_block_size)
	{
	case 1024:
		min_max_calcualtion<1024, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 512:
		min_max_calcualtion<512, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 256:
		min_max_calcualtion<256, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 128:
		min_max_calcualtion<128, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 64:
		min_max_calcualtion<64, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 32:
		min_max_calcualtion<32, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
        break;
	case 16:
		min_max_calcualtion<16, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 8:
		min_max_calcualtion<8, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	case 4:
		min_max_calcualtion<4, true> << <grid, block, new_block_size*2, stream1 >> > (min_max_result_device, min_max_result_device, grid_size*2, grid_size*2);
		break;
	}
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(min_max_result_host, min_max_result_device, mem_grid_size * 2, cudaMemcpyDeviceToHost);

    min_val_host = min_max_result_host[0];
    max_val_host = min_max_result_host[1];

    // END :   ----------------- FINDING MIN-MAX -----------------
    // START : -----------------  SUBSTRACT MIN  -----------------

    // Subtract Min value from all pixels
	block.x = 16;
	block.y = 16;
	grid.x = nWidth/block.x + (nWidth%block.x == 0 ? 0 : 1);
	grid.y = nHeight/block.y + (nHeight%block.y == 0 ? 0 : 1);
	substract_min << <grid, block, 0, stream1 >> > (input_d,output_d,min_val_host,nWidth,nHeight);
    cudaDeviceSynchronize();

    // END :   ----------------- SUBSTRACT MIN -----------------
    // START : -----------------    SCALING    -----------------

    //Integer-Result Scaling
	int nScaleFactor = 0;
	int nPower = 1;
	while (nPower * 255.0f / (max_val_host - min_val_host) < 255.0f)
	{
		nScaleFactor++;
		nPower *= 2;
	}
	nPower /= 2;
	uint8_t nConstant = static_cast<uint8_t>(255.0f / (max_val_host - min_val_host) * nPower);

	scale_with_constant << <grid, block, 0, stream1 >> > (output_d,nConstant, nScaleFactor,nWidth,nHeight);

    // END :   -----------------    SCALING    -----------------

    // Stop recording timer_1
    float et; //elapsed time
	cudaEventRecord(stop_1, 0);
	cudaEventSynchronize(stop_1);
	cudaEventElapsedTime(&et, start_1, stop_1);
	printf("GPU Process Elapsed Time = %f ms\n", et);
    
    cudaMemcpy(out_image, output_d, mem_image_size, cudaMemcpyDeviceToHost);

    // Stop recording timer_0
    cudaEventRecord(stop_0, 0);
	cudaEventSynchronize(stop_0);
    cudaEventElapsedTime(&et, start_0, stop_0);

    // Destroy/deallocate times
    cudaEventDestroy(start_0); cudaEventDestroy(stop_0);
	cudaEventDestroy(start_1); cudaEventDestroy(stop_1);
	printf("GPU Total Elapsed Time = %f ms\n", et);

    // Free the allocated memory on both GPU and CPU
    free(input_h); 
    free(min_max_result_host);
    cudaFree(input_d);
    cudaFree(min_max_result_device);
}

int main() {
    int width;  // image width
    int height; // image height
    int bpp;    // bytes per pixel if the image was RGB (not used)

    // Load a grayscale bmp image to an unsigned integer array with its height and weight. (uint8_t is an alias for "unsigned char")

    // Choose one of the inputs
    // uint8_t *image = stbi_load("./samples/640x426.bmp", &width, &height, &bpp, NUM_CHANNELS);
    uint8_t *image = stbi_load("./samples/1280x843.bmp", &width, &height, &bpp, NUM_CHANNELS);
    // uint8_t *image = stbi_load("./samples/1920x1280.bmp", &width, &height, &bpp, NUM_CHANNELS);
    // uint8_t *image = stbi_load("./samples/5184x3456.bmp", &width, &height, &bpp, NUM_CHANNELS);
    
    uint8_t *enhanced_image; // Gpu will write the processed image to that. Image variable will be proccessed by cpu later.

    int image_size = width * height; // size of flatten image (to array)
    enhanced_image = new uint8_t[image_size]; // Allocation

    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp / 3); // Image is grayscale, so bpp / 3;
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);  

    printf("Run on GPU \n");
    RunOnGPU(image, enhanced_image, image_size, width, height); // image variable won't be modified for further CPU process
    stbi_write_bmp("./out_img_gpu.bmp", width, height, 1, enhanced_image); // Write image array into a bmp file
    
    printf("Run on CPU \n");
    RunOnCPU(image, image_size);
    stbi_write_bmp("./out_img_cpu.bmp", width, height, 1, image); // Write image array into a bmp file

    printf("Processing on both GPU and CPU is finished, well done! \n");

    // Deallocate memory
    stbi_image_free(image);
    free(enhanced_image);
}
