#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "gpu_convolution.h"
#include "cuda.h"

unsigned int divUp(const unsigned int& a, const unsigned int& b)
{
	if (a % b != 0) {
		return a / b + 1;
	}
	else {
		return a / b;
	}
}

const unsigned int MAX_FILTER_SIZE = 25;
__device__ __constant__ float d_cFilterKernel[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

__global__ void filterImageGlobal(float* d_sourceImagePtr, float* d_maskPtr, float* d_outImagePtr,
						int width, int height, int paddedWidth, int paddedHeight,
						int filterWidth, int filterHeight)
{
	const int s = floor(float(filterWidth) / 2);
	const int i = blockIdx.y * blockDim.y + threadIdx.y + s;
	const int j = blockIdx.x * blockDim.x + threadIdx.x + s;

	int filterRowIndex = 0;
	int sourceImgRowIndex = 0;
	float pixelSum = 0;
	int sourceImgIndex = 0;
	int maskIndex = 0;

	// Check out of bounds thread idx
	if( j >= s && j < paddedWidth - s &&
			i >= s && i < paddedHeight - s) {

		// Apply convolution
		for (int h = -s;  h <= s; h++) {
			filterRowIndex = (h + s) * filterWidth;
	    	sourceImgRowIndex = (h + i) * paddedWidth;
	    	for (int w = -s; w <= s; w++) {
	    		sourceImgIndex = w + j + sourceImgRowIndex;
	    		maskIndex = (w + s) + filterRowIndex;
	    		pixelSum += d_sourceImagePtr[sourceImgIndex] * d_maskPtr[maskIndex];
	    	}
		}

		// Thresholding overflowing pixel's values
		if (pixelSum < 0) {
			pixelSum = 0;
		}
		else if (pixelSum > 255) {
			pixelSum = 255;
		}

		// Write pixel on the output image
		d_outImagePtr[(j - s) + (i - s) * width] = pixelSum;
		pixelSum = 0;
	}
}

__global__ void filterImageConstant(float* d_sourceImagePtr, float* d_outImagePtr,
						int width, int height, int paddedWidth, int paddedHeight,
						int filterWidth, int filterHeight)
{
	const int s = floor(float(filterWidth) / 2);
	const int i = blockIdx.y * blockDim.y + threadIdx.y + s;
	const int j = blockIdx.x * blockDim.x + threadIdx.x + s;

	int filterRowIndex = 0;
	int sourceImgRowIndex = 0;
	float pixelSum = 0;
	int sourceImgIndex = 0;
	int maskIndex = 0;

	// Check out of bounds thread idx
	if( j >= s && j < paddedWidth - s &&
			i >= s && i < paddedHeight - s) {

		// Apply convolution
		for (int h = -s;  h <= s; h++) {
			filterRowIndex = (h + s) * filterWidth;
	    	sourceImgRowIndex = (h + i) * paddedWidth;
	    	for (int w = -s; w <= s; w++) {
	    		sourceImgIndex = w + j + sourceImgRowIndex;
	    		maskIndex = (w + s) + filterRowIndex;
	    		// Here we use the kernel in constant memory
	    		pixelSum += d_sourceImagePtr[sourceImgIndex] * d_cFilterKernel[maskIndex];
	    	}
		}

		// Thresholding overflowing pixel's values
		if (pixelSum < 0) {
			pixelSum = 0;
		}
		else if (pixelSum > 255) {
			pixelSum = 255;
		}

		// Write pixel on the output image
		d_outImagePtr[(j - s) + (i - s) * width] = pixelSum;
		pixelSum = 0;
	}
}

__global__ void filterImageShared(float* d_sourceImagePtr, float* d_outImagePtr,
									int paddedWidth, int paddedHeight,
									int blockWidth, int blockHeight,
									int surroundingPixels,
									int width, int height,
									int filterWidth, int filterHeight)
{
	// Each block will share the same data, enabling a faster memory access.
	// Global memory access for each thread will be: number of tile sub blocks * threads
	// instead of 9 * threads

	// Tile shared array (dynamically sized by kernel launcher)
	extern __shared__ float s_data[];

	// Evaluate tile's size
	int tileWidth = blockWidth + 2 * surroundingPixels;
	int tileHeight = blockHeight + 2 * surroundingPixels;

	// Evaluates number of sub blocks
	int noSubBlocks = int(ceil(float(tileHeight) / float(blockDim.y)));

	// Get start and end coordinates for blocks
	int blockStartCol = blockIdx.x * blockWidth + surroundingPixels;
	int blockEndCol = blockStartCol + blockWidth;
	int blockStartRow = blockIdx.y * blockHeight + surroundingPixels;
	int blockEndRow = blockStartRow + blockHeight;

	// Get start and end coordinates for tiles
	int tileStartCol = blockStartCol - surroundingPixels;
	int tileEndCol = blockEndCol + surroundingPixels;
	int tileEndClampedCol = min(tileEndCol, paddedWidth);

	int tileStartRow = blockStartRow - surroundingPixels;
	int tileEndRow = blockEndRow + surroundingPixels;
	int tileEndClampedRow = min(tileEndRow, paddedHeight);

	// Pixel position in tile
	int tilePixelPosCol = threadIdx.x;
	// Input image pixel column position
	int iPixelPosCol = tileStartCol + tilePixelPosCol;

	int tilePixelPosRow = 0;
	int iPixelPosRow = 0;
	int iPixelPos = 0;
	int tilePixelPos = 0;
	for(int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++) {
		tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
		iPixelPosRow = tileStartRow + tilePixelPosRow;

		// Check if the pixel is in the image
		if(iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow) {
			iPixelPos = iPixelPosRow * paddedWidth + iPixelPosCol;
	      tilePixelPos = tilePixelPosRow * tileWidth + tilePixelPosCol;
	      // Load the pixel in the shared memory
	      s_data[tilePixelPos] = d_sourceImagePtr[iPixelPos];
	    }
	}

	int oPixelPosCol = 0;
	int oPixelPosRow = 0;
	int oPixelPos = 0;
	float pixelSum = 0;
	int tilePixelPosOffset = 0;
	int maskIndex = 0;

	// Wait for threads loading data in tiles
	__syncthreads();

	for(int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++) {

	    tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
	    iPixelPosRow = tileStartRow + tilePixelPosRow;

	    // Check if the pixel is in the tile and image.
	    // Pixels in the tile padding are exclude from evaluation.
	     if( iPixelPosCol >= tileStartCol + surroundingPixels &&
	    		iPixelPosCol < tileEndClampedCol - surroundingPixels &&
	    		iPixelPosRow >= tileStartRow + surroundingPixels &&
	    		iPixelPosRow < tileEndClampedRow - surroundingPixels ) {

	    	 // Evaluate pixel position for output image
	    	 oPixelPosCol = iPixelPosCol - surroundingPixels;
	    	 oPixelPosRow = iPixelPosRow - surroundingPixels;
	    	 oPixelPos = oPixelPosRow * width + oPixelPosCol;

	    	 tilePixelPos = tilePixelPosRow * tileWidth + tilePixelPosCol;

	    	 // Apply convolution
	    	 for (int h = -surroundingPixels;  h <= surroundingPixels; h++) {
	    		 for (int w = -surroundingPixels; w <= surroundingPixels; w++) {
	    			 tilePixelPosOffset = h * tileWidth + w;
	    			 maskIndex = (h + surroundingPixels) * filterWidth + (w + surroundingPixels);
	    			 pixelSum += s_data[tilePixelPos + tilePixelPosOffset] * d_cFilterKernel[maskIndex];
				}
			}

	    	// Thresholding overflowing pixel's values
			if (pixelSum < 0) {
				pixelSum = 0;
			}
			else if (pixelSum > 255) {
				pixelSum = 255;
			}

			// Write pixel on the output image
			d_outImagePtr[oPixelPos] = pixelSum;
			pixelSum = 0;
	    }
	}
}

void run(const float* sourceImage,
        float* outImage,
        const float* mask,
        int width, int height,
        int paddedWidth, int paddedHeight,
        int filterWidth, int filterHeight)
{
	const int blockWidth = 32;
	const int blockHeight = 32;

	float *d_sourceImagePtr;
	float *d_outImagePtr;
	float *d_maskPtr;

	const int sourceImgSize = sizeof(float) * paddedWidth * paddedHeight;
	const int maskSize = sizeof(float) * filterWidth * filterHeight;
	const int outImageSize = sizeof(float) * width * height;

	int copyDuration = 0;

	// Allocate device memory for images and filter
	auto t3 = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&d_sourceImagePtr, sourceImgSize);
	cudaMalloc((void**)&d_maskPtr, maskSize);
	cudaMalloc((void**)&d_outImagePtr, outImageSize);
	auto t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	// Transfer data from host to device memory
	t3 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_sourceImagePtr, sourceImage, sourceImgSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_maskPtr, mask, maskSize, cudaMemcpyHostToDevice);
	t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	// Allocates block size and grid size
	dim3 threadsPerBlock(blockWidth, blockHeight);
	dim3 blocksPerGrid(divUp(width, blockWidth), divUp(height, blockHeight));

	printf("Blocks: %d, Threads: %d\n", width / threadsPerBlock.x * height / threadsPerBlock.y, blockWidth * blockHeight);

	auto t1 = std::chrono::high_resolution_clock::now();

	filterImageGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_sourceImagePtr, d_maskPtr, d_outImagePtr,
					 width,  height,  paddedWidth,  paddedHeight,
					 filterWidth,  filterHeight);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

	// Waits for threads to finish work
	cudaDeviceSynchronize();

	auto t2 = std::chrono::high_resolution_clock::now();
	auto filterDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Filtering multi Execution time: " << filterDuration << std::endl;

	// Transfer resulting image back
	t3 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(outImage, d_outImagePtr, outImageSize, cudaMemcpyDeviceToHost);
	t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
	std::cout << "Copy Execution time: " << copyDuration << std::endl;

	// Cleanup after kernel execution
	cudaFree(d_sourceImagePtr);
	cudaFree(d_maskPtr);
	cudaFree(d_outImagePtr);
}

void runConstant(const float* sourceImage,
        float* outImage,
        const float* mask,
        int width, int height,
        int paddedWidth, int paddedHeight,
        int filterWidth, int filterHeight)
{
	const int blockWidth = 32;
	const int blockHeight = 32;

	float *d_sourceImagePtr;
	float *d_outImagePtr;

	const int sourceImgSize = sizeof(float) * paddedWidth * paddedHeight;
	const int maskSize = sizeof(float) * filterWidth * filterHeight;
	const int outImageSize = sizeof(float) * width * height;

	int copyDuration = 0;

	// Allocate device memory for images and filter
	auto t3 = std::chrono::high_resolution_clock::now();
	cudaMalloc((void**)&d_sourceImagePtr, sourceImgSize);
	cudaMalloc((void**)&d_outImagePtr, outImageSize);
	auto t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	// Transfer data from host to device memory
	t3 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_sourceImagePtr, sourceImage, sourceImgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cFilterKernel, mask, maskSize, 0, cudaMemcpyHostToDevice);
	t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	// Allocates block size and grid size
	dim3 threadsPerBlock(blockWidth, blockHeight);
	dim3 blocksPerGrid(divUp(width, blockWidth), divUp(height, blockHeight));

	printf("Blocks: %d, Threads: %d\n", width / threadsPerBlock.x * height / threadsPerBlock.y, blockWidth * blockHeight);

	auto t1 = std::chrono::high_resolution_clock::now();

	filterImageConstant<<<blocksPerGrid, threadsPerBlock>>>(d_sourceImagePtr, d_outImagePtr,
					 width,  height,  paddedWidth,  paddedHeight,
					 filterWidth,  filterHeight);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

	// Waits for threads to finish work
	cudaDeviceSynchronize();

	auto t2 = std::chrono::high_resolution_clock::now();
	auto filterDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Filtering multi Execution time: " << filterDuration << std::endl;

	// Transfer resulting image back
	t3 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(outImage, d_outImagePtr, outImageSize, cudaMemcpyDeviceToHost);
	t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
	std::cout << "Copy Execution time: " << copyDuration << std::endl;

	// Cleanup after kernel execution
	cudaFree(d_sourceImagePtr);
	cudaFree(d_outImagePtr);
}

void runShared(const float* sourceImage,
        		float* outImage,
        		const float* mask,
        		int width, int height,
        		int paddedWidth, int paddedHeight,
        		int filterWidth, int filterHeight)
{
	float *d_sourceImagePtr;
	float *d_outImagePtr;

	const int blockWidth = 32;
	const int blockHeight = 32;
	const int surroundingPixels = floor(filterWidth / 2);

	// TIles includes block size + block padding
	const int tileWidth = blockWidth + 2 * surroundingPixels;
	const int tileHeight = blockHeight + 2 * surroundingPixels;

	// Thread block height will be less than its width.
	// This way we can use bigger kernel size without
	// exceeding thread limit
	const int threadBlockHeight = 8;

	// Evaluate images and kernel size
	const int sourceImgSize = sizeof(float) * paddedWidth * paddedHeight;
	const int maskSize = sizeof(float) * filterWidth * filterHeight;
	const int outImageSize = sizeof(float) * width * height;

	dim3 threadsPerBlock(tileWidth, threadBlockHeight);
	dim3 blocksPerGrid(divUp(width, blockWidth),
											divUp(height, blockHeight));

	int noSubBlocks = int(ceil(float(tileHeight) / float(divUp(height, blockHeight))));

	printf("NoSubBlocks: %d\n", noSubBlocks);
	printf("Blocks: %d, Threads: %d\n", divUp(width, blockWidth) * divUp(height, blockHeight), tileWidth * threadBlockHeight);

	// Evaluates the shared memory size
	int sharedMemorySize = tileWidth * tileHeight * sizeof(float);

	int copyDuration = 0;

	auto t3 = std::chrono::high_resolution_clock::now();
	// Allocate device memory for images
	cudaMalloc((void**)&d_sourceImagePtr, sourceImgSize);
	cudaMalloc((void**)&d_outImagePtr, outImageSize);
	auto t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	// Transfer data from host to device memory
	t3 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_sourceImagePtr, sourceImage, sourceImgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_cFilterKernel, mask, maskSize, 0, cudaMemcpyHostToDevice);
	t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	// Launch kernel specifying the shared memory size
	filterImageShared<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(d_sourceImagePtr, d_outImagePtr,
																				paddedWidth, paddedHeight,
																				blockWidth, blockHeight,
																				surroundingPixels,
																				width, height,
																				filterWidth, filterHeight);

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Waits for threads to finish work
	cudaDeviceSynchronize();

	auto t2 = std::chrono::high_resolution_clock::now();
	auto filterDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	std::cout << "Filtering multi Execution time: " << filterDuration << std::endl;

	// Transfer resulting image back
	t3 = std::chrono::high_resolution_clock::now();
	cudaMemcpy(outImage, d_outImagePtr, outImageSize, cudaMemcpyDeviceToHost);
	t4 = std::chrono::high_resolution_clock::now();
	copyDuration += std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
	std::cout << "Copy Execution time: " << copyDuration << std::endl;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// Cleanup after kernel execution
	cudaFree(d_sourceImagePtr);
	cudaFree(d_outImagePtr);
}
