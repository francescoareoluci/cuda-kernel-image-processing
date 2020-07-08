#include <iostream>
#include <chrono>
#include "image.h"
#include "kernel.h"

int main(int argc, char **argv)
{
	printf("===== Multithread kernel convolution =====\n");

	// The first call to the CUDA device will take a lot of time,
	// better do it here
	cudaFree(0);

	Kernel kernel;
	//kernel.setSharpenFilter();
	//kernel.setGaussianFilter(25, 25, 1);
	//kernel.setEdgeDetectionFilter();
	kernel.setGaussianLaplacianFilter();

	Image img;
	img.loadImage("images/1.png");

	Image newMtImg;
	Image newNpImg;

	// Executing multithread filtering for each image
	auto t1 = std::chrono::high_resolution_clock::now();
	img.multithreadFiltering(newMtImg, kernel);
	auto t2 = std::chrono::high_resolution_clock::now();

	auto t3 = std::chrono::high_resolution_clock::now();
	img.applyFilter(newNpImg, kernel);
	auto t4 = std::chrono::high_resolution_clock::now();

	// Evaluating execution times
	auto multithreadDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	auto singleDuration = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

	std::cout << "Multithread Execution time: " << multithreadDuration << std::endl;
	std::cout << "Single thread Execution time: " << singleDuration << std::endl;

	newMtImg.saveImage("output/1_mt.png");
	newNpImg.saveImage("output/1_np.png");
}
