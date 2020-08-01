#include <iostream>
#include <chrono>
#include "image.h"
#include "kernel.h"

#define GAUSSIAN_FILTER_COMMAND             "gaussian"
#define SHARPENING_FILTER_COMMAND         "sharpen"
#define EDGE_DETECTION_FILTER_COMMAND  "edge_detect"
#define LAPLACIAN_FILTER_COMMAND            "laplacian"
#define GAUSSIAN_LAPLACIAN_COMMAND      "gaussian_laplacian"

#define CUDA_GLOBAL		"global"
#define CUDA_CONSTANT	"constant"
#define CUDA_SHARED		"shared"

enum class FilterType
{
    GAUSSIAN_FILTER,
    SHARPEN_FILTER,
    EDGE_DETECTION,
    LAPLACIAN_FILTER,
    GAUSSIAN_LAPLACIAN_FILTER
};

int main(int argc, char **argv)
{
	std::cout << "===== Multithread kernel convolution =====" << std::endl;

	 // Check command line parameters
	 if (argc < 3) {
		 std::cerr << "Usage: " << argv[0] << " filter_type image_path cuda_mem_tye" << std::endl;
		 std::cerr << "filter_type: <gaussian | sharpen | edge_detect | alt_edge_detect>" << std::endl;
	    std::cerr << "image_path: specify the image path" << std::endl;
	    std::cerr << "(optional) cuda_mem_type: <global | constant | shared>. Default: shared" << std::endl;
	     return 1;
	}

	 FilterType filterType;
	 std::string cmdFilter = std::string(argv[1]);
	  if (cmdFilter == GAUSSIAN_FILTER_COMMAND) {
		  filterType = FilterType::GAUSSIAN_FILTER;
	 }
	 else if (cmdFilter == SHARPENING_FILTER_COMMAND) {
		  filterType = FilterType::SHARPEN_FILTER;
	 }
	 else if (cmdFilter == EDGE_DETECTION_FILTER_COMMAND) {
		 filterType = FilterType::EDGE_DETECTION;
	 }
	 else if (cmdFilter == LAPLACIAN_FILTER_COMMAND) {
		 filterType = FilterType::LAPLACIAN_FILTER;
	 }
	 else if (cmdFilter == GAUSSIAN_LAPLACIAN_COMMAND) {
		 filterType = FilterType::GAUSSIAN_LAPLACIAN_FILTER;
	 }
	 else {
		 std::cerr << "Invalid filter type " << cmdFilter << std::endl;
	    std::cerr << "filter_type: <gaussian | sharpen | edge_detect | laplacian | gaussian_laplacian >" << std::endl;
	     return 1;
	}

   Kernel filter = Kernel();
	switch (filterType)
	{
	 	 case FilterType::GAUSSIAN_FILTER:
	 		 filter.setGaussianFilter(7, 7, 1);
	     break;

	     case FilterType::SHARPEN_FILTER:
	    	 filter.setSharpenFilter();
	     break;

	     case FilterType::EDGE_DETECTION:
	    	 filter.setEdgeDetectionFilter();
	     break;

	     case FilterType::LAPLACIAN_FILTER:
	    	 filter.setLaplacianFilter();
	     break;

	     case FilterType::GAUSSIAN_LAPLACIAN_FILTER:
	    	 filter.setGaussianLaplacianFilter();
	     break;

	     default:
	    	 std::cerr << "Unable to find requested filter, switching to gaussian..." << std::endl;
	       filter.setGaussianFilter(5, 5, 2);
	      break;
	}
	filter.printKernel();

	CudaMemType cudaType = CudaMemType::SHARED;
	if (argc == 4) {
		std::string cudaMemCmd = std::string(argv[3]);
		if (cudaMemCmd == CUDA_GLOBAL)
			cudaType = CudaMemType::GLOBAL;
		else if(cudaMemCmd == CUDA_CONSTANT)
			cudaType = CudaMemType::CONSTANT;
		else if(cudaMemCmd == CUDA_SHARED)
			cudaType = CudaMemType::SHARED;
	}

	Image img;
	bool loadResult = img.loadImage(argv[2]);
	if (!loadResult) {
		std::cerr << "Unable to load image " << argv[2] << std::endl;
		return 1;
	}

	Image newMtImg;
	Image newNpImg;

	// The first call to the CUDA device will take a lot of time,
	// better do it here
	cudaFree(0);

	// Executing multithread filtering for each image
	auto t1 = std::chrono::high_resolution_clock::now();
	bool cudaResult = img.multithreadFiltering(newMtImg, filter, cudaType);
	auto t2 = std::chrono::high_resolution_clock::now();

	auto t3 = std::chrono::high_resolution_clock::now();
	bool sequentialResult = img.applyFilter(newNpImg, filter);
	auto t4 = std::chrono::high_resolution_clock::now();

	// Evaluating execution times and save results
	if (cudaResult) {
		auto multithreadDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
		std::cout << "Multithread Execution time: " << multithreadDuration << " μs" << std::endl;
		newMtImg.saveImage("output/1_mt.png");
	}

	if (sequentialResult) {
		auto singleDuration = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
		std::cout << "Sequential Execution time: " << singleDuration << " μs" << std::endl;
		newNpImg.saveImage("output/1_np.png");
	}
}
