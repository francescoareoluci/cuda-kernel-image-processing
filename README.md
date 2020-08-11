# CUDA C++ Multithreading Kernel image processing

This repository contains a CUDA C++ application that can be used to process an image using Kernel convolution. The application can process .png images and the resulting image will be a grayscale png.

## Compile the application

In order to run the application, the CUDA toolkit must be installed. To install it, please follow this link: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Moreover, the png++ library (a libpng wrapper) must be installed. To install it using apt:

> sudo apt install libpng++-dev

A Makefile is provided in order to compile the application. In order to compile run the following command in the root directory:

> make

## Application usage

A main controller (main.cu) has been written to test the developed classes that are used to load images (image.h, images.cpp), to build a kernel (kernel.h, kernel.cpp) and to filter the images (gpu_convolution.cu, gpu_convolution.h). The main file will load image from the requested path, and will write the output image in output/ folder. The application run the kernel processing on the loaded image two times: the first time it will run a parallel processing with the specified CUDA kernel type, the second time it will run a sequential processing. Execution times for the two runs will be printed on the command line.
To launch the main application:

**Usage: ./kernel_convolution filter_type image_path cuda_mem_tye** <br>
	**filter_type**: <gaussian | sharpen | edge_detect | laplacian | gaussian_laplacian> <br>
	**image_path**: specify the image path <br>
	**(optional) cuda_mem_type**: <global | constant | shared>. Default: shared <br>

