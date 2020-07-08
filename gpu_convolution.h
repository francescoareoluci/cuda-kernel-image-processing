/*
 * gpu_convolution.h
 *
 *  Created on: Jul 4, 2020
 *      Author: francesco
 */

#ifndef GPU_CONVOLUTION_H_
#define GPU_CONVOLUTION_H_

/*
 * @brief: This function will launch a CUDA kernel to calculate image
 * 					convolution. The launched kernel will use global memory for
 * 					source image and kernel matrix
 */
void run(const float* sourceImage,
        float* outImage,
        const float* mask,
        int width, int height,
        int paddedWidth, int paddedHeight,
        int filterWidth, int filterHeight);

/*
 * @brief: This function will launch a CUDA kernel to calculate image
 * 					convolution. The launched kernel will use global memory for
 * 					source image and constant memory for kernel matrix
 */
void runConstant(const float* sourceImage,
        float* outImage,
        const float* mask,
        int width, int height,
        int paddedWidth, int paddedHeight,
        int filterWidth, int filterHeight);

/*
 * @brief: This function will launch a CUDA kernel to calculate image
 * 					convolution. The launched kernel will use shared memory to
 * 					load tiles of the source image and constant memory for kernel matrix
 */
void runShared(const float* sourceImage,
        float* outImage,
        const float* mask,
        int width, int height,
        int paddedWidth, int paddedHeight,
        int filterWidth, int filterHeight);


#endif /* GPU_CONVOLUTION_H_ */
