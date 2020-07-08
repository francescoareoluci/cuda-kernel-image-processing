#include "kernel.h"
#include <iostream>
#include <cmath>


#define SHARPEN_FILTER_MAX      5
#define SHARPEN_FILTER_MIN     -1
#define LAPLACIAN_FILTER_MAX    4
#define LAPLACIAN_FILTER_MIN    -1
#define LINE_DETECTOR_MAX       8
#define LINE_DETECTOR_MIN       -1


Kernel::Kernel()
{
	this->m_filterMatrix = std::vector<float>(0);
	this->m_filterWidth = 0;
	this->m_filterHeight = 0;
}

void Kernel::printKernel() const
{
    int height = m_filterHeight;
    int width = m_filterWidth;

    if (height == 0 || width == 0)
    {
        std::cout << "Kernel has not been set up" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Kernel ===" << std::endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << (float)m_filterMatrix[j + i * width] << " ";
        }
        std::cout << "" << std::endl;
    }
    std::cout << "==============" << std::endl;
    std::cout << std::endl;
}

bool Kernel::setGaussianFilter(const int height, const int width, const float stdDev)
{
    std::cout << "Building gaussian filter..." << std::endl;

    if (height != width || height % 2 == 0 || width % 2 == 0) {
        std::cerr << "Height and Width values are not valid" << std::endl;
        std::cerr << "Width and height should have the same values" << std::endl;

        return false;
    }

    if (stdDev <= 0) {
        std::cerr << "Standard deviation value is not valid" << std::endl;
        std::cerr << "Standard deviation value must be positive" << std::endl;

        return false;
    }

    std::vector<float> kernel(width * height);
    float sum = 0.0;

    int middleHeight = int(height / 2);
    int middleWidth = int(width / 2);

    for (int i = -middleHeight; i <= middleHeight; i++) {
        for (int j = -middleWidth; j <= middleWidth; j++) {
            float cellValue = exp(- (i * i + j * j) / (2 * stdDev * stdDev)) / (2 * M_PI * stdDev * stdDev);
            kernel[(j + middleWidth) + ( i + middleHeight) * width] = cellValue;
            sum += cellValue;
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            kernel[j + i * width] /= sum;
        }
    }

    m_filterMatrix = kernel;
    m_filterWidth = width;
    m_filterHeight = height;

    return true;
}

bool Kernel::setSharpenFilter()
{
    std::vector<float> kernel(3 * 3);
    this->buildKernelCommon(kernel, SHARPEN_FILTER_MAX, SHARPEN_FILTER_MIN, 3, 3);

    kernel[0] = 0.0;
    kernel[2] = 0.0;
    kernel[kernel.size() - 3] = 0.0;
    kernel[kernel.size() - 1] = 0.0;

    m_filterMatrix = kernel;
    m_filterWidth = 3;
    m_filterHeight = 3;

    return true;
}

bool Kernel::setEdgeDetectionFilter()
{
    std::vector<float> kernel(3 * 3);
    this->buildKernelCommon(kernel, LINE_DETECTOR_MAX, LINE_DETECTOR_MIN, 3, 3);

    m_filterMatrix = kernel;
    m_filterWidth = 3;
    m_filterHeight = 3;

    return true;
}

bool Kernel::setLaplacianFilter()
{
    std::vector<float> kernel(3 * 3);
    this->buildKernelCommon(kernel, LAPLACIAN_FILTER_MAX, LAPLACIAN_FILTER_MIN, 3, 3);

    kernel[0] = 0.0;
    kernel[2] = 0.0;
    kernel[kernel.size() - 3] = 0.0;
    kernel[kernel.size() - 1] = 0.0;

    m_filterMatrix = kernel;
    m_filterWidth = 3;
    m_filterHeight = 3;

    return true;
}

bool Kernel::setGaussianLaplacianFilter()
{
    std::vector<float> kernel(5 * 5);

    int max = 16;
    int min = -2;
    int med = -1;

    kernel[12] = max;
    kernel[2] = med;
    kernel[6] = med;
    kernel[8] = med;
    kernel[10] = med;
    kernel[14] = med;
    kernel[16] = med;
    kernel[18] = med;
    kernel[22] = med;
    kernel[7] = min;
    kernel[11] = min;
    kernel[13] = min;
    kernel[17] = min;

    m_filterMatrix = kernel;
    m_filterWidth = 5;
    m_filterHeight = 5;

    return true;
}

bool Kernel::buildKernelCommon(std::vector<float> &kernel, int max, int min, int height, int width)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if ((i == int(height / 2)) && (j == int(width / 2))) {
                kernel[j + i * width] = max;
            }
            else {
                kernel[j + i * width] = min;
            }
        }
    }

    return true;
}

int Kernel::getKernelWidth() const
{
    return m_filterWidth;
}

int Kernel::getKernelHeight() const
{
    return m_filterHeight;
}

std::vector<float> Kernel::getKernel() const
{
    return this->m_filterMatrix;
}
