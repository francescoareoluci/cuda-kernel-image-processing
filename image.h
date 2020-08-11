#ifndef IMAGE_H_
#define IMAGE_H_

#include <vector>
#include <thread>
#include "kernel.h"

enum class CudaMemType
{
	GLOBAL,
	CONSTANT,
	SHARED
};

class Image
{
    public:
        Image();

        /*
         *  @brief: Dtor
         */
        ~Image() {
            m_image.clear();
            std::vector<float>().swap(m_image);
        }

        /*
         * @brief: get loaded image width
         */
        int getImageWidth() const;

        /*
         * @brief: get loaded image height
         */
        int getImageHeight() const;

        /*
         * @brief: get number of channels of the image
         */
        int getImageChannels() const;

        /*
         * @brief: set the image given another linearized vector
         *
         * @params: source: the matrix to be set as state
         * @return: true is successfull, false otherwise
         */
        bool setImage(const std::vector<float>& source, int width, int height);

        /*
         * @brief: return the matrix state
         *
         * @return: the matrix state
         */
        std::vector<float> getImage() const;

        /*
         * @brief: load an image from filename path
         *
         * @params: filename: the path of the image to be loaded
         * @return: true is successfull, false otherwise
         */
        bool loadImage(const char *filename);

        /*
         * @brief: save an image in filename path
         *
         * @params: filename: the path where to save the image
         * @return: true is successfull, false otherwise
         */
        bool saveImage(const char *filename) const;

        /*o
         * @brief: apply a kernel to the image and pass
         *         result in resultingImage object
         *
         * @params[out]: resultingImage: the image object where the matrix will be saved
         * @params[in]: kernel: kernel to be applied to the image
         * @return: true if successful, false otherwise
         */
        bool applyFilter(Image& resultingImage, const Kernel& kernel) const;

        /*
         * @brief: apply a kernel to the image and save it's state.
         *          This method can be used to iterate a convolution
         *          on the same image more than 1 times.
         *
         * @params[in]: kernel: kernel to be applied to the image
         * @return: true if successful, false otherwise
         */
        bool applyFilter(const Kernel& kernel);

        /*
         * @brief: apply a CUDA multithread convolution to the image 
         *          and pass result in resultingImage object
         *
         * @params[out]: resultingImage: the image object where the matrix will be saved
         * @params[in]: kernel: kernel to be applied to the image
         * @params[in]: cudaType: specify the type of CUDA kernel to be used
         * @return: true if successful, false otherwise
         */
        bool multithreadFiltering(Image& resultingImage, const Kernel& kernel, const CudaMemType cudaType);

    private:
        /*
         * @brief: A common method to apply the kernel to the image
         */
        std::vector<float> applyFilterCommon(const Kernel& kernel) const;

        /*
         * @brief: return a border-replicated padded matrix using matrix state
         *          and requested padding
         */
        std::vector<float> buildReplicatePaddedImage(const int paddingHeight,
                                                    const int paddingWidth) const;

        /*
         * @brief: return a zero padded matrix using matrix state
         *          and requested padding
         */
        std::vector<float> buildZeroPaddingImage(const int paddingHeight,
                                                const int paddingWidth) const;

        std::vector<float> m_image;	    ///< Linearized matrix containing the image pixels' values
        int m_imageWidth;               ///< Matrix width
        int m_imageHeight;              ///< Matrix height
};

#endif /* IMAGE_H_ */
