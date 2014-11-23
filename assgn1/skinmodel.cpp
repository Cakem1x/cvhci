#include "skinmodel.h"
#include <cmath>
#include <iostream>

#define voodoo ((SkinProbMap*) pimpl)

class SkinProbMap {
  public:
    SkinProbMap() {
      int sizes[] = {256, 256, 256};
      map = new cv::Mat(3, sizes, CV_64FC1);
      total_skin = 0;
    }

    ~SkinProbMap(){}

    double add_training_skin_pixel(cv::Vec3b bgr) {
      total_skin++;
      map->at<double>(bgr[0], bgr[1], bgr[2]) += 1;

      return map->at<double>(bgr[0], bgr[1], bgr[2]);
    }

    float test_skin_pixel(cv::Vec3b bgr) {
       double div = map->at<double>(bgr[0], bgr[1], bgr[2]) / total_skin;
       return div;
    }

  private:
    cv::Mat* map;
    double total_skin;
};

/// Constructor
SkinModel::SkinModel()
{
  SkinProbMap* sm = new SkinProbMap();
  pimpl = (SkinModelPimpl*) sm;
}

/// Destructor
SkinModel::~SkinModel() 
{
  delete voodoo;
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
  for (int row = 0; row < img.rows; ++row) {
    for (int col = 0; col < img.cols; ++col) {
      if (mask(row,col) == 255) {
        double total_skin = voodoo->add_training_skin_pixel(img(row,col));
        //std::cout << "Adding skin pixel at [" << (int)img(row,col)[0] << "," << (int)img(row,col)[1] << "," << (int)img(row,col)[2]  << "] , there were " << total_skin << "skin pixels with the same CrCb value." << std::endl;
      }
    }
  }
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

    for (int row = 0; row < img.rows; ++row) {
      for (int col = 0; col < img.cols; ++col) {
        double skin_prob = voodoo->test_skin_pixel(img(row,col));
        //std::cout << "Probability for skin at [" << (int)img(row,col)[0] << "," << (int)img(row,col)[1] << "," << (int)img(row,col)[2] << "] is " << skin_prob << std::endl;
        if ((int)(skin_prob*700000) < 255) {
          skin(row,col) = (int)(skin_prob*700000);
        } else {
          skin(row,col) = 255;
        }
			}
    }

    return skin;
}

