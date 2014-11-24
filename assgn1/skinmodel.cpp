#include "skinmodel.h"
#include <cmath>
#include <iostream>
#include <assert.h>

#define voodoo ((SkinProbMap*) pimpl)

class SkinProbMap {
  public:
    SkinProbMap() {
      int sizes[] = {256, 256, 256};
      map = new cv::Mat(3, sizes, CV_16UC2);
      total_skin = 0;
      total_non_skin = 0;
    }

    ~SkinProbMap(){}

    /// Adds a training point bgr with the information whether or not this point is a skin point
    /// @returns the number of pixels with which this model has already been trained.
    int add_training_skin_pixel(cv::Vec3b bgr, bool is_skin) {
      cv::Vec<unsigned short, 2>* color_point = &(map->at <cv::Vec<unsigned short, 2> > (bgr[0], bgr[1], bgr[2]));
      if (is_skin) {
        total_skin++;
        (*color_point)[0] += 1;
      } else {
        total_non_skin++;
      }
      (*color_point)[1] += 1;

      //std::cout << "Total skin + total_non_skin is " << total_skin + total_non_skin << std::endl;

      return total_skin + total_non_skin;
    }

    double add_training_non_skin_pixel() {
      return total_non_skin++;
    }

    /// Returns the probability for Skin (S) for a given color (X), so P(S|X).
    double test_skin_pixel(cv::Vec3b bgr) {
      double P_S_X, P_X_S, P_X, P_S; // P(S|X), P(X|S), P(X), P(S)
      assert(total_skin > 0);
      assert(total_non_skin > 0);
      cv::Vec<unsigned short, 2> color_point = map->at<cv::Vec<unsigned short, 2> >(bgr[0], bgr[1], bgr[2]);

      P_X_S = (double)color_point[0] / (double)total_skin;
      P_X = (double)color_point[1] / (double)(total_non_skin + total_skin);
      P_S = (double)total_skin / (double)(total_non_skin + total_skin);
      P_S_X = (P_X_S/P_X)*P_S;

      //std::cout << "P(S|X) = P(X|S) / P(X) * P(S) = " << P_S_X << " = " << P_X_S << " / " << P_X << " * " << P_S << std::endl;
      return P_S_X; // = P(S|X)
    }

  private:
    /// First channel counts how many skin pixels there were with this color, second channel counts how man pixels there were with this color in total.
    cv::Mat* map;
    int total_skin;
    int total_non_skin;
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
      // train for pixel at img(row,col)
      voodoo->add_training_skin_pixel(img(row,col), mask(row,col) == 255);
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
        double P_X_S = voodoo->test_skin_pixel(img(row,col)); //P(X|S)
        // Scale the probability to somewhere between 0 and 255:
        if ((int)(P_X_S*100) < 255) {
          skin(row,col) = (int)(P_X_S*100);
        } else {
          skin(row,col) = 255;
        }
			}
    }

    return skin;
}

