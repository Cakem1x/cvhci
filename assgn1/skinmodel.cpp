#include "skinmodel.h"
#include <cmath>
#include <iostream>

using namespace std;

class RegionGrower {
  public:
    RegionGrower(cv::Mat1d &prob_field)
      : prob_field(prob_field) { }

    /** Adds all points as seeds whose probability is bigger than the set threshold
     * @returns how many seeds have been found.
     */
    int find_seeds(double seed_threshold) {
      this->seed_threshold = seed_threshold;

      for (int row = 0; row < prob_field.rows; ++row) {
        for (int col = 0; col < prob_field.cols; ++col) {
          if (prob_field(row, col) > seed_threshold) {
            seeds.push_back(std::tuple<int, int>(row, col));
          }
        }
      }

      return seeds.size();
    }

    cv::Mat1d grow(double grow_threshold) {
      if (seeds.size() <= 0) {
        std::cout << "Warning: No seeds found. Forgot to call find_seeds?" << std::endl;
      }

      while (seeds.size() > 0) {
        std::tuple<int, int> curr_pos = seeds.back();
        seeds.pop_back();

        for (int d_row = -1; d_row <= 1; d_row++) {
          for (int d_col = -1; d_col <= 1; d_col++) {
            int row = std::get<0>(curr_pos)+ d_row;
            int col = std::get<1>(curr_pos)+ d_col; 
            if (row >= 0 && col >= 0 && row < prob_field.rows && col < prob_field.cols) {
              double new_prob = (prob_field(row, col) + prob_field(std::get<0>(curr_pos), std::get<1>(curr_pos))) / 2.0;
              //std::cout << "avg prob is " << new_prob << ", threshold is " << grow_threshold;
              // Only grow if the probabilites were different
              if (new_prob != prob_field(row, col) && new_prob > grow_threshold) {
                prob_field(row, col) = new_prob;
                seeds.push_back(std::tuple<int, int>(row, col));
                std::cout << " --> growing!" << std::endl;
              } else {
                //std::cout << " --> didn't grow." << std::endl;
              }
            }
          }
        }
      }

      return prob_field;
    }

  private:
    cv::Mat1b prob_field;
    std::vector<std::tuple<int, int> > seeds;
    double seed_threshold;
};

/// Constructor
SkinModel::SkinModel()
{
}

/// Destructor
SkinModel::~SkinModel() 
{
}

/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
    //--- IMPLEMENT THIS ---//
}

/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	//--- IMPLEMENT THIS ---//
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining()
{
	//--- IMPLEMENT THIS ---//
}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

	//--- IMPLEMENT THIS ---//
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {

			if (false)
				skin(row, col) = rand()%256;

			if (false)
				skin(row, col) = img(row,col)[2];

			if (true) {
			
				cv::Vec3b bgr = img(row, col);
				if (bgr[2] > bgr[1] && bgr[1] > bgr[0]) 
					skin(row, col) = 2*(bgr[2] - bgr[0]);
			}
        }
    }

    return skin;
}

