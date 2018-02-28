/*
 * Author Kenneth Chaney
 *
 * Handles tracking of features across an image stream
 */

#pragma once

#include <msckf_mono/types.h>

namespace msckf_mono {
  typedef cv::Point2f Point2f;
  typedef std::vector<cv::Point2f> Point2fVector;

  typedef <typename _Scalar>
    class CornerTracker {
      public:
        CornerTracker();
  
        void configure(std::pair<int, int> grid_size,
                       double detection_threshold,
                       int window_size,
                       double ransac_threshold,
                       double min_eigen_thresh,
                       int max_level,
                       int max_iters,
                       double epsilon);
  
        void add_gyroscope_reading(GyroscopeReading<_Scalar> imu, double timestamp);
        void add_image(cv::Mat image, double timestamp);
  
        void get_current_features();
        void get_prev_features();
  
        void draw_tracks();
  
        void clear();
  
      private:
        void two_point_ransac();
        void integrate_gyro();
        void detect_features();
    };
}
