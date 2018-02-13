#pragma once

// Standard
#include <vector>
#include <map>
#include <unordered_map>
#include <stdlib.h>
#include <math.h>
#include <iostream>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"

// Fast corner detection
#include <fast/fast.h>

// Eigen
#include <Eigen/Dense>

namespace corner_detector
{
typedef cv::Point2f Point2f; 
typedef std::vector<cv::Point2f> Point2fVector;
typedef std::vector<size_t> IdVector;

class CornerDetector{
public:
  CornerDetector(int n_rows=8, int n_cols=10, double detection_threshold=40.0);
  ~CornerDetector() { };
  void detect_features(const cv::Mat& image, std::vector<cv::Point2f>& features);
  void set_grid_position(const cv::Point2f& pos);
  void set_grid_size(int n_rows, int n_cols);

  int get_n_rows(){return grid_n_rows_;}
  int get_n_cols(){return grid_n_cols_;}

  float shiTomasiScore(const cv::Mat& img, int u, int v);
  int sub2ind(const cv::Point2f& sub);
private:
  void zero_occupancy_grid();
  std::vector<bool> occupancy_grid_;
  // Size of each grid rectangle in pixels
  int grid_n_rows_, grid_n_cols_, grid_width_, grid_height_;
  // Threshold for corner score
  double detection_threshold_;
}; // CornerDetector class

class CornerTracker
{
  public:
    CornerTracker(int window_size=31,
                  double min_eigen_threshold=0.001,
                  int max_level=3,
                  int termcrit_max_iters=50,
                  double termcirt_epsilon=0.01);

    ~CornerTracker() {};

    void configure(double window_size,
                   double min_eigen_threshold,
                   int max_level,
                   int termcrit_max_iters,
                   double termcirt_epsilon);

    void track_features(cv::Mat img_1, cv::Mat img_2, Point2fVector& points1, Point2fVector& points2, IdVector& id1, IdVector& id2);

  private:
    cv::Size window_size_;

    double min_eigen_threshold_;
    int max_level_;

    cv::TermCriteria termination_criteria_;
};

class TrackVisualizer
{
  public:
    TrackVisualizer();

    void add_current_features(Point2fVector& features, IdVector& feature_ids);
    void add_predicted(Point2fVector& features, IdVector& feature_ids);

    cv::Mat draw_tracks(cv::Mat img);

  private:
    std::unordered_map<int, Point2fVector> feature_tracks_;
    std::unordered_map<int, Point2f> predicted_pts_;
};

class TrackHandler
{
  public:
    TrackHandler(const cv::Mat K);
    ~TrackHandler();

    Eigen::Array<bool, Eigen::Dynamic, 1>
      twoPointRansac(const Eigen::Matrix3d& dR,
          const std::vector<Eigen::Vector2d,
          Eigen::aligned_allocator<Eigen::Vector2d>>& old_points_in,
          const std::vector<Eigen::Vector2d,
          Eigen::aligned_allocator<Eigen::Vector2d>>& new_points_in);

    void add_gyro_reading(Eigen::Vector3d& gyro_reading);
    cv::Mat integrate_gyro();
    void predict_features(Point2fVector& predicted_pts);

    void track_features(cv::Mat img, Point2fVector& features, IdVector& feature_ids, double cur_time);
    void clear_tracks();
    size_t get_next_feature_id(){return next_feature_id_;}

    Point2fVector get_prev_features(){return prev_features_;}
    IdVector get_prev_ids(){return prev_feature_ids_;}

    void set_grid_size(int n_rows, int n_cols);
    void set_ransac_threshold(double rt);

    cv::Mat get_track_image();

  private:
    double ransac_threshold_;
    CornerDetector detector_;
    CornerTracker tracker_;
    
    double cur_time_;

    cv::Mat prev_img_;
    Point2fVector prev_features_;
    IdVector prev_feature_ids_;
    double prev_time_;

    Eigen::Vector3d gyro_accum_;
    size_t n_gyro_readings_;

    size_t next_feature_id_;

    const cv::Mat K_;
    const cv::Mat K_inv_;

    TrackVisualizer visualizer_;
};

} // corner_detector namespace
