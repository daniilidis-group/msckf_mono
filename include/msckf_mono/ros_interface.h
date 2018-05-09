#ifndef MSCKF_MONO_ROS_INTERFACE_H_
#define MSCKF_MONO_ROS_INTERFACE_H_

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <msckf_mono/types.h>
#include <msckf_mono/msckf.h>
#include <msckf_mono/corner_detector.h>
#include <atomic>

namespace msckf_mono
{
  class RosInterface {
    public:
      RosInterface(ros::NodeHandle nh);

      void imuCallback(const sensor_msgs::ImuConstPtr& imu);

      void imageCallback(const sensor_msgs::ImageConstPtr& msg);

      void publish_core();

      void publish_extra(const ros::Time& publish_time);

    private:
      ros::NodeHandle nh_;
      image_transport::ImageTransport it_;

      std::string subscribe_topic_;
      image_transport::Subscriber image_sub_;
      image_transport::Publisher track_image_pub_;

      ros::Subscriber imu_sub_;

      void load_parameters();

      bool debug_;

      std::atomic<bool> is_calibrating_imu_;
      bool is_first_imu_;
      std::vector<std::tuple<double, imuReading<float>>> imu_queue_;
      double prev_imu_time_;

      void setup_msckf();
      std::shared_ptr<MSCKF<float>> msckf_;
      msckf_mono::Camera<float> camera_;
      msckf_mono::noiseParams<float> noise_params_;
      Eigen::Matrix<float,12,1> Q_imu_vars_;
      Eigen::Matrix<float,15,1> IMUCovar_vars_;
      msckf_mono::MSCKFParams<float> msckf_params_;
      msckf_mono::imuState<float> init_imu_state_;

      void setup_track_handler();
      std::shared_ptr<corner_detector::TrackHandler> track_handler_;

      msckf_mono::Matrix4<float> T_cam_imu_;
      msckf_mono::Matrix3<float> R_cam_imu_;
      msckf_mono::Vector3<float> p_cam_imu_;

      std::string camera_model_;
      cv::Mat K_;
      std::string distortion_model_;
      cv::Mat dist_coeffs_;

      int n_grid_cols_;
      int n_grid_rows_;
      float ransac_threshold_;
  };
}

#endif
