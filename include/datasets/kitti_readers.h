#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <array>
#include <map>
#include <memory>

#include <opencv2/core/core.hpp>
#include <msckf_mono/types.h>
#include <msckf_mono/matrix_utils.h>

namespace kitti_dataset
{
  /*
   * The KITTI dataset has the following format for storing sensor data.
   *
   * - Date
   *   |- calib_cam_to_cam.txt
   *   |- calib_imu_to_velo.txt
   *   |- calib_velo_to_cam.txt
   *   |- Sequence
   *      |- sensor_a
   *         |- timestamps.txt
   *         |- data
   *            |- 0000000000.<filetype>
   *            |- 0000000001.<filetype>
   *      |- sensor_b
   *         |- timestamps.txt
   *         |- data
   *            |- 0000000000.<filetype>
   *            |- 0000000001.<filetype>
   *
   * The sensor data filenames are strings padded to 10 digits.
   *
   *
   * KITTI defines the velodyne as an intermediate frame.
   */

  using kitti_calib_file=std::map<std::string, std::vector<double>>;

  std::vector<double> line_to_vector(std::string& text);
  kitti_calib_file file_to_map(std::string fn);

  class CamCalib
  {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      CamCalib();

      void load_from_map(kitti_calib_file data, std::string cam_name);

      Eigen::Matrix<float,2,1> S;
      msckf_mono::Matrix3<float> K;
      msckf_mono::Matrix3<float> R;
      msckf_mono::Vector3<float> T;
      Eigen::Matrix<float,2,1> S_rect;
      msckf_mono::Matrix3<float> R_rect;
      msckf_mono::Matrix34<float> P_rect;

      msckf_mono::Matrix4<float> R_rect_pad;
  };

  class Calib
  {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      Calib(std::string folder);

      kitti_calib_file cam_to_cam;
      kitti_calib_file imu_to_velo;
      kitti_calib_file velo_to_cam;

      CamCalib cam0;
      CamCalib cam1;
      CamCalib cam2;
      CamCalib cam3;

      msckf_mono::Matrix4<float> T_velo_imu;
      msckf_mono::Matrix4<float> T_cam0unrect_velo;

      msckf_mono::Matrix4<float> T_cam0_velo;
      msckf_mono::Matrix4<float> T_cam1_velo;
      msckf_mono::Matrix4<float> T_cam2_velo;
      msckf_mono::Matrix4<float> T_cam3_velo;

      msckf_mono::Matrix4<float> T_cam0_imu;
      msckf_mono::Matrix4<float> T_cam1_imu;
      msckf_mono::Matrix4<float> T_cam2_imu;
      msckf_mono::Matrix4<float> T_cam3_imu;

      msckf_mono::Matrix3<float> K_cam0;
      msckf_mono::Matrix3<float> K_cam1;
      msckf_mono::Matrix3<float> K_cam2;
      msckf_mono::Matrix3<float> K_cam3;
  };
  
  class Sensor
  {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      Sensor(std::string name, std::string folder, std::string sensor_suffix);

      double get_time();
      bool has_next();
      bool next();
      void reset();

      std::string get_name();

      msckf_mono::Transform<float> get_kitti_transform(std::string frame);

      std::vector<double> get_calib_line(std::string fn, std::string key);

      // This isn't particularly expensive. So just have this around for each instance??
      void load_kitti_calib();

    protected:
      std::string name_;
      std::string folder_;
      std::string sensor_suffix_;

      std::vector<std::pair<double, std::string>> sensor_readings_;
      std::vector<std::pair<double, std::string>>::iterator sensor_readings_iter_;
  };

  class Image : public Sensor
  {
    public:
      Image(std::string name, std::string folder, std::shared_ptr<Calib> calib);

      cv::Mat get_data();

      msckf_mono::Matrix3<float> get_K();
      msckf_mono::Vector3<float> get_p_BS();
      msckf_mono::Quaternion<float> get_q_BS();
      double get_dT();

      msckf_mono::Camera<float> get_camera();

    private:
      msckf_mono::Vector3<float> p_BS_;
      msckf_mono::Quaternion<float> q_BS_;
      msckf_mono::Matrix3<float> K_;
  };

  class Depth : public Sensor
  {
    public:
      Depth(std::string name, std::string folder, std::shared_ptr<Calib> calib);

      cv::Mat get_data();
  };

  enum oxtsIndex {
    lat,lon,alt, // Latitude, longitude, altitude
    roll,pitch,yaw,
    vn,ve,vf,vl,vu, // Velocity North, East, Forward, Left, Up
    ax,ay,az, // Acceleration x,y,z
    af,al,au, // Acceleration Forward, Left, Up
    wx,wy,wz, // Angular Velocity x,y,z
    wf,wl,wu, // Angular Velocity Forward, Left, Up
    pos_accuracy,vel_accuracy, // Accuracy Estimates
    navstat,numsats,
    posmode,velmode,orimode
  };

  class OXTS : public Sensor
  {
    public:
      OXTS(std::string name, std::string folder);

      std::string get_name();

      std::array<double, 30> read_oxts_file(std::string filenum);

    protected:
      std::string oxts_name_;
  };

  class IMU : public OXTS
  {
    public:
      IMU(std::string folder);

      msckf_mono::imuReading<float> get_data();
  };

  class GroundTruth : public OXTS
  {
    public:
      GroundTruth(std::string folder);

      msckf_mono::imuState<float> get_data();

    private:
      bool has_origin_;
      double scale_;
      msckf_mono::imuState<float> origin_;
  };
}
