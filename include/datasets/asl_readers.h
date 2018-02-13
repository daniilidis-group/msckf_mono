#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <msckf_mono/measurement.h>
#include <msckf_mono/imustate.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace asl_dataset
{
  class Sensor
  {
    public:
      Sensor(std::string name, std::string folder);

    protected:
      std::string name_;
      std::string folder_;

      size_t cur_time_;

      template<typename Lambda>
      void read_csv(Lambda line_func)
      {
        std::string csv_name(folder_+std::string("/data.csv"));


        std::ifstream f;
        f.open(csv_name.c_str());

        if(!f.is_open()){
          return;
        }

        std::string header;
        std::getline(f,header);

        while(!f.eof())
        {
          std::string sline;
          std::getline(f,sline);
          std::stringstream ssline;
          ssline << sline;
          line_func(ssline);
        }
      }

      template<typename Lambda>
      void read_config(Lambda config_reader_func)
      {
        std::string config_name(folder_+std::string("/sensor.yaml"));
        cv::FileStorage fs2;
        fs2.open(config_name.c_str(), cv::FileStorage::READ && cv::FileStorage::FORMAT_YAML, "");
        if(!fs2.isOpened()){
          throw std::runtime_error(config_name+" not opened");
        }
        config_reader_func(fs2);
      }
  };

  class Camera : public Sensor
  {
    public:
      Camera(std::string name, std::string folder);

      size_t get_time();

      cv::Mat get_data();

      bool has_next();

      bool next();

      cv::Mat get_T_BS();
      Eigen::Vector3d get_p_BS();
      Eigen::Quaterniond get_q_BS();
      std::string get_camera_model();
      cv::Mat get_K();
      std::string get_dist_model();
      cv::Mat get_dist_coeffs();
      double get_dT();

    private:
      Eigen::Vector3d p_BS_;
      Eigen::Quaterniond q_BS_;

      cv::Mat T_BS_;

      double rate_hz_;
      int width_;
      int height_;
      std::string camera_model_;
      cv::Mat K_; // fu, fv, cu, cv
      std::string distortion_model_;
      cv::Mat distortion_coefficients_;

      std::vector<std::pair<size_t, std::string>> image_list_;
      std::vector<std::pair<size_t, std::string>>::iterator list_iter_;
  };

  class IMU : public Sensor
  {
    public:
      IMU(std::string name, std::string folder);

      size_t get_time();
      msckf::measurement get_data();
      bool next();
      bool has_next();

      cv::Mat get_T_BS();
      Eigen::Vector3d get_p_BS();
      Eigen::Quaterniond get_q_BS();
      double get_dT();
      double get_gnd();
      double get_grw();
      double get_and();
      double get_arw();

    private:
      double gyroscope_noise_density_;
      double gyroscope_random_walk_;
      double accelerometer_noise_density_;
      double accelerometer_random_walk_;

      double dT_;

      cv::Mat T_BS_;
      Eigen::Vector3d p_BS_;
      Eigen::Quaterniond q_BS_;

      std::vector<std::pair<size_t, msckf::measurement>> reading_list_;
      std::vector<std::pair<size_t, msckf::measurement>>::iterator list_iter_;
  };

  class GroundTruth : public Sensor
  {
    public:
      GroundTruth(std::string name, std::string folder);

      size_t get_time();
      msckf::imuState get_data();
      bool next();
      bool has_next();

    private:
      cv::Mat T_BS_;
      Eigen::Vector3d p_BS_;
      Eigen::Quaterniond q_BS_;


      std::vector<std::pair<size_t, msckf::imuState>> reading_list_;
      std::vector<std::pair<size_t, msckf::imuState>>::iterator list_iter_;
  };
}
