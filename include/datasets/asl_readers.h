#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <msckf_mono/types.h>

namespace asl_dataset
{
  using timestamp = unsigned long long;
  
  class Sensor
  {
    public:
      Sensor(std::string name, std::string folder);

    protected:
      std::string name_;
      std::string folder_;

      timestamp cur_time_;

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

      timestamp get_time();

      cv::Mat get_data();

      bool has_next();

      bool next();

      cv::Mat get_T_BS();
      msckf_mono::Vector3<float> get_p_BS();
      msckf_mono::Quaternion<float> get_q_BS();
      std::string get_camera_model();
      cv::Mat get_K();
      std::string get_dist_model();
      cv::Mat get_dist_coeffs();
      double get_dT();

    private:
      msckf_mono::Vector3<float> p_BS_;
      msckf_mono::Quaternion<float> q_BS_;

      cv::Mat T_BS_;

      double rate_hz_;
      int width_;
      int height_;
      std::string camera_model_;
      cv::Mat K_; // fu, fv, cu, cv
      std::string distortion_model_;
      cv::Mat distortion_coefficients_;

      std::vector<std::pair<timestamp, std::string>> image_list_;
      std::vector<std::pair<timestamp, std::string>>::iterator list_iter_;
  };

  class IMU : public Sensor
  {
    public:
      IMU(std::string name, std::string folder);

      timestamp get_time();
      msckf_mono::imuReading<float> get_data();

      bool next();
      bool has_next();

      cv::Mat get_T_BS();
      msckf_mono::Vector3<float> get_p_BS();
      msckf_mono::Quaternion<float> get_q_BS();
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
      msckf_mono::Vector3<float> p_BS_;
      msckf_mono::Quaternion<float> q_BS_;

      std::vector<std::pair<timestamp, msckf_mono::imuReading<float>>> reading_list_;
      std::vector<std::pair<timestamp, msckf_mono::imuReading<float>>>::iterator list_iter_;
  };

  class GroundTruth : public Sensor
  {
    public:
      GroundTruth(std::string name, std::string folder);

      timestamp get_time();
      msckf_mono::imuState<float> get_data();

      bool next();
      bool has_next();

    private:
      cv::Mat T_BS_;
      msckf_mono::Vector3<float> p_BS_;
      msckf_mono::Quaternion<float> q_BS_;


      std::vector<std::pair<timestamp, msckf_mono::imuState<float>>> reading_list_;
      std::vector<std::pair<timestamp, msckf_mono::imuState<float>>>::iterator list_iter_;
  };
}
