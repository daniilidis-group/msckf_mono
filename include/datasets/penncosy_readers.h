#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <utility>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <msckf_mono/types.h>

namespace penncosy_dataset
{
  class Sensor
  {
    public:
      // folder
      // |- data
      //    |- sensor_name
      //       |- sensor yamls
      //       |- seq
      //          |- csv_name
      //          |- sub_sensor_name
      //             |- images
      Sensor(std::string folder, std::string sensor_name, std::string seq, std::string csv_name, std::string sub_sensor_name="");

      size_t get_cur_time();


    protected:
      std::string sensor_name_;
      std::string sub_sensor_name_;
      std::string csv_name_;
      std::string sequence_;
      std::string folder_;

      size_t cur_time_;

      template<typename Lambda>
      void read_csv(Lambda line_func)
      {
        std::string fullpath(folder_+std::string("/data/")+
                             sensor_name_+std::string("/")+
                             sequence_+std::string("/")+
                             csv_name_);

        std::cout << "Opening file " << fullpath << std::endl;

        std::ifstream f;
        f.open(fullpath.c_str());

        if(!f.is_open()){
          return;
        }

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
      void read_config(Lambda config_reader_func, std::string yaml_name)
      {
        std::string config_name(folder_+std::string("/data/")+
                                sensor_name_+std::string("/")+yaml_name);
        cv::FileStorage fs2;
        fs2.open(config_name.c_str(), cv::FileStorage::READ && cv::FileStorage::FORMAT_YAML, "");
        if(!fs2.isOpened()){
          throw std::runtime_error(config_name+" not opened");
        }
        config_reader_func(fs2);
      }
  };

  class VICamera : public Sensor
  {
    public:
      VICamera(std::string folder, std::string sensor_name, std::string seq, std::string csv_name, std::string sub_sensor_name="");

      double get_time();

      cv::Mat get_data();

      bool next();

      bool has_next();

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

      std::vector<double> reading_list_;
      std::vector<double>::iterator list_iter_;
  };


  class VI_IMU : public Sensor
  {
    public:
      VI_IMU(std::string folder, std::string sensor_name, std::string seq);

      double get_time();

      msckf_mono::measurement get_data();

      bool next();

      bool has_next();

      cv::Mat get_T_BS();
      Eigen::Vector3d get_p_BS();
      Eigen::Quaterniond get_q_BS();
      double get_dT();

    private:
      double dT_;

      cv::Mat T_BS_;
      Eigen::Vector3d p_BS_;
      Eigen::Quaterniond q_BS_;

      std::vector<std::pair<double, msckf_mono::measurement>> reading_list_;
      std::vector<std::pair<double, msckf_mono::measurement>>::iterator list_iter_;
  };
}
