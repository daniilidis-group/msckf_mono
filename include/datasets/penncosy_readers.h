#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <msckf_mono/measurement.h>
#include <msckf_mono/imustate.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace penncosy_dataset
{
  class Sensor
  {
    public:
      // folder
      // |- data
      //    |- sensor_name
      //       |- seq
      //          |- csv_name
      //          |- sub_sensor_name
      //             |- images
      Sensor(std::string folder, std::string sensor_name, std::string seq, std::string csv_name, std::string sub_sensor_name="") :
        sensor_name_(sensor_name), sequence_(seq), folder_(folder), sub_sensor_name_(sub_sensor_name), csv_name_(csv_name), cur_time_(0)
      {
      }

      size_t get_cur_time(){return cur_time_;}


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
      void read_config(Lambda config_reader_func)
      {
        config_reader_func();
      }
  };

  class Camera : public Sensor
  {
    public:
      Camera(std::string folder, std::string sensor_name, std::string seq, std::string csv_name, std::string sub_sensor_name="") :
        Sensor(folder, sensor_name, seq, csv_name, sub_sensor_name)
      {
        read_config( [&]()->bool
        {
          std::stringstream transform_line;

          Eigen::Matrix3d R_BS;
          transform_line >> R_BS(0,0);
          transform_line >> R_BS(0,1);
          transform_line >> R_BS(0,2);
          transform_line >>p_BS_(0);
          transform_line >> R_BS(1,0);
          transform_line >> R_BS(1,1);
          transform_line >> R_BS(1,2);
          transform_line >>p_BS_(1);
          transform_line >> R_BS(2,0);
          transform_line >> R_BS(2,1);
          transform_line >> R_BS(2,2);
          transform_line >>p_BS_(2);

          q_BS_ = Eigen::Quaterniond(R_BS);
          return true;
        } );

        read_csv( [&](std::stringstream& s)->bool
        {
          if( s.rdbuf()->in_avail() == 0 )
            return false;

          double p;
          s >> p; // timestamp [s]
          reading_list_.push_back(p);

          return true;
        } );

        list_iter_ = reading_list_.begin();
      }

      double get_time()
      {
        return *list_iter_;
      }

      cv::Mat get_data()
      {
      // folder
      // |- data
      //    |- sensor_name
      //       |- seq
      //          |- csv_name
      //          |- sub_sensor_name
      //             |- images
      //
        std::stringstream stream;
        stream << folder_ << "/data/" << sensor_name_ <<
                  "/" << sequence_ << "/" << sub_sensor_name_ <<
                  "/frame_" << std::setfill('0') << std::setw(4) << (size_t)(list_iter_ - reading_list_.begin())+1 << ".png";

        cv::Mat img = cv::imread(stream.str(), CV_LOAD_IMAGE_GRAYSCALE);
        if(img.rows == 0 && img.cols == 0){
          std::cout << "Error opening image " << stream.str() << std::endl;
        }
        return img;
      }

      bool next()
      {
        if(!has_next())
          return false;
        ++list_iter_;
        return true;
      }

      bool has_next()
      {
        if(reading_list_.size()==0)
          return false;
        return list_iter_ != reading_list_.end();
      }


      cv::Mat get_T_BS(){return T_BS_;}
      Eigen::Vector3d get_p_BS(){return p_BS_;}
      Eigen::Quaterniond get_q_BS(){return q_BS_;}

    private:

      cv::Mat T_BS_;
      Eigen::Vector3d p_BS_;
      Eigen::Quaterniond q_BS_;

      std::vector<double> reading_list_;
      std::vector<double>::iterator list_iter_;
  };


  class VI_IMU : public Sensor
  {
    public:
      VI_IMU(std::string folder, std::string sensor_name, std::string seq) :
        Sensor(folder, sensor_name, seq, "imu.txt", "visensor")
      {
        dT_ = 1./200.; // frequency of the VI sensor IMU

        read_config( [&]()->bool
        {
          std::stringstream transform_line;

          Eigen::Matrix3d R_BS;
          transform_line >> R_BS(0,0);
          transform_line >> R_BS(0,1);
          transform_line >> R_BS(0,2);
          transform_line >>p_BS_(0);
          transform_line >> R_BS(1,0);
          transform_line >> R_BS(1,1);
          transform_line >> R_BS(1,2);
          transform_line >>p_BS_(1);
          transform_line >> R_BS(2,0);
          transform_line >> R_BS(2,1);
          transform_line >> R_BS(2,2);
          transform_line >>p_BS_(2);

          q_BS_ = Eigen::Quaterniond(R_BS);
          return true;
        } );

        read_csv( [&](std::stringstream& s)->bool
        {
          if( s.rdbuf()->in_avail() == 0 )
            return false;

          std::pair<double, msckf::measurement> p;
          s >> p.first; // timestamp [s]

          s >> p.second.a[0]; // a_RS_S_x [m s^-2]
          s >> p.second.a[1]; // a_RS_S_y [m s^-2]
          s >> p.second.a[2]; // a_RS_S_z [m s^-2]

          s >> p.second.omega[0]; // w_RS_S_x [rad s^-1]
          s >> p.second.omega[1]; // w_RS_S_y [rad s^-1]
          s >> p.second.omega[2]; // w_RS_S_z [rad s^-1]

          p.second.dT = dT_;

          reading_list_.push_back(p);
          return true;
        } );

        list_iter_ = reading_list_.begin();
      }

      double get_time()
      {
        return list_iter_->first;
      }

      msckf::measurement get_data()
      {
        return list_iter_->second;
      }

      bool next()
      {
        if(!has_next())
          return false;
        ++list_iter_;
        return true;
      }

      bool has_next()
      {
        if(reading_list_.size()==0)
          return false;
        return list_iter_ != reading_list_.end();
      }


      cv::Mat get_T_BS(){return T_BS_;}
      Eigen::Vector3d get_p_BS(){return p_BS_;}
      Eigen::Quaterniond get_q_BS(){return q_BS_;}
      double get_dT(){return dT_;}

    private:
      double dT_;

      cv::Mat T_BS_;
      Eigen::Vector3d p_BS_;
      Eigen::Quaterniond q_BS_;

      std::vector<std::pair<double, msckf::measurement>> reading_list_;
      std::vector<std::pair<double, msckf::measurement>>::iterator list_iter_;
  };
}
