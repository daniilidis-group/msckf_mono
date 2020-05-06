#include <datasets/penncosy_readers.h>
#include <opencv2/highgui/highgui.hpp>

namespace penncosy_dataset
{
  Sensor::Sensor(std::string folder, std::string sensor_name, std::string seq, std::string csv_name, std::string sub_sensor_name) :
    sensor_name_(sensor_name), sequence_(seq), folder_(folder), sub_sensor_name_(sub_sensor_name), csv_name_(csv_name), cur_time_(0)
    {
    }

  size_t Sensor::get_cur_time(){return cur_time_;}

  VICamera::VICamera(std::string folder, std::string sensor_name, std::string seq, std::string csv_name, std::string cam_prefix) :
    Sensor(folder, sensor_name, seq, csv_name, cam_prefix+std::string("_cam_frames"))
    {
      read_config( [&](cv::FileStorage& fs)->bool
          {
          cv::FileNode t_bs_fn = fs["T_BS"];
          int rows = (int)t_bs_fn["rows"];
          int cols = (int)t_bs_fn["cols"];
          T_BS_ = cv::Mat(rows, cols, CV_32FC1);
          cv::FileNodeIterator it = t_bs_fn["data"].begin();
          for(int i=0; i<rows; i++){
          for(int j=0; j<cols; j++){
          T_BS_.at<float>(i, j) = (float)(*it);
          ++it;
          }
          }

          Eigen::Matrix3d R_BS;
          R_BS << T_BS_.at<float>(0,0), T_BS_.at<float>(0,1), T_BS_.at<float>(0,2),
          T_BS_.at<float>(1,0), T_BS_.at<float>(1,1), T_BS_.at<float>(1,2),
          T_BS_.at<float>(2,0), T_BS_.at<float>(2,1), T_BS_.at<float>(2,2);

          q_BS_ = Eigen::Quaterniond(R_BS).inverse();
          p_BS_ << T_BS_.at<float>(0,3), T_BS_.at<float>(1,3), T_BS_.at<float>(2,3);

          fs["rate_hz"] >> rate_hz_;

          std::vector<int> res;
          fs["resolution"] >> res;
          width_ = res.at(0);
          height_ = res.at(1);

          fs["camera_model"] >> camera_model_;

          std::vector<float> intrinsics;
          fs["intrinsics"] >> intrinsics; // fu, fv, cu, cv
          K_ = cv::Mat::eye(3, 3, CV_32F);
          K_.at<float>(0,0) = intrinsics[0];
          K_.at<float>(1,1) = intrinsics[1];
          K_.at<float>(0,2) = intrinsics[2];
          K_.at<float>(1,2) = intrinsics[3];

          fs["distortion_model"] >> distortion_model_;

          std::vector<float> coeffs;
          fs["distortion_coefficients"] >> coeffs;
          distortion_coefficients_ = cv::Mat::zeros(coeffs.size(), 1, CV_32F);
          int i = 0;
          for(auto c : coeffs){
            distortion_coefficients_.at<float>(i++) = c;
          }

          return true;
          }, cam_prefix+std::string(".yaml") );

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

  double VICamera::get_time()
  {
    return *list_iter_;
  }

  cv::Mat VICamera::get_data()
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

  bool VICamera::next()
  {
    if(!has_next())
      return false;
    ++list_iter_;
    return true;
  }

  bool VICamera::has_next()
  {
    if(reading_list_.size()==0)
      return false;
    return list_iter_ != reading_list_.end();
  }

  cv::Mat VICamera::get_T_BS()
  {
    return T_BS_;
  }

  Eigen::Vector3d VICamera::get_p_BS()
  {
    return p_BS_;
  }
  Eigen::Quaterniond VICamera::get_q_BS()
  {
    return q_BS_;
  }
  std::string VICamera::get_camera_model()
  {
    return camera_model_;
  }
  cv::Mat VICamera::get_K()
  {
    return K_;
  }
  std::string VICamera::get_dist_model()
  {
    return distortion_model_;
  }
  cv::Mat VICamera::get_dist_coeffs()
  {
    return distortion_coefficients_;
  }
  double VICamera::get_dT()
  {
    return 1./rate_hz_;
  }


  VI_IMU::VI_IMU(std::string folder, std::string sensor_name, std::string seq) :
    Sensor(folder, sensor_name, seq, "imu.txt", "visensor")
  {
    dT_ = 1./200.; // frequency of the VI sensor IMU

    read_config( [&](cv::FileStorage& fs)->bool
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
        }, std::string("imu.yaml") );

    read_csv( [&](std::stringstream& s)->bool
        {
        if( s.rdbuf()->in_avail() == 0 )
        return false;

        std::pair<double, msckf_mono::measurement> p;
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

  double VI_IMU::get_time()
  {
    return list_iter_->first;
  }

  msckf_mono::measurement VI_IMU::get_data()
  {
    return list_iter_->second;
  }

  bool VI_IMU::next()
  {
    if(!has_next())
      return false;
    ++list_iter_;
    return true;
  }

  bool VI_IMU::has_next()
  {
    if(reading_list_.size()==0)
      return false;
    return list_iter_ != reading_list_.end();
  }


  cv::Mat VI_IMU::get_T_BS(){return T_BS_;}
  Eigen::Vector3d VI_IMU::get_p_BS(){return p_BS_;}
  Eigen::Quaterniond VI_IMU::get_q_BS(){return q_BS_;}
  double VI_IMU::get_dT(){return dT_;}
}
