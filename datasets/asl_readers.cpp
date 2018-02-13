#include <datasets/asl_readers.h>
#include <opencv2/highgui/highgui.hpp>

namespace asl_dataset
{
  Sensor::Sensor(std::string name, std::string folder) :
    name_(name), folder_(folder), cur_time_(0)
  {
  }

  Camera::Camera(std::string name, std::string folder) : Sensor(name, folder)
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
        } );
    read_csv( [&](std::stringstream& s)->bool
        {
        if( s.rdbuf()->in_avail() == 0 )
        return false;

        std::pair<size_t, std::string> p;
        s >> p.first; // timestamp [ns]
        char ch;
        s >> ch; // handle the comma
        s >> p.second; // filename
        image_list_.push_back(p);
        return true;
        } );

    list_iter_ = image_list_.begin();
  }

  size_t Camera::get_time()
  {
    return list_iter_->first;
  }

  cv::Mat Camera::get_data()
  {
    std::string fn = folder_ + "/data/" + list_iter_->second;
    cv::Mat img = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
    if(img.rows == 0 && img.cols == 0){
      std::cout << "Error opening image " << fn << std::endl;
    }
    return img;
  }

  bool Camera::has_next()
  {
    if(image_list_.size()==0)
      return false;

    return list_iter_ != image_list_.end();
  }

  bool Camera::next()
  {
    if(!has_next())
      return false;
    ++list_iter_;
    return true;
  }

  cv::Mat Camera::get_T_BS()
  {
    return T_BS_;
  }

  Eigen::Vector3d Camera::get_p_BS()
  {
    return p_BS_;
  }
  Eigen::Quaterniond Camera::get_q_BS()
  {
    return q_BS_;
  }
  std::string Camera::get_camera_model()
  {
    return camera_model_;
  }
  cv::Mat Camera::get_K()
  {
    return K_;
  }
  std::string Camera::get_dist_model()
  {
    return distortion_model_;
  }
  cv::Mat Camera::get_dist_coeffs()
  {
    return distortion_coefficients_;
  }
  double Camera::get_dT()
  {
    return 1./rate_hz_;
  }

  IMU::IMU(std::string name, std::string folder) : Sensor(name, folder)
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
        q_BS_ = Eigen::Quaterniond(R_BS);
        p_BS_ << T_BS_.at<float>(0,3), T_BS_.at<float>(1,3), T_BS_.at<float>(2,3);


        fs["rate_hz"] >> dT_;
        dT_ = 1.0/dT_;

        fs["gyroscope_noise_density"] >> gyroscope_noise_density_;
        fs["gyroscope_random_walk"] >> gyroscope_random_walk_;
        fs["accelerometer_noise_density"] >> accelerometer_noise_density_;
        fs["accelerometer_random_walk"] >> accelerometer_random_walk_;

        return true;
        } );

    read_csv( [&](std::stringstream& s)->bool
        {
        if( s.rdbuf()->in_avail() == 0 )
        return false;

        std::pair<size_t, msckf::measurement> p;
        char c;
        s >> p.first; // timestamp [ns]
        s >> c;
        s >> p.second.omega[0]; // w_RS_S_x [rad s^-1]
        s >> c;
        s >> p.second.omega[1]; // w_RS_S_y [rad s^-1]
        s >> c;
        s >> p.second.omega[2]; // w_RS_S_z [rad s^-1]
        s >> c;
        s >> p.second.a[0]; // a_RS_S_x [m s^-2]
        s >> c;
        s >> p.second.a[1]; // a_RS_S_y [m s^-2]
        s >> c;
        s >> p.second.a[2]; // a_RS_S_z [m s^-2]
        p.second.dT = dT_;

        reading_list_.push_back(p);
        return true;
        } );

    list_iter_ = reading_list_.begin();
  }

  size_t IMU::get_time()
  {
    return list_iter_->first;
  }

  msckf::measurement IMU::get_data()
  {
    return list_iter_->second;
  }

  bool IMU::next()
  {
    if(!has_next())
      return false;
    ++list_iter_;
    return true;
  }

  bool IMU::has_next()
  {
    if(reading_list_.size()==0)
      return false;
    return list_iter_ != reading_list_.end();
  }


  cv::Mat IMU::get_T_BS(){return T_BS_;}
  Eigen::Vector3d IMU::get_p_BS(){return p_BS_;}
  Eigen::Quaterniond IMU::get_q_BS(){return q_BS_;}
  double IMU::get_dT(){return dT_;}
  double IMU::get_gnd(){return gyroscope_noise_density_;}
  double IMU::get_grw(){return gyroscope_random_walk_;}
  double IMU::get_and(){return accelerometer_noise_density_;}
  double IMU::get_arw(){return accelerometer_random_walk_;}

  GroundTruth::GroundTruth(std::string name, std::string folder) : Sensor(name, folder)
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
        q_BS_ = Eigen::Quaterniond(R_BS);
        p_BS_ << T_BS_.at<float>(0,3), T_BS_.at<float>(1,3), T_BS_.at<float>(2,3);

        return true;
        } );

    read_csv( [&](std::stringstream& s)->bool
        {
        if( s.rdbuf()->in_avail() == 0 )
        return false;

        std::pair<size_t, msckf::imuState> p;
        char c;
        p.second.p_I_G_null = Eigen::Vector3d::Zero();
        p.second.v_I_G_null = Eigen::Vector3d::Zero();
        p.second.q_IG_null = Eigen::Quaterniond::Identity();

        s >> p.first;
        s >> c;

        s >> p.second.p_I_G[0]; //  p_RS_R_x [m]
        s >> c;
        s >> p.second.p_I_G[1]; //  p_RS_R_y [m]
        s >> c;
        s >> p.second.p_I_G[2]; //  p_RS_R_z [m]
        s >> c;

        s >> p.second.q_IG.w(); //  q_RS_w []
        s >> c;
        s >> p.second.q_IG.x(); //  q_RS_x []
        s >> c;
        s >> p.second.q_IG.y(); //  q_RS_y []
        s >> c;
        s >> p.second.q_IG.z(); //  q_RS_z []
        s >> c;

        s >> p.second.v_I_G[0]; //  v_RS_R_x [m s^-1]
        s >> c;
        s >> p.second.v_I_G[1]; //  v_RS_R_y [m s^-1]
        s >> c;
        s >> p.second.v_I_G[2]; //  v_RS_R_z [m s^-1]
        s >> c;

        s >> p.second.b_g[0]; //  b_w_RS_S_x [rad s^-1]
        s >> c;
        s >> p.second.b_g[1]; //  b_w_RS_S_y [rad s^-1]
        s >> c;
        s >> p.second.b_g[2]; //  b_w_RS_S_z [rad s^-1]
        s >> c;

        s >> p.second.b_a[0]; //  b_w_RS_S_x [rad s^-1]
        s >> c;
        s >> p.second.b_a[1]; //  b_w_RS_S_y [rad s^-1]
        s >> c;
        s >> p.second.b_a[2]; //  b_w_RS_S_z [rad s^-1]

        p.second.g[0] = 0.0;
        p.second.g[1] = 0.0;
        p.second.g[2] = -9.81;

        p.second.v_I_G = p.second.q_IG * p.second.v_I_G;
        p.second.q_IG = p.second.q_IG.inverse();
        p.second.p_I_G = p.second.p_I_G;
        // Check this

        reading_list_.push_back(p);
        return true;
        } );

    list_iter_ = reading_list_.begin();
  }

  size_t GroundTruth::get_time()
  {
    return list_iter_->first;
  }

  msckf::imuState GroundTruth::get_data()
  {
    return list_iter_->second;
  }

  bool GroundTruth::next()
  {
    if(!has_next())
      return false;
    ++list_iter_;
    return true;
  }

  bool GroundTruth::has_next()
  {
    if(reading_list_.size()==0)
      return false;
    return list_iter_ != reading_list_.end();
  }
}
