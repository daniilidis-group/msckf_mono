#include <datasets/kitti_readers.h>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

namespace kitti_dataset
{
  std::vector<double> line_to_vector(const std::string& text){
    std::vector<double> data;

    std::stringstream ssline;
    ssline << text;
    std::string sdata;

    while(std::getline(ssline, sdata, ' ')){
      data.push_back(std::stod(sdata));
    }

    return data;
  }

  std::map<std::string, std::vector<double>> file_to_map(std::string fn) {
    std::map<std::string, std::vector<double>> out;

    std::ifstream f;
    f.open(fn.c_str());

    if(!f.is_open()){
      std::cout << "Failed opening: " << fn << std::endl;
      return out;
    }

    int i = 0;
    std::string line;

    while(std::getline(f, line)) {
      int key_index = line.find(':');
      std::string key = line.substr(0, key_index);
      if(key != "calib_time"){
        std::vector<double> data = line_to_vector(line.substr(key_index+2));
        out[key] = data;
      }
    }

    return out;
  }

  CamCalib::CamCalib() {
  }

  void CamCalib::load_from_map(kitti_calib_file data, std::string cam_name) {
    msckf_mono::vector_to_eigen(data[std::string("S_")+cam_name], S);
    msckf_mono::vector_to_eigen(data[std::string("K_")+cam_name], K);
    msckf_mono::vector_to_eigen(data[std::string("R_")+cam_name], R);
    msckf_mono::vector_to_eigen(data[std::string("T_")+cam_name], T);
    msckf_mono::vector_to_eigen(data[std::string("S_rect_")+cam_name], S_rect);
    msckf_mono::vector_to_eigen(data[std::string("R_rect_")+cam_name], R_rect);
    msckf_mono::vector_to_eigen(data[std::string("P_rect_")+cam_name], P_rect);

    R_rect_pad.Identity();
    R_rect_pad.block<3,3>(0,0) = R_rect;
  }

  Calib::Calib(std::string folder) {
    cam_to_cam = file_to_map(folder + "/../calib_cam_to_cam.txt");
    imu_to_velo = file_to_map(folder + "/../calib_imu_to_velo.txt");
    velo_to_cam = file_to_map(folder + "/../calib_velo_to_cam.txt");

    cam0.load_from_map(cam_to_cam, "00");
    cam1.load_from_map(cam_to_cam, "01");
    cam2.load_from_map(cam_to_cam, "02");
    cam3.load_from_map(cam_to_cam, "03");

    msckf_mono::Matrix3<float> R =  msckf_mono::vector_to_eigen<float,3,3,double>(imu_to_velo["R"]);
    msckf_mono::Vector3<float> t =  msckf_mono::vector_to_eigen<float,3,1,double>(imu_to_velo["T"]);

    msckf_mono::transform_from_mats(
        msckf_mono::vector_to_eigen<float,3,3,double>(imu_to_velo["R"]),
        msckf_mono::vector_to_eigen<float,3,1,double>(imu_to_velo["T"]),
        T_velo_imu
        );

    msckf_mono::transform_from_mats(
        msckf_mono::vector_to_eigen<float,3,3,double>(velo_to_cam["R"]),
        msckf_mono::vector_to_eigen<float,3,1,double>(velo_to_cam["T"]),
        T_cam0unrect_velo
        );

    K_cam0 = cam0.P_rect.block<3,3>(0,0);
    K_cam1 = cam1.P_rect.block<3,3>(0,0);
    K_cam2 = cam2.P_rect.block<3,3>(0,0);
    K_cam3 = cam3.P_rect.block<3,3>(0,0);

    msckf_mono::Matrix4<float> T0 = msckf_mono::Matrix4<float>::Identity();
    msckf_mono::Matrix4<float> T1 = msckf_mono::Matrix4<float>::Identity();
    msckf_mono::Matrix4<float> T2 = msckf_mono::Matrix4<float>::Identity();
    msckf_mono::Matrix4<float> T3 = msckf_mono::Matrix4<float>::Identity();

    T0(0,3) = cam0.P_rect(0,3) / cam0.P_rect(0,0);
    T1(0,3) = cam1.P_rect(0,3) / cam1.P_rect(0,0);
    T2(0,3) = cam2.P_rect(0,3) / cam2.P_rect(0,0);
    T3(0,3) = cam3.P_rect(0,3) / cam3.P_rect(0,0);

    T_cam0_velo = T0 * (cam0.R_rect_pad * T_cam0unrect_velo);
    T_cam1_velo = T1 * (cam0.R_rect_pad * T_cam0unrect_velo);
    T_cam2_velo = T2 * (cam0.R_rect_pad * T_cam0unrect_velo);
    T_cam3_velo = T3 * (cam0.R_rect_pad * T_cam0unrect_velo);

    T_cam0_imu = T_cam0_velo * T_velo_imu;
    T_cam1_imu = T_cam1_velo * T_velo_imu;
    T_cam2_imu = T_cam2_velo * T_velo_imu;
    T_cam3_imu = T_cam3_velo * T_velo_imu;
  }

  Sensor::Sensor(std::string name, std::string folder, std::string sensor_suffix) :
    name_(name), folder_(folder), sensor_suffix_(sensor_suffix)
  {
    std::string sensor_timestamp_name( folder_ + std::string("/") + name_ + std::string("/timestamps.txt") );

    std::ifstream f;
    f.open(sensor_timestamp_name.c_str());

    if(!f.is_open()){
      std::cout << "Failed opening: " << sensor_timestamp_name << std::endl;
      return;
    }

    double starting_time = -1.0;
    int line_num = 0;

    while(!f.eof()){
      std::tm t = {};
      std::string sline;
      std::getline(f,sline);
      std::stringstream ssline;
      ssline << sline;
      ssline >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
      if (ssline.fail()){
        break;
      } else {
        // Compute the time relative to the start of the sequence
        double time = double(mktime(&t)) + std::stod(sline.substr(sline.find('.')));
        starting_time = (starting_time<0.0) ? time : starting_time;
        time -= starting_time;

        // Construct the filename
        std::string filename = std::to_string(line_num++);

        filename = std::string(10 - filename.size(), '0') + filename;

        sensor_readings_.push_back(std::make_pair(time, filename));
      }
    }
    sensor_readings_iter_ = sensor_readings_.end();
  }

  bool Sensor::has_next() {
    return sensor_readings_iter_ != (sensor_readings_.end()-1);
  }

  bool Sensor::next() {
    if(!has_next()){
      return false;
    }

    if(get_time() < 0.0 ) {
      sensor_readings_iter_ = sensor_readings_.begin();
    } else {
      ++sensor_readings_iter_;
    }

    return sensor_readings_iter_ != sensor_readings_.end();
  }

  void Sensor::reset() {
    sensor_readings_iter_ = sensor_readings_.end();
  }

  double Sensor::get_time() {
    if(sensor_readings_iter_!=sensor_readings_.end()){
      return sensor_readings_iter_->first;
    }else{
      return -1.0;
    }
  }
  
  std::string Sensor::get_name() {
    return name_;
  }

  Image::Image(std::string name, std::string folder, std::shared_ptr<Calib> calib) : Sensor(name, folder, "png"){
    msckf_mono::Matrix4<float> T_cam_imu;
    if (name == "image_00") {
      K_ = calib->K_cam0;
      T_cam_imu = calib->T_cam0_imu;
    } else if (name == "image_01") {
      K_ = calib->K_cam1;
      T_cam_imu = calib->T_cam1_imu;
    } else if (name == "image_02") {
      K_ = calib->K_cam2;
      T_cam_imu = calib->T_cam2_imu;
    } else if (name == "image_03") {
      K_ = calib->K_cam3;
      T_cam_imu = calib->T_cam3_imu;
    } else {
      assert(false);
    }
    q_BS_ = msckf_mono::Quaternion<float>(T_cam_imu.block<3,3>(0,0));
    p_BS_ = T_cam_imu.block<3,1>(0,3);
  }

  cv::Mat Image::get_data() {
    std::string fn = folder_ + "/" + name_ + "/data/" + sensor_readings_iter_->second + "." + sensor_suffix_;
    cv::Mat img = cv::imread(fn, CV_LOAD_IMAGE_GRAYSCALE);
    if(img.rows == 0 && img.cols == 0){
      std::cout << "Error opening image " << fn << std::endl;
    }
    return img;
  }

  msckf_mono::Matrix3<float> Image::get_K() {
    return K_;
  }

  msckf_mono::Camera<float> Image::get_camera() {
    msckf_mono::Camera<float> cam;
    cam.f_u = K_(0,0);
    cam.f_v = K_(1,1);
    cam.c_u = K_(0,2);
    cam.c_v = K_(1,2);

    cam.q_CI = q_BS_;
    cam.p_C_I = p_BS_;

    return cam;
  }

  OXTS::OXTS(std::string name, std::string folder) : Sensor("oxts", folder, "txt"), oxts_name_(name){
  }

  std::array<double, 30> OXTS::read_oxts_file(std::string filenum=std::string()){
    if(filenum.empty()){
      filenum = sensor_readings_iter_->second;
    }

    std::string fn = folder_ + "/" + name_ + "/data/" + filenum + "." + sensor_suffix_;

    std::ifstream f;
    f.open(fn.c_str());
    std::string line;

    std::array<double, 30> data;
    int i = 0;

    while(std::getline(f, line, ' ')) {
      data[i++] = std::stod(line);
    }

    return data;
  }

  std::string OXTS::get_name(){
    return oxts_name_;
  }

  IMU::IMU(std::string folder) : OXTS("imu", folder) {
  }

  msckf_mono::imuReading<float> IMU::get_data() {
    auto oxts_data = read_oxts_file();

    msckf_mono::imuReading<float> data;

    data.omega[0] = oxts_data[wf];
    data.omega[1] = oxts_data[wl];
    data.omega[2] = oxts_data[wu];

    data.a[0] = oxts_data[af];
    data.a[1] = oxts_data[al];
    data.a[2] = oxts_data[au];

    data.dT = 0.1;

    return data;
  }

  GroundTruth::GroundTruth(std::string folder) : OXTS("gt", folder), has_origin_(false) {
  }

  msckf_mono::imuState<float> GroundTruth::get_data() {
    auto oxts_data = read_oxts_file();

    msckf_mono::imuState<float> data;
    data.p_I_G = msckf_mono::Vector3<float>::Zero();

    double er = 6378137.;
    if(!has_origin_){
      scale_ = cos(oxts_data[lat] * 3.1415926535 / 180.);
    }

    data.p_I_G(0) = scale_ * oxts_data[lon] * 3.1415926535 * er / 180.;
    data.p_I_G(1) = scale_ * er * log(tan((90. + oxts_data[lat]) * 3.1415926535 / 360.));
    data.p_I_G(2) = oxts_data[alt];

    data.q_IG = Eigen::AngleAxisf(oxts_data[roll], Eigen::Vector3f::UnitX())
              * Eigen::AngleAxisf(oxts_data[pitch], Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(oxts_data[yaw], Eigen::Vector3f::UnitZ());

    data.q_IG = data.q_IG.inverse();

    data.v_I_G(0) = oxts_data[vf];
    data.v_I_G(1) = oxts_data[vl];
    data.v_I_G(2) = oxts_data[vu];

    data.v_I_G = data.q_IG.inverse() * data.v_I_G;

    data.b_g = msckf_mono::Vector3<float>::Zero();
    data.b_a = msckf_mono::Vector3<float>::Zero();

    if(!has_origin_){
      origin_.p_I_G = data.p_I_G;
      has_origin_ = true;
    }

    data.p_I_G -= origin_.p_I_G;

    return data;
  }
}
