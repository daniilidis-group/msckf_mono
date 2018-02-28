#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <msckf_mono/corner_detector.h>

#include <datasets/data_synchronizers.h>
#include <datasets/penncosy_readers.h>
#include <msckf_mono/msckf.h>

#include <msckf_mono/CamStates.h>


using namespace penncosy_dataset;
using namespace synchronizer;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;


  std::shared_ptr<VI_IMU> imu0;
  std::shared_ptr<VICamera> cam0;

  std::string seq = "af";
  std::string path = "/home/ken/datasets/penncosyvio";

  imu0.reset(new VI_IMU(path, "visensor", seq));
  cam0.reset(new VICamera(path, "visensor", seq, "timestamps_cameras.txt", "right"));
  ros::Rate r_cam(0.25*1.0/cam0->get_dT());

  auto sync = make_synchronizer(imu0, cam0);

  //msckf_mono::MSCKF msckf;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher raw_img_pub = it.advertise("image", 1);
  image_transport::Publisher track_img_pub = it.advertise("image_track", 1);

  msckf_mono::Camera camera;
  auto K = cam0->get_K();
  camera.f_u = K.at<float>(0,0);
  camera.f_v = K.at<float>(1,1);
  camera.c_u = K.at<float>(0,2);
  camera.c_v = K.at<float>(1,2);

  //camera.q_CI = cam0->get_q_BS();
  //camera.p_C_I = cam0->get_p_BS();

  ////camera.q_CI = cam0->get_q_BS();
  ////camera.p_C_I = cam0->get_p_BS();

  //double feature_cov;
  //nh.param<double>("feature_covariance", feature_cov, 7);

  //msckf_mono::noiseParams noise_params;
  //noise_params.u_var_prime = pow(feature_cov/camera.f_u,2);
  //noise_params.v_var_prime = pow(feature_cov/camera.f_v,2);

  //Eigen::Matrix<double,12,1> Q_imu_vars;
  //double w_var, dbg_var, a_var, dba_var;
  //nh.param<double>("imu_vars/w_var", w_var, 1e-5);
  //nh.param<double>("imu_vars/dbg_var", dbg_var, 3.6733e-5);
  //nh.param<double>("imu_vars/a_var", a_var, 1e-3);
  //nh.param<double>("imu_vars/dba_var", dba_var, 7e-4);
  //Q_imu_vars << w_var, 	w_var, 	w_var,
  //              dbg_var,dbg_var,dbg_var,
  //              a_var,	a_var,	a_var,
  //              dba_var,dba_var,dba_var;
  //noise_params.Q_imu = Q_imu_vars.asDiagonal();

  //Eigen::Matrix<double,15,1> IMUCovar_vars;
  //double q_var_init, bg_var_init, v_var_init, ba_var_init, p_var_init;
  //nh.param<double>("imu_covars/q_var_init", q_var_init, 1e-5);
  //nh.param<double>("imu_covars/bg_var_init", bg_var_init, 1e-2);
  //nh.param<double>("imu_covars/v_var_init", v_var_init, 1e-2);
  //nh.param<double>("imu_covars/ba_var_init", ba_var_init, 1e-2);
  //nh.param<double>("imu_covars/p_var_init", p_var_init, 1e-12);
  //IMUCovar_vars << q_var_init, q_var_init, q_var_init,
  //                 bg_var_init,bg_var_init,bg_var_init,
  //                 v_var_init, v_var_init, v_var_init,
  //                 ba_var_init,ba_var_init,ba_var_init,
  //                 p_var_init, p_var_init, p_var_init;
  //noise_params.initial_imu_covar = IMUCovar_vars.asDiagonal();

  //msckf_mono::MSCKFParams msckf_params;
  //nh.param<double>("max_gn_cost_norm", msckf_params.max_gn_cost_norm, 11);
  //msckf_params.max_gn_cost_norm = pow(msckf_params.max_gn_cost_norm/camera.f_u, 2);
  //nh.param<double>("translation_threshold", msckf_params.translation_threshold, 0.05);
  //nh.param<double>("min_rcond", msckf_params.min_rcond, 3e-12);

  //nh.param<double>("keyframe_transl_dist", msckf_params.redundancy_angle_thresh, 0.005);
  //nh.param<double>("keyframe_rot_dist", msckf_params.redundancy_distance_thresh, 0.05);

  //int max_tl, min_tl, max_cs;
  //nh.param<int>("max_track_length", max_tl, 1000);	// set to inf to wait for features to go out of view
  //nh.param<int>("min_track_length", min_tl, 3);		// set to infinity to dead-reckon only
  //nh.param<int>("max_cam_states",   max_cs, 20);


  //msckf_params.max_track_length = max_tl;
  //msckf_params.min_track_length = min_tl;
  //msckf_params.max_cam_states = max_cs;

  corner_detector::TrackHandler th(cam0->get_K());

  //double ransac_threshold;
  //nh.param<double>("ransac_threshold", ransac_threshold, 0.000002);

  int n_grid_rows, n_grid_cols;
  nh.param<int>("n_grid_rows", n_grid_rows, 8);
  nh.param<int>("n_grid_cols", n_grid_cols, 8);
  th.set_grid_size(n_grid_rows, n_grid_cols);

  //int state_k = 0;
  
  Eigen::Quaterniond q_CI = cam0->get_q_BS() * imu0->get_q_BS().inverse();

  while(sync.has_next() && ros::ok()){
    auto data_pack = sync.get_data();
    auto imu_reading = std::get<0>(data_pack);
    auto image_reading = std::get<1>(data_pack);

    if(imu_reading){
      ros::Time cur_ros_time;
      cur_ros_time.fromNSec(imu0->get_time());

      auto imu_data = imu_reading.get();

      Eigen::Vector3d cam_frame_av = q_CI.inverse() * imu_data.omega;//Eigen::Vector3d::Zero(); //
      th.add_gyro_reading(cam_frame_av);

      if (image_reading){
        corner_detector::Point2fVector points;
        corner_detector::IdVector ids;
        cv::Mat img = image_reading.get();

        auto boundry_id = th.get_next_feature_id();
        th.track_features(img, points, ids, ((double)cam0->get_time()));

        if(raw_img_pub.getNumSubscribers()>0){
          cv_bridge::CvImage out_img;
          out_img.header.frame_id = "cam0"; // Same timestamp and tf frame as input image
          out_img.header.stamp = cur_ros_time;
          out_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1; // Or whatever
          out_img.image = img; // Your cv::Mat
          raw_img_pub.publish(out_img.toImageMsg());
        }

        if(track_img_pub.getNumSubscribers()>0){
          cv_bridge::CvImage out_img;
          out_img.header.frame_id = "cam0"; // Same timestamp and tf frame as input image
          out_img.header.stamp = cur_ros_time;
          out_img.encoding = sensor_msgs::image_encodings::TYPE_8UC3; // Or whatever
          out_img.image = th.get_track_image(); // Your cv::Mat
          track_img_pub.publish(out_img.toImageMsg());
        }
        r_cam.sleep();
      }
    }

    sync.next();
  }
}
