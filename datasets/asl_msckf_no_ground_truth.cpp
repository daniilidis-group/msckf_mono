#include <iostream>
#include <string>

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
#include <datasets/asl_readers.h>

#include <msckf_mono/msckf.h>

#include <msckf_mono/CamStates.h>
#include <msckf_mono/ros_interface.h>

using namespace asl_dataset;
using namespace synchronizer;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  std::string data_set;
  double calib_end, calib_start;
  if(!nh.getParam("data_set_path", data_set)){
    std::cerr << "Must define a data_set_path" << std::endl;
    return 0;
  }
  if(!nh.getParam("stand_still_start", calib_start)){
    std::cerr << "Must define when the system starts a standstill" << std::endl;
    return 0;
  }
  if(!nh.getParam("stand_still_end", calib_end)){
    std::cerr << "Must define when the system stops a standstill" << std::endl;
    return 0;
  }

  ROS_INFO_STREAM("Accessing dataset at " << data_set);

  std::shared_ptr<IMU> imu0;
  std::shared_ptr<Camera> cam0;

  imu0.reset(new IMU("imu0", data_set+"/imu0"));
  cam0.reset(new Camera("cam0", data_set+"/cam0"));

  Synchronizer<IMU, Camera> sync(imu0, cam0);

  std::shared_ptr<msckf_mono::MSCKF<float>> msckf(new msckf_mono::MSCKF<float>());

  msckf_mono::Camera<float> camera;
  auto K = cam0->get_K();
  camera.f_u = K.at<float>(0,0);
  camera.f_v = K.at<float>(1,1);
  camera.c_u = K.at<float>(0,2);
  camera.c_v = K.at<float>(1,2);

  camera.q_CI = cam0->get_q_BS();
  camera.p_C_I = cam0->get_p_BS();

  msckf_mono::noiseParams<float> noise_params;
  msckf_mono::MSCKFParams<float> msckf_params;
  std::tie(noise_params, msckf_params) = msckf_mono::fetch_params<float>(nh, camera.f_u, camera.f_v);

  std::shared_ptr<corner_detector::TrackHandler> track_handler;
  track_handler.reset(new corner_detector::TrackHandler(cam0->get_K(), cam0->get_dist_coeffs(), "radtan"));

  float ransac_threshold;
  nh.param<float>("ransac_threshold", ransac_threshold, 0.000002);
  track_handler->set_ransac_threshold(ransac_threshold);

  int n_grid_rows, n_grid_cols;
  nh.param<int>("n_grid_rows", n_grid_rows, 8);
  nh.param<int>("n_grid_cols", n_grid_cols, 8);
  track_handler->set_grid_size(n_grid_rows, n_grid_cols);

  int state_k = 0;

  // start from standstill
  while(imu0->get_time()<calib_start && sync.has_next()){
    sync.next();
  }

  Eigen::Vector3f accel_accum;
  Eigen::Vector3f gyro_accum;
  int num_readings = 0;

  accel_accum.setZero();
  gyro_accum.setZero();

  while(imu0->get_time()<calib_end && sync.has_next()){
    auto data_pack = sync.get_data();
    auto imu_reading = std::get<0>(data_pack);

    if(imu_reading){
      msckf_mono::imuReading<float> imu_data = imu_reading.get();
      accel_accum += imu_data.a;
      gyro_accum += imu_data.omega;
      num_readings++;
    }

    sync.next();
  }

  Eigen::Vector3f accel_mean = accel_accum / num_readings;
  Eigen::Vector3f gyro_mean = gyro_accum / num_readings;

  msckf_mono::imuState<float> firstImuState;
  firstImuState.b_g = gyro_mean;
  firstImuState.g << 0.0, 0.0, -9.81;
  firstImuState.q_IG = Eigen::Quaternionf::FromTwoVectors(-firstImuState.g, accel_mean);

  firstImuState.b_a = firstImuState.q_IG*firstImuState.g + accel_mean;

  firstImuState.p_I_G.setZero();
  firstImuState.v_I_G.setZero();

  msckf->initialize(camera, noise_params, msckf_params, firstImuState);
  msckf_mono::imuState<float> imu_state = msckf->getImuState();
  msckf_mono::imuReading<float> imu_data = imu0->get_data();
  auto q = imu_state.q_IG;

  ROS_INFO_STREAM("\nInitial IMU State" <<
    "\n--p_I_G " << imu_state.p_I_G.transpose() <<
    "\n--q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.x() <<
    "\n--v_I_G " << imu_state.v_I_G.transpose() <<
    "\n--b_a " << imu_state.b_a.transpose() <<
    "\n--b_g " << imu_state.b_g.transpose() <<
    "\n--a " << imu_data.a.transpose()<<
    "\n--g " << imu_state.g.transpose()<<
    "\n--world_adjusted_a " << (q.toRotationMatrix().transpose()*(imu_data.a-imu_state.b_a)).transpose());

  ros::Rate r_imu(1.0/imu0->get_dT());
  ros::Rate r_cam(1.0/cam0->get_dT());

  ros::Time start_clock_time = ros::Time::now();
  ros::Time start_dataset_time;
  start_dataset_time.fromNSec(imu0->get_time());

  std::shared_ptr<msckf_mono::ROSOutput<float>> ros_output_manager;
  ros_output_manager.reset(new msckf_mono::ROSOutput<float>(nh));
  ros_output_manager->setTrackHandler(track_handler);
  ros_output_manager->setMSCKF(msckf);

  while(sync.has_next() && ros::ok()){
    auto data_pack = sync.get_data();

    auto imu_reading = std::get<0>(data_pack);

    if(imu_reading){
      state_k++;

      msckf_mono::imuReading<float> imu_data = imu_reading.get();
      msckf_mono::imuState<float> prev_imu_state = msckf->getImuState();
      msckf_mono::Quaternion<float> prev_rotation = prev_imu_state.q_IG;
      msckf->propagate(imu_data);

      Eigen::Vector3f cam_frame_av = (camera.q_CI.inverse() * (imu_data.omega-prev_imu_state.b_g));
      track_handler->add_gyro_reading(cam_frame_av);

      if(std::get<1>(data_pack)){
        ros::Time cur_clock_time = ros::Time::now();
        ros::Time cur_dataset_time;
        cur_dataset_time.fromNSec(imu0->get_time());

        float elapsed_dataset_time = (cur_dataset_time - start_dataset_time).toSec();
        float elapsed_clock_time = (cur_clock_time - start_clock_time).toSec();

        cv::Mat img = std::get<1>(data_pack).get();

        track_handler->set_current_image(img, ((float)cam0->get_time())/1e9);

        std::vector<msckf_mono::Vector2<float>, Eigen::aligned_allocator<msckf_mono::Vector2<float>>> cur_features;
        corner_detector::IdVector cur_ids;
        track_handler->tracked_features(cur_features, cur_ids);

        std::vector<msckf_mono::Vector2<float>, Eigen::aligned_allocator<msckf_mono::Vector2<float>>> new_features;
        corner_detector::IdVector new_ids;
        track_handler->new_features(new_features, new_ids);

        {
          msckf->augmentState(state_k, ((float)imu0->get_time())/1e9);
          msckf->update(cur_features, cur_ids);
          msckf->addFeatures(new_features, new_ids);
          msckf->marginalize();
          msckf->pruneRedundantStates();
          msckf->pruneEmptyStates();

          auto imu_state = msckf->getImuState();
          auto q = imu_state.q_IG;

          ros::Time cur_ros_time;
          cur_ros_time.fromNSec(cam0->get_time());


          ros_output_manager->publishImage(cur_ros_time);
          ros_output_manager->publishTracks(cur_ros_time);
          ros_output_manager->publishCamPoses(cur_ros_time);
          ros_output_manager->publishPrunedCamStates(cur_ros_time);
          ros_output_manager->publishMap(cur_ros_time);
          ros_output_manager->publishOdom(cur_ros_time);

          r_cam.sleep();
        }
      }
    }

    sync.next();
  }
}
