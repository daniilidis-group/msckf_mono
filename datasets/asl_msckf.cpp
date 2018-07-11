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
#include <msckf_mono/StageTiming.h>

#include <datasets/data_synchronizers.h>
#include <datasets/asl_readers.h>

#include <msckf_mono/msckf.h>

#include <msckf_mono/CamStates.h>

using namespace asl_dataset;
using namespace synchronizer;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  std::string data_set;
  double calib_end;
  if(!nh.getParam("data_set_path", data_set)){
    std::cerr << "Must define a data_set_path" << std::endl;
    return 0;
  }
  if(!nh.getParam("stand_still_end", calib_end)){
    std::cerr << "Must define when the system stops a standstill" << std::endl;
    return 0;
  }

  ROS_INFO_STREAM("Accessing dataset at " << data_set);

  std::shared_ptr<IMU> imu0;
  std::shared_ptr<Camera> cam0;
  std::shared_ptr<GroundTruth> gt0;

  imu0.reset(new IMU("imu0", data_set+"/imu0"));
  cam0.reset(new Camera("cam0", data_set+"/cam0"));
  gt0.reset(new  GroundTruth("state_groundtruth_estimate0", data_set+"/state_groundtruth_estimate0"));

  Synchronizer<IMU, Camera, GroundTruth> sync(imu0, cam0, gt0);

  msckf_mono::MSCKF<float> msckf;

  msckf_mono::Camera<float> camera;
  auto K = cam0->get_K();
  camera.f_u = K.at<float>(0,0);
  camera.f_v = K.at<float>(1,1);
  camera.c_u = K.at<float>(0,2);
  camera.c_v = K.at<float>(1,2);

  camera.q_CI = cam0->get_q_BS();
  const auto q_CI = camera.q_CI;
  camera.p_C_I = cam0->get_p_BS();

  ROS_INFO_STREAM("Camera\n- q_CI " << q_CI.x() << "," << q_CI.y() << "," << q_CI.z() << "," << q_CI.w() << "\n" <<
                  "- p_C_I " << camera.p_C_I.transpose());

  float feature_cov;
  nh.param<float>("feature_covariance", feature_cov, 7);

  msckf_mono::noiseParams<float> noise_params;
  noise_params.u_var_prime = pow(feature_cov/camera.f_u,2);
  noise_params.v_var_prime = pow(feature_cov/camera.f_v,2);

  Eigen::Matrix<float,12,1> Q_imu_vars;
  float w_var, dbg_var, a_var, dba_var;
  nh.param<float>("imu_vars/w_var", w_var, 1e-5);
  nh.param<float>("imu_vars/dbg_var", dbg_var, 3.6733e-5);
  nh.param<float>("imu_vars/a_var", a_var, 1e-3);
  nh.param<float>("imu_vars/dba_var", dba_var, 7e-4);
  Q_imu_vars << w_var, 	w_var, 	w_var,
                dbg_var,dbg_var,dbg_var,
                a_var,	a_var,	a_var,
                dba_var,dba_var,dba_var;
  noise_params.Q_imu = Q_imu_vars.asDiagonal();

  Eigen::Matrix<float,15,1> IMUCovar_vars;
  float q_var_init, bg_var_init, v_var_init, ba_var_init, p_var_init;
  nh.param<float>("imu_covars/q_var_init", q_var_init, 1e-5);
  nh.param<float>("imu_covars/bg_var_init", bg_var_init, 1e-2);
  nh.param<float>("imu_covars/v_var_init", v_var_init, 1e-2);
  nh.param<float>("imu_covars/ba_var_init", ba_var_init, 1e-2);
  nh.param<float>("imu_covars/p_var_init", p_var_init, 1e-12);
  IMUCovar_vars << q_var_init, q_var_init, q_var_init,
                   bg_var_init,bg_var_init,bg_var_init,
                   v_var_init, v_var_init, v_var_init,
                   ba_var_init,ba_var_init,ba_var_init,
                   p_var_init, p_var_init, p_var_init;
  noise_params.initial_imu_covar = IMUCovar_vars.asDiagonal();

  msckf_mono::MSCKFParams<float> msckf_params;
  nh.param<float>("max_gn_cost_norm", msckf_params.max_gn_cost_norm, 11);
  msckf_params.max_gn_cost_norm = pow(msckf_params.max_gn_cost_norm/camera.f_u, 2);
  nh.param<float>("translation_threshold", msckf_params.translation_threshold, 0.05);
  nh.param<float>("min_rcond", msckf_params.min_rcond, 3e-12);

  nh.param<float>("keyframe_transl_dist", msckf_params.redundancy_angle_thresh, 0.005);
  nh.param<float>("keyframe_rot_dist", msckf_params.redundancy_distance_thresh, 0.05);

  int max_tl, min_tl, max_cs;
  nh.param<int>("max_track_length", max_tl, 1000);	// set to inf to wait for features to go out of view
  nh.param<int>("min_track_length", min_tl, 3);		// set to infinity to dead-reckon only
  nh.param<int>("max_cam_states",   max_cs, 20);


  msckf_params.max_track_length = max_tl;
  msckf_params.min_track_length = min_tl;
  msckf_params.max_cam_states = max_cs;

  corner_detector::TrackHandler th(cam0->get_K(), cam0->get_dist_coeffs(), "radtan");

  float ransac_threshold;
  nh.param<float>("ransac_threshold", ransac_threshold, 0.000002);
  th.set_ransac_threshold(ransac_threshold);

  int n_grid_rows, n_grid_cols;
  nh.param<int>("n_grid_rows", n_grid_rows, 8);
  nh.param<int>("n_grid_cols", n_grid_cols, 8);
  th.set_grid_size(n_grid_rows, n_grid_cols);

  int state_k = 0;

  msckf_mono::imuState<float> closest_gt;
  // start from standstill

  while(imu0->get_time()<calib_end && sync.has_next()){
    auto data_pack = sync.get_data();
    auto gt_reading = std::get<2>(data_pack);

    if(gt_reading){
      closest_gt = gt_reading.get();
    }
    sync.next();
  }

  msckf_mono::imuState<float> firstImuState;
  firstImuState.b_a = closest_gt.b_a;
  firstImuState.b_g = closest_gt.b_g;
  firstImuState.g << 0.0, 0.0, -9.81;
  firstImuState.q_IG = closest_gt.q_IG;
  firstImuState.p_I_G = closest_gt.p_I_G; //Eigen::Vector3d::Zero();
  firstImuState.v_I_G = closest_gt.v_I_G; //Eigen::Vector3d::Zero();

  msckf.initialize(camera, noise_params, msckf_params, firstImuState);
  msckf_mono::imuState<float> imu_state = msckf.getImuState();
  msckf_mono::imuReading<float> imu_data = imu0->get_data();
  auto q = imu_state.q_IG;

  ROS_INFO_STREAM("\nInitial IMU State" <<
    "\n---p_I_G " << imu_state.p_I_G.transpose() <<
    "\n---q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.x() <<
    "\n---v_I_G " << imu_state.v_I_G.transpose() <<
    "\n---b_a " << imu_state.b_a.transpose() <<
    "\n---b_g " << imu_state.b_g.transpose() <<
    "\n---a " << imu_data.a.transpose()<<
    "\n---g " << imu_state.g.transpose()<<
    "\n---world_adjusted_a " << (q.toRotationMatrix().transpose()*(imu_data.a-imu_state.b_a)).transpose());
  q = closest_gt.q_IG;
  ROS_INFO_STREAM("Initial GT State" <<
    "\n---p_I_G " << closest_gt.p_I_G.transpose() <<
    "\n---q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.x() <<
    "\n---v_I_G " << closest_gt.v_I_G.transpose() <<
    "\n---b_a " << closest_gt.b_a.transpose() <<
    "\n---b_g " << closest_gt.b_g.transpose());

  ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 100);
  ros::Publisher map_pub = nh.advertise<sensor_msgs::PointCloud2>("map", 100);
  ros::Publisher cam_pose_pub = nh.advertise<geometry_msgs::PoseArray>("cam_state_poses", 100);
  ros::Publisher pruned_cam_states_track_pub = nh.advertise<nav_msgs::Path>("pruned_cam_states_path", 100);
  ros::Publisher cam_state_pub = nh.advertise<msckf_mono::CamStates>("cam_states", 100);

  ros::Publisher imu_track_pub = nh.advertise<nav_msgs::Path>("imu_path", 100);
  nav_msgs::Path imu_path;

  ros::Publisher gt_track_pub = nh.advertise<nav_msgs::Path>("ground_truth_path", 100);
  nav_msgs::Path gt_path;

  ros::Publisher time_state_pub = nh.advertise<msckf_mono::StageTiming>("stage_timing",10);

  image_transport::ImageTransport it(nh);
  image_transport::Publisher raw_img_pub = it.advertise("image", 1);
  image_transport::Publisher track_img_pub = it.advertise("image_track", 1);

  ros::Rate r_imu(1.0/imu0->get_dT());
  ros::Rate r_cam(1.0/cam0->get_dT());

  ros::Time start_clock_time = ros::Time::now();
  ros::Time start_dataset_time;
  start_dataset_time.fromNSec(imu0->get_time());

  while(sync.has_next() && ros::ok()){
    msckf_mono::StageTiming timing_data;
#define TSTART(X) ros::Time start_##X = ros::Time::now();
#define TEND(X) ros::Time end_##X = ros::Time::now();
#define TRECORD(X) {float T = (end_##X-start_##X).toSec();\
                                timing_data.times.push_back(T);\
                                timing_data.stages.push_back(#X);}

    TSTART(get_data);
    auto data_pack = sync.get_data();
    TEND(get_data);
    TRECORD(get_data);

    auto imu_reading = std::get<0>(data_pack);

    auto gt_reading = std::get<2>(data_pack);

    if(gt_reading){
      closest_gt = gt_reading.get();
    }
    if(imu_reading){
      state_k++;

      TSTART(imu_prop);
      msckf_mono::imuReading<float> imu_data = imu_reading.get();
      msckf_mono::imuState<float> prev_imu_state = msckf.getImuState();
      msckf_mono::Quaternion<float> prev_rotation = prev_imu_state.q_IG;
      msckf.propagate(imu_data);

      Eigen::Vector3f cam_frame_av = (camera.q_CI.inverse() * (imu_data.omega-prev_imu_state.b_g));
      th.add_gyro_reading(cam_frame_av);
      TEND(imu_prop);
      TRECORD(imu_prop);

      if(std::get<1>(data_pack)){
        ros::Time cur_clock_time = ros::Time::now();
        ros::Time cur_dataset_time;
        cur_dataset_time.fromNSec(imu0->get_time());

        float elapsed_dataset_time = (cur_dataset_time - start_dataset_time).toSec();
        float elapsed_clock_time = (cur_clock_time - start_clock_time).toSec();


        TSTART(feature_tracking_and_warping);
        cv::Mat img = std::get<1>(data_pack).get();

        th.set_current_image(img, ((float)cam0->get_time())/1e9);

        std::vector<msckf_mono::Vector2<float>, Eigen::aligned_allocator<msckf_mono::Vector2<float>>> cur_features;
        corner_detector::IdVector cur_ids;
        th.tracked_features(cur_features, cur_ids);

        std::vector<msckf_mono::Vector2<float>, Eigen::aligned_allocator<msckf_mono::Vector2<float>>> new_features;
        corner_detector::IdVector new_ids;
        th.new_features(new_features, new_ids);

        if(elapsed_clock_time > elapsed_dataset_time){
          ROS_ERROR("skipping frame");
        }else{
          TEND(feature_tracking_and_warping);
          TRECORD(feature_tracking_and_warping);

          TSTART(msckf_augment_state);
          msckf.augmentState(state_k, ((float)imu0->get_time())/1e9);
          TEND(msckf_augment_state);
          TRECORD(msckf_augment_state);

          TSTART(msckf_update);
          msckf.update(cur_features, cur_ids);
          TEND(msckf_update);
          TRECORD(msckf_update);

          TSTART(msckf_add_features);
          msckf.addFeatures(new_features, new_ids);
          TEND(msckf_add_features);
          TRECORD(msckf_add_features);

          TSTART(msckf_marginalize);
          msckf.marginalize();
          TEND(msckf_marginalize);
          TRECORD(msckf_marginalize);

          TSTART(msckf_prune_redundant);
          msckf.pruneRedundantStates();
          TEND(msckf_prune_redundant);
          TRECORD(msckf_prune_redundant);

          TSTART(msckf_prune_empty_states);
          msckf.pruneEmptyStates();
          TEND(msckf_prune_empty_states);
          TRECORD(msckf_prune_empty_states);

          auto imu_state = msckf.getImuState();
          auto q = imu_state.q_IG;

          TSTART(publishing);
          ros::Time cur_ros_time;
          cur_ros_time.fromNSec(cam0->get_time());
          {
            nav_msgs::Odometry odom;
            odom.header.stamp = cur_ros_time;
            odom.header.frame_id = "map";
            odom.pose.pose.position.x = imu_state.p_I_G[0];
            odom.pose.pose.position.y = imu_state.p_I_G[1];
            odom.pose.pose.position.z = imu_state.p_I_G[2];
            msckf_mono::Quaternion<float> q_out = imu_state.q_IG.inverse();
            odom.pose.pose.orientation.w = q_out.w();
            odom.pose.pose.orientation.x = q_out.x();
            odom.pose.pose.orientation.y = q_out.y();
            odom.pose.pose.orientation.z = q_out.z();
            odom_pub.publish(odom);
          }

          if(raw_img_pub.getNumSubscribers()>0){
            cv_bridge::CvImage out_img;
            out_img.header.frame_id = "cam0"; // Same timestamp and tf frame as input image
            out_img.header.stamp = cur_ros_time;
            out_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1; // Or whatever
            out_img.image    = img; // Your cv::Mat
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

          if(map_pub.getNumSubscribers()>0){
            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> map =
              msckf.getMap();
            pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
            pointcloud->header.frame_id = "map";
            pointcloud->height = 1;
            for (auto& point:map)
            {
              pointcloud->points.push_back(pcl::PointXYZ(point(0),
                    point(1),
                    point(2)));
            }

            pointcloud->width = pointcloud->points.size();
            map_pub.publish(pointcloud);
          }

          if(cam_pose_pub.getNumSubscribers()>0){
            geometry_msgs::PoseArray cam_poses;

            auto msckf_cam_poses = msckf.getCamStates();
            for( auto& cs : msckf_cam_poses ){
              geometry_msgs::Pose p;
              p.position.x = cs.p_C_G[0];
              p.position.y = cs.p_C_G[1];
              p.position.z = cs.p_C_G[2];
              msckf_mono::Quaternion<float> q_out = cs.q_CG.inverse();
              p.orientation.w = q_out.w();
              p.orientation.x = q_out.x();
              p.orientation.y = q_out.y();
              p.orientation.z = q_out.z();
              cam_poses.poses.push_back(p);
            }

            cam_poses.header.frame_id = "map";
            cam_poses.header.stamp = cur_ros_time;

            cam_pose_pub.publish(cam_poses);
          }

          if(cam_state_pub.getNumSubscribers()>0){
            msckf_mono::CamStates cam_states;

            auto msckf_cam_states = msckf.getCamStates();
            for( auto& cs : msckf_cam_states ){
              msckf_mono::CamState ros_cs;

              ros_cs.stamp.fromSec(cs.time);
              ros_cs.id = cs.state_id;

              ros_cs.number_tracked_features = cs.tracked_feature_ids.size();

              auto& p = ros_cs.pose;
              p.position.x = cs.p_C_G[0];
              p.position.y = cs.p_C_G[1];
              p.position.z = cs.p_C_G[2];
              msckf_mono::Quaternion<float> q_out = cs.q_CG.inverse();
              p.orientation.w = q_out.w();
              p.orientation.x = q_out.x();
              p.orientation.y = q_out.y();
              p.orientation.z = q_out.z();

              cam_states.cam_states.push_back(ros_cs);
            }

            cam_state_pub.publish(cam_states);
          }

          if(pruned_cam_states_track_pub.getNumSubscribers()>0){
            nav_msgs::Path pruned_path;
            pruned_path.header.stamp = cur_ros_time;
            pruned_path.header.frame_id = "map";
            for(auto ci : msckf.getPrunedStates()){
              geometry_msgs::PoseStamped ps;

              ps.header.stamp.fromNSec(ci.time);
              ps.header.frame_id = "map";

              ps.pose.position.x = ci.p_C_G[0];
              ps.pose.position.y = ci.p_C_G[1];
              ps.pose.position.z = ci.p_C_G[2];
              msckf_mono::Quaternion<float> q_out = ci.q_CG.inverse();
              ps.pose.orientation.w = q_out.w();
              ps.pose.orientation.x = q_out.x();
              ps.pose.orientation.y = q_out.y();
              ps.pose.orientation.z = q_out.z();

              pruned_path.poses.push_back(ps);
            }

            pruned_cam_states_track_pub.publish(pruned_path);
          }

          {
            gt_path.header.stamp = cur_ros_time;
            gt_path.header.frame_id = "map";
            geometry_msgs::PoseStamped gt_pose;
            gt_pose.header = gt_path.header;
            gt_pose.pose.position.x = closest_gt.p_I_G[0];
            gt_pose.pose.position.y = closest_gt.p_I_G[1];
            gt_pose.pose.position.z = closest_gt.p_I_G[2];
            msckf_mono::Quaternion<float> q_out = closest_gt.q_IG.inverse();
            gt_pose.pose.orientation.w = q_out.w();
            gt_pose.pose.orientation.x = q_out.x();
            gt_pose.pose.orientation.y = q_out.y();
            gt_pose.pose.orientation.z = q_out.z();

            gt_path.poses.push_back(gt_pose);

            gt_track_pub.publish(gt_path);
          }

          {
            imu_path.header.stamp = cur_ros_time;
            imu_path.header.frame_id = "map";
            geometry_msgs::PoseStamped imu_pose;
            imu_pose.header = imu_path.header;
            imu_pose.pose.position.x = imu_state.p_I_G[0];
            imu_pose.pose.position.y = imu_state.p_I_G[1];
            imu_pose.pose.position.z = imu_state.p_I_G[2];
            msckf_mono::Quaternion<float> q_out = imu_state.q_IG.inverse();
            imu_pose.pose.orientation.w = q_out.w();
            imu_pose.pose.orientation.x = q_out.x();
            imu_pose.pose.orientation.y = q_out.y();
            imu_pose.pose.orientation.z = q_out.z();

            imu_path.poses.push_back(imu_pose);

            imu_track_pub.publish(imu_path);
          }
          TEND(publishing);
          TRECORD(publishing);

          time_state_pub.publish(timing_data);

          r_cam.sleep();
        }
      }
    }

    sync.next();
  }
}
