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
  double calib_start;
  double calib_end;
  if(!nh.getParam("data_set_path", data_set)){
    std::cerr << "Must define a data_set_path" << std::endl;
    return 0;
  }
  if(!nh.getParam("stand_still_end", calib_end)){
    std::cerr << "Must define when the system stops a standstill" << std::endl;
    return 0;
  }

  std::cout << "Accessing dataset at " << data_set << std::endl;

  std::shared_ptr<IMU> imu0;
  std::shared_ptr<Camera> cam0;
  std::shared_ptr<Camera> cam1;
  std::shared_ptr<GroundTruth> gt0;

  imu0.reset(new IMU("imu0", data_set+"/imu0"));
  cam0.reset(new Camera("cam0", data_set+"/cam0"));
  cam1.reset(new Camera("cam1", data_set+"/cam1"));
  gt0.reset(new  GroundTruth("state_groundtruth_estimate0", data_set+"/state_groundtruth_estimate0"));

  Synchronizer<IMU, Camera, Camera, GroundTruth> sync(imu0, cam0, cam1, gt0);

  msckf::MSCKF msckf;

  msckf::Camera camera;
  auto K = cam0->get_K();
  camera.f_u = K.at<float>(0,0);
  camera.f_v = K.at<float>(1,1);
  camera.c_u = K.at<float>(0,2);
  camera.c_v = K.at<float>(1,2);

  camera.q_CI = cam0->get_q_BS();
  camera.p_C_I = cam0->get_p_BS();

  double feature_cov;
  nh.param<double>("feature_covariance", feature_cov, 7);

  msckf::noiseParams noise_params;
  noise_params.u_var_prime = pow(feature_cov/camera.f_u,2);
  noise_params.v_var_prime = pow(feature_cov/camera.f_v,2);

  Eigen::Matrix<double,12,1> Q_imu_vars;
  double w_var, dbg_var, a_var, dba_var;
  nh.param<double>("imu_vars/w_var", w_var, 1e-5);
  nh.param<double>("imu_vars/dbg_var", dbg_var, 3.6733e-5);
  nh.param<double>("imu_vars/a_var", a_var, 1e-3);
  nh.param<double>("imu_vars/dba_var", dba_var, 7e-4);
  Q_imu_vars << w_var, 	w_var, 	w_var,
                dbg_var,dbg_var,dbg_var,
                a_var,	a_var,	a_var,
                dba_var,dba_var,dba_var;
  noise_params.Q_imu = Q_imu_vars.asDiagonal();

  Eigen::Matrix<double,15,1> IMUCovar_vars;
  double q_var_init, bg_var_init, v_var_init, ba_var_init, p_var_init;
  nh.param<double>("imu_covars/q_var_init", q_var_init, 1e-5);
  nh.param<double>("imu_covars/bg_var_init", bg_var_init, 1e-2);
  nh.param<double>("imu_covars/v_var_init", v_var_init, 1e-2);
  nh.param<double>("imu_covars/ba_var_init", ba_var_init, 1e-2);
  nh.param<double>("imu_covars/p_var_init", p_var_init, 1e-12);
  IMUCovar_vars << q_var_init, q_var_init, q_var_init,
                   bg_var_init,bg_var_init,bg_var_init,
                   v_var_init, v_var_init, v_var_init,
                   ba_var_init,ba_var_init,ba_var_init,
                   p_var_init, p_var_init, p_var_init;
  noise_params.initial_imu_covar = IMUCovar_vars.asDiagonal();

  msckf::MSCKFParams msckf_params;
  nh.param<double>("max_gn_cost_norm", msckf_params.max_gn_cost_norm, 11);
  msckf_params.max_gn_cost_norm = pow(msckf_params.max_gn_cost_norm/camera.f_u, 2);
  nh.param<double>("translation_threshold", msckf_params.translation_threshold, 0.05);
  nh.param<double>("min_rcond", msckf_params.min_rcond, 3e-12);

  nh.param<double>("keyframe_transl_dist", msckf_params.redundancy_angle_thresh, 0.005);
  nh.param<double>("keyframe_rot_dist", msckf_params.redundancy_distance_thresh, 0.05);

  int max_tl, min_tl, max_cs;
  nh.param<int>("max_track_length", max_tl, 1000);	// set to inf to wait for features to go out of view
  nh.param<int>("min_track_length", min_tl, 3);		// set to infinity to dead-reckon only
  nh.param<int>("max_cam_states",   max_cs, 20);


  msckf_params.max_track_length = max_tl;
  msckf_params.min_track_length = min_tl;
  msckf_params.max_cam_states = max_cs;


  corner_detector::TrackHandler th(cam0->get_K());

  double ransac_threshold;
  nh.param<double>("ransac_threshold", ransac_threshold, 0.000002);

  int n_grid_rows, n_grid_cols;
  nh.param<int>("n_grid_rows", n_grid_rows, 8);
  nh.param<int>("n_grid_cols", n_grid_cols, 8);
  th.set_grid_size(n_grid_rows, n_grid_cols);

  int state_k = 0;

  msckf::imuState closest_gt;
  // start from standstill

  while(imu0->get_time()<calib_end && sync.has_next()){
    auto data_pack = sync.get_data();
    auto gt_reading = std::get<3>(data_pack);

    if(gt_reading){
      closest_gt = gt_reading.get();
    }
    sync.next();
  }

  msckf::imuState firstImuState;
  firstImuState.b_a = closest_gt.b_a;
  firstImuState.b_g = closest_gt.b_g;
  firstImuState.g << 0.0, 0.0, -9.81;
  firstImuState.q_IG = closest_gt.q_IG;
  firstImuState.p_I_G = closest_gt.p_I_G; //Eigen::Vector3d::Zero();
  firstImuState.v_I_G = closest_gt.v_I_G; //Eigen::Vector3d::Zero();

  msckf.initialize(camera, noise_params, msckf_params, firstImuState);
  auto imu_state = msckf.getImuState();
  auto imu_data = imu0->get_data();
  auto& q = imu_state.q_IG;

  std::cout << "\nInitial IMU State" <<
    "\n---p_I_G " << imu_state.p_I_G.transpose() <<
    "\n---q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.x() <<
    "\n---v_I_G " << imu_state.v_I_G.transpose() <<
    "\n---b_a " << imu_state.b_a.transpose() <<
    "\n---b_g " << imu_state.b_g.transpose() <<
    "\n---a " << imu_data.a.transpose()<<
    "\n---g " << imu_state.g.transpose()<<
    "\n---world_adjusted_a " << (q.toRotationMatrix().transpose()*(imu_data.a-imu_state.b_a)).transpose();
  q = closest_gt.q_IG;
  std::cout << "\nInitial GT State" <<
    "\n---p_I_G " << closest_gt.p_I_G.transpose() <<
    "\n---q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.x() <<
    "\n---v_I_G " << closest_gt.v_I_G.transpose() <<
    "\n---b_a " << closest_gt.b_a.transpose() <<
    "\n---b_g " << closest_gt.b_g.transpose() << std::endl;

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
#define TRECORD(X) {double T = (end_##X-start_##X).toSec();\
                                timing_data.times.push_back(T);\
                                timing_data.stages.push_back(#X);}

    TSTART(get_data);
    auto data_pack = sync.get_data();
    TEND(get_data);
    TRECORD(get_data);

    auto imu_reading = std::get<0>(data_pack);

    auto gt_reading = std::get<3>(data_pack);

    if(gt_reading){
      closest_gt = gt_reading.get();
    }
    if(imu_reading){
      state_k++;

      TSTART(imu_prop);
      auto imu_data = imu_reading.get();
      auto prev_imu_state = msckf.getImuState();
      Eigen::Quaterniond prev_rotation = prev_imu_state.q_IG;
      msckf.propagate(imu_data);

      Eigen::Vector3d cam_frame_av = camera.q_CI * (imu_data.omega-prev_imu_state.b_g);
      th.add_gyro_reading(cam_frame_av);
      TEND(imu_prop);
      TRECORD(imu_prop);

      if(std::get<1>(data_pack) && std::get<2>(data_pack)){
        ros::Time cur_clock_time = ros::Time::now();
        ros::Time cur_dataset_time;
        cur_dataset_time.fromNSec(imu0->get_time());

        double elapsed_dataset_time = (cur_dataset_time - start_dataset_time).toSec();
        double elapsed_clock_time = (cur_clock_time - start_clock_time).toSec();


        TSTART(feature_tracking_and_warping);
        corner_detector::Point2fVector points;
        corner_detector::IdVector ids;
        cv::Mat img = std::get<1>(data_pack).get();

        auto boundry_id = th.get_next_feature_id();
        th.track_features(img, points, ids, ((double)cam0->get_time())/1e9);

        if(elapsed_clock_time > elapsed_dataset_time){
          std::cout << "skipping frame" << std::endl;
        }else{

          corner_detector::Point2fVector undistorted_pts;
          cv::undistortPoints(points, undistorted_pts, cam0->get_K(), cam0->get_dist_coeffs());

          std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> updated_features;
          std::vector<size_t> updated_ids;
          std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> new_features;
          std::vector<size_t> new_ids;
          TEND(feature_tracking_and_warping);
          TRECORD(feature_tracking_and_warping);

          TSTART(feature_ransac);
          auto last_updated_id = std::find(ids.begin(),
              ids.end(),
              boundry_id);
          int n_updated = std::distance(ids.begin(), last_updated_id);
          int n_new = ids.size() - n_updated;

          updated_features.reserve(n_updated);
          updated_ids.reserve(n_updated);
          new_features.reserve(n_new);
          new_ids.reserve(n_new);

          for(int i=0; i<undistorted_pts.size(); i++){
            Eigen::Vector2d feature = { undistorted_pts[i].x, undistorted_pts[i].y };
            size_t id = ids[i];
            if(ids[i]<boundry_id){
              updated_features.push_back(feature);
              updated_ids.push_back(id);
            }else{
              new_features.push_back(feature);
              new_ids.push_back(id);
            }
          }

          // Get feature positions of active features from the last iteration for RANSAC.
          corner_detector::Point2fVector undistorted_prev_pts;
          cv::undistortPoints(th.get_prev_features(),
              undistorted_prev_pts,
              cam0->get_K(),
              cam0->get_dist_coeffs());
          corner_detector::IdVector prev_ids = th.get_prev_ids();

          std::vector<Eigen::Vector2d,
            Eigen::aligned_allocator<Eigen::Vector2d>> remaining_prev_pts;
          remaining_prev_pts.reserve(n_updated);
          auto prev_pt_iter = undistorted_prev_pts.begin();

          for (int i=0; i < static_cast<int>(undistorted_prev_pts.size()); ++i) {
            size_t id = prev_ids[i];
            if (std::find(updated_ids.begin(),
                  updated_ids.end(),
                  prev_ids[i])
                != updated_ids.end()) {
              Eigen::Vector2d pt = {undistorted_prev_pts[i].x, undistorted_prev_pts[i].y};
              remaining_prev_pts.push_back(pt);
            }
          }

          auto curr_imu_state = msckf.getImuState();
          Eigen::Quaterniond curr_rotation = curr_imu_state.q_IG;
          Eigen::Matrix3d dR = (curr_rotation * prev_rotation.inverse()).toRotationMatrix();

          Eigen::Array<bool, Eigen::Dynamic, 1> inliers = th.twoPointRansac(
              dR,
              remaining_prev_pts,
              updated_features);
          int n_inliers = inliers.count();

          std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> final_features;
          final_features.reserve(n_inliers);
          corner_detector::IdVector final_ids;
          final_ids.reserve(n_inliers);

          for (int i=0; i < inliers.size(); ++i) {
            if (inliers(i)) {
              final_features.push_back(updated_features[i]);
              final_ids.push_back(updated_ids[i]);
            }
          }
          TEND(feature_ransac);
          TRECORD(feature_ransac);

          TSTART(msckf_augment_state);
          msckf.augmentState(state_k, ((double)imu0->get_time())/1e9);
          TEND(msckf_augment_state);
          TRECORD(msckf_augment_state);

          TSTART(msckf_update);
          msckf.update(final_features, final_ids);
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
            Eigen::Quaterniond q_out = imu_state.q_IG.inverse();
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
            std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> map =
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
              Eigen::Quaterniond q_out = cs.q_CG.inverse();
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
              Eigen::Quaterniond q_out = cs.q_CG.inverse();
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
            //std::cout << "pruned states size " << msckf.getPrunedStates().size() << std::endl;
            for(auto ci : msckf.getPrunedStates()){
              geometry_msgs::PoseStamped ps;

              ps.header.stamp.fromNSec(ci.time);
              ps.header.frame_id = "map";

              ps.pose.position.x = ci.p_C_G[0];
              ps.pose.position.y = ci.p_C_G[1];
              ps.pose.position.z = ci.p_C_G[2];
              Eigen::Quaterniond q_out = ci.q_CG.inverse();
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
            Eigen::Quaterniond q_out = closest_gt.q_IG.inverse();
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
            Eigen::Quaterniond q_out = imu_state.q_IG.inverse();
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
