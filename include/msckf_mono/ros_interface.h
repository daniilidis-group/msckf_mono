#ifndef MSCKF_MONO_ROS_INTERFACE_H_
#define MSCKF_MONO_ROS_INTERFACE_H_

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>

#include <msckf_mono/CamStates.h>
#include <msckf_mono/StageTiming.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <msckf_mono/types.h>
#include <msckf_mono/msckf.h>
#include <msckf_mono/corner_detector.h>
#include <atomic>

namespace msckf_mono
{
  template<typename _S>
    class ROSOutput {
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ROSOutput(ros::NodeHandle nh) : nh_(nh), it_(nh)
        {
          odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 100);
          map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("map", 10);
          cam_pose_pub_ = nh.advertise<geometry_msgs::PoseArray>("cam_poses", 100);
          pruned_cam_states_pub_ = nh.advertise<nav_msgs::Path>("pruned_cam_states_path", 100);
          raw_image_pub_ = it_.advertise("image_raw", 20);
          track_image_pub_ = it_.advertise("track_overlay_image", 20);
        }

        void setTrackHandler(std::shared_ptr<corner_detector::TrackHandler> track_handler) {
          track_handler_ = track_handler;
        }

        void setMSCKF(std::shared_ptr<MSCKF<_S>> msckf) {
          msckf_ = msckf;
        }

        void publishOdom(const ros::Time publish_time) {
          auto imu_state = msckf_->getImuState();
          auto imu_covar = msckf_->getImuCovar();

          nav_msgs::Odometry odom;
          odom.header.stamp = publish_time;
          odom.header.frame_id = "map";
          odom.twist.twist.linear.x = imu_state.v_I_G[0];
          odom.twist.twist.linear.y = imu_state.v_I_G[1];
          odom.twist.twist.linear.z = imu_state.v_I_G[2];

          for(int i=0; i<6; i++){
            for(int j=0; j<6; j++){
              int a = (i<3) ? 12+i : i-3;
              int b = (j<3) ? 12+j : j-3;
              odom.pose.covariance[i*6+j] = imu_covar(a,b);
            }
          }

          odom.pose.pose.position.x = imu_state.p_I_G[0];
          odom.pose.pose.position.y = imu_state.p_I_G[1];
          odom.pose.pose.position.z = imu_state.p_I_G[2];

          Quaternion<_S> q_out = imu_state.q_IG.inverse();
          odom.pose.pose.orientation.w = q_out.w();
          odom.pose.pose.orientation.x = q_out.x();
          odom.pose.pose.orientation.y = q_out.y();
          odom.pose.pose.orientation.z = q_out.z();

          odom_pub_.publish(odom);
        }

        void publishTracks(const ros::Time publish_time) {
          if(track_image_pub_.getNumSubscribers() > 0){
            cv_bridge::CvImage out_img;
            out_img.header.frame_id = "cam0";
            out_img.header.stamp = publish_time;
            out_img.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
            out_img.image = track_handler_->get_track_image();
            track_image_pub_.publish(out_img.toImageMsg());
          }
        }

        void publishImage(const ros::Time publish_time) {
          if(raw_image_pub_.getNumSubscribers() > 0){
            cv_bridge::CvImage out_img;
            out_img.header.frame_id = "cam0";
            out_img.header.stamp = publish_time;
            out_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
            out_img.image = track_handler_->get_image();
            raw_image_pub_.publish(out_img.toImageMsg());
          }
        }

        void publishMap(const ros::Time publish_time) {
          if(map_pub_.getNumSubscribers()>0){
            std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> map =
              msckf_->getMap();

            pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
            pointcloud->header.frame_id = "map";
            pointcloud->height = 1;
            for (auto& point:map) {
              pointcloud->points.push_back(pcl::PointXYZ(point(0),
                    point(1),
                    point(2)));
            }

            pointcloud->width = pointcloud->points.size();
            map_pub_.publish(pointcloud);
          }
        }

        void publishCamPoses(const ros::Time publish_time) {
          if(cam_pose_pub_.getNumSubscribers()>0){
            geometry_msgs::PoseArray cam_poses;

            auto msckf_cam_poses = msckf_->getCamStates();
            for( auto& cs : msckf_cam_poses ){
              geometry_msgs::Pose p;
              p.position.x = cs.p_C_G[0];
              p.position.y = cs.p_C_G[1];
              p.position.z = cs.p_C_G[2];
              Quaternion<_S> q_out = cs.q_CG.inverse();
              p.orientation.w = q_out.w();
              p.orientation.x = q_out.x();
              p.orientation.y = q_out.y();
              p.orientation.z = q_out.z();
              cam_poses.poses.push_back(p);
            }

            cam_poses.header.frame_id = "map";
            cam_poses.header.stamp = publish_time;

            cam_pose_pub_.publish(cam_poses);
          }
        }

        void publishPrunedCamStates(const ros::Time publish_time) {
          if(pruned_cam_states_pub_.getNumSubscribers()>0){
            nav_msgs::Path pruned_path;
            pruned_path.header.stamp = publish_time;
            pruned_path.header.frame_id = "map";
            for(auto ci : msckf_->getPrunedStates()){
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

            pruned_cam_states_pub_.publish(pruned_path);
          }
        }

      private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;

        std::shared_ptr<MSCKF<_S>> msckf_;
        std::shared_ptr<corner_detector::TrackHandler> track_handler_;

        ros::Publisher odom_pub_; // Done
        ros::Publisher map_pub_; // Done
        ros::Publisher cam_pose_pub_; // Done
        ros::Publisher pruned_cam_states_pub_; // Done
        ros::Publisher imu_track_pub_;
        ros::Publisher gt_track_pub_;

        image_transport::Publisher raw_image_pub_;
        image_transport::Publisher track_image_pub_;
    };

  template<typename _S>
    class ROSInput {
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ROSInput(ros::NodeHandle nh);
      private:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;

        image_transport::Subscriber image_sub_;

        ros::Subscriber imu_sub_;
        std::vector<std::tuple<double, imuReading<_S>>> imu_queue_;
    };

  class RosInterface {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      RosInterface(ros::NodeHandle nh);

      void imuCallback(const sensor_msgs::ImuConstPtr& imu);

      void imageCallback(const sensor_msgs::ImageConstPtr& msg);

      void publish_core(const ros::Time& publish_time);

      void publish_extra(const ros::Time& publish_time);

    private:
      ros::NodeHandle nh_;
      image_transport::ImageTransport it_;

      image_transport::Subscriber image_sub_;
      image_transport::Publisher track_image_pub_;
      ros::Publisher odom_pub_;

      ros::Subscriber imu_sub_;

      void load_parameters();

      bool debug_;

      std::vector<std::tuple<double, imuReading<float>>> imu_queue_;
      double prev_imu_time_;

      void setup_track_handler();
      std::shared_ptr<corner_detector::TrackHandler> track_handler_;

      Matrix3<float> R_imu_cam_;
      Vector3<float> p_imu_cam_;

      Matrix3<float> R_cam_imu_;
      Vector3<float> p_cam_imu_;

      std::string camera_model_;
      cv::Mat K_;
      std::string distortion_model_;
      cv::Mat dist_coeffs_;

      int n_grid_cols_;
      int n_grid_rows_;
      float ransac_threshold_;

      enum CalibrationMethod { TimedStandStill };
      CalibrationMethod imu_calibration_method_;

      double stand_still_time_;
      double done_stand_still_time_;

      std::atomic<bool> imu_calibrated_;
      bool can_initialize_imu();
      void initialize_imu();

      int state_k_;
      void setup_msckf();
      MSCKF<float> msckf_;
      Camera<float> camera_;
      noiseParams<float> noise_params_;
      MSCKFParams<float> msckf_params_;
      imuState<float> init_imu_state_;
  };

  template<typename _S>
    std::tuple<noiseParams<_S>, MSCKFParams<_S>> fetch_params(ros::NodeHandle nh, double fu=200., double fv=200.) {
      _S feature_cov;
      nh.param<_S>("feature_covariance", feature_cov, 7);

      msckf_mono::noiseParams<_S> noise_params;
      noise_params.u_var_prime = pow(feature_cov/fu,2);
      noise_params.v_var_prime = pow(feature_cov/fv,2);

      Eigen::Matrix<_S,12,1> Q_imu_vars;
      _S w_var, dbg_var, a_var, dba_var;
      nh.param<_S>("imu_vars/w_var", w_var, 1e-5);
      nh.param<_S>("imu_vars/dbg_var", dbg_var, 3.6733e-5);
      nh.param<_S>("imu_vars/a_var", a_var, 1e-3);
      nh.param<_S>("imu_vars/dba_var", dba_var, 7e-4);
      Q_imu_vars << w_var, 	w_var, 	w_var,
                 dbg_var,dbg_var,dbg_var,
                 a_var,	a_var,	a_var,
                 dba_var,dba_var,dba_var;
      noise_params.Q_imu = Q_imu_vars.asDiagonal();

      Eigen::Matrix<_S,15,1> IMUCovar_vars;
      _S q_var_init, bg_var_init, v_var_init, ba_var_init, p_var_init;
      nh.param<_S>("imu_covars/q_var_init", q_var_init, 1e-5);
      nh.param<_S>("imu_covars/bg_var_init", bg_var_init, 1e-2);
      nh.param<_S>("imu_covars/v_var_init", v_var_init, 1e-2);
      nh.param<_S>("imu_covars/ba_var_init", ba_var_init, 1e-2);
      nh.param<_S>("imu_covars/p_var_init", p_var_init, 1e-12);
      IMUCovar_vars << q_var_init, q_var_init, q_var_init,
                    bg_var_init,bg_var_init,bg_var_init,
                    v_var_init, v_var_init, v_var_init,
                    ba_var_init,ba_var_init,ba_var_init,
                    p_var_init, p_var_init, p_var_init;
      noise_params.initial_imu_covar = IMUCovar_vars.asDiagonal();

      msckf_mono::MSCKFParams<_S> msckf_params;
      nh.param<_S>("max_gn_cost_norm", msckf_params.max_gn_cost_norm, 11.);
      msckf_params.max_gn_cost_norm = pow(msckf_params.max_gn_cost_norm/( (fu+fv)/2.0 ), 2);
      nh.param<_S>("translation_threshold", msckf_params.translation_threshold, 0.05);
      nh.param<_S>("min_rcond", msckf_params.min_rcond, 3e-12);

      nh.param<_S>("keyframe_transl_dist", msckf_params.redundancy_angle_thresh, 0.05);
      nh.param<_S>("keyframe_rot_dist", msckf_params.redundancy_distance_thresh, 0.05);

      int max_tl, min_tl, max_cs;
      nh.param<int>("max_track_length", max_tl, 10);	// set to inf to wait for features to go out of view
      nh.param<int>("min_track_length", min_tl, 3);		// set to infinity to dead-reckon only
      nh.param<int>("max_cam_states",   max_cs, 200);

      msckf_params.max_track_length = max_tl;
      msckf_params.min_track_length = min_tl;
      msckf_params.max_cam_states = max_cs;

      return std::make_tuple(noise_params, msckf_params);
    };
}

#endif
