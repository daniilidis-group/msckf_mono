#include <iostream>
#include <string>

#include <ros/ros.h>

#include <datasets/kitti_readers.h>
#include <datasets/data_synchronizers.h>

#include <msckf_mono/msckf.h>
#include <msckf_mono/ros_interface.h>
#include <msckf_mono/corner_detector.h>
#include <msckf_mono/prettyprint.h>

#include <opencv2/core/eigen.hpp>

using namespace kitti_dataset;
using namespace synchronizer;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_msckf");
    ros::NodeHandle nh;

    std::string dataset;
    double calib_end;
    if(!nh.getParam("dataset_path", dataset)){
        std::cerr << "Must define a dataset_path" << std::endl;
        return 0;
    }

    bool force_gt_state;
    nh.param<bool>("force_gt_state", force_gt_state, true);

    ROS_INFO_STREAM("Accessing dataset at " << dataset);

    std::shared_ptr<Calib> kitti_calib(new Calib(dataset));
    std::shared_ptr<Image> cam0(new Image("image_00", dataset, kitti_calib));
    std::shared_ptr<IMU> imu0(new IMU(dataset));
    std::shared_ptr<GroundTruth> gt(new GroundTruth(dataset));

    auto sync = make_synchronizer(imu0, cam0, gt);

    cv::Mat K;
    cv::Mat dist_coeffs;

    cv::eigen2cv(cam0->get_K(), K);

    std::shared_ptr<corner_detector::TrackHandler> track_handler;
    track_handler.reset(new corner_detector::TrackHandler(K, dist_coeffs, "rectified"));

    std::shared_ptr<msckf_mono::MSCKF<float>> msckf(new msckf_mono::MSCKF<float>());

    gt->next();
    msckf_mono::imuState<float> first_imu = gt->get_data();
    first_imu.g << 0.0,0.0,-9.81;
    gt->reset();

    {
    auto q = first_imu.q_IG;
    ROS_INFO_STREAM("\nFirst IMU State" <<
        "\n--p_I_G " << first_imu.p_I_G.transpose() <<
        "\n--q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.z() <<
        "\n--v_I_G " << first_imu.v_I_G.transpose() <<
        "\n---b_a " << first_imu.b_a.transpose() <<
        "\n---b_g " << first_imu.b_g.transpose() <<
        "\n---g " << first_imu.g.transpose());
    }

    auto msckf_cam = cam0->get_camera();
    std::cout << "Camera" << std::endl;
    std::cout << "- f_u " << msckf_cam.f_u << std::endl;
    std::cout << "- f_v " << msckf_cam.f_v << std::endl;
    std::cout << "- c_u " << msckf_cam.c_u << std::endl;
    std::cout << "- c_v " << msckf_cam.c_v << std::endl;
    std::cout << "-p_C_I " << msckf_cam.p_C_I.transpose() << std::endl;
    std::cout << "-q_CI " << msckf_cam.q_CI.w() << "," << msckf_cam.q_CI.x() << "," << msckf_cam.q_CI.y() << "," << msckf_cam.q_CI.z() << std::endl;

    msckf_mono::noiseParams<float> noise_params;
    msckf_mono::MSCKFParams<float> msckf_params;
    std::tie(noise_params, msckf_params) = msckf_mono::fetch_params<float>(nh);

    msckf->initialize(msckf_cam, noise_params, msckf_params, first_imu);

    // Create all publishers
    std::shared_ptr<msckf_mono::ROSOutput<float>> ros_output_manager;
    ros_output_manager.reset(new msckf_mono::ROSOutput<float>(nh));
    ros_output_manager->setTrackHandler(track_handler);
    ros_output_manager->setMSCKF(msckf);

    int state_k = 0;

    while(sync.next() && ros::ok()){
      state_k++;
      auto data_pack = sync.get_data();

      auto cur_ros_time = ros::Time::now();

      auto imu_reading = std::get<0>(data_pack);
      auto image_reading = std::get<1>(data_pack);
      auto gt_reading = std::get<2>(data_pack);

      if (imu_reading && !force_gt_state){
        msckf->propagate(imu_reading.get());
      }

      if(gt_reading && force_gt_state){
        msckf->propagate_forced_state(gt_reading.get());
      }

      if(image_reading){
        track_handler->set_current_image(image_reading.get(), cam0->get_time());

        std::vector<msckf_mono::Vector2<float>, Eigen::aligned_allocator<msckf_mono::Vector2<float>>> cur_features;
        corner_detector::IdVector cur_ids;
        track_handler->tracked_features(cur_features, cur_ids);

        std::vector<msckf_mono::Vector2<float>, Eigen::aligned_allocator<msckf_mono::Vector2<float>>> new_features;
        corner_detector::IdVector new_ids;
        track_handler->new_features(new_features, new_ids);

        msckf->augmentState(state_k, imu0->get_time());
        msckf->update(cur_features, cur_ids);
        msckf->addFeatures(new_features, new_ids);

        msckf->marginalize();
        msckf->pruneEmptyStates();

        ros_output_manager->publishImage(cur_ros_time);
        ros_output_manager->publishTracks(cur_ros_time);
        ros_output_manager->publishCamPoses(cur_ros_time);
        ros_output_manager->publishPrunedCamStates(cur_ros_time);
        ros_output_manager->publishMap(cur_ros_time);
      }

      ros_output_manager->publishOdom(cur_ros_time);

      {
        msckf_mono::imuState<float> imu_state = msckf->getImuState();
        msckf_mono::imuState<float> gt_state = gt->get_data();
        msckf_mono::imuReading<float> imu_data = imu0->get_data();
        auto q = imu_state.q_IG;
        auto qgt = gt_state.q_IG;

        ROS_INFO_STREAM("\nCur IMU State" <<
            "\n--hat-p_I_G " << imu_state.p_I_G.transpose() <<
            "\n--gt-p_I_G " << gt_state.p_I_G.transpose() <<
            "\n--hat-q_IG " << q.w() << "," << q.x() << "," << q.y() << "," << q.z() <<
            "\n--gt-q_IG " << qgt.w() << "," << qgt.x() << "," << qgt.y() << "," << qgt.z() <<
            "\n--hat-v_I_G " << imu_state.v_I_G.transpose() <<
            "\n--gt-v_I_G " << gt_state.v_I_G.transpose());
      }
    }

    msckf->finish();
    auto cur_ros_time = ros::Time::now();
    ros_output_manager->publishImage(cur_ros_time);
    ros_output_manager->publishTracks(cur_ros_time);
    ros_output_manager->publishCamPoses(cur_ros_time);
    ros_output_manager->publishPrunedCamStates(cur_ros_time);
    ros_output_manager->publishMap(cur_ros_time);
    ros_output_manager->publishOdom(cur_ros_time);
}
