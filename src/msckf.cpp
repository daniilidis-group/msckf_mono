#include <msckf_mono/msckf.h>
namespace msckf {
  MSCKF::MSCKF(){}

  void MSCKF::initialize(const Camera &camera, const noiseParams &noise_params,
      const MSCKFParams &msckf_params,
      const imuState &imu_state) {
    // Constructor:
    camera_ = camera;
    noise_params_ = noise_params;
    msckf_params_ = msckf_params;
    num_feature_tracks_residualized_ = 0;
    imu_state_ = imu_state;
    pos_init_ = imu_state_.p_I_G;
    imu_state_.p_I_G_null = imu_state_.p_I_G;
    imu_state_.v_I_G_null = imu_state_.v_I_G;
    imu_state_.q_IG_null = imu_state_.q_IG;
    imu_covar_ = noise_params.initial_imu_covar;
    last_feature_id_ = 0;

    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (int i = 1; i < 100; ++i) {
      boost::math::chi_squared chi_squared_dist(i);
      chi_squared_test_table[i] = boost::math::quantile(chi_squared_dist, 0.05);
    }
    // TODO: Adjust for 0-sized covar?
  }

  void MSCKF::addFeatures(
      const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &features,
      const vector<size_t> &feature_ids) {
    // Assumes featureIDs match features
    // Original code is a bit confusing here. Seems to allow for repeated feature
    // IDs
    // Will assume feature IDs are unique per feature per call
    // TODO: revisit this assumption if necessary
    for (size_t i = 0; i < features.size(); i++) {
      size_t id = feature_ids[i];
      if (std::find(tracked_feature_ids_.begin(), tracked_feature_ids_.end(),
            id) == tracked_feature_ids_.end()) {
        // New feature
        featureTrack track;
        track.feature_id = feature_ids[i];
        track.observations.push_back(features[i]);

        vector<camState>::iterator cam_state_last = cam_states_.end() - 1;
        cam_state_last->tracked_feature_ids.push_back(feature_ids[i]);

        track.cam_state_indices.push_back(cam_state_last->state_id);

        feature_tracks_.push_back(track);
        tracked_feature_ids_.push_back(feature_ids[i]);

      } else {
        std::cout << "Error, added new feature that was already being tracked" << std::endl;
        return;
      }
    }
  }

  void MSCKF::augmentState(const int &state_id, const double &time) {
    // Compute camera_ pose from current IMU pose
    Quaterniond q_CG = camera_.q_CI * imu_state_.q_IG;

    q_CG.normalize();
    camState cam_state;
    cam_state.last_correlated_id = -1;
    cam_state.q_CG = q_CG;

    cam_state.p_C_G =
      imu_state_.p_I_G + imu_state_.q_IG.inverse() * camera_.p_C_I;

    cam_state.time = time;
    cam_state.state_id = state_id;

    MatrixXd P;
    // Build MSCKF covariance matrix
    if (cam_states_.size()) {
      P = MatrixXd::Zero(15 + cam_covar_.cols(), 15 + cam_covar_.cols());
      P.block<15, 15>(0, 0) = imu_covar_;
      P.block(0, 15, 15, cam_covar_.cols()) = imu_cam_covar_;
      P.block(15, 0, cam_covar_.cols(), 15) = imu_cam_covar_.transpose();
      P.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
    } else {
      P = imu_covar_;
    }

    if (P.determinant() < -0.000001) {
      //ROS_ERROR("Covariance determinant is negative! %f", P.determinant());
    }

    MatrixXd J = MatrixXd::Zero(6, 15 + 6 * cam_states_.size());
    J.block<3, 3>(0, 0) = camera_.q_CI.toRotationMatrix();
    J.block<3, 3>(3, 0) =
      vectorToSkewSymmetric(imu_state_.q_IG.inverse() * camera_.p_C_I);
    J.block<3, 3>(3, 12) = Matrix3d::Identity();

    // Camera State Jacobian
    // MatrixXd J = calcJ(imu_state_, cam_states_);

    MatrixXd tempMat = MatrixXd::Identity(15 + 6 * cam_states_.size() + 6,
        15 + 6 * cam_states_.size());
    tempMat.block(15 + 6 * cam_states_.size(), 0, 6,
        15 + 6 * cam_states_.size()) = J;

    // Augment the MSCKF covariance matrix
    MatrixXd P_aug = tempMat * P * tempMat.transpose();

    MatrixXd P_aug_sym = (P_aug + P_aug.transpose()) / 2.0;

    P_aug = P_aug_sym;

    // Break everything into appropriate structs
    cam_states_.push_back(cam_state);
    imu_covar_ = P_aug.block<15, 15>(0, 0);

    cam_covar_.resize(P_aug.rows() - 15, P_aug.cols() - 15);
    cam_covar_ = P_aug.block(15, 15, P_aug.rows() - 15, P_aug.cols() - 15);

    imu_cam_covar_.resize(15, P_aug.cols() - 15);
    imu_cam_covar_ = P_aug.block(0, 15, 15, P_aug.cols() - 15);

    Eigen::VectorXd cov_diag = imu_covar_.diagonal();
  }

  Quaternion<double> MSCKF::buildUpdateQuat(const Vector3d &deltaTheta) {
    Vector3d deltaq = 0.5 * deltaTheta;
    Quaternion<double> updateQuat;
    // Replaced with squaredNorm() ***1x1 result so using sum instead of creating
    // another variable and then referencing the 0th index value***
    double checkSum = deltaq.squaredNorm();
    if (checkSum > 1) {
      updateQuat.w() = 1;
      updateQuat.x() = -deltaq(0);
      updateQuat.y() = -deltaq(1);
      updateQuat.z() = -deltaq(2);
    } else {
      updateQuat.w() = sqrt(1 - checkSum);
      updateQuat.x() = -deltaq(0);
      updateQuat.y() = -deltaq(1);
      updateQuat.z() = -deltaq(2);
    }

    updateQuat.normalize();

    return updateQuat;
  }

  Matrix<double, 15, 15> MSCKF::calcF(const imuState &imu_state_k,
      const measurement &measurement_k) {
    /* Multiplies the error state in the linearized continuous-time
       error state model */
    Matrix<double, 15, 15> F = Matrix<double, 15, 15>::Zero();
    Vector3d omegaHat, aHat;
    omegaHat = measurement_k.omega - imu_state_k.b_g;
    aHat = measurement_k.a - imu_state_k.b_a;
    Matrix3d C_IG = imu_state_k.q_IG.toRotationMatrix();

    F.block<3, 3>(0, 0) = -vectorToSkewSymmetric(omegaHat);
    F.block<3, 3>(0, 3) = -Matrix3d::Identity();
    F.block<3, 3>(6, 0) = -C_IG.transpose() * vectorToSkewSymmetric(aHat);
    F.block<3, 3>(6, 9) = -C_IG.transpose();
    F.block<3, 3>(12, 6) = Matrix3d::Identity();

    return F;
  }

  Matrix<double, 15, 12> MSCKF::calcG(const imuState &imu_state_k) {
    /* Multiplies the noise vector in the linearized continuous-time
       error state model */
    Matrix<double, 15, 12> G = Matrix<double, 15, 12>::Zero();

    Matrix3d C_IG = imu_state_k.q_IG.toRotationMatrix();

    G.block<3, 3>(0, 0) = -Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -C_IG.transpose();
    G.block<3, 3>(9, 9) = Matrix3d::Identity();

    return G;
  }

  void MSCKF::cost(const Eigen::Isometry3d &T_c0_ci, const Eigen::Vector3d &x,
      const Eigen::Vector2d &z, double &e) const {
    // Compute hi1, hi2, and hi3 as Equation (37).
    const double &alpha = x(0);
    const double &beta = x(1);
    const double &rho = x(2);

    Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) +
      rho * T_c0_ci.translation();
    double &h1 = h(0);
    double &h2 = h(1);
    double &h3 = h(2);

    // Predict the feature observation in ci frame.
    Eigen::Vector2d z_hat(h1 / h3, h2 / h3);

    // Compute the residual.
    e = (z_hat - z).squaredNorm();
    return;
  }

  void MSCKF::jacobian(const Eigen::Isometry3d &T_c0_ci, const Eigen::Vector3d &x,
      const Eigen::Vector2d &z, Eigen::Matrix<double, 2, 3> &J,
      Eigen::Vector2d &r, double &w) const {
    // Compute hi1, hi2, and hi3 as Equation (37).
    const double &alpha = x(0);
    const double &beta = x(1);
    const double &rho = x(2);

    Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) +
      rho * T_c0_ci.translation();
    double &h1 = h(0);
    double &h2 = h(1);
    double &h3 = h(2);

    // Compute the Jacobian.
    Eigen::Matrix3d W;
    W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
    W.rightCols<1>() = T_c0_ci.translation();

    J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
    J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

    // Compute the residual.
    Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
    r = z_hat - z;

    // Compute the weight based on the residual.
    double e = r.norm();
    double huber_epsilon = 0.01;
    if (e <= huber_epsilon)
      w = 1.0;
    else
      w = huber_epsilon / (2 * e);

    return;
  }

  void MSCKF::generateInitialGuess(const Eigen::Isometry3d &T_c1_c2,
      const Eigen::Vector2d &z1,
      const Eigen::Vector2d &z2,
      Eigen::Vector3d &p) const {
    // Construct a least square problem to solve the depth.
    Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

    Eigen::Vector2d A(0.0, 0.0);
    A(0) = m(0) - z2(0) * m(2);
    A(1) = m(1) - z2(1) * m(2);

    Eigen::Vector2d b(0.0, 0.0);
    b(0) = z2(0) * T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
    b(1) = z2(1) * T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

    // Solve for the depth.
    double depth = (A.transpose() * A).inverse() * A.transpose() * b;
    p(0) = z1(0) * depth;
    p(1) = z1(1) * depth;
    p(2) = depth;
    return;
  }

  bool MSCKF::initializePosition(
      const vector<camState> &camStates,
      const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &measurements,
      Vector3d &p_f_G) {
    // Organize camera poses and feature observations properly.
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
      cam_poses(0);

    for (auto &cam : camStates) {
      // This camera pose will take a vector from this camera frame
      // to the world frame.
      Eigen::Isometry3d cam0_pose;
      cam0_pose.linear() = cam.q_CG.toRotationMatrix().transpose();
      cam0_pose.translation() = cam.p_C_G;

      cam_poses.push_back(cam0_pose);
    }

    // All camera poses should be modified such that it takes a
    // vector from the first camera frame in the buffer to this
    // camera frame.
    Eigen::Isometry3d T_c0_w = cam_poses[0];
    for (auto &pose : cam_poses) pose = pose.inverse() * T_c0_w;

    // Generate initial guess
    Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
    generateInitialGuess(cam_poses[cam_poses.size() - 1], measurements[0],
        measurements[measurements.size() - 1], initial_position);
    Eigen::Vector3d solution(initial_position(0) / initial_position(2),
        initial_position(1) / initial_position(2),
        1.0 / initial_position(2));

    // Apply Levenberg-Marquart method to solve for the 3d position.
    double initial_damping = 1e-3;
    double lambda = initial_damping;
    int inner_loop_max_iteration = 10;
    int outer_loop_max_iteration = 10;
    double estimation_precision = 5e-7;
    int inner_loop_cntr = 0;
    int outer_loop_cntr = 0;
    bool is_cost_reduced = false;
    double delta_norm = 0;
    // Compute the initial cost.
    double total_cost = 0.0;
    for (int i = 0; i < cam_poses.size(); ++i) {
      double this_cost = 0.0;
      cost(cam_poses[i], solution, measurements[i], this_cost);
      total_cost += this_cost;
    }

    // Outer loop.
    do {
      Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
      Eigen::Vector3d b = Eigen::Vector3d::Zero();

      for (int i = 0; i < cam_poses.size(); ++i) {
        Eigen::Matrix<double, 2, 3> J;
        Eigen::Vector2d r;
        double w;

        jacobian(cam_poses[i], solution, measurements[i], J, r, w);

        if (w == 1) {
          A += J.transpose() * J;
          b += J.transpose() * r;
        } else {
          double w_square = w * w;
          A += w_square * J.transpose() * J;
          b += w_square * J.transpose() * r;
        }
      }

      // Inner loop.
      // Solve for the delta that can reduce the total cost.
      do {
        Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
        Eigen::Vector3d delta = (A + damper).ldlt().solve(b);
        Eigen::Vector3d new_solution = solution - delta;
        delta_norm = delta.norm();

        double new_cost = 0.0;
        for (int i = 0; i < cam_poses.size(); ++i) {
          double this_cost = 0.0;
          cost(cam_poses[i], new_solution, measurements[i], this_cost);
          new_cost += this_cost;
        }

        if (new_cost < total_cost) {
          is_cost_reduced = true;
          solution = new_solution;
          total_cost = new_cost;
          lambda = lambda / 10 > 1e-10 ? lambda / 10 : 1e-10;
        } else {
          is_cost_reduced = false;
          lambda = lambda * 10 < 1e12 ? lambda * 10 : 1e12;
        }

      } while (inner_loop_cntr++ < inner_loop_max_iteration && !is_cost_reduced);

      inner_loop_cntr = 0;

    } while (outer_loop_cntr++ < outer_loop_max_iteration &&
        delta_norm > estimation_precision);

    // Covert the feature position from inverse depth
    // representation to its 3d coordinate.
    Eigen::Vector3d final_position(solution(0) / solution(2),
        solution(1) / solution(2), 1.0 / solution(2));

    // Check if the solution is valid. Make sure the feature
    // is in front of every camera frame observing it.
    bool is_valid_solution = true;
    for (const auto &pose : cam_poses) {
      Eigen::Vector3d position =
        pose.linear() * final_position + pose.translation();
      if (position(2) <= 0) {
        is_valid_solution = false;
        break;
      }
    }

    double normalized_cost =
      total_cost / (2 * cam_poses.size() * cam_poses.size());

    Eigen::VectorXd cov_diag = imu_covar_.diagonal();

    double pos_covar = cov_diag.segment(12, 3).norm();

    double cost_threshold = msckf_params_.max_gn_cost_norm;
    // if (pos_covar > 2)

    if (normalized_cost > cost_threshold) {
      is_valid_solution = false;
    }

    // printf("Cost is: %f, normalized: %f, target: %f\n", total_cost,
    // normalized_cost, cost_threshold);

    // Convert the feature position to the world frame.
    p_f_G = T_c0_w.linear() * final_position + T_c0_w.translation();

    return is_valid_solution;
  }

  void MSCKF::calcMeasJacobian(const Vector3d &p_f_G,
      const vector<size_t> &camStateIndices,
      MatrixXd &H_o_j, MatrixXd &A_j) {
    // Calculates H_o_j according to Mourikis 2007

    MatrixXd H_f_j = MatrixXd::Zero(2 * camStateIndices.size(), 3);
    MatrixXd H_x_j =
      MatrixXd::Zero(2 * camStateIndices.size(), 15 + 6 * cam_states_.size());

    for (int c_i = 0; c_i < camStateIndices.size(); c_i++) {
      size_t index = camStateIndices[c_i];
      Vector3d p_f_C = cam_states_[index].q_CG.toRotationMatrix() *
        (p_f_G - cam_states_[index].p_C_G);

      double X, Y, Z;

      X = p_f_C(0);
      Y = p_f_C(1);
      Z = p_f_C(2);

      // cout << "p_f_C: " << p_f_C.transpose() << ". X: " << X << ", Y: " << Y <<
      // ", Z: " << Z << endl;

      Matrix<double, 2, 3> J_i;
      J_i << 1, 0, -X / Z, 0, 1, -Y / Z;
      J_i *= 1 / Z;

      // Enforce observability constraint, see propagation for citation
      Matrix<double, 2, 6> A;
      A << J_i * vectorToSkewSymmetric(p_f_C),
        -J_i * cam_states_[index].q_CG.toRotationMatrix();

      Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
      u.head(3) = cam_states_[index].q_CG.toRotationMatrix() * imu_state_.g;
      u.tail(3) =
        vectorToSkewSymmetric(p_f_G - cam_states_[index].p_C_G) * imu_state_.g;

      Matrix<double, 2, 6> H_x =
        A - A * u * (u.transpose() * u).inverse() * u.transpose();
      Matrix<double, 2, 3> H_f = -H_x.block<2, 3>(0, 3);
      H_f_j.block<2, 3>(2 * c_i, 0) = H_f;

      // Potential indexing problem zone
      H_x_j.block<2, 6>(2 * c_i, 15 + 6 * (index)) = H_x;
    }

    int jacobian_row_size = 2 * camStateIndices.size();

    JacobiSVD<MatrixXd> svd_helper(H_f_j, ComputeFullU | ComputeThinV);
    A_j = svd_helper.matrixU().rightCols(jacobian_row_size - 3);
    /*
       MatrixXd H_f_j_transpose = H_f_j.transpose();
       printf("H_f_j is:\n");
       std::cout << H_f_j << std::endl;



       FullPivLU<MatrixXd> lu_decomp(H_f_j_transpose); // for null space computation

       A_j = lu_decomp.kernel();
       */

    H_o_j = A_j.transpose() * H_x_j;
    // std::cout << "H_o_j:\n" << H_o_j << std::endl;
  }

  VectorXd MSCKF::calcResidual(
      const Vector3d &p_f_G, const vector<camState> &camStates,
      const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &observations) {
    // CALCRESIDUAL Calculates the residual for a feature position

    VectorXd r_j = VectorXd::Constant(2 * camStates.size(),
        std::numeric_limits<double>::quiet_NaN());

    int iter = 0;
    for (auto state_i : camStates) {
      Vector3d p_f_C = state_i.q_CG.toRotationMatrix() * (p_f_G - state_i.p_C_G);
      Vector2d zhat_i_j = p_f_C.head<2>() / p_f_C(2);

      r_j.segment<2>(2 * iter) = observations[iter] - zhat_i_j;
      iter++;
    }

    return r_j;
  }

  double MSCKF::cond(MatrixXd M) {
    // Returns condition number calculation
    // Code credit: https://forum.kde.org/viewtopic.php?f=74&t=117430
    JacobiSVD<MatrixXd> svd(M);
    double cond = svd.singularValues()(0) /
      svd.singularValues()(svd.singularValues().size() - 1);
    return cond;
  }

  void MSCKF::findRedundantCamStates(vector<size_t> &rm_cam_state_ids) {
    // Ensure that there are enough cam_states to work with
    if (cam_states_.size() < 5) return;

    double dist_thresh = msckf_params_.redundancy_distance_thresh;
    double angle_thresh = msckf_params_.redundancy_angle_thresh;

    auto last_kf = cam_states_.begin();

    auto kf_pos = last_kf->p_C_G;
    auto kf_q = last_kf->q_CG;
    auto next_cs = cam_states_.begin();
    ++next_cs;
    auto protected_states = cam_states_.end() - 3;

    next_cs = last_kf;
    ++next_cs;
    while(next_cs != protected_states){
      const auto& cam_pos = next_cs->p_C_G;
      const auto& cam_q = next_cs->q_CG;
      double distance = (cam_pos-kf_pos).norm();
      double angle = kf_q.angularDistance(cam_q);
      if(distance<dist_thresh&&angle<angle_thresh){
        rm_cam_state_ids.push_back(next_cs->state_id);
      }else{
        last_kf = next_cs;
        kf_pos = last_kf->p_C_G;
        kf_q = last_kf->q_CG;
      }
      ++next_cs;
      int num_remaining = (cam_states_.size() - rm_cam_state_ids.size());
      if(num_remaining <= msckf_params_.max_cam_states){
        break;
      }
    }

    int num_over_max = (cam_states_.size() - rm_cam_state_ids.size()) - msckf_params_.max_cam_states;
    for(int i=0; i<num_over_max; i++){
      if(rm_cam_state_ids.end() == std::find(rm_cam_state_ids.begin(), rm_cam_state_ids.end(), cam_states_[i].state_id)){
        rm_cam_state_ids.push_back(cam_states_[i].state_id);
      }
    }

    if(rm_cam_state_ids.size()<2){
      rm_cam_state_ids.clear();
    }

    // Sort the elements in the output vector
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());
  }

  void MSCKF::finish() {
    for (size_t i = 0; i < tracked_feature_ids_.size(); i++) {
      vector<size_t> camStateIndices;
      vector<camState> camStatesTemp;
      removeTrackedFeature(tracked_feature_ids_[i], camStatesTemp,
          camStateIndices);

      if (camStatesTemp.size() >= msckf_params_.min_track_length) {
        featureTrackToResidualize track;

        if (feature_tracks_[i].feature_id != tracked_feature_ids_[i]) {
          for (vector<featureTrack>::iterator feat_track =
              feature_tracks_.begin();
              feat_track != feature_tracks_.end(); feat_track++) {
            if (feat_track->feature_id == tracked_feature_ids_[i]) {
              track.feature_id = feat_track->feature_id;
              track.observations = feat_track->observations;
              track.initialized = feat_track->initialized;
              if (feat_track->initialized) track.p_f_G = feat_track->p_f_G;
              break;
            }
          }
        } else {
          track.feature_id = feature_tracks_[i].feature_id;
          track.observations = feature_tracks_[i].observations;
          track.initialized = feature_tracks_[i].initialized;
          if (feature_tracks_[i].initialized)
            track.p_f_G = feature_tracks_[i].p_f_G;
        }

        track.cam_states = camStatesTemp;
        track.cam_state_indices = camStateIndices;

        feature_tracks_to_residualize_.push_back(track);
      }

      tracks_to_remove_.push_back(tracked_feature_ids_[i]);
    }

    marginalize();

    // TODO: Add outputs
  }

  void MSCKF::marginalize() {
    if (!feature_tracks_to_residualize_.empty()) {
      int num_passed, num_rejected, num_ransac, max_length, min_length;
      double max_norm, min_norm;
      num_passed = 0;
      num_rejected = 0;
      num_ransac = 0;
      max_length = -1;
      min_length = numeric_limits<int>::max();
      max_norm = -1;
      min_norm = numeric_limits<double>::infinity();

      vector<bool> valid_tracks;
      vector<Vector3d, Eigen::aligned_allocator<Vector3d>> p_f_G_vec;
      int total_nObs = 0;

      for (auto track = feature_tracks_to_residualize_.begin();
          track != feature_tracks_to_residualize_.end(); track++) {
        if (num_feature_tracks_residualized_ > 3 &&
            !checkMotion(track->observations.front(), track->cam_states)) {
          num_rejected += 1;
          valid_tracks.push_back(false);
          continue;
        }

        Vector3d p_f_G;
        double Jcost, RCOND;

        // Estimate feature 3D location with intersection, LM
        bool isvalid =
          initializePosition(track->cam_states, track->observations, p_f_G);

        if (isvalid) {
          track->initialized = true;
          track->p_f_G = p_f_G;
          map_.push_back(p_f_G);
        }

        p_f_G_vec.push_back(p_f_G);
        int nObs = track->observations.size();

        Vector3d p_f_C1 = (track->cam_states[0].q_CG.toRotationMatrix()) *
          (p_f_G - track->cam_states[0].p_C_G);

        Array3d p_f_G_array = p_f_G.array();

        if (!isvalid)
        {
          num_rejected += 1;
          valid_tracks.push_back(false);
        } else {
          num_passed += 1;
          valid_tracks.push_back(true);
          total_nObs += nObs;
          if (nObs > max_length) {
            max_length = nObs;
          }
          if (nObs < min_length) {
            min_length = nObs;
          }

          num_feature_tracks_residualized_ += 1;
        }
      }

      if (!num_passed) {
        return;
      }
      MatrixXd H_o = Eigen::MatrixXd::Zero(2 * total_nObs - 3 * num_passed,
          15 + 6 * cam_states_.size());
      MatrixXd R_o = Eigen::MatrixXd::Zero(2 * total_nObs - 3 * num_passed,
          2 * total_nObs - 3 * num_passed);
      VectorXd r_o(2 * total_nObs - 3 * num_passed);

      Vector2d rep;
      rep << noise_params_.u_var_prime, noise_params_.v_var_prime;

      int stack_counter = 0;
      for (int iter = 0; iter < feature_tracks_to_residualize_.size(); iter++) {
        if (!valid_tracks[iter]) continue;

        featureTrackToResidualize track = feature_tracks_to_residualize_[iter];

        Vector3d p_f_G = p_f_G_vec[iter];
        VectorXd r_j = calcResidual(p_f_G, track.cam_states, track.observations);

        int nObs = track.observations.size();
        MatrixXd R_j = (rep.replicate(nObs, 1)).asDiagonal();

        // Calculate H_o_j and residual
        MatrixXd H_o_j, A_j;
        calcMeasJacobian(p_f_G, track.cam_state_indices, H_o_j, A_j);

        // Stacked residuals and friends
        VectorXd r_o_j = A_j.transpose() * r_j;
        MatrixXd R_o_j = A_j.transpose() * R_j * A_j;

        if (gatingTest(H_o_j, r_o_j, track.cam_states.size() - 1)) {
          r_o.segment(stack_counter, r_o_j.size()) = r_o_j;
          H_o.block(stack_counter, 0, H_o_j.rows(), H_o_j.cols()) = H_o_j;
          R_o.block(stack_counter, stack_counter, R_o_j.rows(), R_o_j.cols()) =
            R_o_j;

          stack_counter += H_o_j.rows();
        }
      }

      H_o.conservativeResize(stack_counter, H_o.cols());
      r_o.conservativeResize(stack_counter);
      R_o.conservativeResize(stack_counter, stack_counter);

      measurementUpdate(H_o, r_o, R_o);
    }
  }

  void MSCKF::measurementUpdate(const MatrixXd &H_o, const VectorXd &r_o,
      const MatrixXd &R_o) {
    if (r_o.size() != 0) {
      // Build MSCKF covariance matrix
      MatrixXd P = MatrixXd::Zero(15 + cam_covar_.rows(), 15 + cam_covar_.cols());
      P.block<15, 15>(0, 0) = imu_covar_;
      if (cam_covar_.rows() != 0) {
        P.block(0, 15, 15, imu_cam_covar_.cols()) = imu_cam_covar_;
        P.block(15, 0, imu_cam_covar_.cols(), 15) = imu_cam_covar_.transpose();
        P.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
      }

      MatrixXd T_H, Q_1, R_n;
      VectorXd r_n;

      // Put residuals in update-worthy form
      // Calculates T_H matrix according to Mourikis 2007
      HouseholderQR<MatrixXd> qr(H_o);
      MatrixXd Q = qr.householderQ();
      MatrixXd R = qr.matrixQR().triangularView<Upper>();

      VectorXd nonZeroRows = R.rowwise().any();
      int numNonZeroRows = nonZeroRows.sum();

      T_H = MatrixXd::Zero(numNonZeroRows, R.cols());
      Q_1 = MatrixXd::Zero(Q.rows(), numNonZeroRows);

      size_t counter = 0;
      for (size_t r_ind = 0; r_ind < R.rows(); r_ind++) {
        if (nonZeroRows(r_ind) == 1.0) {
          T_H.row(counter) = R.row(r_ind);
          Q_1.col(counter) = Q.col(r_ind);
          counter++;
          if (counter > numNonZeroRows) {
            //ROS_ERROR("More non zero rows than expected in QR decomp");
          }
        }
      }

      r_n = Q_1.transpose() * r_o;
      R_n = Q_1.transpose() * R_o * Q_1;

      // Calculate Kalman Gain
      MatrixXd temp = T_H * P * T_H.transpose() + R_n;
      MatrixXd K = (P * T_H.transpose()) * temp.inverse();

      // State Correction
      VectorXd deltaX = K * r_n;

      // Update IMU state (from updateState matlab function defined in MSCKF.m)
      Quaterniond q_IG_up = buildUpdateQuat(deltaX.head<3>()) * imu_state_.q_IG;

      imu_state_.q_IG = q_IG_up;

      imu_state_.b_g += deltaX.segment<3>(3);
      imu_state_.b_a += deltaX.segment<3>(9);
      imu_state_.v_I_G += deltaX.segment<3>(6);
      imu_state_.p_I_G += deltaX.segment<3>(12);

      // Update Camera states
      for (size_t c_i = 0; c_i < cam_states_.size(); c_i++) {
        Quaterniond q_CG_up = buildUpdateQuat(deltaX.segment<3>(15 + 6 * c_i)) *
          cam_states_[c_i].q_CG;
        cam_states_[c_i].q_CG = q_CG_up.normalized();
        cam_states_[c_i].p_C_G += deltaX.segment<3>(18 + 6 * c_i);
      }

      // Covariance correction
      MatrixXd tempMat = MatrixXd::Identity(15 + 6 * cam_states_.size(),
          15 + 6 * cam_states_.size()) -
        K * T_H;

      MatrixXd P_corrected, P_corrected_transpose;
      P_corrected = tempMat * P * tempMat.transpose() + K * R_n * K.transpose();
      // Enforce symmetry
      P_corrected_transpose = P_corrected.transpose();
      P_corrected += P_corrected_transpose;
      P_corrected /= 2;

      // TODO : Verify need for eig check on P_corrected here (doesn't seem too
      // important for now)
      imu_covar_ = P_corrected.block<15, 15>(0, 0);

      // TODO: Check here
      cam_covar_.resize(P_corrected.rows() - 15, P_corrected.cols() - 15);
      cam_covar_ = P_corrected.block(15, 15, P_corrected.rows() - 15,
          P_corrected.cols() - 15);
      imu_cam_covar_.resize(15, P_corrected.cols() - 15);
      imu_cam_covar_ = P_corrected.block(0, 15, 15, P_corrected.cols() - 15);

      return;
    } else
      return;
  }

  Matrix<double, 4, 4> MSCKF::omegaMat(Vector3d omega) {
    // Compute the omega-matrix of a 3-d vector omega
    Matrix<double, 4, 4> bigOmega = Matrix<double, 4, 4>::Zero();
    bigOmega.block<3, 3>(0, 0) = -vectorToSkewSymmetric(omega);
    bigOmega.block<3, 1>(0, 3) = omega;
    bigOmega.block<1, 3>(3, 0) = -omega.transpose();

    return bigOmega;
  }

  void MSCKF::propagate(measurement &measurement_) {
    MatrixXd F = calcF(imu_state_, measurement_);
    MatrixXd G = calcG(imu_state_);

    imuState imu_state_prop = propogateImuStateRK(imu_state_, measurement_);

    // F * dt
    F *= measurement_.dT;
    MatrixXd Phi = F.exp();

    // Apply observability constraints - enforce nullspace of Phi
    // Ref: Observability-constrained Vision-aided Inertial Navigation, Hesch J.
    // et al. Feb, 2012
    Matrix3d R_kk_1(imu_state_.q_IG_null);
    Phi.block<3, 3>(0, 0) =
      imu_state_prop.q_IG.toRotationMatrix() * R_kk_1.transpose();

    Vector3d u = R_kk_1 * imu_state_.g;
    RowVector3d s = (u.transpose() * u).inverse() * u.transpose();

    Matrix3d A1 = Phi.block<3, 3>(6, 0);
    Vector3d w1 =
      vectorToSkewSymmetric(imu_state_.v_I_G_null - imu_state_prop.v_I_G) *
      imu_state_.g;
    Phi.block<3, 3>(6, 0) = A1 - (A1 * u - w1) * s;

    Matrix3d A2 = Phi.block<3, 3>(12, 0);
    Vector3d w2 =
      vectorToSkewSymmetric(measurement_.dT * imu_state_.v_I_G_null +
          imu_state_.p_I_G_null - imu_state_prop.p_I_G) *
      imu_state_.g;
    Phi.block<3, 3>(12, 0) = A2 - (A2 * u - w2) * s;

    Matrix<double, 15, 15> imu_covar_prop = Phi * imu_covar_ * Phi.transpose() +
      Phi * G * noise_params_.Q_imu *
      G.transpose() * Phi.transpose() *
      measurement_.dT;

    MatrixXd imu_cam_covar_prop = Phi * imu_cam_covar_;

    MatrixXd P;
    // Build MSCKF covariance matrix
    if (cam_states_.size()) {
      P = MatrixXd::Zero(15 + cam_covar_.cols(), 15 + cam_covar_.cols());
      P.block<15, 15>(0, 0) = imu_covar_prop;
      P.block(0, 15, 15, cam_covar_.cols()) = imu_cam_covar_prop;
      P.block(15, 0, cam_covar_.cols(), 15) = imu_cam_covar_prop.transpose();
      P.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
    } else {
      P = imu_covar_prop;
    }

    P += P.transpose().eval();
    P /= 2.0;

    // Apply updates
    imu_state_ = imu_state_prop;
    imu_state_.q_IG_null = imu_state_.q_IG;
    imu_state_.v_I_G_null = imu_state_.v_I_G;
    imu_state_.p_I_G_null = imu_state_.p_I_G;
    imu_covar_ = (imu_covar_prop + imu_covar_prop.transpose()) /
      2.0;  // P.block<15, 15>(0, 0);
    imu_cam_covar_.resize(imu_cam_covar_prop.rows(), imu_cam_covar_prop.cols());
    imu_cam_covar_ = imu_cam_covar_prop;
  }

  imuState MSCKF::propogateImuState(const imuState &imu_state_k,
      const measurement &measurement_k) {
    imuState imuStateProp = imu_state_k;
    double dT = measurement_k.dT;
    Matrix<double, 4, 4> omega_psi =
      0.5 * omegaMat(measurement_k.omega - imu_state_k.b_g) * dT;

    // Note: MSCKF Matlab code assumes quaternion form: -x,-y,-z,w
    //     Eigen quaternion is of form: w,x,y,z
    // Following computation accounts for this change

    Vector4d q_IG(-imu_state_k.q_IG.x(), -imu_state_k.q_IG.y(),
        -imu_state_k.q_IG.z(), imu_state_k.q_IG.w());

    Vector4d q_IG_prop = q_IG + omega_psi * q_IG;

    Quaternion<double> q_IG_prop_quat(q_IG_prop(3), -q_IG_prop(0), -q_IG_prop(1),
        -q_IG_prop(2));
    q_IG_prop_quat.normalize();
    imuStateProp.q_IG = q_IG_prop_quat;

    imuStateProp.v_I_G = (((imu_state_k.q_IG.toRotationMatrix()).transpose()) *
        (measurement_k.a - imu_state_k.b_a) +
        imu_state_k.g) *
      dT +
      imu_state_k.v_I_G;
    imuStateProp.p_I_G = imu_state_k.p_I_G + imu_state_k.v_I_G * dT;

    return imuStateProp;
  }

  imuState MSCKF::propogateImuStateRK(const imuState &imu_state_k,
      const measurement &measurement_k) {
    imuState imuStateProp = imu_state_k;
    double dT = measurement_k.dT;

    Matrix<double, 4, 4> omega_psi =
      0.5 * omegaMat(measurement_k.omega - imu_state_k.b_g);

    // Note: MSCKF Matlab code assumes quaternion form: -x,-y,-z,w
    //     Eigen quaternion is of form: w,x,y,z
    // Following computation accounts for this change

    Vector4d y0, k0, k1, k2, k3, k4, k5, y_t;
    y0(0) = -imu_state_k.q_IG.x();
    y0(1) = -imu_state_k.q_IG.y();
    y0(2) = -imu_state_k.q_IG.z();
    y0(3) = imu_state_k.q_IG.w();

    k0 = omega_psi * (y0);
    k1 = omega_psi * (y0 + (k0 / 4) * dT);
    k2 = omega_psi * (y0 + (k0 / 8 + k1 / 8) * dT);
    k3 = omega_psi * (y0 + (-k1 / 2 + k2) * dT);
    k4 = omega_psi * (y0 + (k0 * 3 / 16 + k3 * 9 / 16) * dT);
    k5 = omega_psi *
      (y0 +
       (-k0 * 3 / 7 + k1 * 2 / 7 + k2 * 12 / 7 - k3 * 12 / 7 + k4 * 8 / 7) *
       dT);

    y_t = y0 + (7 * k0 + 32 * k2 + 12 * k3 + 32 * k4 + 7 * k5) * dT / 90;

    Quaternion<double> q(y_t(3), -y_t(0), -y_t(1), -y_t(2));
    q.normalize();

    imuStateProp.q_IG = q;
    imuStateProp.v_I_G = (((imu_state_k.q_IG.toRotationMatrix()).transpose()) *
        (measurement_k.a - imu_state_k.b_a) +
        imu_state_k.g) *
      dT +
      imu_state_k.v_I_G;

    imuStateProp.p_I_G = imu_state_k.p_I_G + imu_state_k.v_I_G * dT;
    return imuStateProp;
  }

  void MSCKF::pruneEmptyStates() {
    int max_states = msckf_params_.max_cam_states;
    if (cam_states_.size() < max_states) return;
    vector<size_t> deleteIdx;
    deleteIdx.clear();

    size_t num_states = cam_states_.size();

    // Find all cam_states_ with no tracked landmarks and prune them
    auto camState_it = cam_states_.begin();
    size_t num_deleted = 0;
    int camstate_pos = 0;
    int num_cam_states = cam_states_.size();

    int last_to_remove = num_cam_states - max_states-1;

    if(cam_states_.front().tracked_feature_ids.size()){
      return;
    }

    for (int i = 1; i < num_cam_states - max_states; i++) {
      if (cam_states_[i].tracked_feature_ids.size()) {
        last_to_remove = i - 1;
        break;
      }
    }

    for (int i = 0; i <= last_to_remove; ++i) {
      deleteIdx.push_back(camstate_pos + num_deleted);
      pruned_states_.push_back(*camState_it);
      camState_it = cam_states_.erase(camState_it);
      num_deleted++;
    }

    if (deleteIdx.size() != 0) {
      int n_remove = 0;
      int n_keep = 0;
      std::vector<bool> to_keep(num_states, false);
      for (size_t IDx = 0; IDx < num_states; IDx++) {
        if (find(deleteIdx.begin(), deleteIdx.end(), IDx) != deleteIdx.end())
          n_remove++;
        else {
          to_keep[IDx] = true;
          n_keep++;
        }
      }

      int remove_counter = 0;
      int keep_counter = 0;
      VectorXi keepCovarIdx(6 * n_keep);
      VectorXi removeCovarIdx(6 * n_remove);
      for (size_t IDx = 0; IDx < num_states; IDx++) {
        if (!to_keep[IDx]) {
          removeCovarIdx.segment<6>(6 * remove_counter) =
            VectorXi::LinSpaced(6, 6 * IDx, 6 * (IDx + 1) - 1);
          remove_counter++;
        } else {
          keepCovarIdx.segment<6>(6 * keep_counter) =
            VectorXi::LinSpaced(6, 6 * IDx, 6 * (IDx + 1) - 1);
          keep_counter++;
        }
      }

      MatrixXd prunedCamCovar;
      cam_covar_slice(keepCovarIdx, prunedCamCovar);
      cam_covar_.resize(prunedCamCovar.rows(), prunedCamCovar.cols());
      cam_covar_ = prunedCamCovar;

      Matrix<double, 15, Dynamic> prunedImuCamCovar;
      imu_cam_covar_slice(keepCovarIdx, prunedImuCamCovar);
      imu_cam_covar_.resize(prunedImuCamCovar.rows(), prunedImuCamCovar.cols());
      imu_cam_covar_ = prunedImuCamCovar;
    }

    // TODO: Additional outputs = deletedCamCovar (used to compute sigma),
    // deletedCamStates
    }

    void MSCKF::pruneRedundantStates() {
      // Cap number of cam states used in computation to max_cam_states
      if (cam_states_.size() < 20){
        return;
      }

      // Find two camera states to rmoved
      vector<size_t> rm_cam_state_ids;
      rm_cam_state_ids.clear();
      findRedundantCamStates(rm_cam_state_ids);

      // Find size of jacobian matrix
      size_t jacobian_row_size = 0;
      for (auto &feature : feature_tracks_) {
        vector<size_t> involved_cam_state_ids;
        size_t obs_id;
        // Check how many camera states to be removed are associated with a given
        // feature
        for (const auto &cam_id : rm_cam_state_ids) {
          auto obs_it = find(feature.cam_state_indices.begin(),
              feature.cam_state_indices.end(), cam_id);
          if (obs_it != feature.cam_state_indices.end()) {
            involved_cam_state_ids.push_back(cam_id);
            obs_id = distance(feature.cam_state_indices.begin(), obs_it);
          }
        }

        if (involved_cam_state_ids.size() == 0) continue;
        if (involved_cam_state_ids.size() == 1) {
          feature.observations.erase(feature.observations.begin() + obs_id);
          feature.cam_state_indices.erase(feature.cam_state_indices.begin() +
              obs_id);
          continue;
        }

        if (!feature.initialized) {
          vector<camState> feature_associated_cam_states;
          for (const auto &cam_state : cam_states_) {
            if (find(feature.cam_state_indices.begin(),
                  feature.cam_state_indices.end(),
                  cam_state.state_id) != feature.cam_state_indices.end())
              feature_associated_cam_states.push_back(cam_state);
          }
          if (!checkMotion(feature.observations.front(),
                feature_associated_cam_states)) {
            for (const auto &cam_id : involved_cam_state_ids) {
              auto cam_it = find(feature.cam_state_indices.begin(),
                  feature.cam_state_indices.end(), cam_id);
              if (cam_it != feature.cam_state_indices.end()) {
                size_t obs_idx =
                  distance(feature.cam_state_indices.begin(), cam_it);
                feature.cam_state_indices.erase(cam_it);
                feature.observations.erase(feature.observations.begin() + obs_idx);
              }
            }
            continue;
          } else {
            Vector3d p_f_G;
            if (!initializePosition(feature_associated_cam_states,
                  feature.observations, p_f_G)) {
              for (const auto &cam_id : involved_cam_state_ids) {
                auto cam_it = find(feature.cam_state_indices.begin(),
                    feature.cam_state_indices.end(), cam_id);
                if (cam_it != feature.cam_state_indices.end()) {
                  size_t obs_idx =
                    distance(feature.cam_state_indices.begin(), cam_it);
                  feature.cam_state_indices.erase(cam_it);
                  feature.observations.erase(feature.observations.begin() +
                      obs_idx);
                }
              }
              continue;
            } else {
              feature.initialized = true;
              feature.p_f_G = p_f_G;
              map_.push_back(p_f_G);
            }
          }
        }

        jacobian_row_size += 2 * involved_cam_state_ids.size() - 3;
      }

      // Compute Jacobian and Residual
      MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 15 + 6 * cam_states_.size());
      MatrixXd R_x = MatrixXd::Zero(jacobian_row_size, jacobian_row_size);
      VectorXd r_x = VectorXd::Zero(jacobian_row_size);
      int stack_counter = 0;

      Vector2d rep;
      rep << noise_params_.u_var_prime, noise_params_.v_var_prime;

      for (auto &feature : feature_tracks_) {
        vector<size_t> involved_cam_state_ids;
        vector<Vector2d, Eigen::aligned_allocator<Vector2d>> involved_observations;
        for (const auto &cam_id : rm_cam_state_ids) {
          auto cam_it = find(feature.cam_state_indices.begin(),
              feature.cam_state_indices.end(), cam_id);
          if (cam_it != feature.cam_state_indices.end()) {
            involved_cam_state_ids.push_back(cam_id);
            involved_observations.push_back(feature.observations[distance(
                  feature.cam_state_indices.begin(), cam_it)]);
          }
        }

        size_t nObs = involved_cam_state_ids.size();
        if (nObs == 0) continue;

        vector<camState> involved_cam_states;
        vector<size_t> cam_state_indices;
        int cam_state_iter = 0;
        for (const auto &cam_state : cam_states_) {
          if (find(involved_cam_state_ids.begin(), involved_cam_state_ids.end(),
                cam_state.state_id) != involved_cam_state_ids.end()) {
            involved_cam_states.push_back(cam_state);
            cam_state_indices.push_back(cam_state_iter);
          }
          cam_state_iter++;
        }

        // Calculate H_xj and residual
        VectorXd r_j =
          calcResidual(feature.p_f_G, involved_cam_states, involved_observations);

        MatrixXd R_j = (rep.replicate(nObs, 1)).asDiagonal();

        MatrixXd H_x_j, A_j;
        calcMeasJacobian(feature.p_f_G, cam_state_indices, H_x_j, A_j);

        // Stacked residuals and friends
        VectorXd r_x_j = A_j.transpose() * r_j;
        MatrixXd R_x_j = A_j.transpose() * R_j * A_j;

        if (gatingTest(H_x_j, r_x_j, nObs - 1)) {
          r_x.segment(stack_counter, r_x_j.size()) = r_x_j;
          H_x.block(stack_counter, 0, H_x_j.rows(), H_x_j.cols()) = H_x_j;
          R_x.block(stack_counter, stack_counter, R_x_j.rows(), R_x_j.cols()) =
            R_x_j;

          stack_counter += H_x_j.rows();
        }

        // Done, now remove these cam states registrations and corresponding
        // observations from the feature
        for (const auto &cam_id : involved_cam_state_ids) {
          auto cam_it = find(feature.cam_state_indices.begin(),
              feature.cam_state_indices.end(), cam_id);
          if (cam_it != feature.cam_state_indices.end()) {
            feature.cam_state_indices.erase(cam_it);
            feature.observations.erase(
                feature.observations.begin() +
                distance(feature.cam_state_indices.begin(), cam_it));
          }
        }
      }

      H_x.conservativeResize(stack_counter, H_x.cols());
      r_x.conservativeResize(stack_counter);
      R_x.conservativeResize(stack_counter, stack_counter);

      // Perform Measurement Update
      measurementUpdate(H_x, r_x, R_x);

      // Time to prune
      vector<size_t> deleteIdx(0);

      size_t num_states = cam_states_.size();

      // Find all cam states which are marked for deletion
      auto cam_state_it = cam_states_.begin();
      size_t num_deleted = 0;
      int cam_state_pos = 0;

      while (cam_state_it != cam_states_.end()) {
        if (find(rm_cam_state_ids.begin(), rm_cam_state_ids.end(),
              cam_state_it->state_id) != rm_cam_state_ids.end()) {
          // TODO: add to pruned states? If yes, maybe sort states by state id
          deleteIdx.push_back(cam_state_pos + num_deleted);
          pruned_states_.push_back(*cam_state_it);
          cam_state_it = cam_states_.erase(cam_state_it);
          ++num_deleted;
        } else {
          ++cam_state_it;
          ++cam_state_pos;
        }
      }

      if (num_deleted != 0) {
        int n_remove = 0;
        int n_keep = 0;
        vector<bool> to_keep(num_states, false);
        for (size_t IDx = 0; IDx < num_states; ++IDx) {
          if (find(deleteIdx.begin(), deleteIdx.end(), IDx) != deleteIdx.end())
            ++n_remove;
          else {
            to_keep[IDx] = true;
            ++n_keep;
          }
        }

        int remove_counter = 0;
        int keep_counter = 0;
        VectorXi keepCovarIdx(6 * n_keep);
        VectorXi removeCovarIdx(6 * n_remove);

        for (size_t IDx = 0; IDx < num_states; ++IDx) {
          if (!to_keep[IDx]) {
            removeCovarIdx.segment<6>(6 * remove_counter) =
              VectorXi::LinSpaced(6, 6 * IDx, 6 * (IDx + 1) - 1);
            ++remove_counter;
          } else {
            keepCovarIdx.segment<6>(6 * keep_counter) =
              VectorXi::LinSpaced(6, 6 * IDx, 6 * (IDx + 1) - 1);
            ++keep_counter;
          }
        }

        MatrixXd prunedCamCovar;
        cam_covar_slice(keepCovarIdx, prunedCamCovar);

        cam_covar_.resize(prunedCamCovar.rows(), prunedCamCovar.cols());
        cam_covar_ = prunedCamCovar;

        Matrix<double, 15, Dynamic> prunedImuCamCovar;
        imu_cam_covar_slice(keepCovarIdx, prunedImuCamCovar);

        imu_cam_covar_.resize(prunedImuCamCovar.rows(), prunedImuCamCovar.cols());
        imu_cam_covar_ = prunedImuCamCovar;
      }
    }

void MSCKF::removeTrackedFeature(const size_t featureID,
    vector<camState> &featCamStates,
    vector<size_t> &camStateIndices) {
  featCamStates.clear();
  camStateIndices.clear();

  for (size_t c_i = 0; c_i < cam_states_.size(); c_i++) {
    auto feature_iter =
      std::find(cam_states_[c_i].tracked_feature_ids.begin(),
          cam_states_[c_i].tracked_feature_ids.end(), featureID);
    if (feature_iter != cam_states_[c_i].tracked_feature_ids.end()) {
      cam_states_[c_i].tracked_feature_ids.erase(feature_iter);
      camStateIndices.push_back(c_i);
      featCamStates.push_back(cam_states_[c_i]);
    }
  }
}

Vector3d MSCKF::skewSymmetricToVector(Matrix3d Skew) {
  Vector3d Vec;
  Vec(0) = Skew(2, 1);
  Vec(1) = Skew(0, 2);
  Vec(3) = Skew(1, 0);

  return Vec;
}

Vector3d MSCKF::Triangulate(const Vector2d &obs1, const Vector2d &obs2,
    const Matrix3d &C_12, const Vector3d &t_21_1) {
  // Triangulate feature position given 2 observations and the transformation
  // between the 2 observation frames
  // Homogeneous normalized vector representations of observations:
  Vector3d v1(0, 0, 1), v2(0, 0, 1);
  v1.segment<2>(0) = obs1;
  v2.segment<2>(0) = obs2;

  v1.normalize();
  v2.normalize();

  // scalarConstants:= [lambda_1; lambda_2] = lambda
  // A*lambda = t_21_1  -->  lambda = A\t_21_1
  Matrix<double, 3, 2, ColMajor> A_;
  A_ << v1, -C_12 * v2;
  MatrixXd A = A_;

  Vector2d scalarConstants = A.colPivHouseholderQr().solve(t_21_1);

  return scalarConstants(0) * v1;
}

void MSCKF::update(
    const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &measurements,
    const vector<size_t> &feature_ids) {

  feature_tracks_to_residualize_.clear();
  tracks_to_remove_.clear();

  int id_iter = 0;
  // Loop through all features being tracked
  for (auto feature_id : tracked_feature_ids_) {
    // Check if old feature is seen in current measurements
    auto input_feature_ids_iter =
      find(feature_ids.begin(), feature_ids.end(), feature_id);
    bool is_valid = (input_feature_ids_iter != feature_ids.end());

    // If so, get the relevant track
    auto track = feature_tracks_.begin() + id_iter;

    // If we're still tracking this point, add the observation
    if (is_valid) {
      size_t feature_ids_dist =
        distance(feature_ids.begin(), input_feature_ids_iter);
      track->observations.push_back(measurements[feature_ids_dist]);

      auto cam_state_iter = cam_states_.end() - 1;
      cam_state_iter->tracked_feature_ids.push_back(feature_id);

      track->cam_state_indices.push_back(cam_state_iter->state_id);
    }

    // If corner is not valid or track is too long, remove track to be
    // residualized
    if (!is_valid  || (track->observations.size() >=
          msckf_params_.max_track_length))
    {
      featureTrackToResidualize track_to_residualize;
      removeTrackedFeature(feature_id, track_to_residualize.cam_states,
          track_to_residualize.cam_state_indices);

      // If track is long enough, add to the residualized list
      if (track_to_residualize.cam_states.size() >=
          msckf_params_.min_track_length) {
        track_to_residualize.feature_id = track->feature_id;
        track_to_residualize.observations = track->observations;
        track_to_residualize.initialized = track->initialized;
        if (track->initialized) track_to_residualize.p_f_G = track->p_f_G;

        feature_tracks_to_residualize_.push_back(track_to_residualize);
      }

      tracks_to_remove_.push_back(feature_id);
    }

    id_iter++;
  }

  // TODO: Double check this stuff and maybe use use non-pointers for accessing
  // elements so that it only requires one pass

  for (auto feature_id : tracks_to_remove_) {
    auto track_iter = feature_tracks_.begin();
    while (track_iter != feature_tracks_.end()) {
      if (track_iter->feature_id == feature_id) {
        size_t last_id = track_iter->cam_state_indices.back();
        for (size_t index : track_iter->cam_state_indices) {
          for (auto &camstate : cam_states_) {
            if (!camstate.tracked_feature_ids.size() &&
                camstate.state_id == index) {
              camstate.last_correlated_id = last_id;
            }
          }
        }
        track_iter = feature_tracks_.erase(track_iter);
        break;
      } else
        track_iter++;
    }

    auto corresponding_id = std::find(tracked_feature_ids_.begin(),
        tracked_feature_ids_.end(), feature_id);

    if (corresponding_id != tracked_feature_ids_.end()) {
      tracked_feature_ids_.erase(corresponding_id);
    }
  }
}

Matrix3d MSCKF::vectorToSkewSymmetric(Vector3d Vec) {
  // Returns skew-symmetric form of a 3-d vector
  Matrix3d M;
  M << 0, -Vec(2), Vec(1), Vec(2), 0, -Vec(0), -Vec(1), Vec(0), 0;

  return M;
}

// Constraint on track to be marginalized based on Mahalanobis Gating
// High Precision, Consistent EKF-based Visual-Inertial Odometry by Li et al.
bool MSCKF::gatingTest(const MatrixXd &H, const VectorXd &r, const int &dof) {
  MatrixXd P = MatrixXd::Zero(15 + cam_covar_.rows(), 15 + cam_covar_.cols());
  P.block<15, 15>(0, 0) = imu_covar_;
  if (cam_covar_.rows() != 0) {
    P.block(0, 15, 15, imu_cam_covar_.cols()) = imu_cam_covar_;
    P.block(15, 0, imu_cam_covar_.cols(), 15) = imu_cam_covar_.transpose();
    P.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
  }

  MatrixXd P1 = H * P * H.transpose();
  MatrixXd P2 =
    noise_params_.u_var_prime * MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

  if (gamma < chi_squared_test_table[dof]) {
    // cout << "passed" << endl;
    return true;
  } else {
    // cout << "failed" << endl;
    return false;
  }
}

bool MSCKF::checkMotion(const Vector2d first_observation,
    const vector<camState> &cam_states) {
  if (cam_states.size() < 2) {
    return false;
  }
  const camState &first_cam = cam_states.front();
  // const camState& last_cam = cam_states.back();

  Eigen::Isometry3d first_cam_pose;
  first_cam_pose.linear() = first_cam.q_CG.toRotationMatrix().transpose();
  first_cam_pose.translation() = first_cam.p_C_G;
  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  Eigen::Vector3d feature_direction;
  feature_direction << first_observation, 1.0;
  feature_direction = feature_direction / feature_direction.norm();
  feature_direction = first_cam_pose.linear() * feature_direction;

  double max_ortho_translation = 0;

  for (auto second_cam_iter = cam_states.begin() + 1;
      second_cam_iter != cam_states.end(); second_cam_iter++) {
    Eigen::Isometry3d second_cam_pose;
    second_cam_pose.linear() =
      second_cam_iter->q_CG.toRotationMatrix().transpose();
    second_cam_pose.translation() = second_cam_iter->p_C_G;
    // Compute the translation between the first frame
    // and the last frame. We assume the first frame and
    // the last frame will provide the largest motion to
    // speed up the checking process.
    Eigen::Vector3d translation =
      second_cam_pose.translation() - first_cam_pose.translation();
    // translation = translation / translation.norm();
    double parallel_translation = translation.transpose() * feature_direction;
    Eigen::Vector3d orthogonal_translation =
      translation - parallel_translation * feature_direction;
    if (orthogonal_translation.norm() > max_ortho_translation) {
      max_ortho_translation = orthogonal_translation.norm();
    }
  }

  if (max_ortho_translation > msckf_params_.translation_threshold)
    return true;
  else
    return false;
}

void MSCKF::cam_covar_slice(VectorXi& inds, MatrixXd& tmp){
  int inds_size = inds.rows();
  tmp.resize(inds_size, inds_size);
  for(int i=0; i<inds_size; i++){
    for(int j=0; j<inds_size; j++){
      tmp(i,j) = cam_covar_(inds(i), inds(j));
    }
  }
}

void MSCKF::imu_cam_covar_slice(VectorXi& inds, Matrix<double,15,Dynamic>& tmp){
  int inds_size = inds.rows();
  tmp.resize(15, inds_size);
  for(int i=0; i<15; i++){
    for(int j=0; j<inds_size; j++){
      tmp(i,j) = imu_cam_covar_(i, inds(j));
    }
  }
}

}  // End namespace
