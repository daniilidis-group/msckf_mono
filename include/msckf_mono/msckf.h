#ifndef MSCKF_HPP_
#define MSCKF_HPP_

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/StdVector>
#include <boost/math/distributions/chi_squared.hpp>

#include <msckf_mono/types.h>
#include <msckf_mono/matrix_utils.h>

using namespace Eigen;

/* Notes to self:
 *		- Noiseparams in calcGNPoseEst
 *		- The thing with the quaternions being inverted
 */
namespace msckf_mono {
  template <typename _S>
    class MSCKF {
    private: 
      Camera<_S> camera_;
      noiseParams<_S> noise_params_;
      MSCKFParams<_S> msckf_params_;
      // prunedStates;
      std::vector<featureTrack<_S>> feature_tracks_;
      std::vector<size_t> tracked_feature_ids_;

      std::vector<featureTrackToResidualize<_S>> feature_tracks_to_residualize_;
      size_t num_feature_tracks_residualized_;
      std::vector<size_t> tracks_to_remove_;
      size_t last_feature_id_;

      imuState<_S> imu_state_;
      std::vector<camState<_S>> cam_states_;

      std::vector<camState<_S>> pruned_states_;
      std::vector<Vector3<_S>, Eigen::aligned_allocator<Vector3<_S>>> map_;

      Matrix<_S,15,15> imu_covar_;
      MatrixX<_S> cam_covar_;
      Matrix<_S,15,Dynamic> imu_cam_covar_;

      std::vector<_S> chi_squared_test_table;
      Vector3<_S> pos_init_;
      Quaternion<_S> quat_init_;

      Matrix<_S,15,15> F_;
      Matrix<_S,15,15> Phi_;
      Matrix<_S,15,12> G_;

      MatrixX<_S> P_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MSCKF(){}

      // Initializes the filter state and parameters.
      void initialize(const Camera<_S>& camera,
                      const noiseParams<_S>& noise_params,
                      const MSCKFParams<_S>& msckf_params,
                      const imuState<_S>& imu_state) {
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
        chi_squared_test_table.resize(99);
        for (int i = 1; i < 100; ++i) {
          boost::math::chi_squared chi_squared_dist(i);
          chi_squared_test_table[i-1] = boost::math::quantile(chi_squared_dist, 0.05);
        }
        // TODO: Adjust for 0-sized covar?
      }

      // Given an IMU measurement, propagates the latest camera pose to the timestamp in measurement
      // using the acceleration and angular velocity in measurement.
      void propagate(imuReading<_S> &measurement_) {
        calcF(imu_state_, measurement_);
        calcG(imu_state_);

        imuState<_S> imu_state_prop = propogateImuStateRK(imu_state_, measurement_);

        // F * dt
        F_ *= measurement_.dT;

        // Matrix exponential
        Phi_ = F_.exp();

        // Apply observability constraints - enforce nullspace of Phi
        // Ref: Observability-constrained Vision-aided Inertial Navigation, Hesch J.
        // et al. Feb, 2012
        Matrix3<_S> R_kk_1(imu_state_.q_IG_null);
        Phi_.template block<3, 3>(0, 0) =
          imu_state_prop.q_IG.toRotationMatrix() * R_kk_1.transpose();

        Vector3<_S> u = R_kk_1 * imu_state_.g;
        RowVector3<_S> s = (u.transpose() * u).inverse() * u.transpose();

        Matrix3<_S> A1 = Phi_.template block<3, 3>(6, 0);
        Vector3<_S> tmp = imu_state_.v_I_G_null - imu_state_prop.v_I_G;
        Vector3<_S> w1 = vectorToSkewSymmetric(tmp) * imu_state_.g;
        Phi_.template block<3, 3>(6, 0) = A1 - (A1 * u - w1) * s;

        Matrix3<_S> A2 = Phi_.template block<3, 3>(12, 0);
        tmp = measurement_.dT * imu_state_.v_I_G_null +
          imu_state_.p_I_G_null - imu_state_prop.p_I_G;
        Vector3<_S> w2 = vectorToSkewSymmetric(tmp) * imu_state_.g;
        Phi_.template block<3, 3>(12, 0) = A2 - (A2 * u - w2) * s;

        Matrix<_S, 15, 15> imu_covar_prop = Phi_ * (imu_covar_ + G_ * noise_params_.Q_imu * G_.transpose() * measurement_.dT) * Phi_.transpose();


        // Apply updates directly
        imu_state_ = imu_state_prop;
        imu_state_.q_IG_null = imu_state_.q_IG;
        imu_state_.v_I_G_null = imu_state_.v_I_G;
        imu_state_.p_I_G_null = imu_state_.p_I_G;

        imu_covar_ = (imu_covar_prop + imu_covar_prop.transpose()) / 2.0;
        imu_cam_covar_ = Phi_ * imu_cam_covar_;
      }

      // Generates a new camera state and adds it to the full state and covariance.
      void augmentState(const int& state_id, const _S& time) {
        map_.clear();

        // Compute camera_ pose from current IMU pose
        Quaternion<_S> q_CG = camera_.q_CI * imu_state_.q_IG;

        q_CG.normalize();
        camState<_S> cam_state;
        cam_state.last_correlated_id = -1;
        cam_state.q_CG = q_CG;

        cam_state.p_C_G =
          imu_state_.p_I_G + imu_state_.q_IG.inverse() * camera_.p_C_I;

        cam_state.time = time;
        cam_state.state_id = state_id;

        // Build MSCKF covariance matrix
        if (cam_states_.size()) {
          P_.resize(15 + cam_covar_.cols(), 15 + cam_covar_.cols());
          P_.template block<15, 15>(0, 0) = imu_covar_;
          P_.block(0, 15, 15, cam_covar_.cols()) = imu_cam_covar_;
          P_.block(15, 0, cam_covar_.cols(), 15) = imu_cam_covar_.transpose();
          P_.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
        } else {
          P_ = imu_covar_;
        }

        if (P_.determinant() < -0.000001) {
          //ROS_ERROR("Covariance determinant is negative! %f", P.determinant());
        }

        MatrixX<_S> J = MatrixX<_S>::Zero(6, 15 + 6 * cam_states_.size());
        J.template block<3, 3>(0, 0) = camera_.q_CI.toRotationMatrix();
        J.template block<3, 3>(3, 0) =
          vectorToSkewSymmetric(imu_state_.q_IG.inverse() * camera_.p_C_I);
        J.template block<3, 3>(3, 12) = Matrix3<_S>::Identity();

        // Camera<_S> State Jacobian
        // MatrixX<_S> J = calcJ(imu_state_, cam_states_);

        MatrixX<_S> tempMat = MatrixX<_S>::Identity(15 + 6 * cam_states_.size() + 6,
                                                    15 + 6 * cam_states_.size());
        tempMat.block(15 + 6 * cam_states_.size(), 0, 6,
                      15 + 6 * cam_states_.size()) = J;

        // Augment the MSCKF covariance matrix
        MatrixX<_S> P_aug = tempMat * P_ * tempMat.transpose();

        MatrixX<_S> P_aug_sym = (P_aug + P_aug.transpose()) / 2.0;

        P_aug = P_aug_sym;

        // Break everything into appropriate structs
        cam_states_.push_back(cam_state);
        imu_covar_ = P_aug.template block<15, 15>(0, 0);

        cam_covar_.resize(P_aug.rows() - 15, P_aug.cols() - 15);
        cam_covar_ = P_aug.block(15, 15, P_aug.rows() - 15, P_aug.cols() - 15);

        imu_cam_covar_.resize(15, P_aug.cols() - 15);
        imu_cam_covar_ = P_aug.block(0, 15, 15, P_aug.cols() - 15);

        VectorX<_S> cov_diag = imu_covar_.diagonal();
      }

      // Updates the positions of tracked features at the current timestamp.
      void update(const std::vector<Vector2<_S>, Eigen::aligned_allocator<Vector2<_S>>> &measurements,
                  const std::vector<size_t> &feature_ids) {

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
            featureTrackToResidualize<_S> track_to_residualize;
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

      // Adds newly detected features to the filter.
      void addFeatures(const std::vector<Vector2<_S>, Eigen::aligned_allocator<Vector2<_S>>>& features,
                       const std::vector<size_t>& feature_ids) {
        // Assumes featureIDs match features
        // Original code is a bit confusing here. Seems to allow for repeated feature
        // IDs
        // Will assume feature IDs are unique per feature per call
        // TODO: revisit this assumption if necessary
        using camStateIter = typename std::vector<camState<_S>>::iterator;

        for (size_t i = 0; i < features.size(); i++) {
          size_t id = feature_ids[i];
          if (std::find(tracked_feature_ids_.begin(), tracked_feature_ids_.end(),
                        id) == tracked_feature_ids_.end()) {
            // New feature
            featureTrack<_S> track;
            track.feature_id = feature_ids[i];
            track.observations.push_back(features[i]);

            camStateIter cam_state_last = cam_states_.end() - 1;
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

      // Finds feature tracks that have been lost, removes them from the filter, and uses them
      // to update the camera states that observed them.
      void marginalize() {
        if (!feature_tracks_to_residualize_.empty()) {
          int num_passed, num_rejected, num_ransac, max_length, min_length;
          _S max_norm, min_norm;
          num_passed = 0;
          num_rejected = 0;
          num_ransac = 0;
          max_length = -1;
          min_length = std::numeric_limits<int>::max();
          max_norm = -1;
          min_norm = std::numeric_limits<_S>::infinity();

          std::vector<bool> valid_tracks;
          std::vector<Vector3<_S>, Eigen::aligned_allocator<Vector3<_S>>> p_f_G_vec;
          int total_nObs = 0;

          for (auto track = feature_tracks_to_residualize_.begin();
               track != feature_tracks_to_residualize_.end(); track++) {
            if (num_feature_tracks_residualized_ > 3 &&
                !checkMotion(track->observations.front(), track->cam_states)) {
              num_rejected += 1;
              valid_tracks.push_back(false);
              continue;
            }

            Vector3<_S> p_f_G;
            _S Jcost, RCOND;

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

            Vector3<_S> p_f_C1 = (track->cam_states[0].q_CG.toRotationMatrix()) *
              (p_f_G - track->cam_states[0].p_C_G);

            Eigen::Array<_S, 3, 1> p_f_G_array = p_f_G.array();

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
          MatrixX<_S> H_o = MatrixX<_S>::Zero(2 * total_nObs - 3 * num_passed,
                                              15 + 6 * cam_states_.size());
          MatrixX<_S> R_o = MatrixX<_S>::Zero(2 * total_nObs - 3 * num_passed,
                                              2 * total_nObs - 3 * num_passed);
          VectorX<_S> r_o(2 * total_nObs - 3 * num_passed);

          Vector2<_S> rep;
          rep << noise_params_.u_var_prime, noise_params_.v_var_prime;

          int stack_counter = 0;
          for (int iter = 0; iter < feature_tracks_to_residualize_.size(); iter++) {
            if (!valid_tracks[iter]) continue;

            featureTrackToResidualize<_S> track = feature_tracks_to_residualize_[iter];

            Vector3<_S> p_f_G = p_f_G_vec[iter];
            VectorX<_S> r_j = calcResidual(p_f_G, track.cam_states, track.observations);

            int nObs = track.observations.size();
            MatrixX<_S> R_j = (rep.replicate(nObs, 1)).asDiagonal();

            // Calculate H_o_j and residual
            MatrixX<_S> H_o_j, A_j;
            calcMeasJacobian(p_f_G, track.cam_state_indices, H_o_j, A_j);

            // Stacked residuals and friends
            VectorX<_S> r_o_j = A_j.transpose() * r_j;
            MatrixX<_S> R_o_j = A_j.transpose() * R_j * A_j;

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

      // Removes camera states that are not considered 'keyframes' (too close in distance or
      // angle to their neighboring camera states), and marginalizes their observations.
      void pruneRedundantStates() {
        // Cap number of cam states used in computation to max_cam_states
        if (cam_states_.size() < 20){
          return;
        }

        // Find two camera states to rmoved
        std::vector<size_t> rm_cam_state_ids;
        rm_cam_state_ids.clear();
        findRedundantCamStates(rm_cam_state_ids);

        // Find size of jacobian matrix
        size_t jacobian_row_size = 0;
        for (auto &feature : feature_tracks_) {
          std::vector<size_t> involved_cam_state_ids;
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
            std::vector<camState<_S>> feature_associated_cam_states;
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
              Vector3<_S> p_f_G;
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
        MatrixX<_S> H_x = MatrixX<_S>::Zero(jacobian_row_size, 15 + 6 * cam_states_.size());
        MatrixX<_S> R_x = MatrixX<_S>::Zero(jacobian_row_size, jacobian_row_size);
        VectorX<_S> r_x = VectorX<_S>::Zero(jacobian_row_size);
        int stack_counter = 0;

        Vector2<_S> rep;
        rep << noise_params_.u_var_prime, noise_params_.v_var_prime;

        for (auto &feature : feature_tracks_) {
          std::vector<size_t> involved_cam_state_ids;
          std::vector<Vector2<_S>, Eigen::aligned_allocator<Vector2<_S>>> involved_observations;
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

          std::vector<camState<_S>> involved_cam_states;
          std::vector<size_t> cam_state_indices;
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
          VectorX<_S> r_j =
            calcResidual(feature.p_f_G, involved_cam_states, involved_observations);

          MatrixX<_S> R_j = (rep.replicate(nObs, 1)).asDiagonal();

          MatrixX<_S> H_x_j, A_j;
          calcMeasJacobian(feature.p_f_G, cam_state_indices, H_x_j, A_j);

          // Stacked residuals and friends
          VectorX<_S> r_x_j = A_j.transpose() * r_j;
          MatrixX<_S> R_x_j = A_j.transpose() * R_j * A_j;

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
        std::vector<size_t> deleteIdx(0);

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
          std::vector<bool> to_keep(num_states, false);
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

          MatrixX<_S> prunedCamCovar;
          square_slice(cam_covar_, keepCovarIdx, prunedCamCovar);

          cam_covar_.resize(prunedCamCovar.rows(), prunedCamCovar.cols());
          cam_covar_ = prunedCamCovar;

          Matrix<_S, 15, Dynamic> prunedImuCamCovar;
          column_slice(imu_cam_covar_, keepCovarIdx, prunedImuCamCovar);

          imu_cam_covar_.resize(prunedImuCamCovar.rows(), prunedImuCamCovar.cols());
          imu_cam_covar_ = prunedImuCamCovar;
        }
      }

      // Removes camera states that no longer contain any active observations.
      void pruneEmptyStates() {
        int max_states = msckf_params_.max_cam_states;
        if (cam_states_.size() < max_states) return;
        std::vector<size_t> deleteIdx;
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

          MatrixX<_S> prunedCamCovar;
          square_slice(cam_covar_, keepCovarIdx, prunedCamCovar);
          cam_covar_.resize(prunedCamCovar.rows(), prunedCamCovar.cols());
          cam_covar_ = prunedCamCovar;

          Matrix<_S, 15, Dynamic> prunedImuCamCovar;
          column_slice(imu_cam_covar_, keepCovarIdx, prunedImuCamCovar);
          imu_cam_covar_.resize(prunedImuCamCovar.rows(), prunedImuCamCovar.cols());
          imu_cam_covar_ = prunedImuCamCovar;
        }

        // TODO: Additional outputs = deletedCamCovar (used to compute sigma),
        // deletedCamStates
      }

      // Once all images are processed, this method will marginalize any remaining feature tracks
      // and update the final state.
      void finish() {
        for (size_t i = 0; i < tracked_feature_ids_.size(); i++) {
          std::vector<size_t> camStateIndices;
          std::vector<camState<_S>> camStatesTemp;
          removeTrackedFeature(tracked_feature_ids_[i], camStatesTemp,
                               camStateIndices);

          if (camStatesTemp.size() >= msckf_params_.min_track_length) {
            featureTrackToResidualize<_S> track;

            if (feature_tracks_[i].feature_id != tracked_feature_ids_[i]) {
              for (typename std::vector<featureTrack<_S>>::iterator feat_track =
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

      // Calls for info:
      inline size_t getNumCamStates()
      {
        return cam_states_.size();
      }

      inline imuState<_S> getImuState()
      {
        return imu_state_;
      }

      inline std::vector<Vector3<_S>, Eigen::aligned_allocator<Vector3<_S>>> getMap()
      {
        return map_;
      }

      inline Camera<_S> getCamera()
      {
        return camera_;
      }

      inline camState<_S> getCamState(size_t i)
      {
        return cam_states_[i];
      }

      inline std::vector<camState<_S>> getCamStates() const
      {
        return cam_states_;
      }

      inline std::vector<camState<_S>> getPrunedStates()
      {
        std::sort(pruned_states_.begin(), pruned_states_.end(),
                  [](camState<_S> a, camState<_S> b)
                  {
                    return a.state_id < b.state_id;
                  });
        return pruned_states_;
      }

    private:
      Quaternion<_S> buildUpdateQuat(const Vector3<_S> &deltaTheta) {
        Vector3<_S> deltaq = 0.5 * deltaTheta;
        Quaternion<_S> updateQuat;
        // Replaced with squaredNorm() ***1x1 result so using sum instead of creating
        // another variable and then referencing the 0th index value***
        _S checkSum = deltaq.squaredNorm();
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

      inline void calcF(const imuState<_S> &imu_state_k,
                        const imuReading<_S> &measurement_k) {
        /* Multiplies the error state in the linearized continuous-time
           error state model */
        F_.setZero();

        Vector3<_S> omegaHat, aHat;
        omegaHat = measurement_k.omega - imu_state_k.b_g;
        aHat = measurement_k.a - imu_state_k.b_a;
        Matrix3<_S> C_IG = imu_state_k.q_IG.toRotationMatrix();

        F_.template block<3, 3>(0, 0) = -vectorToSkewSymmetric(omegaHat);
        F_.template block<3, 3>(0, 3) = -Matrix3<_S>::Identity();
        F_.template block<3, 3>(6, 0) = -C_IG.transpose() * vectorToSkewSymmetric(aHat);
        F_.template block<3, 3>(6, 9) = -C_IG.transpose();
        F_.template block<3, 3>(12, 6) = Matrix3<_S>::Identity();
      }

      inline void calcG(const imuState<_S> &imu_state_k) {
        /* Multiplies the noise std::vector in the linearized continuous-time
           error state model */
        G_.setZero();

        Matrix3<_S> C_IG = imu_state_k.q_IG.toRotationMatrix();

        G_.template block<3, 3>(0, 0) = -Matrix3<_S>::Identity();
        G_.template block<3, 3>(3, 3) = Matrix3<_S>::Identity();
        G_.template block<3, 3>(6, 6) = -C_IG.transpose();
        G_.template block<3, 3>(9, 9) = Matrix3<_S>::Identity();
      }

      void calcMeasJacobian(const Vector3<_S> &p_f_G,
                            const std::vector<size_t> &camStateIndices,
                            MatrixX<_S> &H_o_j,
                            MatrixX<_S> &A_j) {
        // Calculates H_o_j according to Mourikis 2007

        MatrixX<_S> H_f_j = MatrixX<_S>::Zero(2 * camStateIndices.size(), 3);
        MatrixX<_S> H_x_j =
          MatrixX<_S>::Zero(2 * camStateIndices.size(), 15 + 6 * cam_states_.size());

        for (int c_i = 0; c_i < camStateIndices.size(); c_i++) {
          size_t index = camStateIndices[c_i];
          Vector3<_S> p_f_C = cam_states_[index].q_CG.toRotationMatrix() *
            (p_f_G - cam_states_[index].p_C_G);

          _S X, Y, Z;

          X = p_f_C(0);
          Y = p_f_C(1);
          Z = p_f_C(2);

          // cout << "p_f_C: " << p_f_C.transpose() << ". X: " << X << ", Y: " << Y <<
          // ", Z: " << Z << endl;

          Matrix<_S, 2, 3> J_i;
          J_i << 1, 0, -X / Z, 0, 1, -Y / Z;
          J_i *= 1 / Z;

          // Enforce observability constraint, see propagation for citation
          Matrix<_S, 2, 6> A;
          A << J_i * vectorToSkewSymmetric(p_f_C),
            -J_i * cam_states_[index].q_CG.toRotationMatrix();

          Matrix<_S, 6, 1> u = Matrix<_S, 6, 1>::Zero();
          u.head(3) = cam_states_[index].q_CG.toRotationMatrix() * imu_state_.g;
          Vector3<_S> tmp = p_f_G - cam_states_[index].p_C_G;
          u.tail(3) = vectorToSkewSymmetric(tmp) * imu_state_.g;

          Matrix<_S, 2, 6> H_x =
            A - A * u * (u.transpose() * u).inverse() * u.transpose();
          Matrix<_S, 2, 3> H_f = -H_x.template block<2, 3>(0, 3);
          H_f_j.template block<2, 3>(2 * c_i, 0) = H_f;

          // Potential indexing problem zone
          H_x_j.template block<2, 6>(2 * c_i, 15 + 6 * (index)) = H_x;
        }

        int jacobian_row_size = 2 * camStateIndices.size();

        JacobiSVD<MatrixX<_S>> svd_helper(H_f_j, ComputeFullU | ComputeThinV);
        A_j = svd_helper.matrixU().rightCols(jacobian_row_size - 3);

        H_o_j = A_j.transpose() * H_x_j;
      }

      VectorX<_S> calcResidual(const Vector3<_S> &p_f_G,
                               const std::vector<camState<_S>> &camStates,
                               const std::vector<Vector2<_S>, Eigen::aligned_allocator<Vector2<_S>>> &observations) {
        // CALCRESIDUAL Calculates the residual for a feature position

        VectorX<_S> r_j = VectorX<_S>::Constant(2 * camStates.size(),
                                                std::numeric_limits<_S>::quiet_NaN());

        int iter = 0;
        for (auto state_i : camStates) {
          Vector3<_S> p_f_C = state_i.q_CG.toRotationMatrix() * (p_f_G - state_i.p_C_G);
          Vector2<_S> zhat_i_j = p_f_C.template head<2>() / p_f_C(2);

          r_j.template segment<2>(2 * iter) = observations[iter] - zhat_i_j;
          iter++;
        }

        return r_j;
      }

      bool checkMotion(const Vector2<_S> first_observation,
                       const std::vector<camState<_S>>& cam_states) {
        if (cam_states.size() < 2) {
          return false;
        }
        const camState<_S> &first_cam = cam_states.front();
        // const camState<_S>& last_cam = cam_states.back();

        Isometry3<_S> first_cam_pose;
        first_cam_pose.linear() = first_cam.q_CG.toRotationMatrix().transpose();
        first_cam_pose.translation() = first_cam.p_C_G;
        // Get the direction of the feature when it is first observed.
        // This direction is represented in the world frame.
        Vector3<_S> feature_direction;
        feature_direction << first_observation, 1.0;
        feature_direction = feature_direction / feature_direction.norm();
        feature_direction = first_cam_pose.linear() * feature_direction;

        _S max_ortho_translation = 0;

        for (auto second_cam_iter = cam_states.begin() + 1;
             second_cam_iter != cam_states.end(); second_cam_iter++) {
          Isometry3<_S> second_cam_pose;
          second_cam_pose.linear() =
            second_cam_iter->q_CG.toRotationMatrix().transpose();
          second_cam_pose.translation() = second_cam_iter->p_C_G;
          // Compute the translation between the first frame
          // and the last frame. We assume the first frame and
          // the last frame will provide the largest motion to
          // speed up the checking process.
          Vector3<_S> translation =
            second_cam_pose.translation() - first_cam_pose.translation();
          // translation = translation / translation.norm();
          _S parallel_translation = translation.transpose() * feature_direction;
          Vector3<_S> orthogonal_translation =
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

      void cost(const Isometry3<_S>& T_c0_ci,
                const Vector3<_S>& x, const Vector2<_S>& z,
                _S& e) const {
        // Compute hi1, hi2, and hi3 as Equation (37).
        const _S &alpha = x(0);
        const _S &beta = x(1);
        const _S &rho = x(2);

        Vector3<_S> h = T_c0_ci.linear() * Vector3<_S>(alpha, beta, 1.0) +
          rho * T_c0_ci.translation();
        _S &h1 = h(0);
        _S &h2 = h(1);
        _S &h3 = h(2);

        // Predict the feature observation in ci frame.
        Vector2<_S> z_hat(h1 / h3, h2 / h3);

        // Compute the residual.
        e = (z_hat - z).squaredNorm();
        return;
      }

      void findRedundantCamStates(std::vector<size_t> &rm_cam_state_ids) {
        // Ensure that there are enough cam_states to work with
        if (cam_states_.size() < 5) return;

        _S dist_thresh = msckf_params_.redundancy_distance_thresh;
        _S angle_thresh = msckf_params_.redundancy_angle_thresh;

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
          _S distance = (cam_pos-kf_pos).norm();
          _S angle = kf_q.angularDistance(cam_q);
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

        // Sort the elements in the output std::vector
        std::sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());
      }


      // Constraint on track to be marginalized based on Mahalanobis Gating
      // High Precision, Consistent EKF-based Visual-Inertial Odometry by Li et al.
      bool gatingTest(const MatrixX<_S>& H, const VectorX<_S>& r, const int& dof) {
        MatrixX<_S> P = MatrixX<_S>::Zero(15 + cam_covar_.rows(), 15 + cam_covar_.cols());
        P.template block<15, 15>(0, 0) = imu_covar_;
        if (cam_covar_.rows() != 0) {
          P.block(0, 15, 15, imu_cam_covar_.cols()) = imu_cam_covar_;
          P.block(15, 0, imu_cam_covar_.cols(), 15) = imu_cam_covar_.transpose();
          P.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
        }

        MatrixX<_S> P1 = H * P * H.transpose();
        MatrixX<_S> P2 =
          noise_params_.u_var_prime * MatrixX<_S>::Identity(H.rows(), H.rows());
        _S gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

        if (gamma < chi_squared_test_table[dof+1]) {
          // cout << "passed" << endl;
          return true;
        } else {
          // cout << "failed" << endl;
          return false;
        }
      }

      void generateInitialGuess(const Isometry3<_S>& T_c1_c2, const Vector2<_S>& z1,
                                const Vector2<_S>& z2, Vector3<_S>& p) const {
        // Construct a least square problem to solve the depth.
        Vector3<_S> m = T_c1_c2.linear() * Vector3<_S>(z1(0), z1(1), 1.0);

        Vector2<_S> A(0.0, 0.0);
        A(0) = m(0) - z2(0) * m(2);
        A(1) = m(1) - z2(1) * m(2);

        Vector2<_S> b(0.0, 0.0);
        b(0) = z2(0) * T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
        b(1) = z2(1) * T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

        // Solve for the depth.
        _S depth = (A.transpose() * A).inverse() * A.transpose() * b;
        p(0) = z1(0) * depth;
        p(1) = z1(1) * depth;
        p(2) = depth;
        return;
      }

      bool initializePosition(const std::vector<camState<_S>> &camStates,
                              const std::vector<Vector2<_S>, Eigen::aligned_allocator<Vector2<_S>>> &measurements,
                              Vector3<_S> &p_f_G) {
        // Organize camera poses and feature observations properly.
        std::vector<Isometry3<_S>, Eigen::aligned_allocator<Isometry3<_S>>>
          cam_poses(0);

        for (auto &cam : camStates) {
          // This camera pose will take a std::vector from this camera frame
          // to the world frame.
          Isometry3<_S> cam0_pose;
          cam0_pose.linear() = cam.q_CG.toRotationMatrix().transpose();
          cam0_pose.translation() = cam.p_C_G;

          cam_poses.push_back(cam0_pose);
        }

        // All camera poses should be modified such that it takes a
        // std::vector from the first camera frame in the buffer to this
        // camera frame.
        Isometry3<_S> T_c0_w = cam_poses[0];
        for (auto &pose : cam_poses) pose = pose.inverse() * T_c0_w;

        // Generate initial guess
        Vector3<_S> initial_position(0.0, 0.0, 0.0);
        generateInitialGuess(cam_poses[cam_poses.size() - 1], measurements[0],
                             measurements[measurements.size() - 1], initial_position);
        Vector3<_S> solution(initial_position(0) / initial_position(2),
                             initial_position(1) / initial_position(2),
                             1.0 / initial_position(2));

        // Apply Levenberg-Marquart method to solve for the 3d position.
        _S initial_damping = 1e-3;
        _S lambda = initial_damping;
        int inner_loop_max_iteration = 10;
        int outer_loop_max_iteration = 10;
        _S estimation_precision = 5e-7;
        int inner_loop_cntr = 0;
        int outer_loop_cntr = 0;
        bool is_cost_reduced = false;
        _S delta_norm = 0;
        // Compute the initial cost.
        _S total_cost = 0.0;
        for (int i = 0; i < cam_poses.size(); ++i) {
          _S this_cost = 0.0;
          cost(cam_poses[i], solution, measurements[i], this_cost);
          total_cost += this_cost;
        }

        // Outer loop.
        do {
          Matrix3<_S> A = Matrix3<_S>::Zero();
          Vector3<_S> b = Vector3<_S>::Zero();

          for (int i = 0; i < cam_poses.size(); ++i) {
            Matrix<_S, 2, 3> J;
            Vector2<_S> r;
            _S w;

            jacobian(cam_poses[i], solution, measurements[i], J, r, w);

            if (w == 1) {
              A += J.transpose() * J;
              b += J.transpose() * r;
            } else {
              _S w_square = w * w;
              A += w_square * J.transpose() * J;
              b += w_square * J.transpose() * r;
            }
          }

          // Inner loop.
          // Solve for the delta that can reduce the total cost.
          do {
            Matrix3<_S> damper = lambda * Matrix3<_S>::Identity();
            Vector3<_S> delta = (A + damper).ldlt().solve(b);
            Vector3<_S> new_solution = solution - delta;
            delta_norm = delta.norm();

            _S new_cost = 0.0;
            for (int i = 0; i < cam_poses.size(); ++i) {
              _S this_cost = 0.0;
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
        Vector3<_S> final_position(solution(0) / solution(2),
                                   solution(1) / solution(2), 1.0 / solution(2));

        // Check if the solution is valid. Make sure the feature
        // is in front of every camera frame observing it.
        bool is_valid_solution = true;
        for (const auto &pose : cam_poses) {
          Vector3<_S> position =
            pose.linear() * final_position + pose.translation();
          if (position(2) <= 0) {
            is_valid_solution = false;
            break;
          }
        }

        _S normalized_cost =
          total_cost / (2 * cam_poses.size() * cam_poses.size());

        VectorX<_S> cov_diag = imu_covar_.diagonal();

        _S pos_covar = cov_diag.segment(12, 3).norm();

        if (normalized_cost > msckf_params_.max_gn_cost_norm) {
          is_valid_solution = false;
        }

        // printf("Cost is: %f, normalized: %f, target: %f\n", total_cost,
        // normalized_cost, cost_threshold);

        // Convert the feature position to the world frame.
        p_f_G = T_c0_w.linear() * final_position + T_c0_w.translation();

        return is_valid_solution;
      }

      void jacobian(const Isometry3<_S>& T_c0_ci,
                    const Vector3<_S>& x, const Vector2<_S>& z,
                    Matrix<_S, 2, 3>& J, Vector2<_S>& r,
                    _S& w) const {
        // Compute hi1, hi2, and hi3 as Equation (37).
        const _S &alpha = x(0);
        const _S &beta = x(1);
        const _S &rho = x(2);

        Vector3<_S> h = T_c0_ci.linear() * Vector3<_S>(alpha, beta, 1.0) +
          rho * T_c0_ci.translation();
        _S &h1 = h(0);
        _S &h2 = h(1);
        _S &h3 = h(2);

        // Compute the Jacobian.
        Matrix3<_S> W;
        W.template leftCols<2>() = T_c0_ci.linear().template leftCols<2>();
        W.template rightCols<1>() = T_c0_ci.translation();

        J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
        J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

        // Compute the residual.
        Vector2<_S> z_hat(h1 / h3, h2 / h3);
        r = z_hat - z;

        // Compute the weight based on the residual.
        _S e = r.norm();
        _S huber_epsilon = 0.01;
        if (e <= huber_epsilon)
          w = 1.0;
        else
          w = huber_epsilon / (2 * e);

        return;
      }

      void measurementUpdate(const MatrixX<_S> &H_o,
                             const VectorX<_S> &r_o,
                             const MatrixX<_S> &R_o) {
        if (r_o.size() != 0) {
          // Build MSCKF covariance matrix
          MatrixX<_S> P = MatrixX<_S>::Zero(15 + cam_covar_.rows(), 15 + cam_covar_.cols());
          P.template block<15, 15>(0, 0) = imu_covar_;
          if (cam_covar_.rows() != 0) {
            P.block(0, 15, 15, imu_cam_covar_.cols()) = imu_cam_covar_;
            P.block(15, 0, imu_cam_covar_.cols(), 15) = imu_cam_covar_.transpose();
            P.block(15, 15, cam_covar_.rows(), cam_covar_.cols()) = cam_covar_;
          }

          MatrixX<_S> T_H, Q_1, R_n;
          VectorX<_S> r_n;

          // Put residuals in update-worthy form
          // Calculates T_H matrix according to Mourikis 2007
          HouseholderQR<MatrixX<_S>> qr(H_o);
          MatrixX<_S> Q = qr.householderQ();
          MatrixX<_S> R = qr.matrixQR().template triangularView<Upper>();

          VectorX<_S> nonZeroRows = R.rowwise().any();
          int numNonZeroRows = nonZeroRows.sum();

          T_H = MatrixX<_S>::Zero(numNonZeroRows, R.cols());
          Q_1 = MatrixX<_S>::Zero(Q.rows(), numNonZeroRows);

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
          MatrixX<_S> temp = T_H * P * T_H.transpose() + R_n;
          MatrixX<_S> K = (P * T_H.transpose()) * temp.inverse();

          // State Correction
          VectorX<_S> deltaX = K * r_n;

          // Update IMU state (from updateState matlab function defined in MSCKF.m)
          Quaternion<_S> q_IG_up = buildUpdateQuat(deltaX.template head<3>()) * imu_state_.q_IG;

          imu_state_.q_IG = q_IG_up;

          imu_state_.b_g += deltaX.template segment<3>(3);
          imu_state_.b_a += deltaX.template segment<3>(9);
          imu_state_.v_I_G += deltaX.template segment<3>(6);
          imu_state_.p_I_G += deltaX.template segment<3>(12);

          // Update Camera<_S> states
          for (size_t c_i = 0; c_i < cam_states_.size(); c_i++) {
            Quaternion<_S> q_CG_up = buildUpdateQuat(deltaX.template segment<3>(15 + 6 * c_i)) *
              cam_states_[c_i].q_CG;
            cam_states_[c_i].q_CG = q_CG_up.normalized();
            cam_states_[c_i].p_C_G += deltaX.template segment<3>(18 + 6 * c_i);
          }

          // Covariance correction
          MatrixX<_S> tempMat = MatrixX<_S>::Identity(15 + 6 * cam_states_.size(),
                                                      15 + 6 * cam_states_.size()) -
            K * T_H;

          MatrixX<_S> P_corrected, P_corrected_transpose;
          P_corrected = tempMat * P * tempMat.transpose() + K * R_n * K.transpose();
          // Enforce symmetry
          P_corrected_transpose = P_corrected.transpose();
          P_corrected += P_corrected_transpose;
          P_corrected /= 2;

          if(P_corrected.rows()-15!=cam_covar_.rows()){
            std::cout << "[P:" << P_corrected.rows() << "," << P_corrected.cols() << "]";
            std::cout << "[cam_covar_:" << cam_covar_.rows() << "," << cam_covar_.cols() << "]";
            std::cout << std::endl;
          }

          // TODO : Verify need for eig check on P_corrected here (doesn't seem too
          // important for now)
          imu_covar_ = P_corrected.template block<15, 15>(0, 0);

          // TODO: Check here
          cam_covar_ = P_corrected.block(15, 15, P_corrected.rows() - 15,
                                         P_corrected.cols() - 15);
          imu_cam_covar_ = P_corrected.block(0, 15, 15, P_corrected.cols() - 15);

          return;
        } else
          return;
      }

      imuState<_S> propogateImuStateRK(const imuState<_S> &imu_state_k,
                                       const imuReading<_S> &measurement_k) {
        imuState<_S> imuStateProp = imu_state_k;
        const _S dT(measurement_k.dT);

        Vector3<_S> omega_vec = measurement_k.omega - imu_state_k.b_g;
        Matrix4<_S> omega_psi = 0.5 * omegaMat(omega_vec);

        // Note: MSCKF Matlab code assumes quaternion form: -x,-y,-z,w
        //     Eigen quaternion is of form: w,x,y,z
        // Following computation accounts for this change

        Vector4<_S> y0, k0, k1, k2, k3, k4, k5, y_t;
        y0(0) = -imu_state_k.q_IG.x();
        y0(1) = -imu_state_k.q_IG.y();
        y0(2) = -imu_state_k.q_IG.z();
        y0(3) = imu_state_k.q_IG.w();

        k0 = omega_psi * (y0);
        k1 = omega_psi * (y0 + (k0 / 4.) * dT);
        k2 = omega_psi * (y0 + (k0 / 8. + k1 / 8.) * dT);
        k3 = omega_psi * (y0 + (-k1 / 2. + k2) * dT);
        k4 = omega_psi * (y0 + (k0 * 3. / 16. + k3 * 9. / 16.) * dT);
        k5 = omega_psi *
          (y0 +
           (-k0 * 3. / 7. + k1 * 2. / 7. + k2 * 12. / 7. - k3 * 12. / 7. + k4 * 8. / 7.) *
           dT);

        y_t = y0 + (7. * k0 + 32. * k2 + 12. * k3 + 32. * k4 + 7. * k5) * dT / 90.;

        Quaternion<_S> q(y_t(3), -y_t(0), -y_t(1), -y_t(2));
        q.normalize();

        imuStateProp.q_IG = q;
        Vector3<_S> delta_v_I_G = (((imu_state_k.q_IG.toRotationMatrix()).transpose()) *
                                   (measurement_k.a - imu_state_k.b_a) +
                                   imu_state_k.g) * dT;

        imuStateProp.v_I_G += delta_v_I_G;

        imuStateProp.p_I_G = imu_state_k.p_I_G + imu_state_k.v_I_G * dT;
        return imuStateProp;
      }

      void removeTrackedFeature(const size_t featureID,
                                std::vector<camState<_S>> &featCamStates,
                                std::vector<size_t> &camStateIndices){
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

      Vector3<_S> Triangulate(const Vector2<_S> &obs1,
                              const Vector2<_S> &obs2,
                              const Matrix3<_S> &C_12,
                              const Vector3<_S> &t_21_1){
        // Triangulate feature position given 2 observations and the transformation
        // between the 2 observation frames
        // Homogeneous normalized std::vector representations of observations:
        Vector3<_S> v1(0, 0, 1), v2(0, 0, 1);
        v1.template segment<2>(0) = obs1;
        v2.template segment<2>(0) = obs2;

        v1.normalize();
        v2.normalize();

        // scalarConstants:= [lambda_1; lambda_2] = lambda
        // A*lambda = t_21_1  -->  lambda = A\t_21_1
        Matrix<_S, 3, 2, ColMajor> A_;
        A_ << v1, -C_12 * v2;
        MatrixX<_S> A = A_;

        Vector2<_S> scalarConstants = A.colPivHouseholderQr().solve(t_21_1);

        return scalarConstants(0) * v1;
      }

    };

  // TODO: getIMUState;
  // TODO: other gets
} // End namespace

#endif /* MSCKF_HPP_ */
