/*
 * msckf.hpp
 *
 *  Created on: Feb 5, 2017
 *      Author: sidmys
 */

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

#include "camera.h"
#include "camstates.h"
#include "imustate.h"
#include "measurement.h"
#include "noiseparams.h"

using namespace std;
using namespace Eigen;

/* Notes to self:
 *		- Noiseparams in calcGNPoseEst
 *		- The thing with the quaternions being inverted
 */
namespace msckf {
struct MSCKFParams {
  double max_gn_cost_norm, min_rcond, translation_threshold;
  double redundancy_angle_thresh, redundancy_distance_thresh;
  size_t min_track_length, max_track_length, max_cam_states;
};

struct featureTrackToResidualize {
  size_t feature_id;
  vector<Vector2d, Eigen::aligned_allocator<Vector2d>> observations;

  vector<camState> cam_states;
  vector<size_t> cam_state_indices;

  bool initialized;
  Vector3d p_f_G;

  featureTrackToResidualize() : initialized(false) {}
};

struct featureTrack {
  size_t feature_id;
  vector<Vector2d, Eigen::aligned_allocator<Vector2d>> observations;

  vector<size_t> cam_state_indices; // state_ids of cam states corresponding to observations

  bool initialized;
  Vector3d p_f_G;

  featureTrack() : initialized(false) {}
};

class MSCKF {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MSCKF();
  // Initializes the filter state and parameters.
  void initialize(const Camera& camera,
		  const noiseParams& noise_params,
		  const MSCKFParams& msckf_params,
		  const imuState& imu_state);
  // Given an IMU measurement, propagates the latest camera pose to the timestamp in measurement
  // using the acceleration and angular velocity in measurement.
  void propagate(measurement &measurement_);
  // Generates a new camera state and adds it to the full state and covariance.
  void augmentState(const int& state_k, const double& time);
  // Updates the positions of tracked features at the current timestamp.
  void update(const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &measurements,
	      const vector<size_t> &featureIDs);
  // Adds newly detected features to the filter.
  void addFeatures(const vector<Vector2d, Eigen::aligned_allocator<Vector2d>>& features,
		   const vector<size_t>& featureIDs);
  // Finds feature tracks that have been lost, removes them from the filter, and uses them
  // to update the camera states that observed them.
  void marginalize();
  // Removes camera states that are not considered 'keyframes' (too close in distance or
  // angle to their neighboring camera states), and marginalizes their observations.
  void pruneRedundantStates();
  // Removes camera states that no longer contain any active observations.
  void pruneEmptyStates();
  // Once all images are processed, this method will marginalize any remaining feature tracks
  // and update the final state.
  void finish();

  // Calls for info:
  inline size_t getNumCamStates() {return cam_states_.size();}
  inline imuState getImuState() {return imu_state_;}
  inline vector<Vector3d, Eigen::aligned_allocator<Vector3d>> getMap() {return map_;}
  inline Camera getCamera() { return camera_;}
  inline camState getCamState(size_t i) {return cam_states_[i];}
  inline vector<camState> getCamStates() const { return cam_states_; }
  inline vector<camState> getPrunedStates()
  {
    std::sort(pruned_states_.begin(), pruned_states_.end(),
        [](camState a, camState b)
        {
          return a.state_id < b.state_id;
        });
    return pruned_states_;
  }

 private:
  Camera camera_;
  noiseParams noise_params_;
  MSCKFParams msckf_params_;
  // prunedStates;
  vector<featureTrack> feature_tracks_;
  vector<size_t> tracked_feature_ids_;

  vector<featureTrackToResidualize> feature_tracks_to_residualize_;
  size_t num_feature_tracks_residualized_;
  vector<size_t> tracks_to_remove_;
  size_t last_feature_id_;

  imuState imu_state_;
  vector<camState> cam_states_;

  vector<camState> pruned_states_;
  vector<Vector3d, Eigen::aligned_allocator<Vector3d>> map_;

  Matrix<double,15,15> imu_covar_;
  MatrixXd cam_covar_;
  Matrix<double,15,Dynamic> imu_cam_covar_;

  map<int, double> chi_squared_test_table;
  Vector3d pos_init_;
  Quaterniond quat_init_;

  Quaternion<double> buildUpdateQuat(const Vector3d &deltaTheta);
  Matrix<double,15,15> calcF(const imuState &imuState_k,
			     const measurement &measurement_k);
  Matrix<double,15,12> calcG(const imuState &imuState_k);
  void calcMeasJacobian(const Vector3d &p_f_G,
			const vector<size_t> &camStateIndices,
			MatrixXd &H_o_j,
			MatrixXd &A_j);
  VectorXd calcResidual(const Vector3d &p_f_G,
			const vector<camState> &camStates,
			const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &observations);
  bool checkMotion(const Vector2d first_observation,
                   const vector<camState>& cam_states);
  double cond(MatrixXd M);
  void cost(const Eigen::Isometry3d& T_c0_ci,
                   const Eigen::Vector3d& x, const Eigen::Vector2d& z,
                   double& e) const;
  void findRedundantCamStates(vector<size_t> &rm_cam_state_ids);
  bool gatingTest(const MatrixXd& H, const VectorXd& r, const int& dof);
  void generateInitialGuess(const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
                            const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;
  bool initializePosition(const vector<camState> &camStates,
                          const vector<Vector2d, Eigen::aligned_allocator<Vector2d>> &measurements,
                          Vector3d &p_f_G);
  void jacobian(const Eigen::Isometry3d& T_c0_ci,
                const Eigen::Vector3d& x, const Eigen::Vector2d& z,
                Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
                double& w) const;
  void measurementUpdate(const MatrixXd &H_o, const VectorXd &r_o, const MatrixXd &R_o);
  Matrix<double,4,4> omegaMat(Vector3d omega);
  imuState propogateImuState(const imuState &imuState_k,
			     const measurement &measurement_k);
  imuState propogateImuStateRK(const imuState &imuState_k, const
			       measurement &measurement_k);
  void removeTrackedFeature(const size_t featureID,
			    vector<camState> &featCamStates,
			    vector<size_t> &camStateIndices);
  Vector3d skewSymmetricToVector(Matrix3d Skew);
  Vector3d Triangulate(const Vector2d &obs1,
		       const Vector2d &obs2,
		       const Matrix3d &C_12,
		       const Vector3d &t_21_1);
  Matrix3d vectorToSkewSymmetric(Vector3d Vec);

  void cam_covar_slice(VectorXi& inds, MatrixXd& tmp);
  void imu_cam_covar_slice(VectorXi& inds, Matrix<double,15,Dynamic>& tmp);
};

// TODO: getIMUState;
// TODO: other gets
} // End namespace

#endif /* MSCKF_HPP_ */
