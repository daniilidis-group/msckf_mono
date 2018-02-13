/*
 * camstates.h
 *
 *  Created on: Jan 24, 2017
 *      Author: sidmys
 */

#ifndef CAMSTATES_H_
#define CAMSTATES_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstddef>
#include <vector>

namespace msckf {
struct camState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d p_C_G;
  Eigen::Quaternion<double> q_CG;
  double time;
  int state_id;
  int last_correlated_id;
  std::vector<std::size_t> tracked_feature_ids;
};
}
#endif /* CAMSTATES_H_ */
