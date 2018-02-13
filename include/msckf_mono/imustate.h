/*
 * imustate.h
 *
 *  Created on: Feb 5, 2017
 *      Author: sidmys
 */

#ifndef IMUSTATE_H_
#define IMUSTATE_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace msckf {
struct imuState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d p_I_G, v_I_G, b_g, b_a, g;
  Eigen::Vector3d p_I_G_null, v_I_G_null;
  Eigen::Quaternion<double> q_IG, q_IG_null;
};
}

#endif /* IMUSTATE_H_ */
