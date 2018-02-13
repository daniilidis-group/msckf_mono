/*
 * camera.h
 *
 *  Created on: Feb 5, 2017
 *      Author: sidmys
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace msckf {
struct Camera {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double c_u, c_v, f_u, f_v, b;
  Eigen::Quaternion<double> q_CI;
  Eigen::Vector3d p_C_I;
};
}

#endif /* CAMERA_H_ */
