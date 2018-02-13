/*
 * noiseparams.h
 *
 *  Created on: Jan 24, 2017
 *      Author: sidmys
 */

#ifndef NOISEPARAMS_H_
#define NOISEPARAMS_H_

#include <Eigen/Dense>

namespace msckf {
struct noiseParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double u_var_prime, v_var_prime;
  Eigen::Matrix<double, 12, 12> Q_imu;
  Eigen::Matrix<double, 15, 15> initial_imu_covar;
};
}

#endif /* NOISEPARAMS_H_ */
