/*
 * measurement.h
 *
 *  Created on: Feb 5, 2017
 *      Author: sidmys
 */

#ifndef MEASUREMENT_H_
#define MEASUREMENT_H_

#include <Eigen/Dense>

namespace msckf {
struct measurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d omega, a;
  double dT;
};
}

#endif /* MEASUREMENT_H_ */
