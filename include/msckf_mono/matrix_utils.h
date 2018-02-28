
#pragma once

#include <msckf_mono/types.h>

namespace msckf_mono {
  template <typename _Scalar>
    inline Matrix3<_Scalar> vectorToSkewSymmetric(const Vector3<_Scalar>& Vec) {
      // Returns skew-symmetric form of a 3-d vector
      Matrix3<_Scalar> M;

      M << 0, -Vec(2), Vec(1),
        Vec(2), 0, -Vec(0),
        -Vec(1), Vec(0), 0;

      return M;
    }

  template <typename _Scalar>
    inline Matrix4<_Scalar> omegaMat(const Vector3<_Scalar>& omega) {
      // Compute the omega-matrix of a 3-d vector omega
      Matrix4<_Scalar> bigOmega;
      bigOmega.setZero();

      bigOmega.template block<3, 3>(0, 0) = -vectorToSkewSymmetric(omega);
      bigOmega.template block<3, 1>(0, 3) = omega;
      bigOmega.template block<1, 3>(3, 0) = -omega.transpose();

      return bigOmega;
    }

  template <typename _Scalar>
    inline Vector3<_Scalar> skewSymmetricToVector(const Matrix3<_Scalar>& Skew) {
      Vector3<_Scalar> Vec;
      Vec(0) = Skew(2, 1);
      Vec(1) = Skew(0, 2);
      Vec(3) = Skew(1, 0);

      return Vec;
    }

  template <typename _Scalar>
    _Scalar cond(const MatrixX<_Scalar>& M) {
      // Returns condition number calculation
      // Code credit: https://forum.kde.org/viewtopic.php?f=74&t=117430
      Eigen::JacobiSVD<MatrixX<_Scalar>> svd(M);
      _Scalar cond = svd.singularValues()(0) /
        svd.singularValues()(svd.singularValues().size() - 1);
      return cond;
    }

  // Slice columns and rows at indices inds from in to out
  //
  // in - M x M
  // inds - N x 1
  // out - N x N
  template <typename _Scalar>
    inline void square_slice(const MatrixX<_Scalar>& in,
                             const Eigen::VectorXi& inds,
                             MatrixX<_Scalar>& out ){
      int inds_size = inds.rows();
      out.resize(inds_size, inds_size);
      for(int i=0; i<inds_size; i++){
        for(int j=0; j<inds_size; j++){
          out(i,j) = in(inds(i),inds(j));
        }
      }
    }

  // Slice columns at indices inds from in to out
  //
  // in - R x M
  // inds - N x 1
  // out - R x N
  template <typename _Scalar, int _Rows>
    inline void column_slice(const Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic>& in,
                             const Eigen::VectorXi& inds,
                             Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic>& out){
      int inds_size = inds.rows();
      int rows = in.rows();
      out.resize(rows, inds_size);
      for(int i=0; i<rows; i++){
        for(int j=0; j<inds_size; j++){
          out(i,j) = in(i,inds(j));
        }
      }
    }


}
