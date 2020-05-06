
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

  // Some simple utilities for manipulating rotation and translation pairs
  template<typename _Scalar>
    Transform<_Scalar> compose(const Transform<_Scalar>& a, const Transform<_Scalar>& b){
      Quaternion<_Scalar> R = a.first * b.first;
      Point<_Scalar> t = a.first * b.second + a.second;
      return std::make_pair( R, t );
    }

  template<typename _Scalar>
    Transform<_Scalar> inverse(const Transform<_Scalar>& a){
      Quaternion<_Scalar> R = a.first.inverse();
      Point<_Scalar> t = R * _Scalar(-1.) * a.second;
      return std::make_pair( R, t );
    }

  template<typename _O, int _D0, int _D1, typename _I>
    void vector_to_eigen(const std::vector<_I>& data, Eigen::Matrix<_O,_D0,_D1>& m) {
      assert(data.size() == _D0 * _D1);

      for(int i=0; i<_D0; i++){
        for(int j=0; j<_D1; j++){
          m(i,j) = data[i*_D1 + j];
        }
      }
    }

  template<typename _O, int _D0, int _D1, typename _I>
    Eigen::Matrix<_O,_D0,_D1> vector_to_eigen(const std::vector<_I>& data) {
      Eigen::Matrix<_O,_D0,_D1> m;
      vector_to_eigen<_O,_D0,_D1,_I>(data, m);
      return m;
    }

  template<typename _S>
    void transform_from_mats(const Matrix3<_S> R, const Vector3<_S> t, Matrix4<_S>& T) {
      // for(int i=0; i<3; i++)
      //   for(int j=0;j <3; j++)
      //     T(i,j) = R(i,j);

      // for(int i=0; i<3; i++)
      //   T(i,3) = t(i,0);

      T.template block<3,3>(0,0) = R;
      T.template block<3,1>(0,3) = t;

      T(3,0) = 0.;
      T(3,1) = 0.;
      T(3,2) = 0.;
      T(3,3) = 1.;
    }
}
