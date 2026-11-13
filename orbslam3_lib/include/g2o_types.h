/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "converter.h"

#include <frame.h>
#include <key_frame.h>
#include <math.h>

#include <memory>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sba/types_sba.h>

namespace ORB_SLAM3
{

class key_frame;
class frame;
class geometric_camera;

typedef Eigen::Matrix<double, 6, 1>   vector_6d;
typedef Eigen::Matrix<double, 9, 1>   vector_9d;
typedef Eigen::Matrix<double, 12, 1>  vector_12d;
typedef Eigen::Matrix<double, 15, 1>  vector_15d;
typedef Eigen::Matrix<double, 12, 12> matrix_12d;
typedef Eigen::Matrix<double, 15, 15> matrix_15d;
typedef Eigen::Matrix<double, 9, 9>   matrix_9d;

Eigen::Matrix3d exp_so3(const double x, const double y, const double z);
Eigen::Matrix3d exp_so3(const Eigen::Vector3d &w);

Eigen::Vector3d log_so3(const Eigen::Matrix3d &R);

Eigen::Matrix3d inverse_right_jacobian_so3(const Eigen::Vector3d &v);
Eigen::Matrix3d right_jacobian_so3(const Eigen::Vector3d &v);
Eigen::Matrix3d right_jacobian_so3(const double x, const double y, const double z);

Eigen::Matrix3d skew(const Eigen::Vector3d &w);
Eigen::Matrix3d inverse_right_jacobian_so3(const double x, const double y, const double z);

template <typename T_ = double> Eigen::Matrix<T_, 3, 3> normalize_rotation(const Eigen::Matrix<T_, 3, 3> &R)
{
    Eigen::JacobiSVD<Eigen::Matrix<T_, 3, 3>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

class imu_cam_pose
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    imu_cam_pose() {}
    imu_cam_pose(key_frame *pKF, camera_view cameras);
    imu_cam_pose(frame *pF, const camera_vector &cameras, const imu::calib &imu_calibration);
    imu_cam_pose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, key_frame *pKF, camera_view cameras);

    void set_parameters(const std::vector<Eigen::Matrix3d> &_Rcw,
                        const std::vector<Eigen::Vector3d> &_tcw,
                        const std::vector<Eigen::Matrix3d> &_Rbc,
                        const std::vector<Eigen::Vector3d> &_tbc,
                        const double                       &_bf);

    void            update(const double *pu);                                                  // update in the imu reference
    void            update_world_reference(const double *pu);                                  // update in the world reference
    Eigen::Vector2d project(const Eigen::Vector3d &point_world, int cam_idx = 0) const;        // Mono
    Eigen::Vector3d project_stereo(const Eigen::Vector3d &point_world, int cam_idx = 0) const; // Stereo
    bool            is_depth_positive(const Eigen::Vector3d &point_world, int cam_idx = 0) const;

  public:
    // For IMU
    Eigen::Matrix3d rotation_world_body;
    Eigen::Vector3d translation_world_body;

    // For set of cameras
    std::vector<Eigen::Matrix3d> rotation_camera_world;
    std::vector<Eigen::Vector3d> translation_camera_world;
    std::vector<Eigen::Matrix3d> rotation_camera_body;
    std::vector<Eigen::Matrix3d> rotation_body_camera;
    std::vector<Eigen::Vector3d> translation_camera_body;
    std::vector<Eigen::Vector3d> translation_body_camera;
    double                       bf;
    camera_view                  cameras;

    // For posegraph 4DoF
    Eigen::Matrix3d initial_rotation_world_body;
    Eigen::Matrix3d rotation_delta;

    int its;
};

class inverse_depth_point
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    inverse_depth_point() {}
    inverse_depth_point(double _rho, double _u, double _v, geometric_camera &camera);

    void update(const double *pu);

    double rho;
    double u;
    double v; // they are not variables, observation in the host frame

    double fx;
    double fy;
    double cx;
    double cy;
    double bf; // from host frame

    int its;
};

// Optimizable parameters are IMU pose
class vertex_pose : public g2o::BaseVertex<6, imu_cam_pose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_pose(key_frame *pKF, camera_view cameras) { setEstimate(imu_cam_pose(pKF, cameras)); }
    vertex_pose(frame *pF, const camera_vector &cameras, const imu::calib &imu_calibration) { setEstimate(imu_cam_pose(pF, cameras, imu_calibration)); }

    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        _estimate.update(update_);
        updateCache();
    }
};

class vertex_pose_4dof : public g2o::BaseVertex<4, imu_cam_pose>
{
    // Translation and yaw are the only optimizable variables
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_pose_4dof(key_frame *pKF, camera_view cameras) { setEstimate(imu_cam_pose(pKF, cameras)); }
    vertex_pose_4dof(frame *pF, const camera_vector &cameras, const imu::calib &imu_calibration) { setEstimate(imu_cam_pose(pF, cameras, imu_calibration)); }
    vertex_pose_4dof(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, key_frame *pKF, camera_view cameras) { setEstimate(imu_cam_pose(_Rwc, _twc, pKF, cameras)); }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        double update6DoF[6];
        update6DoF[0] = 0;
        update6DoF[1] = 0;
        update6DoF[2] = update_[0];
        update6DoF[3] = update_[1];
        update6DoF[4] = update_[2];
        update6DoF[5] = update_[3];
        _estimate.update_world_reference(update6DoF);
        updateCache();
    }
};

class vertex_velocity : public g2o::BaseVertex<3, Eigen::Vector3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_velocity() {}
    vertex_velocity(key_frame *pKF);
    vertex_velocity(frame *pF);

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate() + uv);
    }
};

class vertex_gyro_bias : public g2o::BaseVertex<3, Eigen::Vector3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_gyro_bias() {}
    vertex_gyro_bias(key_frame *pKF);
    vertex_gyro_bias(frame *pF);

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate() + ubg);
    }
};

class vertex_acc_bias : public g2o::BaseVertex<3, Eigen::Vector3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_acc_bias() {}
    vertex_acc_bias(key_frame *pKF);
    vertex_acc_bias(frame *pF);

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate() + uba);
    }
};

// Gravity direction vertex
class gravity_direction
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    gravity_direction() {}
    gravity_direction(Eigen::Matrix3d pRwg) : rotation_world_gravity(pRwg) {}

    void update(const double *pu) { rotation_world_gravity = rotation_world_gravity * exp_so3(pu[0], pu[1], 0.0); }

    Eigen::Matrix3d rotation_world_gravity = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rotation_gravity_world = Eigen::Matrix3d::Identity();

    int its = 0;
};

class vertex_gravity_direction : public g2o::BaseVertex<2, gravity_direction>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_gravity_direction() {}
    vertex_gravity_direction(Eigen::Matrix3d pRwg) { setEstimate(gravity_direction(pRwg)); }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        _estimate.update(update_);
        updateCache();
    }
};

// scale vertex
class vertex_scale : public g2o::BaseVertex<1, double>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_scale() { setEstimate(1.0); }
    vertex_scale(double ps) { setEstimate(ps); }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() { setEstimate(1.0); }

    virtual void oplusImpl(const double *update_) { setEstimate(estimate() * exp(*update_)); }
};

// Inverse depth point (just one parameter, inverse depth at the host frame)
class vertex_inverse_depth : public g2o::BaseVertex<1, inverse_depth_point>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    vertex_inverse_depth() {}
    vertex_inverse_depth(double invDepth, double u, double v, geometric_camera &camera) { setEstimate(inverse_depth_point(invDepth, u, v, camera)); }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl(const double *update_)
    {
        _estimate.update(update_);
        updateCache();
    }
};

class edge_mono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, vertex_pose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_mono(int cam_idx_ = 0) : cam_idx(cam_idx_) {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const g2o::VertexSBAPointXYZ *VPoint = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        const vertex_pose            *VPose  = static_cast<const vertex_pose *>(_vertices[1]);
        const Eigen::Vector2d         obs(_measurement);
        _error = obs - VPose->estimate().project(VPoint->estimate(), cam_idx);
    }

    virtual void linearizeOplus();

    bool is_depth_positive()
    {
        const g2o::VertexSBAPointXYZ *VPoint = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        const vertex_pose            *VPose  = static_cast<const vertex_pose *>(_vertices[1]);
        return VPose->estimate().is_depth_positive(VPoint->estimate(), cam_idx);
    }

    Eigen::Matrix<double, 2, 9> get_jacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 2, 9> J;
        J.block<2, 3>(0, 0) = _jacobianOplusXi;
        J.block<2, 6>(0, 3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double, 9, 9> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 2, 9> J;
        J.block<2, 3>(0, 0) = _jacobianOplusXi;
        J.block<2, 6>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }

  public:
    const int cam_idx;
};

class edge_mono_only_pose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, vertex_pose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_mono_only_pose(const Eigen::Vector3f &Xw_, int cam_idx_ = 0) : point_world(Xw_.cast<double>()), cam_idx(cam_idx_) {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_pose    *VPose = static_cast<const vertex_pose *>(_vertices[0]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().project(point_world, cam_idx);
    }

    virtual void linearizeOplus();

    bool is_depth_positive()
    {
        const vertex_pose *VPose = static_cast<const vertex_pose *>(_vertices[0]);
        return VPose->estimate().is_depth_positive(point_world, cam_idx);
    }

    Eigen::Matrix<double, 6, 6> get_hessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

  public:
    const Eigen::Vector3d point_world;
    const int             cam_idx;
};

class edge_stereo : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, vertex_pose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_stereo(int cam_idx_ = 0) : cam_idx(cam_idx_) {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const g2o::VertexSBAPointXYZ *VPoint = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        const vertex_pose            *VPose  = static_cast<const vertex_pose *>(_vertices[1]);
        const Eigen::Vector3d         obs(_measurement);
        _error = obs - VPose->estimate().project_stereo(VPoint->estimate(), cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double, 3, 9> get_jacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 3, 9> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 6>(0, 3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double, 9, 9> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 3, 9> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 6>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }

  public:
    const int cam_idx;
};

class edge_stereo_only_pose : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, vertex_pose>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_stereo_only_pose(const Eigen::Vector3f &Xw_, int cam_idx_ = 0) : point_world(Xw_.cast<double>()), cam_idx(cam_idx_) {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_pose    *VPose = static_cast<const vertex_pose *>(_vertices[0]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - VPose->estimate().project_stereo(point_world, cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double, 6, 6> get_hessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

  public:
    const Eigen::Vector3d point_world; // 3D point coordinates
    const int             cam_idx;
};

class edge_inertial : public g2o::BaseMultiEdge<9, vector_9d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_inertial(imu::preintegrated *pInt);

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void         computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double, 24, 24> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 24> J;
        J.block<9, 6>(0, 0)  = _jacobianOplus[0];
        J.block<9, 3>(0, 6)  = _jacobianOplus[1];
        J.block<9, 3>(0, 9)  = _jacobianOplus[2];
        J.block<9, 3>(0, 12) = _jacobianOplus[3];
        J.block<9, 6>(0, 15) = _jacobianOplus[4];
        J.block<9, 3>(0, 21) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 18, 18> get_hessian_no_pose_1()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 18> J;
        J.block<9, 3>(0, 0)  = _jacobianOplus[1];
        J.block<9, 3>(0, 3)  = _jacobianOplus[2];
        J.block<9, 3>(0, 6)  = _jacobianOplus[3];
        J.block<9, 6>(0, 9)  = _jacobianOplus[4];
        J.block<9, 3>(0, 15) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 9, 9> get_hessian_2()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 9> J;
        J.block<9, 6>(0, 0) = _jacobianOplus[4];
        J.block<9, 3>(0, 6) = _jacobianOplus[5];
        return J.transpose() * information() * J;
    }

    const Eigen::Matrix3d rotation_gyro_bias_jacobian;
    const Eigen::Matrix3d velocity_gyro_bias_jacobian;
    const Eigen::Matrix3d position_gyro_bias_jacobian;
    const Eigen::Matrix3d velocity_accel_bias_jacobian;
    const Eigen::Matrix3d position_accel_bias_jacobian;
    imu::preintegrated   *preintegration;
    const double          dt;
    Eigen::Vector3d       g;
};

// Edge inertial whre gravity is included as optimizable variable and it is not supposed to be pointing in -z axis, as well as scale
class edge_inertial_gs : public g2o::BaseMultiEdge<9, vector_9d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_inertial_gs(imu::preintegrated *pInt);

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void         computeError();
    virtual void linearizeOplus();

    const Eigen::Matrix3d rotation_gyro_bias_jacobian;
    const Eigen::Matrix3d velocity_gyro_bias_jacobian;
    const Eigen::Matrix3d position_gyro_bias_jacobian;
    const Eigen::Matrix3d velocity_accel_bias_jacobian;
    const Eigen::Matrix3d position_accel_bias_jacobian;
    imu::preintegrated   *preintegration;
    const double          dt;
    Eigen::Vector3d       g;
    Eigen::Vector3d       gravity_imu;

    Eigen::Matrix<double, 27, 27> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 27> J;
        J.block<9, 6>(0, 0)  = _jacobianOplus[0];
        J.block<9, 3>(0, 6)  = _jacobianOplus[1];
        J.block<9, 3>(0, 9)  = _jacobianOplus[2];
        J.block<9, 3>(0, 12) = _jacobianOplus[3];
        J.block<9, 6>(0, 15) = _jacobianOplus[4];
        J.block<9, 3>(0, 21) = _jacobianOplus[5];
        J.block<9, 2>(0, 24) = _jacobianOplus[6];
        J.block<9, 1>(0, 26) = _jacobianOplus[7];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 27, 27> get_hessian_2()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 27> J;
        J.block<9, 3>(0, 0)  = _jacobianOplus[2];
        J.block<9, 3>(0, 3)  = _jacobianOplus[3];
        J.block<9, 2>(0, 6)  = _jacobianOplus[6];
        J.block<9, 1>(0, 8)  = _jacobianOplus[7];
        J.block<9, 3>(0, 9)  = _jacobianOplus[1];
        J.block<9, 3>(0, 12) = _jacobianOplus[5];
        J.block<9, 6>(0, 15) = _jacobianOplus[0];
        J.block<9, 6>(0, 21) = _jacobianOplus[4];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 9, 9> get_hessian_3()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 9> J;
        J.block<9, 3>(0, 0) = _jacobianOplus[2];
        J.block<9, 3>(0, 3) = _jacobianOplus[3];
        J.block<9, 2>(0, 6) = _jacobianOplus[6];
        J.block<9, 1>(0, 8) = _jacobianOplus[7];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 1, 1> get_hessian_scale()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 1> J = _jacobianOplus[7];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 3, 3> get_hessian_bias_gyro()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 3> J = _jacobianOplus[2];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 3, 3> get_hessian_bias_acc()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 3> J = _jacobianOplus[3];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 2, 2> get_hessian_gdir()
    {
        linearizeOplus();
        Eigen::Matrix<double, 9, 2> J = _jacobianOplus[6];
        return J.transpose() * information() * J;
    }
};

class edge_gyro_rw : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, vertex_gyro_bias, vertex_gyro_bias>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_gyro_rw() {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_gyro_bias *VG1 = static_cast<const vertex_gyro_bias *>(_vertices[0]);
        const vertex_gyro_bias *VG2 = static_cast<const vertex_gyro_bias *>(_vertices[1]);
        _error                      = VG2->estimate() - VG1->estimate();
    }

    virtual void linearizeOplus()
    {
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
    }

    Eigen::Matrix<double, 6, 6> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 3>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }

    Eigen::Matrix3d get_hessian_2()
    {
        linearizeOplus();
        return _jacobianOplusXj.transpose() * information() * _jacobianOplusXj;
    }
};

class edge_acc_rw : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, vertex_acc_bias, vertex_acc_bias>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_acc_rw() {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_acc_bias *VA1 = static_cast<const vertex_acc_bias *>(_vertices[0]);
        const vertex_acc_bias *VA2 = static_cast<const vertex_acc_bias *>(_vertices[1]);
        _error                     = VA2->estimate() - VA1->estimate();
    }

    virtual void linearizeOplus()
    {
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
    }

    Eigen::Matrix<double, 6, 6> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = _jacobianOplusXi;
        J.block<3, 3>(0, 3) = _jacobianOplusXj;
        return J.transpose() * information() * J;
    }

    Eigen::Matrix3d get_hessian_2()
    {
        linearizeOplus();
        return _jacobianOplusXj.transpose() * information() * _jacobianOplusXj;
    }
};

class constraint_pose_imu
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    constraint_pose_imu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_, const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_, const matrix_15d &H_)
        : rotation_world_body(Rwb_), translation_world_body(twb_), vwb(vwb_), bg(bg_), ba(ba_), information_matrix(H_)
    {
        information_matrix = (information_matrix + information_matrix) / 2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> es(information_matrix);
        Eigen::Matrix<double, 15, 1>                                 eigs = es.eigenvalues();
        for (int i = 0; i < 15; i++)
        {
            if (eigs[i] < 1e-12)
            {
                eigs[i] = 0;
            }
        }
        information_matrix = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
    }

    Eigen::Matrix3d rotation_world_body;
    Eigen::Vector3d translation_world_body;
    Eigen::Vector3d vwb;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    matrix_15d      information_matrix;
};

class edge_prior_pose_imu : public g2o::BaseMultiEdge<15, vector_15d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    edge_prior_pose_imu(constraint_pose_imu *c);

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void         computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double, 15, 15> get_hessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 15, 15> J;
        J.block<15, 6>(0, 0)  = _jacobianOplus[0];
        J.block<15, 3>(0, 6)  = _jacobianOplus[1];
        J.block<15, 3>(0, 9)  = _jacobianOplus[2];
        J.block<15, 3>(0, 12) = _jacobianOplus[3];
        return J.transpose() * information() * J;
    }

    Eigen::Matrix<double, 9, 9> get_hessian_no_pose()
    {
        linearizeOplus();
        Eigen::Matrix<double, 15, 9> J;
        J.block<15, 3>(0, 0) = _jacobianOplus[1];
        J.block<15, 3>(0, 3) = _jacobianOplus[2];
        J.block<15, 3>(0, 6) = _jacobianOplus[3];
        return J.transpose() * information() * J;
    }
    Eigen::Matrix3d rotation_world_body;
    Eigen::Vector3d translation_world_body;
    Eigen::Vector3d vwb;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
};

// Priors for biases
class edge_prior_acc : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, vertex_acc_bias>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_prior_acc(const Eigen::Vector3f &bprior_) : bprior(bprior_.cast<double>()) {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_acc_bias *VA = static_cast<const vertex_acc_bias *>(_vertices[0]);
        _error                    = bprior - VA->estimate();
    }
    virtual void linearizeOplus();

    Eigen::Matrix<double, 3, 3> get_hessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

    const Eigen::Vector3d bprior;
};

class edge_prior_gyro : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, vertex_gyro_bias>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_prior_gyro(const Eigen::Vector3f &bprior_) : bprior(bprior_.cast<double>()) {}

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_gyro_bias *VG = static_cast<const vertex_gyro_bias *>(_vertices[0]);
        _error                     = bprior - VG->estimate();
    }
    virtual void linearizeOplus();

    Eigen::Matrix<double, 3, 3> get_hessian()
    {
        linearizeOplus();
        return _jacobianOplusXi.transpose() * information() * _jacobianOplusXi;
    }

    const Eigen::Vector3d bprior;
};

class edge_4dof : public g2o::BaseBinaryEdge<6, vector_6d, vertex_pose_4dof, vertex_pose_4dof>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    edge_4dof(const Eigen::Matrix4d &deltaT)
    {
        relative_pose        = deltaT;
        relative_rotation    = deltaT.block<3, 3>(0, 0);
        relative_translation = deltaT.block<3, 1>(0, 3);
    }

    virtual bool read(std::istream &is) { return false; }
    virtual bool write(std::ostream &os) const { return false; }

    void computeError()
    {
        const vertex_pose_4dof *VPi = static_cast<const vertex_pose_4dof *>(_vertices[0]);
        const vertex_pose_4dof *VPj = static_cast<const vertex_pose_4dof *>(_vertices[1]);
        _error << log_so3(VPi->estimate().rotation_camera_world[0] * VPj->estimate().rotation_camera_world[0].transpose() * relative_rotation.transpose()),
            VPi->estimate().rotation_camera_world[0] * (-VPj->estimate().rotation_camera_world[0].transpose() * VPj->estimate().translation_camera_world[0]) +
                VPi->estimate().translation_camera_world[0] - relative_translation;
    }

    Eigen::Matrix4d relative_pose;
    Eigen::Matrix3d relative_rotation;
    Eigen::Vector3d relative_translation;
};

} // namespace ORB_SLAM3
