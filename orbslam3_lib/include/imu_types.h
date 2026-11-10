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

#include <mutex>
#include <ostream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

namespace imu
{

const float gravity_value = 9.81;

// IMU measurement (gyro, accelerometer and timestamp)
class point
{
  public:
    point(const float &acc_x, const float &acc_y, const float &acc_z, const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z, const double &timestamp)
        : a(acc_x, acc_y, acc_z), w(ang_vel_x, ang_vel_y, ang_vel_z), t(timestamp)
    {
    }
    point(const cv::Point3f acceleration, const cv::Point3f angular_velocity, const double &timestamp)
        : a(acceleration.x, acceleration.y, acceleration.z), w(angular_velocity.x, angular_velocity.y, angular_velocity.z), t(timestamp)
    {
    }

  public:
    Eigen::Vector3f a;
    Eigen::Vector3f w;
    double          t;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// IMU biases (gyro and accelerometer)
class bias
{
  public:
    bias() : bax(0), bay(0), baz(0), bwx(0), bwy(0), bwz(0) {}
    bias(const float &b_acc_x, const float &b_acc_y, const float &b_acc_z, const float &b_ang_vel_x, const float &b_ang_vel_y, const float &b_ang_vel_z)
        : bax(b_acc_x), bay(b_acc_y), baz(b_acc_z), bwx(b_ang_vel_x), bwy(b_ang_vel_y), bwz(b_ang_vel_z)
    {
    }
    void                 copy_from(bias &source_bias);
    friend std::ostream &operator<<(std::ostream &out, const bias &source_bias);

  public:
    float bax;
    float bay;
    float baz;
    float bwx;
    float bwy;
    float bwz;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// IMU calibration (Tbc, Tcb, noise)
class calib
{
  public:
    calib(const Sophus::SE3<float> &body_to_camera_pose, const float &ng, const float &na, const float &ngw, const float &naw) { set(body_to_camera_pose, ng, na, ngw, naw); }

    calib(const calib &other);
    calib() { is_set = false; }

    void set(const Sophus::SE3<float> &body_to_camera_pose, const float &ng, const float &na, const float &ngw, const float &naw);

  public:
    Sophus::SE3<float>              camera_to_body;
    Sophus::SE3<float>              body_to_camera;
    Eigen::DiagonalMatrix<float, 6> covariance;
    Eigen::DiagonalMatrix<float, 6> covariance_walk;
    bool                            is_set;
};

// Integration of 1 gyro measurement
class integrated_rotation
{
  public:
    integrated_rotation() {}
    integrated_rotation(const Eigen::Vector3f &angular_velocity, const bias &imu_bias, const float &time);

  public:
    float           delta_time;
    Eigen::Matrix3f delta_rotation;
    Eigen::Matrix3f right_jacobian;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Preintegration of Imu Measurements
class preintegrated
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    preintegrated(const bias &original_bias, const calib &calibration);
    preintegrated(preintegrated *previous_preintegration);
    preintegrated() {}
    ~preintegrated() {}

    void      copy_from(preintegrated *previous_preintegration);
    void      initialize(const bias &original_bias);
    void      integrate_new_measurement(const Eigen::Vector3f &acceleration, const Eigen::Vector3f &angular_velocity, const float &time_step);
    void      reintegrate();
    void      merge_previous(preintegrated *previous_preintegration);
    void      set_new_bias(const bias &updated_bias);
    imu::bias get_delta_bias(const bias &updated_bias);

    Eigen::Matrix3f get_delta_rotation(const bias &updated_bias);
    Eigen::Vector3f get_delta_velocity(const bias &updated_bias);
    Eigen::Vector3f get_delta_position(const bias &updated_bias);

    Eigen::Matrix3f get_updated_delta_rotation();
    Eigen::Vector3f get_updated_delta_velocity();
    Eigen::Vector3f get_updated_delta_position();

    Eigen::Matrix3f get_original_delta_rotation();
    Eigen::Vector3f get_original_delta_velocity();
    Eigen::Vector3f get_original_delta_position();

    Eigen::Matrix<float, 6, 1> get_delta_bias();

    bias get_original_bias();
    bias get_updated_bias();

    void print_measurements() const
    {
        spdlog::debug("Printing {} IMU measurements", _measurements.size());
        for (size_t i = 0; i < _measurements.size(); i++)
        {
            spdlog::debug("IMU measurement timestamp: {}", _measurements[i].t);
        }
    }

  public:
    float                           delta_time;
    Eigen::Matrix<float, 15, 15>    covariance;
    Eigen::Matrix<float, 15, 15>    information;
    Eigen::DiagonalMatrix<float, 6> noise_covariance;
    Eigen::DiagonalMatrix<float, 6> noise_random_walk_covariance;

    // Values for the original bias (when integration was computed)
    bias            original_bias;
    Eigen::Matrix3f delta_rotation;
    Eigen::Vector3f delta_velocity;
    Eigen::Vector3f delta_position;
    Eigen::Matrix3f rotation_gyro_bias_jacobian;
    Eigen::Matrix3f velocity_gyro_bias_jacobian;
    Eigen::Matrix3f velocity_accel_bias_jacobian;
    Eigen::Matrix3f position_gyro_bias_jacobian;
    Eigen::Matrix3f position_accel_bias_jacobian;
    Eigen::Vector3f average_acceleration;
    Eigen::Vector3f average_angular_velocity;

  private:
    // Updated bias
    bias _updated_bias;
    // Difference between original and updated bias, used to compute updated preintegration values.
    Eigen::Matrix<float, 6, 1> _bias_difference;

    struct integrable
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        integrable() {}
        integrable(const Eigen::Vector3f &acceleration, const Eigen::Vector3f &angular_velocity, const float &time_step) : a(acceleration), w(angular_velocity), t(time_step) {}
        Eigen::Vector3f a;
        Eigen::Vector3f w;
        float           t;
    };

    std::vector<integrable> _measurements;
    std::mutex              _mutex;
};

// Lie Algebra Functions
Eigen::Matrix3f right_jacobian_so3(const float &x, const float &y, const float &z);
Eigen::Matrix3f right_jacobian_so3(const Eigen::Vector3f &vector);

Eigen::Matrix3f inverse_right_jacobian_so3(const float &x, const float &y, const float &z);
Eigen::Matrix3f inverse_right_jacobian_so3(const Eigen::Vector3f &vector);

Eigen::Matrix3f normalize_rotation(const Eigen::Matrix3f &rotation);

} // namespace imu

} // namespace ORB_SLAM3
