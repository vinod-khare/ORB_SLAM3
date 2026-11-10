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

#include "imu_types.h"

#include <cmath>
#include <iostream>

namespace ORB_SLAM3
{

namespace imu
{

const float eps = 1e-4;

Eigen::Matrix3f normalize_rotation(const Eigen::Matrix3f &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

Eigen::Matrix3f right_jacobian_so3(const float &x, const float &y, const float &z)
{
    Eigen::Matrix3f I;
    I.setIdentity();
    const float     d2 = x * x + y * y + z * z;
    const float     d  = sqrt(d2);
    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);
    if (d < eps)
    {
        return I;
    }
    else
    {
        return I - W * (1.0f - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
    }
}

Eigen::Matrix3f right_jacobian_so3(const Eigen::Vector3f &v) { return right_jacobian_so3(v(0), v(1), v(2)); }

Eigen::Matrix3f inverse_right_jacobian_so3(const float &x, const float &y, const float &z)
{
    Eigen::Matrix3f I;
    I.setIdentity();
    const float     d2 = x * x + y * y + z * z;
    const float     d  = sqrt(d2);
    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);

    if (d < eps)
    {
        return I;
    }
    else
    {
        return I + W / 2 + W * W * (1.0f / d2 - (1.0f + cos(d)) / (2.0f * d * sin(d)));
    }
}

Eigen::Matrix3f inverse_right_jacobian_so3(const Eigen::Vector3f &v) { return inverse_right_jacobian_so3(v(0), v(1), v(2)); }

integrated_rotation::integrated_rotation(const Eigen::Vector3f &angular_velocity, const bias &imu_bias, const float &time)
{
    const float x = (angular_velocity(0) - imu_bias.bwx) * time;
    const float y = (angular_velocity(1) - imu_bias.bwy) * time;
    const float z = (angular_velocity(2) - imu_bias.bwz) * time;

    const float d2 = x * x + y * y + z * z;
    const float d  = sqrt(d2);

    Eigen::Vector3f v;
    v << x, y, z;
    Eigen::Matrix3f W = Sophus::SO3f::hat(v);
    if (d < eps)
    {
        delta_rotation = Eigen::Matrix3f::Identity() + W;
        right_jacobian = Eigen::Matrix3f::Identity();
    }
    else
    {
        delta_rotation = Eigen::Matrix3f::Identity() + W * sin(d) / d + W * W * (1.0f - cos(d)) / d2;
        right_jacobian = Eigen::Matrix3f::Identity() - W * (1.0f - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
    }
}

preintegrated::preintegrated(const bias &bias_value, const calib &calibration)
{
    noise_covariance             = calibration.covariance;
    noise_random_walk_covariance = calibration.covariance_walk;
    initialize(bias_value);
}

// Copy constructor
preintegrated::preintegrated(preintegrated *previous_preintegration)
    : delta_time(previous_preintegration->delta_time), covariance(previous_preintegration->covariance), information(previous_preintegration->information),
      noise_covariance(previous_preintegration->noise_covariance), noise_random_walk_covariance(previous_preintegration->noise_random_walk_covariance),
      original_bias(previous_preintegration->original_bias), delta_rotation(previous_preintegration->delta_rotation), delta_velocity(previous_preintegration->delta_velocity),
      delta_position(previous_preintegration->delta_position), rotation_gyro_bias_jacobian(previous_preintegration->rotation_gyro_bias_jacobian),
      velocity_gyro_bias_jacobian(previous_preintegration->velocity_gyro_bias_jacobian), velocity_accel_bias_jacobian(previous_preintegration->velocity_accel_bias_jacobian),
      position_gyro_bias_jacobian(previous_preintegration->position_gyro_bias_jacobian), position_accel_bias_jacobian(previous_preintegration->position_accel_bias_jacobian),
      average_acceleration(previous_preintegration->average_acceleration), average_angular_velocity(previous_preintegration->average_angular_velocity),
      _updated_bias(previous_preintegration->_updated_bias), _bias_difference(previous_preintegration->_bias_difference), _measurements(previous_preintegration->_measurements)
{
}

void preintegrated::copy_from(preintegrated *previous_preintegration)
{
    delta_time                   = previous_preintegration->delta_time;
    covariance                   = previous_preintegration->covariance;
    information                  = previous_preintegration->information;
    noise_covariance             = previous_preintegration->noise_covariance;
    noise_random_walk_covariance = previous_preintegration->noise_random_walk_covariance;
    original_bias.copy_from(previous_preintegration->original_bias);
    delta_rotation               = previous_preintegration->delta_rotation;
    delta_velocity               = previous_preintegration->delta_velocity;
    delta_position               = previous_preintegration->delta_position;
    rotation_gyro_bias_jacobian  = previous_preintegration->rotation_gyro_bias_jacobian;
    velocity_gyro_bias_jacobian  = previous_preintegration->velocity_gyro_bias_jacobian;
    velocity_accel_bias_jacobian = previous_preintegration->velocity_accel_bias_jacobian;
    position_gyro_bias_jacobian  = previous_preintegration->position_gyro_bias_jacobian;
    position_accel_bias_jacobian = previous_preintegration->position_accel_bias_jacobian;
    average_acceleration         = previous_preintegration->average_acceleration;
    average_angular_velocity     = previous_preintegration->average_angular_velocity;
    _updated_bias.copy_from(previous_preintegration->_updated_bias);
    _bias_difference = previous_preintegration->_bias_difference;
    _measurements    = previous_preintegration->_measurements;
}

void preintegrated::initialize(const bias &bias_value)
{
    delta_rotation.setIdentity();
    delta_velocity.setZero();
    delta_position.setZero();
    rotation_gyro_bias_jacobian.setZero();
    velocity_gyro_bias_jacobian.setZero();
    velocity_accel_bias_jacobian.setZero();
    position_gyro_bias_jacobian.setZero();
    position_accel_bias_jacobian.setZero();
    covariance.setZero();
    information.setZero();
    _bias_difference.setZero();
    original_bias = bias_value;
    _updated_bias = bias_value;
    average_acceleration.setZero();
    average_angular_velocity.setZero();
    delta_time = 0.0f;
    _measurements.clear();
}

void preintegrated::reintegrate()
{
    std::unique_lock<std::mutex>  lock(_mutex);
    const std::vector<integrable> aux = _measurements;
    initialize(_updated_bias);
    for (size_t i = 0; i < aux.size(); i++)
    {
        integrate_new_measurement(aux[i].a, aux[i].w, aux[i].t);
    }
}

void preintegrated::integrate_new_measurement(const Eigen::Vector3f &acceleration, const Eigen::Vector3f &angular_velocity, const float &time_step)
{
    _measurements.push_back(integrable(acceleration, angular_velocity, time_step));

    // Position is updated firstly, as it depends on previously computed velocity and rotation.
    // Velocity is updated secondly, as it depends on previously computed rotation.
    // Rotation is the last to be updated.

    // Matrices to compute covariance
    Eigen::Matrix<float, 9, 9> A;
    A.setIdentity();
    Eigen::Matrix<float, 9, 6> B;
    B.setZero();

    Eigen::Vector3f acc, accW;
    acc << acceleration(0) - original_bias.bax, acceleration(1) - original_bias.bay, acceleration(2) - original_bias.baz;
    accW << angular_velocity(0) - original_bias.bwx, angular_velocity(1) - original_bias.bwy, angular_velocity(2) - original_bias.bwz;

    average_acceleration     = (delta_time * average_acceleration + delta_rotation * acc * time_step) / (delta_time + time_step);
    average_angular_velocity = (delta_time * average_angular_velocity + accW * time_step) / (delta_time + time_step);

    // Update delta position dP and velocity dV (rely on no-updated delta rotation)
    delta_position = delta_position + delta_velocity * time_step + 0.5f * delta_rotation * acc * time_step * time_step;
    delta_velocity = delta_velocity + delta_rotation * acc * time_step;

    // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
    Eigen::Matrix<float, 3, 3> Wacc = Sophus::SO3f::hat(acc);

    A.block<3, 3>(3, 0) = -delta_rotation * time_step * Wacc;
    A.block<3, 3>(6, 0) = -0.5f * delta_rotation * time_step * time_step * Wacc;
    A.block<3, 3>(6, 3) = Eigen::DiagonalMatrix<float, 3>(time_step, time_step, time_step);
    B.block<3, 3>(3, 3) = delta_rotation * time_step;
    B.block<3, 3>(6, 3) = 0.5f * delta_rotation * time_step * time_step;

    // Update position and velocity jacobians wrt bias correction
    position_accel_bias_jacobian = position_accel_bias_jacobian + velocity_accel_bias_jacobian * time_step - 0.5f * delta_rotation * time_step * time_step;
    position_gyro_bias_jacobian  = position_gyro_bias_jacobian + velocity_gyro_bias_jacobian * time_step - 0.5f * delta_rotation * time_step * time_step * Wacc * rotation_gyro_bias_jacobian;
    velocity_accel_bias_jacobian = velocity_accel_bias_jacobian - delta_rotation * time_step;
    velocity_gyro_bias_jacobian  = velocity_gyro_bias_jacobian - delta_rotation * time_step * Wacc * rotation_gyro_bias_jacobian;

    // Update delta rotation
    integrated_rotation rotation_increment(angular_velocity, original_bias, time_step);
    delta_rotation = normalize_rotation(delta_rotation * rotation_increment.delta_rotation);

    // Compute rotation parts of matrices A and B
    A.block<3, 3>(0, 0) = rotation_increment.delta_rotation.transpose();
    B.block<3, 3>(0, 0) = rotation_increment.right_jacobian * time_step;

    // Update covariance
    covariance.block<9, 9>(0, 0)  = A * covariance.block<9, 9>(0, 0) * A.transpose() + B * noise_covariance * B.transpose();
    covariance.block<6, 6>(9, 9) += noise_random_walk_covariance;

    // Update rotation jacobian wrt bias correction
    rotation_gyro_bias_jacobian = rotation_increment.delta_rotation.transpose() * rotation_gyro_bias_jacobian - rotation_increment.right_jacobian * time_step;

    // Total integrated time
    delta_time += time_step;
}

void preintegrated::merge_previous(preintegrated *previous_preintegration)
{
    if (previous_preintegration == this)
    {
        return;
    }

    std::unique_lock<std::mutex> lock1(_mutex);
    std::unique_lock<std::mutex> lock2(previous_preintegration->_mutex);
    bias                         bav;
    bav.bwx = _updated_bias.bwx;
    bav.bwy = _updated_bias.bwy;
    bav.bwz = _updated_bias.bwz;
    bav.bax = _updated_bias.bax;
    bav.bay = _updated_bias.bay;
    bav.baz = _updated_bias.baz;

    const std::vector<integrable> aux1 = previous_preintegration->_measurements;
    const std::vector<integrable> aux2 = _measurements;

    initialize(bav);
    for (size_t i = 0; i < aux1.size(); i++)
    {
        integrate_new_measurement(aux1[i].a, aux1[i].w, aux1[i].t);
    }
    for (size_t i = 0; i < aux2.size(); i++)
    {
        integrate_new_measurement(aux2[i].a, aux2[i].w, aux2[i].t);
    }
}

void preintegrated::set_new_bias(const bias &updated_bias)
{
    std::unique_lock<std::mutex> lock(_mutex);
    _updated_bias = updated_bias;

    _bias_difference(0) = updated_bias.bwx - original_bias.bwx;
    _bias_difference(1) = updated_bias.bwy - original_bias.bwy;
    _bias_difference(2) = updated_bias.bwz - original_bias.bwz;
    _bias_difference(3) = updated_bias.bax - original_bias.bax;
    _bias_difference(4) = updated_bias.bay - original_bias.bay;
    _bias_difference(5) = updated_bias.baz - original_bias.baz;
}

imu::bias preintegrated::get_delta_bias(const bias &bias_value)
{
    std::unique_lock<std::mutex> lock(_mutex);
    return imu::bias(bias_value.bax - original_bias.bax,
                     bias_value.bay - original_bias.bay,
                     bias_value.baz - original_bias.baz,
                     bias_value.bwx - original_bias.bwx,
                     bias_value.bwy - original_bias.bwy,
                     bias_value.bwz - original_bias.bwz);
}

Eigen::Matrix3f preintegrated::get_delta_rotation(const bias &bias_value)
{
    std::unique_lock<std::mutex> lock(_mutex);
    Eigen::Vector3f              dbg;
    dbg << bias_value.bwx - original_bias.bwx, bias_value.bwy - original_bias.bwy, bias_value.bwz - original_bias.bwz;
    return normalize_rotation(delta_rotation * Sophus::SO3f::exp(rotation_gyro_bias_jacobian * dbg).matrix());
}

Eigen::Vector3f preintegrated::get_delta_velocity(const bias &bias_value)
{
    std::unique_lock<std::mutex> lock(_mutex);
    Eigen::Vector3f              dbg, dba;
    dbg << bias_value.bwx - original_bias.bwx, bias_value.bwy - original_bias.bwy, bias_value.bwz - original_bias.bwz;
    dba << bias_value.bax - original_bias.bax, bias_value.bay - original_bias.bay, bias_value.baz - original_bias.baz;
    return delta_velocity + velocity_gyro_bias_jacobian * dbg + velocity_accel_bias_jacobian * dba;
}

Eigen::Vector3f preintegrated::get_delta_position(const bias &bias_value)
{
    std::unique_lock<std::mutex> lock(_mutex);
    Eigen::Vector3f              dbg, dba;
    dbg << bias_value.bwx - original_bias.bwx, bias_value.bwy - original_bias.bwy, bias_value.bwz - original_bias.bwz;
    dba << bias_value.bax - original_bias.bax, bias_value.bay - original_bias.bay, bias_value.baz - original_bias.baz;
    return delta_position + position_gyro_bias_jacobian * dbg + position_accel_bias_jacobian * dba;
}

Eigen::Matrix3f preintegrated::get_updated_delta_rotation()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return normalize_rotation(delta_rotation * Sophus::SO3f::exp(rotation_gyro_bias_jacobian * _bias_difference.head(3)).matrix());
}

Eigen::Vector3f preintegrated::get_updated_delta_velocity()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return delta_velocity + velocity_gyro_bias_jacobian * _bias_difference.head(3) + velocity_accel_bias_jacobian * _bias_difference.tail(3);
}

Eigen::Vector3f preintegrated::get_updated_delta_position()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return delta_position + position_gyro_bias_jacobian * _bias_difference.head(3) + position_accel_bias_jacobian * _bias_difference.tail(3);
}

Eigen::Matrix3f preintegrated::get_original_delta_rotation()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return delta_rotation;
}

Eigen::Vector3f preintegrated::get_original_delta_velocity()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return delta_velocity;
}

Eigen::Vector3f preintegrated::get_original_delta_position()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return delta_position;
}

bias preintegrated::get_original_bias()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return original_bias;
}

bias preintegrated::get_updated_bias()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return _updated_bias;
}

Eigen::Matrix<float, 6, 1> preintegrated::get_delta_bias()
{
    std::unique_lock<std::mutex> lock(_mutex);
    return _bias_difference;
}

void bias::copy_from(bias &original_bias)
{
    bax = original_bias.bax;
    bay = original_bias.bay;
    baz = original_bias.baz;
    bwx = original_bias.bwx;
    bwy = original_bias.bwy;
    bwz = original_bias.bwz;
}

std::ostream &operator<<(std::ostream &out, const bias &original_bias)
{
    if (original_bias.bwx > 0)
    {
        out << " ";
    }
    out << original_bias.bwx << ",";
    if (original_bias.bwy > 0)
    {
        out << " ";
    }
    out << original_bias.bwy << ",";
    if (original_bias.bwz > 0)
    {
        out << " ";
    }
    out << original_bias.bwz << ",";
    if (original_bias.bax > 0)
    {
        out << " ";
    }
    out << original_bias.bax << ",";
    if (original_bias.bay > 0)
    {
        out << " ";
    }
    out << original_bias.bay << ",";
    if (original_bias.baz > 0)
    {
        out << " ";
    }
    out << original_bias.baz;

    return out;
}

void calib::set(const Sophus::SE3<float> &body_to_camera_pose, const float &ng, const float &na, const float &ngw, const float &naw)
{
    is_set           = true;
    const float ng2  = ng * ng;
    const float na2  = na * na;
    const float ngw2 = ngw * ngw;
    const float naw2 = naw * naw;

    // Sophus/Eigen
    body_to_camera = body_to_camera_pose;
    camera_to_body = body_to_camera_pose.inverse();
    covariance.diagonal() << ng2, ng2, ng2, na2, na2, na2;
    covariance_walk.diagonal() << ngw2, ngw2, ngw2, naw2, naw2, naw2;
}

calib::calib(const calib &other)
{
    is_set = other.is_set;
    // Sophus/Eigen parameters
    body_to_camera  = other.body_to_camera;
    camera_to_body  = other.camera_to_body;
    covariance      = other.covariance;
    covariance_walk = other.covariance_walk;
}

} // namespace imu

} // namespace ORB_SLAM3
