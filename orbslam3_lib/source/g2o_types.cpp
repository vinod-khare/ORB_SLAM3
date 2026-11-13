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

#include "g2o_types.h"

#include "converter.h"
#include "imu_types.h"
namespace ORB_SLAM3
{

imu_cam_pose::imu_cam_pose(key_frame *pKF, camera_view cameras) : cameras(cameras), its(0)
{
    // Load IMU pose
    translation_world_body = pKF->get_imu_position().cast<double>();
    rotation_world_body    = pKF->get_imu_rotation().cast<double>();

    // Load camera poses
    int num_cams;
    if (pKF->is_fisheye_stereo)
    {
        num_cams = 2;
    }
    else
    {
        num_cams = 1;
    }

    translation_camera_world.resize(num_cams);
    rotation_camera_world.resize(num_cams);
    translation_camera_body.resize(num_cams);
    rotation_camera_body.resize(num_cams);
    rotation_body_camera.resize(num_cams);
    translation_body_camera.resize(num_cams);

    // Left camera
    translation_camera_world[0] = pKF->get_translation().cast<double>();
    rotation_camera_world[0]    = pKF->get_rotation().cast<double>();
    translation_camera_body[0]  = pKF->imu_calibration.camera_to_body.translation().cast<double>();
    rotation_camera_body[0]     = pKF->imu_calibration.camera_to_body.rotationMatrix().cast<double>();
    rotation_body_camera[0]     = rotation_camera_body[0].transpose();
    translation_body_camera[0]  = pKF->imu_calibration.body_to_camera.translation().cast<double>();
    bf                          = cameras[0]->baseline_fx();

    if (num_cams > 1)
    {
        Eigen::Matrix4d Trl         = pKF->get_relative_pose_trl(*cameras[0]).matrix().cast<double>();
        rotation_camera_world[1]    = Trl.block<3, 3>(0, 0) * rotation_camera_world[0];
        translation_camera_world[1] = Trl.block<3, 3>(0, 0) * translation_camera_world[0] + Trl.block<3, 1>(0, 3);
        translation_camera_body[1]  = Trl.block<3, 3>(0, 0) * translation_camera_body[0] + Trl.block<3, 1>(0, 3);
        rotation_camera_body[1]     = Trl.block<3, 3>(0, 0) * rotation_camera_body[0];
        rotation_body_camera[1]     = rotation_camera_body[1].transpose();
        translation_body_camera[1]  = -rotation_body_camera[1] * translation_camera_body[1];
    }

    // For posegraph 4DoF
    initial_rotation_world_body = rotation_world_body;
    rotation_delta.setIdentity();
}

imu_cam_pose::imu_cam_pose(frame *pF, const camera_vector &cameras, const imu::calib &imu_calibration) : cameras(cameras), its(0)
{
    // Load IMU pose
    translation_world_body = pF->get_imu_position(imu_calibration).cast<double>();
    rotation_world_body    = pF->get_imu_rotation(imu_calibration).cast<double>();

    // Load camera poses
    int num_cams;
    if (is_fisheye_stereo(cameras))
    {
        num_cams = 2;
    }
    else
    {
        num_cams = 1;
    }

    translation_camera_world.resize(num_cams);
    rotation_camera_world.resize(num_cams);
    translation_camera_body.resize(num_cams);
    rotation_camera_body.resize(num_cams);
    rotation_body_camera.resize(num_cams);
    translation_body_camera.resize(num_cams);

    // Left camera
    translation_camera_world[0] = pF->get_pose().translation().cast<double>();
    rotation_camera_world[0]    = pF->get_pose().rotationMatrix().cast<double>();
    translation_camera_body[0]  = imu_calibration.camera_to_body.translation().cast<double>();
    rotation_camera_body[0]     = imu_calibration.camera_to_body.rotationMatrix().cast<double>();
    rotation_body_camera[0]     = rotation_camera_body[0].transpose();
    translation_body_camera[0]  = imu_calibration.body_to_camera.translation().cast<double>();
    bf                          = cameras[0]->baseline_fx();

    if (num_cams > 1)
    {
        Eigen::Matrix4d Trl         = cameras[0]->left_to_right_pose().inverse().matrix().cast<double>();
        rotation_camera_world[1]    = Trl.block<3, 3>(0, 0) * rotation_camera_world[0];
        translation_camera_world[1] = Trl.block<3, 3>(0, 0) * translation_camera_world[0] + Trl.block<3, 1>(0, 3);
        translation_camera_body[1]  = Trl.block<3, 3>(0, 0) * translation_camera_body[0] + Trl.block<3, 1>(0, 3);
        rotation_camera_body[1]     = Trl.block<3, 3>(0, 0) * rotation_camera_body[0];
        rotation_body_camera[1]     = rotation_camera_body[1].transpose();
        translation_body_camera[1]  = -rotation_body_camera[1] * translation_camera_body[1];
    }

    // For posegraph 4DoF
    initial_rotation_world_body = rotation_world_body;
    rotation_delta.setIdentity();
}

imu_cam_pose::imu_cam_pose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, key_frame *pKF, camera_view cameras) : cameras(cameras), its(0)
{
    // This is only for posegrpah, we do not care about multicamera
    translation_camera_world.resize(1);
    rotation_camera_world.resize(1);
    translation_camera_body.resize(1);
    rotation_camera_body.resize(1);
    rotation_body_camera.resize(1);
    translation_body_camera.resize(1);

    translation_camera_body[0]  = pKF->imu_calibration.camera_to_body.translation().cast<double>();
    rotation_camera_body[0]     = pKF->imu_calibration.camera_to_body.rotationMatrix().cast<double>();
    rotation_body_camera[0]     = rotation_camera_body[0].transpose();
    translation_body_camera[0]  = pKF->imu_calibration.body_to_camera.translation().cast<double>();
    translation_world_body      = _Rwc * translation_camera_body[0] + _twc;
    rotation_world_body         = _Rwc * rotation_camera_body[0];
    rotation_camera_world[0]    = _Rwc.transpose();
    translation_camera_world[0] = -rotation_camera_world[0] * _twc;
    bf                          = cameras[0]->baseline_fx();

    // For posegraph 4DoF
    initial_rotation_world_body = rotation_world_body;
    rotation_delta.setIdentity();
}

void imu_cam_pose::set_parameters(
    const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc, const std::vector<Eigen::Vector3d> &_tbc, const double &_bf)
{
    rotation_body_camera     = _Rbc;
    translation_body_camera  = _tbc;
    rotation_camera_world    = _Rcw;
    translation_camera_world = _tcw;
    const int num_cams       = rotation_body_camera.size();
    rotation_camera_body.resize(num_cams);
    translation_camera_body.resize(num_cams);

    for (size_t i = 0; i < translation_camera_body.size(); i++)
    {
        rotation_camera_body[i]    = rotation_body_camera[i].transpose();
        translation_camera_body[i] = -rotation_camera_body[i] * translation_body_camera[i];
    }
    rotation_world_body    = rotation_camera_world[0].transpose() * rotation_camera_body[0];
    translation_world_body = rotation_camera_world[0].transpose() * (translation_camera_body[0] - translation_camera_world[0]);

    bf = _bf;
}

Eigen::Vector2d imu_cam_pose::project(const Eigen::Vector3d &point_world, int cam_idx) const
{
    Eigen::Vector3d Xc = rotation_camera_world[cam_idx] * point_world + translation_camera_world[cam_idx];

    return cameras[cam_idx]->project(Xc);
}

Eigen::Vector3d imu_cam_pose::project_stereo(const Eigen::Vector3d &point_world, int cam_idx) const
{
    Eigen::Vector3d Pc = rotation_camera_world[cam_idx] * point_world + translation_camera_world[cam_idx];
    Eigen::Vector3d pc;
    double          invZ = 1 / Pc(2);
    pc.head(2)           = cameras[cam_idx]->project(Pc);
    pc(2)                = pc(0) - bf * invZ;
    return pc;
}

bool imu_cam_pose::is_depth_positive(const Eigen::Vector3d &point_world, int cam_idx) const
{
    return (rotation_camera_world[cam_idx].row(2) * point_world + translation_camera_world[cam_idx](2)) > 0.0;
}

void imu_cam_pose::update(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];

    // Update body pose
    translation_world_body += rotation_world_body * ut;
    rotation_world_body     = rotation_world_body * exp_so3(ur);

    // Normalize rotation after 5 updates
    its++;
    if (its >= 3)
    {
        normalize_rotation(rotation_world_body);
        its = 0;
    }

    // Update camera poses
    const Eigen::Matrix3d Rbw = rotation_world_body.transpose();
    const Eigen::Vector3d tbw = -Rbw * translation_world_body;

    for (size_t i = 0; i < rotation_camera_world.size(); i++)
    {
        rotation_camera_world[i]    = rotation_camera_body[i] * Rbw;
        translation_camera_world[i] = rotation_camera_body[i] * tbw + translation_camera_body[i];
    }
}

void imu_cam_pose::update_world_reference(const double *pu)
{
    Eigen::Vector3d ur, ut;
    ur << pu[0], pu[1], pu[2];
    ut << pu[3], pu[4], pu[5];

    const Eigen::Matrix3d dR = exp_so3(ur);
    rotation_delta           = dR * rotation_delta;
    rotation_world_body      = rotation_delta * initial_rotation_world_body;
    // Update body pose
    translation_world_body += ut;

    // Normalize rotation after 5 updates
    its++;
    if (its >= 5)
    {
        rotation_delta(0, 2) = 0.0;
        rotation_delta(1, 2) = 0.0;
        rotation_delta(2, 0) = 0.0;
        rotation_delta(2, 1) = 0.0;
        normalize_rotation(rotation_delta);
        its = 0;
    }

    // Update camera pose
    const Eigen::Matrix3d Rbw = rotation_world_body.transpose();
    const Eigen::Vector3d tbw = -Rbw * translation_world_body;

    for (size_t i = 0; i < rotation_camera_world.size(); i++)
    {
        rotation_camera_world[i]    = rotation_camera_body[i] * Rbw;
        translation_camera_world[i] = rotation_camera_body[i] * tbw + translation_camera_body[i];
    }
}

inverse_depth_point::inverse_depth_point(double _rho, double _u, double _v, geometric_camera &camera)
    : rho(_rho), u(_u), v(_v), fx(camera.get_parameter(0)), fy(camera.get_parameter(1)), cx(camera.get_parameter(2)), cy(camera.get_parameter(3)), bf(camera.baseline_fx())
{
}

void inverse_depth_point::update(const double *pu) { rho += *pu; }

bool vertex_pose::read(std::istream &is)
{
    std::vector<Eigen::Matrix<double, 3, 3>> rotation_camera_world(_estimate.rotation_camera_world.size());
    std::vector<Eigen::Matrix<double, 3, 1>> translation_camera_world(_estimate.translation_camera_world.size());
    std::vector<Eigen::Matrix<double, 3, 3>> rotation_body_camera(_estimate.rotation_body_camera.size());
    std::vector<Eigen::Matrix<double, 3, 1>> translation_body_camera(_estimate.translation_body_camera.size());

    const int num_cams = _estimate.rotation_body_camera.size();
    for (int idx = 0; idx < num_cams; idx++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                is >> rotation_camera_world[idx](i, j);
            }
        }
        for (int i = 0; i < 3; i++)
        {
            is >> translation_camera_world[idx](i);
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                is >> rotation_body_camera[idx](i, j);
            }
        }
        for (int i = 0; i < 3; i++)
        {
            is >> translation_body_camera[idx](i);
        }

        float nextParam;
        for (size_t i = 0; i < _estimate.cameras[idx]->size(); i++)
        {
            if (!(is >> nextParam))
            {
                return false;
            }
        }
    }

    double bf;
    is >> bf;
    _estimate.set_parameters(rotation_camera_world, translation_camera_world, rotation_body_camera, translation_body_camera, bf);
    updateCache();

    return true;
}

bool vertex_pose::write(std::ostream &os) const
{
    std::vector<Eigen::Matrix<double, 3, 3>> rotation_camera_world    = _estimate.rotation_camera_world;
    std::vector<Eigen::Matrix<double, 3, 1>> translation_camera_world = _estimate.translation_camera_world;

    std::vector<Eigen::Matrix<double, 3, 3>> rotation_body_camera    = _estimate.rotation_body_camera;
    std::vector<Eigen::Matrix<double, 3, 1>> translation_body_camera = _estimate.translation_body_camera;

    const int num_cams = translation_camera_world.size();

    for (int idx = 0; idx < num_cams; idx++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                os << rotation_camera_world[idx](i, j) << " ";
            }
        }
        for (int i = 0; i < 3; i++)
        {
            os << translation_camera_world[idx](i) << " ";
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                os << rotation_body_camera[idx](i, j) << " ";
            }
        }
        for (int i = 0; i < 3; i++)
        {
            os << translation_body_camera[idx](i) << " ";
        }

        for (size_t i = 0; i < _estimate.cameras[idx]->size(); i++)
        {
            os << _estimate.cameras[idx]->get_parameter(i) << " ";
        }
    }

    os << _estimate.bf << " ";

    return os.good();
}

void edge_mono::linearizeOplus()
{
    const vertex_pose            *VPose  = static_cast<const vertex_pose *>(_vertices[1]);
    const g2o::VertexSBAPointXYZ *VPoint = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);

    const Eigen::Matrix3d &rotation_camera_world    = VPose->estimate().rotation_camera_world[cam_idx];
    const Eigen::Vector3d &translation_camera_world = VPose->estimate().translation_camera_world[cam_idx];
    const Eigen::Vector3d  Xc                       = rotation_camera_world * VPoint->estimate() + translation_camera_world;
    const Eigen::Vector3d  Xb                       = VPose->estimate().rotation_body_camera[cam_idx] * Xc + VPose->estimate().translation_body_camera[cam_idx];
    const Eigen::Matrix3d &rotation_camera_body     = VPose->estimate().rotation_camera_body[cam_idx];

    const Eigen::Matrix<double, 2, 3> proj_jac = VPose->estimate().cameras[cam_idx]->project_jacobian(Xc);
    _jacobianOplusXi                           = -proj_jac * rotation_camera_world;

    Eigen::Matrix<double, 3, 6> SE3deriv;
    double                      x = Xb(0);
    double                      y = Xb(1);
    double                      z = Xb(2);

    SE3deriv << 0.0, z, -y, 1.0, 0.0, 0.0, -z, 0.0, x, 0.0, 1.0, 0.0, y, -x, 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jac * rotation_camera_body * SE3deriv; // TODO optimize this product
}

void edge_mono_only_pose::linearizeOplus()
{
    const vertex_pose *VPose = static_cast<const vertex_pose *>(_vertices[0]);

    const Eigen::Matrix3d &rotation_camera_world    = VPose->estimate().rotation_camera_world[cam_idx];
    const Eigen::Vector3d &translation_camera_world = VPose->estimate().translation_camera_world[cam_idx];
    const Eigen::Vector3d  Xc                       = rotation_camera_world * point_world + translation_camera_world;
    const Eigen::Vector3d  Xb                       = VPose->estimate().rotation_body_camera[cam_idx] * Xc + VPose->estimate().translation_body_camera[cam_idx];
    const Eigen::Matrix3d &rotation_camera_body     = VPose->estimate().rotation_camera_body[cam_idx];

    Eigen::Matrix<double, 2, 3> proj_jac = VPose->estimate().cameras[cam_idx]->project_jacobian(Xc);

    Eigen::Matrix<double, 3, 6> SE3deriv;
    double                      x = Xb(0);
    double                      y = Xb(1);
    double                      z = Xb(2);
    SE3deriv << 0.0, z, -y, 1.0, 0.0, 0.0, -z, 0.0, x, 0.0, 1.0, 0.0, y, -x, 0.0, 0.0, 0.0, 1.0;
    _jacobianOplusXi = proj_jac * rotation_camera_body * SE3deriv; // symbol different becasue of update mode
}

void edge_stereo::linearizeOplus()
{
    const vertex_pose            *VPose  = static_cast<const vertex_pose *>(_vertices[1]);
    const g2o::VertexSBAPointXYZ *VPoint = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);

    const Eigen::Matrix3d &rotation_camera_world    = VPose->estimate().rotation_camera_world[cam_idx];
    const Eigen::Vector3d &translation_camera_world = VPose->estimate().translation_camera_world[cam_idx];
    const Eigen::Vector3d  Xc                       = rotation_camera_world * VPoint->estimate() + translation_camera_world;
    const Eigen::Vector3d  Xb                       = VPose->estimate().rotation_body_camera[cam_idx] * Xc + VPose->estimate().translation_body_camera[cam_idx];
    const Eigen::Matrix3d &rotation_camera_body     = VPose->estimate().rotation_camera_body[cam_idx];
    const double           bf                       = VPose->estimate().bf;
    const double           inv_z2                   = 1.0 / (Xc(2) * Xc(2));

    Eigen::Matrix<double, 3, 3> proj_jac;
    proj_jac.block<2, 3>(0, 0)  = VPose->estimate().cameras[cam_idx]->project_jacobian(Xc);
    proj_jac.block<1, 3>(2, 0)  = proj_jac.block<1, 3>(0, 0);
    proj_jac(2, 2)             += bf * inv_z2;

    _jacobianOplusXi = -proj_jac * rotation_camera_world;

    Eigen::Matrix<double, 3, 6> SE3deriv;
    double                      x = Xb(0);
    double                      y = Xb(1);
    double                      z = Xb(2);

    SE3deriv << 0.0, z, -y, 1.0, 0.0, 0.0, -z, 0.0, x, 0.0, 1.0, 0.0, y, -x, 0.0, 0.0, 0.0, 1.0;

    _jacobianOplusXj = proj_jac * rotation_camera_body * SE3deriv;
}

void edge_stereo_only_pose::linearizeOplus()
{
    const vertex_pose *VPose = static_cast<const vertex_pose *>(_vertices[0]);

    const Eigen::Matrix3d &rotation_camera_world    = VPose->estimate().rotation_camera_world[cam_idx];
    const Eigen::Vector3d &translation_camera_world = VPose->estimate().translation_camera_world[cam_idx];
    const Eigen::Vector3d  Xc                       = rotation_camera_world * point_world + translation_camera_world;
    const Eigen::Vector3d  Xb                       = VPose->estimate().rotation_body_camera[cam_idx] * Xc + VPose->estimate().translation_body_camera[cam_idx];
    const Eigen::Matrix3d &rotation_camera_body     = VPose->estimate().rotation_camera_body[cam_idx];
    const double           bf                       = VPose->estimate().bf;
    const double           inv_z2                   = 1.0 / (Xc(2) * Xc(2));

    Eigen::Matrix<double, 3, 3> proj_jac;
    proj_jac.block<2, 3>(0, 0)  = VPose->estimate().cameras[cam_idx]->project_jacobian(Xc);
    proj_jac.block<1, 3>(2, 0)  = proj_jac.block<1, 3>(0, 0);
    proj_jac(2, 2)             += bf * inv_z2;

    Eigen::Matrix<double, 3, 6> SE3deriv;
    double                      x = Xb(0);
    double                      y = Xb(1);
    double                      z = Xb(2);
    SE3deriv << 0.0, z, -y, 1.0, 0.0, 0.0, -z, 0.0, x, 0.0, 1.0, 0.0, y, -x, 0.0, 0.0, 0.0, 1.0;
    _jacobianOplusXi = proj_jac * rotation_camera_body * SE3deriv;
}

vertex_velocity::vertex_velocity(key_frame *pKF) { setEstimate(pKF->get_velocity().cast<double>()); }

vertex_velocity::vertex_velocity(frame *pF) { setEstimate(pF->get_velocity().cast<double>()); }

vertex_gyro_bias::vertex_gyro_bias(key_frame *pKF) { setEstimate(pKF->get_gyro_bias().cast<double>()); }

vertex_gyro_bias::vertex_gyro_bias(frame *pF)
{
    Eigen::Vector3d bg;
    bg << pF->imu_bias.bwx, pF->imu_bias.bwy, pF->imu_bias.bwz;
    setEstimate(bg);
}

vertex_acc_bias::vertex_acc_bias(key_frame *pKF) { setEstimate(pKF->get_accelerometer_bias().cast<double>()); }

vertex_acc_bias::vertex_acc_bias(frame *pF)
{
    Eigen::Vector3d ba;
    ba << pF->imu_bias.bax, pF->imu_bias.bay, pF->imu_bias.baz;
    setEstimate(ba);
}

edge_inertial::edge_inertial(imu::preintegrated *pInt)
    : rotation_gyro_bias_jacobian(pInt->rotation_gyro_bias_jacobian.cast<double>()), velocity_gyro_bias_jacobian(pInt->velocity_gyro_bias_jacobian.cast<double>()),
      position_gyro_bias_jacobian(pInt->position_gyro_bias_jacobian.cast<double>()), velocity_accel_bias_jacobian(pInt->velocity_accel_bias_jacobian.cast<double>()),
      position_accel_bias_jacobian(pInt->position_accel_bias_jacobian.cast<double>()), preintegration(pInt), dt(pInt->delta_time)
{
    // This edge links 6 vertices
    resize(6);
    g << 0, 0, -imu::gravity_value;

    matrix_9d Info = pInt->covariance.block<9, 9>(0, 0).cast<double>().inverse();
    Info           = (Info + Info.transpose()) / 2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(Info);
    Eigen::Matrix<double, 9, 1>                                eigs = es.eigenvalues();
    for (int i = 0; i < 9; i++)
    {
        if (eigs[i] < 1e-12)
        {
            eigs[i] = 0;
        }
    }
    Info = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
    setInformation(Info);
}

void edge_inertial::computeError()
{
    // TODO Maybe reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const vertex_pose      *VP1 = static_cast<const vertex_pose *>(_vertices[0]);
    const vertex_velocity  *VV1 = static_cast<const vertex_velocity *>(_vertices[1]);
    const vertex_gyro_bias *VG1 = static_cast<const vertex_gyro_bias *>(_vertices[2]);
    const vertex_acc_bias  *VA1 = static_cast<const vertex_acc_bias *>(_vertices[3]);
    const vertex_pose      *VP2 = static_cast<const vertex_pose *>(_vertices[4]);
    const vertex_velocity  *VV2 = static_cast<const vertex_velocity *>(_vertices[5]);
    const imu::bias         b1(VA1->estimate()[0], VA1->estimate()[1], VA1->estimate()[2], VG1->estimate()[0], VG1->estimate()[1], VG1->estimate()[2]);
    const Eigen::Matrix3d   dR = preintegration->get_delta_rotation(b1).cast<double>();
    const Eigen::Vector3d   dV = preintegration->get_delta_velocity(b1).cast<double>();
    const Eigen::Vector3d   dP = preintegration->get_delta_position(b1).cast<double>();

    const Eigen::Vector3d er = log_so3(dR.transpose() * VP1->estimate().rotation_world_body.transpose() * VP2->estimate().rotation_world_body);
    const Eigen::Vector3d ev = VP1->estimate().rotation_world_body.transpose() * (VV2->estimate() - VV1->estimate() - g * dt) - dV;
    const Eigen::Vector3d ep =
        VP1->estimate().rotation_world_body.transpose() * (VP2->estimate().translation_world_body - VP1->estimate().translation_world_body - VV1->estimate() * dt - g * dt * dt / 2) - dP;

    _error << er, ev, ep;
}

void edge_inertial::linearizeOplus()
{
    const vertex_pose      *VP1 = static_cast<const vertex_pose *>(_vertices[0]);
    const vertex_velocity  *VV1 = static_cast<const vertex_velocity *>(_vertices[1]);
    const vertex_gyro_bias *VG1 = static_cast<const vertex_gyro_bias *>(_vertices[2]);
    const vertex_acc_bias  *VA1 = static_cast<const vertex_acc_bias *>(_vertices[3]);
    const vertex_pose      *VP2 = static_cast<const vertex_pose *>(_vertices[4]);
    const vertex_velocity  *VV2 = static_cast<const vertex_velocity *>(_vertices[5]);
    const imu::bias         b1(VA1->estimate()[0], VA1->estimate()[1], VA1->estimate()[2], VG1->estimate()[0], VG1->estimate()[1], VG1->estimate()[2]);
    const imu::bias         db = preintegration->get_delta_bias(b1);
    Eigen::Vector3d         dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1 = VP1->estimate().rotation_world_body;
    const Eigen::Matrix3d Rbw1 = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2 = VP2->estimate().rotation_world_body;

    const Eigen::Matrix3d dR    = preintegration->get_delta_rotation(b1).cast<double>();
    const Eigen::Matrix3d eR    = dR.transpose() * Rbw1 * Rwb2;
    const Eigen::Vector3d er    = log_so3(eR);
    const Eigen::Matrix3d invJr = inverse_right_jacobian_so3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
    // rotation
    _jacobianOplus[0].block<3, 3>(0, 0) = -invJr * Rwb2.transpose() * Rwb1;                                       // OK
    _jacobianOplus[0].block<3, 3>(3, 0) = Sophus::SO3d::hat(Rbw1 * (VV2->estimate() - VV1->estimate() - g * dt)); // OK
    _jacobianOplus[0].block<3, 3>(6, 0) =
        Sophus::SO3d::hat(Rbw1 * (VP2->estimate().translation_world_body - VP1->estimate().translation_world_body - VV1->estimate() * dt - 0.5 * g * dt * dt)); // OK
    // translation
    _jacobianOplus[0].block<3, 3>(6, 3) = -Eigen::Matrix3d::Identity(); // OK

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3, 3>(3, 0) = -Rbw1;      // OK
    _jacobianOplus[1].block<3, 3>(6, 0) = -Rbw1 * dt; // OK

    // Jacobians wrt Gyro 1
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3, 3>(0, 0) = -invJr * eR.transpose() * right_jacobian_so3(rotation_gyro_bias_jacobian * dbg) * rotation_gyro_bias_jacobian; // OK
    _jacobianOplus[2].block<3, 3>(3, 0) = -velocity_gyro_bias_jacobian;                                                                                  // OK
    _jacobianOplus[2].block<3, 3>(6, 0) = -position_gyro_bias_jacobian;                                                                                  // OK

    // Jacobians wrt Accelerometer 1
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3, 3>(3, 0) = -velocity_accel_bias_jacobian; // OK
    _jacobianOplus[3].block<3, 3>(6, 0) = -position_accel_bias_jacobian; // OK

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3, 3>(0, 0) = invJr; // OK
    // translation
    _jacobianOplus[4].block<3, 3>(6, 3) = Rbw1 * Rwb2; // OK

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3, 3>(3, 0) = Rbw1; // OK
}

edge_inertial_gs::edge_inertial_gs(imu::preintegrated *pInt)
    : rotation_gyro_bias_jacobian(pInt->rotation_gyro_bias_jacobian.cast<double>()), velocity_gyro_bias_jacobian(pInt->velocity_gyro_bias_jacobian.cast<double>()),
      position_gyro_bias_jacobian(pInt->position_gyro_bias_jacobian.cast<double>()), velocity_accel_bias_jacobian(pInt->velocity_accel_bias_jacobian.cast<double>()),
      position_accel_bias_jacobian(pInt->position_accel_bias_jacobian.cast<double>()), preintegration(pInt), dt(pInt->delta_time)
{
    // This edge links 8 vertices
    resize(8);
    gravity_imu << 0, 0, -imu::gravity_value;

    matrix_9d Info = pInt->covariance.block<9, 9>(0, 0).cast<double>().inverse();
    Info           = (Info + Info.transpose()) / 2;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(Info);
    Eigen::Matrix<double, 9, 1>                                eigs = es.eigenvalues();
    for (int i = 0; i < 9; i++)
    {
        if (eigs[i] < 1e-12)
        {
            eigs[i] = 0;
        }
    }
    Info = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
    setInformation(Info);
}

void edge_inertial_gs::computeError()
{
    // TODO Maybe reintegrate inertial measurments when difference between linearization point and current estimate is too big
    const vertex_pose              *VP1   = static_cast<const vertex_pose *>(_vertices[0]);
    const vertex_velocity          *VV1   = static_cast<const vertex_velocity *>(_vertices[1]);
    const vertex_gyro_bias         *VG    = static_cast<const vertex_gyro_bias *>(_vertices[2]);
    const vertex_acc_bias          *VA    = static_cast<const vertex_acc_bias *>(_vertices[3]);
    const vertex_pose              *VP2   = static_cast<const vertex_pose *>(_vertices[4]);
    const vertex_velocity          *VV2   = static_cast<const vertex_velocity *>(_vertices[5]);
    const vertex_gravity_direction *VGDir = static_cast<const vertex_gravity_direction *>(_vertices[6]);
    const vertex_scale             *VS    = static_cast<const vertex_scale *>(_vertices[7]);
    const imu::bias                 b(VA->estimate()[0], VA->estimate()[1], VA->estimate()[2], VG->estimate()[0], VG->estimate()[1], VG->estimate()[2]);
    g                        = VGDir->estimate().rotation_world_gravity * gravity_imu;
    const double          s  = VS->estimate();
    const Eigen::Matrix3d dR = preintegration->get_delta_rotation(b).cast<double>();
    const Eigen::Vector3d dV = preintegration->get_delta_velocity(b).cast<double>();
    const Eigen::Vector3d dP = preintegration->get_delta_position(b).cast<double>();

    const Eigen::Vector3d er = log_so3(dR.transpose() * VP1->estimate().rotation_world_body.transpose() * VP2->estimate().rotation_world_body);
    const Eigen::Vector3d ev = VP1->estimate().rotation_world_body.transpose() * (s * (VV2->estimate() - VV1->estimate()) - g * dt) - dV;
    const Eigen::Vector3d ep =
        VP1->estimate().rotation_world_body.transpose() * (s * (VP2->estimate().translation_world_body - VP1->estimate().translation_world_body - VV1->estimate() * dt) - g * dt * dt / 2) - dP;

    _error << er, ev, ep;
}

void edge_inertial_gs::linearizeOplus()
{
    const vertex_pose              *VP1   = static_cast<const vertex_pose *>(_vertices[0]);
    const vertex_velocity          *VV1   = static_cast<const vertex_velocity *>(_vertices[1]);
    const vertex_gyro_bias         *VG    = static_cast<const vertex_gyro_bias *>(_vertices[2]);
    const vertex_acc_bias          *VA    = static_cast<const vertex_acc_bias *>(_vertices[3]);
    const vertex_pose              *VP2   = static_cast<const vertex_pose *>(_vertices[4]);
    const vertex_velocity          *VV2   = static_cast<const vertex_velocity *>(_vertices[5]);
    const vertex_gravity_direction *VGDir = static_cast<const vertex_gravity_direction *>(_vertices[6]);
    const vertex_scale             *VS    = static_cast<const vertex_scale *>(_vertices[7]);
    const imu::bias                 b(VA->estimate()[0], VA->estimate()[1], VA->estimate()[2], VG->estimate()[0], VG->estimate()[1], VG->estimate()[2]);
    const imu::bias                 db = preintegration->get_delta_bias(b);

    Eigen::Vector3d dbg;
    dbg << db.bwx, db.bwy, db.bwz;

    const Eigen::Matrix3d Rwb1                   = VP1->estimate().rotation_world_body;
    const Eigen::Matrix3d Rbw1                   = Rwb1.transpose();
    const Eigen::Matrix3d Rwb2                   = VP2->estimate().rotation_world_body;
    const Eigen::Matrix3d rotation_world_gravity = VGDir->estimate().rotation_world_gravity;
    Eigen::MatrixXd       Gm                     = Eigen::MatrixXd::Zero(3, 2);
    Gm(0, 1)                                     = -imu::gravity_value;
    Gm(1, 0)                                     = imu::gravity_value;
    const double          s                      = VS->estimate();
    const Eigen::MatrixXd dGdTheta               = rotation_world_gravity * Gm;
    const Eigen::Matrix3d dR                     = preintegration->get_delta_rotation(b).cast<double>();
    const Eigen::Matrix3d eR                     = dR.transpose() * Rbw1 * Rwb2;
    const Eigen::Vector3d er                     = log_so3(eR);
    const Eigen::Matrix3d invJr                  = inverse_right_jacobian_so3(er);

    // Jacobians wrt Pose 1
    _jacobianOplus[0].setZero();
    // rotation
    _jacobianOplus[0].block<3, 3>(0, 0) = -invJr * Rwb2.transpose() * Rwb1;
    _jacobianOplus[0].block<3, 3>(3, 0) = Sophus::SO3d::hat(Rbw1 * (s * (VV2->estimate() - VV1->estimate()) - g * dt));
    _jacobianOplus[0].block<3, 3>(6, 0) =
        Sophus::SO3d::hat(Rbw1 * (s * (VP2->estimate().translation_world_body - VP1->estimate().translation_world_body - VV1->estimate() * dt) - 0.5 * g * dt * dt));
    // translation
    _jacobianOplus[0].block<3, 3>(6, 3) = Eigen::DiagonalMatrix<double, 3>(-s, -s, -s);

    // Jacobians wrt Velocity 1
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3, 3>(3, 0) = -s * Rbw1;
    _jacobianOplus[1].block<3, 3>(6, 0) = -s * Rbw1 * dt;

    // Jacobians wrt Gyro bias
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3, 3>(0, 0) = -invJr * eR.transpose() * right_jacobian_so3(rotation_gyro_bias_jacobian * dbg) * rotation_gyro_bias_jacobian;
    _jacobianOplus[2].block<3, 3>(3, 0) = -velocity_gyro_bias_jacobian;
    _jacobianOplus[2].block<3, 3>(6, 0) = -position_gyro_bias_jacobian;

    // Jacobians wrt Accelerometer bias
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3, 3>(3, 0) = -velocity_accel_bias_jacobian;
    _jacobianOplus[3].block<3, 3>(6, 0) = -position_accel_bias_jacobian;

    // Jacobians wrt Pose 2
    _jacobianOplus[4].setZero();
    // rotation
    _jacobianOplus[4].block<3, 3>(0, 0) = invJr;
    // translation
    _jacobianOplus[4].block<3, 3>(6, 3) = s * Rbw1 * Rwb2;

    // Jacobians wrt Velocity 2
    _jacobianOplus[5].setZero();
    _jacobianOplus[5].block<3, 3>(3, 0) = s * Rbw1;

    // Jacobians wrt Gravity direction
    _jacobianOplus[6].setZero();
    _jacobianOplus[6].block<3, 2>(3, 0) = -Rbw1 * dGdTheta * dt;
    _jacobianOplus[6].block<3, 2>(6, 0) = -0.5 * Rbw1 * dGdTheta * dt * dt;

    // Jacobians wrt scale factor
    _jacobianOplus[7].setZero();
    _jacobianOplus[7].block<3, 1>(3, 0) = Rbw1 * (VV2->estimate() - VV1->estimate());
    _jacobianOplus[7].block<3, 1>(6, 0) = Rbw1 * (VP2->estimate().translation_world_body - VP1->estimate().translation_world_body - VV1->estimate() * dt);
}

edge_prior_pose_imu::edge_prior_pose_imu(constraint_pose_imu *c)
{
    resize(4);
    rotation_world_body    = c->rotation_world_body;
    translation_world_body = c->translation_world_body;
    vwb                    = c->vwb;
    bg                     = c->bg;
    ba                     = c->ba;
    setInformation(c->information_matrix);
}

void edge_prior_pose_imu::computeError()
{
    const vertex_pose      *VP = static_cast<const vertex_pose *>(_vertices[0]);
    const vertex_velocity  *VV = static_cast<const vertex_velocity *>(_vertices[1]);
    const vertex_gyro_bias *VG = static_cast<const vertex_gyro_bias *>(_vertices[2]);
    const vertex_acc_bias  *VA = static_cast<const vertex_acc_bias *>(_vertices[3]);

    const Eigen::Vector3d er  = log_so3(rotation_world_body.transpose() * VP->estimate().rotation_world_body);
    const Eigen::Vector3d et  = rotation_world_body.transpose() * (VP->estimate().translation_world_body - translation_world_body);
    const Eigen::Vector3d ev  = VV->estimate() - vwb;
    const Eigen::Vector3d ebg = VG->estimate() - bg;
    const Eigen::Vector3d eba = VA->estimate() - ba;

    _error << er, et, ev, ebg, eba;
}

void edge_prior_pose_imu::linearizeOplus()
{
    const vertex_pose    *VP = static_cast<const vertex_pose *>(_vertices[0]);
    const Eigen::Vector3d er = log_so3(rotation_world_body.transpose() * VP->estimate().rotation_world_body);
    _jacobianOplus[0].setZero();
    _jacobianOplus[0].block<3, 3>(0, 0) = inverse_right_jacobian_so3(er);
    _jacobianOplus[0].block<3, 3>(3, 3) = rotation_world_body.transpose() * VP->estimate().rotation_world_body;
    _jacobianOplus[1].setZero();
    _jacobianOplus[1].block<3, 3>(6, 0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[2].setZero();
    _jacobianOplus[2].block<3, 3>(9, 0) = Eigen::Matrix3d::Identity();
    _jacobianOplus[3].setZero();
    _jacobianOplus[3].block<3, 3>(12, 0) = Eigen::Matrix3d::Identity();
}

void edge_prior_acc::linearizeOplus()
{
    // Jacobian wrt bias
    _jacobianOplusXi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
}

void edge_prior_gyro::linearizeOplus()
{
    // Jacobian wrt bias
    _jacobianOplusXi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
}

// SO3 FUNCTIONS
Eigen::Matrix3d exp_so3(const Eigen::Vector3d &w) { return exp_so3(w[0], w[1], w[2]); }

Eigen::Matrix3d exp_so3(const double x, const double y, const double z)
{
    const double    d2 = x * x + y * y + z * z;
    const double    d  = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
        return normalize_rotation(res);
    }
    else
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W * sin(d) / d + W * W * (1.0 - cos(d)) / d2;
        return normalize_rotation(res);
    }
}

Eigen::Vector3d log_so3(const Eigen::Matrix3d &R)
{
    const double    tr = R(0, 0) + R(1, 1) + R(2, 2);
    Eigen::Vector3d w;
    w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2, (R(1, 0) - R(0, 1)) / 2;
    const double costheta = (tr - 1.0) * 0.5f;
    if (costheta > 1 || costheta < -1)
    {
        return w;
    }
    const double theta = acos(costheta);
    const double s     = sin(theta);
    if (fabs(s) < 1e-5)
    {
        return w;
    }
    else
    {
        return theta * w / s;
    }
}

Eigen::Matrix3d inverse_right_jacobian_so3(const Eigen::Vector3d &v) { return inverse_right_jacobian_so3(v[0], v[1], v[2]); }

Eigen::Matrix3d inverse_right_jacobian_so3(const double x, const double y, const double z)
{
    const double d2 = x * x + y * y + z * z;
    const double d  = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() + W / 2 + W * W * (1.0 / d2 - (1.0 + cos(d)) / (2.0 * d * sin(d)));
    }
}

Eigen::Matrix3d right_jacobian_so3(const Eigen::Vector3d &v) { return right_jacobian_so3(v[0], v[1], v[2]); }

Eigen::Matrix3d right_jacobian_so3(const double x, const double y, const double z)
{
    const double d2 = x * x + y * y + z * z;
    const double d  = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
    if (d < 1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W * (1.0 - cos(d)) / d2 + W * W * (d - sin(d)) / (d2 * d);
    }
}

Eigen::Matrix3d skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0;
    return W;
}

} // namespace ORB_SLAM3
