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

#include "optimizable_types.h"

namespace ORB_SLAM3
{
bool edge_se3_project_xyz_only_pose::read(std::istream &is)
{
    for (int i = 0; i < 2; i++)
    {
        is >> _measurement[i];
    }
    is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
    information()(1, 0) = information()(0, 1);
    return true;
}

bool edge_se3_project_xyz_only_pose::write(std::ostream &os) const
{

    for (int i = 0; i < 2; i++)
    {
        os << measurement()[i] << " ";
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i; j < 2; j++)
        {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

void edge_se3_project_xyz_only_pose::linearizeOplus()
{
    g2o::VertexSE3Expmap *vi        = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d       xyz_trans = vi->estimate().map(point_world);

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];

    Eigen::Matrix<double, 3, 6> SE3deriv;
    SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f, -z, 0.f, x, 0.f, 1.f, 0.f, y, -x, 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXi = -camera.project_jacobian(xyz_trans) * SE3deriv;
}

bool edge_se3_project_xyz_only_pose_to_body::read(std::istream &is)
{
    for (int i = 0; i < 2; i++)
    {
        is >> _measurement[i];
    }
    is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
    information()(1, 0) = information()(0, 1);
    return true;
}

bool edge_se3_project_xyz_only_pose_to_body::write(std::ostream &os) const
{

    for (int i = 0; i < 2; i++)
    {
        os << measurement()[i] << " ";
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i; j < 2; j++)
        {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

void edge_se3_project_xyz_only_pose_to_body::linearizeOplus()
{
    g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    g2o::SE3Quat          T_lw(vi->estimate());
    Eigen::Vector3d       X_l = T_lw.map(point_world);
    Eigen::Vector3d       X_r = right_from_left.map(T_lw.map(point_world));

    double x_w = X_l[0];
    double y_w = X_l[1];
    double z_w = X_l[2];

    Eigen::Matrix<double, 3, 6> SE3deriv;
    SE3deriv << 0.f, z_w, -y_w, 1.f, 0.f, 0.f, -z_w, 0.f, x_w, 0.f, 1.f, 0.f, y_w, -x_w, 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXi = -camera.project_jacobian(X_r) * right_from_left.rotation().toRotationMatrix() * SE3deriv;
}

edge_se3_project_xyz::edge_se3_project_xyz(geometric_camera &camera) : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>(), camera(camera) {}

bool edge_se3_project_xyz::read(std::istream &is)
{
    for (int i = 0; i < 2; i++)
    {
        is >> _measurement[i];
    }
    is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
    information()(1, 0) = information()(0, 1);
    return true;
}

bool edge_se3_project_xyz::write(std::ostream &os) const
{

    for (int i = 0; i < 2; i++)
    {
        os << measurement()[i] << " ";
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i; j < 2; j++)
        {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

void edge_se3_project_xyz::linearizeOplus()
{
    g2o::VertexSE3Expmap   *vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
    g2o::SE3Quat            T(vj->estimate());
    g2o::VertexSBAPointXYZ *vi        = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
    Eigen::Vector3d         xyz       = vi->estimate();
    Eigen::Vector3d         xyz_trans = T.map(xyz);

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];

    // Materialize: `auto` would keep an Eigen expression referencing a destroyed temporary.
    const Eigen::Matrix<double, 2, 3> project_jacobian = -camera.project_jacobian(xyz_trans);

    _jacobianOplusXi = project_jacobian * T.rotation().toRotationMatrix();

    Eigen::Matrix<double, 3, 6> SE3deriv;
    SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f, -z, 0.f, x, 0.f, 1.f, 0.f, y, -x, 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXj = project_jacobian * SE3deriv;
}

edge_se3_project_xyz_to_body::edge_se3_project_xyz_to_body(geometric_camera &camera) : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>(), camera(camera) {}

bool edge_se3_project_xyz_to_body::read(std::istream &is)
{
    for (int i = 0; i < 2; i++)
    {
        is >> _measurement[i];
    }
    is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
    information()(1, 0) = information()(0, 1);
    return true;
}

bool edge_se3_project_xyz_to_body::write(std::ostream &os) const
{

    for (int i = 0; i < 2; i++)
    {
        os << measurement()[i] << " ";
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i; j < 2; j++)
        {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

void edge_se3_project_xyz_to_body::linearizeOplus()
{
    g2o::VertexSE3Expmap   *vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
    g2o::SE3Quat            T_lw(vj->estimate());
    g2o::SE3Quat            T_rw = right_from_left * T_lw;
    g2o::VertexSBAPointXYZ *vi   = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
    Eigen::Vector3d         X_w  = vi->estimate();
    Eigen::Vector3d         X_l  = T_lw.map(X_w);
    Eigen::Vector3d         X_r  = right_from_left.map(T_lw.map(X_w));

    _jacobianOplusXi = -camera.project_jacobian(X_r) * T_rw.rotation().toRotationMatrix();

    double x = X_l[0];
    double y = X_l[1];
    double z = X_l[2];

    Eigen::Matrix<double, 3, 6> SE3deriv;
    SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f, -z, 0.f, x, 0.f, 1.f, 0.f, y, -x, 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXj = -camera.project_jacobian(X_r) * right_from_left.rotation().toRotationMatrix() * SE3deriv;
}

vertex_sim3_expmap::vertex_sim3_expmap(camera_view cameras) : BaseVertex<7, g2o::Sim3>(), cameras(cameras)
{
    _marginalized = false;
    _fix_scale    = false;
}

bool vertex_sim3_expmap::read(std::istream &is)
{
    g2o::Vector7d cam2world;
    for (int i = 0; i < 6; i++)
    {
        is >> cam2world[i];
    }
    is >> cam2world[6];

    for (int observation_index = 0; observation_index < 2; ++observation_index)
    {
        for (size_t parameter_index = 0; parameter_index < cameras[0]->size(); ++parameter_index)
        {
            float parameter;
            if (!(is >> parameter))
            {
                return false;
            }
        }
    }

    setEstimate(g2o::Sim3(cam2world).inverse());
    return true;
}

bool vertex_sim3_expmap::write(std::ostream &os) const
{
    g2o::Sim3     cam2world(estimate().inverse());
    g2o::Vector7d lv = cam2world.log();
    for (int i = 0; i < 7; i++)
    {
        os << lv[i] << " ";
    }

    for (int observation_index = 0; observation_index < 2; ++observation_index)
    {
        for (size_t parameter_index = 0; parameter_index < cameras[0]->size(); ++parameter_index)
        {
            os << cameras[0]->get_parameter(parameter_index) << " ";
        }
    }

    return os.good();
}

edge_sim3_project_xyz::edge_sim3_project_xyz() : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, vertex_sim3_expmap>() {}

bool edge_sim3_project_xyz::read(std::istream &is)
{
    for (int i = 0; i < 2; i++)
    {
        is >> _measurement[i];
    }

    is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
    information()(1, 0) = information()(0, 1);
    return true;
}

bool edge_sim3_project_xyz::write(std::ostream &os) const
{
    for (int i = 0; i < 2; i++)
    {
        os << _measurement[i] << " ";
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i; j < 2; j++)
        {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

edge_inverse_sim3_project_xyz::edge_inverse_sim3_project_xyz() : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, vertex_sim3_expmap>() {}

bool edge_inverse_sim3_project_xyz::read(std::istream &is)
{
    for (int i = 0; i < 2; i++)
    {
        is >> _measurement[i];
    }

    is >> information()(0, 0) >> information()(0, 1) >> information()(1, 1);
    information()(1, 0) = information()(0, 1);
    return true;
}

bool edge_inverse_sim3_project_xyz::write(std::ostream &os) const
{
    for (int i = 0; i < 2; i++)
    {
        os << _measurement[i] << " ";
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = i; j < 2; j++)
        {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

} // namespace ORB_SLAM3
