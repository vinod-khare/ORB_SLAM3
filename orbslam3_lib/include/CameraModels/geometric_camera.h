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

#ifndef CAMERAMODELS_GEOMETRICCAMERA_H
#define CAMERAMODELS_GEOMETRICCAMERA_H

#include "converter.h"
#include "geometric_tools.h"

#include <memory>
#include <span>
#include <vector>

#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace ORB_SLAM3
{
class geometric_camera
{

  public:
    geometric_camera() {}
    geometric_camera(const std::vector<float> &_vParameters) : parameters(_vParameters) {}
    geometric_camera(const geometric_camera &other)
        : parameters(other.parameters), id(other.id), type(other.type), _distortion_coefficients(other._distortion_coefficients.clone()), _baseline(other._baseline),
          _left_to_right_pose(other._left_to_right_pose)
    {
    }
    virtual ~geometric_camera() = default;

    virtual cv::Point2f     project(const cv::Point3f &p3D)     = 0;
    virtual Eigen::Vector2d project(const Eigen::Vector3d &v3D) = 0;
    virtual Eigen::Vector2f project(const Eigen::Vector3f &v3D) = 0;
    virtual Eigen::Vector2f project_mat(const cv::Point3f &p3D) = 0;

    virtual float uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D) = 0;

    virtual Eigen::Vector3f unproject_eigen(const cv::Point2f &p2D) = 0;
    virtual cv::Point3f     unproject(const cv::Point2f &p2D)       = 0;

    virtual Eigen::Matrix<double, 2, 3> project_jacobian(const Eigen::Vector3d &v3D) = 0;

    virtual bool reconstruct_with_two_views(const std::vector<cv::KeyPoint> &vKeys1,
                                            const std::vector<cv::KeyPoint> &vKeys2,
                                            const std::vector<int>          &vMatches12,
                                            Sophus::SE3f                    &T21,
                                            std::vector<cv::Point3f>        &vP3D,
                                            std::vector<bool>               &vbTriangulated) = 0;

    virtual cv::Mat         to_k()       = 0;
    virtual Eigen::Matrix3f to_k_eigen() = 0;

    virtual bool epipolar_constrain(
        geometric_camera &otherCamera, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, const float sigmaLevel, const float unc) = 0;

    float get_parameter(const int i) { return parameters[i]; }
    void  set_parameter(const float p, const size_t i) { parameters[i] = p; }

    const cv::Mat &distortion_coefficients() const { return _distortion_coefficients; }
    void           set_distortion_coefficients(const cv::Mat &coefficients)
    {
        CV_Assert(coefficients.type() == CV_32F && (coefficients.rows == 1 || coefficients.cols == 1) && (coefficients.total() == 4 || coefficients.total() == 5));
        _distortion_coefficients = coefficients.clone();
    }

    float baseline() const { return _baseline; }
    float baseline_fx() { return _baseline * get_parameter(0); }
    void  set_baseline(float baseline) { _baseline = baseline; }

    /** Pose of the right camera in the left camera's coordinates. */
    const Sophus::SE3f &left_to_right_pose() const { return _left_to_right_pose; }
    void                set_left_to_right_pose(const Sophus::SE3f &pose) { _left_to_right_pose = pose; }

    size_t size() { return parameters.size(); }

    virtual bool match_and_triangulate(const cv::KeyPoint &kp1,
                                       const cv::KeyPoint &kp2,
                                       geometric_camera   &pOther,
                                       Sophus::SE3f       &Tcw1,
                                       Sophus::SE3f       &Tcw2,
                                       const float         sigmaLevel1,
                                       const float         sigmaLevel2,
                                       Eigen::Vector3f    &x3Dtriangulated) = 0;

    unsigned int get_id() { return id; }

    unsigned int get_type() { return type; }

    const static unsigned int cam_pinhole = 0;
    const static unsigned int cam_fisheye = 1;

    static long unsigned int next_id;

  protected:
    std::vector<float> parameters;

    unsigned int id;

    unsigned int type;

  private:
    cv::Mat      _distortion_coefficients = cv::Mat::zeros(4, 1, CV_32F);
    float        _baseline{0};
    Sophus::SE3f _left_to_right_pose;
};

using camera_vector = std::vector<std::unique_ptr<geometric_camera>>;
using camera_view   = std::span<const std::unique_ptr<geometric_camera>>;

inline bool is_fisheye_stereo(camera_view cameras) { return cameras.size() == 2 && cameras[0]->get_type() == geometric_camera::cam_fisheye; }
} // namespace ORB_SLAM3

#endif // CAMERAMODELS_GEOMETRICCAMERA_H
