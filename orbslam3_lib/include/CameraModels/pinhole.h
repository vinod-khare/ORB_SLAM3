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

#ifndef CAMERAMODELS_PINHOLE_H
#define CAMERAMODELS_PINHOLE_H

#include "geometric_camera.h"
#include "two_view_reconstruction.h"

#include <assert.h>

namespace ORB_SLAM3
{
class pinhole : public geometric_camera
{

  public:
    pinhole()
    {
        parameters.resize(4);
        id   = next_id++;
        type = cam_pinhole;
    }
    pinhole(const std::vector<float> _vParameters) : geometric_camera(_vParameters), _two_view_reconstruction(nullptr)
    {
        assert(parameters.size() == 4);
        id   = next_id++;
        type = cam_pinhole;
    }

    pinhole(pinhole *pPinhole) : geometric_camera(pPinhole->parameters), _two_view_reconstruction(nullptr)
    {
        assert(parameters.size() == 4);
        id   = next_id++;
        type = cam_pinhole;
    }

    ~pinhole()
    {
        if (_two_view_reconstruction)
        {
            delete _two_view_reconstruction;
        }
    }

    cv::Point2f     project(const cv::Point3f &p3D);
    Eigen::Vector2d project(const Eigen::Vector3d &v3D);
    Eigen::Vector2f project(const Eigen::Vector3f &v3D);
    Eigen::Vector2f project_mat(const cv::Point3f &p3D);

    float uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D);

    Eigen::Vector3f unproject_eigen(const cv::Point2f &p2D);
    cv::Point3f     unproject(const cv::Point2f &p2D);

    Eigen::Matrix<double, 2, 3> project_jacobian(const Eigen::Vector3d &v3D);

    bool reconstruct_with_two_views(const std::vector<cv::KeyPoint> &vKeys1,
                                    const std::vector<cv::KeyPoint> &vKeys2,
                                    const std::vector<int>          &vMatches12,
                                    Sophus::SE3f                    &T21,
                                    std::vector<cv::Point3f>        &vP3D,
                                    std::vector<bool>               &vbTriangulated);

    cv::Mat         to_k();
    Eigen::Matrix3f to_k_eigen();

    bool epipolar_constrain(
        geometric_camera &pCamera2, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, const float sigmaLevel, const float unc);

    bool match_and_triangulate(const cv::KeyPoint &kp1,
                               const cv::KeyPoint &kp2,
                               geometric_camera   &pOther,
                               Sophus::SE3f       &Tcw1,
                               Sophus::SE3f       &Tcw2,
                               const float         sigmaLevel1,
                               const float         sigmaLevel2,
                               Eigen::Vector3f    &x3Dtriangulated)
    {
        return false;
    }

    friend std::ostream &operator<<(std::ostream &os, const pinhole &ph);
    friend std::istream &operator>>(std::istream &os, pinhole &ph);

    bool is_equal(geometric_camera &pCam);

  private:
    // Parameters vector corresponds to
    //       [fx, fy, cx, cy]
    two_view_reconstruction *_two_view_reconstruction;
};
} // namespace ORB_SLAM3

#endif // CAMERAMODELS_PINHOLE_H
