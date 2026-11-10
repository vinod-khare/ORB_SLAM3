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

#include "geometric_tools.h"

#include "CameraModels/GeometricCamera.h"
#include "key_frame.h"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>

namespace ORB_SLAM3
{

bool geometric_tools::solve_pnp_ransac(const std::vector<cv::Point3f> &world_points,
                                       const std::vector<cv::Point2f> &image_points,
                                       const std::vector<float>       &max_errors,
                                       GeometricCamera                &camera,
                                       int                             min_inliers,
                                       Eigen::Matrix4f                &camera_from_world,
                                       std::vector<bool>              &inliers)
{
    inliers.assign(world_points.size(), false);
    if (world_points.size() < static_cast<size_t>(min_inliers) || image_points.size() != world_points.size() || max_errors.size() != world_points.size())
    {
        return false;
    }

    const bool is_fisheye      = camera.GetType() == GeometricCamera::CAM_FISHEYE;
    double     max_pixel_error = 0.0;
    for (size_t i = 0; i < image_points.size(); ++i)
    {
        if (max_errors[i] <= 0.0f)
        {
            return false;
        }
        max_pixel_error = std::max(max_pixel_error, static_cast<double>(max_errors[i]));
    }

    cv::Mat camera_matrix = cv::Mat(cv::Matx33d(camera.getParameter(0), 0.0, camera.getParameter(2), 0.0, camera.getParameter(1), camera.getParameter(3), 0.0, 0.0, 1.0));
    cv::Mat distortion_coefficients;
    if (is_fisheye)
    {
        distortion_coefficients = cv::Mat(cv::Vec4d(camera.getParameter(4), camera.getParameter(5), camera.getParameter(6), camera.getParameter(7)));
    }

    cv::Mat          rotation_vector;
    cv::Mat          translation_vector;
    std::vector<int> ransac_inliers;
    const double     reprojection_threshold = std::sqrt(max_pixel_error);

    try
    {
        bool solved = false;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 12)
        if (is_fisheye)
        {
            solved = cv::fisheye::solvePnPRansac(world_points,
                                                 image_points,
                                                 camera_matrix,
                                                 distortion_coefficients,
                                                 rotation_vector,
                                                 translation_vector,
                                                 false,
                                                 300,
                                                 static_cast<float>(reprojection_threshold),
                                                 0.99,
                                                 ransac_inliers,
                                                 cv::SOLVEPNP_EPNP);
        }
        else
        {
            solved = cv::solvePnPRansac(
                world_points, image_points, camera_matrix, cv::noArray(), rotation_vector, translation_vector, false, 300, reprojection_threshold, 0.99, ransac_inliers, cv::SOLVEPNP_EPNP);
        }
#else
        std::vector<cv::Point2f> pnp_image_points     = image_points;
        cv::Mat                  ransac_camera_matrix = camera_matrix;
        double                   ransac_threshold     = reprojection_threshold;

        if (is_fisheye)
        {
            const double focal_length = std::min(camera.getParameter(0), camera.getParameter(1));

            if (focal_length <= 0.0)
            {
                return false;
            }

            for (size_t i = 0; i < image_points.size(); ++i)
            {
                const cv::Point3f bearing = camera.unproject(image_points[i]);
                if (!std::isfinite(bearing.x) || !std::isfinite(bearing.y) || !std::isfinite(bearing.z) || bearing.z <= 0.0f)
                {
                    return false;
                }
                pnp_image_points[i] = cv::Point2f(bearing.x / bearing.z, bearing.y / bearing.z);
            }

            ransac_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            ransac_threshold     = std::max(0.01, 2.0 * reprojection_threshold / focal_length);
        }

        solved = cv::solvePnPRansac(
            world_points, pnp_image_points, ransac_camera_matrix, cv::noArray(), rotation_vector, translation_vector, false, 300, ransac_threshold, 0.99, ransac_inliers, cv::SOLVEPNP_EPNP);
#endif
        if (!solved || ransac_inliers.size() < static_cast<size_t>(min_inliers))
        {
            return false;
        }

        std::vector<cv::Point3f> refine_world_points;
        std::vector<cv::Point2f> refine_image_points;
        refine_world_points.reserve(ransac_inliers.size());
        refine_image_points.reserve(ransac_inliers.size());

        for (const int index : ransac_inliers)
        {
            refine_world_points.push_back(world_points[index]);
            refine_image_points.push_back(image_points[index]);
        }

        if (is_fisheye)
        {
            cv::fisheye::solvePnP(refine_world_points, refine_image_points, camera_matrix, distortion_coefficients, rotation_vector, translation_vector, true, cv::SOLVEPNP_ITERATIVE);
        }
        else
        {
            cv::solvePnPRefineLM(refine_world_points, refine_image_points, camera_matrix, cv::noArray(), rotation_vector, translation_vector);
        }

        cv::Mat rotation;
        cv::Rodrigues(rotation_vector, rotation);
        rotation.convertTo(rotation, CV_32F);
        translation_vector.convertTo(translation_vector, CV_32F);

        Eigen::Matrix4f candidate_pose = Eigen::Matrix4f::Identity();

        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                candidate_pose(row, col) = rotation.at<float>(row, col);
            }

            candidate_pose(row, 3) = translation_vector.at<float>(row);
        }

        const auto classify_inliers = [&](std::vector<bool> &mask)
        {
            mask.assign(world_points.size(), false);
            int count = 0;

            for (size_t i = 0; i < world_points.size(); ++i)
            {
                const Eigen::Vector3f point(world_points[i].x, world_points[i].y, world_points[i].z);
                const Eigen::Vector3f camera_point = candidate_pose.block<3, 3>(0, 0) * point + candidate_pose.block<3, 1>(0, 3);

                if (camera_point.z() <= 0.0f)
                {
                    continue;
                }

                const Eigen::Vector2f projected = camera.project(camera_point);
                const Eigen::Vector2f observed(image_points[i].x, image_points[i].y);

                if ((projected - observed).squaredNorm() < max_errors[i])
                {
                    mask[i] = true;
                    ++count;
                }
            }

            return count;
        };

        if (classify_inliers(inliers) < min_inliers)
        {
            return false;
        }

        refine_world_points.clear();
        refine_image_points.clear();

        for (size_t i = 0; i < inliers.size(); ++i)
        {
            if (inliers[i])
            {
                refine_world_points.push_back(world_points[i]);
                refine_image_points.push_back(image_points[i]);
            }
        }

        if (is_fisheye)
        {
            cv::fisheye::solvePnP(refine_world_points, refine_image_points, camera_matrix, distortion_coefficients, rotation_vector, translation_vector, true, cv::SOLVEPNP_ITERATIVE);
        }
        else
        {
            cv::solvePnPRefineLM(refine_world_points, refine_image_points, camera_matrix, cv::noArray(), rotation_vector, translation_vector);
        }

        cv::Rodrigues(rotation_vector, rotation);
        rotation.convertTo(rotation, CV_32F);
        translation_vector.convertTo(translation_vector, CV_32F);

        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                candidate_pose(row, col) = rotation.at<float>(row, col);
            }

            candidate_pose(row, 3) = translation_vector.at<float>(row);
        }

        if (classify_inliers(inliers) < min_inliers)
        {
            return false;
        }

        camera_from_world = candidate_pose;

        return true;
    }
    catch (const cv::Exception &)
    {
        return false;
    }
}

Eigen::Matrix3f geometric_tools::compute_f12(key_frame *&key_frame1, key_frame *&key_frame2)
{
    Sophus::SE3<float>                    camera1_from_world      = key_frame1->get_pose();
    Sophus::Matrix3<float>                rotation1_from_world    = camera1_from_world.rotationMatrix();
    Sophus::SE3<float>::TranslationMember translation1_from_world = camera1_from_world.translation();

    Sophus::SE3<float>                    camera2_from_world      = key_frame2->get_pose();
    Sophus::Matrix3<float>                rotation2_from_world    = camera2_from_world.rotationMatrix();
    Sophus::SE3<float>::TranslationMember translation2_from_world = camera2_from_world.translation();

    Sophus::Matrix3<float> rotation1_from_camera2    = rotation1_from_world * rotation2_from_world.transpose();
    Eigen::Vector3f        translation1_from_camera2 = -rotation1_from_camera2 * translation2_from_world + translation1_from_world;

    Eigen::Matrix3f translation_skew = Sophus::SO3f::hat(translation1_from_camera2);

    const Eigen::Matrix3f calibration1 = key_frame1->cameras[0]->toK_();
    const Eigen::Matrix3f calibration2 = key_frame2->cameras[0]->toK_();

    return calibration1.transpose().inverse() * translation_skew * rotation1_from_camera2 * calibration2.inverse();
}

bool geometric_tools::triangulate(
    Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2, Eigen::Matrix<float, 3, 4> &camera1_from_world, Eigen::Matrix<float, 3, 4> &camera2_from_world, Eigen::Vector3f &point3d)
{
    Eigen::Matrix4f system;
    system.block<1, 4>(0, 0) = x_c1(0) * camera1_from_world.block<1, 4>(2, 0) - camera1_from_world.block<1, 4>(0, 0);
    system.block<1, 4>(1, 0) = x_c1(1) * camera1_from_world.block<1, 4>(2, 0) - camera1_from_world.block<1, 4>(1, 0);
    system.block<1, 4>(2, 0) = x_c2(0) * camera2_from_world.block<1, 4>(2, 0) - camera2_from_world.block<1, 4>(0, 0);
    system.block<1, 4>(3, 0) = x_c2(1) * camera2_from_world.block<1, 4>(2, 0) - camera2_from_world.block<1, 4>(1, 0);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(system, Eigen::ComputeFullV);

    Eigen::Vector4f homogeneous_point = svd.matrixV().col(3);

    if (homogeneous_point(3) == 0)
    {
        return false;
    }

    // Euclidean coordinates
    point3d = homogeneous_point.head(3) / homogeneous_point(3);

    return true;
}

} // namespace ORB_SLAM3
