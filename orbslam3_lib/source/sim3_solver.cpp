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

#include "sim3_solver.h"

#include "DUtils/random.h"
#include "descriptor_matcher.h"
#include "key_frame.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

sim3_solver::sim3_solver(key_frame *first_key_frame, key_frame *second_key_frame, const vector<map_point *> &matched_map_points, const bool fixed_scale, vector<key_frame *> matched_key_frames)
    : iterations(0), best_inlier_count(0), fix_scale(fixed_scale), cameras(first_key_frame->cameras)
{
    bool bDifferentKFs = false;
    if (matched_key_frames.empty())
    {
        bDifferentKFs      = true;
        matched_key_frames = vector<key_frame *>(matched_map_points.size(), second_key_frame);
    }

    key_frame_1 = first_key_frame;
    key_frame_2 = second_key_frame;

    vector<map_point *> vpKeyFrameMP1 = first_key_frame->get_map_point_matches();

    initial_match_count = matched_map_points.size();

    map_points_1.reserve(initial_match_count);
    map_points_2.reserve(initial_match_count);
    matched_map_points_12 = matched_map_points;
    matched_indices_1.reserve(initial_match_count);
    camera_points_1.reserve(initial_match_count);
    camera_points_2.reserve(initial_match_count);

    Eigen::Matrix3f Rcw1 = first_key_frame->get_rotation();
    Eigen::Vector3f tcw1 = first_key_frame->get_translation();
    Eigen::Matrix3f Rcw2 = second_key_frame->get_rotation();
    Eigen::Vector3f tcw2 = second_key_frame->get_translation();

    all_indices.reserve(initial_match_count);

    size_t idx = 0;

    key_frame *pKFm = second_key_frame; // Default variable
    for (int i1 = 0; i1 < initial_match_count; i1++)
    {
        if (matched_map_points[i1])
        {
            map_point *pMP1 = vpKeyFrameMP1[i1];
            map_point *pMP2 = matched_map_points[i1];

            if (!pMP1)
            {
                continue;
            }

            if (pMP1->is_bad() || pMP2->is_bad())
            {
                continue;
            }

            if (bDifferentKFs)
            {
                pKFm = matched_key_frames[i1];
            }

            int indexKF1 = get<0>(pMP1->get_index_in_key_frame(first_key_frame));
            int indexKF2 = get<0>(pMP2->get_index_in_key_frame(pKFm));

            if (indexKF1 < 0 || indexKF2 < 0)
            {
                continue;
            }

            const auto &kp1 = first_key_frame->keypoints[indexKF1];
            const auto &kp2 = pKFm->keypoints[indexKF2];

            const float sigmaSquare1 = first_key_frame->level_sigma_squared[kp1.octave];
            const float sigmaSquare2 = pKFm->level_sigma_squared[kp2.octave];

            max_error_1.push_back(9.210 * sigmaSquare1);
            max_error_2.push_back(9.210 * sigmaSquare2);

            map_points_1.push_back(pMP1);
            map_points_2.push_back(pMP2);
            matched_indices_1.push_back(i1);

            Eigen::Vector3f X3D1w = pMP1->get_world_position();
            camera_points_1.push_back(Rcw1 * X3D1w + tcw1);

            Eigen::Vector3f X3D2w = pMP2->get_world_position();
            camera_points_2.push_back(Rcw2 * X3D2w + tcw2);

            all_indices.push_back(idx);
            idx++;
        }
    }

    from_camera_to_image(camera_points_1, points_1_image_1, *cameras[0]);
    from_camera_to_image(camera_points_2, points_2_image_2, *cameras[0]);

    set_ransac_parameters();
}

void sim3_solver::set_ransac_parameters(double probability, int minimum_inliers, int maximum_iterations)
{
    ransac_probability    = probability;
    ransac_min_inliers    = minimum_inliers;
    ransac_max_iterations = maximum_iterations;

    number_of_correspondences = map_points_1.size(); // number of correspondences

    current_inliers.resize(number_of_correspondences);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)ransac_min_inliers / number_of_correspondences;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int iteration_limit;

    if (ransac_min_inliers == number_of_correspondences)
    {
        iteration_limit = 1;
    }
    else
    {
        iteration_limit = ceil(log(1 - ransac_probability) / log(1 - pow(epsilon, 3)));
    }

    ransac_max_iterations = max(1, min(iteration_limit, ransac_max_iterations));

    iterations = 0;
}

Eigen::Matrix4f sim3_solver::iterate(int iteration_limit, bool &no_more_iterations, vector<bool> &inliers, int &inlier_count)
{
    no_more_iterations = false;
    inliers            = vector<bool>(initial_match_count, false);
    inlier_count       = 0;

    if (number_of_correspondences < ransac_min_inliers)
    {
        no_more_iterations = true;
        return Eigen::Matrix4f::Identity();
    }

    vector<size_t> vAvailableIndices;

    Eigen::Matrix3f P3Dc1i;
    Eigen::Matrix3f P3Dc2i;

    int nCurrentIterations = 0;
    while (iterations < ransac_max_iterations && nCurrentIterations < iteration_limit)
    {
        nCurrentIterations++;
        iterations++;

        vAvailableIndices = all_indices;

        // Get min set of points
        for (short i = 0; i < 3; ++i)
        {
            int randi = DUtils::random::random_int(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            P3Dc1i.col(i) = camera_points_1[idx];
            P3Dc2i.col(i) = camera_points_2[idx];

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        if (!compute_sim3(P3Dc1i, P3Dc2i))
        {
            continue;
        }

        check_inliers();

        if (current_inlier_count >= best_inlier_count)
        {
            best_inliers           = current_inliers;
            best_inlier_count      = current_inlier_count;
            best_transformation_12 = transformation_12;
            best_rotation          = rotation_12;
            best_translation       = translation_12;
            best_scale             = scale_12;

            if (current_inlier_count > ransac_min_inliers)
            {
                inlier_count = current_inlier_count;
                for (int i = 0; i < number_of_correspondences; i++)
                {
                    if (current_inliers[i])
                    {
                        inliers[matched_indices_1[i]] = true;
                    }
                }
                return best_transformation_12;
            }
        }
    }

    if (iterations >= ransac_max_iterations)
    {
        no_more_iterations = true;
    }

    return Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f sim3_solver::iterate(int iteration_limit, bool &no_more_iterations, vector<bool> &inliers, int &inlier_count, bool &converged)
{
    no_more_iterations = false;
    converged          = false;
    inliers            = vector<bool>(initial_match_count, false);
    inlier_count       = 0;

    if (number_of_correspondences < ransac_min_inliers)
    {
        no_more_iterations = true;
        return Eigen::Matrix4f::Identity();
    }

    vector<size_t> vAvailableIndices;

    Eigen::Matrix3f P3Dc1i;
    Eigen::Matrix3f P3Dc2i;

    int nCurrentIterations = 0;

    Eigen::Matrix4f bestSim3;

    while (iterations < ransac_max_iterations && nCurrentIterations < iteration_limit)
    {
        nCurrentIterations++;
        iterations++;

        vAvailableIndices = all_indices;

        // Get min set of points
        for (short i = 0; i < 3; ++i)
        {
            int randi = DUtils::random::random_int(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            P3Dc1i.col(i) = camera_points_1[idx];
            P3Dc2i.col(i) = camera_points_2[idx];

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        if (!compute_sim3(P3Dc1i, P3Dc2i))
        {
            continue;
        }

        check_inliers();

        if (current_inlier_count >= best_inlier_count)
        {
            best_inliers           = current_inliers;
            best_inlier_count      = current_inlier_count;
            best_transformation_12 = transformation_12;
            best_rotation          = rotation_12;
            best_translation       = translation_12;
            best_scale             = scale_12;

            if (current_inlier_count > ransac_min_inliers)
            {
                inlier_count = current_inlier_count;
                for (int i = 0; i < number_of_correspondences; i++)
                {
                    if (current_inliers[i])
                    {
                        inliers[matched_indices_1[i]] = true;
                    }
                }
                converged = true;
                return best_transformation_12;
            }
            else
            {
                bestSim3 = best_transformation_12;
            }
        }
    }

    if (iterations >= ransac_max_iterations)
    {
        no_more_iterations = true;
    }

    return bestSim3;
}

Eigen::Matrix4f sim3_solver::find(vector<bool> &inliers_12, int &inlier_count)
{
    bool bFlag;
    return iterate(ransac_max_iterations, bFlag, inliers_12, inlier_count);
}

void sim3_solver::compute_centroid(Eigen::Matrix3f &points, Eigen::Matrix3f &centered_points, Eigen::Vector3f &centroid)
{
    centroid = points.rowwise().sum();
    centroid = centroid / points.cols();
    for (int i = 0; i < points.cols(); i++)
    {
        centered_points.col(i) = points.col(i) - centroid;
    }
}

bool sim3_solver::compute_sim3(Eigen::Matrix3f &points_1, Eigen::Matrix3f &points_2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    Eigen::Matrix3f Pr1; // Relative coordinates to centroid (set 1)
    Eigen::Matrix3f Pr2; // Relative coordinates to centroid (set 2)
    Eigen::Vector3f O1;  // Centroid of points_1
    Eigen::Vector3f O2;  // Centroid of points_2

    compute_centroid(points_1, Pr1, O1);
    compute_centroid(points_2, Pr2, O2);

    // Step 2: Compute M matrix

    Eigen::Matrix3f M = Pr2 * Pr1.transpose();

    // Step 3: Compute N matrix
    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    Eigen::Matrix4f normal_matrix;

    N11 = M(0, 0) + M(1, 1) + M(2, 2);
    N12 = M(1, 2) - M(2, 1);
    N13 = M(2, 0) - M(0, 2);
    N14 = M(0, 1) - M(1, 0);
    N22 = M(0, 0) - M(1, 1) - M(2, 2);
    N23 = M(0, 1) + M(1, 0);
    N24 = M(2, 0) + M(0, 2);
    N33 = -M(0, 0) + M(1, 1) - M(2, 2);
    N34 = M(1, 2) + M(2, 1);
    N44 = -M(0, 0) - M(1, 1) + M(2, 2);

    normal_matrix << N11, N12, N13, N14, N12, N22, N23, N24, N13, N23, N33, N34, N14, N24, N34, N44;

    // Step 4: Eigenvector of the highest eigenvalue
    Eigen::EigenSolver<Eigen::Matrix4f> eigSolver;
    eigSolver.compute(normal_matrix);

    Eigen::Vector4f eval = eigSolver.eigenvalues().real();
    Eigen::Matrix4f evec = eigSolver.eigenvectors().real(); // evec[0] is the quaternion of the desired rotation

    int maxIndex; // should be zero
    eval.maxCoeff(&maxIndex);

    // Building R from the quaternion directly avoids 0/0 in the angle-axis form for identity rotations.
    Eigen::Quaternionf q(evec(0, maxIndex), evec(1, maxIndex), evec(2, maxIndex), evec(3, maxIndex));
    const float        q_norm = q.norm();
    if (!std::isfinite(q_norm) || q_norm < std::numeric_limits<float>::epsilon())
    {
        return false;
    }
    q.coeffs()  /= q_norm;
    rotation_12  = q.toRotationMatrix();

    // Step 5: Rotate set 2
    Eigen::Matrix3f P3 = rotation_12 * Pr2;

    // Step 6: Scale

    if (!fix_scale)
    {
        double cvnom = converter::to_cv_mat(Pr1).dot(converter::to_cv_mat(P3));
        double nom   = (Pr1.array() * P3.array()).sum();
        if (abs(nom - cvnom) > 1e-3)
        {
            std::ostringstream solver_output;
            solver_output << nom;
            spdlog::debug("Sim3 solver numerator mismatch: delta={}, numerator={}", abs(nom - cvnom), solver_output.str());
        }
        Eigen::Array<float, 3, 3> aux_P3;
        aux_P3     = P3.array() * P3.array();
        double den = aux_P3.sum();

        scale_12 = nom / den;
    }
    else
    {
        scale_12 = 1.0f;
    }

    if (!std::isfinite(scale_12) || scale_12 <= 0.0f)
    {
        return false;
    }

    // Step 7: Translation
    translation_12 = O1 - scale_12 * rotation_12 * O2;

    // Step 8: Transformation

    // Step 8.1 T12
    transformation_12.setIdentity();

    Eigen::Matrix3f sR                  = scale_12 * rotation_12;
    transformation_12.block<3, 3>(0, 0) = sR;
    transformation_12.block<3, 1>(0, 3) = translation_12;

    // Step 8.2 T21
    transformation_21.setIdentity();
    Eigen::Matrix3f sRinv = (1.0 / scale_12) * rotation_12.transpose();

    transformation_21.block<3, 3>(0, 0) = sRinv;

    Eigen::Vector3f tinv                = -sRinv * translation_12;
    transformation_21.block<3, 1>(0, 3) = tinv;
    return true;
}

void sim3_solver::check_inliers()
{
    vector<Eigen::Vector2f> vP1im2, vP2im1;
    project(camera_points_2, vP2im1, transformation_12, *cameras[0]);
    project(camera_points_1, vP1im2, transformation_21, *cameras[0]);

    current_inlier_count = 0;

    for (size_t i = 0; i < points_1_image_1.size(); i++)
    {
        Eigen::Vector2f dist1 = points_1_image_1[i] - vP2im1[i];
        Eigen::Vector2f dist2 = vP1im2[i] - points_2_image_2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if (err1 < max_error_1[i] && err2 < max_error_2[i])
        {
            current_inliers[i] = true;
            current_inlier_count++;
        }
        else
        {
            current_inliers[i] = false;
        }
    }
}

Eigen::Matrix4f sim3_solver::get_estimated_transformation() { return best_transformation_12; }

Eigen::Matrix3f sim3_solver::get_estimated_rotation() { return best_rotation; }

Eigen::Vector3f sim3_solver::get_estimated_translation() { return best_translation; }

float sim3_solver::get_estimated_scale() { return best_scale; }

void sim3_solver::project(const vector<Eigen::Vector3f> &world_points, vector<Eigen::Vector2f> &image_points, Eigen::Matrix4f camera_from_world, geometric_camera &camera)
{
    Eigen::Matrix3f Rcw = camera_from_world.block<3, 3>(0, 0);
    Eigen::Vector3f tcw = camera_from_world.block<3, 1>(0, 3);

    image_points.clear();
    image_points.reserve(world_points.size());

    for (size_t i = 0, iend = world_points.size(); i < iend; i++)
    {
        Eigen::Vector3f P3Dc = Rcw * world_points[i] + tcw;
        Eigen::Vector2f pt2D = camera.project(P3Dc);
        image_points.push_back(pt2D);
    }
}

void sim3_solver::from_camera_to_image(const vector<Eigen::Vector3f> &camera_points, vector<Eigen::Vector2f> &image_points, geometric_camera &camera)
{
    image_points.clear();
    image_points.reserve(camera_points.size());

    for (size_t i = 0, iend = camera_points.size(); i < iend; i++)
    {
        Eigen::Vector2f pt2D = camera.project(camera_points[i]);
        image_points.push_back(pt2D);
    }
}

} // namespace ORB_SLAM3
