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

#include "frame.h"

#include "CameraModels/KannalaBrandt8.h"
#include "G2oTypes.h"
#include "geometric_camera.h"
#include "key_frame.h"
#include "converter.h"
#include "descriptor_matcher.h"
#include "map_point.h"

#include <array>
#include <chrono>
#include <cmath>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

long unsigned int frame::next_id              = 0;
bool              frame::initial_computations = true;
float             frame::cx, frame::cy, frame::fx, frame::fy, frame::invfx, frame::invfy;
float             frame::min_x, frame::min_y, frame::max_x, frame::max_y;
float             frame::grid_element_width_inverse, frame::grid_element_height_inverse;

// For stereo fisheye matching

frame::frame()
    : imu_constraint(nullptr), _has_pose(false), _has_velocity(false), imu_preintegrated(nullptr), previous_frame(nullptr), frame_imu_preintegrated(nullptr), reference_key_frame(nullptr),
      _is_set(false), _imu_preintegrated(false)
{
}

// Copy Constructor
frame::frame(const frame &frame)
    : imu_constraint(frame.imu_constraint), _camera_from_world(frame._camera_from_world), _has_pose(false), _left_to_right(frame._left_to_right), _right_to_left(frame._right_to_left),
      _left_to_right_rotation(frame._left_to_right_rotation), _left_to_right_translation(frame._left_to_right_translation), _has_velocity(false),
      place_vocabulary_ptr(frame.place_vocabulary_ptr), timestamp(frame.timestamp), optical_flow_pyramid(frame.optical_flow_pyramid),
      right_optical_flow_pyramid(frame.right_optical_flow_pyramid), calibration_matrix(frame.calibration_matrix.clone()),
      eigen_calibration_matrix(converter::to_matrix3f(frame.calibration_matrix)), distortion_coefficients(frame.distortion_coefficients.clone()), baseline_fx(frame.baseline_fx),
      baseline(frame.baseline), depth_threshold(frame.depth_threshold), feature_count(frame.feature_count), keypoints(frame.keypoints), right_keypoints(frame.right_keypoints),
      is_fisheye_stereo(frame.is_fisheye_stereo), map_points(frame.map_points), right_coordinates(frame.right_coordinates), depths(frame.depths), bow_vector(frame.bow_vector),
      feature_vector(frame.feature_vector), outliers(frame.outliers), close_map_point_count(frame.close_map_point_count), imu_bias(frame.imu_bias), imu_calibration(frame.imu_calibration),
      imu_preintegrated(frame.imu_preintegrated), last_key_frame(frame.last_key_frame), previous_frame(frame.previous_frame), frame_imu_preintegrated(frame.frame_imu_preintegrated),
      id(frame.id), reference_key_frame(frame.reference_key_frame), scale_levels(frame.scale_levels), scale_factor(frame.scale_factor), log_scale_factor(frame.log_scale_factor),
      scale_factors(frame.scale_factors), inverse_scale_factors(frame.inverse_scale_factors), level_sigma_squared(frame.level_sigma_squared),
      inverse_level_sigma_squared(frame.inverse_level_sigma_squared), name_file(frame.name_file), dataset(frame.dataset), _is_set(frame._is_set), _imu_preintegrated(frame._imu_preintegrated),
      _imu_mutex(frame._imu_mutex), mono_left(frame.mono_left), mono_right(frame.mono_right), left_to_right_matches(frame.left_to_right_matches),
      right_to_left_matches(frame.right_to_left_matches), stereo_3d_points(frame.stereo_3d_points)
{
    color_image         = frame.color_image;
    color_image_right   = frame.color_image_right;
    color_is_rgb        = frame.color_is_rgb;
    flow_only           = frame.flow_only;
    flow_candidates     = frame.flow_candidates;
    flow_tracks         = frame.flow_tracks;
    flow_inliers        = frame.flow_inliers;
    flow_occupied_cells = frame.flow_occupied_cells;
    extraction_ms       = frame.extraction_ms;
    stereo_matching_ms  = frame.stereo_matching_ms;
    optical_flow_ms     = frame.optical_flow_ms;
    for (int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
        {
            grid[i][j] = frame.grid[i][j];
            if (frame.is_fisheye_stereo)
            {
                right_grid[i][j] = frame.right_grid[i][j];
            }
        }
    }

    if (frame._has_pose)
    {
        set_pose(frame.get_pose());
    }

    if (frame.has_velocity())
    {
        set_velocity(frame.get_velocity());
    }

    projected_points = frame.projected_points;
    matched_in_image = frame.matched_in_image;
}

std::array<unsigned char, 3> frame::keypoint_color(int index) const
{
    const bool  right     = is_fisheye_stereo && index >= static_cast<int>(keypoints.size());
    const auto &image     = right ? color_image_right : color_image;
    const auto &keys      = right ? right_keypoints : keypoints;
    const int   key_index = right ? index - static_cast<int>(keypoints.size()) : index;
    if (image.empty() || image.depth() != CV_8U || key_index < 0 || static_cast<std::size_t>(key_index) >= keys.size())
    {
        return {255, 255, 255};
    }
    const auto &position = keys[key_index].pt;
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || position.x < 0 || position.y < 0 || position.x >= image.cols || position.y >= image.rows)
    {
        return {255, 255, 255};
    }
    const int   column = std::min(cvRound(position.x), image.cols - 1);
    const int   row    = std::min(cvRound(position.y), image.rows - 1);
    const auto *pixel  = image.ptr<unsigned char>(row) + column * image.channels();
    if (image.channels() == 1)
    {
        return {pixel[0], pixel[0], pixel[0]};
    }
    if (image.channels() == 3 || image.channels() == 4)
    {
        return color_is_rgb ? std::array<unsigned char, 3>{pixel[0], pixel[1], pixel[2]} : std::array<unsigned char, 3>{pixel[2], pixel[1], pixel[0]};
    }
    return {255, 255, 255};
}

const std::vector<cv::Mat> &frame::get_optical_flow_pyramid(const cv::Mat &image, bool right)
{
    auto &pyramid = right ? right_optical_flow_pyramid : optical_flow_pyramid;
    if (pyramid.empty())
    {
        pyramid = build_optical_flow_pyramid(image);
    }

    return pyramid;
}

std::vector<cv::Mat> frame::build_optical_flow_pyramid(const cv::Mat &image)
{
    std::vector<cv::Mat> pyramid;
    cv::buildOpticalFlowPyramid(image, pyramid, cv::Size(21, 21), 3, true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
    return pyramid;
}

frame::frame(frame &previous, double timestamp) : frame(static_cast<const frame &>(previous))
{
    id                      = next_id++;
    this->timestamp         = timestamp;
    previous_frame          = &previous;
    imu_constraint          = nullptr;
    imu_preintegrated       = nullptr;
    frame_imu_preintegrated = nullptr;
    _imu_preintegrated      = false;
    _has_pose               = false;
    _is_set                 = false;
    flow_candidates         = 0;
    flow_tracks             = 0;
    flow_inliers            = 0;
    flow_occupied_cells     = 0;
    extraction_ms           = 0;
    stereo_matching_ms      = 0;
    optical_flow_ms         = 0;

    optical_flow_pyramid.clear();
    right_optical_flow_pyramid.clear();
    left_image.release();
    right_image.release();

    set_tracked_features({}, {});
}

void frame::set_tracked_features(std::vector<keypoint> keys, std::vector<map_point *> points, std::vector<keypoint> right_keys, std::vector<map_point *> right_points)
{
    CV_Assert(keys.size() == points.size());
    CV_Assert(right_keys.size() == right_points.size());
    CV_Assert(is_fisheye_stereo || right_keys.empty());
    flow_only       = true;
    keypoints       = std::move(keys);
    right_keypoints = std::move(right_keys);
    for (auto *view : {&keypoints, &right_keypoints})
    {
        for (auto &key : *view)
        {
            key.descriptor.release();
            if (is_fisheye_stereo)
            {
                key.pt_undistorted = key.pt;
            }
        }
    }
    map_points = std::move(points);
    map_points.insert(map_points.end(), right_points.begin(), right_points.end());
    feature_count = static_cast<int>(map_points.size());
    mono_left = mono_right = -1;
    bow_vector.clear();
    feature_vector.clear();
    outliers.assign(feature_count, false);
    right_coordinates.assign(keypoints.size(), -1.0f);
    depths.assign(keypoints.size(), -1.0f);
    close_map_point_count = 0;
    left_to_right_matches.assign(is_fisheye_stereo ? keypoints.size() : 0, -1);
    right_to_left_matches.assign(right_keypoints.size(), -1);
    stereo_3d_points.clear();
    projected_points.clear();
    matched_in_image.clear();
    if (!is_fisheye_stereo && feature_count > 0)
    {
        _undistort_keypoints();
    }
    for (int x = 0; x < FRAME_GRID_COLS; ++x)
    {
        for (int y = 0; y < FRAME_GRID_ROWS; ++y)
        {
            grid[x][y].clear();
            right_grid[x][y].clear();
        }
    }
    _assign_features_to_grid();
}

void frame::initialize_features(const cv::Mat &left, const camera_vector &cameras, const cv::Mat &right, const std::vector<cv::Mat> &left_pyramid, const std::vector<cv::Mat> &right_pyramid)
{
    flow_only = false;
    bow_vector.clear();
    feature_vector.clear();
    projected_points.clear();
    matched_in_image.clear();
    for (int x = 0; x < FRAME_GRID_COLS; ++x)
    {
        for (int y = 0; y < FRAME_GRID_ROWS; ++y)
        {
            grid[x][y].clear();
            right_grid[x][y].clear();
        }
    }
    is_fisheye_stereo = ORB_SLAM3::is_fisheye_stereo(cameras);
    feature_count     = static_cast<int>(keypoints.size() + (is_fisheye_stereo ? right_keypoints.size() : 0));
    map_points.assign(feature_count, nullptr);
    outliers.assign(feature_count, false);
    right_coordinates.assign(keypoints.size(), -1.0f);
    depths.assign(keypoints.size(), -1.0f);
    left_to_right_matches.clear();
    right_to_left_matches.clear();
    stereo_3d_points.clear();
    close_map_point_count = 0;
    if (!is_fisheye_stereo)
    {
        mono_left = mono_right = -1;
    }
    if (feature_count == 0)
    {
        return;
    }
    if (initial_computations)
    {
        _compute_image_bounds(left);
        grid_element_width_inverse  = static_cast<float>(FRAME_GRID_COLS) / (max_x - min_x);
        grid_element_height_inverse = static_cast<float>(FRAME_GRID_ROWS) / (max_y - min_y);
        fx                          = calibration_matrix.at<float>(0, 0);
        fy                          = calibration_matrix.at<float>(1, 1);
        cx                          = calibration_matrix.at<float>(0, 2);
        cy                          = calibration_matrix.at<float>(1, 2);
        invfx                       = 1.0f / fx;
        invfy                       = 1.0f / fy;
        initial_computations        = false;
    }
    baseline                  = baseline_fx / fx;
    const auto stereo_started = std::chrono::steady_clock::now();
    if (is_fisheye_stereo)
    {
        left_image  = left.clone();
        right_image = right.clone();
        compute_stereo_fisheye_matches(cameras);
    }
    else if (!right.empty())
    {
        compute_stereo_matches(left_pyramid, right_pyramid);
    }
    stereo_matching_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stereo_started).count();
    if (is_fisheye_stereo)
    {
        for (auto &key : keypoints)
        {
            key.pt_undistorted = key.pt;
        }
        for (auto &key : right_keypoints)
        {
            key.pt_undistorted = key.pt;
        }
    }
    else
    {
        _undistort_keypoints();
    }
    _assign_features_to_grid();
}

bool frame::reliable_flow_inliers(const cv::Size &image_size, int minimum_inliers)
{
    flow_inliers        = 0;
    flow_occupied_cells = 0;

    if (image_size.width <= 0 || image_size.height <= 0 || minimum_inliers < 50)
    {
        return false;
    }

    std::array<bool, 16> occupied{};
    int                  inliers = 0;

    for (std::size_t index = 0; index < map_points.size(); ++index)
    {
        if (!map_points[index] || outliers[index])
        {
            continue;
        }

        const auto &point = index < keypoints.size() ? keypoints[index].pt : right_keypoints[index - keypoints.size()].pt;

        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 || point.y < 0 || point.x >= image_size.width || point.y >= image_size.height)
        {
            continue;
        }

        const int x = static_cast<int>(point.x * 4 / image_size.width);
        const int y = static_cast<int>(point.y * 4 / image_size.height);

        occupied[static_cast<std::size_t>(y * 4 + x)] = true;

        ++inliers;
    }

    flow_inliers        = inliers;
    flow_occupied_cells = static_cast<int>(std::count(occupied.begin(), occupied.end(), true));

    return inliers >= minimum_inliers && inliers >= 0.6 * feature_count && flow_occupied_cells >= 6;
}

frame::frame(double              timestamp,
             place_vocabulary   *vocabulary,
             const cv::Mat      &calibration,
             const cv::Mat      &distortion,
             float               baseline_fx,
             float               depth_threshold,
             frame              *previous,
             const IMU::Calib   &imu_calibration,
             const Sophus::SE3f &left_to_right)
    : imu_constraint(nullptr), _has_pose(false), _left_to_right(left_to_right), _right_to_left(left_to_right.inverse()), _left_to_right_rotation(left_to_right.rotationMatrix()),
      _left_to_right_translation(left_to_right.translation()), _has_velocity(false), place_vocabulary_ptr(vocabulary), timestamp(timestamp), calibration_matrix(calibration.clone()),
      eigen_calibration_matrix(converter::to_matrix3f(calibration)), distortion_coefficients(distortion.clone()), baseline_fx(baseline_fx), baseline(0), depth_threshold(depth_threshold),
      feature_count(0), close_map_point_count(0), imu_calibration(imu_calibration), imu_preintegrated(nullptr), last_key_frame(nullptr), previous_frame(previous),
      frame_imu_preintegrated(nullptr), id(next_id++), reference_key_frame(nullptr), scale_levels(0), scale_factor(1), log_scale_factor(0), dataset(0), _is_set(false),
      _imu_preintegrated(false), mono_left(-1), mono_right(-1)
{
    if (previous && previous->has_velocity())
    {
        set_velocity(previous->get_velocity());
    }
    else
    {
        _world_velocity.setZero();
    }
}

void frame::_assign_features_to_grid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS * FRAME_GRID_ROWS;

    int nReserve = 0.5f * feature_count / (nCells);

    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
    {
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
        {
            grid[i][j].reserve(nReserve);
            if (is_fisheye_stereo)
            {
                right_grid[i][j].reserve(nReserve);
            }
        }
    }

    for (int i = 0; i < feature_count; i++)
    {
        const auto &kp = (!is_fisheye_stereo || std::cmp_less(i, keypoints.size())) ? keypoints[i] : right_keypoints[i - static_cast<int>(keypoints.size())];

        int nGridPosX, nGridPosY;
        if (position_in_grid(kp.undistorted(), nGridPosX, nGridPosY))
        {
            if (!is_fisheye_stereo || std::cmp_less(i, keypoints.size()))
            {
                grid[nGridPosX][nGridPosY].push_back(i);
            }
            else
            {
                right_grid[nGridPosX][nGridPosY].push_back(i - static_cast<int>(keypoints.size()));
            }
        }
    }
}

bool frame::is_set() const { return _is_set; }

void frame::set_pose(const Sophus::SE3<float> &Tcw)
{
    _camera_from_world = Tcw;

    update_pose_matrices();
    _is_set   = true;
    _has_pose = true;
}

void frame::set_new_bias(const IMU::Bias &b)
{
    imu_bias = b;
    if (imu_preintegrated)
    {
        imu_preintegrated->set_new_bias(b);
    }
}

void frame::set_velocity(Eigen::Vector3f Vwb)
{
    _world_velocity = Vwb;
    _has_velocity   = true;
}

Eigen::Vector3f frame::get_velocity() const { return _world_velocity; }

void frame::set_imu_pose_velocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    _world_velocity = Vwb;
    _has_velocity   = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    _camera_from_world = imu_calibration.mTcb * Tbw;

    update_pose_matrices();
    _is_set   = true;
    _has_pose = true;
}

void frame::update_pose_matrices()
{
    Sophus::SE3<float> Twc       = _camera_from_world.inverse();
    _world_from_camera_rotation  = Twc.rotationMatrix();
    _camera_origin               = Twc.translation();
    _world_to_camera_rotation    = _camera_from_world.rotationMatrix();
    _world_to_camera_translation = _camera_from_world.translation();
}

Eigen::Matrix<float, 3, 1> frame::get_imu_position() const { return _world_from_camera_rotation * imu_calibration.mTcb.translation() + _camera_origin; }

Eigen::Matrix<float, 3, 3> frame::get_imu_rotation() { return _world_from_camera_rotation * imu_calibration.mTcb.rotationMatrix(); }

Sophus::SE3<float> frame::get_imu_pose() { return _camera_from_world.inverse() * imu_calibration.mTcb; }

Sophus::SE3f frame::get_relative_pose_trl() { return _right_to_left; }

Sophus::SE3f frame::get_relative_pose_tlr() { return _left_to_right; }

Eigen::Matrix3f frame::get_relative_pose_tlr_rotation() { return _left_to_right.rotationMatrix(); }

Eigen::Vector3f frame::get_relative_pose_tlr_translation() { return _left_to_right.translation(); }

bool frame::is_in_frustum(map_point *pMP, float viewingCosLimit, const camera_vector &cameras)
{
    if (!is_fisheye_stereo)
    {
        pMP->track_in_view      = false;
        pMP->track_projection_x = -1;
        pMP->track_projection_y = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float, 3, 1> P = pMP->get_world_position();

        // 3D in camera coordinates
        const Eigen::Matrix<float, 3, 1> Pc      = _world_to_camera_rotation * P + _world_to_camera_translation;
        const float                      Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ  = Pc(2);
        const float  invz = 1.0f / PcZ;
        if (PcZ < 0.0f)
        {
            return false;
        }

        const Eigen::Vector2f uv = cameras[0]->project(Pc);

        if (uv(0) < min_x || uv(0) > max_x)
        {
            return false;
        }
        if (uv(1) < min_y || uv(1) > max_y)
        {
            return false;
        }

        pMP->track_projection_x = uv(0);
        pMP->track_projection_y = uv(1);

        // Check distance is in the scale invariance region of the map_point
        const float           maxDistance = pMP->get_max_distance_invariance();
        const float           minDistance = pMP->get_min_distance_invariance();
        const Eigen::Vector3f PO          = P - _camera_origin;
        const float           dist        = PO.norm();

        if (dist < minDistance || dist > maxDistance)
        {
            return false;
        }

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->get_normal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
        {
            return false;
        }

        // Predict scale in the image
        const int nPredictedLevel = pMP->predict_scale(dist, this);

        // Data used by the tracking
        pMP->track_in_view            = true;
        pMP->track_projection_x       = uv(0);
        pMP->track_projection_x_right = uv(0) - baseline_fx * invz;

        pMP->track_depth = Pc_dist;

        pMP->track_projection_y = uv(1);
        pMP->track_scale_level  = nPredictedLevel;
        pMP->track_view_cos     = viewCos;

        return true;
    }
    else
    {
        pMP->track_in_view           = false;
        pMP->track_in_view_right     = false;
        pMP->track_scale_level       = -1;
        pMP->track_scale_level_right = -1;

        pMP->track_in_view       = is_in_frustum_checks(pMP, viewingCosLimit, *cameras[0]);
        pMP->track_in_view_right = is_in_frustum_checks(pMP, viewingCosLimit, *cameras[1], true);

        return pMP->track_in_view || pMP->track_in_view_right;
    }
}

bool frame::project_point_distort(map_point *pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->get_world_position();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc  = _world_to_camera_rotation * P + _world_to_camera_translation;
    const float          &PcX = Pc(0);
    const float          &PcY = Pc(1);
    const float          &PcZ = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        spdlog::debug("Negative depth: {}", PcZ);
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f / PcZ;
    u                = fx * PcX * invz + cx;
    v                = fy * PcY * invz + cy;

    if (u < min_x || u > max_x)
    {
        return false;
    }
    if (v < min_y || v > max_y)
    {
        return false;
    }

    float u_distort, v_distort;

    float x  = (u - cx) * invfx;
    float y  = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = distortion_coefficients.at<float>(0);
    float k2 = distortion_coefficients.at<float>(1);
    float p1 = distortion_coefficients.at<float>(2);
    float p2 = distortion_coefficients.at<float>(3);
    float k3 = 0;
    if (distortion_coefficients.total() == 5)
    {
        k3 = distortion_coefficients.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f frame::in_reference_coordinates(Eigen::Vector3f pCw) { return _world_to_camera_rotation * pCw + _world_to_camera_translation; }

vector<size_t> frame::get_features_in_area(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(feature_count);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0, (int)floor((x - min_x - factorX) * grid_element_width_inverse));
    if (nMinCellX >= FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - min_x + factorX) * grid_element_width_inverse));
    if (nMaxCellX < 0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0, (int)floor((y - min_y - factorY) * grid_element_height_inverse));
    if (nMinCellY >= FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - min_y + factorY) * grid_element_height_inverse));
    if (nMaxCellY < 0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? grid[ix][iy] : right_grid[ix][iy];
            if (vCell.empty())
            {
                continue;
            }

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const auto &kpUn = (!is_fisheye_stereo || !bRight) ? keypoints[vCell[j]] : right_keypoints[vCell[j]];
                if (bCheckLevels)
                {
                    if (kpUn.octave < minLevel)
                    {
                        continue;
                    }
                    if (maxLevel >= 0)
                    {
                        if (kpUn.octave > maxLevel)
                        {
                            continue;
                        }
                    }
                }

                const float distx = kpUn.pt_undistorted.x - x;
                const float disty = kpUn.pt_undistorted.y - y;

                if (fabs(distx) < factorX && fabs(disty) < factorY)
                {
                    vIndices.push_back(vCell[j]);
                }
            }
        }
    }

    return vIndices;
}

bool frame::position_in_grid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - min_x) * grid_element_width_inverse);
    posY = round((kp.pt.y - min_y) * grid_element_height_inverse);

    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
    {
        return false;
    }

    return true;
}

void frame::compute_bow()
{
    if (bow_vector.empty())
    {
        auto       vCurrentDesc      = descriptor_vector(keypoints);
        const auto right_descriptors = descriptor_vector(right_keypoints);
        vCurrentDesc.insert(vCurrentDesc.end(), right_descriptors.begin(), right_descriptors.end());
        place_vocabulary_ptr->transform(vCurrentDesc, bow_vector, feature_vector, 4);
    }
}

void frame::_undistort_keypoints()
{
    if (distortion_coefficients.at<float>(0) == 0.0)
    {
        for (auto &key : keypoints)
        {
            key.pt_undistorted = key.pt;
        }
        return;
    }

    // Fill matrix with points
    cv::Mat mat(feature_count, 2, CV_32F);

    for (int i = 0; i < feature_count; i++)
    {
        mat.at<float>(i, 0) = keypoints[i].pt.x;
        mat.at<float>(i, 1) = keypoints[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, calibration_matrix, distortion_coefficients, cv::Mat(), calibration_matrix);
    mat = mat.reshape(1);

    for (int i = 0; i < feature_count; i++)
    {
        keypoints[i].pt_undistorted.x = mat.at<float>(i, 0);
        keypoints[i].pt_undistorted.y = mat.at<float>(i, 1);
    }
}

void frame::_compute_image_bounds(const cv::Mat &imLeft)
{
    if (distortion_coefficients.at<float>(0) != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols;
        mat.at<float>(3, 1) = imLeft.rows;

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, calibration_matrix, distortion_coefficients, cv::Mat(), calibration_matrix);
        mat = mat.reshape(1);

        // Undistort corners
        min_x = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        max_x = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        min_y = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        max_y = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        min_x = 0.0f;
        max_x = imLeft.cols;
        min_y = 0.0f;
        max_y = imLeft.rows;
    }
}

void frame::compute_stereo_matches(const std::vector<cv::Mat> &left_pyramid, const std::vector<cv::Mat> &right_pyramid)
{
    right_coordinates = vector<float>(feature_count, -1.0f);
    depths            = vector<float>(feature_count, -1.0f);

    const int descriptor_threshold = (descriptor_matcher::high_distance_threshold + descriptor_matcher::low_distance_threshold) / 2;

    const int nRows = left_pyramid[0].rows;

    // Assign keypoints to row table
    vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

    for (int i = 0; i < nRows; i++)
    {
        vRowIndices[i].reserve(200);
    }

    const int Nr = right_keypoints.size();

    for (int iR = 0; iR < Nr; iR++)
    {
        const cv::KeyPoint &kp   = right_keypoints[iR];
        const float        &kpY  = kp.pt.y;
        const float         r    = 2.0f * scale_factors[right_keypoints[iR].octave];
        const int           maxr = ceil(kpY + r);
        const int           minr = floor(kpY - r);

        for (int yi = minr; yi <= maxr; yi++)
        {
            vRowIndices[yi].push_back(iR);
        }
    }

    // Set limits for search
    const float minZ = baseline;
    const float minD = 0;
    const float maxD = baseline_fx / minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(feature_count);

    for (int iL = 0; iL < feature_count; iL++)
    {
        const cv::KeyPoint &kpL    = keypoints[iL];
        const int          &levelL = kpL.octave;
        const float        &vL     = kpL.pt.y;
        const float        &uL     = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if (vCandidates.empty())
        {
            continue;
        }

        const float minU = uL - maxD;
        const float maxU = uL - minD;

        if (maxU < 0)
        {
            continue;
        }

        float  bestDist = descriptor_matcher::high_distance_threshold;
        size_t bestIdxR = 0;

        const cv::Mat &dL = keypoints[iL].descriptor;

        // Compare descriptor to right keypoints
        const auto ranked_candidates = descriptor_matcher::rank_candidates(dL, right_keypoints, vCandidates);
        for (const auto iR : ranked_candidates)
        {
            const cv::KeyPoint &kpR = right_keypoints[iR];

            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
            {
                continue;
            }

            const float &uR = kpR.pt.x;

            if (uR >= minU && uR <= maxU)
            {
                const cv::Mat &dR   = right_keypoints[iR].descriptor;
                const float    dist = descriptor_matcher::descriptor_distance(dL, dR);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if (bestDist < descriptor_threshold)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0         = right_keypoints[bestIdxR].pt.x;
            const float scaleFactor = inverse_scale_factors[kpL.octave];
            const float scaleduL    = round(kpL.pt.x * scaleFactor);
            const float scaledvL    = round(kpL.pt.y * scaleFactor);
            const float scaleduR0   = round(uR0 * scaleFactor);

            // sliding window search
            const int w  = 5;
            cv::Mat   IL = left_pyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);

            float         bestDist = INT_MAX;
            int           bestincR = 0;
            const int     L        = 5;
            vector<float> vDists;
            vDists.resize(2 * L + 1);

            const float iniu = scaleduR0 + L - w;
            const float endu = scaleduR0 + L + w + 1;
            if (iniu < 0 || endu >= right_pyramid[kpL.octave].cols)
            {
                continue;
            }

            for (int incR = -L; incR <= +L; incR++)
            {
                cv::Mat IR = right_pyramid[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

                float dist = cv::norm(IL, IR, cv::NORM_L1);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestincR = incR;
                }

                vDists[L + incR] = dist;
            }

            if (bestincR == -L || bestincR == L)
            {
                continue;
            }

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];

            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            if (deltaR < -1 || deltaR > 1)
            {
                continue;
            }

            // Re-scaled coordinate
            float bestuR = scale_factors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

            float disparity = (uL - bestuR);

            if (disparity >= minD && disparity < maxD)
            {
                if (disparity <= 0)
                {
                    disparity = 0.01;
                    bestuR    = uL - 0.01;
                }
                depths[iL]            = baseline_fx / disparity;
                right_coordinates[iL] = bestuR;
                vDistIdx.push_back(pair<int, int>(bestDist, iL));
            }
        }
    }

    sort(vDistIdx.begin(), vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = vDistIdx.size() - 1; i >= 0; i--)
    {
        if (vDistIdx[i].first < thDist)
        {
            break;
        }
        else
        {
            right_coordinates[vDistIdx[i].second] = -1;
            depths[vDistIdx[i].second]            = -1;
        }
    }
}

void frame::compute_stereo_from_rgbd(const cv::Mat &imDepth)
{
    right_coordinates = vector<float>(feature_count, -1);
    depths            = vector<float>(feature_count, -1);

    for (int i = 0; i < feature_count; i++)
    {
        const auto &kp = keypoints[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v, u);

        if (d > 0)
        {
            depths[i]            = d;
            right_coordinates[i] = kp.pt_undistorted.x - baseline_fx / d;
        }
    }
}

bool frame::unproject_stereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = depths[i];
    if (z > 0)
    {
        const float     u = keypoints[i].pt_undistorted.x;
        const float     v = keypoints[i].pt_undistorted.y;
        const float     x = (u - cx) * z * invfx;
        const float     y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = _world_from_camera_rotation * x3Dc + _camera_origin;
        return true;
    }
    else
    {
        return false;
    }
}

bool frame::imu_is_preintegrated() const
{
    unique_lock<std::mutex> lock(*_imu_mutex);
    return _imu_preintegrated;
}

void frame::set_integrated()
{
    unique_lock<std::mutex> lock(*_imu_mutex);
    _imu_preintegrated = true;
}

void frame::compute_stereo_fisheye_matches(const camera_vector &cameras)
{
    cv::Mat stereoDescLeft  = descriptor_matrix(keypoints, mono_left);
    cv::Mat stereoDescRight = descriptor_matrix(right_keypoints, mono_right);

    left_to_right_matches = vector<int>(keypoints.size(), -1);
    right_to_left_matches = vector<int>(right_keypoints.size(), -1);
    depths                = vector<float>(keypoints.size(), -1.0f);
    right_coordinates     = vector<float>(keypoints.size(), -1);
    stereo_3d_points      = vector<Eigen::Vector3f>(keypoints.size());
    close_map_point_count = 0;

    // Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    descriptor_matcher::knn_match(stereoDescLeft, stereoDescRight, matches, 2);

    cv::parallel_for_(
        cv::Range(0, static_cast<int>(matches.size())),
        [&](const cv::Range &range)
        {
            for (int index = range.start; index < range.end; ++index)
            {
                const auto &neighbors = matches[index];
                if (neighbors.size() < 2 || !(neighbors[0].distance < neighbors[1].distance * 0.7))
                {
                    continue;
                }
                const auto      left_index  = neighbors[0].queryIdx + mono_left;
                const auto      right_index = neighbors[0].trainIdx + mono_right;
                const auto      sigma_left  = level_sigma_squared[keypoints[left_index].octave];
                const auto      sigma_right = level_sigma_squared[right_keypoints[right_index].octave];
                Eigen::Vector3f point;
                const auto      depth =
                    static_cast<KannalaBrandt8 &>(*cameras[0])
                        .TriangulateMatches(
                            *cameras[1], keypoints[left_index], right_keypoints[right_index], _left_to_right_rotation, _left_to_right_translation, sigma_left, sigma_right, point);
                if (depth > 0.0001f)
                {
                    left_to_right_matches[left_index] = right_index;
                    stereo_3d_points[left_index]      = point;
                    depths[left_index]                = depth;
                }
            }
        },
        matches.size() >= 128 ? 4 : 1);
    for (const auto &neighbors : matches)
    {
        if (!neighbors.empty())
        {
            const auto left_index  = neighbors[0].queryIdx + mono_left;
            const auto right_index = left_to_right_matches[left_index];
            if (right_index >= 0)
            {
                right_to_left_matches[right_index] = left_index;
            }
        }
    }
}

bool frame::is_in_frustum_checks(map_point *pMP, float viewingCosLimit, geometric_camera &camera, bool bRight)
{
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->get_world_position();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if (bRight)
    {
        Eigen::Matrix3f Rrl = _right_to_left.rotationMatrix();
        Eigen::Vector3f trl = _right_to_left.translation();
        mR                  = Rrl * _world_to_camera_rotation;
        mt                  = Rrl * _world_to_camera_translation + trl;
        twc                 = _world_from_camera_rotation * _left_to_right.translation() + _camera_origin;
    }
    else
    {
        mR  = _world_to_camera_rotation;
        mt  = _world_to_camera_translation;
        twc = _camera_origin;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc      = mR * P + mt;
    const float     Pc_dist = Pc.norm();
    const float    &PcZ     = Pc(2);

    // Check positive depth
    if (PcZ < 0.0f)
    {
        return false;
    }

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    uv = camera.project(Pc);

    if (uv(0) < min_x || uv(0) > max_x)
    {
        return false;
    }
    if (uv(1) < min_y || uv(1) > max_y)
    {
        return false;
    }

    // Check distance is in the scale invariance region of the map_point
    const float           maxDistance = pMP->get_max_distance_invariance();
    const float           minDistance = pMP->get_min_distance_invariance();
    const Eigen::Vector3f PO          = P - twc;
    const float           dist        = PO.norm();

    if (dist < minDistance || dist > maxDistance)
    {
        return false;
    }

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->get_normal();

    const float viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
    {
        return false;
    }

    // Predict scale in the image
    const int nPredictedLevel = pMP->predict_scale(dist, this);

    if (bRight)
    {
        pMP->track_projection_x_right = uv(0);
        pMP->track_projection_y_right = uv(1);
        pMP->track_scale_level_right  = nPredictedLevel;
        pMP->track_view_cos_right     = viewCos;
        pMP->track_depth_right        = Pc_dist;
    }
    else
    {
        pMP->track_projection_x = uv(0);
        pMP->track_projection_y = uv(1);
        pMP->track_scale_level  = nPredictedLevel;
        pMP->track_view_cos     = viewCos;
        pMP->track_depth        = Pc_dist;
    }

    return true;
}

Eigen::Vector3f frame::unproject_stereo_fisheye(const int &i) { return _world_from_camera_rotation * stereo_3d_points[i] + _camera_origin; }

} // namespace ORB_SLAM3
