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

#include "TwoViewReconstruction.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{
TwoViewReconstruction::TwoViewReconstruction(const Eigen::Matrix3f &k, float sigma, int iterations) : sigma_{sigma}, max_iterations_{iterations}
{
    if (!k.allFinite() || k(0, 0) <= 0 || k(1, 1) <= 0 || k.determinant() == 0 || !std::isfinite(sigma) || sigma <= 0 || iterations <= 0)
    {
        throw std::invalid_argument("Two-view reconstruction requires valid calibration, positive sigma and iterations");
    }
    cv::eigen2cv(k, calibration_);
    calibration_.convertTo(calibration_, CV_64F);
}

bool TwoViewReconstruction::Reconstruct(const std::vector<cv::KeyPoint> &keys1,
                                        const std::vector<cv::KeyPoint> &keys2,
                                        const std::vector<int>          &matches12,
                                        Sophus::SE3f                    &pose21,
                                        std::vector<cv::Point3f>        &points3d,
                                        std::vector<bool>               &triangulated)
{
    reference_size_ = keys1.size();
    matched_points1_.clear();
    matched_points2_.clear();
    reference_indices_.clear();
    if (matches12.size() > keys1.size())
    {
        spdlog::warn("Two-view reconstruction: match list exceeds reference keypoint count");
        return false;
    }
    for (std::size_t index = 0; index < matches12.size(); ++index)
    {
        const auto match = matches12[index];
        if (match < 0)
        {
            continue;
        }
        if (static_cast<std::size_t>(match) >= keys2.size())
        {
            spdlog::warn("Two-view reconstruction: invalid target keypoint index {}", match);
            return false;
        }
        const auto point1 = keys1[index].pt;
        const auto point2 = keys2[match].pt;
        if (!std::isfinite(point1.x) || !std::isfinite(point1.y) || !std::isfinite(point2.x) || !std::isfinite(point2.y))
        {
            spdlog::warn("Two-view reconstruction: non-finite matched keypoint");
            return false;
        }
        matched_points1_.push_back(point1);
        matched_points2_.push_back(point2);
        reference_indices_.push_back(index);
    }
    if (matched_points1_.size() < 8)
    {
        spdlog::debug("Two-view reconstruction: fewer than eight matches");
        return false;
    }

    auto homography_task =
        std::async(std::launch::async, [this] { return cv::findHomography(matched_points1_, matched_points2_, cv::RANSAC, std::sqrt(5.991) * sigma_, cv::noArray(), max_iterations_, 0.99); });
    const auto fundamental = cv::findFundamentalMat(matched_points1_, matched_points2_, cv::FM_RANSAC, std::sqrt(3.841) * sigma_, 0.99, max_iterations_, cv::noArray());
    const auto homography  = homography_task.get();

    std::vector<bool> homography_inliers(matched_points1_.size(), false);
    std::vector<bool> fundamental_inliers(matched_points1_.size(), false);
    const auto        homography_score  = score_homography(homography, homography_inliers);
    const auto        fundamental_score = score_fundamental(fundamental, fundamental_inliers);
    spdlog::debug("Two-view reconstruction: homography score {}, fundamental score {}", homography_score, fundamental_score);
    if (homography_score + fundamental_score <= 0)
    {
        spdlog::debug("Two-view reconstruction: no valid geometry model");
        return false;
    }
    const auto use_homography = homography_score / (homography_score + fundamental_score) > 0.50f;
    if (use_homography)
    {
        return reconstruct_model(homography, true, homography_inliers, pose21, points3d, triangulated) ||
               (fundamental_score > 0 && reconstruct_model(fundamental, false, fundamental_inliers, pose21, points3d, triangulated));
    }
    // A fundamental matrix can fit a plane perfectly without determining a unique essential matrix.
    return reconstruct_model(fundamental, false, fundamental_inliers, pose21, points3d, triangulated) ||
           (homography_score > 0 && reconstruct_model(homography, true, homography_inliers, pose21, points3d, triangulated));
}

bool TwoViewReconstruction::reconstruct_model(
    const cv::Mat &model, bool use_homography, const std::vector<bool> &inliers, Sophus::SE3f &pose21, std::vector<cv::Point3f> &points3d, std::vector<bool> &triangulated) const
{
    const auto           inlier_count = std::count(inliers.begin(), inliers.end(), true);
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    if (use_homography)
    {
        std::vector<cv::Mat> normals;
        cv::decomposeHomographyMat(model, calibration_, rotations, translations, normals);
    }
    else
    {
        const cv::Mat essential = calibration_.t() * model * calibration_;
        cv::Mat       rotation1;
        cv::Mat       rotation2;
        cv::Mat       translation;
        cv::decomposeEssentialMat(essential, rotation1, rotation2, translation);
        rotations                          = {rotation1, rotation2, rotation1, rotation2};
        const cv::Mat opposite_translation = -translation;
        translations                       = {translation, translation, opposite_translation, opposite_translation};
    }

    int                      best_good        = 0;
    int                      second_best_good = 0;
    float                    best_parallax    = 0;
    Sophus::SE3f             best_pose;
    std::vector<cv::Point3f> best_points;
    std::vector<bool>        best_triangulated;
    for (std::size_t index = 0; index < rotations.size(); ++index)
    {
        Eigen::Matrix3f rotation;
        Eigen::Vector3f translation;
        cv::cv2eigen(rotations[index], rotation);
        cv::cv2eigen(translations[index], translation);
        const auto baseline = translation.norm();
        if (!rotation.allFinite() || !translation.allFinite() || baseline <= 1e-6f)
        {
            continue;
        }
        translation /= baseline;
        std::vector<cv::Point3f> candidate_points;
        std::vector<bool>        candidate_triangulated;
        float                    parallax = 0;
        const auto               good     = check_motion(rotation, translation, inliers, candidate_points, candidate_triangulated, parallax);
        if (good > best_good)
        {
            second_best_good  = best_good;
            best_good         = good;
            best_parallax     = parallax;
            best_pose         = Sophus::SE3f(rotation, translation);
            best_points       = std::move(candidate_points);
            best_triangulated = std::move(candidate_triangulated);
        }
        else if (good > second_best_good)
        {
            second_best_good = good;
        }
    }

    // Retain the original model-specific ambiguity, support and parallax thresholds.
    const auto accepted = use_homography ? second_best_good < 0.75 * best_good && best_parallax >= 1.0f && best_good > 50 && best_good > 0.9 * inlier_count
                                         : second_best_good <= 0.7 * best_good && best_parallax > 1.0f && best_good >= std::max(50, static_cast<int>(0.9 * inlier_count));
    if (!accepted)
    {
        spdlog::debug(
            "Two-view reconstruction: {} rejected (best support {}, runner-up {}, parallax {})", use_homography ? "homography" : "fundamental", best_good, second_best_good, best_parallax);
        return false;
    }
    pose21       = best_pose;
    points3d     = std::move(best_points);
    triangulated = std::move(best_triangulated);
    return true;
}

float TwoViewReconstruction::score_homography(const cv::Mat &homography, std::vector<bool> &inliers) const
{
    if (homography.empty() || !cv::checkRange(homography))
    {
        return 0;
    }
    cv::Mat inverse;
    if (cv::invert(homography, inverse) == 0 || !cv::checkRange(inverse))
    {
        return 0;
    }
    std::vector<cv::Point2f> projected1;
    std::vector<cv::Point2f> projected2;
    cv::perspectiveTransform(matched_points2_, projected1, inverse);
    cv::perspectiveTransform(matched_points1_, projected2, homography);
    float      score    = 0;
    const auto variance = sigma_ * sigma_;
    for (std::size_t index = 0; index < inliers.size(); ++index)
    {
        const auto error1  = matched_points1_[index] - projected1[index];
        const auto error2  = matched_points2_[index] - projected2[index];
        const auto chi1    = error1.dot(error1) / variance;
        const auto chi2    = error2.dot(error2) / variance;
        const auto good1   = std::isfinite(chi1) && chi1 <= 5.991f;
        const auto good2   = std::isfinite(chi2) && chi2 <= 5.991f;
        score             += good1 ? 5.991f - chi1 : 0;
        score             += good2 ? 5.991f - chi2 : 0;
        inliers[index]     = good1 && good2;
    }
    return score;
}

float TwoViewReconstruction::score_fundamental(const cv::Mat &fundamental, std::vector<bool> &inliers) const
{
    if (fundamental.rows != 3 || fundamental.cols != 3 || !cv::checkRange(fundamental))
    {
        return 0;
    }
    std::vector<cv::Vec3f> lines1;
    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(matched_points2_, 2, fundamental, lines1);
    cv::computeCorrespondEpilines(matched_points1_, 1, fundamental, lines2);
    float      score    = 0;
    const auto variance = sigma_ * sigma_;
    for (std::size_t index = 0; index < inliers.size(); ++index)
    {
        const auto &point1     = matched_points1_[index];
        const auto &point2     = matched_points2_[index];
        const auto &line1      = lines1[index];
        const auto &line2      = lines2[index];
        const auto  distance1  = line1[0] * point1.x + line1[1] * point1.y + line1[2];
        const auto  distance2  = line2[0] * point2.x + line2[1] * point2.y + line2[2];
        const auto  chi1       = distance1 * distance1 / variance;
        const auto  chi2       = distance2 * distance2 / variance;
        const auto  good1      = std::isfinite(chi1) && line1[0] * line1[0] + line1[1] * line1[1] > 0 && chi1 <= 3.841f;
        const auto  good2      = std::isfinite(chi2) && line2[0] * line2[0] + line2[1] * line2[1] > 0 && chi2 <= 3.841f;
        score                 += good1 ? 5.991f - chi1 : 0;
        score                 += good2 ? 5.991f - chi2 : 0;
        inliers[index]         = good1 && good2;
    }
    return score;
}

int TwoViewReconstruction::check_motion(const Eigen::Matrix3f    &rotation,
                                        const Eigen::Vector3f    &translation,
                                        const std::vector<bool>  &inliers,
                                        std::vector<cv::Point3f> &points3d,
                                        std::vector<bool>        &triangulated,
                                        float                    &parallax) const
{
    points3d.assign(reference_size_, cv::Point3f{});
    triangulated.assign(reference_size_, false);
    parallax = 0;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    std::vector<std::size_t> indices;
    for (std::size_t index = 0; index < inliers.size(); ++index)
    {
        if (inliers[index])
        {
            points1.push_back(matched_points1_[index]);
            points2.push_back(matched_points2_[index]);
            indices.push_back(reference_indices_[index]);
        }
    }
    if (points1.empty())
    {
        return 0;
    }
    cv::Mat rotation_cv;
    cv::Mat translation_cv;
    cv::eigen2cv(rotation, rotation_cv);
    cv::eigen2cv(translation, translation_cv);
    rotation_cv.convertTo(rotation_cv, CV_64F);
    translation_cv.convertTo(translation_cv, CV_64F);
    cv::Mat projection1 = cv::Mat::zeros(3, 4, CV_64F);
    calibration_.copyTo(projection1.colRange(0, 3));
    cv::Mat extrinsic2 = cv::Mat::zeros(3, 4, CV_64F);
    rotation_cv.copyTo(extrinsic2.colRange(0, 3));
    translation_cv.copyTo(extrinsic2.col(3));
    const cv::Mat projection2 = calibration_ * extrinsic2;
    cv::Mat       homogeneous;
    cv::triangulatePoints(projection1, projection2, points1, points2, homogeneous);
    std::vector<cv::Point3f> candidate_points;
    const cv::Mat            homogeneous_rows = homogeneous.t();
    cv::convertPointsFromHomogeneous(homogeneous_rows, candidate_points);
    std::vector<cv::Point2f> projected1;
    std::vector<cv::Point2f> projected2;
    cv::perspectiveTransform(candidate_points, projected1, projection1);
    cv::perspectiveTransform(candidate_points, projected2, projection2);

    const Eigen::Vector3f center2   = -rotation.transpose() * translation;
    const auto            max_error = 4.0f * sigma_ * sigma_;
    std::vector<float>    cos_parallaxes;
    for (std::size_t index = 0; index < points1.size(); ++index)
    {
        const auto column = static_cast<int>(index);
        const auto weight = homogeneous.at<float>(3, column);
        if (!std::isfinite(weight) || weight == 0)
        {
            continue;
        }
        const auto           &candidate = candidate_points[index];
        const Eigen::Vector3f point(candidate.x, candidate.y, candidate.z);
        if (!point.allFinite())
        {
            continue;
        }
        const Eigen::Vector3f ray2         = point - center2;
        const auto            cos_parallax = point.dot(ray2) / (point.norm() * ray2.norm());
        const Eigen::Vector3f point2       = rotation * point + translation;
        if (!std::isfinite(cos_parallax) || point.z() == 0 || point2.z() == 0 || ((point.z() < 0 || point2.z() < 0) && cos_parallax < 0.99998f))
        {
            continue;
        }
        const auto error1         = projected1[index] - points1[index];
        const auto error2         = projected2[index] - points2[index];
        const auto squared_error1 = error1.dot(error1);
        const auto squared_error2 = error2.dot(error2);
        if (!std::isfinite(squared_error1) || !std::isfinite(squared_error2) || squared_error1 > max_error || squared_error2 > max_error)
        {
            continue;
        }
        cos_parallaxes.push_back(std::clamp(cos_parallax, -1.0f, 1.0f));
        points3d[indices[index]]     = cv::Point3f(point.x(), point.y(), point.z());
        triangulated[indices[index]] = cos_parallax < 0.99998f;
    }
    if (!cos_parallaxes.empty())
    {
        std::sort(cos_parallaxes.begin(), cos_parallaxes.end());
        const auto index = std::min(std::size_t{50}, cos_parallaxes.size() - 1);
        parallax         = static_cast<float>(std::acos(cos_parallaxes[index]) * 180 / CV_PI);
    }
    return static_cast<int>(cos_parallaxes.size());
}

} // namespace ORB_SLAM3
