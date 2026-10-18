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

/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "keypoint_extractor.h"

#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int    PATCH_SIZE      = 31;
const int    HALF_PATCH_SIZE = 15;
const int    EDGE_THRESHOLD  = 19;

static float intensity_centroid_angle(const Mat &image, Point2f pt, const vector<int> &u_max)
{
    int          m_01 = 0, m_10 = 0;

    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
    {
        m_10 += u * center[u];
    }

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d     = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10  += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

keypoint_extractor::keypoint_extractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST, keypoint_detector_type detector_type,
                                       keypoint_descriptor_type descriptor_type)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), iniThFAST(_iniThFAST), minThFAST(_minThFAST), _detector_type(detector_type), _descriptor_type(descriptor_type)
{
    _place_descriptor = cv::ORB::create(nfeatures, 1.2f, 1, 0);
    switch (_descriptor_type)
    {
    case keypoint_descriptor_type::orb:
        _descriptor = _place_descriptor;
        break;
    case keypoint_descriptor_type::sift:
        _descriptor = cv::SIFT::create();
        break;
    case keypoint_descriptor_type::brisk:
        _descriptor = cv::BRISK::create();
        break;
    }
    // Scale-space detectors are restricted to a single scale where possible,
    // since ORB-SLAM's own pyramid already provides the octaves.
    switch (_detector_type)
    {
    case keypoint_detector_type::fast:
    case keypoint_detector_type::agast:
        break;
    case keypoint_detector_type::gftt:
        _detector = cv::GFTTDetector::create(0, 0.01, 1.0, 3, false);
        break;
    case keypoint_detector_type::orb:
        _detector = cv::ORB::create(nfeatures, 1.2f, 1, EDGE_THRESHOLD, 0, 2, cv::ORB::HARRIS_SCORE, PATCH_SIZE, iniThFAST);
        break;
    case keypoint_detector_type::brisk:
        _detector = cv::BRISK::create(iniThFAST, 0);
        break;
    case keypoint_detector_type::sift:
        _detector = cv::SIFT::create();
        break;
    case keypoint_detector_type::akaze:
        _detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 1);
        break;
    }
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++)
    {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++)
    {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor                   = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int   sumFeatures              = 0;
    for (int level = 0; level < nlevels - 1; level++)
    {
        mnFeaturesPerLevel[level]  = cvRound(nDesiredFeaturesPerScale);
        sumFeatures               += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale  *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    // This is for orientation
    //  pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int          v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2  = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
    {
        umax[v] = cvRound(sqrt(hp2 - v * v));
    }

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
        {
            ++v0;
        }
        umax[v] = v0;
        ++v0;
    }
}

static void compute_orientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = intensity_centroid_angle(image, keypoint->pt, umax);
    }
}

void keypoint_distribution_node::divide_node(keypoint_distribution_node &n1, keypoint_distribution_node &n2, keypoint_distribution_node &n3, keypoint_distribution_node &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

    // Define boundaries of childs
    n1.UL           = UL;
    n1.UR           = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL           = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR           = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
            {
                n1.vKeys.push_back(kp);
            }
            else
            {
                n3.vKeys.push_back(kp);
            }
        }
        else if (kp.pt.y < n1.BR.y)
        {
            n2.vKeys.push_back(kp);
        }
        else
        {
            n4.vKeys.push_back(kp);
        }
    }

    if (n1.vKeys.size() == 1)
    {
        n1.bNoMore = true;
    }
    if (n2.vKeys.size() == 1)
    {
        n2.bNoMore = true;
    }
    if (n3.vKeys.size() == 1)
    {
        n3.bNoMore = true;
    }
    if (n4.vKeys.size() == 1)
    {
        n4.bNoMore = true;
    }
}

static bool compareNodes(pair<int, keypoint_distribution_node *> &e1, pair<int, keypoint_distribution_node *> &e2)
{
    if (e1.first < e2.first)
    {
        return true;
    }
    else if (e1.first > e2.first)
    {
        return false;
    }
    else
    {
        if (e1.second->UL.x < e2.second->UL.x)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

vector<cv::KeyPoint> keypoint_extractor::distribute_keypoints(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX, const int &maxX, const int &minY, const int &maxY, const int &N,
                                                              const int &level)
{
    // Compute how many initial nodes
    const int                            nIni = std::max(1, cvRound(static_cast<float>(maxX - minX) / (maxY - minY)));

    const float                          hX   = static_cast<float>(maxX - minX) / nIni;

    list<keypoint_distribution_node>     lNodes;

    vector<keypoint_distribution_node *> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++)
    {
        keypoint_distribution_node ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    list<keypoint_distribution_node>::iterator lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->vKeys.size() == 1)
        {
            lit->bNoMore = true;
            lit++;
        }
        else if (lit->vKeys.empty())
        {
            lit = lNodes.erase(lit);
        }
        else
        {
            lit++;
        }
    }

    bool                                            bFinish   = false;

    int                                             iteration = 0;

    vector<pair<int, keypoint_distribution_node *>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        int prevSize  = lNodes.size();

        lit           = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end())
        {
            if (lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                keypoint_distribution_node n1, n2, n3, n4;
                lit->divide_node(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        else if (((int)lNodes.size() + nToExpand * 3) > N)
        {

            while (!bFinish)
            {

                prevSize                                                                  = lNodes.size();

                vector<pair<int, keypoint_distribution_node *>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end(), compareNodes);
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    keypoint_distribution_node n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->divide_node(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                    {
                        break;
                    }
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                {
                    bFinish = true;
                }
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for (const keypoint_distribution_node &node : lNodes)
    {
        const auto bestKey = std::max_element(node.vKeys.begin(), node.vKeys.end(), [](const cv::KeyPoint &left, const cv::KeyPoint &right) { return left.response < right.response; });
        vResultKeys.push_back(*bestKey);
    }

    return vResultKeys;
}

void keypoint_extractor::detect_corners_in_cell(const cv::Mat &cell, std::vector<cv::KeyPoint> &keypoints, int threshold) const
{
    if (_detector_type == keypoint_detector_type::agast)
    {
        AGAST(cell, keypoints, threshold, true);
    }
    else
    {
        FAST(cell, keypoints, threshold, true);
    }
}

void keypoint_extractor::detect_keypoints(int level, int min_x, int max_x, int min_y, int max_y, std::vector<cv::KeyPoint> &keypoints)
{
    vector<cv::KeyPoint> detected;
    _detector->detect(mvImagePyramid[level], detected);

    // Keep points far enough from the border for the ORB descriptor, relative to (min_x, min_y).
    const float width  = static_cast<float>(max_x - min_x);
    const float height = static_cast<float>(max_y - min_y);
    keypoints.reserve(detected.size());
    for (cv::KeyPoint &kp : detected)
    {
        kp.pt.x -= min_x;
        kp.pt.y -= min_y;
        if (kp.pt.x >= 0.f && kp.pt.x < width && kp.pt.y >= 0.f && kp.pt.y < height)
        {
            keypoints.push_back(kp);
        }
    }
}

void keypoint_extractor::compute_keypoints(vector<vector<KeyPoint>> &allKeypoints, vector<vector<KeyPoint>> &display_keypoints)
{
    allKeypoints.resize(nlevels);
    display_keypoints.resize(nlevels);

    const float W = 35;

    if (_detector)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
            const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;
            if (maxBorderX <= minBorderX || maxBorderY <= minBorderY)
            {
                continue;
            }

            vector<cv::KeyPoint> vToDistributeKeys;
            detect_keypoints(level, minBorderX, maxBorderX, minBorderY, maxBorderY, vToDistributeKeys);

            vector<KeyPoint> &keypoints = allKeypoints[level];
            keypoints                   = distribute_keypoints(vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);
            for (KeyPoint &kp : keypoints)
            {
                kp.pt.x += minBorderX;
                kp.pt.y += minBorderY;
                display_keypoints[level].push_back(kp);
                kp.octave = level;
                kp.size   = static_cast<int>(PATCH_SIZE * mvScaleFactor[level]);
            }
            compute_orientation(mvImagePyramid[level], keypoints, umax);
        }
        return;
    }

    for (int level = 0; level < nlevels; ++level)
    {
        const int            minBorderX = EDGE_THRESHOLD - 3;
        const int            minBorderY = minBorderX;
        const int            maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int            maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width  = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        if (width <= 0 || height <= 0)
        {
            continue;
        }

        const int nCols = std::max(1, static_cast<int>(width / W));
        const int nRows = std::max(1, static_cast<int>(height / W));
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float       maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
            {
                continue;
            }
            if (maxY > maxBorderY)
            {
                maxY = maxBorderY;
            }

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float       maxX = iniX + wCell + 6;

                if (iniX >= maxBorderX - 6)
                {
                    continue;
                }

                if (maxX > maxBorderX)
                {
                    maxX = maxBorderX;
                }

                vector<cv::KeyPoint> vKeysCell;

                detect_corners_in_cell(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, iniThFAST);

                if (vKeysCell.empty())
                {
                    detect_corners_in_cell(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, minThFAST);
                }

                if (!vKeysCell.empty())
                {
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        vector<KeyPoint> &keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        keypoints                 = distribute_keypoints(vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps            = keypoints.size();
        for (int i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            display_keypoints[level].push_back(keypoints[i]);
            keypoints[i].octave = level;
            keypoints[i].size   = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level)
    {
        compute_orientation(mvImagePyramid[level], allKeypoints[level], umax);
    }
}

int keypoint_extractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints, OutputArray _descriptors, std::vector<int> &vLappingArea)
{
    display_keypoints_.clear();
    _keypoints.clear();
    _place_descriptors.release();
    _descriptors.release();
    if (_image.empty())
    {
        return -1;
    }

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    compute_pyramid(image);
    vector<vector<KeyPoint>> allKeypoints;
    vector<vector<KeyPoint>> display_keypoints;
    compute_keypoints(allKeypoints, display_keypoints);
    vector<cv::KeyPoint> output_keys;
    vector<cv::KeyPoint> output_display;
    Mat                  descriptors;
    Mat                  place_descriptors;
    for (int level = 0; level < nlevels; ++level)
    {
        auto local_keys = allKeypoints[level];
        if (local_keys.empty())
        {
            continue;
        }
        for (std::size_t index = 0; index < local_keys.size(); ++index)
        {
            local_keys[index].class_id = static_cast<int>(index);
            local_keys[index].octave   = 0;
            local_keys[index].size     = _descriptor_type == keypoint_descriptor_type::sift ? PATCH_SIZE / 6.0f : PATCH_SIZE;
        }
        Mat level_descriptors;
        _descriptor->compute(mvImagePyramid[level], local_keys, level_descriptors);
        auto place_keys = local_keys;
        Mat  level_place_descriptors;
        if (_descriptor_type == keypoint_descriptor_type::orb)
        {
            level_place_descriptors = level_descriptors;
        }
        else
        {
            for (auto &key : place_keys)
            {
                key.size = PATCH_SIZE;
            }
            _place_descriptor->compute(mvImagePyramid[level], place_keys, level_place_descriptors);
        }
        std::size_t descriptor_index = 0;
        for (std::size_t place_index = 0; place_index < place_keys.size(); ++place_index)
        {
            const int original_index = place_keys[place_index].class_id;
            while (descriptor_index < local_keys.size() && local_keys[descriptor_index].class_id != original_index)
            {
                ++descriptor_index;
            }
            CV_Assert(descriptor_index < local_keys.size());
            auto key          = allKeypoints[level][original_index];
            auto display_key  = display_keypoints[level][original_index];
            key.pt           *= mvScaleFactor[level];
            display_key.pt   *= mvScaleFactor[level];
            display_key.size *= mvScaleFactor[level];
            output_keys.push_back(key);
            output_display.push_back(display_key);
            descriptors.push_back(level_descriptors.row(static_cast<int>(descriptor_index)));
            place_descriptors.push_back(level_place_descriptors.row(static_cast<int>(place_index)));
        }
    }
    const auto count = output_keys.size();
    _keypoints.resize(count);
    display_keypoints_.resize(count);
    if (count == 0)
    {
        return 0;
    }
    _descriptors.create(static_cast<int>(count), descriptors.cols, descriptors.type());
    Mat output_descriptors = _descriptors.getMat();
    _place_descriptors.create(static_cast<int>(count), 32, CV_8U);
    int mono_index   = 0;
    int stereo_index = static_cast<int>(count) - 1;
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto &key            = output_keys[index];
        const int   target         = key.pt.x >= vLappingArea[0] && key.pt.x <= vLappingArea[1] ? stereo_index-- : mono_index++;
        _keypoints[target]         = key;
        display_keypoints_[target] = output_display[index];
        descriptors.row(static_cast<int>(index)).copyTo(output_descriptors.row(target));
        place_descriptors.row(static_cast<int>(index)).copyTo(_place_descriptors.row(target));
    }
    return mono_index;
}

void keypoint_extractor::compute_pyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size  sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        Size  wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        Mat   temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if (level != 0)
        {
            resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101 + BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101);
        }
    }
}

} // namespace ORB_SLAM3
