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

#pragma once

#include "Frame.h"
#include "GeometricCamera.h"
#include "KannalaBrandt8.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "Pinhole.h"

#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace ORB_SLAM3
{
class KeyFrameDatabase;
class place_vocabulary;

class Atlas
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Atlas() = default;
    Atlas(int initKFid);
    ~Atlas();

    void CreateNewMap();
    void ChangeMap(Map *pMap);

    unsigned long int GetLastInitKFid();

    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);

    GeometricCamera               *AddCamera(GeometricCamera *pCam);
    std::vector<GeometricCamera *> GetAllCameras();

    void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);
    void InformNewBigChange();
    int  GetLastBigChangeIdx();

    long unsigned int MapPointsInMap();
    long unsigned     KeyFramesInMap();

    std::vector<KeyFrame *> GetAllKeyFrames();
    std::vector<MapPoint *> GetAllMapPoints();
    std::vector<MapPoint *> GetReferenceMapPoints();

    std::vector<Map *> GetAllMaps();

    int CountMaps();

    void clearMap();

    void clearAtlas();

    Map *GetCurrentMap();

    void SetMapBad(Map *pMap);

    bool isInertial();
    void SetInertialSensor();
    void SetImuInitialized();
    bool isImuInitialized();

    std::map<long unsigned int, KeyFrame *> GetAtlasKeyframes();

    long unsigned int GetNumLivedKF();

    long unsigned int GetNumLivedMP();

  protected:
    void CreateNewMapLocked();

    std::set<Map *> mspMaps;
    // Retain maps removed from the active set while SLAM objects may still reference them.
    std::set<Map *> mspOwnedMaps;
    Map            *mpCurrentMap{nullptr};

    // Camera lifetime is managed by the settings/serialization owner, not by Atlas.
    std::vector<GeometricCamera *> mvpCameras;

    unsigned long int mnLastInitKFidMap{0};

    std::mutex mMutexAtlas;

}; // class Atlas

} // namespace ORB_SLAM3
