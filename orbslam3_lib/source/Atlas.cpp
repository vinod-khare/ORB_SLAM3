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

#include "Atlas.h"

#include "GeometricCamera.h"
#include "KannalaBrandt8.h"
#include "Pinhole.h"

#include <algorithm>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace ORB_SLAM3
{

Atlas::Atlas(int initKFid) : mnLastInitKFidMap(initKFid) { CreateNewMap(); }

Atlas::~Atlas()
{
    for (Map *map : mspOwnedMaps)
    {
        delete map;
    }
}

void Atlas::CreateNewMap()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    CreateNewMapLocked();
}

void Atlas::CreateNewMapLocked()
{
    if (mpCurrentMap)
    {
        if (!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
        {
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid() + 1; // The init KF is the next of current maximum
        }

        mpCurrentMap->SetStoredMap();
    }

    mpCurrentMap = new Map(mnLastInitKFidMap);
    mspOwnedMaps.insert(mpCurrentMap);
    mpCurrentMap->SetCurrentMap();
    mspMaps.insert(mpCurrentMap);
    spdlog::info("Created map {}", mpCurrentMap->GetId());
}

void Atlas::ChangeMap(Map *pMap)
{
    if (!pMap)
    {
        throw std::invalid_argument("Cannot change to a null map");
    }

    std::lock_guard<std::mutex> lock(mMutexAtlas);
    if (!mspMaps.contains(pMap) || pMap->IsBad())
    {
        throw std::invalid_argument("Cannot change to a map that is not active in this atlas");
    }

    if (mpCurrentMap)
    {
        mpCurrentMap->SetStoredMap();
    }

    mpCurrentMap = pMap;
    mpCurrentMap->SetCurrentMap();
    spdlog::info("Changed to map {}", mpCurrentMap->GetId());
}

unsigned long int Atlas::GetLastInitKFid()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mnLastInitKFidMap;
}

void Atlas::AddKeyFrame(KeyFrame *pKF)
{
    Map *pMapKF = pKF->GetMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddMapPoint(MapPoint *pMP)
{
    Map *pMapMP = pMP->GetMap();
    pMapMP->AddMapPoint(pMP);
}

GeometricCamera *Atlas::AddCamera(GeometricCamera *pCam)
{
    if (!pCam)
    {
        throw std::invalid_argument("Cannot add a null camera to the atlas");
    }

    std::lock_guard<std::mutex> lock(mMutexAtlas);
    for (GeometricCamera *existing_camera : mvpCameras)
    {
        if (!existing_camera || pCam->GetType() != existing_camera->GetType())
        {
            continue;
        }

        if (pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            if (static_cast<Pinhole *>(existing_camera)->IsEqual(pCam))
            {
                return existing_camera;
            }
        }
        else if (pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            if (static_cast<KannalaBrandt8 *>(existing_camera)->IsEqual(pCam))
            {
                return existing_camera;
            }
        }
    }

    mvpCameras.push_back(pCam);
    return pCam;
}

std::vector<GeometricCamera *> Atlas::GetAllCameras()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mvpCameras;
}

void Atlas::SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs)
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mpCurrentMap->SetReferenceMapPoints(vpMPs);
}

void Atlas::InformNewBigChange()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mpCurrentMap->InformNewBigChange();
}

int Atlas::GetLastBigChangeIdx()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetLastBigChangeIdx();
}

long unsigned int Atlas::MapPointsInMap()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->MapPointsInMap();
}

long unsigned Atlas::KeyFramesInMap()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->KeyFramesInMap();
}

std::vector<KeyFrame *> Atlas::GetAllKeyFrames()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllKeyFrames();
}

std::vector<MapPoint *> Atlas::GetAllMapPoints()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetAllMapPoints();
}

std::vector<MapPoint *> Atlas::GetReferenceMapPoints()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->GetReferenceMapPoints();
}

std::vector<Map *> Atlas::GetAllMaps()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    std::vector<Map *>          vMaps(mspMaps.begin(), mspMaps.end());
    std::sort(vMaps.begin(), vMaps.end(), [](Map *left, Map *right) { return left->GetId() < right->GetId(); });
    return vMaps;
}

int Atlas::CountMaps()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return static_cast<int>(mspMaps.size());
}

void Atlas::clearMap()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    if (mpCurrentMap)
    {
        mpCurrentMap->clear();
    }
}

void Atlas::clearAtlas()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mspMaps.clear();
    mpCurrentMap      = nullptr;
    mnLastInitKFidMap = 0;
}

Map *Atlas::GetCurrentMap()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    if (!mpCurrentMap || mpCurrentMap->IsBad())
    {
        CreateNewMapLocked();
    }

    return mpCurrentMap;
}

void Atlas::SetMapBad(Map *pMap)
{
    if (!pMap)
    {
        throw std::invalid_argument("Cannot mark a null map as bad");
    }

    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mspOwnedMaps.insert(pMap);
    mspMaps.erase(pMap);
    pMap->SetBad();
}

bool Atlas::isInertial()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->IsInertial();
}

void Atlas::SetInertialSensor()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mpCurrentMap->SetInertialSensor();
}

void Atlas::SetImuInitialized()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    mpCurrentMap->SetImuInitialized();
}

bool Atlas::isImuInitialized()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    return mpCurrentMap->isImuInitialized();
}

long unsigned int Atlas::GetNumLivedKF()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    long unsigned int           num = 0;
    for (Map *pMap_i : mspMaps)
    {
        num += pMap_i->GetAllKeyFrames().size();
    }

    return num;
}

long unsigned int Atlas::GetNumLivedMP()
{
    std::lock_guard<std::mutex> lock(mMutexAtlas);
    long unsigned int           num = 0;
    for (Map *pMap_i : mspMaps)
    {
        num += pMap_i->GetAllMapPoints().size();
    }

    return num;
}

std::map<long unsigned int, KeyFrame *> Atlas::GetAtlasKeyframes()
{
    std::lock_guard<std::mutex>             lock(mMutexAtlas);
    std::map<long unsigned int, KeyFrame *> mpIdKFs;
    for (Map *pMap_i : mspMaps)
    {
        std::vector<KeyFrame *> vpKFs_Mi = pMap_i->GetAllKeyFrames();

        for (KeyFrame *pKF_j_Mi : vpKFs_Mi)
        {
            mpIdKFs[pKF_j_Mi->mnId] = pKF_j_Mi;
        }
    }

    return mpIdKFs;
}

} // namespace ORB_SLAM3
