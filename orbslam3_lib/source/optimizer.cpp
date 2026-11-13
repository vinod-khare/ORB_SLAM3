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

#include "optimizer.h"

#include "converter.h"
#include "g2o_compat.h"
#include "g2o_types.h"
#include "optimizable_types.h"

#include <complex>
#include <memory>
#include <mutex>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <spdlog/spdlog.h>
#include <unsupported/Eigen/MatrixFunctions>

namespace ORB_SLAM3
{
template <typename BlockSolverType> BlockSolverType *MakeBlockSolver(typename BlockSolverType::LinearSolverType *linearSolver)
{
    return new BlockSolverType(std::unique_ptr<typename BlockSolverType::LinearSolverType>(linearSolver));
}

template <typename AlgorithmType> AlgorithmType *MakeOptimizationAlgorithm(g2o::Solver *solver) { return new AlgorithmType(std::unique_ptr<g2o::Solver>(solver)); }

bool sortByVal(const pair<map_point *, int> &a, const pair<map_point *, int> &b) { return (a.second < b.second); }

void optimizer::global_bundle_adjustment(camera_view cameras, map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<key_frame *> vpKFs = pMap->get_all_key_frames();
    vector<map_point *> vpMP  = pMap->get_all_map_points();
    bundle_adjustment(cameras, vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void optimizer::bundle_adjustment(
    camera_view cameras, const vector<key_frame *> &vpKFs, const vector<map_point *> &vpMP, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    map *pMap = vpKFs[0]->get_map();

    g2o::SparseOptimizer                    optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = MakeBlockSolver<g2o::BlockSolver_6_3>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
    {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    long unsigned int maxKFid = 0;

    const int nExpectedSize = (vpKFs.size()) * vpMP.size();

    vector<ORB_SLAM3::edge_se3_project_xyz *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::edge_se3_project_xyz_to_body *> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    // Set key_frame vertices

    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKF = vpKFs[i];
        if (pKF->is_bad())
        {
            continue;
        }
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float>    Tcw  = pKF->get_pose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKF->id);
        vSE3->setFixed(pKF->id == pMap->get_initial_key_frame_id());
        optimizer.addVertex(vSE3);
        if (pKF->id > maxKFid)
        {
            maxKFid = pKF->id;
        }
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set map_point vertices
    for (size_t i = 0; std::cmp_less(i, vpMP.size()); i++)
    {
        map_point *pMP = vpMP[i];
        if (pMP->is_bad())
        {
            continue;
        }
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_world_position().cast<double>());
        const int id = pMP->id + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();

        int nEdges = 0;
        // SET EDGES
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            key_frame *pKF = mit->first;
            if (pKF->is_bad() || pKF->id > maxKFid)
            {
                continue;
            }
            if (optimizer.vertex(id) == NULL || optimizer.vertex(pKF->id) == NULL)
            {
                continue;
            }
            nEdges++;

            const int leftIndex = get<0>(mit->second);

            if (leftIndex != -1 && pKF->right_coordinates[get<0>(mit->second)] < 0)
            {
                const auto &kpUn = pKF->keypoints[leftIndex];

                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                ORB_SLAM3::edge_se3_project_xyz *e = new ORB_SLAM3::edge_se3_project_xyz(*cameras[0]);

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->id)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->inverse_level_sigma_squared[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else if (leftIndex != -1 && pKF->right_coordinates[leftIndex] >= 0) // Stereo observation
            {
                const auto &kpUn = pKF->keypoints[leftIndex];

                Eigen::Matrix<double, 3, 1> obs;
                const float                 kp_ur = pKF->right_coordinates[get<0>(mit->second)];
                obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->id)));
                e->setMeasurement(obs);
                const float    &invSigma2 = pKF->inverse_level_sigma_squared[kpUn.octave];
                Eigen::Matrix3d Info      = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = cameras[0]->get_parameter(0);
                e->fy = cameras[0]->get_parameter(1);
                e->cx = cameras[0]->get_parameter(2);
                e->cy = cameras[0]->get_parameter(3);
                e->bf = cameras[0]->baseline_fx();

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMP);
            }

            if (pKF->is_fisheye_stereo)
            {
                int rightIndex = get<1>(mit->second);

                if (rightIndex != -1 && std::cmp_less(rightIndex, pKF->right_keypoints.size()))
                {
                    rightIndex -= static_cast<int>(pKF->keypoints.size());

                    Eigen::Matrix<double, 2, 1> obs;
                    cv::KeyPoint                kp = pKF->right_keypoints[rightIndex];
                    obs << kp.pt.x, kp.pt.y;

                    ORB_SLAM3::edge_se3_project_xyz_to_body *e = new ORB_SLAM3::edge_se3_project_xyz_to_body(*cameras[1]);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->id)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->inverse_level_sigma_squared[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);

                    Sophus::SE3f Trl   = pKF->get_relative_pose_trl(*cameras[0]);
                    e->right_from_left = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                    optimizer.addEdge(e);
                    vpEdgesBody.push_back(e);
                    vpEdgeKFBody.push_back(pKF);
                    vpMapPointEdgeBody.push_back(pMP);
                }
            }
        }

        if (nEdges == 0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    spdlog::debug("BA: End of the optimization");

    // Recover optimized data
    // Keyframes
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKF = vpKFs[i];
        if (pKF->is_bad())
        {
            continue;
        }
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->id));

        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == pMap->get_origin_key_frame()->id)
        {
            pKF->set_pose(Sophus::SE3f(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>()));
        }
        else
        {
            pKF->camera_pose_global_ba  = Sophus::SE3d(SE3quat.rotation(), SE3quat.translation()).cast<float>();
            pKF->global_ba_key_frame_id = nLoopKF;

            Sophus::SE3f    mTwc        = pKF->get_pose_inverse();
            Sophus::SE3f    mTcGBA_c    = pKF->camera_pose_global_ba * mTwc;
            Eigen::Vector3f vector_dist = mTcGBA_c.translation();
            double          dist        = vector_dist.norm();
            if (dist > 1)
            {
                int                 numMonoBadPoints = 0, numMonoOptPoints = 0;
                int                 numStereoBadPoints = 0, numStereoOptPoints = 0;
                vector<map_point *> vpMonoMPsOpt, vpStereoMPsOpt;

                for (size_t i2 = 0, iend = vpEdgesMono.size(); i2 < iend; i2++)
                {
                    ORB_SLAM3::edge_se3_project_xyz *e       = vpEdgesMono[i2];
                    map_point                       *pMP     = vpMapPointEdgeMono[i2];
                    key_frame                       *pKFedge = vpEdgeKFMono[i2];

                    if (pKF != pKFedge)
                    {
                        continue;
                    }

                    if (pMP->is_bad())
                    {
                        continue;
                    }

                    if (e->chi2() > 5.991 || !e->is_depth_positive())
                    {
                        numMonoBadPoints++;
                    }
                    else
                    {
                        numMonoOptPoints++;
                        vpMonoMPsOpt.push_back(pMP);
                    }
                }

                for (size_t i2 = 0, iend = vpEdgesStereo.size(); i2 < iend; i2++)
                {
                    g2o::EdgeStereoSE3ProjectXYZ *e       = vpEdgesStereo[i2];
                    map_point                    *pMP     = vpMapPointEdgeStereo[i2];
                    key_frame                    *pKFedge = vpEdgeKFMono[i2];

                    if (pKF != pKFedge)
                    {
                        continue;
                    }

                    if (pMP->is_bad())
                    {
                        continue;
                    }

                    if (e->chi2() > 7.815 || !e->isDepthPositive())
                    {
                        numStereoBadPoints++;
                    }
                    else
                    {
                        numStereoOptPoints++;
                        vpStereoMPsOpt.push_back(pMP);
                    }
                }
            }
        }
    }

    // Points
    for (size_t i = 0; std::cmp_less(i, vpMP.size()); i++)
    {
        if (vbNotIncludedMP[i])
        {
            continue;
        }

        map_point *pMP = vpMP[i];

        if (pMP->is_bad())
        {
            continue;
        }
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->id + maxKFid + 1));

        if (nLoopKF == pMap->get_origin_key_frame()->id)
        {
            pMP->set_world_position(vPoint->estimate().cast<float>());
            pMP->update_normal_and_depth(*cameras[0]);
        }
        else
        {
            pMP->global_ba_position     = vPoint->estimate().cast<float>();
            pMP->global_ba_key_frame_id = nLoopKF;
        }
    }
}

void optimizer::full_inertial_ba(camera_view             cameras,
                                 map                    *pMap,
                                 int                     its,
                                 const bool              bFixLocal,
                                 const long unsigned int nLoopId,
                                 bool                   *pbStopFlag,
                                 bool                    bInit,
                                 float                   priorG,
                                 float                   priorA,
                                 Eigen::VectorXd        *vSingVal,
                                 bool                   *bHess)
{
    long unsigned int         maxKFid = pMap->get_max_key_frame_id();
    const vector<key_frame *> vpKFs   = pMap->get_all_key_frames();
    const vector<map_point *> vpMPs   = pMap->get_all_map_points();

    // Setup optimizer
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
    {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    int nNonFixed = 0;

    // Set key_frame vertices
    key_frame *pIncKF = nullptr;
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }
        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        pIncKF      = pKFi;
        bool bFixed = false;
        if (bFixLocal)
        {
            bFixed = (pKFi->local_ba_key_frame_id >= (maxKFid - 1)) || (pKFi->fixed_ba_key_frame_id >= (maxKFid - 1));
            if (!bFixed)
            {
                nNonFixed++;
            }
            VP->setFixed(bFixed);
        }
        optimizer.addVertex(VP);

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = new vertex_velocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->id) + 1);
            VV->setFixed(bFixed);
            optimizer.addVertex(VV);
            if (!bInit)
            {
                vertex_gyro_bias *VG = new vertex_gyro_bias(pKFi);
                VG->setId(maxKFid + 3 * (pKFi->id) + 2);
                VG->setFixed(bFixed);
                optimizer.addVertex(VG);
                vertex_acc_bias *VA = new vertex_acc_bias(pKFi);
                VA->setId(maxKFid + 3 * (pKFi->id) + 3);
                VA->setFixed(bFixed);
                optimizer.addVertex(VA);
            }
        }
    }

    if (bInit && !pIncKF)
    {
        return;
    }

    if (bInit)
    {
        vertex_gyro_bias *VG = new vertex_gyro_bias(pIncKF);
        VG->setId(4 * maxKFid + 2);
        VG->setFixed(false);
        optimizer.addVertex(VG);
        vertex_acc_bias *VA = new vertex_acc_bias(pIncKF);
        VA->setId(4 * maxKFid + 3);
        VA->setFixed(false);
        optimizer.addVertex(VA);
    }

    if (bFixLocal)
    {
        if (nNonFixed < 3)
        {
            return;
        }
    }

    // IMU links
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];

        if (!pKFi->previous_key_frame)
        {
            spdlog::debug("NOT INERTIAL LINK TO PREVIOUS FRAME!");
            continue;
        }

        if (pKFi->previous_key_frame && pKFi->id <= maxKFid)
        {
            if (pKFi->is_bad() || pKFi->previous_key_frame->id > maxKFid)
            {
                continue;
            }
            if (pKFi->has_imu && pKFi->previous_key_frame->has_imu)
            {
                pKFi->imu_preintegrated->set_new_bias(pKFi->previous_key_frame->get_imu_bias());
                g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->previous_key_frame->id);
                g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 1);

                g2o::HyperGraph::Vertex *VG1;
                g2o::HyperGraph::Vertex *VA1;
                g2o::HyperGraph::Vertex *VG2 = nullptr;
                g2o::HyperGraph::Vertex *VA2 = nullptr;
                if (!bInit)
                {
                    VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 2);
                    VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 3);
                    VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2);
                    VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3);
                }
                else
                {
                    VG1 = optimizer.vertex(4 * maxKFid + 2);
                    VA1 = optimizer.vertex(4 * maxKFid + 3);
                }

                g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->id);
                g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1);

                if (!bInit)
                {
                    if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
                    {
                        spdlog::error("Unable to build inertial edge: required graph vertex is missing");
                        continue;
                    }
                }
                else
                {
                    if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2)
                    {
                        spdlog::error("Unable to build inertial edge: required graph vertex is missing");
                        continue;
                    }
                }

                edge_inertial *ei = new edge_inertial(pKFi->imu_preintegrated);
                ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
                ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
                ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
                ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
                ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
                ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

                g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
                ei->setRobustKernel(rki);
                rki->setDelta(sqrt(16.92));

                optimizer.addEdge(ei);

                if (!bInit)
                {
                    edge_gyro_rw *egr = new edge_gyro_rw();
                    egr->setVertex(0, VG1);
                    egr->setVertex(1, VG2);
                    Eigen::Matrix3d InfoG = pKFi->imu_preintegrated->covariance.block<3, 3>(9, 9).cast<double>().inverse();
                    egr->setInformation(InfoG);
                    egr->computeError();
                    optimizer.addEdge(egr);

                    edge_acc_rw *ear = new edge_acc_rw();
                    ear->setVertex(0, VA1);
                    ear->setVertex(1, VA2);
                    Eigen::Matrix3d InfoA = pKFi->imu_preintegrated->covariance.block<3, 3>(12, 12).cast<double>().inverse();
                    ear->setInformation(InfoA);
                    ear->computeError();
                    optimizer.addEdge(ear);
                }
            }
            else
            {
                spdlog::warn("Keyframe {} or its predecessor {} has no IMU preintegration", pKFi->id, pKFi->previous_key_frame->id);
            }
        }
    }

    if (bInit)
    {
        g2o::HyperGraph::Vertex *VG = optimizer.vertex(4 * maxKFid + 2);
        g2o::HyperGraph::Vertex *VA = optimizer.vertex(4 * maxKFid + 3);

        // Add prior to comon biases
        Eigen::Vector3f bprior;
        bprior.setZero();

        edge_prior_acc *epa = new edge_prior_acc(bprior);
        epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
        double infoPriorA = priorA; //
        epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epa);

        edge_prior_gyro *epg = new edge_prior_gyro(bprior);
        epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
        double infoPriorG = priorG; //
        epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
        optimizer.addEdge(epg);
    }

    const float thHuberMono   = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid * 5;

    vector<bool> vbNotIncludedMP(vpMPs.size(), false);

    for (size_t i = 0; std::cmp_less(i, vpMPs.size()); i++)
    {
        map_point              *pMP    = vpMPs[i];
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_world_position().cast<double>());
        unsigned long id = pMP->id + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();

        bool bAllFixed = true;

        // Set edges
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (pKFi->id > maxKFid)
            {
                continue;
            }

            if (!pKFi->is_bad())
            {
                const int leftIndex = get<0>(mit->second);

                if (leftIndex != -1 && pKFi->right_coordinates[get<0>(mit->second)] < 0) // Monocular observation
                {
                    const auto                 &kpUn = pKFi->keypoints[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                    edge_mono *e = new edge_mono(0);

                    g2o::OptimizableGraph::Vertex *VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id));
                    if (bAllFixed)
                    {
                        if (!VP->fixed())
                        {
                            bAllFixed = false;
                        }
                    }

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
                else if (leftIndex != -1 && pKFi->right_coordinates[leftIndex] >= 0) // stereo observation
                {
                    const auto                 &kpUn  = pKFi->keypoints[leftIndex];
                    const float                 kp_ur = pKFi->right_coordinates[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                    edge_stereo *e = new edge_stereo(0);

                    g2o::OptimizableGraph::Vertex *VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id));
                    if (bAllFixed)
                    {
                        if (!VP->fixed())
                        {
                            bAllFixed = false;
                        }
                    }

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, VP);
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];

                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }

                if (pKFi->is_fisheye_stereo)
                { // Monocular right observation
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1 && std::cmp_less(rightIndex, pKFi->right_keypoints.size()))
                    {
                        rightIndex -= static_cast<int>(pKFi->keypoints.size());

                        Eigen::Matrix<double, 2, 1> obs;
                        const auto                 &kpUn = pKFi->right_keypoints[rightIndex];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        edge_mono *e = new edge_mono(1);

                        g2o::OptimizableGraph::Vertex *VP = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id));
                        if (bAllFixed)
                        {
                            if (!VP->fixed())
                            {
                                bAllFixed = false;
                            }
                        }

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, VP);
                        e->setMeasurement(obs);
                        const float invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                    }
                }
            }
        }

        if (bAllFixed)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
    }

    if (pbStopFlag)
    {
        if (*pbStopFlag)
        {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover optimized data
    // Keyframes
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }
        vertex_pose *VP = static_cast<vertex_pose *>(optimizer.vertex(pKFi->id));
        if (nLoopId == 0)
        {
            Sophus::SE3f Tcw(VP->estimate().rotation_camera_world[0].cast<float>(), VP->estimate().translation_camera_world[0].cast<float>());
            pKFi->set_pose(Tcw);
        }
        else
        {
            pKFi->camera_pose_global_ba  = Sophus::SE3f(VP->estimate().rotation_camera_world[0].cast<float>(), VP->estimate().translation_camera_world[0].cast<float>());
            pKFi->global_ba_key_frame_id = nLoopId;
        }
        if (pKFi->has_imu)
        {
            vertex_velocity *VV = static_cast<vertex_velocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1));
            if (nLoopId == 0)
            {
                pKFi->set_velocity(VV->estimate().cast<float>());
            }
            else
            {
                pKFi->velocity_world_global_ba = VV->estimate().cast<float>();
            }

            vertex_gyro_bias *VG;
            vertex_acc_bias  *VA;
            if (!bInit)
            {
                VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2));
                VA = static_cast<vertex_acc_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3));
            }
            else
            {
                VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(4 * maxKFid + 2));
                VA = static_cast<vertex_acc_bias *>(optimizer.vertex(4 * maxKFid + 3));
            }

            vector_6d vb;
            vb << VG->estimate(), VA->estimate();
            imu::bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
            if (nLoopId == 0)
            {
                pKFi->set_new_bias(b);
            }
            else
            {
                pKFi->bias_global_ba = b;
            }
        }
    }

    // Points
    for (size_t i = 0; std::cmp_less(i, vpMPs.size()); i++)
    {
        if (vbNotIncludedMP[i])
        {
            continue;
        }

        map_point              *pMP    = vpMPs[i];
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->id + iniMPid + 1));

        if (nLoopId == 0)
        {
            pMP->set_world_position(vPoint->estimate().cast<float>());
            pMP->update_normal_and_depth(*cameras[0]);
        }
        else
        {
            pMP->global_ba_position     = vPoint->estimate().cast<float>();
            pMP->global_ba_key_frame_id = nLoopId;
        }
    }

    pMap->increase_change_index();
}

int optimizer::pose_optimization(frame *pFrame, const camera_vector &cameras)
{
    g2o::SparseOptimizer                    optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = MakeBlockSolver<g2o::BlockSolver_6_3>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set Frame vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3<float>    Tcw  = pFrame->get_pose();
    vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set map_point vertices
    const int N = pFrame->feature_count;

    vector<ORB_SLAM3::edge_se3_project_xyz_only_pose *>         vpEdgesMono;
    vector<ORB_SLAM3::edge_se3_project_xyz_only_pose_to_body *> vpEdgesMono_FHR;
    vector<size_t>                                              vnIndexEdgeMono, vnIndexEdgeRight;
    vpEdgesMono.reserve(N);
    vpEdgesMono_FHR.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeRight.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    vector<size_t>                                 vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono   = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(map_point::global_mutex);

        for (int i = 0; std::cmp_less(i, N); i++)
        {
            map_point *pMP = pFrame->map_points[i];
            if (pMP)
            {
                // Conventional SLAM
                if (!is_fisheye_stereo(cameras))
                {
                    // Monocular observation
                    if (pFrame->right_coordinates[i] < 0)
                    {
                        nInitialCorrespondences++;
                        pFrame->outliers[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const auto                 &kpUn = pFrame->keypoints[i];
                        obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                        ORB_SLAM3::edge_se3_project_xyz_only_pose *e = new ORB_SLAM3::edge_se3_project_xyz_only_pose(*cameras[0]);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);
                        e->point_world = pMP->get_world_position().cast<double>();

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    else // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->outliers[i] = false;

                        Eigen::Matrix<double, 3, 1> obs;
                        const auto                 &kpUn  = pFrame->keypoints[i];
                        const float                &kp_ur = pFrame->right_coordinates[i];
                        obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float     invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave];
                        Eigen::Matrix3d Info      = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = cameras[0]->get_parameter(0);
                        e->fy = cameras[0]->get_parameter(1);
                        e->cx = cameras[0]->get_parameter(2);
                        e->cy = cameras[0]->get_parameter(3);
                        e->bf = cameras[0]->baseline_fx();
                        e->Xw = pMP->get_world_position().cast<double>();

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }
                // SLAM with respect a rigid body
                else
                {
                    nInitialCorrespondences++;

                    cv::KeyPoint kpUn;

                    if (std::cmp_less(i, pFrame->keypoints.size()))
                    { // Left camera observation
                        kpUn = pFrame->keypoints[i];

                        pFrame->outliers[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        ORB_SLAM3::edge_se3_project_xyz_only_pose *e = new ORB_SLAM3::edge_se3_project_xyz_only_pose(*cameras[0]);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);
                        e->point_world = pMP->get_world_position().cast<double>();

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    else
                    {
                        kpUn = pFrame->right_keypoints[i - pFrame->keypoints.size()];

                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        pFrame->outliers[i] = false;

                        ORB_SLAM3::edge_se3_project_xyz_only_pose_to_body *e = new ORB_SLAM3::edge_se3_project_xyz_only_pose_to_body(*cameras[1]);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);
                        e->point_world = pMP->get_world_position().cast<double>();

                        const auto right_from_left = cameras[0]->left_to_right_pose().inverse();
                        e->right_from_left         = g2o::SE3Quat(right_from_left.unit_quaternion().cast<double>(), right_from_left.translation().cast<double>());

                        optimizer.addEdge(e);

                        vpEdgesMono_FHR.push_back(e);
                        vnIndexEdgeRight.push_back(i);
                    }
                }
            }
        }
    }

    if (nInitialCorrespondences < 3)
    {
        return 0;
    }

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]   = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int   its[4]        = {10, 10, 10, 10};

    int nBad = 0;
    for (size_t it = 0; it < 4; it++)
    {
        Tcw = pFrame->get_pose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::edge_se3_project_xyz_only_pose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Mono[it])
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->outliers[idx] = false;
                e->setLevel(0);
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        for (size_t i = 0, iend = vpEdgesMono_FHR.size(); i < iend; i++)
        {
            ORB_SLAM3::edge_se3_project_xyz_only_pose_to_body *e = vpEdgesMono_FHR[i];

            const size_t idx = vnIndexEdgeRight[i];

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Mono[it])
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->outliers[idx] = false;
                e->setLevel(0);
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->outliers[idx] = false;
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        if (optimizer.edges().size() < 10)
        {
            break;
        }
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *vSE3_recov    = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat          SE3quat_recov = vSE3_recov->estimate();
    Sophus::SE3<float>    pose(SE3quat_recov.rotation().cast<float>(), SE3quat_recov.translation().cast<float>());
    pFrame->set_pose(pose);

    return nInitialCorrespondences - nBad;
}

void optimizer::local_bundle_adjustment(camera_view cameras, key_frame *pKF, bool *pbStopFlag, map *pMap, int &num_fixedKF, int &num_OptKF, int &num_MPs, int &num_edges)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<key_frame *> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    pKF->local_ba_key_frame_id = pKF->id;
    map *pCurrentMap           = pKF->get_map();

    const vector<key_frame *> vNeighKFs = pKF->get_vector_covisible_key_frames();
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        key_frame *pKFi             = vNeighKFs[i];
        pKFi->local_ba_key_frame_id = pKF->id;
        if (!pKFi->is_bad() && pKFi->get_map() == pCurrentMap)
        {
            lLocalKeyFrames.push_back(pKFi);
        }
    }

    // Local MapPoints seen in Local KeyFrames
    num_fixedKF = 0;
    list<map_point *> lLocalMapPoints;
    set<map_point *>  sNumObsMP;
    for (list<key_frame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        key_frame *pKFi = *lit;
        if (pKFi->id == pMap->get_initial_key_frame_id())
        {
            num_fixedKF = 1;
        }
        vector<map_point *> vpMPs = pKFi->get_map_point_matches();
        for (vector<map_point *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            map_point *pMP = *vit;
            if (pMP)
            {
                if (!pMP->is_bad() && pMP->get_map() == pCurrentMap)
                {

                    if (pMP->local_ba_key_frame_id != pKF->id)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->local_ba_key_frame_id = pKF->id;
                    }
                }
            }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<key_frame *> lFixedCameras;
    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        std::map<key_frame *, tuple<int, int>> observations = (*lit)->get_observations();
        for (std::map<key_frame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (pKFi->local_ba_key_frame_id != pKF->id && pKFi->fixed_ba_key_frame_id != pKF->id)
            {
                pKFi->fixed_ba_key_frame_id = pKF->id;
                if (!pKFi->is_bad() && pKFi->get_map() == pCurrentMap)
                {
                    lFixedCameras.push_back(pKFi);
                }
            }
        }
    }
    num_fixedKF = lFixedCameras.size() + num_fixedKF;

    if (num_fixedKF == 0)
    {
        spdlog::debug("LM-LBA: There are 0 fixed KF in the optimizations, LBA aborted");
        return;
    }

    // Setup optimizer
    g2o::SparseOptimizer                    optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = MakeBlockSolver<g2o::BlockSolver_6_3>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    if (pMap->is_inertial())
    {
        solver->setUserLambdaInit(100.0);
    }

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
    {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    unsigned long maxKFid = 0;

    // DEBUG LBA
    std::set<unsigned long> optimized_keyframes;
    std::set<unsigned long> fixed_keyframes;

    // Set Local key_frame vertices
    for (list<key_frame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        key_frame            *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float>    Tcw  = pKFi->get_pose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->id);
        vSE3->setFixed(pKFi->id == pMap->get_initial_key_frame_id());
        optimizer.addVertex(vSE3);
        if (pKFi->id > maxKFid)
        {
            maxKFid = pKFi->id;
        }
        // DEBUG LBA
        optimized_keyframes.insert(pKFi->id);
    }
    num_OptKF = lLocalKeyFrames.size();

    // Set Fixed key_frame vertices
    for (list<key_frame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        key_frame            *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float>    Tcw  = pKFi->get_pose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->id);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->id > maxKFid)
        {
            maxKFid = pKFi->id;
        }
        // DEBUG LBA
        fixed_keyframes.insert(pKFi->id);
    }

    {
        std::unique_lock lock(pCurrentMap->viewer_mutex, std::try_to_lock);
        if (lock.owns_lock())
        {
            pCurrentMap->optimized_key_frames.swap(optimized_keyframes);
            pCurrentMap->fixed_key_frames.swap(fixed_keyframes);
        }
    }

    // Set map_point vertices
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    vector<ORB_SLAM3::edge_se3_project_xyz *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<ORB_SLAM3::edge_se3_project_xyz_to_body *> vpEdgesBody;
    vpEdgesBody.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFBody;
    vpEdgeKFBody.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeBody;
    vpMapPointEdgeBody.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono   = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    int nPoints = 0;

    int nEdges = 0;

    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map_point              *pMP    = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_world_position().cast<double>());
        int id = pMP->id + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        nPoints++;

        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();

        // Set edges
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (!pKFi->is_bad() && pKFi->get_map() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                // Monocular observation
                if (leftIndex != -1 && pKFi->right_coordinates[get<0>(mit->second)] < 0)
                {
                    const auto                 &kpUn = pKFi->keypoints[leftIndex];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                    ORB_SLAM3::edge_se3_project_xyz *e = new ORB_SLAM3::edge_se3_project_xyz(*cameras[0]);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);

                    nEdges++;
                }
                else if (leftIndex != -1 && pKFi->right_coordinates[get<0>(mit->second)] >= 0) // Stereo observation
                {
                    const auto                 &kpUn = pKFi->keypoints[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    const float                 kp_ur = pKFi->right_coordinates[get<0>(mit->second)];
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                    e->setMeasurement(obs);
                    const float    &invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];
                    Eigen::Matrix3d Info      = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = cameras[0]->get_parameter(0);
                    e->fy = cameras[0]->get_parameter(1);
                    e->cx = cameras[0]->get_parameter(2);
                    e->cy = cameras[0]->get_parameter(3);
                    e->bf = cameras[0]->baseline_fx();

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);

                    nEdges++;
                }

                if (pKFi->is_fisheye_stereo)
                {
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1)
                    {
                        rightIndex -= static_cast<int>(pKFi->keypoints.size());

                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint                kp = pKFi->right_keypoints[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        ORB_SLAM3::edge_se3_project_xyz_to_body *e = new ORB_SLAM3::edge_se3_project_xyz_to_body(*cameras[1]);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->inverse_level_sigma_squared[kp.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        Sophus::SE3f Trl   = pKFi->get_relative_pose_trl(*cameras[0]);
                        e->right_from_left = g2o::SE3Quat(Trl.unit_quaternion().cast<double>(), Trl.translation().cast<double>());

                        optimizer.addEdge(e);
                        vpEdgesBody.push_back(e);
                        vpEdgeKFBody.push_back(pKFi);
                        vpMapPointEdgeBody.push_back(pMP);

                        nEdges++;
                    }
                }
            }
        }
    }
    num_edges = nEdges;

    if (pbStopFlag)
    {
        if (*pbStopFlag)
        {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    vector<pair<key_frame *, map_point *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesBody.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::edge_se3_project_xyz *e   = vpEdgesMono[i];
        map_point                       *pMP = vpMapPointEdgeMono[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > 5.991 || !e->is_depth_positive())
        {
            key_frame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesBody.size(); i < iend; i++)
    {
        ORB_SLAM3::edge_se3_project_xyz_to_body *e   = vpEdgesBody[i];
        map_point                               *pMP = vpMapPointEdgeBody[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > 5.991 || !e->is_depth_positive())
        {
            key_frame *pKFi = vpEdgeKFBody[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e   = vpEdgesStereo[i];
        map_point                    *pMP = vpMapPointEdgeStereo[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            key_frame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->map_update_mutex);

    if (!vToErase.empty())
    {
        for (size_t i = 0; std::cmp_less(i, vToErase.size()); i++)
        {
            key_frame *pKFi = vToErase[i].first;
            map_point *pMPi = vToErase[i].second;
            pKFi->erase_map_point_match(pMPi);
            pMPi->erase_observation(pKFi);
        }
    }

    // Recover optimized data
    // Keyframes
    for (list<key_frame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        key_frame            *pKFi    = *lit;
        g2o::VertexSE3Expmap *vSE3    = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->id));
        g2o::SE3Quat          SE3quat = vSE3->estimate();
        Sophus::SE3f          Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());
        pKFi->set_pose(Tiw);
    }

    // Points
    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map_point              *pMP    = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->id + maxKFid + 1));
        pMP->set_world_position(vPoint->estimate().cast<float>());
        pMP->update_normal_and_depth(*cameras[0]);
    }

    pMap->increase_change_index();
}

void optimizer::optimize_essential_graph(camera_view                                    cameras,
                                         map                                           *pMap,
                                         key_frame                                     *pLoopKF,
                                         key_frame                                     *pCurKF,
                                         const loop_closing::key_frame_and_pose        &NonCorrectedSim3,
                                         const loop_closing::key_frame_and_pose        &CorrectedSim3,
                                         const std::map<key_frame *, set<key_frame *>> &LoopConnections,
                                         const bool                                    &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3                   *solver_ptr   = MakeBlockSolver<g2o::BlockSolver_7_3>(linearSolver);
    g2o::OptimizationAlgorithmLevenberg    *solver       = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<key_frame *> vpKFs = pMap->get_all_key_frames();
    const vector<map_point *> vpMPs = pMap->get_all_map_points();

    const unsigned int nMaxKFid = pMap->get_max_key_frame_id();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap *>                        vpVertices(nMaxKFid + 1);

    vector<Eigen::Vector3d> vZvectors(nMaxKFid + 1); // For debugging
    Eigen::Vector3d         z_vec;
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100;

    // Set key_frame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        key_frame *pKF = vpKFs[i];
        if (pKF->is_bad())
        {
            continue;
        }
        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->id;

        loop_closing::key_frame_and_pose::const_iterator it = CorrectedSim3.find(pKF);

        if (it != CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->get_pose().cast<double>();
            g2o::Sim3    Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if (pKF->id == pMap->get_initial_key_frame_id())
        {
            VSim3->setFixed(true);
        }

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi] = vScw[nIDi].rotation() * z_vec; // For debugging

        vpVertices[nIDi] = VSim3;
    }

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    // Set Loop edges
    int count_loop = 0;
    for (std::map<key_frame *, set<key_frame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
    {
        key_frame              *pKF           = mit->first;
        const long unsigned int nIDi          = pKF->id;
        const set<key_frame *> &spConnections = mit->second;
        const g2o::Sim3         Siw           = vScw[nIDi];
        const g2o::Sim3         Swi           = Siw.inverse();

        for (set<key_frame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->id;
            if ((nIDi != pCurKF->id || nIDj != pLoopKF->id) && pKF->get_weight(*sit) < minFeat)
            {
                continue;
            }

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // Set normal edges
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        key_frame *pKF = vpKFs[i];

        const int nIDi = pKF->id;

        g2o::Sim3 Swi;

        loop_closing::key_frame_and_pose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if (iti != NonCorrectedSim3.end())
        {
            Swi = (iti->second).inverse();
        }
        else
        {
            Swi = vScw[nIDi].inverse();
        }

        key_frame *pParentKF = pKF->get_parent();

        // Spanning tree edge
        if (pParentKF)
        {
            int nIDj = pParentKF->id;

            g2o::Sim3 Sjw;

            loop_closing::key_frame_and_pose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if (itj != NonCorrectedSim3.end())
            {
                Sjw = itj->second;
            }
            else
            {
                Sjw = vScw[nIDj];
            }

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<key_frame *> sLoopEdges = pKF->get_loop_edges();
        for (set<key_frame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            key_frame *pLKF = *sit;
            if (pLKF->id < pKF->id)
            {
                g2o::Sim3 Slw;

                loop_closing::key_frame_and_pose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                {
                    Slw = itl->second;
                }
                else
                {
                    Slw = vScw[pLKF->id];
                }

                g2o::Sim3      Sli = Slw * Swi;
                g2o::EdgeSim3 *el  = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->id)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<key_frame *> vpConnectedKFs = pKF->get_covisibles_by_weight(minFeat);
        for (vector<key_frame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            key_frame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && !pKF->has_child(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
            {
                if (!pKFn->is_bad() && pKFn->id < pKF->id)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->id, pKFn->id), max(pKF->id, pKFn->id))))
                    {
                        continue;
                    }

                    g2o::Sim3 Snw;

                    loop_closing::key_frame_and_pose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                    {
                        Snw = itn->second;
                    }
                    else
                    {
                        Snw = vScw[pKFn->id];
                    }

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->id)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        // Inertial edges if inertial
        if (pKF->has_imu && pKF->previous_key_frame)
        {
            g2o::Sim3                                        Spw;
            loop_closing::key_frame_and_pose::const_iterator itp = NonCorrectedSim3.find(pKF->previous_key_frame);
            if (itp != NonCorrectedSim3.end())
            {
                Spw = itp->second;
            }
            else
            {
                Spw = vScw[pKF->previous_key_frame->id];
            }

            g2o::Sim3      Spi = Spw * Swi;
            g2o::EdgeSim3 *ep  = new g2o::EdgeSim3();
            ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->previous_key_frame->id)));
            ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            ep->setMeasurement(Spi);
            ep->information() = matLambda;
            optimizer.addEdge(ep);
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    unique_lock<mutex> lock(pMap->map_update_mutex);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];

        const int nIDi = pKFi->id;

        g2o::VertexSim3Expmap *VSim3        = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3              CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi]                 = CorrectedSiw.inverse();
        double s                            = CorrectedSiw.scale();

        Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
        pKFi->set_pose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        map_point *pMP = vpMPs[i];

        if (pMP->is_bad())
        {
            continue;
        }

        int nIDr;
        if (pMP->corrected_by_key_frame_id == pCurKF->id)
        {
            nIDr = pMP->corrected_reference_key_frame_id;
        }
        else
        {
            key_frame *pRefKF = pMP->get_reference_key_frame();
            nIDr              = pRefKF->id;
        }

        g2o::Sim3 Srw          = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double, 3, 1> eigP3Dw          = pMP->get_world_position().cast<double>();
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->set_world_position(eigCorrectedP3Dw.cast<float>());

        pMP->update_normal_and_depth(*cameras[0]);
    }

    // TODO Check this changeindex
    pMap->increase_change_index();
}

void optimizer::optimize_essential_graph(camera_view          cameras,
                                         key_frame           *pCurKF,
                                         vector<key_frame *> &vpFixedKFs,
                                         vector<key_frame *> &vpFixedCorrectedKFs,
                                         vector<key_frame *> &vpNonFixedKFs,
                                         vector<map_point *> &vpNonCorrectedMPs)
{
    spdlog::trace("Opt_Essential: There are {} KFs fixed in the merged map", vpFixedKFs.size());
    spdlog::trace("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map");
    spdlog::trace("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map");
    spdlog::trace("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map");

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3                   *solver_ptr   = MakeBlockSolver<g2o::BlockSolver_7_3>(linearSolver);
    g2o::OptimizationAlgorithmLevenberg    *solver       = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    map               *pMap     = pCurKF->get_map();
    const unsigned int nMaxKFid = pMap->get_max_key_frame_id();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap *>                        vpVertices(nMaxKFid + 1);

    vector<bool> vpGoodPose(nMaxKFid + 1);
    vector<bool> vpBadPose(nMaxKFid + 1);

    const int minFeat = 100;

    for (key_frame *pKFi : vpFixedKFs)
    {
        if (pKFi->is_bad())
        {
            continue;
        }

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->id;

        Sophus::SE3d Tcw = pKFi->get_pose().cast<double>();
        g2o::Sim3    Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

        vCorrectedSwc[nIDi] = Siw.inverse();
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi]  = false;
    }
    spdlog::trace("Opt_Essential: vpFixedKFs loaded");

    set<unsigned long> sIdKF;
    for (key_frame *pKFi : vpFixedCorrectedKFs)
    {
        if (pKFi->is_bad())
        {
            continue;
        }

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->id;

        Sophus::SE3d Tcw = pKFi->get_pose().cast<double>();
        g2o::Sim3    Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

        vCorrectedSwc[nIDi] = Siw.inverse();
        VSim3->setEstimate(Siw);

        Sophus::SE3d Tcw_bef = pKFi->camera_pose_before_merge.cast<double>();
        vScw[nIDi]           = g2o::Sim3(Tcw_bef.unit_quaternion(), Tcw_bef.translation(), 1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi]  = true;
    }

    for (key_frame *pKFi : vpNonFixedKFs)
    {
        if (pKFi->is_bad())
        {
            continue;
        }

        const int nIDi = pKFi->id;

        if (sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
        {
            continue;
        }

        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        Sophus::SE3d Tcw = pKFi->get_pose().cast<double>();
        g2o::Sim3    Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi]  = true;
    }

    vector<key_frame *> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(), vpFixedKFs.begin(), vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(), vpFixedCorrectedKFs.begin(), vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(), vpNonFixedKFs.begin(), vpNonFixedKFs.end());
    set<key_frame *> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    for (key_frame *pKFi : vpKFs)
    {
        int       num_connections = 0;
        const int nIDi            = pKFi->id;

        g2o::Sim3 correctedSwi;
        g2o::Sim3 Swi;

        if (vpGoodPose[nIDi])
        {
            correctedSwi = vCorrectedSwc[nIDi];
        }
        if (vpBadPose[nIDi])
        {
            Swi = vScw[nIDi].inverse();
        }

        key_frame *pParentKFi = pKFi->get_parent();

        // Spanning tree edge
        if (pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->id;

            g2o::Sim3 Sjw;
            bool      bHasRelation = false;

            if (vpGoodPose[nIDi] && vpGoodPose[nIDj])
            {
                Sjw          = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            }
            else if (vpBadPose[nIDi] && vpBadPose[nIDj])
            {
                Sjw          = vScw[nIDj];
                bHasRelation = true;
            }

            if (bHasRelation)
            {
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
                num_connections++;
            }
        }

        // Loop edges
        const set<key_frame *> sLoopEdges = pKFi->get_loop_edges();
        for (set<key_frame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            key_frame *pLKF = *sit;
            if (spKFs.find(pLKF) != spKFs.end() && pLKF->id < pKFi->id)
            {
                g2o::Sim3 Slw;
                bool      bHasRelation = false;

                if (vpGoodPose[nIDi] && vpGoodPose[pLKF->id])
                {
                    Slw          = vCorrectedSwc[pLKF->id].inverse();
                    bHasRelation = true;
                }
                else if (vpBadPose[nIDi] && vpBadPose[pLKF->id])
                {
                    Slw          = vScw[pLKF->id];
                    bHasRelation = true;
                }

                if (bHasRelation)
                {
                    g2o::Sim3      Sli = Slw * Swi;
                    g2o::EdgeSim3 *el  = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->id)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<key_frame *> vpConnectedKFs = pKFi->get_covisibles_by_weight(minFeat);
        for (vector<key_frame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            key_frame *pKFn = *vit;
            if (pKFn && pKFn != pParentKFi && !pKFi->has_child(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if (!pKFn->is_bad() && pKFn->id < pKFi->id)
                {

                    g2o::Sim3 Snw          = vScw[pKFn->id];
                    bool      bHasRelation = false;

                    if (vpGoodPose[nIDi] && vpGoodPose[pKFn->id])
                    {
                        Snw          = vCorrectedSwc[pKFn->id].inverse();
                        bHasRelation = true;
                    }
                    else if (vpBadPose[nIDi] && vpBadPose[pKFn->id])
                    {
                        Snw          = vScw[pKFn->id];
                        bHasRelation = true;
                    }

                    if (bHasRelation)
                    {
                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->id)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        num_connections++;
                    }
                }
            }
        }

        if (num_connections == 0)
        {
            spdlog::trace("Opt_Essential: KF " + to_string(pKFi->id) + " has 0 connections");
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->map_update_mutex);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (key_frame *pKFi : vpNonFixedKFs)
    {
        if (pKFi->is_bad())
        {
            continue;
        }

        const int nIDi = pKFi->id;

        g2o::VertexSim3Expmap *VSim3        = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3              CorrectedSiw = VSim3->estimate();
        vCorrectedSwc[nIDi]                 = CorrectedSiw.inverse();
        double       s                      = CorrectedSiw.scale();
        Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation() / s);

        pKFi->camera_pose_before_merge = pKFi->get_pose();
        pKFi->world_pose_before_merge  = pKFi->get_pose_inverse();
        pKFi->set_pose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (map_point *pMPi : vpNonCorrectedMPs)
    {
        if (pMPi->is_bad())
        {
            continue;
        }

        key_frame *pRefKF = pMPi->get_reference_key_frame();
        while (pRefKF->is_bad())
        {
            if (!pRefKF)
            {
                spdlog::trace("MP " + to_string(pMPi->id) + " without a valid reference KF");
                break;
            }

            pMPi->erase_observation(pRefKF);
            pRefKF = pMPi->get_reference_key_frame();
        }

        if (vpBadPose[pRefKF->id])
        {
            Sophus::SE3f TNonCorrectedwr = pRefKF->world_pose_before_merge;
            Sophus::SE3f Twr             = pRefKF->get_pose_inverse();

            Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->get_world_position();
            pMPi->set_world_position(eigCorrectedP3Dw);

            pMPi->update_normal_and_depth(*cameras[0]);
        }
        else
        {
            spdlog::error("Map point reference keyframe belongs to another map");
        }
    }
}

int optimizer::optimize_sim3(camera_view                  cameras,
                             key_frame                   *pKF1,
                             key_frame                   *pKF2,
                             vector<map_point *>         &vpMatches1,
                             g2o::Sim3                   &g2oS12,
                             const float                  th2,
                             const bool                   bFixScale,
                             Eigen::Matrix<double, 7, 7> &mAcumHessian,
                             const bool                   bAllPoints)
{
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Camera poses
    const Eigen::Matrix3f R1w = pKF1->get_rotation();
    const Eigen::Vector3f t1w = pKF1->get_translation();
    const Eigen::Matrix3f R2w = pKF2->get_rotation();
    const Eigen::Vector3f t2w = pKF2->get_translation();

    // Set Sim3 vertex
    ORB_SLAM3::vertex_sim3_expmap *vSim3 = new ORB_SLAM3::vertex_sim3_expmap(cameras);
    vSim3->_fix_scale                    = bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    optimizer.addVertex(vSim3);

    // Set map_point vertices
    const int                                          N            = vpMatches1.size();
    const vector<map_point *>                          vpMapPoints1 = pKF1->get_map_point_matches();
    vector<ORB_SLAM3::edge_sim3_project_xyz *>         vpEdges12;
    vector<ORB_SLAM3::edge_inverse_sim3_project_xyz *> vpEdges21;
    vector<size_t>                                     vnIndexEdge;
    vector<bool>                                       vbIsInKF2;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);
    vbIsInKF2.reserve(2 * N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs          = 0;
    int nInKF2           = 0;
    int nOutKF2          = 0;
    int nMatchWithoutMP  = 0;

    vector<int> vIdsOnlyInKF2;

    for (int i = 0; std::cmp_less(i, N); i++)
    {
        if (!vpMatches1[i])
        {
            continue;
        }

        map_point *pMP1 = vpMapPoints1[i];
        map_point *pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = get<0>(pMP2->get_index_in_key_frame(pKF2));

        Eigen::Vector3f P3D1c;
        Eigen::Vector3f P3D2c;

        if (pMP1 && pMP2)
        {
            if (!pMP1->is_bad() && !pMP2->is_bad())
            {
                g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f         P3D1w   = pMP1->get_world_position();
                P3D1c                           = R1w * P3D1w + t1w;
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f         P3D2w   = pMP2->get_world_position();
                P3D2c                           = R2w * P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            // TODO The 3D position in KF1 doesn't exist
            if (!pMP2->is_bad())
            {
                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f         P3D2w   = pMP2->get_world_position();
                P3D2c                           = R2w * P3D2w + t2w;
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }
            continue;
        }

        if (i2 < 0 && !bAllPoints)
        {
            spdlog::trace("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints));
            continue;
        }

        if (P3D2c(2) < 0)
        {
            spdlog::trace("Sim3: Z coordinate is negative");
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const auto                 &kpUn1 = pKF1->keypoints[i];
        obs1 << kpUn1.pt_undistorted.x, kpUn1.pt_undistorted.y;

        ORB_SLAM3::edge_sim3_project_xyz *e12 = new ORB_SLAM3::edge_sim3_project_xyz();

        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->inverse_level_sigma_squared[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double, 2, 1> obs2;
        int                         octave2 = 0;
        bool                        inKF2;
        if (i2 >= 0)
        {
            const auto &kpUn2 = pKF2->keypoints[i2];
            obs2 << kpUn2.pt_undistorted.x, kpUn2.pt_undistorted.y;
            octave2 = kpUn2.octave;
            inKF2   = true;

            nInKF2++;
        }
        else
        {
            float invz = 1 / P3D2c(2);
            float x    = P3D2c(0) * invz;
            float y    = P3D2c(1) * invz;

            obs2 << x, y;
            inKF2 = false;
            nOutKF2++;
        }

        ORB_SLAM3::edge_inverse_sim3_project_xyz *e21 = new ORB_SLAM3::edge_inverse_sim3_project_xyz();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->inverse_level_sigma_squared[octave2];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad       = 0;
    int nBadOutKF2 = 0;
    for (size_t i = 0; std::cmp_less(i, vpEdges12.size()); i++)
    {
        ORB_SLAM3::edge_sim3_project_xyz         *e12 = vpEdges12[i];
        ORB_SLAM3::edge_inverse_sim3_project_xyz *e21 = vpEdges21[i];
        if (!e12 || !e21)
        {
            continue;
        }

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx      = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<map_point *>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<ORB_SLAM3::edge_sim3_project_xyz *>(NULL);
            vpEdges21[i] = static_cast<ORB_SLAM3::edge_inverse_sim3_project_xyz *>(NULL);
            nBad++;

            if (!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        // Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations;
    if (nBad > 0)
    {
        nMoreIterations = 10;
    }
    else
    {
        nMoreIterations = 5;
    }

    if (nCorrespondences - nBad < 10)
    {
        return 0;
    }

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn      = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for (size_t i = 0; std::cmp_less(i, vpEdges12.size()); i++)
    {
        ORB_SLAM3::edge_sim3_project_xyz         *e12 = vpEdges12[i];
        ORB_SLAM3::edge_inverse_sim3_project_xyz *e21 = vpEdges21[i];
        if (!e12 || !e21)
        {
            continue;
        }

        e12->computeError();
        e21->computeError();

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx      = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<map_point *>(NULL);
        }
        else
        {
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12                             = vSim3_recov->estimate();

    return nIn;
}

void optimizer::local_inertial_ba(camera_view cameras, key_frame *pKF, bool *pbStopFlag, map *pMap, int &num_fixedKF, int &num_OptKF, int &num_MPs, int &num_edges, bool bLarge, bool bRecInit)
{
    map *pCurrentMap = pKF->get_map();

    int maxOpt = 10;
    int opt_it = 10;
    if (bLarge)
    {
        maxOpt = 25;
        opt_it = 4;
    }
    const int           Nd      = std::min((int)pCurrentMap->key_frame_count() - 2, maxOpt);
    const unsigned long maxKFid = pKF->id;

    vector<key_frame *>       vpOptimizableKFs;
    const vector<key_frame *> vpNeighsKFs = pKF->get_vector_covisible_key_frames();
    list<key_frame *>         lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->local_ba_key_frame_id = pKF->id;
    for (int i = 1; std::cmp_less(i, Nd); i++)
    {
        if (vpOptimizableKFs.back()->previous_key_frame)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->previous_key_frame);
            vpOptimizableKFs.back()->local_ba_key_frame_id = pKF->id;
        }
        else
        {
            break;
        }
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<map_point *> lLocalMapPoints;
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        vector<map_point *> vpMPs = vpOptimizableKFs[i]->get_map_point_matches();
        for (vector<map_point *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            map_point *pMP = *vit;
            if (pMP)
            {
                if (!pMP->is_bad())
                {
                    if (pMP->local_ba_key_frame_id != pKF->id)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->local_ba_key_frame_id = pKF->id;
                    }
                }
            }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    list<key_frame *> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->previous_key_frame)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->previous_key_frame);
        vpOptimizableKFs.back()->previous_key_frame->fixed_ba_key_frame_id = pKF->id;
    }
    else
    {
        vpOptimizableKFs.back()->local_ba_key_frame_id = 0;
        vpOptimizableKFs.back()->fixed_ba_key_frame_id = pKF->id;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for (int i = 0, iend = vpNeighsKFs.size(); i < iend; i++)
    {
        if (lpOptVisKFs.size() >= maxCovKF)
        {
            break;
        }

        key_frame *pKFi = vpNeighsKFs[i];
        if (pKFi->local_ba_key_frame_id == pKF->id || pKFi->fixed_ba_key_frame_id == pKF->id)
        {
            continue;
        }
        pKFi->local_ba_key_frame_id = pKF->id;
        if (!pKFi->is_bad() && pKFi->get_map() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<map_point *> vpMPs = pKFi->get_map_point_matches();
            for (vector<map_point *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                map_point *pMP = *vit;
                if (pMP)
                {
                    if (!pMP->is_bad())
                    {
                        if (pMP->local_ba_key_frame_id != pKF->id)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->local_ba_key_frame_id = pKF->id;
                        }
                    }
                }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    const int maxFixKF = 200;

    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        std::map<key_frame *, tuple<int, int>> observations = (*lit)->get_observations();
        for (std::map<key_frame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (pKFi->local_ba_key_frame_id != pKF->id && pKFi->fixed_ba_key_frame_id != pKF->id)
            {
                pKFi->fixed_ba_key_frame_id = pKF->id;
                if (!pKFi->is_bad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
        if (lFixedKeyFrames.size() >= maxFixKF)
        {
            break;
        }
    }

    [[maybe_unused]] bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    if (bLarge)
    {
        g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
        solver->setUserLambdaInit(1e-2); // to avoid iterating for finding optimal lambda
        optimizer.setAlgorithm(solver);
    }
    else
    {
        g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }

    // Set Local temporal key_frame vertices
    N = vpOptimizableKFs.size();
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpOptimizableKFs[i];

        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = new vertex_velocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->id) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            vertex_gyro_bias *VG = new vertex_gyro_bias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->id) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            vertex_acc_bias *VA = new vertex_acc_bias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->id) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local visual key_frame vertices
    for (list<key_frame *>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        key_frame   *pKFi = *it;
        vertex_pose *VP   = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(false);
        optimizer.addVertex(VP);
    }

    // Set Fixed key_frame vertices
    for (list<key_frame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        key_frame   *pKFi = *lit;
        vertex_pose *VP   = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if (pKFi->has_imu) // This should be done only for keyframe just before temporal window
        {
            vertex_velocity *VV = new vertex_velocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->id) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            vertex_gyro_bias *VG = new vertex_gyro_bias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->id) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            vertex_acc_bias *VA = new vertex_acc_bias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->id) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<edge_inertial *> vei(N, (edge_inertial *)NULL);
    vector<edge_gyro_rw *>  vegr(N, (edge_gyro_rw *)NULL);
    vector<edge_acc_rw *>   vear(N, (edge_acc_rw *)NULL);

    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpOptimizableKFs[i];

        if (!pKFi->previous_key_frame)
        {
            spdlog::error("Missing inertial link to previous frame");
            continue;
        }
        if (pKFi->has_imu && pKFi->previous_key_frame->has_imu && pKFi->imu_preintegrated)
        {
            pKFi->imu_preintegrated->set_new_bias(pKFi->previous_key_frame->get_imu_bias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 1);
            g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 2);
            g2o::HyperGraph::Vertex *VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 3);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->id);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1);
            g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2);
            g2o::HyperGraph::Vertex *VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3);

            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                spdlog::error("Unable to build inertial edge: required graph vertex is missing");
                continue;
            }

            vei[i] = new edge_inertial(pKFi->imu_preintegrated);

            vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
            vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
            vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

            if (i == N - 1 || bRecInit)
            {
                // All inertial residuals are included without robust cost function, but not that one linking the
                // last optimizable keyframe inside of the local window and the first fixed keyframe out. The
                // information matrix for this measurement is also downweighted. This is done to avoid accumulating
                // error due to fixing variables.
                g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
                vei[i]->setRobustKernel(rki);
                if (i == N - 1)
                {
                    vei[i]->setInformation(vei[i]->information() * 1e-2);
                }
                rki->setDelta(sqrt(16.92));
            }
            optimizer.addEdge(vei[i]);

            vegr[i] = new edge_gyro_rw();
            vegr[i]->setVertex(0, VG1);
            vegr[i]->setVertex(1, VG2);
            Eigen::Matrix3d InfoG = pKFi->imu_preintegrated->covariance.block<3, 3>(9, 9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new edge_acc_rw();
            vear[i]->setVertex(0, VA1);
            vear[i]->setVertex(1, VA2);
            Eigen::Matrix3d InfoA = pKFi->imu_preintegrated->covariance.block<3, 3>(12, 12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);

            optimizer.addEdge(vear[i]);
        }
        else
        {
            spdlog::error("Unable to build inertial edge");
        }
    }

    // Set map_point vertices
    const int nExpectedSize = (N + lFixedKeyFrames.size()) * lLocalMapPoints.size();

    // Mono
    vector<edge_mono *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<edge_stereo *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono   = sqrt(5.991);
    const float chi2Mono2     = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2   = 7.815;

    const unsigned long iniMPid = maxKFid * 5;

    std::map<int, int> mVisEdges;
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi     = vpOptimizableKFs[i];
        mVisEdges[pKFi->id] = 0;
    }
    for (list<key_frame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        mVisEdges[(*lit)->id] = 0;
    }

    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map_point              *pMP    = *lit;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_world_position().cast<double>());

        unsigned long id = pMP->id + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();

        // Create visual constraints
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (pKFi->local_ba_key_frame_id != pKF->id && pKFi->fixed_ba_key_frame_id != pKF->id)
            {
                continue;
            }

            if (!pKFi->is_bad() && pKFi->get_map() == pCurrentMap)
            {
                const int leftIndex = get<0>(mit->second);

                int left_octave = 0;

                // Monocular left observation
                if (leftIndex != -1 && pKFi->right_coordinates[leftIndex] < 0)
                {
                    mVisEdges[pKFi->id]++;

                    const auto &kpUn = pKFi->keypoints[leftIndex];
                    left_octave      = kpUn.octave;
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                    edge_mono *e = new edge_mono(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs);

                    const float &invSigma2 = pKFi->inverse_level_sigma_squared[left_octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                // Stereo-observation
                else if (leftIndex != -1) // Stereo observation
                {
                    const auto &kpUn = pKFi->keypoints[leftIndex];
                    left_octave      = kpUn.octave;
                    mVisEdges[pKFi->id]++;

                    const float                 kp_ur = pKFi->right_coordinates[leftIndex];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                    edge_stereo *e = new edge_stereo(0);

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs.head(2));

                    const float &invSigma2 = pKFi->inverse_level_sigma_squared[left_octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }

                // Monocular right observation
                if (pKFi->is_fisheye_stereo)
                {
                    int rightIndex = get<1>(mit->second);

                    if (rightIndex != -1)
                    {
                        rightIndex -= static_cast<int>(pKFi->keypoints.size());
                        mVisEdges[pKFi->id]++;

                        Eigen::Matrix<double, 2, 1> obs;
                        cv::KeyPoint                kp = pKFi->right_keypoints[rightIndex];
                        obs << kp.pt.x, kp.pt.y;

                        edge_mono *e = new edge_mono(1);

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                        e->setMeasurement(obs);

                        // Add here uncerteinty
                        const float unc2 = cameras[0]->uncertainty2(obs);

                        const float &invSigma2 = pKFi->inverse_level_sigma_squared[left_octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                }
            }
        }
    }

    for (std::map<int, int>::iterator mit = mVisEdges.begin(), mend = mVisEdges.end(); mit != mend; mit++)
    {
        assert(mit->second >= 3);
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it); // Originally to 2
    float err_end = optimizer.activeRobustChi2();
    if (pbStopFlag)
    {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    vector<pair<key_frame *, map_point *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        edge_mono *e      = vpEdgesMono[i];
        map_point *pMP    = vpMapPointEdgeMono[i];
        bool       bClose = pMP->track_depth < 10.f;

        if (pMP->is_bad())
        {
            continue;
        }

        if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->is_depth_positive())
        {
            key_frame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Stereo
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        edge_stereo *e   = vpEdgesStereo[i];
        map_point   *pMP = vpMapPointEdgeStereo[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > chi2Stereo2)
        {
            key_frame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->map_update_mutex);

    // TODO: Some convergence problems have been detected here
    if ((2 * err < err_end || isnan(err) || isnan(err_end)) && !bLarge) // bGN)
    {
        spdlog::error("Local-inertial bundle adjustment failed");
        return;
    }

    if (!vToErase.empty())
    {
        for (size_t i = 0; std::cmp_less(i, vToErase.size()); i++)
        {
            key_frame *pKFi = vToErase[i].first;
            map_point *pMPi = vToErase[i].second;
            pKFi->erase_map_point_match(pMPi);
            pMPi->erase_observation(pKFi);
        }
    }

    for (list<key_frame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        (*lit)->fixed_ba_key_frame_id = 0;
    }

    // Recover optimized data
    // Local temporal Keyframes
    N = vpOptimizableKFs.size();
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpOptimizableKFs[i];

        vertex_pose *VP = static_cast<vertex_pose *>(optimizer.vertex(pKFi->id));
        Sophus::SE3f Tcw(VP->estimate().rotation_camera_world[0].cast<float>(), VP->estimate().translation_camera_world[0].cast<float>());
        pKFi->set_pose(Tcw);
        pKFi->local_ba_key_frame_id = 0;

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = static_cast<vertex_velocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1));
            pKFi->set_velocity(VV->estimate().cast<float>());
            vertex_gyro_bias *VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2));
            vertex_acc_bias  *VA = static_cast<vertex_acc_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3));
            vector_6d         b;
            b << VG->estimate(), VA->estimate();
            pKFi->set_new_bias(imu::bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    // Local visual key_frame
    for (list<key_frame *>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        key_frame   *pKFi = *it;
        vertex_pose *VP   = static_cast<vertex_pose *>(optimizer.vertex(pKFi->id));
        Sophus::SE3f Tcw(VP->estimate().rotation_camera_world[0].cast<float>(), VP->estimate().translation_camera_world[0].cast<float>());
        pKFi->set_pose(Tcw);
        pKFi->local_ba_key_frame_id = 0;
    }

    // Points
    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map_point              *pMP    = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->id + iniMPid + 1));
        pMP->set_world_position(vPoint->estimate().cast<float>());
        pMP->update_normal_and_depth(*cameras[0]);
    }

    pMap->increase_change_index();
}

Eigen::MatrixXd optimizer::marginalize(const Eigen::MatrixXd &H, const int &start, const int &end)
{
    // Goal
    // a  | ab | ac       a*  | 0 | ac*
    // ba | b  | bc  -->  0   | 0 | 0
    // ca | cb | c        ca* | 0 | c*

    // Size of block before block to marginalize
    const int a = start;
    // Size of block to marginalize
    const int b = end - start + 1;
    // Size of block after block to marginalize
    const int c = H.cols() - (end + 1);

    // Reorder as follows:
    // a  | ab | ac       a  | ac | ab
    // ba | b  | bc  -->  ca | c  | cb
    // ca | cb | c        ba | bc | b

    Eigen::MatrixXd Hn = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    if (a > 0)
    {
        Hn.block(0, 0, a, a)     = H.block(0, 0, a, a);
        Hn.block(0, a + c, a, b) = H.block(0, a, a, b);
        Hn.block(a + c, 0, b, a) = H.block(a, 0, b, a);
    }
    if (a > 0 && c > 0)
    {
        Hn.block(0, a, a, c) = H.block(0, a + b, a, c);
        Hn.block(a, 0, c, a) = H.block(a + b, 0, c, a);
    }
    if (c > 0)
    {
        Hn.block(a, a, c, c)     = H.block(a + b, a + b, c, c);
        Hn.block(a, a + c, c, b) = H.block(a + b, a, c, b);
        Hn.block(a + c, a, b, c) = H.block(a, a + b, b, c);
    }
    Hn.block(a + c, a + c, b, b) = H.block(a, a, b, b);

    // Perform marginalization (Schur complement)
    Eigen::JacobiSVD<Eigen::MatrixXd>                     svd(Hn.block(a + c, a + c, b, b), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singularValues_inv = svd.singularValues();
    for (int i = 0; i < b; ++i)
    {
        if (singularValues_inv(i) > 1e-6)
        {
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        }
        else
        {
            singularValues_inv(i) = 0;
        }
    }
    Eigen::MatrixXd invHb        = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
    Hn.block(0, 0, a + c, a + c) = Hn.block(0, 0, a + c, a + c) - Hn.block(0, a + c, a + c, b) * invHb * Hn.block(a + c, 0, b, a + c);
    Hn.block(a + c, a + c, b, b) = Eigen::MatrixXd::Zero(b, b);
    Hn.block(0, a + c, a + c, b) = Eigen::MatrixXd::Zero(a + c, b);
    Hn.block(a + c, 0, b, a + c) = Eigen::MatrixXd::Zero(b, a + c);

    // Inverse reorder
    // a*  | ac* | 0       a*  | 0 | ac*
    // ca* | c*  | 0  -->  0   | 0 | 0
    // 0   | 0   | 0       ca* | 0 | c*
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(H.rows(), H.cols());
    if (a > 0)
    {
        res.block(0, 0, a, a) = Hn.block(0, 0, a, a);
        res.block(0, a, a, b) = Hn.block(0, a + c, a, b);
        res.block(a, 0, b, a) = Hn.block(a + c, 0, b, a);
    }
    if (a > 0 && c > 0)
    {
        res.block(0, a + b, a, c) = Hn.block(0, a, a, c);
        res.block(a + b, 0, c, a) = Hn.block(a, 0, c, a);
    }
    if (c > 0)
    {
        res.block(a + b, a + b, c, c) = Hn.block(a, a, c, c);
        res.block(a + b, a, c, b)     = Hn.block(a, a + c, c, b);
        res.block(a, a + b, b, c)     = Hn.block(a + c, a, b, c);
    }

    res.block(a, a, b, b) = Hn.block(a + c, a + c, b, b);

    return res;
}

void optimizer::inertial_optimization(camera_view      cameras,
                                      map             *pMap,
                                      Eigen::Matrix3d &Rwg,
                                      double          &scale,
                                      Eigen::Vector3d &bg,
                                      Eigen::Vector3d &ba,
                                      bool             bMono,
                                      Eigen::MatrixXd &covInertial,
                                      bool             bFixedVel,
                                      bool             bGauss,
                                      float            priorG,
                                      float            priorA)
{
    spdlog::trace("inertial optimization");
    int                       its     = 200;
    long unsigned int         maxKFid = pMap->get_max_key_frame_id();
    const vector<key_frame *> vpKFs   = pMap->get_all_key_frames();

    // Setup optimizer
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);

    if (priorG != 0.f)
    {
        solver->setUserLambdaInit(1e3);
    }

    optimizer.setAlgorithm(solver);

    // Set key_frame vertices (fixed poses and optimizable velocities)
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }
        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        vertex_velocity *VV = new vertex_velocity(pKFi);
        VV->setId(maxKFid + (pKFi->id) + 1);
        if (bFixedVel)
        {
            VV->setFixed(true);
        }
        else
        {
            VV->setFixed(false);
        }

        optimizer.addVertex(VV);
    }

    // Biases
    vertex_gyro_bias *VG = new vertex_gyro_bias(vpKFs.front());
    VG->setId(maxKFid * 2 + 2);
    if (bFixedVel)
    {
        VG->setFixed(true);
    }
    else
    {
        VG->setFixed(false);
    }
    optimizer.addVertex(VG);
    vertex_acc_bias *VA = new vertex_acc_bias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    if (bFixedVel)
    {
        VA->setFixed(true);
    }
    else
    {
        VA->setFixed(false);
    }

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    edge_prior_acc *epa = new edge_prior_acc(bprior);
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    edge_prior_gyro *epg = new edge_prior_gyro(bprior);
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    vertex_gravity_direction *VGDir = new vertex_gravity_direction(Rwg);
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    vertex_scale *VS = new vertex_scale(scale);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(!bMono); // Fixed for stereo case
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<edge_inertial_gs *> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<key_frame *, key_frame *>> vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];

        if (pKFi->previous_key_frame && pKFi->id <= maxKFid)
        {
            if (pKFi->is_bad() || pKFi->previous_key_frame->id > maxKFid)
            {
                continue;
            }
            if (!pKFi->imu_preintegrated)
            {
                spdlog::error("Missing preintegrated IMU measurement");
            }

            pKFi->imu_preintegrated->set_new_bias(pKFi->previous_key_frame->get_imu_bias());
            g2o::HyperGraph::Vertex *VP1   = optimizer.vertex(pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VV1   = optimizer.vertex(maxKFid + (pKFi->previous_key_frame->id) + 1);
            g2o::HyperGraph::Vertex *VP2   = optimizer.vertex(pKFi->id);
            g2o::HyperGraph::Vertex *VV2   = optimizer.vertex(maxKFid + (pKFi->id) + 1);
            g2o::HyperGraph::Vertex *VG    = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex *VA    = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex *VS    = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                spdlog::error("Unable to build inertial edge: required graph vertex is missing");

                continue;
            }
            edge_inertial_gs *ei = new edge_inertial_gs(pKFi->imu_preintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->previous_key_frame, pKFi));
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    std::set<g2o::HyperGraph::Edge *> setEdges = optimizer.edges();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    scale = VS->estimate();

    // Recover optimized data
    // Biases
    VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<vertex_acc_bias *>(optimizer.vertex(maxKFid * 2 + 3));
    vector_6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();
    scale = VS->estimate();

    imu::bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);
    Rwg = VGDir->estimate().rotation_world_gravity;

    // Keyframes velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }

        vertex_velocity *VV = static_cast<vertex_velocity *>(optimizer.vertex(maxKFid + (pKFi->id) + 1));
        Eigen::Vector3d  Vw = VV->estimate(); // Velocity is scaled after
        pKFi->set_velocity(Vw.cast<float>());

        if ((pKFi->get_gyro_bias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->set_new_bias(b);
            if (pKFi->imu_preintegrated)
            {
                pKFi->imu_preintegrated->reintegrate();
            }
        }
        else
        {
            pKFi->set_new_bias(b);
        }
    }
}

void optimizer::inertial_optimization(camera_view cameras, map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG, float priorA)
{
    int                       its     = 200; // Check number of iterations
    long unsigned int         maxKFid = pMap->get_max_key_frame_id();
    const vector<key_frame *> vpKFs   = pMap->get_all_key_frames();

    // Setup optimizer
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);

    // Set key_frame vertices (fixed poses and optimizable velocities)
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }
        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        vertex_velocity *VV = new vertex_velocity(pKFi);
        VV->setId(maxKFid + (pKFi->id) + 1);
        VV->setFixed(false);

        optimizer.addVertex(VV);
    }

    // Biases
    vertex_gyro_bias *VG = new vertex_gyro_bias(vpKFs.front());
    VG->setId(maxKFid * 2 + 2);
    VG->setFixed(false);
    optimizer.addVertex(VG);

    vertex_acc_bias *VA = new vertex_acc_bias(vpKFs.front());
    VA->setId(maxKFid * 2 + 3);
    VA->setFixed(false);

    optimizer.addVertex(VA);
    // prior acc bias
    Eigen::Vector3f bprior;
    bprior.setZero();

    edge_prior_acc *epa = new edge_prior_acc(bprior);
    epa->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
    double infoPriorA = priorA;
    epa->setInformation(infoPriorA * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epa);
    edge_prior_gyro *epg = new edge_prior_gyro(bprior);
    epg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
    double infoPriorG = priorG;
    epg->setInformation(infoPriorG * Eigen::Matrix3d::Identity());
    optimizer.addEdge(epg);

    // Gravity and scale
    vertex_gravity_direction *VGDir = new vertex_gravity_direction(Eigen::Matrix3d::Identity());
    VGDir->setId(maxKFid * 2 + 4);
    VGDir->setFixed(true);
    optimizer.addVertex(VGDir);
    vertex_scale *VS = new vertex_scale(1.0);
    VS->setId(maxKFid * 2 + 5);
    VS->setFixed(true); // Fixed since scale is obtained from already well initialized map
    optimizer.addVertex(VS);

    // Graph edges
    // IMU links with gravity and scale
    vector<edge_inertial_gs *> vpei;
    vpei.reserve(vpKFs.size());
    vector<pair<key_frame *, key_frame *>> vppUsedKF;
    vppUsedKF.reserve(vpKFs.size());

    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];

        if (pKFi->previous_key_frame && pKFi->id <= maxKFid)
        {
            if (pKFi->is_bad() || pKFi->previous_key_frame->id > maxKFid)
            {
                continue;
            }

            pKFi->imu_preintegrated->set_new_bias(pKFi->previous_key_frame->get_imu_bias());
            g2o::HyperGraph::Vertex *VP1   = optimizer.vertex(pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VV1   = optimizer.vertex(maxKFid + (pKFi->previous_key_frame->id) + 1);
            g2o::HyperGraph::Vertex *VP2   = optimizer.vertex(pKFi->id);
            g2o::HyperGraph::Vertex *VV2   = optimizer.vertex(maxKFid + (pKFi->id) + 1);
            g2o::HyperGraph::Vertex *VG    = optimizer.vertex(maxKFid * 2 + 2);
            g2o::HyperGraph::Vertex *VA    = optimizer.vertex(maxKFid * 2 + 3);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(maxKFid * 2 + 4);
            g2o::HyperGraph::Vertex *VS    = optimizer.vertex(maxKFid * 2 + 5);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                spdlog::error("Unable to build inertial edge: required graph vertex is missing");

                continue;
            }
            edge_inertial_gs *ei = new edge_inertial_gs(pKFi->imu_preintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));

            vpei.push_back(ei);

            vppUsedKF.push_back(make_pair(pKFi->previous_key_frame, pKFi));
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(its);

    // Recover optimized data
    // Biases
    VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(maxKFid * 2 + 2));
    VA = static_cast<vertex_acc_bias *>(optimizer.vertex(maxKFid * 2 + 3));
    vector_6d vb;
    vb << VG->estimate(), VA->estimate();
    bg << VG->estimate();
    ba << VA->estimate();

    imu::bias b(vb[3], vb[4], vb[5], vb[0], vb[1], vb[2]);

    // Keyframes velocities and biases
    const int N = vpKFs.size();
    for (size_t i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }

        vertex_velocity *VV = static_cast<vertex_velocity *>(optimizer.vertex(maxKFid + (pKFi->id) + 1));
        Eigen::Vector3d  Vw = VV->estimate();
        pKFi->set_velocity(Vw.cast<float>());

        if ((pKFi->get_gyro_bias() - bg.cast<float>()).norm() > 0.01)
        {
            pKFi->set_new_bias(b);
            if (pKFi->imu_preintegrated)
            {
                pKFi->imu_preintegrated->reintegrate();
            }
        }
        else
        {
            pKFi->set_new_bias(b);
        }
    }
}

void optimizer::inertial_optimization(camera_view cameras, map *pMap, Eigen::Matrix3d &Rwg, double &scale)
{
    int                       its     = 10;
    long unsigned int         maxKFid = pMap->get_max_key_frame_id();
    const vector<key_frame *> vpKFs   = pMap->get_all_key_frames();

    // Setup optimizer
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmGaussNewton>(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set key_frame vertices (all variables are fixed)
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];
        if (pKFi->id > maxKFid)
        {
            continue;
        }
        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        vertex_velocity *VV = new vertex_velocity(pKFi);
        VV->setId(maxKFid + 1 + (pKFi->id));
        VV->setFixed(true);
        optimizer.addVertex(VV);

        // Vertex of fixed biases
        vertex_gyro_bias *VG = new vertex_gyro_bias(vpKFs.front());
        VG->setId(2 * (maxKFid + 1) + (pKFi->id));
        VG->setFixed(true);
        optimizer.addVertex(VG);
        vertex_acc_bias *VA = new vertex_acc_bias(vpKFs.front());
        VA->setId(3 * (maxKFid + 1) + (pKFi->id));
        VA->setFixed(true);
        optimizer.addVertex(VA);
    }

    // Gravity and scale
    vertex_gravity_direction *VGDir = new vertex_gravity_direction(Rwg);
    VGDir->setId(4 * (maxKFid + 1));
    VGDir->setFixed(false);
    optimizer.addVertex(VGDir);
    vertex_scale *VS = new vertex_scale(scale);
    VS->setId(4 * (maxKFid + 1) + 1);
    VS->setFixed(false);
    optimizer.addVertex(VS);

    // Graph edges
    int count_edges = 0;
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];

        if (pKFi->previous_key_frame && pKFi->id <= maxKFid)
        {
            if (pKFi->is_bad() || pKFi->previous_key_frame->id > maxKFid)
            {
                continue;
            }

            g2o::HyperGraph::Vertex *VP1   = optimizer.vertex(pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VV1   = optimizer.vertex((maxKFid + 1) + pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VP2   = optimizer.vertex(pKFi->id);
            g2o::HyperGraph::Vertex *VV2   = optimizer.vertex((maxKFid + 1) + pKFi->id);
            g2o::HyperGraph::Vertex *VG    = optimizer.vertex(2 * (maxKFid + 1) + pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VA    = optimizer.vertex(3 * (maxKFid + 1) + pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VGDir = optimizer.vertex(4 * (maxKFid + 1));
            g2o::HyperGraph::Vertex *VS    = optimizer.vertex(4 * (maxKFid + 1) + 1);
            if (!VP1 || !VV1 || !VG || !VA || !VP2 || !VV2 || !VGDir || !VS)
            {
                spdlog::debug("Error" + to_string(VP1->id()) + ", " + to_string(VV1->id()) + ", " + to_string(VG->id()) + ", " + to_string(VA->id()) + ", " + to_string(VP2->id()) + ", " +
                              to_string(VV2->id()) + ", " + to_string(VGDir->id()) + ", " + to_string(VS->id()));

                continue;
            }
            count_edges++;
            edge_inertial_gs *ei = new edge_inertial_gs(pKFi->imu_preintegrated);
            ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            ei->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG));
            ei->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA));
            ei->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            ei->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));
            ei->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VGDir));
            ei->setVertex(7, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VS));
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            ei->setRobustKernel(rk);
            rk->setDelta(1.f);
            optimizer.addEdge(ei);
        }
    }

    // Compute error for different scales
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    [[maybe_unused]] float err = optimizer.activeRobustChi2();
    optimizer.optimize(its);
    optimizer.computeActiveErrors();
    [[maybe_unused]] float err_end = optimizer.activeRobustChi2();
    // Recover optimized data
    scale = VS->estimate();
    Rwg   = VGDir->estimate().rotation_world_gravity;
}

void optimizer::local_bundle_adjustment(camera_view cameras, key_frame *pMainKF, vector<key_frame *> vpAdjustKF, vector<key_frame *> vpFixedKF, bool *pbStopFlag)
{
    [[maybe_unused]] bool bShowImages = false;

    vector<map_point *> vpMPs;

    g2o::SparseOptimizer                    optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = MakeBlockSolver<g2o::BlockSolver_6_3>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);
    optimizer.setAlgorithm(solver);

    optimizer.setVerbose(false);

    if (pbStopFlag)
    {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    long unsigned int maxKFid = 0;
    set<key_frame *>  spKeyFrameBA;

    map *pCurrentMap = pMainKF->get_map();

    // Set fixed key_frame vertices
    int numInsertedPoints = 0;
    for (key_frame *pKFi : vpFixedKF)
    {
        if (pKFi->is_bad() || pKFi->get_map() != pCurrentMap)
        {
            spdlog::debug("ERROR LBA: KF is bad or is not in the current map");
            continue;
        }

        pKFi->local_ba_merge_key_frame_id = pMainKF->id;

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float>    Tcw  = pKFi->get_pose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->id);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->id > maxKFid)
        {
            maxKFid = pKFi->id;
        }

        set<map_point *> spViewMPs = pKFi->get_map_points();
        for (map_point *pMPi : spViewMPs)
        {
            if (pMPi)
            {
                if (!pMPi->is_bad() && pMPi->get_map() == pCurrentMap)
                {

                    if (pMPi->local_ba_merge_key_frame_id != pMainKF->id)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->local_ba_merge_key_frame_id = pMainKF->id;
                        numInsertedPoints++;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    // Set non fixed Keyframe vertices
    set<key_frame *> spAdjustKF(vpAdjustKF.begin(), vpAdjustKF.end());
    numInsertedPoints = 0;
    for (key_frame *pKFi : vpAdjustKF)
    {
        if (pKFi->is_bad() || pKFi->get_map() != pCurrentMap)
        {
            continue;
        }

        pKFi->local_ba_merge_key_frame_id = pMainKF->id;

        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        Sophus::SE3<float>    Tcw  = pKFi->get_pose();
        vSE3->setEstimate(g2o::SE3Quat(Tcw.unit_quaternion().cast<double>(), Tcw.translation().cast<double>()));
        vSE3->setId(pKFi->id);
        optimizer.addVertex(vSE3);
        if (pKFi->id > maxKFid)
        {
            maxKFid = pKFi->id;
        }

        set<map_point *> spViewMPs = pKFi->get_map_points();
        for (map_point *pMPi : spViewMPs)
        {
            if (pMPi)
            {
                if (!pMPi->is_bad() && pMPi->get_map() == pCurrentMap)
                {
                    if (pMPi->local_ba_merge_key_frame_id != pMainKF->id)
                    {
                        vpMPs.push_back(pMPi);
                        pMPi->local_ba_merge_key_frame_id = pMainKF->id;
                        numInsertedPoints++;
                    }
                }
            }
        }

        spKeyFrameBA.insert(pKFi);
    }

    const int nExpectedSize = (vpAdjustKF.size() + vpFixedKF.size()) * vpMPs.size();

    vector<ORB_SLAM3::edge_se3_project_xyz *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set map_point vertices
    std::map<key_frame *, int> mpObsKFs;
    std::map<key_frame *, int> mpObsFinalKFs;
    std::map<map_point *, int> mpObsMPs;
    for (unsigned int i = 0; std::cmp_less(i, vpMPs.size()); ++i)
    {
        map_point *pMPi = vpMPs[i];
        if (pMPi->is_bad())
        {
            continue;
        }

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMPi->get_world_position().cast<double>());
        const int id = pMPi->id + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<key_frame *, tuple<int, int>> observations = pMPi->get_observations();
        int                                          nEdges       = 0;
        // SET EDGES
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            key_frame *pKF = mit->first;
            if (pKF->is_bad() || pKF->id > maxKFid || pKF->local_ba_merge_key_frame_id != pMainKF->id || !pKF->get_map_point(get<0>(mit->second)))
            {
                continue;
            }

            nEdges++;

            const auto &kpUn = pKF->keypoints[get<0>(mit->second)];

            if (pKF->right_coordinates[get<0>(mit->second)] < 0) // Monocular
            {
                mpObsMPs[pMPi]++;
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                ORB_SLAM3::edge_se3_project_xyz *e = new ORB_SLAM3::edge_se3_project_xyz(*cameras[0]);

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->id)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->inverse_level_sigma_squared[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKF);
                vpMapPointEdgeMono.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsMPs[pMPi] += 2;
                Eigen::Matrix<double, 3, 1> obs;
                const float                 kp_ur = pKF->right_coordinates[get<0>(mit->second)];
                obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->id)));
                e->setMeasurement(obs);
                const float    &invSigma2 = pKF->inverse_level_sigma_squared[kpUn.octave];
                Eigen::Matrix3d Info      = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber3D);

                e->fx = cameras[0]->get_parameter(0);
                e->fy = cameras[0]->get_parameter(1);
                e->cx = cameras[0]->get_parameter(2);
                e->cy = cameras[0]->get_parameter(3);
                e->bf = cameras[0]->baseline_fx();

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKF);
                vpMapPointEdgeStereo.push_back(pMPi);

                mpObsKFs[pKF]++;
            }
        }
    }

    if (pbStopFlag)
    {
        if (*pbStopFlag)
        {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
    {
        if (*pbStopFlag)
        {
            bDoMore = false;
        }
    }

    std::map<unsigned long int, int> mWrongObsKF;
    if (bDoMore)
    {
        // Check inlier observations
        int badMonoMP = 0, badStereoMP = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::edge_se3_project_xyz *e   = vpEdgesMono[i];
            map_point                       *pMP = vpMapPointEdgeMono[i];

            if (pMP->is_bad())
            {
                continue;
            }

            if (e->chi2() > 5.991 || !e->is_depth_positive())
            {
                e->setLevel(1);
                badMonoMP++;
            }
            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e   = vpEdgesStereo[i];
            map_point                    *pMP = vpMapPointEdgeStereo[i];

            if (pMP->is_bad())
            {
                continue;
            }

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
                badStereoMP++;
            }

            e->setRobustKernel(0);
        }
        spdlog::debug("[BA]: First optimization(Huber), there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " stereo bad edges");

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<key_frame *, map_point *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());
    set<map_point *> spErasedMPs;
    set<key_frame *> spErasedKFs;

    // Check inlier observations
    int badMonoMP = 0, badStereoMP = 0;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        ORB_SLAM3::edge_se3_project_xyz *e   = vpEdgesMono[i];
        map_point                       *pMP = vpMapPointEdgeMono[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > 5.991 || !e->is_depth_positive())
        {
            key_frame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
            mWrongObsKF[pKFi->id]++;
            badMonoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e   = vpEdgesStereo[i];
        map_point                    *pMP = vpMapPointEdgeStereo[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            key_frame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
            mWrongObsKF[pKFi->id]++;
            badStereoMP++;

            spErasedMPs.insert(pMP);
            spErasedKFs.insert(pKFi);
        }
    }

    spdlog::trace("[BA]: Second optimization, there are " + to_string(badMonoMP) + " monocular and " + to_string(badStereoMP) + " sterero bad edges");

    // Get Map Mutex
    unique_lock<mutex> lock(pMainKF->get_map()->map_update_mutex);

    if (!vToErase.empty())
    {
        for (size_t i = 0; std::cmp_less(i, vToErase.size()); i++)
        {
            key_frame *pKFi = vToErase[i].first;
            map_point *pMPi = vToErase[i].second;
            pKFi->erase_map_point_match(pMPi);
            pMPi->erase_observation(pKFi);
        }
    }
    for (unsigned int i = 0; std::cmp_less(i, vpMPs.size()); ++i)
    {
        map_point *pMPi = vpMPs[i];
        if (pMPi->is_bad())
        {
            continue;
        }

        const std::map<key_frame *, tuple<int, int>> observations = pMPi->get_observations();
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            key_frame *pKF = mit->first;
            if (pKF->is_bad() || pKF->id > maxKFid || pKF->local_ba_key_frame_id != pMainKF->id || !pKF->get_map_point(get<0>(mit->second)))
            {
                continue;
            }

            if (pKF->right_coordinates[get<0>(mit->second)] < 0) // Monocular
            {
                mpObsFinalKFs[pKF]++;
            }
            else // RGBD or Stereo
            {
                mpObsFinalKFs[pKF]++;
            }
        }
    }

    // Recover optimized data
    // Keyframes
    for (key_frame *pKFi : vpAdjustKF)
    {
        if (pKFi->is_bad())
        {
            continue;
        }

        g2o::VertexSE3Expmap *vSE3    = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFi->id));
        g2o::SE3Quat          SE3quat = vSE3->estimate();
        Sophus::SE3f          Tiw(SE3quat.rotation().cast<float>(), SE3quat.translation().cast<float>());

        int                 numMonoBadPoints = 0, numMonoOptPoints = 0;
        int                 numStereoBadPoints = 0, numStereoOptPoints = 0;
        vector<map_point *> vpMonoMPsOpt, vpStereoMPsOpt;
        vector<map_point *> vpMonoMPsBad, vpStereoMPsBad;

        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            ORB_SLAM3::edge_se3_project_xyz *e       = vpEdgesMono[i];
            map_point                       *pMP     = vpMapPointEdgeMono[i];
            key_frame                       *pKFedge = vpEdgeKFMono[i];

            if (pKFi != pKFedge)
            {
                continue;
            }

            if (pMP->is_bad())
            {
                continue;
            }

            if (e->chi2() > 5.991 || !e->is_depth_positive())
            {
                numMonoBadPoints++;
                vpMonoMPsBad.push_back(pMP);
            }
            else
            {
                numMonoOptPoints++;
                vpMonoMPsOpt.push_back(pMP);
            }
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e       = vpEdgesStereo[i];
            map_point                    *pMP     = vpMapPointEdgeStereo[i];
            key_frame                    *pKFedge = vpEdgeKFMono[i];

            if (pKFi != pKFedge)
            {
                continue;
            }

            if (pMP->is_bad())
            {
                continue;
            }

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                numStereoBadPoints++;
                vpStereoMPsBad.push_back(pMP);
            }
            else
            {
                numStereoOptPoints++;
                vpStereoMPsOpt.push_back(pMP);
            }
        }

        pKFi->set_pose(Tiw);
    }

    // Points
    for (map_point *pMPi : vpMPs)
    {
        if (pMPi->is_bad())
        {
            continue;
        }

        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMPi->id + maxKFid + 1));
        pMPi->set_world_position(vPoint->estimate().cast<float>());
        pMPi->update_normal_and_depth(*cameras[0]);
    }
}

void optimizer::merge_inertial_ba(camera_view cameras, key_frame *pCurrKF, key_frame *pMergeKF, bool *pbStopFlag, map *pMap, loop_closing::key_frame_and_pose &corrPoses)
{
    const int           Nd      = 6;
    const unsigned long maxKFid = pCurrKF->id;

    vector<key_frame *> vpOptimizableKFs;
    vpOptimizableKFs.reserve(2 * Nd);

    // For cov KFS, inertial parameters are not optimized
    const int           maxCovKF = 30;
    vector<key_frame *> vpOptimizableCovKFs;
    vpOptimizableCovKFs.reserve(maxCovKF);

    // Add sliding window for current KF
    vpOptimizableKFs.push_back(pCurrKF);
    pCurrKF->local_ba_key_frame_id = pCurrKF->id;
    for (int i = 1; std::cmp_less(i, Nd); i++)
    {
        if (vpOptimizableKFs.back()->previous_key_frame)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->previous_key_frame);
            vpOptimizableKFs.back()->local_ba_key_frame_id = pCurrKF->id;
        }
        else
        {
            break;
        }
    }

    list<key_frame *> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->previous_key_frame)
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back()->previous_key_frame);
        vpOptimizableKFs.back()->previous_key_frame->local_ba_key_frame_id = pCurrKF->id;
    }
    else
    {
        vpOptimizableCovKFs.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Add temporal neighbours to merge KF (previous and next KFs)
    vpOptimizableKFs.push_back(pMergeKF);
    pMergeKF->local_ba_key_frame_id = pCurrKF->id;

    // Previous KFs
    for (int i = 1; i < (Nd / 2); i++)
    {
        if (vpOptimizableKFs.back()->previous_key_frame)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->previous_key_frame);
            vpOptimizableKFs.back()->local_ba_key_frame_id = pCurrKF->id;
        }
        else
        {
            break;
        }
    }

    // We fix just once the old map
    if (vpOptimizableKFs.back()->previous_key_frame)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->previous_key_frame);
        vpOptimizableKFs.back()->previous_key_frame->fixed_ba_key_frame_id = pCurrKF->id;
    }
    else
    {
        vpOptimizableKFs.back()->local_ba_key_frame_id = 0;
        vpOptimizableKFs.back()->fixed_ba_key_frame_id = pCurrKF->id;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Next KFs
    if (pMergeKF->next_key_frame)
    {
        vpOptimizableKFs.push_back(pMergeKF->next_key_frame);
        vpOptimizableKFs.back()->local_ba_key_frame_id = pCurrKF->id;
    }

    while (vpOptimizableKFs.size() < (2 * Nd))
    {
        if (vpOptimizableKFs.back()->next_key_frame)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->next_key_frame);
            vpOptimizableKFs.back()->local_ba_key_frame_id = pCurrKF->id;
        }
        else
        {
            break;
        }
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by optimizable keyframes
    list<map_point *>          lLocalMapPoints;
    std::map<map_point *, int> mLocalObs;
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        vector<map_point *> vpMPs = vpOptimizableKFs[i]->get_map_point_matches();
        for (vector<map_point *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            // Using mnBALocalForKF we avoid redundance here, one MP can not be added several times to lLocalMapPoints
            map_point *pMP = *vit;
            if (pMP)
            {
                if (!pMP->is_bad())
                {
                    if (pMP->local_ba_key_frame_id != pCurrKF->id)
                    {
                        mLocalObs[pMP] = 1;
                        lLocalMapPoints.push_back(pMP);
                        pMP->local_ba_key_frame_id = pCurrKF->id;
                    }
                    else
                    {
                        mLocalObs[pMP]++;
                    }
                }
            }
        }
    }

    std::vector<std::pair<map_point *, int>> pairs;
    pairs.reserve(mLocalObs.size());
    for (auto itr = mLocalObs.begin(); itr != mLocalObs.end(); ++itr)
    {
        pairs.push_back(*itr);
    }
    sort(pairs.begin(), pairs.end(), sortByVal);

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    int i = 0;
    for (vector<pair<map_point *, int>>::iterator lit = pairs.begin(), lend = pairs.end(); lit != lend; lit++, i++)
    {
        std::map<key_frame *, tuple<int, int>> observations = lit->first->get_observations();
        if (i >= maxCovKF)
        {
            break;
        }
        for (std::map<key_frame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (pKFi->local_ba_key_frame_id != pCurrKF->id && pKFi->fixed_ba_key_frame_id != pCurrKF->id) // If optimizable or already included...
            {
                pKFi->local_ba_key_frame_id = pCurrKF->id;
                if (!pKFi->is_bad())
                {
                    vpOptimizableCovKFs.push_back(pKFi);
                    break;
                }
            }
        }
    }

    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);

    solver->setUserLambdaInit(1e3);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Set Local key_frame vertices
    N = vpOptimizableKFs.size();
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpOptimizableKFs[i];

        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = new vertex_velocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->id) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            vertex_gyro_bias *VG = new vertex_gyro_bias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->id) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            vertex_acc_bias *VA = new vertex_acc_bias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->id) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Local cov keyframes vertices
    int Ncov = vpOptimizableCovKFs.size();
    for (int i = 0; std::cmp_less(i, Ncov); i++)
    {
        key_frame *pKFi = vpOptimizableCovKFs[i];

        vertex_pose *VP = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(false);
        optimizer.addVertex(VP);

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = new vertex_velocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->id) + 1);
            VV->setFixed(false);
            optimizer.addVertex(VV);
            vertex_gyro_bias *VG = new vertex_gyro_bias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->id) + 2);
            VG->setFixed(false);
            optimizer.addVertex(VG);
            vertex_acc_bias *VA = new vertex_acc_bias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->id) + 3);
            VA->setFixed(false);
            optimizer.addVertex(VA);
        }
    }

    // Set Fixed key_frame vertices
    for (list<key_frame *>::iterator lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
    {
        key_frame   *pKFi = *lit;
        vertex_pose *VP   = new vertex_pose(pKFi, cameras);
        VP->setId(pKFi->id);
        VP->setFixed(true);
        optimizer.addVertex(VP);

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = new vertex_velocity(pKFi);
            VV->setId(maxKFid + 3 * (pKFi->id) + 1);
            VV->setFixed(true);
            optimizer.addVertex(VV);
            vertex_gyro_bias *VG = new vertex_gyro_bias(pKFi);
            VG->setId(maxKFid + 3 * (pKFi->id) + 2);
            VG->setFixed(true);
            optimizer.addVertex(VG);
            vertex_acc_bias *VA = new vertex_acc_bias(pKFi);
            VA->setId(maxKFid + 3 * (pKFi->id) + 3);
            VA->setFixed(true);
            optimizer.addVertex(VA);
        }
    }

    // Create intertial constraints
    vector<edge_inertial *> vei(N, (edge_inertial *)NULL);
    vector<edge_gyro_rw *>  vegr(N, (edge_gyro_rw *)NULL);
    vector<edge_acc_rw *>   vear(N, (edge_acc_rw *)NULL);
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpOptimizableKFs[i];

        if (!pKFi->previous_key_frame)
        {
            spdlog::debug("NOT INERTIAL LINK TO PREVIOUS FRAME!!!!");
            continue;
        }
        if (pKFi->has_imu && pKFi->previous_key_frame->has_imu && pKFi->imu_preintegrated)
        {
            pKFi->imu_preintegrated->set_new_bias(pKFi->previous_key_frame->get_imu_bias());
            g2o::HyperGraph::Vertex *VP1 = optimizer.vertex(pKFi->previous_key_frame->id);
            g2o::HyperGraph::Vertex *VV1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 1);
            g2o::HyperGraph::Vertex *VG1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 2);
            g2o::HyperGraph::Vertex *VA1 = optimizer.vertex(maxKFid + 3 * (pKFi->previous_key_frame->id) + 3);
            g2o::HyperGraph::Vertex *VP2 = optimizer.vertex(pKFi->id);
            g2o::HyperGraph::Vertex *VV2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1);
            g2o::HyperGraph::Vertex *VG2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2);
            g2o::HyperGraph::Vertex *VA2 = optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3);

            if (!VP1 || !VV1 || !VG1 || !VA1 || !VP2 || !VV2 || !VG2 || !VA2)
            {
                spdlog::error("Unable to build inertial edge: required graph vertex is missing");
                continue;
            }

            vei[i] = new edge_inertial(pKFi->imu_preintegrated);

            vei[i]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP1));
            vei[i]->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV1));
            vei[i]->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VG1));
            vei[i]->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VA1));
            vei[i]->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VP2));
            vei[i]->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(VV2));

            // TODO Uncomment
            g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
            vei[i]->setRobustKernel(rki);
            rki->setDelta(sqrt(16.92));
            optimizer.addEdge(vei[i]);

            vegr[i] = new edge_gyro_rw();
            vegr[i]->setVertex(0, VG1);
            vegr[i]->setVertex(1, VG2);
            Eigen::Matrix3d InfoG = pKFi->imu_preintegrated->covariance.block<3, 3>(9, 9).cast<double>().inverse();
            vegr[i]->setInformation(InfoG);
            optimizer.addEdge(vegr[i]);

            vear[i] = new edge_acc_rw();
            vear[i]->setVertex(0, VA1);
            vear[i]->setVertex(1, VA2);
            Eigen::Matrix3d InfoA = pKFi->imu_preintegrated->covariance.block<3, 3>(12, 12).cast<double>().inverse();
            vear[i]->setInformation(InfoA);
            optimizer.addEdge(vear[i]);
        }
        else
        {
            spdlog::debug("ERROR building inertial edge");
        }
    }

    spdlog::debug("end inserting inertial edges");

    // Set map_point vertices
    const int nExpectedSize = (N + Ncov + lFixedKeyFrames.size()) * lLocalMapPoints.size();

    // Mono
    vector<edge_mono *> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Stereo
    vector<edge_stereo *> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<key_frame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<map_point *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono   = sqrt(5.991);
    const float chi2Mono2     = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2   = 7.815;

    const unsigned long iniMPid = maxKFid * 5;

    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map_point *pMP = *lit;
        if (!pMP)
        {
            continue;
        }

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_world_position().cast<double>());

        unsigned long id = pMP->id + iniMPid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const std::map<key_frame *, tuple<int, int>> observations = pMP->get_observations();

        // Create visual constraints
        for (std::map<key_frame *, tuple<int, int>>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            key_frame *pKFi = mit->first;

            if (!pKFi)
            {
                continue;
            }

            if ((pKFi->local_ba_key_frame_id != pCurrKF->id) && (pKFi->fixed_ba_key_frame_id != pCurrKF->id))
            {
                continue;
            }

            if (pKFi->id > maxKFid)
            {
                continue;
            }

            if (optimizer.vertex(id) == NULL || optimizer.vertex(pKFi->id) == NULL)
            {
                continue;
            }

            if (!pKFi->is_bad())
            {
                const auto &kpUn = pKFi->keypoints[get<0>(mit->second)];

                if (pKFi->right_coordinates[get<0>(mit->second)] < 0) // Monocular observation
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y;

                    edge_mono *e = new edge_mono();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // stereo observation
                {
                    const float                 kp_ur = pKFi->right_coordinates[get<0>(mit->second)];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                    edge_stereo *e = new edge_stereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->id)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->inverse_level_sigma_squared[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if (pbStopFlag)
    {
        optimizer.setForceStopFlag(pbStopFlag);
    }

    if (pbStopFlag)
    {
        if (*pbStopFlag)
        {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(8);

    vector<pair<key_frame *, map_point *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    // Mono
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        edge_mono *e   = vpEdgesMono[i];
        map_point *pMP = vpMapPointEdgeMono[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > chi2Mono2)
        {
            key_frame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Stereo
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        edge_stereo *e   = vpEdgesStereo[i];
        map_point   *pMP = vpMapPointEdgeStereo[i];

        if (pMP->is_bad())
        {
            continue;
        }

        if (e->chi2() > chi2Stereo2)
        {
            key_frame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->map_update_mutex);
    if (!vToErase.empty())
    {
        for (size_t i = 0; std::cmp_less(i, vToErase.size()); i++)
        {
            key_frame *pKFi = vToErase[i].first;
            map_point *pMPi = vToErase[i].second;
            pKFi->erase_map_point_match(pMPi);
            pMPi->erase_observation(pKFi);
        }
    }

    // Recover optimized data
    // Keyframes
    for (int i = 0; std::cmp_less(i, N); i++)
    {
        key_frame *pKFi = vpOptimizableKFs[i];

        vertex_pose *VP = static_cast<vertex_pose *>(optimizer.vertex(pKFi->id));
        Sophus::SE3f Tcw(VP->estimate().rotation_camera_world[0].cast<float>(), VP->estimate().translation_camera_world[0].cast<float>());
        pKFi->set_pose(Tcw);

        Sophus::SE3d Tiw = pKFi->get_pose().cast<double>();
        g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
        corrPoses[pKFi] = g2oSiw;

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = static_cast<vertex_velocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1));
            pKFi->set_velocity(VV->estimate().cast<float>());
            vertex_gyro_bias *VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2));
            vertex_acc_bias  *VA = static_cast<vertex_acc_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3));
            vector_6d         b;
            b << VG->estimate(), VA->estimate();
            pKFi->set_new_bias(imu::bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    for (int i = 0; std::cmp_less(i, Ncov); i++)
    {
        key_frame *pKFi = vpOptimizableCovKFs[i];

        vertex_pose *VP = static_cast<vertex_pose *>(optimizer.vertex(pKFi->id));
        Sophus::SE3f Tcw(VP->estimate().rotation_camera_world[0].cast<float>(), VP->estimate().translation_camera_world[0].cast<float>());
        pKFi->set_pose(Tcw);

        Sophus::SE3d Tiw = pKFi->get_pose().cast<double>();
        g2o::Sim3    g2oSiw(Tiw.unit_quaternion(), Tiw.translation(), 1.0);
        corrPoses[pKFi] = g2oSiw;

        if (pKFi->has_imu)
        {
            vertex_velocity *VV = static_cast<vertex_velocity *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 1));
            pKFi->set_velocity(VV->estimate().cast<float>());
            vertex_gyro_bias *VG = static_cast<vertex_gyro_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 2));
            vertex_acc_bias  *VA = static_cast<vertex_acc_bias *>(optimizer.vertex(maxKFid + 3 * (pKFi->id) + 3));
            vector_6d         b;
            b << VG->estimate(), VA->estimate();
            pKFi->set_new_bias(imu::bias(b[3], b[4], b[5], b[0], b[1], b[2]));
        }
    }

    // Points
    for (list<map_point *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map_point              *pMP    = *lit;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->id + iniMPid + 1));
        pMP->set_world_position(vPoint->estimate().cast<float>());
        pMP->update_normal_and_depth(*cameras[0]);
    }

    pMap->increase_change_index();
}

int optimizer::pose_inertial_optimization_last_key_frame(frame *pFrame, const camera_vector &cameras, const imu::calib &imu_calibration, bool bRecInit)
{
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmGaussNewton>(solver_ptr);
    optimizer.setVerbose(false);
    optimizer.setAlgorithm(solver);

    int nInitialMonoCorrespondences   = 0;
    int nInitialStereoCorrespondences = 0;
    int nInitialCorrespondences       = 0;

    // Set Frame vertex
    vertex_pose *VP = new vertex_pose(pFrame, cameras, imu_calibration);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    vertex_velocity *VV = new vertex_velocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    vertex_gyro_bias *VG = new vertex_gyro_bias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    vertex_acc_bias *VA = new vertex_acc_bias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set map_point vertices
    const int  N      = pFrame->feature_count;
    const int  Nleft  = static_cast<int>(pFrame->keypoints.size());
    const bool bRight = pFrame->is_fisheye_stereo;

    vector<edge_mono_only_pose *>   vpEdgesMono;
    vector<edge_stereo_only_pose *> vpEdgesStereo;
    vector<size_t>                  vnIndexEdgeMono;
    vector<size_t>                  vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono   = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(map_point::global_mutex);

        for (int i = 0; std::cmp_less(i, N); i++)
        {
            map_point *pMP = pFrame->map_points[i];
            if (pMP)
            {
                // Left monocular observation
                if ((!bRight && pFrame->right_coordinates[i] < 0) || std::cmp_less(i, Nleft))
                {
                    const auto &kpUn     = pFrame->keypoints[i];
                    const auto &position = std::cmp_less(i, Nleft) ? kpUn.pt : kpUn.pt_undistorted;

                    nInitialMonoCorrespondences++;
                    pFrame->outliers[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << position.x, position.y;

                    edge_mono_only_pose *e = new edge_mono_only_pose(pMP->get_world_position(), 0);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs);

                    const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if (!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->outliers[i] = false;

                    const auto                 &kpUn  = pFrame->keypoints[i];
                    const float                 kp_ur = pFrame->right_coordinates[i];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                    edge_stereo_only_pose *e = new edge_stereo_only_pose(pMP->get_world_position());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if (bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->outliers[i] = false;

                    const auto                 &kpUn = pFrame->right_keypoints[i - Nleft];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    edge_mono_only_pose *e = new edge_mono_only_pose(pMP->get_world_position(), 1);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs);

                    const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }
    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    key_frame   *pKF = pFrame->last_key_frame;
    vertex_pose *VPk = new vertex_pose(pKF, cameras);
    VPk->setId(4);
    VPk->setFixed(true);
    optimizer.addVertex(VPk);
    vertex_velocity *VVk = new vertex_velocity(pKF);
    VVk->setId(5);
    VVk->setFixed(true);
    optimizer.addVertex(VVk);
    vertex_gyro_bias *VGk = new vertex_gyro_bias(pKF);
    VGk->setId(6);
    VGk->setFixed(true);
    optimizer.addVertex(VGk);
    vertex_acc_bias *VAk = new vertex_acc_bias(pKF);
    VAk->setId(7);
    VAk->setFixed(true);
    optimizer.addVertex(VAk);

    edge_inertial *ei = new edge_inertial(pFrame->imu_preintegrated);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    edge_gyro_rw *egr = new edge_gyro_rw();
    egr->setVertex(0, VGk);
    egr->setVertex(1, VG);
    Eigen::Matrix3d InfoG = pFrame->imu_preintegrated->covariance.block<3, 3>(9, 9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    edge_acc_rw *ear = new edge_acc_rw();
    ear->setVertex(0, VAk);
    ear->setVertex(1, VA);
    Eigen::Matrix3d InfoA = pFrame->imu_preintegrated->covariance.block<3, 3>(12, 12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    float chi2Mono[4]   = {12, 7.5, 5.991, 5.991};
    float chi2Stereo[4] = {15.6, 9.8, 7.815, 7.815};

    int its[4] = {10, 10, 10, 10};

    int nBad           = 0;
    int nBadMono       = 0;
    int nBadStereo     = 0;
    int nInliersMono   = 0;
    int nInliersStereo = 0;
    int nInliers       = 0;
    for (size_t it = 0; it < 4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad            = 0;
        nBadMono        = 0;
        nBadStereo      = 0;
        nInliers        = 0;
        nInliersMono    = 0;
        nInliersStereo  = 0;
        float chi2close = 1.5 * chi2Mono[it];

        // For monocular observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            edge_mono_only_pose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2   = e->chi2();
            bool        bClose = pFrame->map_points[idx]->track_depth < 10.f;

            if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->is_depth_positive())
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->outliers[idx] = false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        // For stereo observations
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            edge_stereo_only_pose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1); // not included in next optimization
                nBadStereo++;
            }
            else
            {
                pFrame->outliers[idx] = false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad     = nBadMono + nBadStereo;

        if (optimizer.edges().size() < 10)
        {
            break;
        }
    }

    // If not too much tracks, recover not too bad points
    if ((nInliers < 30) && !bRecInit)
    {
        nBad                                 = 0;
        const float            chi2MonoOut   = 18.f;
        const float            chi2StereoOut = 24.f;
        edge_mono_only_pose   *e1;
        edge_stereo_only_pose *e2;
        for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1               = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2() < chi2MonoOut)
            {
                pFrame->outliers[idx] = false;
            }
            else
            {
                nBad++;
            }
        }
        for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2               = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2() < chi2StereoOut)
            {
                pFrame->outliers[idx] = false;
            }
            else
            {
                nBad++;
            }
        }
    }

    // Recover optimized pose, velocity and biases
    pFrame->set_imu_pose_velocity(VP->estimate().rotation_world_body.cast<float>(), VP->estimate().translation_world_body.cast<float>(), VV->estimate().cast<float>(), imu_calibration);
    vector_6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->imu_bias = imu::bias(b[3], b[4], b[5], b[0], b[1], b[2]);

    // Recover Hessian, marginalize keyFframe states and generate new prior for frame
    Eigen::Matrix<double, 15, 15> H;
    H.setZero();

    H.block<9, 9>(0, 0)   += ei->get_hessian_2();
    H.block<3, 3>(9, 9)   += egr->get_hessian_2();
    H.block<3, 3>(12, 12) += ear->get_hessian_2();

    int tot_in = 0, tot_out = 0;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        edge_mono_only_pose *e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if (!pFrame->outliers[idx])
        {
            H.block<6, 6>(0, 0) += e->get_hessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        edge_stereo_only_pose *e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if (!pFrame->outliers[idx])
        {
            H.block<6, 6>(0, 0) += e->get_hessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
    }

    pFrame->imu_constraint = new constraint_pose_imu(VP->estimate().rotation_world_body, VP->estimate().translation_world_body, VV->estimate(), VG->estimate(), VA->estimate(), H);

    return nInitialCorrespondences - nBad;
}

int optimizer::pose_inertial_optimization_last_frame(frame *pFrame, const camera_vector &cameras, const imu::calib &imu_calibration, bool bRecInit)
{
    g2o::SparseOptimizer                 optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmGaussNewton>(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences   = 0;
    int nInitialStereoCorrespondences = 0;
    int nInitialCorrespondences       = 0;

    // Set Current Frame vertex
    vertex_pose *VP = new vertex_pose(pFrame, cameras, imu_calibration);
    VP->setId(0);
    VP->setFixed(false);
    optimizer.addVertex(VP);
    vertex_velocity *VV = new vertex_velocity(pFrame);
    VV->setId(1);
    VV->setFixed(false);
    optimizer.addVertex(VV);
    vertex_gyro_bias *VG = new vertex_gyro_bias(pFrame);
    VG->setId(2);
    VG->setFixed(false);
    optimizer.addVertex(VG);
    vertex_acc_bias *VA = new vertex_acc_bias(pFrame);
    VA->setId(3);
    VA->setFixed(false);
    optimizer.addVertex(VA);

    // Set map_point vertices
    const int  N      = pFrame->feature_count;
    const int  Nleft  = static_cast<int>(pFrame->keypoints.size());
    const bool bRight = pFrame->is_fisheye_stereo;

    vector<edge_mono_only_pose *>   vpEdgesMono;
    vector<edge_stereo_only_pose *> vpEdgesStereo;
    vector<size_t>                  vnIndexEdgeMono;
    vector<size_t>                  vnIndexEdgeStereo;
    vpEdgesMono.reserve(N);
    vpEdgesStereo.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float thHuberMono   = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(map_point::global_mutex);

        for (int i = 0; std::cmp_less(i, N); i++)
        {
            map_point *pMP = pFrame->map_points[i];
            if (pMP)
            {
                // Left monocular observation
                if ((!bRight && pFrame->right_coordinates[i] < 0) || std::cmp_less(i, Nleft))
                {
                    const auto &kpUn     = pFrame->keypoints[i];
                    const auto &position = std::cmp_less(i, Nleft) ? kpUn.pt : kpUn.pt_undistorted;

                    nInitialMonoCorrespondences++;
                    pFrame->outliers[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << position.x, position.y;

                    edge_mono_only_pose *e = new edge_mono_only_pose(pMP->get_world_position(), 0);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs);

                    const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                // Stereo observation
                else if (!bRight)
                {
                    nInitialStereoCorrespondences++;
                    pFrame->outliers[i] = false;

                    const auto                 &kpUn  = pFrame->keypoints[i];
                    const float                 kp_ur = pFrame->right_coordinates[i];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt_undistorted.x, kpUn.pt_undistorted.y, kp_ur;

                    edge_stereo_only_pose *e = new edge_stereo_only_pose(pMP->get_world_position());

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs.head(2));

                    const float &invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }

                // Right monocular observation
                if (bRight && i >= Nleft)
                {
                    nInitialMonoCorrespondences++;
                    pFrame->outliers[i] = false;

                    const auto                 &kpUn = pFrame->right_keypoints[i - Nleft];
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    edge_mono_only_pose *e = new edge_mono_only_pose(pMP->get_world_position(), 1);

                    e->setVertex(0, VP);
                    e->setMeasurement(obs);

                    // Add here uncerteinty
                    const float unc2 = cameras[0]->uncertainty2(obs);

                    const float invSigma2 = pFrame->inverse_level_sigma_squared[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // Set Previous Frame Vertex
    frame *pFp = pFrame->previous_frame;

    vertex_pose *VPk = new vertex_pose(pFp, cameras, imu_calibration);
    VPk->setId(4);
    VPk->setFixed(false);
    optimizer.addVertex(VPk);
    vertex_velocity *VVk = new vertex_velocity(pFp);
    VVk->setId(5);
    VVk->setFixed(false);
    optimizer.addVertex(VVk);
    vertex_gyro_bias *VGk = new vertex_gyro_bias(pFp);
    VGk->setId(6);
    VGk->setFixed(false);
    optimizer.addVertex(VGk);
    vertex_acc_bias *VAk = new vertex_acc_bias(pFp);
    VAk->setId(7);
    VAk->setFixed(false);
    optimizer.addVertex(VAk);

    edge_inertial *ei = new edge_inertial(pFrame->frame_imu_preintegrated);

    ei->setVertex(0, VPk);
    ei->setVertex(1, VVk);
    ei->setVertex(2, VGk);
    ei->setVertex(3, VAk);
    ei->setVertex(4, VP);
    ei->setVertex(5, VV);
    optimizer.addEdge(ei);

    edge_gyro_rw *egr = new edge_gyro_rw();
    egr->setVertex(0, VGk);
    egr->setVertex(1, VG);
    Eigen::Matrix3d InfoG = pFrame->imu_preintegrated->covariance.block<3, 3>(9, 9).cast<double>().inverse();
    egr->setInformation(InfoG);
    optimizer.addEdge(egr);

    edge_acc_rw *ear = new edge_acc_rw();
    ear->setVertex(0, VAk);
    ear->setVertex(1, VA);
    Eigen::Matrix3d InfoA = pFrame->imu_preintegrated->covariance.block<3, 3>(12, 12).cast<double>().inverse();
    ear->setInformation(InfoA);
    optimizer.addEdge(ear);

    if (!pFp->imu_constraint)
    {
        spdlog::debug("pFp->imu_constraint does not exist!!!\nPrevious Frame " + to_string(pFp->id));
    }

    edge_prior_pose_imu *ep = new edge_prior_pose_imu(pFp->imu_constraint);

    ep->setVertex(0, VPk);
    ep->setVertex(1, VVk);
    ep->setVertex(2, VGk);
    ep->setVertex(3, VAk);
    g2o::RobustKernelHuber *rkp = new g2o::RobustKernelHuber;
    ep->setRobustKernel(rkp);
    rkp->setDelta(5);
    optimizer.addEdge(ep);

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]   = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {15.6f, 9.8f, 7.815f, 7.815f};
    const int   its[4]        = {10, 10, 10, 10};

    int nBad           = 0;
    int nBadMono       = 0;
    int nBadStereo     = 0;
    int nInliersMono   = 0;
    int nInliersStereo = 0;
    int nInliers       = 0;
    for (size_t it = 0; it < 4; it++)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad            = 0;
        nBadMono        = 0;
        nBadStereo      = 0;
        nInliers        = 0;
        nInliersMono    = 0;
        nInliersStereo  = 0;
        float chi2close = 1.5 * chi2Mono[it];

        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            edge_mono_only_pose *e = vpEdgesMono[i];

            const size_t idx    = vnIndexEdgeMono[i];
            bool         bClose = pFrame->map_points[idx]->track_depth < 10.f;

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if ((chi2 > chi2Mono[it] && !bClose) || (bClose && chi2 > chi2close) || !e->is_depth_positive())
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->outliers[idx] = false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            edge_stereo_only_pose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->outliers[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->outliers[idx] = true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->outliers[idx] = false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if (it == 2)
            {
                e->setRobustKernel(0);
            }
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad     = nBadMono + nBadStereo;

        if (optimizer.edges().size() < 10)
        {
            break;
        }
    }

    if ((nInliers < 30) && !bRecInit)
    {
        nBad                                 = 0;
        const float            chi2MonoOut   = 18.f;
        const float            chi2StereoOut = 24.f;
        edge_mono_only_pose   *e1;
        edge_stereo_only_pose *e2;
        for (size_t i = 0, iend = vnIndexEdgeMono.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeMono[i];
            e1               = vpEdgesMono[i];
            e1->computeError();
            if (e1->chi2() < chi2MonoOut)
            {
                pFrame->outliers[idx] = false;
            }
            else
            {
                nBad++;
            }
        }
        for (size_t i = 0, iend = vnIndexEdgeStereo.size(); i < iend; i++)
        {
            const size_t idx = vnIndexEdgeStereo[i];
            e2               = vpEdgesStereo[i];
            e2->computeError();
            if (e2->chi2() < chi2StereoOut)
            {
                pFrame->outliers[idx] = false;
            }
            else
            {
                nBad++;
            }
        }
    }

    nInliers = nInliersMono + nInliersStereo;

    // Recover optimized pose, velocity and biases
    pFrame->set_imu_pose_velocity(VP->estimate().rotation_world_body.cast<float>(), VP->estimate().translation_world_body.cast<float>(), VV->estimate().cast<float>(), imu_calibration);
    vector_6d b;
    b << VG->estimate(), VA->estimate();
    pFrame->imu_bias = imu::bias(b[3], b[4], b[5], b[0], b[1], b[2]);

    // Recover Hessian, marginalize previous frame states and generate new prior for frame
    Eigen::Matrix<double, 30, 30> H;
    H.setZero();

    H.block<24, 24>(0, 0) += ei->get_hessian();

    Eigen::Matrix<double, 6, 6> Hgr  = egr->get_hessian();
    H.block<3, 3>(9, 9)             += Hgr.block<3, 3>(0, 0);
    H.block<3, 3>(9, 24)            += Hgr.block<3, 3>(0, 3);
    H.block<3, 3>(24, 9)            += Hgr.block<3, 3>(3, 0);
    H.block<3, 3>(24, 24)           += Hgr.block<3, 3>(3, 3);

    Eigen::Matrix<double, 6, 6> Har  = ear->get_hessian();
    H.block<3, 3>(12, 12)           += Har.block<3, 3>(0, 0);
    H.block<3, 3>(12, 27)           += Har.block<3, 3>(0, 3);
    H.block<3, 3>(27, 12)           += Har.block<3, 3>(3, 0);
    H.block<3, 3>(27, 27)           += Har.block<3, 3>(3, 3);

    H.block<15, 15>(0, 0) += ep->get_hessian();

    int tot_in = 0, tot_out = 0;
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        edge_mono_only_pose *e = vpEdgesMono[i];

        const size_t idx = vnIndexEdgeMono[i];

        if (!pFrame->outliers[idx])
        {
            H.block<6, 6>(15, 15) += e->get_hessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        edge_stereo_only_pose *e = vpEdgesStereo[i];

        const size_t idx = vnIndexEdgeStereo[i];

        if (!pFrame->outliers[idx])
        {
            H.block<6, 6>(15, 15) += e->get_hessian();
            tot_in++;
        }
        else
        {
            tot_out++;
        }
    }

    H = marginalize(H, 0, 14);

    pFrame->imu_constraint =
        new constraint_pose_imu(VP->estimate().rotation_world_body, VP->estimate().translation_world_body, VV->estimate(), VG->estimate(), VA->estimate(), H.block<15, 15>(15, 15));
    delete pFp->imu_constraint;
    pFp->imu_constraint = NULL;

    return nInitialCorrespondences - nBad;
}

void optimizer::optimize_essential_graph_4dof(camera_view                                    cameras,
                                              map                                           *pMap,
                                              key_frame                                     *pLoopKF,
                                              key_frame                                     *pCurKF,
                                              const loop_closing::key_frame_and_pose        &NonCorrectedSim3,
                                              const loop_closing::key_frame_and_pose        &CorrectedSim3,
                                              const std::map<key_frame *, set<key_frame *>> &LoopConnections)
{

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolverX::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX                   *solver_ptr   = MakeBlockSolver<g2o::BlockSolverX>(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = MakeOptimizationAlgorithm<g2o::OptimizationAlgorithmLevenberg>(solver_ptr);

    optimizer.setAlgorithm(solver);

    const vector<key_frame *> vpKFs = pMap->get_all_key_frames();
    const vector<map_point *> vpMPs = pMap->get_all_map_points();

    const unsigned int nMaxKFid = pMap->get_max_key_frame_id();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);

    vector<vertex_pose_4dof *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;
    // Set key_frame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        key_frame *pKF = vpKFs[i];
        if (pKF->is_bad())
        {
            continue;
        }

        vertex_pose_4dof *V4DoF;

        const int nIDi = pKF->id;

        loop_closing::key_frame_and_pose::const_iterator it = CorrectedSim3.find(pKF);

        if (it != CorrectedSim3.end())
        {
            vScw[nIDi]          = it->second;
            const g2o::Sim3 Swc = it->second.inverse();
            Eigen::Matrix3d Rwc = Swc.rotation().toRotationMatrix();
            Eigen::Vector3d twc = Swc.translation();
            V4DoF               = new vertex_pose_4dof(Rwc, twc, pKF, cameras);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->get_pose().cast<double>();
            g2o::Sim3    Siw(Tcw.unit_quaternion(), Tcw.translation(), 1.0);

            vScw[nIDi] = Siw;
            V4DoF      = new vertex_pose_4dof(pKF, cameras);
        }

        if (pKF == pLoopKF)
        {
            V4DoF->setFixed(true);
        }

        V4DoF->setId(nIDi);
        V4DoF->setMarginalized(false);

        optimizer.addVertex(V4DoF);
        vpVertices[nIDi] = V4DoF;
    }
    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

    // Edge used in posegraph has still 6Dof, even if updates of camera poses are just in 4DoF
    Eigen::Matrix<double, 6, 6> matLambda = Eigen::Matrix<double, 6, 6>::Identity();
    matLambda(0, 0)                       = 1e3;
    matLambda(1, 1)                       = 1e3;
    matLambda(0, 0)                       = 1e3;

    // Set Loop edges
    [[maybe_unused]] edge_4dof *e_loop;
    for (std::map<key_frame *, set<key_frame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
    {
        key_frame              *pKF           = mit->first;
        const long unsigned int nIDi          = pKF->id;
        const set<key_frame *> &spConnections = mit->second;
        const g2o::Sim3         Siw           = vScw[nIDi];

        for (set<key_frame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->id;
            if ((nIDi != pCurKF->id || nIDj != pLoopKF->id) && pKF->get_weight(*sit) < minFeat)
            {
                continue;
            }

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sij = Siw * Sjw.inverse();
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3)             = 1.;

            edge_4dof *e = new edge_4dof(Tij);
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));

            e->information() = matLambda;
            e_loop           = e;
            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // 1. Set normal edges
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        key_frame *pKF = vpKFs[i];

        const int nIDi = pKF->id;

        g2o::Sim3 Siw;

        // Use noncorrected poses for posegraph edges
        loop_closing::key_frame_and_pose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if (iti != NonCorrectedSim3.end())
        {
            Siw = iti->second;
        }
        else
        {
            Siw = vScw[nIDi];
        }

        // 1.1.0 Spanning tree edge
        key_frame *pParentKF = static_cast<key_frame *>(NULL);
        if (pParentKF)
        {
            int nIDj = pParentKF->id;

            g2o::Sim3 Swj;

            loop_closing::key_frame_and_pose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if (itj != NonCorrectedSim3.end())
            {
                Swj = (itj->second).inverse();
            }
            else
            {
                Swj = vScw[nIDj].inverse();
            }

            g2o::Sim3       Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3)             = 1.;

            edge_4dof *e = new edge_4dof(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.1.1 Inertial edges
        key_frame *prevKF = pKF->previous_key_frame;
        if (prevKF)
        {
            int nIDj = prevKF->id;

            g2o::Sim3 Swj;

            loop_closing::key_frame_and_pose::const_iterator itj = NonCorrectedSim3.find(prevKF);

            if (itj != NonCorrectedSim3.end())
            {
                Swj = (itj->second).inverse();
            }
            else
            {
                Swj = vScw[nIDj].inverse();
            }

            g2o::Sim3       Sij = Siw * Swj;
            Eigen::Matrix4d Tij;
            Tij.block<3, 3>(0, 0) = Sij.rotation().toRotationMatrix();
            Tij.block<3, 1>(0, 3) = Sij.translation();
            Tij(3, 3)             = 1.;

            edge_4dof *e = new edge_4dof(Tij);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // 1.2 Loop edges
        const set<key_frame *> sLoopEdges = pKF->get_loop_edges();
        for (set<key_frame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            key_frame *pLKF = *sit;
            if (pLKF->id < pKF->id)
            {
                g2o::Sim3 Swl;

                loop_closing::key_frame_and_pose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                {
                    Swl = itl->second.inverse();
                }
                else
                {
                    Swl = vScw[pLKF->id].inverse();
                }

                g2o::Sim3       Sil = Siw * Swl;
                Eigen::Matrix4d Til;
                Til.block<3, 3>(0, 0) = Sil.rotation().toRotationMatrix();
                Til.block<3, 1>(0, 3) = Sil.translation();
                Til(3, 3)             = 1.;

                edge_4dof *e = new edge_4dof(Til);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->id)));
                e->information() = matLambda;
                optimizer.addEdge(e);
            }
        }

        // 1.3 Covisibility graph edges
        const vector<key_frame *> vpConnectedKFs = pKF->get_covisibles_by_weight(minFeat);
        for (vector<key_frame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            key_frame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && pKFn != prevKF && pKFn != pKF->next_key_frame && !pKF->has_child(pKFn) && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->is_bad() && pKFn->id < pKF->id)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->id, pKFn->id), max(pKF->id, pKFn->id))))
                    {
                        continue;
                    }

                    g2o::Sim3 Swn;

                    loop_closing::key_frame_and_pose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                    {
                        Swn = itn->second.inverse();
                    }
                    else
                    {
                        Swn = vScw[pKFn->id].inverse();
                    }

                    g2o::Sim3       Sin = Siw * Swn;
                    Eigen::Matrix4d Tin;
                    Tin.block<3, 3>(0, 0) = Sin.rotation().toRotationMatrix();
                    Tin.block<3, 1>(0, 3) = Sin.translation();
                    Tin(3, 3)             = 1.;
                    edge_4dof *e          = new edge_4dof(Tin);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->id)));
                    e->information() = matLambda;
                    optimizer.addEdge(e);
                }
            }
        }
    }

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->map_update_mutex);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; std::cmp_less(i, vpKFs.size()); i++)
    {
        key_frame *pKFi = vpKFs[i];

        const int nIDi = pKFi->id;

        vertex_pose_4dof *Vi = static_cast<vertex_pose_4dof *>(optimizer.vertex(nIDi));
        Eigen::Matrix3d   Ri = Vi->estimate().rotation_camera_world[0];
        Eigen::Vector3d   ti = Vi->estimate().translation_camera_world[0];

        g2o::Sim3 CorrectedSiw = g2o::Sim3(Ri, ti, 1.);
        vCorrectedSwc[nIDi]    = CorrectedSiw.inverse();

        Sophus::SE3d Tiw(CorrectedSiw.rotation(), CorrectedSiw.translation());
        pKFi->set_pose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        map_point *pMP = vpMPs[i];

        if (pMP->is_bad())
        {
            continue;
        }

        int nIDr;

        key_frame *pRefKF = pMP->get_reference_key_frame();
        nIDr              = pRefKF->id;

        g2o::Sim3 Srw          = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double, 3, 1> eigP3Dw          = pMP->get_world_position().cast<double>();
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->set_world_position(eigCorrectedP3Dw.cast<float>());

        pMP->update_normal_and_depth(*cameras[0]);
    }
    pMap->increase_change_index();
}

} // namespace ORB_SLAM3
