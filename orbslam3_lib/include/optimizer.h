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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "g2o_compat.h"
#include "frame.h"
#include "key_frame.h"
#include "loop_closing.h"
#include "map.h"
#include "map_point.h"

#include <math.h>

#include <map>
#include <memory>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

namespace ORB_SLAM3
{

class loop_closing;

class optimizer
{
  public:
    void static bundle_adjustment(
        const std::vector<key_frame *> &vpKF, const std::vector<map_point *> &vpMP, int nIterations = 5, bool *pbStopFlag = NULL, const unsigned long nLoopKF = 0, const bool bRobust = true);
    void static global_bundle_adjustment(map *pMap, int nIterations = 5, bool *pbStopFlag = NULL, const unsigned long nLoopKF = 0, const bool bRobust = true);
    void static full_inertial_ba(map                *pMap,
                                 int                 its,
                                 const bool          bFixLocal  = false,
                                 const unsigned long nLoopKF    = 0,
                                 bool               *pbStopFlag = NULL,
                                 bool                bInit      = false,
                                 float               priorG     = 1e2,
                                 float               priorA     = 1e6,
                                 Eigen::VectorXd    *vSingVal   = NULL,
                                 bool               *bHess      = NULL);

    void static local_bundle_adjustment(key_frame *pKF, bool *pbStopFlag, map *pMap, int &num_fixedKF, int &num_OptKF, int &num_MPs, int &num_edges);

    int static pose_optimization(frame *pFrame, const camera_vector &cameras);
    int static pose_inertial_optimization_last_key_frame(frame *pFrame, const camera_vector &cameras, const imu::calib &imu_calibration, bool bRecInit = false);
    int static pose_inertial_optimization_last_frame(frame *pFrame, const camera_vector &cameras, const imu::calib &imu_calibration, bool bRecInit = false);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static optimize_essential_graph(map                                           *pMap,
                                         key_frame                                     *pLoopKF,
                                         key_frame                                     *pCurKF,
                                         const loop_closing::key_frame_and_pose        &NonCorrectedSim3,
                                         const loop_closing::key_frame_and_pose        &CorrectedSim3,
                                         const std::map<key_frame *, set<key_frame *>> &LoopConnections,
                                         const bool                                    &bFixScale);
    void static optimize_essential_graph(
        key_frame *pCurKF, vector<key_frame *> &vpFixedKFs, vector<key_frame *> &vpFixedCorrectedKFs, vector<key_frame *> &vpNonFixedKFs, vector<map_point *> &vpNonCorrectedMPs);

    // For inertial loopclosing
    void static optimize_essential_graph_4dof(map                                           *pMap,
                                              key_frame                                     *pLoopKF,
                                              key_frame                                     *pCurKF,
                                              const loop_closing::key_frame_and_pose        &NonCorrectedSim3,
                                              const loop_closing::key_frame_and_pose        &CorrectedSim3,
                                              const std::map<key_frame *, set<key_frame *>> &LoopConnections);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono) (NEW)
    static int optimize_sim3(key_frame                   *pKF1,
                             key_frame                   *pKF2,
                             std::vector<map_point *>    &vpMatches1,
                             g2o::Sim3                   &g2oS12,
                             const float                  th2,
                             const bool                   bFixScale,
                             Eigen::Matrix<double, 7, 7> &mAcumHessian,
                             const bool                   bAllPoints = false);

    // For inertial systems

    void static local_inertial_ba(key_frame *pKF, bool *pbStopFlag, map *pMap, int &num_fixedKF, int &num_OptKF, int &num_MPs, int &num_edges, bool bLarge = false, bool bRecInit = false);
    void static merge_inertial_ba(key_frame *pCurrKF, key_frame *pMergeKF, bool *pbStopFlag, map *pMap, loop_closing::key_frame_and_pose &corrPoses);

    // Local BA in welding area when two maps are merged
    void static local_bundle_adjustment(key_frame *pMainKF, vector<key_frame *> vpAdjustKF, vector<key_frame *> vpFixedKF, bool *pbStopFlag);

    // Marginalize block element (start:end,start:end). Perform Schur complement.
    // Marginalized elements are filled with zeros.
    static Eigen::MatrixXd marginalize(const Eigen::MatrixXd &H, const int &start, const int &end);

    // Inertial pose-graph
    void static inertial_optimization(map             *pMap,
                                      Eigen::Matrix3d &Rwg,
                                      double          &scale,
                                      Eigen::Vector3d &bg,
                                      Eigen::Vector3d &ba,
                                      bool             bMono,
                                      Eigen::MatrixXd &covInertial,
                                      bool             bFixedVel = false,
                                      bool             bGauss    = false,
                                      float            priorG    = 1e2,
                                      float            priorA    = 1e6);
    void static inertial_optimization(map *pMap, Eigen::Vector3d &bg, Eigen::Vector3d &ba, float priorG = 1e2, float priorA = 1e6);
    void static inertial_optimization(map *pMap, Eigen::Matrix3d &Rwg, double &scale);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

} // namespace ORB_SLAM3

#endif // OPTIMIZER_H
