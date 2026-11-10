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

#ifndef ORB_SLAM3_OPTIMIZABLETYPES_H
#define ORB_SLAM3_OPTIMIZABLETYPES_H

#include "CameraModels/geometric_camera.h"
#include "G2oCompat.h"

#include <memory>

#include <Eigen/Geometry>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/sim3/sim3.h>

namespace ORB_SLAM3
{
class EdgeSE3ProjectXYZOnlyPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeSE3ProjectXYZOnlyPose(geometric_camera &camera) : camera(camera) {}

    bool read(std::istream &is);

    bool write(std::ostream &os) const;

    void computeError()
    {
        const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector2d             obs(_measurement);
        _error = obs - camera.project(v1->estimate().map(Xw));
    }

    bool isDepthPositive()
    {
        const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        return (v1->estimate().map(Xw))(2) > 0.0;
    }

    virtual void linearizeOplus();

    Eigen::Vector3d  Xw;
    geometric_camera &camera;
};

class EdgeSE3ProjectXYZOnlyPoseToBody : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeSE3ProjectXYZOnlyPoseToBody(geometric_camera &camera) : camera(camera) {}

    bool read(std::istream &is);

    bool write(std::ostream &os) const;

    void computeError()
    {
        const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector2d             obs(_measurement);
        _error = obs - camera.project((mTrl * v1->estimate()).map(Xw));
    }

    bool isDepthPositive()
    {
        const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        return ((mTrl * v1->estimate()).map(Xw))(2) > 0.0;
    }

    virtual void linearizeOplus();

    Eigen::Vector3d  Xw;
    geometric_camera &camera;

    g2o::SE3Quat mTrl;
};

class EdgeSE3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeSE3ProjectXYZ(geometric_camera &camera);

    bool read(std::istream &is);

    bool write(std::ostream &os) const;

    void computeError()
    {
        const g2o::VertexSE3Expmap   *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const g2o::VertexSBAPointXYZ *v2 = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        Eigen::Vector2d               obs(_measurement);
        _error = obs - camera.project(v1->estimate().map(v2->estimate()));
    }

    bool isDepthPositive()
    {
        const g2o::VertexSE3Expmap   *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const g2o::VertexSBAPointXYZ *v2 = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        return ((v1->estimate().map(v2->estimate()))(2) > 0.0);
    }

    virtual void linearizeOplus();

    geometric_camera &camera;
};

class EdgeSE3ProjectXYZToBody : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit EdgeSE3ProjectXYZToBody(geometric_camera &camera);

    bool read(std::istream &is);

    bool write(std::ostream &os) const;

    void computeError()
    {
        const g2o::VertexSE3Expmap   *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const g2o::VertexSBAPointXYZ *v2 = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        Eigen::Vector2d               obs(_measurement);
        _error = obs - camera.project((mTrl * v1->estimate()).map(v2->estimate()));
    }

    bool isDepthPositive()
    {
        const g2o::VertexSE3Expmap   *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const g2o::VertexSBAPointXYZ *v2 = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        return ((mTrl * v1->estimate()).map(v2->estimate()))(2) > 0.0;
    }

    virtual void linearizeOplus();

    geometric_camera &camera;
    g2o::SE3Quat     mTrl;
};

class VertexSim3Expmap : public g2o::BaseVertex<7, g2o::Sim3>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit VertexSim3Expmap(const camera_vector &cameras);
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    virtual void setToOriginImpl() { _estimate = g2o::Sim3(); }

    virtual void oplusImpl(const double *update_)
    {
        Eigen::Map<g2o::Vector7d> update(const_cast<double *>(update_));

        if (_fix_scale)
        {
            update[6] = 0;
        }

        g2o::Sim3 s(update);
        setEstimate(s * estimate());
    }

    const camera_vector &cameras;

    bool _fix_scale;
};

class EdgeSim3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, ORB_SLAM3::VertexSim3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3ProjectXYZ();
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    void computeError()
    {
        const ORB_SLAM3::VertexSim3Expmap *v1 = static_cast<const ORB_SLAM3::VertexSim3Expmap *>(_vertices[1]);
        const g2o::VertexSBAPointXYZ      *v2 = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);

        Eigen::Vector2d obs(_measurement);
        _error = obs - v1->cameras[0]->project(v1->estimate().map(v2->estimate()));
    }
};

class EdgeInverseSim3ProjectXYZ : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseSim3ProjectXYZ();
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    void computeError()
    {
        const ORB_SLAM3::VertexSim3Expmap *v1 = static_cast<const ORB_SLAM3::VertexSim3Expmap *>(_vertices[1]);
        const g2o::VertexSBAPointXYZ      *v2 = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);

        Eigen::Vector2d obs(_measurement);
        _error = obs - v1->cameras[0]->project((v1->estimate().inverse().map(v2->estimate())));
    }
};

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_OPTIMIZABLETYPES_H
