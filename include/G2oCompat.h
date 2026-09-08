#ifndef ORB_SLAM3_G2OCOMPAT_H
#define ORB_SLAM3_G2OCOMPAT_H

#include <g2o/types/sba/types_six_dof_expmap.h>

namespace g2o
{
using Vector7d = Vector7;
using VertexSBAPointXYZ = VertexPointXYZ;
}

#endif