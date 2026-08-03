#include <opencv2/core/core.hpp>

namespace orbslam3
{
class frame_mono
{
public:
  double  timestamp;
  cv::Mat image;
};
} // namespace orbslam3