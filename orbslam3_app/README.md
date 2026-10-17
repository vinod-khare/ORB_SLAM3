# ORB-SLAM3 App User Guide

`orbslam3_app` replays image sequences from folders through the ORB-SLAM3 library. It supports monocular and stereo input, optionally paired with IMU measurements. It is an offline dataset runner, not a live-camera or RGB-D application.

Playback is unpaced. A persistent worker reads, resizes, and applies grayscale/CLAHE preprocessing ahead of tracking, with at most three prepared frames queued. Tracking remains sequential, preserving image order, timestamps, and IMU associations; input failures are reported on the main thread.

Stereo input uses an additional persistent right-camera worker, overlapping left/right decoding and preprocessing with separate CLAHE instances. A single-image right-camera handoff keeps buffering bounded. Monocular input does not start this additional worker.

When optical flow is enabled, the prefetch worker also builds the left-image PyrLK pyramid, including derivatives, after CLAHE. Tracking reuses it in both flow directions and on the next frame. If SLAM must resize or rectify the input internally, pyramid construction remains lazy inside tracking so it uses the transformed image.

## Optical-flow tracking

Set `Tracking.useOpticalFlow: 1` to use a flow-first frontend in monocular/stereo SLAM, including inertial modes. Setting it to `0` retains descriptor-based tracking. RGB-D and localization-only mode retain descriptor-based tracking.

- Healthy ordinary frames track existing mapped left-camera observations directly with pyramidal LK. They do **not** detect features, compute ORB descriptors, snap endpoints to newly detected features, or run temporal/local-map descriptor searches.
- Motion/IMU prediction seeds LK when the camera projection is in the input image's coordinate system. Forward/backward consistency and photometric checks reject unreliable tracks. Up to 40 unique map points per cell of a 4-by-4 image grid are tracked (640 total).
- At least 60% of sampled tracks must survive LK validation. Pose optimization then validates geometry before accepting flow: at least 50 inliers, a 60% inlier ratio, and coverage of at least six grid cells are required, including after local-map pose/inertial optimization.
- Weak flow or local-map tracking triggers current-image ORB extraction and descriptor recovery. Initialization and relocalization also use descriptors. Frames refresh features at least every `max(2, Camera.fps / 4)` frames (integer division), even when flow remains healthy, to replenish observations and acquire stereo depth.
- A flow frame selected for keyframe insertion is first promoted to a fully extracted frame and revalidated against the map. Keyframes always contain current-image descriptors and, for stereo, fresh stereo correspondences/depth. Ordinary stereo flow frames use existing 2D–3D map correspondences, not newly triangulated depth.
- Keyframe selection compares flow inliers against the sampled left-camera population, rather than treating the deliberately bounded track count as a loss of the full stereo feature population.

Shutdown reports flow-only frames, feature-extracted frames, and flow recovery attempts. The app also reports playback throughput including input/prefetch waits; the existing median/mean tracking times exclude those waits. Compare both timing measures and trajectory accuracy on the same sequence before drawing performance conclusions.

## Viewer keypoints

The **ORB-SLAM3: Current Frame** window displays tracked features using OpenCV's rich keypoint style: circles show feature size and radial lines show orientation (when available). Green marks map-point matches; blue marks visual-odometry matches. Both camera views use this style, with keypoint positions and sizes scaled together. Initialization correspondence lines and tracking statistics are unchanged.

The viewer preserves the selected detector's native size and orientation, separately from the ORB patch sizes and orientations used for descriptors and matching. Sizes are converted from pyramid-level to input-image coordinates. Detectors that do not report orientation (such as FAST, AGAST, and GFTT) display circles without orientation lines. OpenCV does not assign a separate symbol to each detector type. Flow-only frames move the last detected markers to their tracked positions; their size and orientation remain from the last detection until features are refreshed.

The Pangolin map viewer and OpenCV frame viewer are owned by the app layer. The core `ORB_SLAM3` library runs without UI dependencies; `--no-viewer` skips loading viewer settings and starting the UI.

## Build

Install the dependencies used by the project (see [`install_dependencies.sh`](../install_dependencies.sh)), then configure and build from the repository root:

```sh
cmake -S . -B .build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build .build/Release -j20
```

If dependencies are installed through vcpkg, add `-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake` to the configure command, using your vcpkg checkout path.

The executable is `.build/Release/orbslam3_app/orbslam3_app`. To inspect its current options:

```sh
./.build/Release/orbslam3_app/orbslam3_app --help
```

## Quick Start

The checked-in [`TUM-VI.yaml`](../config/TUM-VI.yaml) expects a TUM-VI sequence and Kalibr camchain calibration at the paths in its `data` and `calibration` sections. Download/extract that data and adjust those paths (or create a settings YAML for your sequence), then run stereo SLAM:

```sh
./.build/Release/orbslam3_app/orbslam3_app \
  --vocab vocab/ORBvoc.txt \
  --settings config/TUM-VI.yaml \
  --slam-type stereo \
  --output-folder output/corridor \
  --output trajectory \
  --save-ply
```

Use `--slam-type stereo-inertial` to include the configured IMU data. For monocular modes, only the left image folder is required; use `mono` or `mono-inertial` respectively.

Dataset paths can be stored under `data` in the settings YAML or overridden on the command line. For example:

```sh
--data.root /datasets/sequence/mav0 --data.left cam0/data --data.right cam1/data
```

Paths in `data.root`, `data.left`, `data.right`, and `data.imu` are resolved relative to the settings file's directory when specified in YAML. A `--data.root` override is relative to the current working directory; the individual image/IMU paths may be absolute or relative to that root. Calibration file paths in YAML are relative to the settings file.

## Input Data

- Images may be `.png`, `.jpg`, or `.jpeg`. With no timestamps file, image filename stems must be numeric nanoseconds, for example `1403636579763555584.png`; frames are ordered by the numeric timestamp.
- Stereo images are paired by identical filename in the left and right folders.
- Pass `--times-file FILE` to use an external timestamp list. `--timestamps-type` accepts `auto`, `filename_ns`, `timestamp_ns`, or `utc`. `filename_ns` associates entries with numeric image filename stems; `timestamp_ns` associates each timestamp with the corresponding sorted image; `utc` accepts UTC date/time lines. The default `auto` detects the format where possible.
- In inertial modes, `data.imu` points to a CSV with seven numeric columns and no header: `timestamp_ns,gx,gy,gz,ax,ay,az`. Timestamps must be ordered. Blank lines and lines beginning with `#` are ignored.
- The camera and IMU calibration, camera model, and IMU noise parameters belong in the settings YAML. The provided TUM-VI configuration uses a Kalibr camchain file.

## Options

| Option | Description |
| --- | --- |
| `--vocab`, `--settings` | Required ORB vocabulary and settings YAML paths. These may also be supplied as the first two positional arguments. |
| `--slam-type` | `mono` (default), `mono-inertial`, `stereo`, or `stereo-inertial`. |
| `--data.root`, `--data.left`, `--data.right`, `--data.imu` | Dataset root and sensor data paths; command-line values override corresponding YAML values. |
| `--times-file`, `--timestamps-type` | Optional timestamp list and its format. |
| `--frames-skip N` | Skip the first N image frames. |
| `--frames-stride N` | Process every Nth frame; default is 1. |
| `--frames-take N` | Maximum number of frames; 0 (default) processes all remaining frames. |
| `--output-folder DIR` | Write output files into this directory; it is created if missing. |
| `--output NAME` | Set the trajectory basename. Without this option, trajectory files are `CameraTrajectory.txt` and `KeyFrameTrajectory.txt`; with it, files are `f_NAME.txt` and `kf_NAME.txt`. |
| `--save-ply` | Export map points to `pointcloud.ply` in the output folder. |
| `--save-colmap` | Export a COLMAP-compatible sparse model into the output folder. |
| `--no-viewer` | Run headless without the Pangolin viewer. |

## Features Available in This App

- Offline folder replay for monocular, stereo, monocular-inertial, and stereo-inertial SLAM.
- Image timestamps from numeric filenames or external nanosecond/UTC timestamp files.
- Stereo pairing by matching image filenames, plus time-associated IMU CSV input.
- Frame skipping, fixed-stride sampling, and frame-count limits for shorter runs.
- Pangolin SLAM visualization and camera/keyframe trajectory exports.
- Optional map-point PLY and COLMAP-compatible sparse-model exports.
- Configurable feature detector in the settings YAML: FAST, AGAST, GFTT, ORB, BRISK, SIFT, and AKAZE. The detector changes keypoint detection; descriptors remain ORB.
- Optional flow-first tracking with descriptor recovery through `Tracking.useOpticalFlow` in the settings YAML.
- Median and mean tracking-time summary after each run.

The executable currently does not provide live-camera, RGB-D, or ROS input. Those capabilities described in the upstream ORB-SLAM3 documentation are outside this app's current input interface.

## Accuracy Regression Test

[`orbslam3_tests/regression/regression.py`](../orbslam3_tests/regression/regression.py) runs stereo SLAM headless on TUM-VI corridor1, aligns the trajectory to the mocap ground truth with evo, and fails if ATE RMSE or the tracked-frame ratio is worse than [`baseline.yaml`](../orbslam3_tests/regression/baseline.yaml) beyond its tolerance.

```sh
python3 -m venv .venv && .venv/bin/pip install -r orbslam3_tests/regression/requirements.txt
source .venv/bin/activate
cmake -S . -B .build/Release -DORBSLAM3_REGRESSION_TEST=ON
cmake --build .build/Release -j20
ctest --test-dir .build/Release -L regression --output-on-failure   # regression only
ctest --test-dir .build/Release -LE regression                      # unit tests only
```

After an intentional accuracy change, refresh the baseline and commit it:

```sh
.venv/bin/python orbslam3_tests/regression/regression.py --output-folder .build/Release/regression/stereo --skip-run --update-baseline
```