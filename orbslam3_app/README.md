# ORB-SLAM3 App User Guide

`orbslam3_app` replays image sequences from folders through the ORB-SLAM3 library. It supports monocular and stereo input, optionally paired with IMU measurements. It is an offline dataset runner, not a live-camera or RGB-D application.

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

## Features Available in This App

- Offline folder replay for monocular, stereo, monocular-inertial, and stereo-inertial SLAM.
- Image timestamps from numeric filenames or external nanosecond/UTC timestamp files.
- Stereo pairing by matching image filenames, plus time-associated IMU CSV input.
- Frame skipping, fixed-stride sampling, and frame-count limits for shorter runs.
- Pangolin SLAM visualization and camera/keyframe trajectory exports.
- Optional map-point PLY and COLMAP-compatible sparse-model exports.
- Configurable feature detector in the settings YAML: FAST, AGAST, GFTT, ORB, BRISK, SIFT, and AKAZE. The detector changes keypoint detection; descriptors remain ORB.
- Optional sparse LK optical-flow associations through `Tracking.useOpticalFlow` in the settings YAML.
- Median and mean tracking-time summary after each run.

The executable currently does not provide live-camera, RGB-D, or ROS input. Those capabilities described in the upstream ORB-SLAM3 documentation are outside this app's current input interface.