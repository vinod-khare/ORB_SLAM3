# ORB-SLAM3 App User Guide

`orbslam3_app` replays image sequences from folders through the ORB-SLAM3 library. It supports monocular and stereo input, optionally paired with IMU measurements. It is an offline dataset runner, not a live-camera or RGB-D application.

Playback is unpaced. A persistent worker reads, resizes, and applies grayscale/CLAHE preprocessing ahead of tracking, with at most three prepared frames queued. Tracking remains sequential, preserving image order, timestamps, and IMU associations; input failures are reported on the main thread.

Stereo input uses an additional persistent right-camera worker, overlapping left/right decoding and preprocessing with separate CLAHE instances. A single-image right-camera handoff keeps buffering bounded. Monocular input does not start this additional worker.

When optical flow is enabled, the prefetch worker also builds the left-image PyrLK pyramid, including derivatives, after CLAHE. Tracking reuses it in both flow directions and on the next frame. If SLAM must resize or rectify the input internally, pyramid construction remains lazy inside tracking so it uses the transformed image.

## Optical-flow tracking

Set `tracking.use_optical_flow: 1` to use a flow-first frontend in monocular/stereo SLAM, including inertial modes. Setting it to `0` retains descriptor-based tracking. The command-line option `--tracking.use_optical_flow` enables flow; use `--tracking.use_optical_flow=false` to disable it. RGB-D and localization-only mode retain descriptor-based tracking.

- Healthy ordinary frames track existing mapped left-camera observations directly with pyramidal LK. They do **not** detect features, compute ORB descriptors, snap endpoints to newly detected features, or run temporal/local-map descriptor searches.
- Motion/IMU prediction seeds LK when the camera projection is in the input image's coordinate system. Forward/backward consistency and photometric checks reject unreliable tracks. Up to 40 unique map points per cell of a 4-by-4 image grid are tracked (640 total).
- At least 60% of sampled tracks must survive LK validation. Pose optimization then validates geometry before accepting flow: at least 50 inliers, a 60% inlier ratio, and coverage of at least six grid cells are required, including after local-map pose/inertial optimization.
- Weak flow or local-map tracking triggers current-image ORB extraction and descriptor recovery. Initialization and relocalization also use descriptors. Frames refresh features at least every `max(2, Camera.fps / 4)` frames (integer division), even when flow remains healthy, to replenish observations and acquire stereo depth.
- A flow frame selected for keyframe insertion is first promoted to a fully extracted frame and revalidated against the map. Keyframes always contain current-image descriptors and, for stereo, fresh stereo correspondences/depth. Ordinary stereo flow frames use existing 2D–3D map correspondences, not newly triangulated depth.
- Keyframe selection compares flow inliers against the sampled left-camera population, rather than treating the deliberately bounded track count as a loss of the full stereo feature population.

Shutdown reports flow-only frames, feature-extracted frames, and flow recovery attempts. The app also reports playback throughput including input/prefetch waits; the mean frontend time excludes those waits. Compare both timing measures and trajectory accuracy on the same sequence before drawing performance conclusions.

## Processing Statistics

Each processed frame logs statistics through `spdlog`, using the configured logging
pattern. Playback ends with mean preprocessing/frontend times and totals for
completed backend jobs. All durations are steady-clock wall times in milliseconds.

The app also writes `statistics.csv` to `--output-folder` (or the current working
directory when omitted), replacing any existing file. It contains a header and
one numeric row per processed frame, including frames where tracking fails.
Columns include frame ID, dataset `timestamp` in seconds, detected/matched counts,
preprocessing/frontend milliseconds, processed-frame count, elapsed playback
seconds, throughput FPS, cumulative preprocessing/frontend milliseconds, and each
backend's completed-job count, last milliseconds, and total milliseconds. Backend
values are snapshots at frame completion, not work attributed to that frame or
the final shutdown totals. CSV output is independent of logging and viewer settings.

- **Detected keypoints:** fresh detections across the input camera views. Flow-only
  frames report zero; descriptor recovery and keyframe promotion report their fresh
  detections.
- **Matched keypoints:** final non-outlier feature observations associated with map
  points, counted only when tracking succeeds. These are not tentative descriptor
  candidates or the number of stereo depth matches. Stereo observations can share
  a map point.
- **Preprocessing:** elapsed input preparation in the prefetch worker, including
  image loading, resize, grayscale, CLAHE, optional optical-flow pyramid creation,
  and stereo synchronization. Waiting for space in the prepared-frame queue is
  excluded. This is preparation latency, not summed left/right CPU time.
- **Frontend:** the synchronous `TrackMonocular`/`TrackStereo` call, including
  internal image adaptation, feature extraction, matching, and pose estimation.
  Viewer publication and statistics logging are excluded.
- **Backend:** latest completed job time, job count, and cumulative elapsed work
  for local mapping (including local BA), loop detection/correction/merging, and
  global bundle adjustment. Idle worker sleeps are excluded; aborted jobs are
  included. Independent workers overlap, so their durations must not be added to
  frontend time to estimate frame latency. No completed jobs is shown as zero
  jobs, rather than an implied zero-cost backend. Shutdown joins backend workers
  before sampling its summary; pending queued jobs are not guaranteed to run.
- **Throughput:** processed frames divided by elapsed playback time, including
  input waits and preceding viewer/logging work, rather than reciprocal tracking
  time or dataset timestamps. The final Playback line includes the full loop.

`System::statistics().get_snapshot()` returns a thread-safe value copy containing
the latest frame ID, dataset timestamp, counts and timings, playback totals/FPS,
and the three backend samples. Collection has bounded storage and no logging or
UI dependency. `System` owns the collector directly; workers borrow it and are
joined on shutdown or destruction. A future ImGui panel can poll this API and
append samples to its own plot buffers, using the frame ID/job counts to detect new samples. Workers
never render or hold the statistics mutex while a consumer logs or plots.

## Viewer keypoints

The camera panes in the **ORB-SLAM3** window display tracked features using OpenCV's rich keypoint style: circles show feature size and radial lines show orientation (when available). Green marks map-point matches; blue marks visual-odometry matches. Both camera views use this style, with keypoint positions and sizes scaled together. Initialization correspondence lines and tracking statistics are unchanged.

The viewer preserves the selected detector's native size and orientation, separately from the ORB patch sizes and orientations used for descriptors and matching. Sizes are converted from pyramid-level to input-image coordinates. Detectors that do not report orientation (such as FAST, AGAST, and GFTT) display circles without orientation lines. OpenCV does not assign a separate symbol to each detector type. Flow-only frames move the last detected markers to their tracked positions; their size and orientation remain from the last detection until features are refreshed.

Hello ImGui owns one native window with resizable, dockable Controls, Map, and camera panes (separate left/right panes in stereo). VTK renders the 3D map off-screen; both that image and OpenCV overlays appear as ImGui textures. Drag the map to orbit, use Shift-drag or the middle/right mouse button to pan, and scroll to zoom. The controls retain follow-camera, camera/top views, layer visibility, localization, stepping, reset, and stop/export. Pane layouts and window geometry persist between runs; the layout button restores the default arrangement.

The UI runs on a separate, lower-priority thread capped at 30 FPS. Camera image snapshots are demand-driven: the SLAM thread uses `try_lock` and drops an image update rather than waiting for the UI. Camera poses are queued through a short mutex-protected handoff so the full trajectory retains every tracked frame; the UI only swaps that queue while locked, never rendering under the lock. Local-BA color publication remains non-blocking. All image annotation, VTK rendering, and texture uploads happen on the UI thread. Map access retains the core's existing short data-access locks; rendering never holds those locks. Reset explicitly pauses map reads to protect object lifetimes. Closing the window detaches visualization and lets playback continue; Stop ends playback and uses the normal trajectory-export path.

The core `ORB_SLAM3` library runs without UI dependencies; `--no-viewer` skips loading viewer settings and starting the UI. Neither Pangolin nor OpenCV HighGUI is required by the app.

## Ground Truth

Display a ground-truth trajectory as an orange line in the map pane:

```yaml
ground_truth:
  path: "../dataset/mav0/mocap0/data.csv"
  visible: true
```

Or use `--ground_truth.path /datasets/trajectory.txt --ground_truth.visible`.
Command-line values override YAML; `--ground_truth.visible=false` starts with the
overlay hidden. YAML paths are relative to the settings file; command-line paths
are relative to the working directory. Visibility defaults to false. A configured
file is loaded even when initially hidden, and the **Ground truth** checkbox under
**Map layers** toggles it during playback. Without a path, that checkbox is disabled.
Headless runs do not load ground truth.

Supported formats are TUM text (`timestamp_seconds tx ty tz qx qy qz qw`, camera
poses) and EuRoC/TUM-VI CSV (`timestamp_nanoseconds,px,py,pz,qw,qx,qy,qz`, IMU/body
poses; additional columns are allowed). Blank lines and `#` comments are ignored.
Timestamps must increase strictly, and pose values must be finite with nonzero
quaternions. Invalid or unreadable files report an error rather than being ignored.
For CSV, the Kalibr `cam0.T_cam_imu` transform from `calibration.file`, when present,
converts body poses to camera poses; otherwise the frames are assumed identical.

Ground truth is aligned to the active map using Eigen's least-squares rigid
alignment, with scale fitting only for non-inertial monocular SLAM. Translation is
interpolated using lerp and orientation using quaternion slerp at each keyframe's
timestamp, including between distant samples; no nearest-timestamp cutoff is used.
The camera offset is applied after interpolation. Timestamps outside the ground-truth
range are not extrapolated. The overlay appears once there are at least three
non-collinear reference positions. That first valid alignment is frozen for the
viewer lifetime: optimization, keyframe culling, resets, visibility toggles, and
playback beyond the reference time range never refit or move ground truth.

## Camera Trajectory

The **Camera trajectory** layer shows every successfully tracked frame, not just
keyframes. It defaults to visible and remains independent of the keyframe layer.
It records the live pose estimates as published, rather than retroactively
reconstructing an optimized export trajectory. Tracking loss, resets, and map
changes start disconnected segments instead of drawing artificial bridges.
Hiding the layer does not discard its history.

The camera trail is a 3-pixel teal line; ground truth is a 7-pixel orange line.
The narrower camera line is drawn in front where paths coincide, leaving the
orange edges visible. Matching color swatches beside both layer checkboxes
identify them.

```yaml
camera_trajectory:
  visible: true
```

Override this using `--camera_trajectory.visible=false` (or
`--camera_trajectory.visible` to enable it).

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

- Images may be `.png`, `.jpg`, or `.jpeg`. Image filename stems must be numeric nanoseconds, for example `1403636579763555584.png`; frames are ordered by the numeric timestamp.
- Stereo images are paired by identical filename in the left and right folders.
- In inertial modes, `data.imu` points to a CSV with seven numeric columns and no header: `timestamp_ns,gx,gy,gz,ax,ay,az`. Timestamps must be ordered. Blank lines and lines beginning with `#` are ignored.
- `calibration.file` points to the Kalibr camchain YAML providing camera models, intrinsics, distortion, resolution, and camera/IMU extrinsics.
- In inertial modes, `calibration.imu_file` must point to a Kalibr IMU YAML containing `gyroscope_noise_density`, `accelerometer_noise_density`, `gyroscope_random_walk`, `accelerometer_random_walk`, and `update_rate`. All five values must be finite and positive. Relative paths are resolved from the settings file, not the camchain file.
- IMU noise and frequency must come from that calibration file. The former settings keys `IMU.NoiseGyro`, `IMU.NoiseAcc`, `IMU.GyroWalk`, `IMU.AccWalk`, and `IMU.Frequency` are no longer supported. `IMU.InsertKFsWhenLost` remains an optional settings entry.

## Options

| Option                                                     | Description                                                                                                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--vocab`, `--settings`                                    | Required ORB vocabulary and settings YAML paths. These may also be supplied as the first two positional arguments.                                                             |
| `--slam-type`                                              | `mono` (default), `mono-inertial`, `stereo`, or `stereo-inertial`.                                                                                                             |
| `--data.root`, `--data.left`, `--data.right`, `--data.imu` | Dataset root and sensor data paths; command-line values override corresponding YAML values.                                                                                    |
| `--frames-skip N`                                          | Skip the first N image frames.                                                                                                                                                 |
| `--frames-stride N`                                        | Process every Nth frame; default is 1.                                                                                                                                         |
| `--frames-take N`                                          | Maximum number of frames; 0 (default) processes all remaining frames.                                                                                                          |
| `--output-folder DIR`                                      | Write output files into this directory; it is created if missing.                                                                                                              |
| `--output NAME`                                            | Set the trajectory basename. Without this option, trajectory files are `CameraTrajectory.txt` and `KeyFrameTrajectory.txt`; with it, files are `f_NAME.txt` and `kf_NAME.txt`. |
| `--save-ply`                                               | Export map points to `pointcloud.ply` in the output folder.                                                                                                                    |
| `--save-colmap`                                            | Export a COLMAP-compatible sparse model into the output folder.                                                                                                                |
| `--keypoint_detector.type`                                 | Keypoint detector (`fast`, `agast`, `gftt`, `orb`, `brisk`, `sift`, `akaze`); overrides `keypoint_detector.type` in the settings YAML.                                         |
| `--keypoint_descriptor.type`                               | Independent OpenCV descriptor (`orb`, `sift`, `brisk`). |
| `--tracking.matcher`                                       | OpenCV matcher (`brute_force`, `flann`); overrides `tracking.matcher` in YAML. Hamming/LSH for binary data, L2/KD-tree for SIFT. |
| `--tracking.use_optical_flow [true/false]`                  | Enable or disable flow-first tracking; overrides YAML `tracking.use_optical_flow` (default: false). |
| `--dbow_version`                                           | Place-recognition backend (`2` or `3`); overrides YAML `dbow_version`. Default: `2`. |
| `--no-viewer`                                              | Run headless without the ImGui viewer.                                                                                                                                      |
| `--ground_truth.path`                                     | TUM trajectory or EuRoC/TUM-VI CSV file; overrides YAML `ground_truth.path`. |
| `--ground_truth.visible [true/false]`                      | Initial ground-truth visibility; overrides YAML `ground_truth.visible` (default: false). |
| `--camera_trajectory.visible [true/false]`                 | Initial full camera-trail visibility; overrides YAML `camera_trajectory.visible` (default: true). |
| `--logging.pattern`                                       | spdlog output pattern; overrides `logging.pattern` in YAML. Default: `%+` (spdlog's built-in default). |

## DBoW place recognition

Both DBoW2 and DBoW3 are linked into the application. Install `dbow2` and `dbow3`
through vcpkg before configuring the build. DBoW2 remains the default.

Select DBoW3 with `--dbow_version 3`, or in the settings YAML:

```yaml
dbow_version: 3
```

Only integer values `2` and `3` are accepted; the command line overrides YAML.
Both backends use the same ORB-SLAM text vocabulary (`ORBvoc.txt`) and the separate
ORB place-recognition descriptors, regardless of the tracking descriptor selected.
Use this text format when switching between backends; native DBoW3 binary/YAML
vocabularies are not interchangeable with the DBoW2 text loader.

During playback, use **Controls > Tracking > DBoW version** to switch backends.
The selected backend loads the same vocabulary file between tracked frames;
tracking and vocabulary queries wait until loading finishes. Keep that file
unchanged during playback. Word/node IDs and saved-atlas vector types remain
unchanged, so existing keyframes and the inverted index remain valid without a
map reset. A failed switch is logged and shown in the GUI, retaining the previous
backend. GUI changes apply only to the current run; they do not rewrite YAML.

## Logging

Set the spdlog output pattern in the settings YAML:

```yaml
logging:
  pattern: "[%H:%M:%S] [%^%l%$] %v"
```

Or override it with `--logging.pattern '[%H:%M:%S] [%^%l%$] %v'`.
Quote patterns to preserve `%` characters and spaces. With neither setting, logs
use spdlog's project default (`%+`), including date, time with milliseconds,
logger name (when nonempty), colored level, and message. `%v` prints only the message.
See the [spdlog pattern flags](https://github.com/gabime/spdlog/wiki/Custom-formatting#pattern-flags).

## Features Available in This App

- Offline folder replay for monocular, stereo, monocular-inertial, and stereo-inertial SLAM.
- Image timestamps from numeric nanosecond filename stems.
- Stereo pairing by matching image filenames, plus time-associated IMU CSV input.
- Frame skipping, fixed-stride sampling, and frame-count limits for shorter runs.
- Dockable ImGui/VTK SLAM visualization and camera/keyframe trajectory exports.
- Optional map-point PLY and COLMAP-compatible sparse-model exports.
- Independently configurable OpenCV detectors, descriptors (ORB, SIFT, BRISK), and matchers (brute force, FLANN). The existing vocabulary uses a separate ORB place-recognition descriptor channel; see the root README's Feature Pipeline section.
- Optional flow-first tracking with descriptor recovery through `tracking.use_optical_flow` in YAML or `--tracking.use_optical_flow` on the command line.
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
## TUM-VI Stereo Benchmark

[`orbslam3_tests/benchmark/tumvi_benchmark.py`](../orbslam3_tests/benchmark/tumvi_benchmark.py)
runs all extracted `dataset-*` directories under `.data/tumvi` sequentially,
headless, in stereo mode, processing every frame. Build the app first and install
the existing regression dependencies in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r orbslam3_tests/regression/requirements.txt
.venv/bin/python orbslam3_tests/benchmark/tumvi_benchmark.py
```

Each sequence needs `mav0/cam0/data`, `mav0/cam1/data`, `dso/camchain.yaml`,
and `mav0/mocap0/data.csv` for accuracy evaluation. The benchmark copies the
selected settings template, replacing dataset/calibration paths with absolute
per-sequence paths; it never edits the template. Feature settings are preserved.
Missing inputs, crashes, timeouts, and evaluation failures are reported without
stopping the remaining sequences; any failure gives a nonzero exit status.

Every execution creates a new run under `.data/benchmark/tumvi/<UTC timestamp>`.
Use `--run-name NAME` for a memorable name and `--notes TEXT` to describe the
experiment. Names must be unique; even `--overwrite` cannot replace a saved run.
Each run contains `summary.md`, `summary.csv`, and
`summary.json`, plus a directory per sequence containing generated settings,
`app.log`, trajectories, and the original per-frame `statistics.csv`.
Reports are updated after every sequence. JSON includes pooled aggregates;
CSV contains one row per sequence, including failure details and timeout/crash
classification. An interrupted run retains completed reports and is marked
`interrupted` in its manifest; an abruptly killed process may remain `running`.

`run.json` records UTC start/end times, notes, sequence selection, settings,
timeout, Git commit/branch/dirty status, executable and vocabulary SHA-256 hashes,
and host/runtime information. `template.yaml` snapshots the input settings;
`source.patch` captures tracked working-tree changes when present. Untracked
paths are listed in Git status but their contents are not archived. The binary
hash identifies the executed build; the recorded Git commit does not guarantee
that the executable was rebuilt from that commit. Dataset fingerprints include
calibration, IMU, timestamp and ground-truth file hashes and stereo image
inventories (names and sizes, not image contents).

- Accuracy: translation ATE RMSE after timestamp matching (10 ms tolerance),
  camera-to-body conversion, and SE(3) alignment without scale correction;
  tracked-frame ratio and matched-pose count. Tracking coverage is retained even
  if mocap evaluation fails. TUM-VI mocap generally covers only
  the start/end room, not the entire route. Pooled ATE weights squared errors by
  matched-pose count after independent per-sequence alignments.
- Frame statistics: mean, median, p95, min, and max of detected/matched keypoint
  counts and preprocessing/frontend milliseconds, both per sequence and pooled.
- Cumulative statistics: frame count, elapsed playback time, throughput, summed
  preprocessing/frontend time, and worker job counts/times. Worker totals are
  last-frame CSV snapshots, may exclude shutdown work, and overlap in time.
  Playback throughput excludes startup/shutdown; `wall_seconds` includes them
  for new runs and is unavailable when re-evaluating. Failed/incomplete runs do
  not enter pooled frame statistics; accuracy failures retain complete statistics.

Options: `--datasets-root DIR`, `--output-folder DIR`, `--app PATH`,
`--vocab PATH`, `--settings PATH`, repeated `--sequence dataset-room1_512_16`,
and `--timeout SECONDS` (per sequence, default unlimited). `--output-folder` is
the parent directory for new runs. With `--skip-run`, it instead names an existing
run directory: saved outputs/settings are re-evaluated to stdout without changing
any reports. For recorded runs, pass the original sequence selection. Legacy
directories containing `summary.json` can also be compared, without provenance.

To compare stereo SLAM with and without optical-flow tracking on every locally
extracted TUM-VI sequence, run:

```sh
.venv/bin/python orbslam3_tests/benchmark/tumvi_benchmark.py \
  --compare-optical-flow --run-name stereo-optical-flow
```

This runs the complete selected dataset once with `tracking.use_optical_flow: 0`
and once with `tracking.use_optical_flow: 1`, explicitly overriding the template
while preserving its other feature/tracking settings. Both use the same binary,
vocabulary, frame selection, and stereo mode without IMU. Install all desired
sequences under `--datasets-root` first; this command does not download datasets.
Use a unique run name for each repetition and keep the machine otherwise idle.

Results are saved under `.data/benchmark/tumvi/stereo-optical-flow/`, with
`without_optical_flow/` and `with_optical_flow/` holding the usual per-sequence
logs, statistics, trajectories, summaries, and provenance. The parent contains
`comparison.md`, `comparison.csv`, and `comparison.json`: baseline is flow off,
candidate is flow on. Both modes are attempted even if individual sequences
fail; any sequence failure returns a nonzero exit status. Ctrl+C stops the batch
and preserves completed reports. Repeat runs before drawing performance
conclusions; one pair is not a statistical significance test.

Record a baseline and an experiment, rebuilding the executable after algorithm
changes or selecting a different feature-settings template with `--settings`:

```sh
.venv/bin/python orbslam3_tests/benchmark/tumvi_benchmark.py --run-name baseline
# Rebuild after code changes, then:
.venv/bin/python orbslam3_tests/benchmark/tumvi_benchmark.py --run-name candidate --notes "Describe the algorithm or settings change"
.venv/bin/python orbslam3_tests/benchmark/tumvi_benchmark.py \
  --compare .data/benchmark/tumvi/baseline .data/benchmark/tumvi/candidate \
  --comparison-output .data/benchmark/comparisons/baseline-vs-candidate
```

Comparison needs no SLAM execution or original datasets. It prints a Markdown
report and optionally writes `comparison.md`, `comparison.json` and metric rows
in `comparison.csv`. Reports show provenance/settings changes, every sequence's
status and errors, and absolute/percentage metric deltas (candidate minus
baseline). Percentages are omitted for zero baselines. Accuracy/latency decreases
and coverage/throughput increases are labeled by direction, not statistical
significance. Counts are contextual. Pooled deltas are omitted when sequence
sets, processed frame counts or outcomes differ, avoiding survivor bias; paired
per-sequence metrics remain available. Dataset, host, vocabulary and timeout
changes trigger a warning. A comparison exits zero on successful report
generation, even when metrics regress; it is not a CI threshold gate. Repeat
runs under the same conditions to distinguish improvements from timing noise
and multithreaded tracking variability.

Run the fast synthetic integration checks without processing a dataset:

```sh
.venv/bin/python -m unittest discover -s orbslam3_tests/benchmark -p 'tumvi_benchmark_test.py'
```

## Keypoint Detector Benchmark

[`orbslam3_tests/benchmark/detector_benchmark.py`](../orbslam3_tests/benchmark/detector_benchmark.py) runs `orbslam3_app` headless once per detector (selected with `--keypoint_detector.type`, or `keypoint_detector: { type: ... }` in the settings YAML), scores each trajectory against mocap ground truth using the regression test's evaluation, and prints a Markdown table. Descriptor and matcher choices come from the settings YAML.

```sh
.venv/bin/python orbslam3_tests/benchmark/detector_benchmark.py \
  --benchmark.detectors fast,agast,gftt,orb,brisk,sift,akaze \
  --benchmark.slam_type stereo
```

Other options: `--benchmark.output_folder DIR` and `--benchmark.skip_run` (re-evaluate existing trajectories). Per-detector logs and trajectories are written under `.data/benchmark/<detector>/`.

Results: TUM-VI corridor1 (512_16), stereo, `config/TUM-VI.yaml` with optical flow enabled, Release build, one run per detector (the tracker is multithreaded, so results vary slightly between runs). ATE RMSE is measured only where mocap exists (start/end of the sequence); track times exclude input waits, playback fps includes them.

| Detector | ATE RMSE (m) | Tracked ratio | Mean track (ms) | Median track (ms) | Playback (frames/s) |
| -------- | -----------: | ------------: | --------------: | ----------------: | ------------------: |
| fast     |       0.0935 |         1.000 |            8.87 |              4.65 |                55.8 |
| agast    |       0.0689 |         1.000 |            7.73 |              4.70 |                56.8 |
| gftt     |       0.0329 |         1.000 |            8.57 |              4.96 |                58.8 |
| orb      |       0.1046 |         1.000 |            7.06 |              4.82 |                56.9 |
| brisk    |       0.0161 |         1.000 |            9.52 |              4.84 |                56.7 |
| sift     |       0.0850 |         1.000 |           21.20 |              4.70 |                39.0 |
| akaze    |       0.1006 |         1.000 |           14.09 |              4.19 |                47.7 |

With optical flow most frames skip detection, so median times are nearly identical; the detector cost shows up in the mean (keyframes and recovery frames). SIFT and AKAZE are the slowest; BRISK and GFTT were the most accurate in this run.
