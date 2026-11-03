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
For CSV, the Kalibr `cam0.T_cam_imu` transform from `data.calibration.file`, when present,
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
  --vocab vocab/ORBvoc.dbow3 \
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
- `data.calibration.file` points to the Kalibr camchain YAML providing camera models, intrinsics, distortion, resolution, and camera/IMU extrinsics.
- In inertial modes, `data.calibration.imu_file` must point to a Kalibr IMU YAML containing `gyroscope_noise_density`, `accelerometer_noise_density`, `gyroscope_random_walk`, `accelerometer_random_walk`, and `update_rate`. All five values must be finite and positive. Calibration paths are resolved from the settings file, not `data.root` or the camchain file.
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
| `--save-ply`                                               | Export map points with sampled RGB colors to `pointcloud.ply` in the output folder.                                                                                           |
| `--save-colmap`                                            | Export a COLMAP-compatible sparse model into the output folder.                                                                                                                |
| `--keypoint_detector.type`                                 | Keypoint detector (`fast`, `agast`, `gftt`, `orb`, `brisk`, `sift`, `akaze`, `star`, `harris_laplace`, `msd`, `tbmr`, `surf`); overrides `keypoint_detector.type` in YAML. |
| `--keypoint_descriptor.type`                               | Independent descriptor (`orb`, `sift`, `brisk`, `brief`, `freak`, `latch`, `beblid`, `teblid`, `daisy`, `vgg`, `boostdesc`, `surf`); overrides `keypoint_descriptor.type` in YAML. SURF requires OpenCV nonfree support. |
| `--tracking.matcher`                                       | OpenCV matcher (`brute_force`, `flann`); overrides `tracking.matcher` in YAML. Hamming/LSH for binary data, L2/KD-tree for floating-point data. |
| `--tracking.use_optical_flow [true/false]`                  | Enable or disable flow-first tracking; overrides YAML `tracking.use_optical_flow` (default: false). |
| `--dbow_version`                                           | Place-recognition backend (`2` or `3`); overrides YAML `dbow_version`. Default: `2`. |
| `--no-viewer`                                              | Run headless without the ImGui viewer.                                                                                                                                      |
| `--ground_truth.path`                                     | TUM trajectory or EuRoC/TUM-VI CSV file; overrides YAML `ground_truth.path`. |
| `--ground_truth.visible [true/false]`                      | Initial ground-truth visibility; overrides YAML `ground_truth.visible` (default: false). |
| `--camera_trajectory.visible [true/false]`                 | Initial full camera-trail visibility; overrides YAML `camera_trajectory.visible` (default: true). |
| `--logging.pattern`                                       | spdlog output pattern; overrides `logging.pattern` in YAML. Default: `%+` (spdlog's built-in default). |
| `--logging.level`                                         | spdlog minimum level; overrides `logging.level` in YAML. Default: `info`. |

Map points use colors sampled from the original camera images, before CLAHE or
grayscale conversion. The viewer and PLY export share the same stored RGB values;
export does not need the source images to remain on disk. Grayscale datasets,
including TUM-VI, produce grayscale points. Points without a source pixel are white.

## DBoW place recognition

Both DBoW2 and DBoW3 are linked into the application. Install `dbow2` and `dbow3`
through vcpkg before configuring the build. DBoW2 remains the default.

Select DBoW3 with `--dbow_version 3`, or in the settings YAML:

```yaml
dbow_version: 3
```

Only integer values `2` and `3` are accepted; the command line overrides YAML.
Both backends use the same compressed DBoW3 binary vocabulary (`vocab/ORBvoc.dbow3`)
and the separate ORB place-recognition descriptors, regardless of the tracking
descriptor selected. DBoW2 imports the binary tree without changing node IDs,
word IDs, child order, descriptors, or double-precision weights. Text and YAML
vocabularies are rejected by both backends, regardless of filename extension.

The shipped vocabulary was losslessly converted from the original ORB text file:
971,814 words and 1,082,072 non-root nodes, reduced from 145,250,924 to 50,559,373
bytes (65.2% smaller). Compression changes storage only, not tree size or accuracy.
Do not prune or retrain this file to reduce size while reusing an existing atlas.

During playback, use **Controls > Tracking > DBoW version** to switch backends.
The selected backend loads the same vocabulary file between tracked frames;
tracking and vocabulary queries wait until loading finishes. Keep that file
unchanged during playback. Word/node IDs and saved-atlas vector types remain
unchanged, so existing keyframes and the inverted index remain valid without a
map reset. A failed switch is logged and shown in the GUI, retaining the previous
backend. GUI changes apply only to the current run; they do not rewrite YAML.

### Training A Vocabulary

Vocabularies belong to a descriptor algorithm and its settings, not to the storage
format or detector alone. SIFT's 128 floating-point values require a different
trained tree and distance metric from ORB's 32-byte binary descriptors. Even
equal-sized descriptors such as ORB and BRIEF are not interchangeable.
However, **SIFT tracking does not require a SIFT vocabulary in this app**: place
recognition always uses the separate ORB channel. Both runtime backends therefore
require a 32-byte ORB vocabulary. Non-ORB training outputs are for external use or
a future descriptor-specific place-recognition pipeline; they do not enable that
pipeline automatically. A file's shape alone cannot identify its algorithm, so
keep descriptor names and training settings with generated files.

Build the `vocabulary_trainer` CMake target, then run the dependency-free script:

```bash
bash tools/generate_vocabulary.sh \
  --sequence .data/tumvi/dataset-corridor1_512_16/dso/cam0/images \
  --output vocab/corridor-orb.dbow3 --descriptor orb \
  --stride 10 --max-frames 500 --features 1000 --branching 10 --levels 5
```

Use the camera's image directory, not a dataset root containing multiple cameras,
vignettes, and output images. Images are recursively collected and lexically sorted;
timestamp names should sort chronologically. `--stride` samples every Nth image,
and `--max-frames` bounds sampled frames, including featureless images. The trainer
uses the app's extractor (eight pyramid levels, scale factor 1.2, FAST thresholds
20/7), supports the same detector/descriptor names, and defaults to FAST + ORB.
Training stores at most `max-frames * features` descriptors in RAM; clustering and
the tree consume additional memory. The default tree has at most 100,000 words;
larger branching/depth values increase file size, training time, and memory use.
Compact trees must be evaluated for loop-closure/relocalization recall, not just size.

For a standalone SIFT vocabulary, choose `--descriptor sift` and a distinct output
such as `vocab/corridor-sift.dbow3`. SURF requires OpenCV nonfree support. Output
always uses compressed binary storage, TF-IDF weighting, and L1 scoring. Training
checks a binary round-trip against every sampled image and refuses to overwrite an
existing file. `--seed` defaults to 42; `BUILD_TYPE=Debug` selects a Debug trainer.
Use `bash tools/generate_vocabulary.sh --help` for all options.

Train on representative, varied imagery, preferably separate from evaluation
sequences; a single corridor can overfit. New vocabularies change word identities:
start a fresh atlas and keep the chosen file unchanged during playback.

## Logging

Set the spdlog minimum level and output pattern in the settings YAML:

```yaml
logging:
  level: "info"
  pattern: "[%H:%M:%S] [%^%l%$] %v"
```

Override either value independently with `--logging.level warning` or
`--logging.pattern '[%H:%M:%S] [%^%l%$] %v'`. Levels are `trace`, `debug`,
`info` (default), `warning`, `error`, `critical`, and `off`; spdlog aliases
`warn` and `err` are also accepted. Invalid levels are rejected.
The viewer's **Logging** controls start with the resolved YAML/CLI settings and
change the active level or pattern at runtime. Apply a pattern with the checkmark
or Enter. Runtime changes do not rewrite the YAML file.
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
- Independently configurable OpenCV detectors and descriptors, including contrib xfeatures2d algorithms, and matchers (brute force, FLANN). The existing vocabulary uses a separate ORB place-recognition descriptor channel; see the root README's Feature Pipeline section for supported types and build requirements.
- Optional flow-first tracking with descriptor recovery through `tracking.use_optical_flow` in YAML or `--tracking.use_optical_flow` on the command line.
- Median and mean tracking-time summary after each run.

The executable currently does not provide live-camera, RGB-D, or ROS input. Those capabilities described in the upstream ORB-SLAM3 documentation are outside this app's current input interface.

## TUM-VI Stereo Benchmark

[`orbslam3_tests/benchmark/tumvi_benchmark.py`](../orbslam3_tests/benchmark/tumvi_benchmark.py)
runs all extracted `dataset-*` directories under `.data/tumvi` sequentially,
headless, in stereo mode, processing every frame. Build the app first and install
the benchmark dependencies in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r orbslam3_tests/benchmark/requirements.txt
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
  New CSVs also report `extraction_ms` (left/right extraction wall time, including
  their overlap), `stereo_matching_ms` (matching and triangulation),
  `optical_flow_ms` (forward/backward LK, including lazy pyramid construction),
  `input_wait_ms` (waiting for prefetch), and `viewer_publication_ms` (camera image
  and pose handoff on the playback thread). The first three are frontend
  sub-stages, not additional time; extraction also includes pyramid generation.
  Input preprocessing overlaps tracking, so do not sum these columns to estimate
  throughput. Older CSVs without the optional columns remain supported.
- Tracking quality: separate left/right feature counts, flow candidates,
  surviving tracks, geometric flow inliers, and occupied cells in a 4x4 grid.
  The original `detected_keypoints` column remains the sum of left and right
  observations; flow candidates are left-camera map-point tracks, not detections.
  CSV also records `flow_only`, `features_extracted`, `tracking_state` (2: OK,
  3: recently lost, 4: lost), `map_id`, and `map_change_index`. Summaries count
  tracking-loss transitions, map switches, feature extraction frames, and major
  map corrections. Corrections can include loop closure, merging, or bundle
  adjustment; they are not a count of loop detections. Detailed events remain
  in `app.log`. Preprocessing time includes synchronization waits and contention,
  so it is not an isolated measurement of image-processing CPU time.
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

### Stereo Throughput

The TUM-VI sample uses `tracking.matcher: brute_force`: at about 1,000 binary
descriptors per image, exact Hamming matching avoids rebuilding a FLANN LSH
index every frame. Override with `--tracking.matcher flann` for comparisons;
larger descriptor sets and floating-point descriptors need separate measurements.
Fisheye triangulation uses up to four OpenCV parallel stripes for at least 128
queries; smaller sets stay serial. Reverse match conflicts are committed in the
original order, preserving deterministic results. Left/right feature extraction,
image prefetch, local mapping, and loop closing already overlap on worker threads.

The viewer already runs on a separate `std::jthread` at 30 FPS. Camera-image
publication is demand-driven and uses `try_lock` to drop stale display updates;
map snapshots coalesce per map, and camera poses queue for the viewer to drain.
No live tracker object is passed to the renderer. Reset/localization commands
are intentionally handled by playback between frames. A small
`viewer_publication_ms` with a large viewer-on throughput penalty indicates
resource contention, not a blocking image handoff. Compare `--no-viewer` first,
especially with software OpenGL rendering. Another unbounded display queue
would retain stale frames rather than solve that contention.

Pyramids are built once **per image and layout**, not once globally: each camera
has a feature pyramid at `keypoint_extractor.scale_factor` (normally 1.2), while
left-image LK uses a separate 2x pyramid with interleaved derivatives and different
border requirements. Prefetch builds the LK pyramid, playback transfers it into
the frame, and forward/backward LK and the next frame reuse that cache. Lazy
construction covers library callers without prefetch. Feature refresh preserves
the LK cache, and a new timestamp clears it. These layouts cannot be substituted
without changing feature scales or LK behavior.

Right-image temporal flow and left-to-right LK stereo are not enabled by these
changes. They could save right extraction only with calibrated fisheye projection
seeds, forward/backward checks, geometric triangulation/reprojection validation,
spatial coverage checks, and descriptor recovery on keyframes or tracking loss.
Running additional LK while retaining both extractions adds work; measure it
against exact stereo matching before accepting that accuracy tradeoff.

To compare stereo SLAM with and without optical-flow tracking on every locally
extracted TUM-VI sequence, run:

```sh
.venv/bin/python orbslam3_tests/benchmark/tumvi_benchmark.py \
  --compare-optical-flow --repetitions 3 --run-name stereo-optical-flow-v2
```

This runs three pairs by default, each covering the complete selected dataset
with `tracking.use_optical_flow: 0` and `tracking.use_optical_flow: 1`, alternating
off/on then on/off execution order to reduce ordering bias. It overrides the template
while preserving its other feature/tracking settings. Both use the same binary,
vocabulary, frame selection, and stereo mode without IMU. Install all desired
sequences under `--datasets-root` first; this command does not download datasets.
Use a unique run name for each experiment and keep the machine otherwise idle.
The paired run snapshots the settings and checks the executable and vocabulary
hashes before each variant; do not rebuild or edit inputs during a benchmark.

Results are saved under `.data/benchmark/tumvi/stereo-optical-flow-v2/`, with
`repetition_01/`, `repetition_02/`, etc. Each repetition contains
`without_optical_flow/`, `with_optical_flow/`, and its comparison reports.
The parent `comparison.md`, `comparison.csv`, and `comparison.json` contain
medians and min/max ranges across repetitions, including median paired deltas:
baseline is flow off, candidate is flow on. Metrics missing from any repetition
are omitted with a warning instead of averaging only survivors. With
`--repetitions 1`, variant directories and the single comparison live directly
under the run directory, preserving the original layout.
Both modes are attempted even if individual sequences
fail; any sequence failure returns a nonzero exit status. Ctrl+C stops the batch
and preserves completed reports. Neither repeated medians nor a single pair
establish statistical significance.

Stereo optical flow is accuracy-first: it estimates a pose from up to 1,024
spatially distributed established map-point tracks, checks geometric inliers
and coverage, then refreshes features and stereo depth in both images on every
frame. Descriptor matching and local-map optimization still run on the refreshed
features. Weak flow falls back to the motion prediction and normal matching.
This keeps feature counts comparable to flow-off tracking and retains stereo
constraints, at the cost of much of the earlier flow-only speedup. Monocular
tracking retains its bounded flow-only path with periodic/quality-based refresh.

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
per-sequence metrics remain available. New runs archive matched ground-truth
timestamps, reference positions, and unaligned body-frame estimates in
`accuracy_samples.csv`. Comparisons intersect these timestamps and independently
SE(3)-align both estimates on exactly that subset, reporting `shared_ate_rmse_m`
and `shared_matched_poses`. Prefer shared ATE when tracking coverage differs;
ordinary ATE and coverage remain visible. Legacy runs without archived samples
remain comparable, with a warning that shared-timestamp accuracy is unavailable.
Dataset, host, vocabulary, timeout, source provenance, and executable
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
