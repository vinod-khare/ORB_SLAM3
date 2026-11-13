# ORB-SLAM3

### V1.0, December 22th, 2021
**Authors:** Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, [José M. M. Montiel](http://webdiis.unizar.es/~josemari/), [Juan D. Tardós](http://webdiis.unizar.es/~jdtardos/).

The [Changelog](Changelog.md) describes the upstream release history.

ORB-SLAM3 is a SLAM library for visual, visual-inertial, and multi-map SLAM. The library supports monocular, stereo, and RGB-D sensors with pinhole and fisheye camera models. This repository's app is an offline, folder-based runner for monocular and stereo image sequences, optionally paired with IMU data. It does not currently provide live-camera, RGB-D, or ROS input.

See the [app guide](orbslam3_app/README.md) for build setup, dataset formats, calibration, runtime options, exports, and benchmarks.

This software is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), developed by Raul Mur-Artal, Juan D. Tardós, J. M. M. Montiel, and Dorian Galvez-Lopez ([DBoW2](https://github.com/dorian3d/DBoW2)).

<a href="https://youtu.be/HyLNq-98LRo" target="_blank"><img src="https://img.youtube.com/vi/HyLNq-98LRo/0.jpg" alt="ORB-SLAM3" width="240" height="180" /></a>

## Related Publications

[ORB-SLAM3] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, *IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021*. [PDF](https://arxiv.org/abs/2007.11898).

[IMU-Initialization] Carlos Campos, J. M. M. Montiel and Juan D. Tardós, **Inertial-Only Optimization for Visual-Inertial Initialization**, *ICRA 2020*. [PDF](https://arxiv.org/pdf/2003.05766.pdf).

[ORBSLAM-Atlas] Richard Elvira, J. M. M. Montiel and Juan D. Tardós, **ORBSLAM-Atlas: a robust and accurate multi-map system**, *IROS 2019*. [PDF](https://arxiv.org/pdf/1908.11585.pdf).

[ORBSLAM-VI] Raúl Mur-Artal and Juan D. Tardós, **Visual-inertial monocular SLAM with map reuse**, *IEEE Robotics and Automation Letters*, vol. 2 no. 2, pp. 796-803, 2017. [PDF](https://arxiv.org/pdf/1610.05949.pdf).

[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós, **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**, *IEEE Transactions on Robotics*, vol. 33, no. 5, pp. 1255-1262, 2017. [PDF](https://arxiv.org/pdf/1610.06475.pdf).

[Monocular] Raúl Mur-Artal, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**, *IEEE Transactions on Robotics*, vol. 31, no. 5, pp. 1147-1163, 2015. [PDF](https://arxiv.org/pdf/1502.00956.pdf).

[DBoW2 Place Recognition] Dorian Gálvez-López and Juan D. Tardós, **Bags of Binary Words for Fast Place Recognition in Image Sequences**, *IEEE Transactions on Robotics*, vol. 28, no. 5, pp. 1188-1197, 2012. [PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf).

## License

ORB-SLAM3 is released under the [GPLv3 license](LICENSE). Install dependencies using [`install_dependencies.sh`](install_dependencies.sh). For commercial licensing inquiries, contact the authors at orbslam (at) unizar (dot) es.

If you use ORB-SLAM3 in academic work, cite:

```bibtex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map {SLAM}},
  author={Campos, Carlos AND Elvira, Richard AND G\'omez, Juan J. AND Montiel, Jos\'e M. M. AND Tard\'os, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={6},
  pages={1874-1890},
  year={2021}
}
```

## Prerequisites

Building requires a C++23-capable compiler and CMake 3.25 or newer. The dependency setup script installs the project's system and vcpkg dependencies. When using a separate vcpkg checkout, provide its toolchain file during CMake configuration.

## Build

Configure and build from the repository root:

```sh
cmake -S . -B .build/Release -DCMAKE_BUILD_TYPE=Release
cmake --build .build/Release -j20
```

For vcpkg, add `-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake` to the configure command. The build produces the `ORB_SLAM3` library and folder-based dataset runner. DBoW3 is optional: when found, the build also enables its backend and the vocabulary trainer. Otherwise DBoW2 remains available, including support for the shipped compressed vocabularies. Selecting `dbow_version: 3` or `--dbow_version 3` without DBoW3 reports `Code not compiled with DBoW3`.

See the [app guide](orbslam3_app/README.md) for a quick start and complete application instructions.

## Formatting C++ Code

Run `./format.sh` to format C++ sources, headers, and inline/template files using the repository's `.clang-format`. Install `clang-format` and ensure it is available on `PATH` first. The script works from any working directory and includes tracked and non-ignored untracked files, skipping deleted files and symbolic links. Ignored build and dependency directories are excluded.

## Feature Pipeline

Monocular initialization uses OpenCV RANSAC homography/fundamental fitting, motion decomposition, perspective/epipolar transforms, and triangulation. The existing symmetric model scores, motion-ambiguity checks, reprojection limits, and minimum support/parallax requirements remain in place. The score-selected model is tried first; if its reconstruction fails, the other valid model is tried with the same acceptance checks. This handles planar scenes whose fundamental fit does not uniquely determine motion. Returned points retain reference-keypoint indexing and unit-baseline scale. OpenCV's sampling, normalization, and homography hypotheses differ from the original implementation, so initialization decisions and trajectories need not be identical.

Detection, description, and matching are independently selectable. Descriptors are computed by OpenCV.

`Tracking` owns the extractors and supplies keypoints, descriptors, and scale metadata to `frame` for all sensor modes, including deferred optical-flow recovery. Frames retain neither camera models nor calibration: camera models and IMU calibration are passed to operations that need them. Stereo extraction runs concurrently, and its image pyramids are passed explicitly for stereo depth computation; frames retain no extractor references.

Keyframes retain neither camera models nor camera calibration. Geometry operations receive cameras explicitly; keyframes retain only the stereo observation layout copied from their source frame. Tracking, mapping, and loop closing supply the shared camera models to matching and optimization. Intrinsics, distortion, stereo baseline/extrinsics, and the metric close-depth cutoff (`baseline * Stereo.ThDepth`) come from those models, which must outlive the processing workers and solvers that use them.

Each `keypoint` stores original image coordinates in `pt` and geometry coordinates in `pt_undistorted`, without a separate undistorted keypoint vector in frames or keyframes. Rectified and native fisheye observations use the original coordinates for both fields. `GetTrackedKeyPointsUn()` continues to return undistorted coordinates in `pt`.

```yaml
keypoint:
  detector:
    type: FAST
    n_features: 1024
    scale_factor: 1.2
    n_levels: 8
    initial_threshold: 20
    minimum_threshold: 7
  descriptor:
    type: SIFT
tracking:
  matcher: brute_force
```

- Detectors: `FAST`, `AGAST`, `GFTT`, `ORB`, `BRISK`, `SIFT`, `AKAZE`, `STAR`, `HARRIS_LAPLACE`, `MSD`, `TBMR`, `SURF` (case-insensitive).
- Binary descriptors: `ORB` (32 bytes, default), `BRISK` (64), `BRIEF` (32, orientation-aware), `FREAK` (64), `LATCH` (32), `BEBLID` (64), `TEBLID` (32), `BOOSTDESC` (32, BINBOOST-256).
- Floating-point descriptors: `SIFT` (128 values), `SURF` (64), `DAISY` (200, full normalization with orientation), `VGG` (120). All supported descriptors can describe points from any supported detector. SURF detection and description require OpenCV nonfree support.
- AKAZE descriptors are not offered because OpenCV requires AKAZE-specific scale-space metadata. LUCID requires color input, while the extractor receives grayscale images. AffineFeature2D wrappers, image-level PCT signatures, and module matching utilities are not standalone detector/descriptor selections. TBMR supplies ordinary keypoints; affine regions are not retained.
- Matchers: `brute_force` (default) or `flann`. Binary descriptors use Hamming distance and FLANN LSH; floating-point descriptors use L2 distance and FLANN KD-trees. SLAM projection, pyramid-level, epipolar, ratio, and map-point checks remain in place around OpenCV candidate ranking.
- Exhaustive ranking inside geometric/vocabulary candidate sets uses OpenCV brute force even in FLANN mode so all candidates remain available for geometric filtering. Unconstrained nearest-neighbor searches use the selected backend.
- Detectors run across the feature pyramid with per-level spatial distribution. FAST/AGAST use the configured corner thresholds; ORB/BRISK use `initial_threshold`. Scale-space detectors use one internal scale/octave where possible; other parameters use OpenCV defaults. BEBLID/TEBLID and VGG/BoostDesc sampling factors match the extractor's normalized 31-pixel keypoint size.
- Place recognition uses the selected descriptor, with no separate descriptor channel. Both DBoW2 and DBoW3 accept binary (`CV_8U`) and floating-point (`CV_32F`) vocabularies; the vocabulary must be trained for the selected descriptor type and settings.
- Vocabulary-guided matching retains at least two tree levels below the root (or uses leaves in shallower trees). This preserves the six-level ORB vocabulary's existing grouping while preventing shallow vocabularies, such as the four-level SIFT tree, from collapsing all features into one exhaustive matching bucket. BoW words and weights are unchanged.
- Internal acceptance thresholds use 32-byte Hamming-equivalent units: binary Hamming distances are multiplied by `32 / descriptor_bytes`; floating-point L2 distances are divided by four. New floating-point descriptors are normalized to L2 magnitude 512. Descriptor choices can require accuracy tuning and do not guarantee equivalent trajectories.
- Saved atlases retain only the selected descriptor channel. Older ORB atlases can be loaded; an atlas cannot be reused with a different tracking-descriptor format. New atlases require this version of the code.

Command-line selections override YAML:

```sh
.build/Release/orbslam3_app/orbslam3_app vocab/ORBvoc.dbow3 config/TUM-VI.yaml \
  --slam-type stereo --keypoint.detector.type star \
  --keypoint.descriptor.type orb --tracking.matcher flann
```

When selecting a non-ORB descriptor, use a vocabulary trained for that same descriptor. For the available DBoW backends, supported vocabulary format, and runtime switching, see [DBoW place recognition in the app guide](orbslam3_app/README.md#dbow-place-recognition).

The supplied [TUM-VI configuration](config/TUM-VI.yaml) selects SIFT, so
[`run.sh`](run.sh) uses `vocab/TUMVI_SIFT.dbow3`. To run it headlessly on a
limited number of frames:

```sh
.build/Release/orbslam3_app/orbslam3_app vocab/TUMVI_SIFT.dbow3 config/TUM-VI.yaml \
  --slam-type stereo --no-viewer --frames-take 100 --output-folder .build/sift-smoke
```

Changing the descriptor also requires changing the vocabulary passed to the app.
An ORB/SIFT format mismatch is rejected before tracking or the viewer starts; it
is not a failure to detect keypoints.
