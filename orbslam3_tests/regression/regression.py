#!/usr/bin/env python3
"""End-to-end accuracy regression: run orbslam3_app headless and compare ATE against a stored baseline."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app", type=Path, default=REPO / ".build/Release/orbslam3_app/orbslam3_app")
    p.add_argument("--vocab", type=Path, default=REPO / "vocab/ORBvoc.txt")
    p.add_argument("--settings", type=Path, default=REPO / "config/TUM-VI.yaml")
    p.add_argument("--slam-type", default="stereo")
    p.add_argument("--output-folder", type=Path, default=REPO / ".data/regression")
    p.add_argument("--baseline", type=Path, default=Path(__file__).with_name("baseline.yaml"))
    p.add_argument("--update-baseline", action="store_true", help="Overwrite baseline metrics with this run")
    p.add_argument("--skip-run", action="store_true", help="Re-evaluate the existing trajectory without running SLAM")
    return p.parse_args()


def load_settings(path):
    # OpenCV-style "%YAML:1.0" header is not valid for PyYAML.
    text = path.read_text()
    if text.startswith("%YAML:"):
        text = text.split("\n", 1)[1]
    return yaml.safe_load(text)


def load_estimate(path, t_cam_imu):
    # Rows: timestamp_ns tx ty tz qx qy qz qw, poses are T_world_cam0 (non-inertial modes).
    data = np.loadtxt(path, ndmin=2)
    traj = PoseTrajectory3D(positions_xyz=data[:, 1:4], orientations_quat_wxyz=data[:, [7, 4, 5, 6]], timestamps=data[:, 0] / 1e9)
    # Ground truth is the IMU body pose, so express the estimate as T_world_body = T_world_cam * T_cam_imu.
    traj.transform(t_cam_imu, right_mul=True)
    return traj


def evaluate(args, matched_output=None):
    settings = load_settings(args.settings)
    settings_dir = args.settings.resolve().parent
    data_root = settings_dir / settings["data"]["root"]
    camchain = yaml.safe_load((settings_dir / settings["calibration"]["file"]).read_text())
    t_cam_imu = np.array(camchain["cam0"]["T_cam_imu"])
    num_images = sum(1 for _ in (data_root / settings["data"]["left"]).glob("*.png"))

    with (args.output_folder / "statistics.csv").open(newline="") as statistics_file:
        statistics = list(csv.DictReader(statistics_file))
    if len(statistics) != num_images:
        raise ValueError(f"Expected {num_images} statistics rows, got {len(statistics)}")
    for frame_number, sample in enumerate(statistics, start=1):
        if int(sample["frames"]) != frame_number or not all(np.isfinite(float(value)) for value in sample.values()):
            raise ValueError(f"Invalid statistics for frame {frame_number}")

    est = load_estimate(args.output_folder / "f_trajectory.txt", t_cam_imu)
    tracked_ratio = est.num_poses / num_images
    ref = file_interface.read_euroc_csv_trajectory(str(data_root / "mocap0/data.csv"))
    ref, est = sync.associate_trajectories(ref, est, max_diff=0.01)
    if matched_output is not None:
        np.savetxt(
            matched_output,
            np.column_stack((ref.timestamps, ref.positions_xyz, est.positions_xyz)),
            delimiter=",",
            header="timestamp,reference_x,reference_y,reference_z,estimate_x,estimate_y,estimate_z",
        )
    est.align(ref, correct_scale=args.slam_type == "mono")

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, est))
    return {"ate_rmse_m": float(ape.get_statistic(metrics.StatisticsType.rmse)), "tracked_ratio": float(tracked_ratio), "matched_poses": int(ref.num_poses)}


def main():
    args = parse_args()
    if not args.skip_run:
        cmd = [str(args.app), "--vocab", str(args.vocab), "--settings", str(args.settings), "--slam-type", args.slam_type, "--output-folder", str(args.output_folder), "--output", "trajectory", "--no-viewer"]
        subprocess.run(cmd, check=True)

    result = evaluate(args)
    print(f"[{args.slam_type}] {result}")

    baseline = yaml.safe_load(args.baseline.read_text()) if args.baseline.exists() else {}
    if args.update_baseline:
        baseline.setdefault("tolerance", {"ate_rel": 0.25, "tracked_abs": 0.05})
        baseline.setdefault("metrics", {})[args.slam_type] = result
        args.baseline.write_text(yaml.safe_dump(baseline, sort_keys=False))
        print(f"Baseline updated: {args.baseline}")
        return 0

    expected = baseline.get("metrics", {}).get(args.slam_type)
    if expected is None:
        print(f"No baseline for '{args.slam_type}' in {args.baseline}; run with --update-baseline", file=sys.stderr)
        return 1

    tol = baseline["tolerance"]
    failures = []
    if result["ate_rmse_m"] > expected["ate_rmse_m"] * (1 + tol["ate_rel"]):
        failures.append(f"ATE RMSE {result['ate_rmse_m']:.4f} m > baseline {expected['ate_rmse_m']:.4f} m (+{tol['ate_rel']:.0%})")
    if result["tracked_ratio"] < expected["tracked_ratio"] - tol["tracked_abs"]:
        failures.append(f"tracked ratio {result['tracked_ratio']:.3f} < baseline {expected['tracked_ratio']:.3f} (-{tol['tracked_abs']})")

    for f in failures:
        print(f"REGRESSION: {f}", file=sys.stderr)
    if not failures and result["ate_rmse_m"] < expected["ate_rmse_m"] * (1 - tol["ate_rel"]):
        print("Accuracy improved beyond tolerance; consider --update-baseline")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
