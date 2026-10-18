#!/usr/bin/env python3
"""Compare odometry accuracy (ATE vs mocap) and speed across keypoint detectors; prints a Markdown table."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression"))
import regression  # noqa: E402

REPO = regression.REPO
ALL_DETECTORS = "fast,agast,gftt,orb,brisk,sift,akaze"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app", type=Path, default=REPO / ".build/Release/orbslam3_app/orbslam3_app")
    p.add_argument("--vocab", type=Path, default=REPO / "vocab/ORBvoc.txt")
    p.add_argument("--settings", type=Path, default=REPO / "config/TUM-VI.yaml")
    p.add_argument("--benchmark.slam_type", dest="slam_type", default="stereo")
    p.add_argument("--benchmark.detectors", dest="detectors", default=ALL_DETECTORS, help="Comma-separated detector names")
    p.add_argument("--benchmark.output_folder", dest="output_folder", type=Path, default=REPO / ".data/benchmark")
    p.add_argument("--benchmark.skip_run", dest="skip_run", action="store_true", help="Re-evaluate existing trajectories without running SLAM")
    return p.parse_args()


def run_detector(args, detector):
    out = args.output_folder / detector
    log = out / "app.log"
    if not args.skip_run:
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(args.app), "--vocab", str(args.vocab), "--settings", str(args.settings), "--slam-type", args.slam_type, "--orb_extractor.detector", detector,
               "--output-folder", str(out), "--output", "trajectory", "--no-viewer"]
        with log.open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    text = log.read_text(errors="replace")
    timing = {k: float(m.group(1)) * 1e3 for k, pat in (("median_ms", r"median tracking time: (\S+)"), ("mean_ms", r"mean tracking time: (\S+)")) if (m := re.search(pat, text))}
    fps = re.search(r"\(([\d.]+) frames/s\)", text)
    timing["fps"] = float(fps.group(1)) if fps else float("nan")
    accuracy = regression.evaluate(argparse.Namespace(settings=args.settings, output_folder=out, slam_type=args.slam_type))
    return {**timing, **accuracy}


def main():
    args = parse_args()
    rows = []
    for detector in args.detectors.split(","):
        try:
            rows.append((detector, run_detector(args, detector)))
        except Exception as e:  # a failing detector must not hide the others
            print(f"{detector} failed: {e}", file=sys.stderr)
            rows.append((detector, None))

    print(f"Slam type: `{args.slam_type}`, settings: `{args.settings.name}`\n")
    print("| Detector | ATE RMSE (m) | Tracked ratio | Mean track (ms) | Median track (ms) | Playback (frames/s) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for detector, r in rows:
        if r is None:
            print(f"| {detector} | failed | - | - | - | - |")
        else:
            print(f"| {detector} | {r['ate_rmse_m']:.4f} | {r['tracked_ratio']:.3f} | {r['mean_ms']:.2f} | {r['median_ms']:.2f} | {r['fps']:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
