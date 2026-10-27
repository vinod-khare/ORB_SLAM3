#!/usr/bin/env python3
"""Run headless stereo SLAM on locally extracted TUM-VI sequences."""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression"))
import regression  # noqa: E402

REPO = regression.REPO
FRAME_FIELDS = (
    "detected_keypoints",
    "matched_keypoints",
    "preprocessing_ms",
    "frontend_ms",
)
WORKERS = ("local_mapping", "loop_closing", "global_bundle_adjustment")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=REPO / ".data/tumvi")
    parser.add_argument(
        "--app", type=Path, default=REPO / ".build/Release/orbslam3_app/orbslam3_app"
    )
    parser.add_argument("--vocab", type=Path, default=REPO / "vocab/ORBvoc.txt")
    parser.add_argument("--settings", type=Path, default=REPO / "config/TUM-VI.yaml")
    parser.add_argument(
        "--output-folder", type=Path, default=REPO / ".data/benchmark/tumvi"
    )
    parser.add_argument(
        "--sequence",
        action="append",
        help="Sequence directory name; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Evaluate existing outputs without running SLAM",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previous per-sequence run artifacts",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Per-sequence timeout in seconds (0: unlimited)",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    return args


def discover_sequences(root, names=None):
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")
    sequences = sorted(path for path in root.glob("dataset-*") if path.is_dir())
    if names:
        missing = set(names) - {path.name for path in sequences}
        if missing:
            raise ValueError(f"Unknown sequences: {', '.join(sorted(missing))}")
        sequences = [path for path in sequences if path.name in names]
    if not sequences:
        raise ValueError(f"No dataset-* sequence directories found in {root}")
    return sequences


def write_settings(template, sequence, output):
    settings = regression.load_settings(template)
    root = sequence / "mav0"
    required = (root / "cam0/data", root / "cam1/data", sequence / "dso/camchain.yaml")
    for path in required:
        if not path.exists():
            raise ValueError(f"Missing sequence input: {path}")
    if not any((root / "cam0/data").glob("*.png")) or not any(
        (root / "cam1/data").glob("*.png")
    ):
        raise ValueError(f"Empty stereo image directory: {sequence}")
    settings["data"] = {
        "root": str(root),
        "left": "cam0/data",
        "right": "cam1/data",
        "imu": "imu0/data.csv",
    }
    settings["calibration"] = {"file": str(sequence / "dso/camchain.yaml")}
    imu = sequence / "dso/imu_config.yaml"
    if imu.is_file():
        settings["calibration"]["imu_file"] = str(imu)
    settings["ground_truth"] = {"path": str(root / "mocap0/data.csv"), "visible": False}
    path = output / "settings.yaml"
    path.write_text("%YAML:1.0\n---\n" + yaml.safe_dump(settings, sort_keys=False))
    return path


def read_statistics(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No frame statistics in {path}")
    samples = {field: [] for field in FRAME_FIELDS}
    for index, row in enumerate(rows, start=1):
        if int(row["frames"]) != index or not all(
            np.isfinite(float(value)) for value in row.values()
        ):
            raise ValueError(f"Invalid statistics row {index} in {path}")
        for field in FRAME_FIELDS:
            value = float(row[field])
            if value < 0:
                raise ValueError(f"Negative {field} in row {index} of {path}")
            samples[field].append(value)
    last = rows[-1]
    counters = {
        field: float(last[field])
        for field in ("elapsed_seconds", "preprocessing_total_ms", "frontend_total_ms")
    }
    for worker in WORKERS:
        counters[f"{worker}_jobs"] = int(last[f"{worker}_jobs"])
        counters[f"{worker}_total_ms"] = float(last[f"{worker}_total_ms"])
    return samples, counters


def summarize(samples, counters):
    result = {"frames": len(samples[FRAME_FIELDS[0]]), **counters}
    elapsed = counters["elapsed_seconds"]
    result["throughput_fps"] = result["frames"] / elapsed if elapsed > 0 else None
    for field, values in samples.items():
        for name, value in (
            ("mean", np.mean(values)),
            ("median", np.median(values)),
            ("p95", np.percentile(values, 95)),
            ("min", min(values)),
            ("max", max(values)),
        ):
            result[f"{field}_{name}"] = float(value)
    return result


def run_sequence(args, sequence):
    output = args.output_folder / sequence.name
    output.mkdir(parents=True, exist_ok=True)
    settings = output / "settings.yaml"
    wall_seconds = None
    if not args.skip_run:
        artifacts = [
            output / name
            for name in (
                "statistics.csv",
                "f_trajectory.txt",
                "kf_trajectory.txt",
                "app.log",
                "settings.yaml",
            )
        ]
        if any(path.exists() for path in artifacts) and not args.overwrite:
            raise ValueError(f"Existing run in {output}; use --overwrite or --skip-run")
        for path in artifacts:
            path.unlink(missing_ok=True)
        settings = write_settings(args.settings, sequence, output)
        command = [
            str(args.app),
            "--vocab",
            str(args.vocab),
            "--settings",
            str(settings),
            "--slam-type",
            "stereo",
            "--output-folder",
            str(output),
            "--output",
            "trajectory",
            "--no-viewer",
            "--frames-take",
            "0",
        ]
        started = time.monotonic()
        with (output / "app.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout or None,
            )
        wall_seconds = time.monotonic() - started
    samples, counters = read_statistics(output / "statistics.csv")
    expected = sum(1 for _ in (sequence / "mav0/cam0/data").glob("*.png"))
    if len(samples[FRAME_FIELDS[0]]) != expected:
        raise ValueError(
            f"Expected {expected} frames, got {len(samples[FRAME_FIELDS[0]])}"
        )
    result = {
        "sequence": sequence.name,
        "status": "ok",
        "wall_seconds": wall_seconds,
        **summarize(samples, counters),
    }
    try:
        result.update(
            regression.evaluate(
                argparse.Namespace(
                    settings=settings, output_folder=output, slam_type="stereo"
                )
            )
        )
    except Exception as error:
        result.update(status="accuracy_failed", error=str(error))
    return result, samples, counters


def write_reports(output, results, pooled, totals):
    aggregate = summarize(pooled, totals) if pooled[FRAME_FIELDS[0]] else {}
    successful = [row for row in results if row["status"] == "ok"]
    if successful:
        matched = sum(row["matched_poses"] for row in successful)
        aggregate["ate_rmse_m"] = float(
            np.sqrt(
                sum(row["ate_rmse_m"] ** 2 * row["matched_poses"] for row in successful)
                / matched
            )
        )
        aggregate["matched_poses"] = matched
        aggregate["tracked_ratio"] = sum(
            row["tracked_ratio"] * row["frames"] for row in successful
        ) / sum(row["frames"] for row in successful)
    aggregate["statistics_sequences"] = sum("frames" in row for row in results)
    aggregate["accuracy_sequences"] = len(successful)
    aggregate["failed_sequences"] = sum(row["status"] != "ok" for row in results)
    (output / "summary.json").write_text(
        json.dumps(
            {"sequences": results, "aggregate": aggregate}, indent=2, allow_nan=False
        )
        + "\n"
    )
    columns = list(dict.fromkeys(key for row in results for key in row))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "| Sequence | Status | Frames | ATE RMSE (m) | Tracked ratio | Frontend mean / p95 (ms) | Preprocess mean (ms) | FPS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results + [
        {
            "sequence": "ALL (pooled)",
            "status": f"{len(successful)}/{len(results)} accuracy",
            **aggregate,
        }
    ]:

        def display(key):
            value = row.get(key)
            return f"{value:.3f}" if value is not None else "-"

        lines.append(
            f"| {row['sequence']} | {row['status']} | {row.get('frames', '-')} | {display('ate_rmse_m')} | {display('tracked_ratio')} | {display('frontend_ms_mean')} / {display('frontend_ms_p95')} | {display('preprocessing_ms_mean')} | {display('throughput_fps')} |"
        )
    lines.extend(
        [
            "",
            "ATE: timestamp-associated (10 ms), body-frame translation error after per-sequence SE(3) alignment; no scale correction. TUM-VI mocap covers only parts of most sequences.",
            "",
            "Pooled ATE weights each successful sequence by matched poses; frame distributions pool all complete statistics runs. Worker counters are last-frame snapshots, exclude work completed after that snapshot, and overlap in time.",
        ]
    )
    for row in results:
        if "error" in row:
            lines.extend(["", f"{row['sequence']}: {row['error']}"])
    report = "\n".join(lines) + "\n"
    (output / "summary.md").write_text(report)
    return report


def main(argv=None):
    args = parse_args(argv)
    for field in ("datasets_root", "app", "vocab", "settings", "output_folder"):
        setattr(args, field, getattr(args, field).resolve())
    sequences = discover_sequences(args.datasets_root, args.sequence)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    results = []
    pooled = {field: [] for field in FRAME_FIELDS}
    totals = {}
    try:
        for index, sequence in enumerate(sequences, start=1):
            print(f"[{index}/{len(sequences)}] {sequence.name}", flush=True)
            try:
                result, samples, counters = run_sequence(args, sequence)
                for field in FRAME_FIELDS:
                    pooled[field].extend(samples[field])
                for field, value in counters.items():
                    totals[field] = totals.get(field, 0) + value
            except Exception as error:
                result = {
                    "sequence": sequence.name,
                    "status": "failed",
                    "error": str(error),
                }
                print(f"{sequence.name}: {error}", file=sys.stderr)
            results.append(result)
            write_reports(args.output_folder, results, pooled, totals)
    except KeyboardInterrupt:
        print("Interrupted; completed sequence reports are preserved.", file=sys.stderr)
        return 130
    print((args.output_folder / "summary.md").read_text())
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    sys.exit(main())
