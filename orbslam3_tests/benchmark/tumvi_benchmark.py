#!/usr/bin/env python3
"""Record and compare headless stereo SLAM runs on local TUM-VI sequences."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
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
        "--run-name", help="Unique run directory name (default: UTC timestamp)"
    )
    parser.add_argument(
        "--notes", default="", help="Description of the change being tested"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE", "CANDIDATE"),
        help="Compare two saved run directories without running SLAM",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="Write comparison.json, .csv and .md in this directory",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        help="Sequence directory name; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Evaluate existing outputs; --output-folder must name an existing run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated: runs are isolated and existing run names cannot be overwritten",
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
    if args.compare and (
        args.skip_run or args.run_name or args.sequence or args.overwrite
    ):
        parser.error("--compare cannot be combined with run or re-evaluation options")
    if args.comparison_output and not args.compare:
        parser.error("--comparison-output requires --compare")
    return args


def file_hash(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_signature(sequence):
    signature = {
        name: file_hash(sequence / name)
        for name in (
            "dso/camchain.yaml",
            "dso/imu_config.yaml",
            "mav0/mocap0/data.csv",
            "mav0/cam0/data.csv",
            "mav0/cam1/data.csv",
            "mav0/imu0/data.csv",
        )
    }
    for camera in ("cam0", "cam1"):
        inventory = [
            (path.name, path.stat().st_size)
            for path in sorted((sequence / "mav0" / camera / "data").glob("*.png"))
        ]
        signature[camera] = {
            "frames": len(inventory),
            "inventory_sha256": hashlib.sha256(
                json.dumps(inventory).encode()
            ).hexdigest(),
        }
    return signature


def create_run(args, sequences):
    started = datetime.now(timezone.utc).isoformat()
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if name in (".", "..") or Path(name).name != name or not name:
        raise ValueError("--run-name must be a single directory name")
    output = args.output_folder / name
    output.mkdir(parents=True, exist_ok=False)
    args.output_folder = output

    def git(*arguments):
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(REPO), *arguments], stderr=subprocess.DEVNULL
                )
                .decode("utf-8", errors="replace")
                .rstrip()
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    patch = git("diff", "HEAD", "--binary")
    if patch:
        (output / "source.patch").write_text(patch + "\n")
    settings = regression.load_settings(args.settings)
    (output / "template.yaml").write_text(
        "%YAML:1.0\n---\n" + yaml.safe_dump(settings, sort_keys=False)
    )
    manifest = {
        "schema_version": 1,
        "run_id": name,
        "started_at": started,
        "finished_at": None,
        "status": "running",
        "notes": args.notes,
        "command": sys.argv,
        "sequences": [sequence.name for sequence in sequences],
        "datasets_root": str(args.datasets_root),
        "datasets": {
            sequence.name: dataset_signature(sequence) for sequence in sequences
        },
        "timeout_seconds": args.timeout,
        "slam_type": "stereo",
        "settings": settings,
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain"),
            "diff_sha256": (
                hashlib.sha256(patch.encode()).hexdigest() if patch else None
            ),
        },
        "artifacts": {
            key: {"path": str(path), "sha256": file_hash(path)}
            for key, path in (
                ("app", args.app),
                ("vocab", args.vocab),
                ("settings", args.settings),
            )
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "yaml": yaml.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_PROC_BIND",
                )
            },
        },
    }
    (output / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def finish_run(output, manifest, status):
    if manifest is not None:
        manifest.update(
            status=status, finished_at=datetime.now(timezone.utc).isoformat()
        )
        (output / "run.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )


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
    if not args.skip_run:
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
        "expected_frames": expected,
        "wall_seconds": wall_seconds,
        **summarize(samples, counters),
    }
    try:
        trajectory = regression.load_estimate(output / "f_trajectory.txt", np.eye(4))
        result.update(
            tracked_frames=trajectory.num_poses,
            tracked_ratio=float(trajectory.num_poses / expected),
        )
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


def write_reports(output, results, pooled, totals, persist=True):
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
    tracked = [row for row in results if "tracked_ratio" in row]
    if tracked:
        aggregate["tracked_ratio"] = sum(
            row["tracked_ratio"] * row["frames"] for row in tracked
        ) / sum(row["frames"] for row in tracked)
    aggregate["statistics_sequences"] = sum("frames" in row for row in results)
    aggregate["accuracy_sequences"] = len(successful)
    aggregate["failed_sequences"] = sum(row["status"] != "ok" for row in results)
    if persist:
        (output / "summary.json").write_text(
            json.dumps(
                {"sequences": results, "aggregate": aggregate},
                indent=2,
                allow_nan=False,
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
    if persist:
        (output / "summary.md").write_text(report)
    return report


def flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def compare_runs(baseline, candidate):
    reports = []
    manifests = []
    warnings = []
    for path in (baseline, candidate):
        reports.append(json.loads((path / "summary.json").read_text()))
        manifest_path = path / "run.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        if not manifest:
            warnings.append(f"{path}: legacy run has no provenance manifest")
        elif manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported run schema in {manifest_path}")
        elif manifest.get("status") != "completed":
            warnings.append(
                f"{path}: run is {manifest.get('status')}; results may be partial"
            )
        manifests.append(manifest)
    metadata = [
        flatten(
            {
                key: manifest.get(key)
                for key in (
                    "git",
                    "settings",
                    "artifacts",
                    "host",
                    "datasets",
                    "datasets_root",
                    "timeout_seconds",
                    "notes",
                    "slam_type",
                )
            }
        )
        for manifest in manifests
    ]
    changes = [
        {
            "field": key,
            "baseline": metadata[0].get(key),
            "candidate": metadata[1].get(key),
        }
        for key in sorted(metadata[0].keys() | metadata[1].keys())
        if metadata[0].get(key) != metadata[1].get(key)
    ]
    if any(
        change["field"].startswith(
            ("datasets.", "host.", "artifacts.vocab.", "timeout_seconds", "slam_type")
        )
        for change in changes
    ):
        warnings.append(
            "Dataset, host, vocabulary or execution conditions differ; metric deltas may not reflect algorithm changes alone"
        )
    indexed = [
        {row["sequence"]: row for row in report["sequences"]} for report in reports
    ]
    statuses = []
    deltas = []

    def add_metrics(name, before, after):
        for metric in sorted(before.keys() & after.keys()):
            old, new = before[metric], after[metric]
            if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
                continue
            if not np.isfinite(old) or not np.isfinite(new):
                raise ValueError(f"Non-finite {metric} in {name}")
            delta = new - old
            direction = 0
            if (
                metric
                in ("ate_rmse_m", "failed_sequences", "wall_seconds", "elapsed_seconds")
                or "_ms" in metric
            ):
                direction = -1
            elif metric in ("tracked_ratio", "throughput_fps", "accuracy_sequences"):
                direction = 1
            deltas.append(
                {
                    "sequence": name,
                    "metric": metric,
                    "baseline": old,
                    "candidate": new,
                    "delta": delta,
                    "percent": 100 * delta / abs(old) if old else None,
                    "assessment": (
                        "unchanged"
                        if delta == 0
                        else (
                            "context"
                            if not direction
                            else "improved" if delta * direction > 0 else "regressed"
                        )
                    ),
                }
            )

    for name in sorted(indexed[0].keys() | indexed[1].keys()):
        before, after = (rows.get(name, {}) for rows in indexed)
        statuses.append(
            {
                "sequence": name,
                "baseline": before.get("status", "missing"),
                "candidate": after.get("status", "missing"),
                "baseline_error": before.get("error"),
                "candidate_error": after.get("error"),
            }
        )
        add_metrics(name, before, after)
    same_workload = indexed[0].keys() == indexed[1].keys() and all(
        indexed[0][name].get("frames") == indexed[1][name].get("frames")
        and indexed[0][name]["status"] == indexed[1][name]["status"]
        for name in indexed[0]
    )
    if same_workload:
        add_metrics("ALL (pooled)", reports[0]["aggregate"], reports[1]["aggregate"])
    else:
        warnings.append(
            "Pooled deltas omitted: sequence sets, processed frame counts or outcomes differ; compare individual sequences"
        )
    return {
        "schema_version": 1,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "warnings": warnings,
        "changes": changes,
        "sequences": statuses,
        "metrics": deltas,
    }


def comparison_report(comparison):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        f"# Benchmark Comparison",
        "",
        f"Baseline: {comparison['baseline']}",
        f"Candidate: {comparison['candidate']}",
        "",
    ]
    lines.extend(f"Warning: {warning}" for warning in comparison["warnings"])
    lines.extend(
        [
            "",
            "## Changes",
            "",
            "| Field | Baseline | Candidate |",
            "| --- | --- | --- |",
        ]
    )
    for change in comparison["changes"]:
        lines.append(
            "| "
            + " | ".join(
                cell(json.dumps(change[key], sort_keys=True))
                for key in ("field", "baseline", "candidate")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Sequence Outcomes",
            "",
            "| Sequence | Baseline | Candidate | Baseline error | Candidate error |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in comparison["sequences"]:
        lines.append(
            "| "
            + " | ".join(
                cell(row[key] or "-")
                for key in (
                    "sequence",
                    "baseline",
                    "candidate",
                    "baseline_error",
                    "candidate_error",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Metric Deltas",
            "",
            "Candidate minus baseline; percentages are unavailable for a zero baseline. Assessments show direction, not statistical significance.",
            "",
            "| Sequence | Metric | Baseline | Candidate | Delta | Change (%) | Assessment |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison["metrics"]:
        values = (
            [cell(row["sequence"]), row["metric"]]
            + [
                f"{row[key]:.6g}" if row[key] is not None else "-"
                for key in ("baseline", "candidate", "delta", "percent")
            ]
            + [row["assessment"]]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    if args.compare:
        comparison = compare_runs(*(path.resolve() for path in args.compare))
        report = comparison_report(comparison)
        if args.comparison_output:
            output = args.comparison_output.resolve()
            if output in [path.resolve() for path in args.compare]:
                raise ValueError("Comparison output must not be a source run directory")
            output.mkdir(parents=True, exist_ok=True)
            (output / "comparison.json").write_text(
                json.dumps(comparison, indent=2, allow_nan=False) + "\n"
            )
            (output / "comparison.md").write_text(report)
            with (output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "sequence",
                        "metric",
                        "baseline",
                        "candidate",
                        "delta",
                        "percent",
                        "assessment",
                    ),
                )
                writer.writeheader()
                writer.writerows(comparison["metrics"])
        print(report)
        return 0
    for field in ("datasets_root", "app", "vocab", "settings", "output_folder"):
        setattr(args, field, getattr(args, field).resolve())
    sequences = discover_sequences(args.datasets_root, args.sequence)
    manifest = None
    if args.skip_run:
        manifest_path = args.output_folder / "run.json"
        if manifest_path.exists():
            recorded = json.loads(manifest_path.read_text())
            if recorded["sequences"] != [sequence.name for sequence in sequences]:
                raise ValueError(
                    "Re-evaluation must use the recorded sequence selection"
                )
    else:
        manifest = create_run(args, sequences)
    print(f"Run directory: {args.output_folder}", flush=True)
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
                    "failure_kind": (
                        "timeout"
                        if isinstance(error, subprocess.TimeoutExpired)
                        else (
                            "crash"
                            if isinstance(error, subprocess.CalledProcessError)
                            else "input_or_output"
                        )
                    ),
                    "error": str(error),
                }
                print(f"{sequence.name}: {error}", file=sys.stderr)
            results.append(result)
            report = write_reports(
                args.output_folder, results, pooled, totals, persist=not args.skip_run
            )
    except KeyboardInterrupt:
        finish_run(args.output_folder, manifest, "interrupted")
        print("Interrupted; completed sequence reports are preserved.", file=sys.stderr)
        return 130
    finish_run(args.output_folder, manifest, "completed")
    print(report)
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    sys.exit(main())
