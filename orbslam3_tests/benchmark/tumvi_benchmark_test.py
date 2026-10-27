"""Synthetic integration checks; no SLAM process or real dataset required."""

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

import tumvi_benchmark as benchmark


class TumviBenchmarkTest(unittest.TestCase):
    def test_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            output = root / "results"
            template = root / "template.yaml"
            template.write_text("%YAML:1.0\nkeypoint_detector: {type: FAST}\n")
            for name in ("dataset-room1", "dataset-room2", "dataset-broken"):
                sequence = datasets / name
                sequence.mkdir(parents=True)
                if name == "dataset-broken":
                    continue
                for camera in ("cam0", "cam1"):
                    images = sequence / "mav0" / camera / "data"
                    images.mkdir(parents=True)
                    for index in range(4):
                        (images / f"{index}.png").touch()
                (sequence / "dso").mkdir()
                transform = np.eye(4)
                transform[0, 3] = 0.1
                (sequence / "dso/camchain.yaml").write_text(
                    yaml.safe_dump({"cam0": {"T_cam_imu": transform.tolist()}})
                )
                mocap = sequence / "mav0/mocap0"
                mocap.mkdir()
                with (mocap / "data.csv").open("w") as stream:
                    stream.write("#timestamp,x,y,z,qw,qx,qy,qz\n")
                    for index, position in enumerate(
                        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
                    ):
                        stream.write(
                            f"{1_000_000_000 + index * 50_000_000},{position[0] + 0.1},{position[1]},{position[2]},1,0,0,0\n"
                        )

            def fake_run(command, **kwargs):
                self.assertEqual(command[command.index("--slam-type") + 1], "stereo")
                self.assertIn("--no-viewer", command)
                destination = Path(command[command.index("--output-folder") + 1])
                settings_path = Path(command[command.index("--settings") + 1])
                self.assertTrue(settings_path.read_text().startswith("%YAML:1.0\n"))
                settings = benchmark.regression.load_settings(settings_path)
                self.assertEqual(
                    Path(settings["data"]["root"]).parent.name, destination.name
                )
                self.assertTrue(Path(settings["calibration"]["file"]).is_file())
                rows = []
                for index in range(4):
                    row = {field: index + 1 for field in benchmark.FRAME_FIELDS}
                    row.update(
                        frames=index + 1,
                        elapsed_seconds=(index + 1) / 10,
                        preprocessing_total_ms=10,
                        frontend_total_ms=10,
                    )
                    for worker in benchmark.WORKERS:
                        row.update(
                            {f"{worker}_jobs": index + 1, f"{worker}_total_ms": 20}
                        )
                    rows.append(row)
                with (destination / "statistics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with (destination / "f_trajectory.txt").open("w") as stream:
                    for index, position in enumerate(
                        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
                    ):
                        stream.write(
                            f"{1_000_000_000 + index * 50_000_000} {position[0]} {position[1]} {position[2]} 0 0 0 1\n"
                        )

            arguments = [
                "--datasets-root",
                str(datasets),
                "--output-folder",
                str(output),
                "--settings",
                str(template),
                "--run-name",
                "baseline",
            ]
            with patch.object(
                benchmark.platform, "platform", return_value="test-host"
            ), patch.object(
                benchmark.subprocess, "check_output", return_value=b"test"
            ), patch.object(
                benchmark.subprocess, "run", side_effect=fake_run
            ) as run:
                self.assertEqual(benchmark.main(arguments), 1)
                self.assertEqual(run.call_count, 2)
            output = output / "baseline"
            manifest = json.loads((output / "run.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["git"]["commit"], "test")
            self.assertTrue((output / "template.yaml").is_file())
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["aggregate"]["frames"], 8)
            self.assertEqual(report["aggregate"]["accuracy_sequences"], 2)
            self.assertEqual(report["aggregate"]["failed_sequences"], 1)
            self.assertAlmostEqual(report["aggregate"]["ate_rmse_m"], 0)
            self.assertEqual(report["aggregate"]["frontend_ms_mean"], 2.5)
            self.assertAlmostEqual(report["aggregate"]["throughput_fps"], 10)
            self.assertEqual(report["aggregate"]["local_mapping_jobs"], 8)
            with patch.object(
                benchmark.platform, "platform", return_value="test-host"
            ), patch.object(
                benchmark.subprocess, "check_output", return_value=b"test"
            ), patch.object(
                benchmark.subprocess, "run", side_effect=fake_run
            ) as run:
                self.assertEqual(
                    benchmark.main(
                        arguments + ["--run-name", "flow", "--compare-optical-flow"]
                    ),
                    1,
                )
                self.assertEqual(run.call_count, 4)
            paired = output.parent / "flow"
            variants = []
            for enabled, name in (
                (0, "without_optical_flow"),
                (1, "with_optical_flow"),
            ):
                variant = json.loads((paired / name / "run.json").read_text())
                self.assertEqual(variant["sequences"], manifest["sequences"])
                self.assertEqual(
                    variant["settings"]["tracking"].pop("use_optical_flow"), enabled
                )
                variants.append(variant["settings"])
                for sequence in ("dataset-room1", "dataset-room2"):
                    settings = benchmark.regression.load_settings(
                        paired / name / sequence / "settings.yaml"
                    )
                    self.assertEqual(settings["tracking"]["use_optical_flow"], enabled)
            self.assertEqual(variants[0], variants[1])
            comparison = json.loads((paired / "comparison.json").read_text())
            self.assertEqual(len(comparison["sequences"]), 3)
            self.assertTrue((paired / "comparison.csv").is_file())
            self.assertTrue((paired / "comparison.md").is_file())
            self.assertEqual(
                json.loads((paired / "run.json").read_text())["status"], "completed"
            )
            saved_summary = (output / "summary.json").read_bytes()
            saved_manifest = (output / "run.json").read_bytes()
            with patch.object(
                benchmark.platform, "platform", return_value="test-host"
            ), patch.object(
                benchmark.subprocess, "check_output", return_value=b"test"
            ), patch.object(
                benchmark.subprocess, "run", side_effect=fake_run
            ):
                self.assertEqual(
                    benchmark.main(arguments + ["--run-name", "candidate"]), 1
                )
            self.assertEqual((output / "summary.json").read_bytes(), saved_summary)
            self.assertTrue((output.parent / "candidate/run.json").is_file())
            with self.assertRaises(FileExistsError):
                benchmark.main(arguments)
            arguments = [
                "--datasets-root",
                str(datasets),
                "--output-folder",
                str(output),
            ]
            with patch.object(benchmark.subprocess, "run") as run:
                self.assertEqual(benchmark.main(arguments + ["--skip-run"]), 1)
                run.assert_not_called()
                self.assertEqual((output / "summary.json").read_bytes(), saved_summary)
                self.assertEqual((output / "run.json").read_bytes(), saved_manifest)
                with self.assertRaises(ValueError):
                    benchmark.main(
                        arguments + ["--skip-run", "--sequence", "dataset-room1"]
                    )
                run.assert_not_called()
            (output / "dataset-room1/statistics.csv").write_text("frames\n1\n")
            self.assertEqual(
                benchmark.main(arguments + ["--skip-run"]),
                1,
            )
            with patch.object(
                benchmark.regression,
                "evaluate",
                side_effect=ValueError("no mocap matches"),
            ):
                args = benchmark.parse_args(arguments + ["--skip-run"])
                result, _, _ = benchmark.run_sequence(args, datasets / "dataset-room2")
                self.assertEqual(result["status"], "accuracy_failed")
                self.assertEqual(result["tracked_frames"], 4)
                self.assertEqual(result["tracked_ratio"], 1)

    def test_interruption_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = root / "datasets"
            (datasets / "dataset-room").mkdir(parents=True)
            template = root / "settings.yaml"
            template.write_text("keypoint_detector: {type: FAST}\n")
            arguments = [
                "--datasets-root",
                str(datasets),
                "--output-folder",
                str(root / "runs"),
                "--settings",
                str(template),
            ]
            for name, error, expected in (
                ("interrupted", KeyboardInterrupt(), 130),
                ("timeout", subprocess.TimeoutExpired("app", 1), 1),
                ("crash", subprocess.CalledProcessError(1, "app"), 1),
            ):
                with patch.object(
                    benchmark.platform, "platform", return_value="test-host"
                ), patch.object(
                    benchmark.subprocess, "check_output", return_value=b"test"
                ), patch.object(
                    benchmark, "run_sequence", side_effect=error
                ):
                    self.assertEqual(
                        benchmark.main(arguments + ["--run-name", name]), expected
                    )
                manifest = json.loads((root / "runs" / name / "run.json").read_text())
                self.assertEqual(
                    manifest["status"],
                    "interrupted" if expected == 130 else "completed",
                )
                self.assertIsNotNone(manifest["finished_at"])
                if expected != 130:
                    report = json.loads(
                        (root / "runs" / name / "summary.json").read_text()
                    )
                    self.assertEqual(report["sequences"][0]["failure_kind"], name)
            with patch.object(
                benchmark.platform, "platform", return_value="test-host"
            ), patch.object(
                benchmark.subprocess, "check_output", return_value=b"test"
            ), patch.object(
                benchmark, "run_sequence", side_effect=KeyboardInterrupt()
            ) as run:
                self.assertEqual(
                    benchmark.main(
                        arguments + ["--run-name", "flow", "--compare-optical-flow"]
                    ),
                    130,
                )
                self.assertEqual(run.call_count, 1)
            paired = root / "runs/flow"
            self.assertEqual(
                json.loads((paired / "run.json").read_text())["status"], "interrupted"
            )
            self.assertFalse((paired / "with_optical_flow").exists())

    def test_optical_flow_options(self):
        for option in ("--skip-run", "--overwrite", "--compare"):
            arguments = ["--compare-optical-flow", option]
            if option == "--compare":
                arguments.extend(["baseline", "candidate"])
            with self.subTest(option=option), self.assertRaises(SystemExit):
                benchmark.parse_args(arguments)

    def test_missing_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                benchmark.discover_sequences(Path(directory), ["missing"])

    def test_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "baseline", root / "candidate"]
            for index, path in enumerate(paths):
                path.mkdir()
                (path / "run.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "completed",
                            "git": {"commit": str(index)},
                            "settings": {"detector": {"type": str(index)}},
                        }
                    )
                )
                (path / "summary.json").write_text(
                    json.dumps(
                        {
                            "sequences": [
                                {
                                    "sequence": "room",
                                    "status": "ok",
                                    "frames": 10,
                                    "ate_rmse_m": 0.1 + index * 0.1,
                                    "tracked_ratio": 0.8 + index * 0.1,
                                    "throughput_fps": 10 + index * 5,
                                    "frontend_ms_mean": 0 + index,
                                    "wall_seconds": None,
                                }
                            ],
                            "aggregate": {"ate_rmse_m": 0.1 + index * 0.1},
                        }
                    )
                )
            comparison = benchmark.compare_runs(*paths)
            metrics = {
                row["metric"]: row
                for row in comparison["metrics"]
                if row["sequence"] == "room"
            }
            self.assertEqual(metrics["ate_rmse_m"]["assessment"], "regressed")
            self.assertAlmostEqual(metrics["ate_rmse_m"]["percent"], 100)
            self.assertEqual(metrics["tracked_ratio"]["assessment"], "improved")
            self.assertEqual(metrics["throughput_fps"]["delta"], 5)
            self.assertIsNone(metrics["frontend_ms_mean"]["percent"])
            self.assertTrue(
                any(
                    row["field"] == "settings.detector.type"
                    for row in comparison["changes"]
                )
            )
            self.assertTrue(
                any(row["sequence"] == "ALL (pooled)" for row in comparison["metrics"])
            )
            with patch.object(benchmark.subprocess, "run") as run:
                self.assertEqual(
                    benchmark.main(
                        [
                            "--compare",
                            *map(str, paths),
                            "--comparison-output",
                            str(root / "diff"),
                        ]
                    ),
                    0,
                )
                run.assert_not_called()
            self.assertTrue((root / "diff/comparison.csv").is_file())
            changed = json.loads((paths[1] / "summary.json").read_text())
            changed["sequences"][0].update(status="accuracy_failed", error="no matches")
            changed["sequences"].append(
                {"sequence": "new", "status": "failed", "error": "timeout"}
            )
            (paths[1] / "summary.json").write_text(json.dumps(changed))
            comparison = benchmark.compare_runs(*paths)
            self.assertTrue(comparison["warnings"])
            self.assertFalse(
                any(row["sequence"] == "ALL (pooled)" for row in comparison["metrics"])
            )
            self.assertEqual(comparison["sequences"][0]["baseline"], "missing")
            self.assertIn("no matches", benchmark.comparison_report(comparison))


if __name__ == "__main__":
    unittest.main()
