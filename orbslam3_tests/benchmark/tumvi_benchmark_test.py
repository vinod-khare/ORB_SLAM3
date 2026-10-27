"""Synthetic integration checks; no SLAM process or real dataset required."""

import csv
import json
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
            ]
            with patch.object(benchmark.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(benchmark.main(arguments), 1)
                self.assertEqual(run.call_count, 2)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["aggregate"]["frames"], 8)
            self.assertEqual(report["aggregate"]["accuracy_sequences"], 2)
            self.assertEqual(report["aggregate"]["failed_sequences"], 1)
            self.assertAlmostEqual(report["aggregate"]["ate_rmse_m"], 0)
            self.assertEqual(report["aggregate"]["frontend_ms_mean"], 2.5)
            self.assertAlmostEqual(report["aggregate"]["throughput_fps"], 10)
            self.assertEqual(report["aggregate"]["local_mapping_jobs"], 8)
            with patch.object(benchmark.subprocess, "run") as run:
                self.assertEqual(
                    benchmark.main(
                        arguments + ["--skip-run", "--sequence", "dataset-room1"]
                    ),
                    0,
                )
                run.assert_not_called()
                self.assertEqual(
                    benchmark.main(arguments + ["--sequence", "dataset-room1"]), 1
                )
                run.assert_not_called()
            (output / "dataset-room1/statistics.csv").write_text("frames\n1\n")
            self.assertEqual(
                benchmark.main(
                    arguments + ["--skip-run", "--sequence", "dataset-room1"]
                ),
                1,
            )

    def test_missing_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                benchmark.discover_sequences(Path(directory), ["missing"])


if __name__ == "__main__":
    unittest.main()
