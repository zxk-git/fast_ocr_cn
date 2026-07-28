import os
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
LAUNCHER = PROJECT_ROOT / "scripts" / "train_cblprd_torch.sh"


class TorchLauncherTest(unittest.TestCase):
    def run_launcher(
        self, *args: str, visible_devices: str = "existing", gpu_env: str | None = None
    ) -> list[str]:
        self.assertTrue(LAUNCHER.is_file(), f"launcher missing: {LAUNCHER}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            dataset_root = root / "dataset"
            for split in ("train", "val"):
                split_dir = dataset_root / split
                split_dir.mkdir(parents=True)
                (split_dir / "annotations.csv").write_text(
                    "image_path,plate_text\nimages/sample.jpg,京A12345\n",
                    encoding="utf-8",
                )

            capture_path = root / "capture.txt"
            fake_cli = root / "fake-fast-plate-ocr"
            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"${KERAS_BACKEND}\" \"${NO_ALBUMENTATIONS_UPDATE}\" "
                "\"${CUDA_VISIBLE_DEVICES:-}\" \"$@\" "
                '> "${CAPTURE_PATH}"\n',
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "CAPTURE_PATH": str(capture_path),
                    "CUDA_VISIBLE_DEVICES": visible_devices,
                    "FAST_PLATE_OCR_BIN": str(fake_cli),
                    "FAST_PLATE_OCR_DATASET_ROOT": str(dataset_root),
                }
            )
            if gpu_env is not None:
                env["FAST_PLATE_OCR_GPU"] = gpu_env
            else:
                env.pop("FAST_PLATE_OCR_GPU", None)
            subprocess.run(
                [str(LAUNCHER), *args],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )
            return capture_path.read_text(encoding="utf-8").splitlines()

    def test_launcher_sets_torch_backend_and_forwards_overrides(self) -> None:
        captured = self.run_launcher("--batch-size", "8")

        self.assertEqual(captured[0], "torch")
        self.assertEqual(captured[1], "1")
        self.assertEqual(captured[2], "existing")
        self.assertEqual(captured[3], "train")
        self.assertIn(str(PROJECT_ROOT / "models" / "cct_s_v2.yaml"), captured)
        self.assertTrue(any(path.endswith("/train/annotations.csv") for path in captured))
        self.assertTrue(any(path.endswith("/val/annotations.csv") for path in captured))
        self.assertIn("--epochs", captured)
        self.assertIn("200", captured)
        self.assertEqual(captured[-2:], ["--batch-size", "8"])

    def test_quick_mode_runs_once_with_three_epochs(self) -> None:
        captured = self.run_launcher("--quick")

        self.assertEqual(captured.count("train"), 1)
        self.assertNotIn("--quick", captured)
        epochs_index = captured.index("--epochs")
        self.assertEqual(captured[epochs_index + 1], "3")
        output_index = captured.index("--output-dir")
        self.assertEqual(
            captured[output_index + 1], str(PROJECT_ROOT / "trained_models" / "quick_test")
        )

    def test_gpu_option_sets_cuda_visible_devices_without_forwarding_it(self) -> None:
        captured = self.run_launcher("--gpu", "2")

        self.assertEqual(captured[2], "2")
        self.assertEqual(captured[3], "train")
        self.assertNotIn("--gpu", captured)

    def test_gpu_environment_variable_selects_gpu(self) -> None:
        captured = self.run_launcher(gpu_env="3")

        self.assertEqual(captured[2], "3")

    def test_rejects_invalid_gpu_id(self) -> None:
        completed = subprocess.run(
            [str(LAUNCHER), "--gpu", "invalid"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Invalid GPU ID", completed.stderr)


if __name__ == "__main__":
    unittest.main()
