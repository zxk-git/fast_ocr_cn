import os
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_pytorch_env.sh"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_onnx_torch.sh"


class PytorchShellScriptsTest(unittest.TestCase):
    def test_setup_keeps_pytorch_and_installs_onnx_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            capture_path = root / "python-calls.txt"
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${CAPTURE_PATH}\"\n"
                "if [[ \"$*\" == *'torch.__version__'* ]]; then echo '2.4.0|0.19.0'; fi\n"
                "if [[ \"$*\" == *'find_spec(\"tensorflow\")'* ]]; then exit 1; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update({"CAPTURE_PATH": str(capture_path), "PYTHON_BIN": str(fake_python)})

            subprocess.run(
                [str(SETUP_SCRIPT), "--skip-verify"],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )

            calls = capture_path.read_text(encoding="utf-8")
            self.assertIn("requirements/train-torch.txt", calls)
            self.assertIn("onnx==1.17.0", calls)
            self.assertIn("onnxscript==0.1.0", calls)
            self.assertNotIn("pip install torch", calls)
            self.assertNotIn("pip install tensorflow", calls)

    def test_export_uses_latest_model_and_torch_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            train_dir = root / "trained"
            old_run = train_dir / "old"
            new_run = train_dir / "new"
            old_run.mkdir(parents=True)
            new_run.mkdir(parents=True)
            old_model = old_run / "best.keras"
            new_model = new_run / "best.keras"
            old_model.touch()
            new_model.touch()
            os.utime(old_model, (1, 1))
            os.utime(new_model, (2, 2))
            (new_run / "plate_config.yaml").write_text("max_plate_slots: 8\n")

            capture_path = root / "export-call.txt"
            fake_cli = root / "fast-plate-ocr"
            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"${KERAS_BACKEND}\" \"$@\" > \"${CAPTURE_PATH}\"\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            fake_python = root / "python"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "CAPTURE_PATH": str(capture_path),
                    "FAST_PLATE_OCR_BIN": str(fake_cli),
                    "FAST_PLATE_OCR_TRAIN_OUTPUT_DIR": str(train_dir),
                    "PYTHON_BIN": str(fake_python),
                }
            )

            subprocess.run(
                [str(EXPORT_SCRIPT), "--no-simplify"],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )

            captured = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(captured[0], "torch")
            self.assertEqual(captured[1], "export")
            self.assertIn(str(new_model), captured)
            self.assertIn(str(new_run / "plate_config.yaml"), captured)
            self.assertIn("onnx", captured)
            self.assertEqual(captured[-1], "--no-simplify")


if __name__ == "__main__":
    unittest.main()
