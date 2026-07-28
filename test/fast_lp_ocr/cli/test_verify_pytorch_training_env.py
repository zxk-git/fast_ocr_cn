import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from scripts.verify_pytorch_training_env import validate_annotations


VERIFIER = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "verify_pytorch_training_env.py"


class TrainingEnvironmentVerificationTest(unittest.TestCase):
    def test_requires_explicit_torch_backend(self) -> None:
        env = os.environ.copy()
        env.pop("KERAS_BACKEND", None)
        completed = subprocess.run(
            [sys.executable, str(VERIFIER)],
            cwd=VERIFIER.parents[1],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Set KERAS_BACKEND=torch", completed.stderr)

    def test_accepts_valid_annotations_and_referenced_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            image_dir = split_dir / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"test")
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/sample.jpg,京A12345\n",
                encoding="utf-8",
            )

            self.assertEqual(validate_annotations(csv_path, set("京A12345_"), 8), 1)

    def test_rejects_out_of_alphabet_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            image_dir = split_dir / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"test")
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/sample.jpg,京B12345\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside the configured alphabet"):
                validate_annotations(csv_path, set("京A12345_"), 8)

    def test_rejects_plate_text_longer_than_max_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            image_dir = split_dir / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"test")
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/sample.jpg,京A1234567\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exceeds 8 slots"):
                validate_annotations(csv_path, set("京A1234567_"), 8)

    def test_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            (split_dir / "images").mkdir()
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text("image_path\nimages/sample.jpg\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing columns"):
                validate_annotations(csv_path, set("京A12345_"), 8)

    def test_rejects_missing_referenced_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            (split_dir / "images").mkdir()
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/missing.jpg,京A12345\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing referenced image"):
                validate_annotations(csv_path, set("京A12345_"), 8)

    def test_rejects_image_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            image_dir = split_dir / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"test")
            (image_dir / "extra.jpg").write_bytes(b"test")
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/sample.jpg,京A12345\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "1 rows but 2 images"):
                validate_annotations(csv_path, set("京A12345_"), 8)


if __name__ == "__main__":
    unittest.main()
