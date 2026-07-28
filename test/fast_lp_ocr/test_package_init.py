import subprocess
import sys


def test_training_modules_import_without_onnxruntime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['onnxruntime'] = None; "
                "from fast_plate_ocr.train.model.config import PlateConfig; "
                "print(PlateConfig.__name__)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PlateConfig"
