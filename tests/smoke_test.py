"""End-to-end smoke test on a tiny synthetic dataset.

Verifies that preprocessing, training, evaluation and image/video prediction
all run. It does NOT measure real detection accuracy.

    python tests/smoke_test.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import evaluate  # noqa: E402
import predict  # noqa: E402
import preprocess  # noqa: E402
import train  # noqa: E402


def synthetic(fake: bool, rng, size=96):
    """'Real' = smooth gradient blob; 'fake' = same + periodic high-frequency grid."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = np.stack([np.sin(xx * rng.uniform(1, 4)), np.cos(yy * rng.uniform(1, 4)),
                     xx * yy], -1)
    img = (base - base.min()) / (np.ptp(base) + 1e-6) * 200 + rng.uniform(0, 40)
    if fake:
        img += 25 * ((np.indices((size, size)).sum(0) % 4) < 2)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="deepfake_smoke_"))
    rng = np.random.default_rng(0)
    try:
        raw = tmp / "raw"
        for label, fake in (("real", False), ("fake", True)):
            (raw / label).mkdir(parents=True)
            for i in range(60):
                cv2.imwrite(str(raw / label / f"{i}.png"), synthetic(fake, rng))
            writer = cv2.VideoWriter(str(raw / label / "clip.avi"),
                                     cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
            for _ in range(30):
                writer.write(synthetic(fake, rng))
            writer.release()

        faces = tmp / "faces"
        preprocess.main(["--input", str(raw), "--output", str(faces), "--no-detect",
                         "--size", "64", "--frames-per-video", "5"])

        model_path = tmp / "model.keras"
        reports = tmp / "reports"
        train.main(["--data-dir", str(faces), "--arch", "cnn", "--img-size", "64",
                    "--epochs", "4", "--batch-size", "16", "--model-path", str(model_path),
                    "--report-dir", str(reports)])
        metrics = evaluate.main(["--data-dir", str(faces), "--model-path", str(model_path),
                                 "--report-dir", str(reports)])

        img = tmp / "probe.png"
        cv2.imwrite(str(img), synthetic(True, rng, 200))
        results = predict.main([str(img), str(raw / "fake" / "clip.avi"),
                                "--model-path", str(model_path), "--frames", "6"])

        expected = ["training_curves.png", "confusion_matrix.png", "roc_curve.png", "metrics.json"]
        missing = [f for f in expected if not (reports / f).exists()]
        assert not missing, f"missing reports: {missing}"
        assert len(results) == 2 and all(r["label"] in ("Original", "Deepfake") for r in results)
        print(f"\nSMOKE TEST PASSED (synthetic test accuracy {metrics['accuracy']:.2f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
