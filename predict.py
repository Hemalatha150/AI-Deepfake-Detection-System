"""Command-line prediction for images and videos.

    python predict.py photo.jpg clip.mp4
    python predict.py clip.mp4 --frames 30 --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config

from src.predictor import DeepfakeDetector, is_video, load_rgb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--frames", type=int, default=config.FRAMES_PER_VIDEO)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    args = ap.parse_args(argv)

    detector = DeepfakeDetector(args.model_path)
    results = []
    for path in args.files:
        if is_video(path):
            r = detector.predict_video(path, args.frames, args.threshold)
            summary = {"file": str(path), "type": "video", "label": r["label"],
                       "confidence": round(r["confidence"], 4),
                       "fake_probability": round(r["fake_probability"], 4),
                       "frames_analyzed": r["frames_analyzed"],
                       "frames_with_faces": r["frames_with_faces"],
                       "fake_frame_ratio": round(r["fake_frame_ratio"], 4),
                       "temporal_jitter": round(r["temporal_jitter"], 4)}
        else:
            r = detector.predict_image(load_rgb(path), args.threshold)
            summary = {"file": str(path), "type": "image", "label": r["label"],
                       "confidence": round(r["confidence"], 4),
                       "fake_probability": round(r["fake_probability"], 4),
                       "faces": 0 if r["used_full_image"] else len(r["faces"])}
        results.append(summary)
        if not args.json:
            print(f"{path.name}: {summary['label']} "
                  f"(confidence {summary['confidence']:.1%}, P(fake)={summary['fake_probability']:.3f})")
    if args.json:
        print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
