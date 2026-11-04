"""Build a face-crop dataset from raw images and videos.

Input layouts supported (folder names are case-insensitive):

  1) Unsplit:   <input>/real/...   <input>/fake/...
     Sources are split into train/val/test *per source file*, so frames of
     the same video never end up in both training and test data.
  2) Pre-split: <input>/{train,val,test}/{real,fake}/...  (split is kept)

Each class folder may contain images, videos, or both (sub-folders allowed).

Example:
    python preprocess.py --input data/raw --output data/faces --frames-per-video 15
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

import config
from src.data import CLASS_ALIASES, SPLIT_ALIASES, find_dir
from src.face_detector import FaceDetector
from src.predictor import load_rgb
from src.video import sample_frames


def collect_sources(class_dir: Path) -> list[Path]:
    exts = config.IMAGE_EXTS | config.VIDEO_EXTS
    return sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in exts)


def save_jpg(rgb, path: Path, size: int):
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(str(path))


def process_source(src: Path, out_dir: Path, detector: FaceDetector | None,
                   frames_per_video: int, size: int) -> int:
    """Extract the main face from an image, or from sampled frames of a video."""
    if src.suffix.lower() in config.VIDEO_EXTS:
        frames = [(f"_f{idx:05d}", rgb) for idx, _, rgb in sample_frames(src, frames_per_video)]
    else:
        frames = [("", load_rgb(src))]

    saved = 0
    stem = f"{src.parent.name}_{src.stem}"
    for suffix, rgb in frames:
        if detector is None:
            crop = rgb
        else:
            faces = detector.extract_faces(rgb, max_faces=1)
            if not faces:
                continue
            crop = faces[0]["crop"]
        save_jpg(crop, out_dir / f"{stem}{suffix}.jpg", size)
        saved += 1
    return saved


def split_sources(sources: list[Path], val: float, test: float, seed: int):
    rng = random.Random(seed)
    sources = sources[:]
    rng.shuffle(sources)
    n_test, n_val = int(len(sources) * test), int(len(sources) * val)
    return {"test": sources[:n_test], "val": sources[n_test:n_test + n_val],
            "train": sources[n_test + n_val:]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=config.RAW_DIR)
    ap.add_argument("--output", type=Path, default=config.FACES_DIR)
    ap.add_argument("--frames-per-video", type=int, default=15)
    ap.add_argument("--size", type=int, default=config.IMG_SIZE, help="saved crop size (px)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--no-detect", action="store_true",
                    help="skip face detection (inputs are already face crops)")
    ap.add_argument("--detector", default="auto", choices=["auto", *FaceDetector.BACKENDS])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    detector = None if args.no_detect else FaceDetector(args.detector)
    if detector:
        print(f"Face detector: {detector.backend}")

    # plan: list of (split, label, source)
    plan = []
    presplit = find_dir(args.input, SPLIT_ALIASES["train"]) is not None
    for label, aliases in CLASS_ALIASES.items():
        if presplit:
            for split, s_aliases in SPLIT_ALIASES.items():
                split_dir = find_dir(args.input, s_aliases)
                class_dir = find_dir(split_dir, aliases) if split_dir else None
                if class_dir:
                    plan += [(split, label, s) for s in collect_sources(class_dir)]
        else:
            class_dir = find_dir(args.input, aliases)
            if class_dir is None:
                raise SystemExit(f"Missing '{config.CLASS_NAMES[label]}' folder in {args.input}")
            for split, items in split_sources(collect_sources(class_dir), args.val,
                                              args.test, args.seed).items():
                plan += [(split, label, s) for s in items]

    if not plan:
        raise SystemExit(f"No images or videos found in {args.input}")

    counts = {}
    for split, label, src in tqdm(plan, desc="Extracting faces"):
        out_dir = args.output / split / config.CLASS_NAMES[label]
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            n = process_source(src, out_dir, detector, args.frames_per_video, args.size)
        except Exception as exc:
            tqdm.write(f"Skipping {src}: {exc}")
            continue
        counts[(split, label)] = counts.get((split, label), 0) + n

    print("\nFace crops written to", args.output)
    for split in SPLIT_ALIASES:
        row = ", ".join(f"{config.CLASS_NAMES[l]}={counts.get((split, l), 0)}" for l in (0, 1))
        print(f"  {split:5s}: {row}")


if __name__ == "__main__":
    main()
