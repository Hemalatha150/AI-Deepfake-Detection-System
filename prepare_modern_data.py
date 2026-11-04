"""Add modern AI-generated data (diffusion models, face swaps) to the datasets.

Sources (downloaded into data/downloads/ first, see README):
  * DeepFakeFace (OpenRL/DeepFakeFace) - real IMDB-WIKI faces plus fakes made with
    Stable Diffusion 1.5 (text2img), SD inpainting and InsightFace face swapping.
  * Tiny-GenImage (TheKernel01/Tiny-GenImage) - real photos plus images from
    Midjourney, Stable Diffusion 1.4/1.5, ADM, GLIDE, Wukong, VQDM and BigGAN.

Outputs:
  data/faces/{split}/{real,fake}/dff_*/   face crops for the face model
  data/scenes/{split}/{real,fake}/...      whole images for the scene model

Every image is processed identically whatever its source, so that file format, size or
aspect ratio cannot become a shortcut for telling real from fake (AI images are mostly
square PNGs, real photos mostly rectangular JPEGs):
  * PNG / lossless sources get one realistic JPEG pass (quality 75-95) first;
  * images are centre-cropped to a square and saved as 256x256 JPEG (quality 95).

    python prepare_modern_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

import config

DOWNLOADS = config.DATA_DIR / "downloads"
SIZE = 256
DFF_FAKE = ["text2img", "inpainting", "insight"]


def split_of(key: str) -> str:
    """Deterministic 80/10/10 split. DeepFakeFace fakes share relative paths with the
    real image they were derived from, so one key keeps a whole family together."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else "val" if h < 20 else "train"


_rng = random.Random(0)


def to_rgb(data: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img = img.convert("RGB")
    except Exception:
        return None
    if fmt != "JPEG":   # simulate the compression real photos have already been through
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=_rng.randint(75, 95))
        img = Image.open(buf).convert("RGB")
    return np.asarray(img)


def center_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return rgb[y:y + s, x:x + s]


def save(rgb: np.ndarray, path: Path, square: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if square:
        rgb = center_square(rgb)
    img = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(str(path))


# ------------------------------------------------------------- DeepFakeFace
def dff_members(zf: zipfile.ZipFile) -> dict[str, str]:
    """relative key (without the top folder) -> member name"""
    out = {}
    for name in zf.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            key = "/".join(name.split("/")[1:]) or name
            out[key] = name
    return out


def prepare_dff(n_real: int, n_fake_each: int, scene_share: float, seed: int, detector):
    zips = {k: zipfile.ZipFile(DOWNLOADS / "dff" / f"{k}.zip") for k in ["wiki", *DFF_FAKE]}
    members = {k: dff_members(z) for k, z in zips.items()}
    rng = random.Random(seed)
    plan = [("wiki", 0, n_real)] + [(k, 1, n_fake_each) for k in DFF_FAKE]
    counts = {}
    for subset, label, n in plan:
        keys = sorted(members[subset])
        rng.shuffle(keys)
        kept = 0
        for key in tqdm(keys, desc=f"DFF {subset}"):
            if kept >= n:
                break
            rgb = to_rgb(zips[subset].read(members[subset][key]))
            if rgb is None:
                continue
            split = split_of(key)
            stem = f"dff_{subset}_{hashlib.md5(key.encode()).hexdigest()[:12]}.jpg"
            cls = config.CLASS_NAMES[label]
            # Whole image -> scene dataset (a share of them).
            if rng.random() < scene_share:
                save(rgb, config.SCENES_DIR / split / cls / f"dff_{subset}" / stem)
            # Face crop (same detector + margin as at inference) -> face dataset.
            faces = detector.extract_faces(rgb, max_faces=1)
            if not faces:
                continue
            save(faces[0]["crop"], config.FACES_DIR / split / cls / f"dff_{subset}" / stem)
            kept += 1
            counts[(subset, split)] = counts.get((subset, split), 0) + 1
    print("DeepFakeFace face crops:", counts)


# ------------------------------------------------------------- GenImage
GEN_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]


def prepare_genimage():
    counts = {}
    for parquet in sorted((DOWNLOADS / "genimage").glob("*.parquet")):
        is_val_shard = parquet.name.startswith("val")
        pf = pq.ParquetFile(parquet)
        for batch in tqdm(pf.iter_batches(batch_size=256, columns=["image", "label", "generator"]),
                          desc=parquet.name, total=pf.metadata.num_row_groups):
            for img, label, gen in zip(batch.column("image").to_pylist(),
                                       batch.column("label").to_pylist(),
                                       batch.column("generator").to_pylist()):
                rgb = to_rgb(img["bytes"])
                if rgb is None:
                    continue
                key = f"{parquet.stem}_{counts.get(parquet.name, 0)}"
                counts[parquet.name] = counts.get(parquet.name, 0) + 1
                # Official validation shards become our test set; train shards are
                # split into train / val.
                split = "test" if is_val_shard else ("val" if split_of(key) != "train" else "train")
                gen_name = GEN_NAMES[gen] if isinstance(gen, int) and gen < len(GEN_NAMES) else str(gen)
                save(rgb, config.SCENES_DIR / split / config.CLASS_NAMES[label]
                     / f"genimage_{gen_name}" / f"{key}.jpg")
    print("GenImage images:", counts)


def add_existing_faces_to_scenes(n_each: int, seed: int):
    """FFHQ / StyleGAN / LFW images are whole pictures too - reuse some for the scene model."""
    rng = random.Random(seed)
    for split in ("train", "val", "test"):
        n = n_each if split == "train" else n_each // 8
        for cls in ("real", "fake"):
            src = config.FACES_DIR / split / cls
            files = sorted(p for p in src.glob("*.jpg"))           # 140k dataset (top level)
            lfw = sorted((src / "lfw").glob("*.jpg")) if cls == "real" else []
            rng.shuffle(files)
            rng.shuffle(lfw)
            picks = files[:n] + lfw[:n // 2]
            for p in picks:
                rgb = to_rgb(p.read_bytes())
                if rgb is not None:
                    tag = "lfw" if p.parent.name == "lfw" else ("ffhq" if cls == "real" else "stylegan")
                    save(rgb, config.SCENES_DIR / split / cls / tag / p.name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dff-real", type=int, default=4500)
    ap.add_argument("--dff-fake-each", type=int, default=1500)
    ap.add_argument("--dff-scene-share", type=float, default=0.5)
    ap.add_argument("--existing-to-scenes", type=int, default=1500)
    ap.add_argument("--skip-dff", action="store_true")
    ap.add_argument("--skip-genimage", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not args.skip_genimage:
        prepare_genimage()
    add_existing_faces_to_scenes(args.existing_to_scenes, args.seed)
    if not args.skip_dff:
        from src.face_detector import FaceDetector
        prepare_dff(args.dff_real, args.dff_fake_each, args.dff_scene_share, args.seed,
                    FaceDetector("auto"))

    for root in (config.FACES_DIR, config.SCENES_DIR):
        print(f"\n{root}")
        for split in ("train", "val", "test"):
            row = ", ".join(f"{c}={sum(1 for _ in (root / split / c).rglob('*.jpg'))}"
                            for c in ("real", "fake") if (root / split / c).exists())
            print(f"  {split:5s}: {row}")


if __name__ == "__main__":
    main()
