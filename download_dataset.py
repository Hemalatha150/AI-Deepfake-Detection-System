"""Download the "140k Real and Fake Faces" dataset from Hugging Face.

70k real faces (Flickr-Faces-HQ) + 70k StyleGAN-generated faces, already
cropped and split. Images are written to:

    data/faces/{train,val,test}/{real,fake}/*.jpg

The full dataset is ~4 GB in 10 shards. Use --train-shards/--eval-shards to
download only part of it (each train shard has ~16.7k images, each
validation/test shard ~10k).

    python download_dataset.py                      # 1 train + 1 val + 1 test shard
    python download_dataset.py --train-shards 6 --eval-shards 2   # everything
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

import config

REPO = "https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces/resolve/main/data"
SHARDS = {"train": ("train", 6), "val": ("validation", 2), "test": ("test", 2)}
LABELS = {0: "real", 1: "fake"}

# Extra REAL faces from everyday cameras (news photos). Adding them stops the model
# from treating "not a clean Flickr photo" as a sign of manipulation, which is what
# made genuine webcam / phone recordings come out as fake.
LFW_URL = "https://huggingface.co/api/datasets/marcelohaps/lfw/parquet/default/train/0.parquet"


def add_lfw(cache: Path, out_root: Path, train_limit: int, seed: int = 42):
    """Split LFW by identity (80/10/10) into <out>/<split>/real/lfw/."""
    import random
    parquet = download(LFW_URL, cache / "lfw.parquet")
    table = pq.read_table(parquet, columns=["image", "identity"])
    ids = sorted(set(table.column("identity").to_pylist()))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    split_of = {i: "test" if k < n * 0.1 else "val" if k < n * 0.2 else "train"
                for k, i in enumerate(ids)}
    counts = {"train": 0, "val": 0, "test": 0}
    for img, ident in zip(table.column("image").to_pylist(), table.column("identity").to_pylist()):
        split = split_of[ident]
        if split == "train" and counts["train"] >= train_limit:
            continue
        d = out_root / split / "real" / "lfw"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"lfw_{counts[split]:05d}.jpg").write_bytes(img["bytes"])
        counts[split] += 1
    print(f"LFW real faces added: {counts}")


def download(url: str, dest: Path):
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out, \
            tqdm(total=int(resp.headers.get("Content-Length", 0)), unit="B",
                 unit_scale=True, desc=dest.name) as bar:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def extract(parquet: Path, out_root: Path, split: str, limit_per_class: int | None):
    counts = {0: 0, 1: 0}
    for label in LABELS.values():
        (out_root / split / label).mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=512, columns=["image", "label"]):
        for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            if limit_per_class and counts[label] >= limit_per_class:
                continue
            path = out_root / split / LABELS[label] / f"{parquet.stem}_{counts[label]:06d}.jpg"
            path.write_bytes(img["bytes"])
            counts[label] += 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=config.FACES_DIR)
    ap.add_argument("--cache", type=Path, default=config.DATA_DIR / "downloads")
    ap.add_argument("--train-shards", type=int, default=1, choices=range(1, 7))
    ap.add_argument("--eval-shards", type=int, default=1, choices=range(1, 3))
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="max images per class per shard")
    ap.add_argument("--keep-parquet", action="store_true")
    ap.add_argument("--extra-real", choices=["lfw"], default=None,
                    help="also add real-world camera faces (LFW) to the real class")
    ap.add_argument("--extra-only", action="store_true", help="only add the extra real faces")
    ap.add_argument("--extra-train-limit", type=int, default=6000)
    args = ap.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)

    if args.extra_real == "lfw":
        add_lfw(args.cache, args.output, args.extra_train_limit)
    if args.extra_only:
        return

    for split, (remote, total) in SHARDS.items():
        n = args.train_shards if split == "train" else args.eval_shards
        for i in range(n):
            name = f"{remote}-{i:05d}-of-{total:05d}.parquet"
            parquet = download(f"{REPO}/{name}", args.cache / name)
            counts = extract(parquet, args.output, split, args.limit_per_class)
            print(f"{split}: {name} -> real={counts[0]}, fake={counts[1]}")
            if not args.keep_parquet:
                parquet.unlink()
    print(f"\nDataset ready in {args.output}")


if __name__ == "__main__":
    main()
