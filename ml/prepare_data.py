"""Build a YOLO-format dataset from a config, and record where it came from.

Currently implements the Open Images source. That choice is a licensing decision,
not a technical one: WIDER FACE is the obvious face-detection dataset, but it ships
under CC-BY-NC-ND, and a model trained on it is arguably a derivative work -- a poor
foundation for something linked from job applications. Open Images annotations are
CC-BY-4.0 and its images are CC-BY-2.0, so the weights this produces are safe to
publish.

Open Images boxes are already normalised to [0,1], so the conversion to YOLO's
`class cx cy w h` is exact -- no image dimensions are needed and no rounding error
creeps in from a resize.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ANNOTATION_URL = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
DATA_ROOT = Path(__file__).resolve().parent / "datasets"


def download(url: str, target: Path) -> Path:
    """Fetch to a temp file then rename, so an interrupted run cannot leave a
    half-written file that a later run mistakes for a complete one."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        partial.write_bytes(response.read())
    partial.replace(target)
    return target


def collect_boxes(split: str, wanted: set[str]) -> dict[str, list[tuple[float, ...]]]:
    """Stream the split's annotation CSV and keep only the classes we asked for."""
    csv_path = DATA_ROOT / "openimages" / f"{split}-annotations-bbox.csv"
    print(f"[prepare] fetching {split} annotations ...", flush=True)
    download(ANNOTATION_URL.format(split=split), csv_path)

    boxes: dict[str, list[tuple[float, ...]]] = {}
    kept = skipped = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["LabelName"] not in wanted:
                continue
            # Group boxes span a crowd rather than one face, and depictions are
            # drawings of faces. Both teach the model the wrong thing about what a
            # face looks like and what counts as one instance.
            if row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1":
                skipped += 1
                continue
            x0, x1 = float(row["XMin"]), float(row["XMax"])
            y0, y1 = float(row["YMin"]), float(row["YMax"])
            w, h = x1 - x0, y1 - y0
            if w <= 0.002 or h <= 0.002:  # degenerate slivers
                skipped += 1
                continue
            boxes.setdefault(row["ImageID"], []).append(((x0 + x1) / 2, (y0 + y1) / 2, w, h))
            kept += 1
    print(f"[prepare] {split}: {kept} boxes over {len(boxes)} images ({skipped} filtered out)")
    return boxes


def fetch_split(split: str, boxes: dict[str, list[tuple[float, ...]]], limit: int,
                out_root: Path, subset: str, workers: int = 16) -> int:
    image_ids = sorted(boxes)
    random.Random(0).shuffle(image_ids)  # deterministic subset across re-runs
    image_ids = image_ids[:limit]

    image_dir = out_root / "images" / subset
    label_dir = out_root / "labels" / subset
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    def one(image_id: str) -> bool:
        target = image_dir / f"{image_id}.jpg"
        try:
            download(IMAGE_URL.format(split=split, image_id=image_id), target)
        except Exception:
            return False
        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes[image_id]]
        (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, i): i for i in image_ids}
        for future in as_completed(futures):
            if future.result():
                done += 1
            else:
                failed += 1
            if (done + failed) % 250 == 0:
                print(f"[prepare] {subset}: {done + failed}/{len(image_ids)}", flush=True)
    print(f"[prepare] {subset}: {done} images written, {failed} failed")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = cfg.get("data") or {}
    if source.get("source") != "openimages":
        sys.exit(f"unsupported data source: {source.get('source')!r}")

    wanted = {entry["id"] for entry in source["classes"]}
    names = [entry["name"] for entry in source["classes"]]
    out_root = DATA_ROOT / cfg["id"]

    # Distinct Open Images splits, so train and val share no images by construction
    # rather than by a random split we would have to trust.
    for split, subset, limit in (
        (source["train_split"], "train", int(source["max_train"])),
        (source["val_split"], "val", int(source["max_val"])),
    ):
        boxes = collect_boxes(split, wanted)
        fetch_split(split, boxes, limit, out_root, subset)

    dataset_yaml = out_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "names": {i: n for i, n in enumerate(names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"[prepare] wrote {dataset_yaml}")
    for subset in ("train", "val"):
        count = len(list((out_root / "images" / subset).glob("*.jpg")))
        print(f"[prepare] {subset}: {count} images")


if __name__ == "__main__":
    main()
