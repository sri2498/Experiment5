# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert COCO-style SAR ship annotations (e.g. raw HRSID train2017.json/test2017.json) to YOLO layout.

The HRSID distribution ships 800x800 JPEG chips with COCO JSON annotations. This tool builds the directory
layout Ultralytics expects and writes a ready-to-train data YAML:

    out/
      images/{split}/*.jpg   (copied, or hard-linked with --link)
      labels/{split}/*.txt   (class cx cy w h, normalised)
      data.yaml

Example:
    python research/coco_to_yolo.py \
        --images /data/HRSID/images \
        --json train=/data/HRSID/annotations/train2017.json test=/data/HRSID/annotations/test2017.json \
        --out /content/datasets/hrsid

Note: if your HRSID copy is already in YOLO format (images + txt labels), you do not need this script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def convert_split(images_dir: Path, json_path: Path, out_dir: Path, split: str, link: bool = False) -> list[str]:
    """Convert one COCO JSON into YOLO images/labels for `split`. Returns the ordered class names."""
    data = json.loads(json_path.read_text())
    cat_ids = sorted(c["id"] for c in data["categories"])
    id2idx = {cid: i for i, cid in enumerate(cat_ids)}
    names = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]

    anns = defaultdict(list)
    for a in data.get("annotations", []):
        if not a.get("iscrowd", 0):
            anns[a["image_id"]].append(a)

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img, n_box = 0, 0
    for img in data["images"]:
        fname = Path(img["file_name"]).name
        src = images_dir / fname
        if not src.exists():
            print(f"[warn] missing image: {src}")
            continue
        dst = img_out / fname
        if not dst.exists():
            if link:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        w, h = float(img["width"]), float(img["height"])
        lines = []
        for a in anns.get(img["id"], []):
            x, y, bw, bh = a["bbox"]  # COCO xywh, absolute
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{id2idx[a['category_id']]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
    print(f"[{split}] {n_img} images, {n_box} boxes -> {img_out}")
    return names


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="directory containing all chip images")
    ap.add_argument("--json", nargs="+", required=True, help="split=path pairs, e.g. train=/x/train2017.json")
    ap.add_argument("--out", type=Path, required=True, help="output dataset root")
    ap.add_argument("--link", action="store_true", help="hard-link images instead of copying")
    args = ap.parse_args()

    names, splits = [], []
    for pair in args.json:
        split, _, path = pair.partition("=")
        assert path, f"expected split=path, got {pair!r}"
        names = convert_split(args.images, Path(path), args.out, split, args.link)
        splits.append(split)

    yaml_lines = [f"path: {args.out.resolve()}"]
    yaml_lines.append(f"train: images/{'train' if 'train' in splits else splits[0]}")
    val = "val" if "val" in splits else ("test" if "test" in splits else splits[-1])
    yaml_lines.append(f"val: images/{val}")
    if "test" in splits:
        yaml_lines.append("test: images/test")
    yaml_lines.append("names:")
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
