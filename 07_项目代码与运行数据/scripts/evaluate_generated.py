from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from ebsd_diffusion.metrics import compare_descriptor_sets, describe_image, descriptors_as_dict


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated RGB IPF+GB images with real images.")
    parser.add_argument("--real", type=Path, required=True, nargs="+")
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("08_训练与生成结果/06_评价报告/evaluation.json"),
    )
    parser.add_argument("--gb-threshold", type=float, default=0.18)
    args = parser.parse_args()

    generated_paths = sorted(
        path for path in args.generated_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not generated_paths:
        raise FileNotFoundError(f"No generated images found under {args.generated_dir}")

    all_paths = list(args.real) + generated_paths
    sizes = []
    for path in all_paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    crop_size = (min(size[0] for size in sizes), min(size[1] for size in sizes))
    if any(width > crop_size[0] * 2 or height > crop_size[1] * 2 for width, height in sizes):
        print(
            "Warning: evaluation images differ by more than 2x in size; "
            "metrics use only a centered common crop and are diagnostic only."
        )

    real_desc = [describe_image(path, args.gb_threshold, crop_size=crop_size) for path in args.real]
    generated_desc = [
        describe_image(path, args.gb_threshold, crop_size=crop_size) for path in generated_paths
    ]
    report = compare_descriptor_sets(real_desc, generated_desc)
    report["created_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["gb_threshold"] = float(args.gb_threshold)
    report["evaluation_crop_width"] = crop_size[0]
    report["evaluation_crop_height"] = crop_size[1]
    report["real_files"] = [str(path.resolve()) for path in args.real]
    report["generated_files"] = [str(path.resolve()) for path in generated_paths]
    report["per_image"] = {
        "real": [descriptors_as_dict(value) for value in real_desc],
        "generated": [descriptors_as_dict(value) for value in generated_desc],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"per_image"}}, ensure_ascii=False, indent=2))
    print(f"Saved report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
