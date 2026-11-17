from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an EBSD image manifest for alloys 1-22.")
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--compositions", type=Path, default=Path("data/compositions.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--pattern", default="{alloy_id}.png")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Find one file containing both IPF and GB under image_dir/<alloy_id>/.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compositions = pd.read_csv(args.compositions)
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for record in compositions.to_dict("records"):
        alloy_id = int(record["alloy_id"])
        if args.discover:
            path = discover_ipfgb_image(args.image_dir, alloy_id)
        else:
            path = (args.image_dir / args.pattern.format(alloy_id=alloy_id)).resolve()
        if not path.is_file():
            missing.append(str(path))
            if not args.allow_missing:
                continue
            width = height = None
        else:
            with Image.open(path) as image:
                width, height = image.size

        rows.append(
            {
                "alloy_id": alloy_id,
                "image_path": str(path),
                "width": width,
                "height": height,
                "use_for_vae": bool(path.is_file()),
                "use_for_diffusion": bool(path.is_file()) and as_bool(record["conditional_ready"]),
                "notes": "GH706 standard-range midpoint used as modeling condition" if alloy_id == 22 else "",
            }
        )

    if missing and not args.allow_missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} expected images. First paths:\n{preview}\n"
            "Use --pattern or --allow-missing, or edit the manifest manually."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} manifest rows to {args.output.resolve()}")
    if missing:
        print(f"Missing image paths recorded: {len(missing)}")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_ipfgb_image(root: Path, alloy_id: int) -> Path:
    directory = (root / str(alloy_id)).resolve()
    if not directory.is_dir():
        return directory / "MISSING_IPF_GB_IMAGE"
    extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    candidates = [
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and re.search("ipf", path.name, re.IGNORECASE)
        and re.search("gb", path.name, re.IGNORECASE)
    ]
    if len(candidates) > 1:
        listing = "\n".join(str(path) for path in candidates)
        raise ValueError(f"Alloy {alloy_id} has multiple IPF+GB candidates:\n{listing}")
    return candidates[0] if candidates else directory / "MISSING_IPF_GB_IMAGE"


if __name__ == "__main__":
    main()
