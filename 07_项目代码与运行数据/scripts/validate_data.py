from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EBSD manifest and image consistency.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--compositions", type=Path, default=Path("data/compositions.csv"))
    parser.add_argument("--heat-treatment", type=Path, default=Path("data/heat_treatment.csv"))
    parser.add_argument("--patch-size", type=int, default=192)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    compositions = pd.read_csv(args.compositions)
    heat_treatment = pd.read_csv(args.heat_treatment)
    known_ids = set(compositions["alloy_id"].astype(int))
    errors: list[str] = []
    dimensions: dict[tuple[int, int], int] = {}

    heat_treatment_ids = set(heat_treatment["alloy_id"].astype(int))
    if heat_treatment_ids != known_ids:
        errors.append(
            "heat_treatment.csv alloy IDs differ from compositions.csv: "
            f"missing={sorted(known_ids - heat_treatment_ids)}, "
            f"extra={sorted(heat_treatment_ids - known_ids)}"
        )
    process_columns = [column for column in heat_treatment.columns if column != "alloy_id"]
    unique_processes = heat_treatment[process_columns].drop_duplicates()
    if len(unique_processes) != 1:
        errors.append(
            "Expected one fixed heat-treatment process for all 22 alloys, "
            f"found {len(unique_processes)} distinct rows"
        )

    for record in manifest.to_dict("records"):
        alloy_id = int(record["alloy_id"])
        path = Path(str(record["image_path"]))
        if alloy_id not in known_ids:
            errors.append(f"Manifest alloy {alloy_id} is absent from compositions.csv")
        if not path.is_file():
            errors.append(f"Missing image: {path}")
            continue
        with Image.open(path) as image:
            if image.mode not in {"RGB", "RGBA"}:
                errors.append(f"Image is not RGB/RGBA: {path} ({image.mode})")
            if min(image.size) < args.patch_size:
                errors.append(f"Image is smaller than patch size: {path} {image.size}")
            dimensions[image.size] = dimensions.get(image.size, 0) + 1

    duplicates = manifest[manifest.duplicated(["alloy_id", "image_path"], keep=False)]
    if not duplicates.empty:
        errors.append("Manifest contains duplicate alloy_id/image_path rows")

    print(f"Rows: {len(manifest)}")
    print(f"Alloys: {sorted(manifest['alloy_id'].astype(int).unique().tolist())}")
    print(f"Image dimensions: {dimensions}")
    print(f"Heat-treatment rows: {len(heat_treatment)}; distinct processes: {len(unique_processes)}")
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()

