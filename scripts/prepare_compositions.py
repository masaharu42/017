from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ELEMENTS = ["Ni", "Fe", "Cr", "Nb", "Ta", "Co", "Mo", "Al", "Ti"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the composition workbook to a model-ready CSV.")
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/compositions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_excel(args.workbook, sheet_name="合金成分汇总", header=2)
    gh706_ranges = load_gh706_ranges(args.workbook)
    table = table.rename(columns={"编号": "alloy_id", "来源/备注": "source_note"})
    table = table.loc[pd.to_numeric(table["alloy_id"], errors="coerce").notna()].copy()
    table["alloy_id"] = table["alloy_id"].astype(int)

    output_rows: list[dict[str, object]] = []
    for record in table.to_dict("records"):
        alloy_id = int(record["alloy_id"])
        row: dict[str, object] = {
            "alloy_id": alloy_id,
            "conditional_ready": True,
            "composition_status": (
                "standard_range_midpoint_modeling_value"
                if alloy_id == 22
                else "exact_or_nominal_point"
            ),
            "source_note": record.get("source_note", ""),
        }

        numeric: dict[str, float] = {}
        balance_element: str | None = None
        for element in ELEMENTS:
            value = record.get(element)
            if isinstance(value, str) and value.strip() == "余量":
                balance_element = element
                numeric[element] = np.nan
            elif pd.isna(value):
                numeric[element] = 0.0
            else:
                numeric[element] = float(value)

        if balance_element:
            numeric[balance_element] = 100.0 - sum(
                value for name, value in numeric.items() if name != balance_element and not np.isnan(value)
            )

        total = float(sum(numeric.values()))
        if abs(total - 100.0) > 0.05:
            raise ValueError(f"Alloy {alloy_id} sums to {total:.4f} wt.%, expected 100 wt.%.")

        for element, value in numeric.items():
            row[element] = value
            if alloy_id == 22 and element in gh706_ranges:
                row[f"{element}_low"], row[f"{element}_high"] = gh706_ranges[element]
            else:
                row[f"{element}_low"] = value
                row[f"{element}_high"] = value
        row["composition_total"] = total
        row["exclusion_reason"] = ""
        output_rows.append(row)

    output = pd.DataFrame(output_rows).sort_values("alloy_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")

    exact = output.loc[output["conditional_ready"]]
    singleton_features = []
    for element in ELEMENTS:
        nonzero = int((exact[element].abs() > 1.0e-12).sum())
        if nonzero == 1:
            singleton_features.append(element)

    print(f"Wrote {len(output)} rows to {args.output.resolve()}")
    print(f"Condition-ready rows: {len(exact)}")
    print("Range-derived midpoint rows: [22]")
    print(f"Singleton non-zero features: {singleton_features}")


def parse_range(value: object) -> tuple[float, float]:
    if value is None or pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number, number
    text = str(value).strip().replace("≤", "<=")
    if text == "余量":
        return np.nan, np.nan
    upper = re.fullmatch(r"<=\s*([0-9.]+)", text)
    if upper:
        return 0.0, float(upper.group(1))
    interval = re.fullmatch(r"([0-9.]+)\s*-\s*([0-9.]+)", text)
    if interval:
        return float(interval.group(1)), float(interval.group(2))
    number = float(text)
    return number, number


def load_gh706_ranges(workbook: Path) -> dict[str, tuple[float, float]]:
    table = pd.read_excel(workbook, sheet_name="21-22号成分", header=9, usecols="A:B")
    ranges: dict[str, tuple[float, float]] = {}
    for record in table.to_dict("records"):
        element = str(record.get("元素", "")).strip()
        if element in ELEMENTS:
            low, high = parse_range(record.get("质量分数 wt.%"))
            if not (np.isnan(low) or np.isnan(high)):
                ranges[element] = (low, high)
    return ranges


if __name__ == "__main__":
    main()
