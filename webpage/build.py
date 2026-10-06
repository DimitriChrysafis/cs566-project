#!/usr/bin/env python3
"""Regenerate the results tables in index.html from results/*.csv.

Rows are grouped by a 'method' (preferred) or 'condition' column and the
absrel / rmse / delta1 columns are averaged. Missing or unparsable csvs
leave the existing placeholder rows untouched. Stdlib only, idempotent.
"""

import csv
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "webpage" / "index.html"
RESULTS = ROOT / "results"

METRICS = ("absrel", "rmse", "delta1")
GROUP_COLS = ("method", "condition")
SPLIT_COLS = ("split", "region")

LABELS = {
    "aligned_dav2": "Aligned DA-V2",
    "focus_raw": "DFF raw",
    "focus": "DFF + all-focus smooth",
    "scale_fit": "DA-V2 + scale fit",
    "refined": "Refined (ours)",
    "combined": "Combined (ours)",
    "dav2_lsq": "DA-V2 (scale+shift)",
    "dav2_median": "DA-V2 (median scale)",
    "dav2s_lsq": "DA-V2 Small (scale+shift)",
    "dav2s_irls": "DA-V2 Small (robust IRLS)",
    "dav2s_median": "DA-V2 Small (median scale)",
    "dav2b_lsq": "DA-V2 Base (scale+shift)",
    "dav2l_lsq": "DA-V2 Large (scale+shift)",
    "hypersim_none": "DA-V2 Metric Indoor (unaligned)",
    "hypersim_lsq": "DA-V2 Metric Indoor (aligned)",
    "dff_raw": "DFF raw",
    "dff": "DFF + all-focus smooth",
}

JOBS = (
    ("NYU", [(RESULTS / "nyu_models.csv", {"crop": "eigen"}),
             (RESULTS / "nyu_dav2.csv", None)]),
    ("SYNTH", [(RESULTS / "refine_synth.csv", {"set": "eval"})]),
)


def read_csv(path):
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames or [], list(reader)
    except OSError:
        return [], []


def find_col(fields, name):
    for f in fields:
        if f and f.strip().lower() == name:
            return f
    return None


def aggregate_per_image(path):
    """nyu_dav2.csv is one row per image with d1 and med_* twins."""
    fields, rows = read_csv(path)
    if not rows:
        return [], {}
    need = ("absrel", "rmse", "d1", "med_absrel", "med_rmse", "med_d1")
    cols = {c: find_col(fields, c) for c in need}
    if not all(cols.values()):
        return [], {}
    groups = {"dav2_lsq": {"absrel": [], "rmse": [], "delta1": []},
              "dav2_median": {"absrel": [], "rmse": [], "delta1": []}}
    for r in rows:
        for key, trio in (("dav2_lsq", ("absrel", "rmse", "d1")),
                          ("dav2_median", ("med_absrel", "med_rmse", "med_d1"))):
            for m, c in zip(METRICS, trio):
                try:
                    groups[key][m].append(float(r[cols[c]]))
                except (ValueError, TypeError):
                    continue
    return ["dav2_lsq", "dav2_median"], groups


def aggregate(specs):
    """Return ordered {group: {metric: [values]}} pooled across csv files.

    specs: list of (path, filters). filters is a dict of column -> required
    value applied to rows, or None. Once a file contributes rows, later
    fallback files are skipped.
    """
    groups = {}
    order = []
    for path, filters in specs:
        if order:
            break
        if not path.exists():
            continue
        fields, rows = read_csv(path)
        if not fields or not rows:
            continue
        key_col = next((c for c in (find_col(fields, g) for g in GROUP_COLS) if c), None)
        if key_col is None:
            if path.name == "nyu_dav2.csv":
                return aggregate_per_image(path)
            continue
        if filters:
            rows = [r for r in rows if all(
                (r.get(k) or "").strip() == v for k, v in filters.items())]
            if not rows:
                continue
        mcols = {m: find_col(fields, m) for m in METRICS}
        if not any(mcols.values()):
            continue
        split_col = next((c for c in (find_col(fields, s) for s in SPLIT_COLS) if c), None)
        if split_col and any((r.get(split_col) or "").strip().lower() == "all" for r in rows):
            rows = [r for r in rows if (r.get(split_col) or "").strip().lower() == "all"]
        for row in rows:
            key = (row.get(key_col) or "").strip()
            if not key:
                continue
            if key not in groups:
                groups[key] = {m: [] for m in METRICS}
                order.append(key)
            for m, col in mcols.items():
                if col is None:
                    continue
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                try:
                    groups[key][m].append(float(raw))
                except ValueError:
                    continue
    return order, groups


def table_rows(order, groups):
    rows = []
    for key in order:
        cells = []
        for m in METRICS:
            vals = groups[key][m]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "\u2014")
        rows.append(
            f"<tr><td>{html.escape(LABELS.get(key.lower(), key))}</td>"
            f'<td class="num">{cells[0]}</td>'
            f'<td class="num">{cells[1]}</td>'
            f'<td class="num">{cells[2]}</td></tr>'
        )
    return rows


def replace_region(text, name, rows):
    start = f"<!--RESULTS:{name}-->"
    end = f"<!--/RESULTS:{name}-->"
    i = text.find(start)
    j = text.find(end, i + len(start)) if i != -1 else -1
    if i == -1 or j == -1:
        return text, False
    line_start = text.rfind("\n", 0, i) + 1
    indent = text[line_start:i]
    body = "\n".join(indent + r for r in rows)
    return text[: i + len(start)] + "\n" + body + "\n" + indent + text[j:], True


def main():
    if not INDEX.exists():
        sys.exit(f"index.html not found at {INDEX}")
    text = INDEX.read_text(encoding="utf-8")

    for name, paths in JOBS:
        order, groups = aggregate(paths)
        if not order:
            print(f"{name}: no usable csv data, placeholder kept")
            continue
        text, ok = replace_region(text, name, table_rows(order, groups))
        if ok:
            print(f"{name}: wrote {len(order)} row(s)")
        else:
            print(f"{name}: markers not found in index.html, skipped")

    INDEX.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
