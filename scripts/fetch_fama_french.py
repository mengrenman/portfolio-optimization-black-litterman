#!/usr/bin/env python3
"""Build the bundled Fama-French factor CSVs from the French data library.

Downloads (or, with ``--source-dir``, reads already-downloaded copies of) the
daily 3-factor, daily 5-factor, and monthly 3-factor zip files published by
Kenneth French, keeps only rows on or after ``--start``, and writes compact
LF-terminated CSVs to ``--output-dir`` (default ``data/raw/factors``).

Each raw file opens with a few lines of free-text preamble, a header row
beginning with an empty date-column name (``",Mkt-RF,..."``), one data row per
period, then a blank line, a copyright footer, and -- for the monthly file
only -- a second "Annual Factors" block. Only the first data block (up to
that first blank line) is parsed; the annual block is never read as monthly
data. Values are copied verbatim (whitespace-stripped, still in percent) from
the source text, so a published "0.10" is written out as "0.10", not
round-tripped through a float.

Usage:
    python scripts/fetch_fama_french.py
    python scripts/fetch_fama_french.py --source-dir /path/to/already-downloaded/zips
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# (zip filename, member filename inside the zip, output filename, factor
# columns as published, whether dates are monthly ("YYYYMM") rather than
# daily ("YYYYMMDD")).
_JOBS: tuple[tuple[str, str, str, tuple[str, ...], bool], ...] = (
    (
        "F-F_Research_Data_Factors_daily_CSV.zip",
        "F-F_Research_Data_Factors_daily.csv",
        "ff3_daily.csv",
        ("Mkt-RF", "SMB", "HML", "RF"),
        False,
    ),
    (
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "F-F_Research_Data_5_Factors_2x3_daily.csv",
        "ff5_daily.csv",
        ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
        False,
    ),
    (
        "F-F_Research_Data_Factors_CSV.zip",
        "F-F_Research_Data_Factors.csv",
        "ff3_monthly.csv",
        ("Mkt-RF", "SMB", "HML", "RF"),
        True,
    ),
)


def parse_ff_block(text: str) -> tuple[list[str], list[list[str]]]:
    """Parse the first data block of a raw Fama-French factor CSV export.

    Finds the header row that begins with an empty date-column name
    (``",Mkt-RF,..."``) and reads data rows until the first blank line after
    it. Everything past that blank line -- a copyright footer and, for the
    monthly file, a second "Annual Factors" block -- is ignored.

    Args:
        text: The decoded contents of one raw Fama-French CSV file.

    Returns:
        A tuple ``(columns, rows)``. ``columns`` are the factor column names
        exactly as published (e.g. ``["Mkt-RF", "SMB", "HML", "RF"]``).
        ``rows`` is a list of ``[date, value, value, ...]`` where every field
        is the source text with surrounding whitespace stripped -- still a
        string, still in percent, never parsed to float.

    Raises:
        ValueError: If no ``,Mkt-RF`` header row is found.
    """
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith(",Mkt-RF")), None)
    if header_idx is None:
        raise ValueError("Could not find a ',Mkt-RF' header row in the source text.")

    columns = [c.strip() for c in lines[header_idx].split(",")[1:]]

    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        if line.strip() == "":
            break
        rows.append([field.strip() for field in line.split(",")])

    return columns, rows


def _read_zip_member(zip_bytes: bytes, member: str) -> str:
    """Decode one member file out of an in-memory zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return archive.read(member).decode("utf-8")


def _fetch_zip_bytes(zip_name: str, source_dir: Path | None) -> bytes:
    """Return the raw bytes of one French-library zip file.

    Args:
        zip_name: The zip file's name, exactly as published.
        source_dir: When given, the zip is read from this directory instead
            of being downloaded.

    Returns:
        The zip file's raw bytes.

    Raises:
        FileNotFoundError: If ``source_dir`` is given but does not contain
            ``zip_name``.
    """
    if source_dir is not None:
        path = source_dir / zip_name
        if not path.exists():
            raise FileNotFoundError(f"Source zip not found: {path}")
        return path.read_bytes()

    url = f"{BASE_URL}/{zip_name}"
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        return response.read()


def _select_rows(rows: list[list[str]], start_token: str, *, monthly: bool) -> list[list[str]]:
    """Trim raw rows to ``start_token`` and reformat their date column.

    Args:
        rows: Raw ``[date, value, ...]`` rows as returned by
            :func:`parse_ff_block`.
        start_token: The earliest date to keep, as ``YYYYMMDD`` (daily) or
            ``YYYYMM`` (monthly); comparable lexicographically to the raw
            date strings since both are zero-padded and equal length.
        monthly: Whether the raw dates are ``YYYYMM`` rather than
            ``YYYYMMDD``.

    Returns:
        Rows on or after ``start_token``, with the date column rewritten to
        ``YYYY-MM`` (monthly) or ``YYYY-MM-DD`` (daily). Values are left
        untouched.
    """
    selected: list[list[str]] = []
    for raw_date, *values in rows:
        if raw_date < start_token:
            continue
        date = f"{raw_date[:4]}-{raw_date[4:6]}" if monthly else f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        selected.append([date, *values])
    return selected


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write ``rows`` to ``path`` as an LF-terminated CSV with ``header``."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    """Build the bundled Fama-French factor CSVs and print their coverage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory holding already-downloaded French zip files, read instead of the network.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/factors"),
        help="Directory to write the bundled CSVs to (default: data/raw/factors).",
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Earliest date to keep, as YYYY-MM-DD (default: 2018-01-01).",
    )
    args = parser.parse_args()

    try:
        date.fromisoformat(args.start)
    except ValueError:
        parser.error(f"--start must be YYYY-MM-DD, got {args.start!r}")

    daily_token = args.start.replace("-", "")
    monthly_token = daily_token[:6]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for zip_name, member, out_name, expected_columns, monthly in _JOBS:
        zip_bytes = _fetch_zip_bytes(zip_name, args.source_dir)
        text = _read_zip_member(zip_bytes, member)
        columns, rows = parse_ff_block(text)
        if tuple(columns) != expected_columns:
            raise ValueError(f"{member}: expected columns {expected_columns}, got {tuple(columns)}")

        start_token = monthly_token if monthly else daily_token
        selected = _select_rows(rows, start_token, monthly=monthly)
        if not selected:
            raise ValueError(f"{member}: no rows on or after {args.start}")

        out_path = args.output_dir / out_name
        _write_csv(out_path, ["date", *columns], selected)

        print(f"{out_name}: {len(selected)} rows, {selected[0][0]} to {selected[-1][0]} -> {out_path}")


if __name__ == "__main__":
    main()
