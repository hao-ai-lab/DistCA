#!/usr/bin/env python3
"""
Parse NCCL-tests output files (e.g. all_gather_2_0.txt) and
merge their tables into a single CSV.

Each CSV row contains
    op,ngpu,a,
    size_bytes,count_elems,
    t_us_out,algbw_out_gbs,busbw_out_gbs,
    t_us_in ,algbw_in_gbs ,busbw_in_gbs
"""

import argparse, csv, glob, re
from pathlib import Path

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument(
    "-d", "--dir", default=".", help="directory with *.txt benchmark files"
)
p.add_argument(
    "-o", "--out", default="nccl_perf.csv", help="CSV file to create/overwrite"
)
args = p.parse_args()

# ---------- helpers ----------
def parse_filename(name: str):
    """all_gather_2_0.txt  ->  ('all_gather', 2, 0)"""
    stem = Path(name).stem
    op, ngpu, dtype = stem.rsplit("-", 2)
    return op, int(ngpu), dtype


def is_data_row(line: str) -> bool:
    """True if the line starts with a decimal size (no '#')."""
    return re.match(r"^\s*\d", line) is not None


def parse_row(tokens):
    """tokens = line.split(); returns the 10 numeric columns we care about."""
    size, count = int(tokens[0]), int(tokens[1])
    (
        t_out,
        alg_out,
        bus_out,
        _,  # #wrong column we ignore
        t_in,
        alg_in,
        bus_in,
    ) = map(float, tokens[5:12])
    return size, count, t_out, alg_out, bus_out, t_in, alg_in, bus_in


# ---------- scan files ----------
rows = []
for fname in glob.glob(f"{args.dir}/*-*-*.perf.txt"):
    op, ngpu, dtype = parse_filename(fname)
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if is_data_row(line):
                toks = line.split()
                (
                    size,
                    count,
                    t_out,
                    alg_out,
                    bus_out,
                    t_in,
                    alg_in,
                    bus_in,
                ) = parse_row(toks)
                rows.append(
                    [
                        op,
                        ngpu,
                        dtype,
                        size,
                        count,
                        t_out,
                        alg_out,
                        bus_out,
                        t_in,
                        alg_in,
                        bus_in,
                    ]
                )

# ---------- write CSV ----------
header = [
    "op",
    "ngpu",
    "dtype",
    "size_bytes",
    "count_elems",
    "t_us_out",
    "algbw_out_gbs",
    "busbw_out_gbs",
    "t_us_in",
    "algbw_in_gbs",
    "busbw_in_gbs",
]

with open(args.out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Parsed {len(rows)} rows → {args.out}")