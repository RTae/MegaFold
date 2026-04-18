#!/usr/bin/env python3
"""
Query NCU report files for specific metrics.

Usage:
    # List all available metric names from a report
    python query_ncu_metrics.py list-metrics

    # Show key metrics for all reports in a table
    python query_ncu_metrics.py summary

    # Query specific metrics (substring match)
    python query_ncu_metrics.py query "sm__throughput" "dram__throughput"

    # Query with kernel name filter
    python query_ncu_metrics.py query "sm__throughput" --filter-kernel "_attn_bwd"

    # Export to CSV
    python query_ncu_metrics.py summary --output results.csv

    # Use a different reports directory
    python query_ncu_metrics.py summary --reports-dir ./ncu_reports
"""

import argparse
import csv
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

FILENAME_RE = re.compile(
    r"^(?P<impl>[a-z0-9]+)_(?P<mode>fwd|bwd|full)_ctx(?P<n_ctx>\d+)\.ncu-rep$"
)

# Key metrics for the summary view
SUMMARY_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_issue_stalled_mio_throttle_pct.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "smsp__inst_executed_op_global_red.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_red_hit_rate.pct",
    "lts__t_sectors_srcunit_tex_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "lts__t_sectors_srcunit_tex_op_red_lookup_miss.sum"
]


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return {
        "impl": m.group("impl"),
        "mode": m.group("mode"),
        "n_ctx": int(m.group("n_ctx")),
    }


def get_report_files(reports_dir: Path):
    reports = sorted(reports_dir.glob("*_ctx*.ncu-rep"))
    if not reports:
        print(f"No .ncu-rep files in {reports_dir}", file=sys.stderr)
        sys.exit(1)
    return reports


def run_ncu_csv(report_path: Path):
    """Import a report and return (header_list, units_list, data_rows)."""
    cmd = ["ncu", "--import", str(report_path), "--csv", "--page", "raw"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"WARN: ncu --import failed on {report_path.name}", file=sys.stderr)
        return [], [], []

    lines = result.stdout.splitlines()
    if len(lines) < 3:
        return [], [], []

    reader = csv.reader(StringIO("\n".join(lines)))
    rows = list(reader)
    header = rows[0] if rows else []
    units = rows[1] if len(rows) > 1 else []
    data = rows[2:] if len(rows) > 2 else []
    return header, units, data


def cmd_list_metrics(args):
    """List all metric names available in the first report file."""
    reports = get_report_files(Path(args.reports_dir))
    report = reports[0]
    print(f"Metrics from: {report.name}")
    print(f"{'='*80}")

    header, units, _ = run_ncu_csv(report)
    # Skip the first 11 standard columns (ID, Process ID, etc.)
    for i, (name, unit) in enumerate(zip(header, units)):
        name = name.strip('"')
        unit = unit.strip('"')
        if i < 11:
            continue
        print(f"  {name}  [{unit}]" if unit else f"  {name}")

    print(f"\nTotal: {len(header) - 11} metrics (+ 11 standard columns)")


def cmd_query(args):
    """Query reports for metrics matching the given substrings."""
    reports = get_report_files(Path(args.reports_dir))
    patterns = args.metrics

    rows_out = []
    matched_cols = None

    for report in reports:
        meta = parse_filename(report)
        if not meta:
            continue
        if args.filter_impl and meta["impl"] not in args.filter_impl:
            continue
        if args.filter_mode and meta["mode"] not in args.filter_mode:
            continue

        header, units, data = run_ncu_csv(report)
        if not header:
            continue

        # Find columns matching any pattern
        if matched_cols is None:
            matched_cols = []
            for i, col in enumerate(header):
                col_clean = col.strip('"')
                if any(p.lower() in col_clean.lower() for p in patterns):
                    matched_cols.append((i, col_clean, units[i].strip('"') if i < len(units) else ""))

        if not matched_cols:
            print(f"No metrics matched: {patterns}", file=sys.stderr)
            return

        for row in data:
            kernel_name = row[4].strip('"') if len(row) > 4 else "?"
            if args.filter_kernel and args.filter_kernel not in kernel_name:
                continue
            entry = {
                "impl": meta["impl"],
                "mode": meta["mode"],
                "n_ctx": meta["n_ctx"],
                "kernel": kernel_name,
            }
            for col_idx, col_name, _ in matched_cols:
                entry[col_name] = row[col_idx].strip('"') if col_idx < len(row) else ""
            rows_out.append(entry)

    if not rows_out:
        print("No matching rows found.", file=sys.stderr)
        return

    # Print or save
    fieldnames = ["impl", "mode", "n_ctx", "kernel"] + [c[1] for c in matched_cols]
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote {len(rows_out)} rows to {args.output}")
    else:
        _print_table(fieldnames, rows_out, matched_cols)


def cmd_summary(args):
    """Show key metrics for all reports."""
    args.metrics = [m for m in SUMMARY_METRICS]
    args.filter_kernel = args.filter_kernel if hasattr(args, "filter_kernel") else None
    args.filter_impl = args.filter_impl if hasattr(args, "filter_impl") else None
    args.filter_mode = args.filter_mode if hasattr(args, "filter_mode") else None
    cmd_query(args)


def _print_table(fieldnames, rows, matched_cols):
    """Print a formatted table to stdout."""
    # Truncate kernel name for display
    max_kernel = 50
    display_rows = []
    for row in rows:
        r = dict(row)
        k = r.get("kernel", "")
        if len(k) > max_kernel:
            r["kernel"] = "..." + k[-(max_kernel - 3):]
        display_rows.append(r)

    # Compute column widths
    widths = {}
    for f in fieldnames:
        widths[f] = max(len(f), max((len(str(r.get(f, ""))) for r in display_rows), default=0))
        widths[f] = min(widths[f], 60)

    # Header
    hdr = " | ".join(f.rjust(widths[f]) if f not in ("impl", "mode", "kernel") else f.ljust(widths[f])
                      for f in fieldnames)
    print(hdr)
    print("-" * len(hdr))

    # Rows
    for r in display_rows:
        line = " | ".join(
            str(r.get(f, "")).rjust(widths[f]) if f not in ("impl", "mode", "kernel")
            else str(r.get(f, "")).ljust(widths[f])
            for f in fieldnames
        )
        print(line)

    print(f"\n{len(rows)} rows")


def main():
    ap = argparse.ArgumentParser(description="Query NCU report metrics")
    ap.add_argument("--reports-dir", default="./ncu_reports")
    sub = ap.add_subparsers(dest="command", required=True)

    # list-metrics
    sub.add_parser("list-metrics", help="List all available metric names")

    # summary
    sp = sub.add_parser("summary", help="Show key metrics for all reports")
    sp.add_argument("--output", default=None, help="Save to CSV")
    sp.add_argument("--filter-kernel", default=None)
    sp.add_argument("--filter-impl", nargs="*", default=None)
    sp.add_argument("--filter-mode", nargs="*", default=None)

    # query
    qp = sub.add_parser("query", help="Query specific metrics by name substring")
    qp.add_argument("metrics", nargs="+", help="Metric name substrings to match")
    qp.add_argument("--output", default=None, help="Save to CSV")
    qp.add_argument("--filter-kernel", default=None)
    qp.add_argument("--filter-impl", nargs="*", default=None)
    qp.add_argument("--filter-mode", nargs="*", default=None)

    args = ap.parse_args()

    if args.command == "list-metrics":
        cmd_list_metrics(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
