#!/usr/bin/env python3
"""Generate a roofline plot from Nsight Compute summary CSV data."""

from __future__ import annotations

import argparse
import csv
import functools
from pathlib import Path
import subprocess
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HBM_BW_BYTES_PER_S = 3.35e12
PEAK_FLOPS_PER_S = 989e12

BATCH = 1
N_HEADS = 4
HEAD_DIM = 32

FLOP_FACTORS = {
    "fwd": 4,
    "bwd": 10,
}

TARGET_NS = [256, 384, 512, 1024]

KERNEL_MAP: dict[str, dict[str, list[str]]] = {
    "fwd": {
        "fa1_bias": ["_fwd_kernel"],
        "fa4": ["FlashAttentionForwardSm90", "flash_attncuteflash_fwd_sm90"],
        "flashbias": ["_fwd_kernel"],
        "megafold": ["_attn_fwd"],
        "sdpa": ["cublas"],
        "sdpa_no_bias": ["flash_fwd_kernel", "flash_fwd_splitkv_kernel"],
    },
    "bwd": {
        "fa1_bias": ["_bwd_kernel"],
        "fa4": ["FlashAttentionBackwardSm90", "flash_attncuteflash_bwd_sm90"],
        "flashbias": ["_bwd_kernel"],
        "megafold": ["_attn_bwd_dk_dv", "_attn_bwd_dq"],
        "sdpa": ["cublas"],
        "sdpa_no_bias": ["flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("summary.csv"),
        help="Path to the summary.csv exported from query_ncu_metrics.py",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("ncu_reports"),
        help="Optional NCU reports directory used to infer metric units from a .ncu-rep file.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(FLOP_FACTORS),
        default="bwd",
        help="Attention mode to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("roofline_backward.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "n_ctx",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "gpu__time_duration.sum",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def unit_scale(unit: str) -> float:
    scales = {
        "byte": 1.0,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9,
        "nsecond": 1e-9,
        "usecond": 1e-6,
        "us": 1e-6,
        "msecond": 1e-3,
        "second": 1.0,
        "s": 1.0,
    }
    cleaned = unit.strip().strip('"')
    if cleaned not in scales:
        raise KeyError(f"Unsupported NCU unit: {cleaned}")
    return scales[cleaned]


@functools.lru_cache(maxsize=None)
def infer_metric_scales(report_path: Path) -> dict[str, float]:
    default = {
        "dram__bytes_read.sum": 1.0,
        "dram__bytes_write.sum": 1.0,
        "gpu__time_duration.sum": 1e-6,
    }
    if not report_path.exists():
        return default

    cmd = ["ncu", "--import", str(report_path), "--csv", "--page", "raw"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return default

    rows = list(csv.reader(StringIO(result.stdout)))
    if len(rows) < 2:
        return default

    header = rows[0]
    units = rows[1]
    scales = dict(default)
    for metric in scales:
        if metric in header:
            idx = header.index(metric)
            scales[metric] = unit_scale(units[idx])
    return scales


def normalize_metrics(df: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["dram__bytes_read_bytes"] = np.nan
    out["dram__bytes_write_bytes"] = np.nan
    out["gpu__time_duration_seconds"] = np.nan

    for (impl, mode, n_ctx), row_idx in out.groupby(["impl", "mode", "n_ctx"], sort=False).groups.items():
        report_path = reports_dir / f"{impl}_{mode}_ctx{int(n_ctx)}.ncu-rep"
        scales = infer_metric_scales(report_path)
        out.loc[row_idx, "dram__bytes_read_bytes"] = (
            out.loc[row_idx, "dram__bytes_read.sum"] * scales["dram__bytes_read.sum"]
        )
        out.loc[row_idx, "dram__bytes_write_bytes"] = (
            out.loc[row_idx, "dram__bytes_write.sum"] * scales["dram__bytes_write.sum"]
        )
        out.loc[row_idx, "gpu__time_duration_seconds"] = (
            out.loc[row_idx, "gpu__time_duration.sum"] * scales["gpu__time_duration.sum"]
        )

    return out


def aggregate_kernels(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["impl", "mode", "n_ctx", "kernel"], as_index=False)[
            ["dram__bytes_read_bytes", "dram__bytes_write_bytes", "gpu__time_duration_seconds", "gpu__time_duration.sum"]
        ]
        .sum()
    )
    grouped["bytes"] = grouped["dram__bytes_read_bytes"] + grouped["dram__bytes_write_bytes"]
    return grouped


def pick_main(group: pd.DataFrame, impl: str, mode: str) -> pd.Series:
    patterns = KERNEL_MAP.get(mode, {}).get(impl, [])
    if patterns:
        mask = group["kernel"].fillna("").apply(lambda value: any(pattern in value for pattern in patterns))
        matched = group[mask]
        if not matched.empty:
            return matched.sort_values("gpu__time_duration.sum", ascending=False).iloc[0]
    return group.sort_values("gpu__time_duration.sum", ascending=False).iloc[0]


def select_points(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    aggregated = aggregate_kernels(df)
    filtered = aggregated[(aggregated["mode"] == mode) & (aggregated["n_ctx"].isin(TARGET_NS))].copy()

    rows: list[pd.Series] = []
    for (impl, n_ctx), group in filtered.groupby(["impl", "n_ctx"], sort=True):
        rows.append(pick_main(group, impl=impl, mode=mode))

    selected = pd.DataFrame(rows).reset_index(drop=True)
    flop_factor = FLOP_FACTORS[mode]
    selected["flops"] = flop_factor * BATCH * N_HEADS * (selected["n_ctx"] ** 2) * HEAD_DIM
    selected["time_s"] = selected["gpu__time_duration_seconds"]
    selected = selected[(selected["bytes"] > 0) & (selected["time_s"] > 0)].copy()
    selected["intensity"] = selected["flops"] / selected["bytes"]
    selected["tflops_s"] = (selected["flops"] / selected["time_s"]) / 1e12
    return selected.sort_values(["impl", "n_ctx"]).reset_index(drop=True)


def plot_roofline(points: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    x = np.logspace(-1, 2, 512)
    roofline_tflops = np.minimum(x * HBM_BW_BYTES_PER_S, PEAK_FLOPS_PER_S) / 1e12
    ax.plot(x, roofline_tflops, color="black", linewidth=2.2, label="H100 roofline")
    ax.axhline(PEAK_FLOPS_PER_S / 1e12, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    impls = sorted(points["impl"].unique())
    colors = {impl: color for impl, color in zip(impls, plt.cm.tab10.colors, strict=False)}
    markers = {
        256: "o",
        384: "s",
        512: "^",
        1024: "D",
    }

    for row in points.itertuples(index=False):
        label = f"{row.impl} / N={int(row.n_ctx)}"
        ax.scatter(
            row.intensity,
            row.tflops_s,
            s=90,
            marker=markers.get(int(row.n_ctx), "o"),
            color=colors[row.impl],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.01, 2000)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved performance (TFLOP/s)")
    ax.set_title("Roofline — backward attention kernels @ H100")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    uniq: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        uniq.setdefault(label, handle)
    ax.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")


def print_summary(points: pd.DataFrame) -> None:
    cols = [
        "impl",
        "mode",
        "n_ctx",
        "kernel",
        "bytes",
        "gpu__time_duration.sum",
        "intensity",
        "tflops_s",
    ]
    printable = points[cols].copy()
    printable = printable.rename(columns={"gpu__time_duration.sum": "time_us"})
    printable["bytes"] = printable["bytes"].map(lambda x: f"{x:.3e}")
    printable["time_us"] = printable["time_us"].map(lambda x: f"{x:.3f}")
    printable["intensity"] = printable["intensity"].map(lambda x: f"{x:.3f}")
    printable["tflops_s"] = printable["tflops_s"].map(lambda x: f"{x:.3f}")
    print(printable.to_string(index=False))


def main() -> None:
    args = parse_args()
    df = load_summary(args.summary_csv)
    normalized = normalize_metrics(df, args.reports_dir)
    points = select_points(normalized, mode=args.mode)
    if points.empty:
        raise SystemExit("No matching points found for the requested mode / context sizes.")
    plot_roofline(points, args.output)
    print_summary(points)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()