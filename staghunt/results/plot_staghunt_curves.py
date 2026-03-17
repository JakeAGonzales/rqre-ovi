#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linestyle": "-",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "legend.frameon": False,
        }
    )


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="valid")


def _infer_label(npz_path: Path) -> str:
    stem = npz_path.stem
    for suffix in ["_staghunt_returns", "_returns"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _display_label(raw_label: str) -> str:
    mapping = {
        "qre": "QRE",
        "rqe": "RQRE",
        "nqovi": "NQOVI",
        "linear_ppo": "Linear PPO",
    }
    return mapping.get(raw_label.strip().lower(), raw_label.upper())


def _load_total_sparse(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "r0" not in data or "r1" not in data:
        raise KeyError(f"Missing r0/r1 arrays in {npz_path}")
    r0 = np.asarray(data["r0"], dtype=np.float64)
    r1 = np.asarray(data["r1"], dtype=np.float64)
    n = min(r0.size, r1.size)
    return r0[:n] + r1[:n]


def _load_metric_if_present(npz_path: Path, key: str) -> np.ndarray | None:
    data = np.load(npz_path)
    if key not in data:
        return None
    return np.asarray(data[key], dtype=np.float64)


def _method_colors(labels: List[str]) -> Dict[str, str]:
    color_map = {
        "QRE": "#1B9E77",
        "RQRE": "#D95F02",
        "NQOVI": "#7570B3",
        "Linear PPO": "#E7298A",
    }
    fallback = ["#66A61E", "#E6AB02", "#A6761D", "#666666"]
    out: Dict[str, str] = {}
    for i, label in enumerate(labels):
        disp = _display_label(label)
        out[label] = color_map.get(disp, fallback[i % len(fallback)])
    return out


def _plot_composite(
    sparse_series: Dict[str, np.ndarray],
    interactions: Dict[str, np.ndarray],
    ss_pct: Dict[str, np.ndarray],
    hh_pct: Dict[str, np.ndarray],
    mixed_pct: Dict[str, np.ndarray],
    window: int,
    out_path: Path,
) -> None:
    _set_plot_style()

    labels = sorted(
        set(sparse_series.keys())
        | set(interactions.keys())
        | set(ss_pct.keys())
        | set(hh_pct.keys())
        | set(mixed_pct.keys())
    )
    colors = _method_colors(labels)

    fig = plt.figure(figsize=(13.5, 12.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.25], hspace=0.35)

    # Top: sparse return
    ax_top = fig.add_subplot(gs[0])
    for label in labels:
        if label not in sparse_series:
            continue
        ax_top.plot(
            _moving_average(sparse_series[label], window),
            linewidth=2.3,
            color=colors[label],
            label=_display_label(label),
        )
    ax_top.set_title("Stag Hunt: Sparse Team Return")
    ax_top.set_xlabel("Episode")
    ax_top.set_ylabel(f"Sparse Return MA{window}")
    ax_top.legend(loc="best")

    # Middle: interactions
    ax_mid = fig.add_subplot(gs[1])
    for label in labels:
        if label not in interactions:
            continue
        ax_mid.plot(
            _moving_average(interactions[label], window),
            linewidth=2.3,
            color=colors[label],
            label=_display_label(label),
        )
    ax_mid.set_title("Stag Hunt: Interactions per Episode")
    ax_mid.set_xlabel("Episode")
    ax_mid.set_ylabel(f"Count MA{window}")
    ax_mid.legend(loc="best")

    # Bottom: three smaller outcome panels
    gs_bottom = gs[2].subgridspec(1, 3, wspace=0.25)
    axes = [fig.add_subplot(gs_bottom[i]) for i in range(3)]
    panels = [("SS (Stag-Stag)", ss_pct), ("HH (Hare-Hare)", hh_pct), ("Mixed", mixed_pct)]

    for idx, (ax, (title, data_map)) in enumerate(zip(axes, panels)):
        for label in labels:
            if label not in data_map:
                continue
            ax.plot(
                _moving_average(data_map[label], window),
                linewidth=2.2,
                color=colors[label],
                label=_display_label(label),
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylim(0, 100)
        if idx == 0:
            ax.set_ylabel(f"Outcome % MA{window}")
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-ready Stag Hunt method comparison plots")
    parser.add_argument(
        "--npz",
        nargs="+",
        default=None,
        help="One or more *_staghunt_returns.npz files. If omitted, auto-discovers in staghunt/results.",
    )
    parser.add_argument("--window", type=int, default=100, help="Moving average window")
    parser.add_argument(
        "--output",
        type=str,
        default="staghunt/results/staghunt_method_comparison.png",
        help="Output figure path for the composite plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.npz:
        paths: List[Path] = [Path(p) for p in args.npz]
    else:
        paths = sorted(Path("staghunt/results").glob("*_staghunt_returns.npz"))
        if not paths:
            raise FileNotFoundError("No *_staghunt_returns.npz files found in staghunt/results")

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"NPZ not found: {p}")

    sparse_series: Dict[str, np.ndarray] = {}
    interactions: Dict[str, np.ndarray] = {}
    ss_pct: Dict[str, np.ndarray] = {}
    hh_pct: Dict[str, np.ndarray] = {}
    mixed_pct: Dict[str, np.ndarray] = {}

    for p in paths:
        label = _infer_label(p)
        sparse_series[label] = _load_total_sparse(p)

        inter = _load_metric_if_present(p, "interactions")
        if inter is not None:
            interactions[label] = inter

        ss = _load_metric_if_present(p, "ss")
        hh = _load_metric_if_present(p, "hh")
        mixed = _load_metric_if_present(p, "mixed")
        if ss is not None and hh is not None and mixed is not None:
            n = min(ss.size, hh.size, mixed.size)
            ss_n = ss[:n]
            hh_n = hh[:n]
            mixed_n = mixed[:n]
            totals = np.maximum(ss_n + hh_n + mixed_n, 1.0)
            ss_pct[label] = 100.0 * (ss_n / totals)
            hh_pct[label] = 100.0 * (hh_n / totals)
            mixed_pct[label] = 100.0 * (mixed_n / totals)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_composite(sparse_series, interactions, ss_pct, hh_pct, mixed_pct, args.window, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
