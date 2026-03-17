#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from overcooked.agents import load_nqovi, load_qre, load_rqe
from overcooked.lfa import extract_obs
from overcooked_jaxmarl import Overcooked, overcooked_layouts


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class PolicyInfo:
    algorithm: str
    path: Path
    label: str
    tau: Optional[float] = None
    eps: Optional[float] = None
    sort_key: Tuple = field(default_factory=tuple)


@dataclass
class PairMetrics:
    mean_team: float
    std_team: float
    mean_ego: float
    std_ego: float
    mean_team_sparse: float
    std_team_sparse: float
    mean_ego_sparse: float
    std_ego_sparse: float
    num_episodes: int


# ── Policy discovery (shared with eval_id_robustness) ────────────────

def _parse_rqe_filename(name: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(r"tau([\d.]+)_([\d.]+)_eps([\d.]+)_([\d.]+)", name)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(3))


def _parse_qre_filename(name: str) -> Optional[float]:
    m = re.search(r"eps([\d.]+)_([\d.]+)", name)
    if not m:
        return None
    return float(m.group(1))


def _parse_nqovi_filename(name: str) -> str:
    """Extract equilibrium selection rule from nqovi filename, e.g. 'welfare'."""
    m = re.search(r"nqovi_overcooked_agent_(\w+)\.pkl", name)
    if m:
        return m.group(1)
    m = re.search(r"nash-(\w+)", name)
    if m:
        return m.group(1)
    return "unknown"



def discover_policies(results_dirs: List[Path]) -> List[PolicyInfo]:
    policies: List[PolicyInfo] = []
    seen: set = set()

    for rdir in results_dirs:
        if not rdir.is_dir():
            print(f"  Warning: {rdir} does not exist, skipping")
            continue
        for pkl in sorted(rdir.glob("*.pkl")):
            if pkl.resolve() in seen:
                continue
            seen.add(pkl.resolve())
            name = pkl.name

            if name.startswith("rqe_") or "_rqe_overcooked_agent_" in name:
                tau, eps = _parse_rqe_filename(name)
                if tau is None:
                    continue
                policies.append(PolicyInfo(
                    algorithm="rqe", path=pkl,
                    label=f"RQE (\u03c4={tau}, \u03b5={eps})",
                    tau=tau, eps=eps, sort_key=("rqe", tau, eps),
                ))
            elif name.startswith("qre_") or "_qre_overcooked_agent_" in name:
                eps = _parse_qre_filename(name)
                if eps is None:
                    continue
                policies.append(PolicyInfo(
                    algorithm="qre", path=pkl,
                    label=f"QRE (\u03b5={eps})",
                    eps=eps, sort_key=("qre", eps),
                ))
            elif name.startswith("nqovi_") or "_nqovi_overcooked_agent" in name:
                selection = _parse_nqovi_filename(name)
                label = f"NQOVI ({selection})"
                policies.append(PolicyInfo(
                    algorithm="nqovi", path=pkl,
                    label=label,
                    sort_key=("nqovi", selection),
                ))

    policies.sort(key=lambda p: p.sort_key)
    return policies


# ── Agent loading ─────────────────────────────────────────────────────

def _load_agent(policy_path: Path, algorithm: str):
    if algorithm == "nqovi":
        return load_nqovi(str(policy_path))
    if algorithm == "qre":
        return load_qre(str(policy_path))
    if algorithm == "rqe":
        return load_rqe(str(policy_path))
    raise ValueError(f"Unknown algorithm: {algorithm}")


# ── Episode runner ────────────────────────────────────────────────────

def _run_crossplay_episode(
    env: Overcooked,
    agent_row,
    agent_col,
    max_steps: int,
    episode_seed: int,
    self_play: bool,
) -> Tuple[float, float, float, float]:
    """Run one cross-play episode. Returns (dense_team, dense_ego, sparse_team, sparse_ego).

    self_play=True: both actions from agent_row's single select_action call.
    self_play=False: a0 from agent_row, a1 from agent_col.
    """
    np.random.seed(episode_seed)
    key = jax.random.PRNGKey(episode_seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    total_ego = 0.0
    total_team = 0.0
    total_ego_sparse = 0.0
    total_team_sparse = 0.0

    for h in range(max_steps):
        obs = extract_obs(state, env)

        if self_play:
            a0, a1 = agent_row.select_action(obs, h)
        else:
            a0, _ = agent_row.select_action(obs, h)
            _, a1 = agent_col.select_action(obs, h)

        actions = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        key, step_key = jax.random.split(key)
        _, state, rewards, dones, infos = env.step(step_key, state, actions)

        sparse_r0 = float(rewards["agent_0"])
        sparse_r1 = float(rewards["agent_1"])
        r0 = sparse_r0 + float(infos["shaped_reward"]["agent_0"])
        r1 = sparse_r1 + float(infos["shaped_reward"]["agent_1"])
        total_ego += r0
        total_team += r0 + r1
        total_ego_sparse += sparse_r0
        total_team_sparse += sparse_r0 + sparse_r1

        if bool(dones["__all__"]):
            break

    return total_team, total_ego, total_team_sparse, total_ego_sparse


def evaluate_pair(
    env: Overcooked,
    agent_row,
    agent_col,
    episodes: int,
    max_steps: int,
    base_seed: int,
    self_play: bool,
) -> Tuple[PairMetrics, Dict[str, np.ndarray]]:
    """Returns (aggregated metrics, raw per-episode arrays)."""
    team_rewards: List[float] = []
    ego_rewards: List[float] = []
    team_rewards_sparse: List[float] = []
    ego_rewards_sparse: List[float] = []

    for ep in range(episodes):
        seed = base_seed + ep * 10_000
        team_r, ego_r, team_s, ego_s = _run_crossplay_episode(
            env, agent_row, agent_col, max_steps, seed, self_play,
        )
        team_rewards.append(team_r)
        ego_rewards.append(ego_r)
        team_rewards_sparse.append(team_s)
        ego_rewards_sparse.append(ego_s)

    team_arr = np.asarray(team_rewards, dtype=np.float64)
    ego_arr = np.asarray(ego_rewards, dtype=np.float64)
    team_arr_s = np.asarray(team_rewards_sparse, dtype=np.float64)
    ego_arr_s = np.asarray(ego_rewards_sparse, dtype=np.float64)

    metrics = PairMetrics(
        mean_team=float(np.mean(team_arr)),
        std_team=float(np.std(team_arr)),
        mean_ego=float(np.mean(ego_arr)),
        std_ego=float(np.std(ego_arr)),
        mean_team_sparse=float(np.mean(team_arr_s)),
        std_team_sparse=float(np.std(team_arr_s)),
        mean_ego_sparse=float(np.mean(ego_arr_s)),
        std_ego_sparse=float(np.std(ego_arr_s)),
        num_episodes=int(team_arr.size),
    )
    raw = {
        "dense_team": team_arr,
        "dense_ego": ego_arr,
        "sparse_team": team_arr_s,
        "sparse_ego": ego_arr_s,
    }
    return metrics, raw


# ── Plotting ──────────────────────────────────────────────────────────

def _set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.grid": False,
        "figure.facecolor": "white",
    })


def _short_label(p: PolicyInfo) -> str:
    """Compact multi-line label for heatmap axes."""
    if p.algorithm == "nqovi":
        selection = _parse_nqovi_filename(p.path.name)
        return f"NQOVI\n({selection})"
    if p.algorithm == "qre":
        return f"QRE\n\u03b5={p.eps}"
    if p.algorithm == "rqe":
        return f"RQE\n\u03c4={p.tau}\n\u03b5={p.eps}"
    return p.label


def _algo_color_bar(policies: List[PolicyInfo]) -> List[str]:
    """Color for axis tick labels by algorithm family."""
    cmap = {"rqe": "#7E57C2", "qre": "#1976D2", "nqovi": "#4CAF50"}
    return [cmap.get(p.algorithm, "black") for p in policies]


def plot_heatmap(
    matrix: np.ndarray,
    row_policies: List[PolicyInfo],
    col_policies: List[PolicyInfo],
    out_path: Path,
    metric_name: str = "Team Reward",
    title: str = "Cross-Play: Team Reward",
    fmt_str: str = "{:.0f}",
) -> None:
    _set_plot_style()

    n_rows, n_cols = matrix.shape
    row_labels = [_short_label(p) for p in row_policies]
    col_labels = [_short_label(p) for p in col_policies]
    row_colors = _algo_color_bar(row_policies)
    col_colors = _algo_color_bar(col_policies)

    # Self-play cells: same policy path
    col_path_to_j = {p.path: j for j, p in enumerate(col_policies)}
    self_play_cells = set()
    for i, rp in enumerate(row_policies):
        if rp.path in col_path_to_j:
            self_play_cells.add((i, col_path_to_j[rp.path]))

    cell_w = max(0.7, min(1.5, 8.0 / n_cols))
    cell_h = max(0.5, min(1.2, 10.0 / n_rows))
    fig_w = n_cols * cell_w + 3.0
    fig_h = n_rows * cell_h + 2.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Color scale (NaN-robust)
    vals = matrix.flatten()
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        finite = np.array([0.0, 1.0])
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmax - vmin < 1.0:
        mid = (vmin + vmax) / 2
        vmin, vmax = mid - 0.5, mid + 0.5
    vmid = float(np.clip(np.median(finite), vmin + 1e-3 * (vmax - vmin), vmax - 1e-3 * (vmax - vmin)))

    cmap_colors = ["#D55E00", "#E8915A", "#F5E6D3", "#C6DAE8", "#5A9BC8", "#0072B2"]
    cmap = LinearSegmentedColormap.from_list("crossplay", cmap_colors)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=max(7, 11 - max(n_rows, n_cols) // 5), color="gray")
                continue
            rel = (val - vmin) / max(vmax - vmin, 1e-6)
            text_color = "white" if rel < 0.15 or rel > 0.85 else "black"
            fontweight = "bold" if (i, j) in self_play_cells else "normal"
            ax.text(j, i, fmt_str.format(val), ha="center", va="center",
                    fontsize=max(7, 11 - max(n_rows, n_cols) // 5), fontweight=fontweight,
                    color=text_color)

    # Highlight self-play cells
    for (i, j) in self_play_cells:
        ax.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.5,
        ))

    # Axes
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, ha="center")
    ax.set_yticklabels(row_labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    for tick_label, color in zip(ax.get_xticklabels(), col_colors):
        tick_label.set_color(color)
    for tick_label, color in zip(ax.get_yticklabels(), row_colors):
        tick_label.set_color(color)

    ax.set_xlabel("Agent 1 (Partner)", fontsize=12, labelpad=8)
    ax.set_ylabel("Agent 0 (Ego)", fontsize=12)
    ax.set_title(title, fontweight="bold", pad=14, y=1.08 + 0.01 * n_rows)

    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label(metric_name, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved heatmap: {out_path}")


def plot_self_vs_cross_bar(
    matrix: np.ndarray,
    row_policies: List[PolicyInfo],
    col_policies: List[PolicyInfo],
    out_path: Path,
) -> None:
    """Bar chart: self-play reward vs mean cross-play reward per ego policy."""
    _set_plot_style()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.facecolor"] = "#F8F6F0"

    n_rows = len(row_policies)
    col_path_to_j = {p.path: j for j, p in enumerate(col_policies)}

    self_play = np.full(n_rows, np.nan)
    for i, rp in enumerate(row_policies):
        if rp.path in col_path_to_j:
            self_play[i] = matrix[i, col_path_to_j[rp.path]]

    cross_play_mean = np.array([
        np.mean([matrix[i, j] for j in range(len(col_policies))
                 if row_policies[i].path != col_policies[j].path])
        if any(row_policies[i].path != col_policies[j].path for j in range(len(col_policies)))
        else np.nan
        for i in range(n_rows)
    ])

    x = np.arange(n_rows)
    width = 0.35
    tick_colors_map = {"rqe": "#7E57C2", "qre": "#1976D2", "nqovi": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(max(8, n_rows * 0.9), 5.5))
    valid_self = ~np.isnan(self_play)
    if valid_self.any():
        ax.bar(x[valid_self] - width / 2, self_play[valid_self], width, label="Self-play",
               color="#7E57C2", alpha=0.85, edgecolor="white", linewidth=0.5)
    valid_cross = ~np.isnan(cross_play_mean)
    if valid_cross.any():
        ax.bar(x[valid_cross] + width / 2, cross_play_mean[valid_cross], width,
               label="Cross-play (mean)", color="#1976D2", alpha=0.85, edgecolor="white", linewidth=0.5)

    labels = [p.label for p in row_policies]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    for tick_label, p in zip(ax.get_xticklabels(), row_policies):
        tick_label.set_color(tick_colors_map.get(p.algorithm, "black"))

    ax.set_ylabel("Team Reward")
    ax.set_title("Self-Play vs Cross-Play Performance", fontweight="bold")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D5D8DC")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved bar chart: {out_path}")


# ── CSV ───────────────────────────────────────────────────────────────

def _write_csv(
    matrix_team: np.ndarray,
    matrix_ego: np.ndarray,
    matrix_team_sparse: np.ndarray,
    matrix_ego_sparse: np.ndarray,
    row_policies: List[PolicyInfo],
    col_policies: List[PolicyInfo],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent0_label", "agent0_algorithm", "agent0_tau", "agent0_eps",
            "agent1_label", "agent1_algorithm", "agent1_tau", "agent1_eps",
            "self_play",
            "mean_team_reward", "mean_ego_reward",
            "mean_team_sparse", "mean_ego_sparse",
        ])
        for i, pi in enumerate(row_policies):
            for j, pj in enumerate(col_policies):
                self_play = pi.path == pj.path
                writer.writerow([
                    pi.label, pi.algorithm,
                    f"{pi.tau}" if pi.tau is not None else "",
                    f"{pi.eps}" if pi.eps is not None else "",
                    pj.label, pj.algorithm,
                    f"{pj.tau}" if pj.tau is not None else "",
                    f"{pj.eps}" if pj.eps is not None else "",
                    "true" if self_play else "false",
                    f"{matrix_team[i, j]:.4f}",
                    f"{matrix_ego[i, j]:.4f}",
                    f"{matrix_team_sparse[i, j]:.4f}",
                    f"{matrix_ego_sparse[i, j]:.4f}",
                ])
    print(f"Saved CSV: {out_csv}")


def _write_raw_csv(
    raw_rows: List[Dict[str, Any]],
    out_csv: Path,
) -> None:
    """Save one row per (agent0, agent1, episode) for full plot reconstruction."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent0_label", "agent0_algorithm", "agent0_tau", "agent0_eps",
            "agent1_label", "agent1_algorithm", "agent1_tau", "agent1_eps",
            "self_play", "episode",
            "dense_team", "dense_ego",
            "sparse_team", "sparse_ego",
        ])
        for row in raw_rows:
            writer.writerow([
                row["agent0_label"], row["agent0_algorithm"],
                row["agent0_tau"], row["agent0_eps"],
                row["agent1_label"], row["agent1_algorithm"],
                row["agent1_tau"], row["agent1_eps"],
                "true" if row["self_play"] else "false",
                row["episode"],
                f"{row['dense_team']:.8f}",
                f"{row['dense_ego']:.8f}",
                f"{row['sparse_team']:.8f}",
                f"{row['sparse_ego']:.8f}",
            ])
    print(f"Saved raw CSV: {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    _project_root = str(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Cross-play heatmap evaluation")
    parser.add_argument("--batch", type=str, default="all",
                        help="Which batch: '1', '2', '1,2', or 'all'")
    parser.add_argument("--results-root", type=str,
                        default=str(Path(_project_root) / "results"))
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="overcooked/evaluation")
    parser.add_argument("--algorithms", type=str, default="",
                        help="Comma-separated filter, e.g. 'rqe,qre'. Empty = all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.layout not in overcooked_layouts:
        raise ValueError(f"Unknown layout: {args.layout}. "
                         f"Options: {list(overcooked_layouts.keys())}")

    root = Path(args.results_root)
    if args.batch == "all":
        batch_dirs = sorted(root.glob("batch*/results"))
        if not batch_dirs:
            # Flat layout: root itself contains PKL files directly
            batch_dirs = [root]
    else:
        batch_dirs = []
        for b in args.batch.split(","):
            b = b.strip()
            d = root / f"batch{b}" / "results"
            if not d.is_dir():
                raise FileNotFoundError(
                    f"Batch folder not found: {d}\n"
                    f"Available: {[p.parent.name for p in sorted(root.glob('batch*/results'))]}"
                )
            batch_dirs.append(d)

    algo_filter = set()
    if args.algorithms:
        algo_filter = {a.strip().lower() for a in args.algorithms.split(",")}

    policies = discover_policies(batch_dirs)
    if algo_filter:
        policies = [p for p in policies if p.algorithm in algo_filter]

    if len(policies) < 2:
        print(f"Need at least 2 policies for cross-play, found {len(policies)}.")
        sys.exit(1)

    # Full N×N square: every policy plays as both ego and partner
    row_policies = policies
    col_policies = policies
    n_rows = len(row_policies)
    n_cols = len(col_policies)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Overcooked(
        layout=overcooked_layouts[args.layout],
        max_steps=args.horizon,
        cooperative=True,
    )

    print("=" * 72)
    print("CROSS-PLAY EVALUATION")
    print("=" * 72)
    print(f"Layout: {args.layout}  |  Horizon: {args.horizon}  |  "
          f"Episodes/pair: {args.episodes}")
    print(f"Batch dirs: {[str(d) for d in batch_dirs]}")
    print(f"\nPolicies ({n_rows}):")
    for p in row_policies:
        print(f"  {p.label:30s}  {p.path.name}")
    print(f"\n{n_rows * n_cols} pairs to evaluate ({n_rows}×{n_cols} full square)")
    print("=" * 72)

    # Load all agents once; row and col share the same list
    all_agents = []
    for p in row_policies:
        print(f"Loading {p.label}...")
        all_agents.append(_load_agent(p.path, p.algorithm))
    row_agents = all_agents
    col_agents = all_agents

    # Evaluate all pairs
    matrix_team = np.zeros((n_rows, n_cols), dtype=np.float64)
    matrix_ego = np.zeros((n_rows, n_cols), dtype=np.float64)
    matrix_team_sparse = np.zeros((n_rows, n_cols), dtype=np.float64)
    matrix_ego_sparse = np.zeros((n_rows, n_cols), dtype=np.float64)
    raw_rows: List[Dict[str, Any]] = []

    total_pairs = n_rows * n_cols
    t_start = time.time()
    pair_count = 0

    for i in range(n_rows):
        for j in range(n_cols):
            pair_count += 1
            self_play = (row_policies[i].path == col_policies[j].path)
            tag = "self-play" if self_play else "cross-play"
            print(f"\n[{pair_count}/{total_pairs}] {row_policies[i].label} vs "
                  f"{col_policies[j].label} ({tag})")

            metrics, raw = evaluate_pair(
                env=env,
                agent_row=row_agents[i],
                agent_col=col_agents[j],
                episodes=args.episodes,
                max_steps=args.horizon,
                base_seed=args.seed,
                self_play=self_play,
            )
            matrix_team[i, j] = metrics.mean_team
            matrix_ego[i, j] = metrics.mean_ego
            matrix_team_sparse[i, j] = metrics.mean_team_sparse
            matrix_ego_sparse[i, j] = metrics.mean_ego_sparse

            pi, pj = row_policies[i], col_policies[j]
            for ep_idx in range(len(raw["dense_team"])):
                raw_rows.append({
                    "agent0_label": pi.label,
                    "agent0_algorithm": pi.algorithm,
                    "agent0_tau": f"{pi.tau}" if pi.tau is not None else "",
                    "agent0_eps": f"{pi.eps}" if pi.eps is not None else "",
                    "agent1_label": pj.label,
                    "agent1_algorithm": pj.algorithm,
                    "agent1_tau": f"{pj.tau}" if pj.tau is not None else "",
                    "agent1_eps": f"{pj.eps}" if pj.eps is not None else "",
                    "self_play": self_play,
                    "episode": ep_idx,
                    "dense_team": raw["dense_team"][ep_idx],
                    "dense_ego": raw["dense_ego"][ep_idx],
                    "sparse_team": raw["sparse_team"][ep_idx],
                    "sparse_ego": raw["sparse_ego"][ep_idx],
                })

            print(f"  team={metrics.mean_team:7.2f} +/- {metrics.std_team:5.2f}  "
                  f"ego={metrics.mean_ego:7.2f} +/- {metrics.std_ego:5.2f}")

            elapsed = time.time() - t_start
            remaining = elapsed / pair_count * (total_pairs - pair_count)
            print(f"  Elapsed: {elapsed:.0f}s  |  Est. remaining: {remaining:.0f}s")

    # Summary table
    print("\n" + "=" * 72)
    print("TEAM REWARD MATRIX (rows=Agent 0, cols=Agent 1)")
    print("=" * 72)
    header = f"{'':>30s}" + "".join(f"{p.label:>14s}" for p in col_policies)
    print(header)
    for i in range(n_rows):
        row_str = f"{row_policies[i].label:>30s}"
        for j in range(n_cols):
            marker = " *" if row_policies[i].path == col_policies[j].path else "  "
            row_str += f"{matrix_team[i, j]:12.1f}{marker}"
        print(row_str)
    print("(* = self-play)")

    batch_tag = f"batch{args.batch}" if args.batch != "all" else "all_batches"

    # Save CSVs first so data is never lost to a plot crash
    _write_csv(matrix_team, matrix_ego, matrix_team_sparse, matrix_ego_sparse,
               row_policies, col_policies, output_dir / f"crossplay_metrics_{batch_tag}.csv")
    _write_raw_csv(raw_rows,
                   output_dir / f"crossplay_raw_episodes_{batch_tag}.csv")

    # Plots
    plot_heatmap(
        matrix_team, row_policies, col_policies,
        out_path=output_dir / f"crossplay_team_reward_{batch_tag}.png",
        metric_name="Team Reward (dense)",
        title="Cross-Play: Team Dense Reward",
    )

    plot_heatmap(
        matrix_ego, row_policies, col_policies,
        out_path=output_dir / f"crossplay_ego_reward_{batch_tag}.png",
        metric_name="Ego Reward (dense)",
        title="Cross-Play: Ego Dense Reward",
    )

    plot_heatmap(
        matrix_team_sparse, row_policies, col_policies,
        out_path=output_dir / f"crossplay_team_sparse_{batch_tag}.png",
        metric_name="Team Reward (sparse)",
        title="Cross-Play: Team Sparse Reward",
    )

    plot_heatmap(
        matrix_ego_sparse, row_policies, col_policies,
        out_path=output_dir / f"crossplay_ego_sparse_{batch_tag}.png",
        metric_name="Ego Reward (sparse)",
        title="Cross-Play: Ego Sparse Reward",
    )

    # Degradation: normalize each row by that agent's own self-play (diagonal) value.
    diag_self_play_sparse = np.diag(matrix_team_sparse)
    safe_diag = np.where(np.abs(diag_self_play_sparse) < 1e-6, 1e-6, diag_self_play_sparse)
    matrix_pct_sparse = matrix_team_sparse / safe_diag[:, None] * 100.0

    plot_heatmap(
        matrix_pct_sparse, row_policies, col_policies,
        out_path=output_dir / f"crossplay_degradation_sparse_{batch_tag}.png",
        metric_name="% of Self-Play (sparse)",
        title="Cross-Play: Sparse Reward as % of Self-Play",
        fmt_str="{:.0f}%",
    )

    plot_self_vs_cross_bar(
        matrix_team, row_policies, col_policies,
        out_path=output_dir / f"crossplay_self_vs_cross_{batch_tag}.png",
    )

    total = time.time() - t_start
    print(f"\nDone. Total time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
