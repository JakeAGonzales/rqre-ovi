#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from staghunt.staghunt import reset, step
from staghunt.agents.qre import load_agent as qre_load
from staghunt.agents.rqe import load_agent as rqe_load
from staghunt.agents.nqovi import load_agent as nqovi_load
from staghunt.agents.ppo import load_agent as ppo_load


LOADERS = {"qre": qre_load, "rqe": rqe_load, "nqovi": nqovi_load, "linear_ppo": ppo_load}

DISPLAY = {
    "nqovi": "NQOVI",
    "qre": "QRE",
    "rqe": "RQRE",
    "linear_ppo": "Lin. PPO",
}


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "legend.frameon": False,
        }
    )


@dataclass
class CrossplayResult:
    algo_0: str
    algo_1: str
    mean_team: float
    std_team: float
    mean_r0: float
    std_r0: float
    mean_r1: float
    std_r1: float
    num_episodes: int


def _run_crossplay_episode(
    agent_0,
    agent_1,
    max_steps: int,
    episode_seed: int,
) -> Tuple[float, float, float]:
    """Run one episode: agent_0 controls player 0, agent_1 controls player 1."""
    np.random.seed(episode_seed)
    key = jax.random.PRNGKey(episode_seed)
    state = reset(key)

    total_r0 = 0.0
    total_r1 = 0.0
    horizon = min(agent_0.H, agent_1.H, max_steps)

    for h in range(horizon):
        # agent_0 picks action for player 0
        a0_0, a0_1 = agent_0.select_action(state, h)
        # agent_1 picks action for player 1
        a1_0, a1_1 = agent_1.select_action(state, h)

        # Use agent_0's choice for player 0 and agent_1's choice for player 1
        actions = jnp.array([a0_0, a1_1], dtype=jnp.int32)
        next_state = step(state, actions)

        total_r0 += float(next_state.last_rewards[0])
        total_r1 += float(next_state.last_rewards[1])

        state = next_state
        if bool(state.done):
            break

    return total_r0 + total_r1, total_r0, total_r1


def evaluate_pair(
    agent_0,
    agent_1,
    algo_0: str,
    algo_1: str,
    episodes: int,
    max_steps: int,
    base_seed: int,
) -> CrossplayResult:
    team_rewards: List[float] = []
    r0_rewards: List[float] = []
    r1_rewards: List[float] = []

    for ep in range(episodes):
        seed = base_seed + ep * 1000 + hash((algo_0, algo_1)) % 10000
        team, r0, r1 = _run_crossplay_episode(agent_0, agent_1, max_steps, seed)
        team_rewards.append(team)
        r0_rewards.append(r0)
        r1_rewards.append(r1)

    team_arr = np.array(team_rewards)
    r0_arr = np.array(r0_rewards)
    r1_arr = np.array(r1_rewards)

    return CrossplayResult(
        algo_0=algo_0,
        algo_1=algo_1,
        mean_team=float(np.mean(team_arr)),
        std_team=float(np.std(team_arr)),
        mean_r0=float(np.mean(r0_arr)),
        std_r0=float(np.std(r0_arr)),
        mean_r1=float(np.mean(r1_arr)),
        std_r1=float(np.std(r1_arr)),
        num_episodes=episodes,
    )


def _plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    fmt: str = ".1f",
    cmap: str = "YlOrRd",
) -> None:
    _set_plot_style()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im = ax.imshow(matrix, cmap=cmap, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    display_labels = [DISPLAY.get(l, l.upper()) for l in labels]
    ax.set_xticklabels(display_labels)
    ax.set_yticklabels(display_labels)

    ax.set_xlabel("Agent 1 policy")
    ax.set_ylabel("Agent 0 policy")
    ax.set_title(title)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def _write_csv(results: List[CrossplayResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent0_algo", "agent1_algo",
            "mean_team", "std_team",
            "mean_r0", "std_r0",
            "mean_r1", "std_r1",
            "num_episodes",
        ])
        for r in results:
            writer.writerow([
                r.algo_0, r.algo_1,
                f"{r.mean_team:.4f}", f"{r.std_team:.4f}",
                f"{r.mean_r0:.4f}", f"{r.std_r0:.4f}",
                f"{r.mean_r1:.4f}", f"{r.std_r1:.4f}",
                r.num_episodes,
            ])
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-play evaluation: pair algorithms against each other"
    )
    parser.add_argument("--algorithms", type=str, default="nqovi,qre,rqe,linear_ppo")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="staghunt/results")
    parser.add_argument("--nqovi-path", type=str, default="")
    parser.add_argument("--qre-path", type=str, default="")
    parser.add_argument("--rqe-path", type=str, default="")
    parser.add_argument("--linear-ppo-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="staghunt/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algos = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_paths = {
        "nqovi": args.nqovi_path,
        "qre": args.qre_path,
        "rqe": args.rqe_path,
        "linear_ppo": getattr(args, "linear_ppo_path", ""),
    }

    # Load agents
    agents = {}
    for algo in algos:
        if explicit_paths.get(algo):
            path = Path(explicit_paths[algo])
        else:
            path = results_dir / f"{algo}_staghunt_agent.pkl"
        if not path.exists():
            print(f"[WARN] Missing {algo.upper()} at {path}, skipping.")
            continue
        agents[algo] = LOADERS[algo](str(path))
        print(f"Loaded {algo.upper()} from {path}")

    available = [a for a in algos if a in agents]
    if len(available) < 2:
        raise FileNotFoundError("Need at least 2 agent checkpoints for cross-play")

    n = len(available)
    print()
    print("=" * 60)
    print("CROSS-PLAY EVALUATION - STAG HUNT")
    print("=" * 60)
    print(f"Algorithms: {[DISPLAY.get(a, a) for a in available]}")
    print(f"Episodes per pair: {args.episodes}")
    print(f"Horizon: {args.horizon}")
    print("=" * 60)

    # Evaluate all pairs (including self-play on diagonal)
    all_results: List[CrossplayResult] = []
    team_matrix = np.zeros((n, n))
    r0_matrix = np.zeros((n, n))
    r1_matrix = np.zeros((n, n))

    for i, algo_0 in enumerate(available):
        for j, algo_1 in enumerate(available):
            print(f"\n{DISPLAY.get(algo_0, algo_0)} (A0) vs {DISPLAY.get(algo_1, algo_1)} (A1):")
            res = evaluate_pair(
                agent_0=agents[algo_0],
                agent_1=agents[algo_1],
                algo_0=algo_0,
                algo_1=algo_1,
                episodes=args.episodes,
                max_steps=args.horizon,
                base_seed=args.seed,
            )
            all_results.append(res)
            team_matrix[i, j] = res.mean_team
            r0_matrix[i, j] = res.mean_r0
            r1_matrix[i, j] = res.mean_r1
            print(f"  Team: {res.mean_team:.1f} ± {res.std_team:.1f}  "
                  f"A0: {res.mean_r0:.1f} ± {res.std_r0:.1f}  "
                  f"A1: {res.mean_r1:.1f} ± {res.std_r1:.1f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("CROSS-PLAY MATRIX (Team Reward)")
    print("=" * 60)
    header = "A0 \\ A1".ljust(10) + "  ".join(f"{DISPLAY.get(a, a):>8s}" for a in available)
    print(header)
    for i, algo_0 in enumerate(available):
        row = f"{DISPLAY.get(algo_0, algo_0):10s}" + "  ".join(
            f"{team_matrix[i, j]:8.1f}" for j in range(n)
        )
        print(row)

    # Plots
    _plot_heatmap(team_matrix, available,
                  "Cross-Play: Team Reward", output_dir / "crossplay_team_reward.png")
    _plot_heatmap(r0_matrix, available,
                  "Cross-Play: Agent 0 Reward", output_dir / "crossplay_agent0_reward.png")

    _write_csv(all_results, output_dir / "crossplay_metrics.csv")


if __name__ == "__main__":
    main()
