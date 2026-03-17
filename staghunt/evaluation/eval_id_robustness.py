#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# Allow direct script execution: python staghunt/evaluation/eval_id_robustness.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from staghunt.staghunt import NUM_ACTIONS, reset, step
from staghunt.agents.qre import load_agent as qre_load
from staghunt.agents.rqe import load_agent as rqe_load
from staghunt.agents.nqovi import load_agent as nqovi_load
from staghunt.agents.ppo import load_agent as ppo_load


LOADERS = {"qre": qre_load, "rqe": rqe_load, "nqovi": nqovi_load, "linear_ppo": ppo_load}

COLORS = {
    "qre": "#1B9E77",
    "rqe": "#D95F02",
    "nqovi": "#7570B3",
    "linear_ppo": "#E7298A",
}

LABELS = {
    "nqovi": "NQOVI (Nash)",
    "qre": "QRE",
    "rqe": "RQRE",
    "linear_ppo": "Linear PPO",
}

MARKERS = {"nqovi": "o", "qre": "s", "rqe": "D", "linear_ppo": "^"}


@dataclass
class DeltaMetrics:
    mean_team: float
    std_team: float
    mean_ego: float
    std_ego: float
    num_episodes: int


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


def _parse_algorithms(raw: str) -> List[str]:
    allowed = {"nqovi", "qre", "rqe", "linear_ppo"}
    algos = [x.strip().lower() for x in raw.split(",") if x.strip()]
    invalid = [a for a in algos if a not in allowed]
    if invalid:
        raise ValueError(f"Unknown algorithms: {invalid}. Allowed: {sorted(allowed)}")

    out: List[str] = []
    seen = set()
    for a in algos:
        if a not in seen:
            seen.add(a)
            out.append(a)
    if not out:
        raise ValueError("--algorithms must contain at least one algorithm")
    return out


def _resolve_policy_path(results_dir: Path, algorithm: str, explicit_path: str) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Provided --{algorithm}-path does not exist: {p}")
        return p

    path = results_dir / f"{algorithm}_staghunt_agent.pkl"
    if path.exists():
        return path
    return None


def _run_episode_id_noisy_partner(
    agent,
    delta: float,
    max_steps: int,
    episode_seed: int,
) -> tuple[float, float]:
    np.random.seed(episode_seed)
    noise_rng = np.random.default_rng(episode_seed + 17)

    key = jax.random.PRNGKey(episode_seed)
    state = reset(key)

    total_team = 0.0
    total_ego = 0.0

    for h in range(min(agent.H, max_steps)):
        a0, a1 = agent.select_action(state, h)

        # In-distribution noisy partner: perturb only agent_1 with probability delta.
        if noise_rng.random() < delta:
            a1 = int(noise_rng.integers(NUM_ACTIONS))

        actions = jnp.array([a0, a1], dtype=jnp.int32)
        next_state = step(state, actions)

        r0 = float(next_state.last_rewards[0])
        r1 = float(next_state.last_rewards[1])

        total_ego += r0
        total_team += r0 + r1

        state = next_state
        if bool(state.done):
            break

    return total_team, total_ego


def _evaluate_algorithm(
    algorithm: str,
    agent,
    deltas: List[float],
    episodes: int,
    max_steps: int,
    base_seed: int,
) -> Dict[float, DeltaMetrics]:
    out: Dict[float, DeltaMetrics] = {}

    for delta in deltas:
        team_rewards: List[float] = []
        ego_rewards: List[float] = []

        for ep in range(episodes):
            episode_seed = int(base_seed + 10_000 * ep + round(1000.0 * delta))
            team_r, ego_r = _run_episode_id_noisy_partner(
                agent=agent,
                delta=delta,
                max_steps=max_steps,
                episode_seed=episode_seed,
            )
            team_rewards.append(team_r)
            ego_rewards.append(ego_r)

        team_arr = np.asarray(team_rewards, dtype=np.float64)
        ego_arr = np.asarray(ego_rewards, dtype=np.float64)

        out[delta] = DeltaMetrics(
            mean_team=float(np.mean(team_arr)),
            std_team=float(np.std(team_arr)),
            mean_ego=float(np.mean(ego_arr)),
            std_ego=float(np.std(ego_arr)),
            num_episodes=int(team_arr.size),
        )

        print(
            f"[{algorithm.upper()}] delta={delta:.2f} "
            f"team={out[delta].mean_team:.2f}±{out[delta].std_team:.2f} "
            f"ego={out[delta].mean_ego:.2f}±{out[delta].std_ego:.2f}"
        )

    return out


def _plot_metric(
    results: Dict[str, Dict[float, DeltaMetrics]],
    algorithms: List[str],
    deltas: List[float],
    out_path: Path,
    metric_key: str,
    std_key: str,
    ylabel: str,
    title: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for algo in algorithms:
        if algo not in results:
            continue
        y = np.array([getattr(results[algo][d], metric_key) for d in deltas], dtype=np.float64)
        std = np.array([getattr(results[algo][d], std_key) for d in deltas], dtype=np.float64)

        ax.plot(
            deltas,
            y,
            marker=MARKERS[algo],
            color=COLORS[algo],
            linewidth=2.5,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=LABELS[algo],
            zorder=3,
        )
        ax.fill_between(deltas, y - std, y + std, color=COLORS[algo], alpha=0.16, zorder=2)

    ax.set_xlabel("Partner noise probability $\\delta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(deltas)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlim(min(deltas), max(deltas))
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot: {out_path}")


def _write_csv(
    results: Dict[str, Dict[float, DeltaMetrics]],
    algorithms: List[str],
    deltas: List[float],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "delta",
                "mean_team_reward",
                "std_team_reward",
                "mean_agent0_reward",
                "std_agent0_reward",
                "num_episodes",
            ]
        )

        for algo in algorithms:
            if algo not in results:
                continue
            for d in deltas:
                m = results[algo][d]
                writer.writerow(
                    [
                        algo,
                        f"{d:.6f}",
                        f"{m.mean_team:.8f}",
                        f"{m.std_team:.8f}",
                        f"{m.mean_ego:.8f}",
                        f"{m.std_ego:.8f}",
                        m.num_episodes,
                    ]
                )

    print(f"Saved CSV: {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ID robustness eval on Stag Hunt: random partner actions with probability delta"
    )
    parser.add_argument("--algorithms", type=str, default="nqovi,qre,rqe,linear_ppo")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per delta per algorithm")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deltas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    parser.add_argument("--results-dir", type=str, default="staghunt/results")
    parser.add_argument("--nqovi-path", type=str, default="")
    parser.add_argument("--qre-path", type=str, default="")
    parser.add_argument("--rqe-path", type=str, default="")
    parser.add_argument("--linear-ppo-path", type=str, default="")

    parser.add_argument("--output-dir", type=str, default="staghunt/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithms = _parse_algorithms(args.algorithms)

    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    if not deltas:
        raise ValueError("--deltas must contain at least one value")
    deltas = sorted(deltas)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    explicit_paths = {
        "nqovi": args.nqovi_path,
        "qre": args.qre_path,
        "rqe": args.rqe_path,
        "linear_ppo": getattr(args, "linear_ppo_path", ""),
    }

    policy_paths: Dict[str, Path] = {}
    for algo in algorithms:
        p = _resolve_policy_path(results_dir=results_dir, algorithm=algo, explicit_path=explicit_paths.get(algo, ""))
        if p is None:
            print(f"[WARN] Missing policy for {algo.upper()} at {results_dir}/{algo}_staghunt_agent.pkl. Skipping.")
            continue
        policy_paths[algo] = p

    if not policy_paths:
        raise FileNotFoundError("No policy checkpoints found for selected algorithms")

    print("=" * 72)
    print("SELF PLAY ROBUSTNESS EVALUATION (NOISY PARTNER) - STAG HUNT")
    print("=" * 72)
    print(f"Algorithms: {list(policy_paths.keys())}")
    print(f"Horizon: {args.horizon}")
    print("Reward metric: sparse only")
    print(f"Episodes per delta: {args.episodes}")
    print(f"Deltas: {deltas}")
    print(f"Results dir: {results_dir.resolve()}")
    for algo, path in policy_paths.items():
        print(f"Policy for {algo.upper()}: {path}")
    print("=" * 72)

    all_results: Dict[str, Dict[float, DeltaMetrics]] = {}
    for algo in [a for a in algorithms if a in policy_paths]:
        print(f"\n--- Evaluating {algo.upper()} ---")
        agent = LOADERS[algo](str(policy_paths[algo]))
        all_results[algo] = _evaluate_algorithm(
            algorithm=algo,
            agent=agent,
            deltas=deltas,
            episodes=args.episodes,
            max_steps=args.horizon,
            base_seed=args.seed,
        )

    ran_algorithms = [a for a in algorithms if a in all_results]

    _plot_metric(
        results=all_results,
        algorithms=ran_algorithms,
        deltas=deltas,
        out_path=output_dir / "id_noisy_partner_team_reward_vs_delta.png",
        metric_key="mean_team",
        std_key="std_team",
        ylabel="Team reward",
        title="Self Play Robustness: Team Reward vs Partner Noise",
    )

    _plot_metric(
        results=all_results,
        algorithms=ran_algorithms,
        deltas=deltas,
        out_path=output_dir / "id_noisy_partner_agent0_reward_vs_delta.png",
        metric_key="mean_ego",
        std_key="std_ego",
        ylabel="Agent 0 reward",
        title="Self Play Robustness: Agent 0 Reward vs Partner Noise",
    )

    _write_csv(
        results=all_results,
        algorithms=ran_algorithms,
        deltas=deltas,
        out_csv=output_dir / "id_noisy_partner_metrics.csv",
    )


if __name__ == "__main__":
    main()
