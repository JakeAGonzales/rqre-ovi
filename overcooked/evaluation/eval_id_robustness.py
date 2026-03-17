#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from overcooked.agents import load_nqovi, load_qre, load_rqe
from overcooked.lfa import extract_obs
from overcooked_jaxmarl import Overcooked, overcooked_layouts


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class PolicyInfo:
    algorithm: str          # "rqe", "qre", "nqovi"
    path: Path
    label: str              # plot legend label
    tau: Optional[float] = None
    eps: Optional[float] = None
    sort_key: Tuple = field(default_factory=tuple)


@dataclass
class DeltaMetrics:
    mean_team: float
    std_team: float
    mean_ego: float
    std_ego: float
    cvar_ego: float
    mean_team_sparse: float
    std_team_sparse: float
    mean_ego_sparse: float
    std_ego_sparse: float
    cvar_ego_sparse: float
    num_episodes: int


# ── Policy discovery ──────────────────────────────────────────────────

def _parse_rqe_filename(name: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract tau and eps from e.g. 'rqe_overcooked_agent_tau0.5_0.5_eps0.1_0.1.pkl'."""
    m = re.search(r"tau([\d.]+)_([\d.]+)_eps([\d.]+)_([\d.]+)", name)
    if not m:
        return None, None
    tau = float(m.group(1))
    eps = float(m.group(3))
    return tau, eps


def _parse_qre_filename(name: str) -> Optional[float]:
    """Extract eps from e.g. 'qre_overcooked_agent_eps0.3_0.3.pkl'."""
    m = re.search(r"eps([\d.]+)_([\d.]+)", name)
    if not m:
        return None
    return float(m.group(1))


def _parse_nqovi_filename(name: str) -> str:
    """Extract equilibrium selection rule from nqovi filename, e.g. 'welfare'."""
    m = re.search(r"nqovi_overcooked_agent_(\w+)\.pkl", name)
    if m:
        return m.group(1)
    # Fall back to nash-{rule} in the prefix part
    m = re.search(r"nash-(\w+)", name)
    if m:
        return m.group(1)
    return "unknown"


def discover_policies(results_dirs: List[Path]) -> List[PolicyInfo]:
    """Find all .pkl policies and parse their hyperparameters."""
    policies: List[PolicyInfo] = []
    seen_paths: set = set()

    for rdir in results_dirs:
        if not rdir.is_dir():
            print(f"  Warning: {rdir} does not exist, skipping")
            continue
        for pkl in sorted(rdir.glob("*.pkl")):
            if pkl.resolve() in seen_paths:
                continue
            seen_paths.add(pkl.resolve())
            name = pkl.name

            if name.startswith("rqe_") or "_rqe_overcooked_agent_" in name:
                tau, eps = _parse_rqe_filename(name)
                if tau is None:
                    continue
                label = f"RQE (\u03c4={tau}, \u03b5={eps})"
                policies.append(PolicyInfo(
                    algorithm="rqe", path=pkl, label=label,
                    tau=tau, eps=eps, sort_key=("rqe", tau, eps),
                ))

            elif name.startswith("qre_") or "_qre_overcooked_agent_" in name:
                eps = _parse_qre_filename(name)
                if eps is None:
                    continue
                label = f"QRE (\u03b5={eps})"
                policies.append(PolicyInfo(
                    algorithm="qre", path=pkl, label=label,
                    eps=eps, sort_key=("qre", eps),
                ))

            elif name.startswith("nqovi_") or "_nqovi_overcooked_agent" in name:
                selection = _parse_nqovi_filename(name)
                label = f"NQOVI ({selection})"
                policies.append(PolicyInfo(
                    algorithm="nqovi", path=pkl, label=label,
                    sort_key=("nqovi", selection),
                ))

    policies.sort(key=lambda p: p.sort_key)
    return policies


# ── Evaluation ────────────────────────────────────────────────────────

def _compute_cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    cutoff = max(1, int(np.ceil(alpha * sorted_vals.size)))
    return float(np.mean(sorted_vals[:cutoff]))


def _action_towards(pos_self: np.ndarray, pos_target: np.ndarray) -> int:
    """Return the action (0-3) that moves pos_self one step closer to pos_target.

    Breaks ties by preferring the horizontal axis. Falls back to stay (4) when
    the agents are already co-located.
    Actions: up=0, down=1, right=2, left=3, stay=4.
    Positions are (x, y) where x increases rightward and y increases downward.
    """
    dx = int(pos_target[0]) - int(pos_self[0])
    dy = int(pos_target[1]) - int(pos_self[1])
    if dx == 0 and dy == 0:
        return 4  # stay
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 3  # right or left
    return 1 if dy > 0 else 0      # down or up


def _load_agent(policy_path: Path, algorithm: str):
    if algorithm == "nqovi":
        return load_nqovi(str(policy_path))
    if algorithm == "qre":
        return load_qre(str(policy_path))
    if algorithm == "rqe":
        return load_rqe(str(policy_path))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_episode_id_noisy_partner(
    env: Overcooked,
    agent,
    delta: float,
    max_steps: int,
    episode_seed: int,
    perturbation: str = "random",
    defect_action: int = 0,
) -> Tuple[float, float, float, float]:
    """Returns (dense_team, dense_ego, sparse_team, sparse_ego).

    perturbation="random" : partner takes a uniformly random action with prob delta.
    perturbation="defect" : partner defects to `defect_action` with prob delta.
    """
    np.random.seed(episode_seed)
    noise_rng = np.random.default_rng(episode_seed + 17)

    key = jax.random.PRNGKey(episode_seed)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key)

    total_ego = 0.0
    total_team = 0.0
    total_ego_sparse = 0.0
    total_team_sparse = 0.0

    for h in range(max_steps):
        obs = extract_obs(state, env)
        a0, a1 = agent.select_action(obs, h)

        if noise_rng.random() < delta:
            if perturbation == "defect":
                a1 = defect_action
            elif perturbation == "adversary":
                pos0 = np.array(state.agent_pos[0])
                pos1 = np.array(state.agent_pos[1])
                a1 = _action_towards(pos1, pos0)
            else:  # "random"
                a1 = int(noise_rng.integers(env.action_space("agent_1").n))

        actions = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        key, step_key = jax.random.split(key)
        _, state, rewards, dones, infos = env.step(step_key, state, actions)

        sparse_r0 = float(rewards["agent_0"])
        sparse_r1 = float(rewards["agent_1"])
        shaped_r0 = float(infos["shaped_reward"]["agent_0"])
        shaped_r1 = float(infos["shaped_reward"]["agent_1"])

        r0 = sparse_r0 + shaped_r0
        r1 = sparse_r1 + shaped_r1
        total_ego += r0
        total_team += r0 + r1
        total_ego_sparse += sparse_r0
        total_team_sparse += sparse_r0 + sparse_r1

        if bool(dones["__all__"]):
            break

    return total_team, total_ego, total_team_sparse, total_ego_sparse


def evaluate_policy(
    policy: PolicyInfo,
    env: Overcooked,
    deltas: List[float],
    episodes: int,
    max_steps: int,
    base_seed: int,
    cvar_alpha: float,
    perturbation: str = "random",
    defect_action: int = 0,
) -> Tuple[Dict[float, DeltaMetrics], Dict[float, Dict[str, np.ndarray]]]:
    """Returns (aggregated metrics, raw per-episode arrays keyed by delta)."""
    agent = _load_agent(policy.path, policy.algorithm)
    out: Dict[float, DeltaMetrics] = {}
    raw: Dict[float, Dict[str, np.ndarray]] = {}

    for delta in deltas:
        team_rewards: List[float] = []
        ego_rewards: List[float] = []
        team_rewards_sparse: List[float] = []
        ego_rewards_sparse: List[float] = []

        for ep in range(episodes):
            episode_seed = int(base_seed + 10_000 * ep + round(1000.0 * delta))
            team_r, ego_r, team_s, ego_s = _run_episode_id_noisy_partner(
                env=env, agent=agent, delta=delta,
                max_steps=max_steps, episode_seed=episode_seed,
                perturbation=perturbation, defect_action=defect_action,
            )
            team_rewards.append(team_r)
            ego_rewards.append(ego_r)
            team_rewards_sparse.append(team_s)
            ego_rewards_sparse.append(ego_s)

        team_arr = np.asarray(team_rewards, dtype=np.float64)
        ego_arr = np.asarray(ego_rewards, dtype=np.float64)
        team_arr_s = np.asarray(team_rewards_sparse, dtype=np.float64)
        ego_arr_s = np.asarray(ego_rewards_sparse, dtype=np.float64)

        out[delta] = DeltaMetrics(
            mean_team=float(np.mean(team_arr)),
            std_team=float(np.std(team_arr)),
            mean_ego=float(np.mean(ego_arr)),
            std_ego=float(np.std(ego_arr)),
            cvar_ego=_compute_cvar(ego_arr, cvar_alpha),
            mean_team_sparse=float(np.mean(team_arr_s)),
            std_team_sparse=float(np.std(team_arr_s)),
            mean_ego_sparse=float(np.mean(ego_arr_s)),
            std_ego_sparse=float(np.std(ego_arr_s)),
            cvar_ego_sparse=_compute_cvar(ego_arr_s, cvar_alpha),
            num_episodes=int(team_arr.size),
        )
        raw[delta] = {
            "dense_team": team_arr,
            "dense_ego": ego_arr,
            "sparse_team": team_arr_s,
            "sparse_ego": ego_arr_s,
        }

        print(
            f"  [{policy.label}] delta={delta:.2f}  "
            f"team={out[delta].mean_team:7.2f} +/- {out[delta].std_team:5.2f}  "
            f"ego={out[delta].mean_ego:7.2f} +/- {out[delta].std_ego:5.2f}  "
            f"CVaR={out[delta].cvar_ego:7.2f}"
        )

    return out, raw


# ── Plotting ──────────────────────────────────────────────────────────

def _set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.facecolor": "#F8F6F0",
        "figure.facecolor": "white",
    })


def _assign_colors(policies: List[PolicyInfo]) -> Dict[str, str]:
    """Assign colors: RQE=purples, QRE=blues, NQOVI=green."""
    colors: Dict[str, str] = {}

    rqe_policies = [p for p in policies if p.algorithm == "rqe"]
    qre_policies = [p for p in policies if p.algorithm == "qre"]

    purple_cmap = cm.get_cmap("Purples", max(len(rqe_policies) + 3, 5))
    for i, p in enumerate(rqe_policies):
        colors[p.label] = matplotlib.colors.rgb2hex(
            purple_cmap(0.35 + 0.6 * i / max(len(rqe_policies) - 1, 1))
        )

    blue_cmap = cm.get_cmap("Blues", max(len(qre_policies) + 3, 5))
    for i, p in enumerate(qre_policies):
        colors[p.label] = matplotlib.colors.rgb2hex(
            blue_cmap(0.4 + 0.55 * i / max(len(qre_policies) - 1, 1))
        )

    for p in policies:
        if p.algorithm == "nqovi":
            colors[p.label] = "#4CAF50"

    return colors


_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "X", "*", "P"]


def _plot_metric(
    results: Dict[str, Dict[float, DeltaMetrics]],
    policies: List[PolicyInfo],
    colors: Dict[str, str],
    deltas: List[float],
    out_path: Path,
    metric_getter: Callable[[DeltaMetrics], float],
    std_getter: Callable[[DeltaMetrics], float],
    ylabel: str,
    title: str,
    xlabel: str = "Partner noise probability $\\delta$",
) -> None:
    _set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, p in enumerate(policies):
        if p.label not in results:
            continue
        data = results[p.label]
        y = np.array([metric_getter(data[d]) for d in deltas], dtype=np.float64)
        std = np.array([std_getter(data[d]) for d in deltas], dtype=np.float64)
        color = colors[p.label]
        marker = _MARKERS[i % len(_MARKERS)]

        ax.plot(
            deltas, y, marker=marker, color=color,
            linewidth=2.0, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.7,
            label=p.label, zorder=3,
        )
        ax.fill_between(deltas, y - std, y + std, color=color, alpha=0.12, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(deltas)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlim(min(deltas), max(deltas))
    ax.set_ylim(bottom=0.0)
    ax.legend(
        frameon=True, facecolor="white", edgecolor="#D5D8DC",
        bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot: {out_path}")


def _write_csv(
    results: Dict[str, Dict[float, DeltaMetrics]],
    policies: List[PolicyInfo],
    deltas: List[float],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "label", "tau", "eps", "delta",
            "mean_team_reward", "std_team_reward",
            "mean_ego_reward", "std_ego_reward",
            "cvar_ego_reward",
            "mean_team_sparse", "std_team_sparse",
            "mean_ego_sparse", "std_ego_sparse",
            "cvar_ego_sparse",
            "num_episodes",
        ])
        for p in policies:
            if p.label not in results:
                continue
            for d in deltas:
                m = results[p.label][d]
                writer.writerow([
                    p.algorithm, p.label,
                    f"{p.tau}" if p.tau is not None else "",
                    f"{p.eps}" if p.eps is not None else "",
                    f"{d:.6f}",
                    f"{m.mean_team:.8f}", f"{m.std_team:.8f}",
                    f"{m.mean_ego:.8f}", f"{m.std_ego:.8f}",
                    f"{m.cvar_ego:.8f}",
                    f"{m.mean_team_sparse:.8f}", f"{m.std_team_sparse:.8f}",
                    f"{m.mean_ego_sparse:.8f}", f"{m.std_ego_sparse:.8f}",
                    f"{m.cvar_ego_sparse:.8f}",
                    m.num_episodes,
                ])
    print(f"Saved CSV: {out_csv}")


def _write_raw_csv(
    raw_results: Dict[str, Dict[float, Dict[str, np.ndarray]]],
    policies: List[PolicyInfo],
    deltas: List[float],
    out_csv: Path,
) -> None:
    """Save one row per (policy, delta, episode) for full plot reconstruction."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "label", "tau", "eps",
            "delta", "episode",
            "dense_team", "dense_ego",
            "sparse_team", "sparse_ego",
        ])
        for p in policies:
            if p.label not in raw_results:
                continue
            for d in deltas:
                if d not in raw_results[p.label]:
                    continue
                arrs = raw_results[p.label][d]
                n_ep = len(arrs["dense_team"])
                for ep in range(n_ep):
                    writer.writerow([
                        p.algorithm, p.label,
                        f"{p.tau}" if p.tau is not None else "",
                        f"{p.eps}" if p.eps is not None else "",
                        f"{d:.6f}", ep,
                        f"{arrs['dense_team'][ep]:.8f}",
                        f"{arrs['dense_ego'][ep]:.8f}",
                        f"{arrs['sparse_team'][ep]:.8f}",
                        f"{arrs['sparse_ego'][ep]:.8f}",
                    ])
    print(f"Saved raw CSV: {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ID noisy-partner robustness evaluation across all discovered policies"
    )
    parser.add_argument("--batch", type=str, default="all",
                        help="Which batch to evaluate: '1', '2', or 'all'")
    _project_root = str(Path(__file__).resolve().parents[2])
    parser.add_argument("--results-root", type=str,
                        default=str(Path(_project_root) / "results"),
                        help="Root directory containing batch folders")
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episodes per delta per policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deltas", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--cvar-alpha", type=float, default=0.20)
    parser.add_argument("--output-dir", type=str, default="overcooked/evaluation")
    parser.add_argument("--skip-cvar-plot", action="store_true")
    parser.add_argument("--algorithms", type=str, default="",
                        help="Comma-separated filter, e.g. 'rqe,qre'. Empty = all.")
    parser.add_argument("--perturbation", type=str, default="random",
                        choices=["random", "defect", "adversary"],
                        help="Partner perturbation type: 'random' (uniform noise), "
                             "'defect' (fixed action with prob delta), or "
                             "'adversary' (move toward ego agent with prob delta).")
    parser.add_argument("--defect-action", type=int, default=0,
                        help="Action index the partner defects to when --perturbation=defect "
                             "(default 0 = move up).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deltas = sorted([float(x.strip()) for x in args.deltas.split(",") if x.strip()])
    if not deltas:
        raise ValueError("--deltas must contain at least one value")

    if args.layout not in overcooked_layouts:
        raise ValueError(f"Unknown layout: {args.layout}. "
                         f"Options: {list(overcooked_layouts.keys())}")

    # Resolve batch directories
    root = Path(args.results_root)
    if args.batch == "all":
        batch_dirs = sorted(root.glob("batch*/results"))
        if not batch_dirs:
            # Flat layout: root itself contains PKL files directly
            batch_dirs = [root]
    else:
        for b in args.batch.split(","):
            b = b.strip()
            d = root / f"batch{b}" / "results"
            if not d.is_dir():
                raise FileNotFoundError(
                    f"Batch folder not found: {d}\n"
                    f"Available: {[p.parent.name for p in sorted(root.glob('batch*/results'))]}"
                )
        batch_dirs = [root / f"batch{b.strip()}" / "results"
                      for b in args.batch.split(",")]

    algo_filter = set()
    if args.algorithms:
        algo_filter = {a.strip().lower() for a in args.algorithms.split(",")}

    # Discover
    policies = discover_policies(batch_dirs)
    if algo_filter:
        policies = [p for p in policies if p.algorithm in algo_filter]

    if not policies:
        print("No policies found. Check --results-root and --batch.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Overcooked(
        layout=overcooked_layouts[args.layout],
        max_steps=args.horizon,
        cooperative=True,
    )

    # Human-readable description of the perturbation mode
    if args.perturbation == "defect":
        from overcooked_jaxmarl.overcooked import Actions as _Actions
        defect_name = next(
            (a.name for a in _Actions if a.value == args.defect_action),
            str(args.defect_action),
        )
        perturb_desc = f"defect→{defect_name}(a={args.defect_action})"
    elif args.perturbation == "adversary":
        perturb_desc = "adversary (move toward ego)"
    else:
        perturb_desc = "random"

    # Print summary
    print("=" * 72)
    print("ID ROBUSTNESS EVALUATION (NOISY SELF-PARTNER)")
    print("=" * 72)
    print(f"Layout: {args.layout}  |  Horizon: {args.horizon}  |  "
          f"Episodes/delta: {args.episodes}")
    print(f"Perturbation: {perturb_desc}  |  Deltas: {deltas}")
    print(f"Batch dirs: {[str(d) for d in batch_dirs]}")
    print(f"\nDiscovered {len(policies)} policies:")
    for p in policies:
        print(f"  {p.label:30s}  {p.path.name}")
    print("=" * 72)

    # Evaluate
    all_results: Dict[str, Dict[float, DeltaMetrics]] = {}
    all_raw: Dict[str, Dict[float, Dict[str, np.ndarray]]] = {}
    colors = _assign_colors(policies)
    t_start = time.time()

    for idx, policy in enumerate(policies, 1):
        print(f"\n--- [{idx}/{len(policies)}] {policy.label} ---")
        metrics, raw = evaluate_policy(
            policy=policy, env=env, deltas=deltas,
            episodes=args.episodes, max_steps=args.horizon,
            base_seed=args.seed, cvar_alpha=args.cvar_alpha,
            perturbation=args.perturbation, defect_action=args.defect_action,
        )
        all_results[policy.label] = metrics
        all_raw[policy.label] = raw
        elapsed = time.time() - t_start
        remaining = elapsed / idx * (len(policies) - idx)
        print(f"  Elapsed: {elapsed:.0f}s  |  Est. remaining: {remaining:.0f}s")

    batch_tag = f"batch{args.batch}" if args.batch != "all" else "all_batches"
    if args.perturbation == "defect":
        perturb_tag = f"defect{args.defect_action}"
    elif args.perturbation == "adversary":
        perturb_tag = "adversary"
    else:
        perturb_tag = "random"
    tag = f"{batch_tag}_{perturb_tag}"

    # Save CSVs first so data is never lost to a plot crash
    _write_csv(all_results, policies, deltas,
               output_dir / f"id_robustness_metrics_{tag}.csv")
    _write_raw_csv(all_raw, policies, deltas,
                   output_dir / f"id_robustness_raw_episodes_{tag}.csv")

    # Plots
    if args.perturbation == "defect":
        noise_xlabel = f"Partner defect probability $\\delta$ (action={defect_name})"
    elif args.perturbation == "adversary":
        noise_xlabel = "Partner adversary probability $\\delta$ (move toward ego)"
    else:
        noise_xlabel = "Partner noise probability $\\delta$"

    _plot_metric(
        results=all_results, policies=policies, colors=colors,
        deltas=deltas,
        out_path=output_dir / f"id_robustness_team_reward_{tag}.png",
        metric_getter=lambda m: m.mean_team,
        std_getter=lambda m: m.std_team,
        ylabel="Team reward (dense)",
        title=f"ID Robustness [{perturb_tag}]: Team Dense Reward vs Partner Perturbation",
        xlabel=noise_xlabel,
    )

    _plot_metric(
        results=all_results, policies=policies, colors=colors,
        deltas=deltas,
        out_path=output_dir / f"id_robustness_ego_reward_{tag}.png",
        metric_getter=lambda m: m.mean_ego,
        std_getter=lambda m: m.std_ego,
        ylabel="Ego reward (dense)",
        title=f"ID Robustness [{perturb_tag}]: Ego Dense Reward vs Partner Perturbation",
        xlabel=noise_xlabel,
    )

    _plot_metric(
        results=all_results, policies=policies, colors=colors,
        deltas=deltas,
        out_path=output_dir / f"id_robustness_team_sparse_{tag}.png",
        metric_getter=lambda m: m.mean_team_sparse,
        std_getter=lambda m: m.std_team_sparse,
        ylabel="Team reward (sparse)",
        title=f"ID Robustness [{perturb_tag}]: Team Sparse Reward vs Partner Perturbation",
        xlabel=noise_xlabel,
    )

    _plot_metric(
        results=all_results, policies=policies, colors=colors,
        deltas=deltas,
        out_path=output_dir / f"id_robustness_ego_sparse_{tag}.png",
        metric_getter=lambda m: m.mean_ego_sparse,
        std_getter=lambda m: m.std_ego_sparse,
        ylabel="Ego reward (sparse)",
        title=f"ID Robustness [{perturb_tag}]: Ego Sparse Reward vs Partner Perturbation",
        xlabel=noise_xlabel,
    )

    if not args.skip_cvar_plot:
        _plot_metric(
            results=all_results, policies=policies, colors=colors,
            deltas=deltas,
            out_path=output_dir / f"id_robustness_ego_cvar_{tag}.png",
            metric_getter=lambda m: m.cvar_ego,
            std_getter=lambda _m: 0.0,
            ylabel=f"Ego CVaR@{args.cvar_alpha:.2f} (dense)",
            title=f"ID Robustness [{perturb_tag}]: Ego Tail Dense Reward vs Partner Perturbation",
            xlabel=noise_xlabel,
        )
        _plot_metric(
            results=all_results, policies=policies, colors=colors,
            deltas=deltas,
            out_path=output_dir / f"id_robustness_ego_cvar_sparse_{tag}.png",
            metric_getter=lambda m: m.cvar_ego_sparse,
            std_getter=lambda _m: 0.0,
            ylabel=f"Ego CVaR@{args.cvar_alpha:.2f} (sparse)",
            title=f"ID Robustness [{perturb_tag}]: Ego Tail Sparse Reward vs Partner Perturbation",
            xlabel=noise_xlabel,
        )

    total = time.time() - t_start
    print(f"\nDone. Total time: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
