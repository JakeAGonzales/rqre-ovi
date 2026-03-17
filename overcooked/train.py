#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overcooked.agents.nqovi import (
    NQOVIOvercooked,
    save_agent as nqovi_save,
)
from overcooked.agents.qre import QREOvercooked, save_agent as qre_save
from overcooked.agents.rqe import RQEOvercooked, save_agent as rqe_save
from overcooked.lfa import extract_obs, get_feature_dim
from overcooked_jaxmarl import Overcooked, overcooked_layouts
from overcooked_jaxmarl.visualize import animate_rollout

AGENT_REWARD_SCALE = 20.0
TRAIN_SPARSE_COEF = 1.0
TRAIN_SHAPED_COEF = 1.0


def make_agent(args: argparse.Namespace, feature_dim: int):
    if args.algorithm == "qre":
        return (
            QREOvercooked(
                feature_dim=feature_dim,
                horizon=args.horizon,
                lam=args.lam,
                beta=args.beta,
                eps1=args.eps1,
                eps2=args.eps2,
                qre_max_iter=args.qre_max_iter,
                buffer_size=args.buffer_size,
                reward_scale=AGENT_REWARD_SCALE,
            ),
            qre_save,
        )
    if args.algorithm == "rqe":
        return (
            RQEOvercooked(
                feature_dim=feature_dim,
                horizon=args.horizon,
                lam=args.lam,
                beta=args.beta,
                eps1=args.eps1,
                eps2=args.eps2,
                tau1=args.tau1,
                tau2=args.tau2,
                buffer_size=args.buffer_size,
                reward_scale=AGENT_REWARD_SCALE,
            ),
            rqe_save,
        )
    if args.algorithm == "nqovi":
        return (
            NQOVIOvercooked(
                feature_dim=feature_dim,
                horizon=args.horizon,
                lam=args.lam,
                beta=args.beta,
                buffer_size=args.buffer_size,
                reward_scale=AGENT_REWARD_SCALE,
                nash_selection=args.nash_selection,
            ),
            nqovi_save,
        )
    raise ValueError(f"Unknown algorithm: {args.algorithm}")


def _agent_filename(args: argparse.Namespace) -> str:
    if args.algorithm == "qre":
        return f"qre_overcooked_agent_eps{args.eps1}_{args.eps2}.pkl"
    if args.algorithm == "rqe":
        return f"rqe_overcooked_agent_tau{args.tau1}_{args.tau2}_eps{args.eps1}_{args.eps2}.pkl"
    return f"nqovi_overcooked_agent_{args.nash_selection}.pkl"


def plot_training_curves(output_dir: Path, algorithm: str):
    data_path = output_dir / f"{algorithm}_overcooked_returns.npz"
    if not data_path.exists():
        return

    data = np.load(data_path)
    if "sparse_r0" in data and "sparse_r1" in data:
        r0 = data["sparse_r0"]
        r1 = data["sparse_r1"]
    else:
        r0 = data["r0"]
        r1 = data["r1"]
    total = r0 + r1

    window = min(100, len(total))
    if window <= 1:
        return

    def ma(x):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(ma(total), linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Sparse Return (MA, w={window})")
    ax.set_title(f"{algorithm.upper()} Overcooked Sparse Returns")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / f"{algorithm}_overcooked_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {out}")


def generate_gif(
    agent: Any,
    env: Overcooked,
    algorithm: str,
    output_dir: Path,
    num_episodes: int,
    seed: int,
    fps: int,
):
    key = jax.random.PRNGKey(seed + 1000)
    state_seq = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key)
        state_seq.append((state, 0.0))

        for h in range(agent.H):
            obs = extract_obs(state, env)
            a0, a1 = agent.select_action(obs, h)
            actions = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}

            key, step_key = jax.random.split(key)
            _, next_state, rewards, dones, _ = env.step(step_key, state, actions)

            r0 = float(rewards["agent_0"])
            r1 = float(rewards["agent_1"])
            state_seq.append((next_state, r0 + r1))

            state = next_state
            if bool(dones["__all__"]):
                break

    gif_path = output_dir / f"{algorithm}_overcooked_rollout.gif"
    animate_rollout(state_seq, env, filename=str(gif_path), fps=fps)
    print(f"Saved rollout GIF to {gif_path}")


def train(args: argparse.Namespace):
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    layout = overcooked_layouts[args.layout]
    env = Overcooked(layout=layout, max_steps=args.horizon, cooperative=args.cooperative)

    key, init_key = jax.random.split(key)
    _, init_state = env.reset(init_key)
    feature_dim = get_feature_dim(env)

    agent, save_fn = make_agent(args, feature_dim)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_r0, ep_r1, ep_lengths = [], [], []
    ep_sparse_r0, ep_sparse_r1 = [], []
    ep_onion_in_pot, ep_soup_pickup, ep_delivery_events, ep_missed_ready = [], [], [], []
    deliveries = 0
    t_start = time.time()

    for k in range(1, args.episodes + 1):
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key)

        total_r0, total_r1 = 0.0, 0.0
        total_sparse_r0, total_sparse_r1 = 0.0, 0.0
        total_onion_in_pot = 0.0
        total_soup_pickup = 0.0
        total_delivery_events = 0.0
        total_missed_ready = 0.0
        any_delivery = False
        steps = 0

        for h in range(agent.H):
            obs = extract_obs(state, env)
            a0, a1 = agent.select_action(obs, h)
            actions = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}

            key, step_key = jax.random.split(key)
            _, next_state, rewards, dones, infos = env.step(step_key, state, actions)

            sparse_r0 = float(rewards["agent_0"])
            sparse_r1 = float(rewards["agent_1"])
            shaped_r0 = float(infos["shaped_reward"]["agent_0"])
            shaped_r1 = float(infos["shaped_reward"]["agent_1"])

            r0 = TRAIN_SPARSE_COEF * sparse_r0 + TRAIN_SHAPED_COEF * shaped_r0
            r1 = TRAIN_SPARSE_COEF * sparse_r1 + TRAIN_SHAPED_COEF * shaped_r1
            done = bool(dones["__all__"])
            next_obs = extract_obs(next_state, env)

            agent.store_transition(h, obs, a0, a1, next_obs, r0, r1, done)

            total_r0 += r0
            total_r1 += r1
            total_sparse_r0 += sparse_r0
            total_sparse_r1 += sparse_r1

            debug_events = infos.get("debug_events", None)
            if debug_events is not None:
                total_onion_in_pot += float(debug_events["onion_in_pot"])
                total_soup_pickup += float(debug_events["soup_pickup"])
                total_delivery_events += float(debug_events["delivery"])
                total_missed_ready += float(debug_events["missed_ready"])

            if sparse_r0 >= args.delivery_reward or sparse_r1 >= args.delivery_reward:
                any_delivery = True

            state = next_state
            steps += 1
            if done:
                break

        if any_delivery:
            deliveries += 1

        ep_r0.append(total_r0)
        ep_r1.append(total_r1)
        ep_lengths.append(steps)
        ep_sparse_r0.append(total_sparse_r0)
        ep_sparse_r1.append(total_sparse_r1)
        ep_onion_in_pot.append(total_onion_in_pot)
        ep_soup_pickup.append(total_soup_pickup)
        ep_delivery_events.append(total_delivery_events)
        ep_missed_ready.append(total_missed_ready)

        if args.update_freq > 0 and k % args.update_freq == 0:
            agent.update()

        if k % 50 == 0:
            recent = min(50, len(ep_r0))
            mean_r0 = float(np.mean(ep_r0[-recent:]))
            mean_r1 = float(np.mean(ep_r1[-recent:]))
            mean_total = mean_r0 + mean_r1
            mean_sparse_total = float(np.mean(np.asarray(ep_sparse_r0[-recent:]) + np.asarray(ep_sparse_r1[-recent:])))
            mean_len = float(np.mean(ep_lengths[-recent:]))
            mean_onion_in_pot = float(np.mean(ep_onion_in_pot[-recent:]))
            mean_soup_pickup = float(np.mean(ep_soup_pickup[-recent:]))
            mean_delivery_events = float(np.mean(ep_delivery_events[-recent:]))
            mean_missed_ready = float(np.mean(ep_missed_ready[-recent:]))
            delivery_rate = 100.0 * deliveries / float(k)
            elapsed = time.time() - t_start
            eps_per_s = k / max(elapsed, 1e-6)
            print(
                f"Ep {k:5d} | Ret_tot: {mean_total:7.2f} | Ret_0: {mean_r0:7.2f} | "
                f"Ret_1: {mean_r1:7.2f} | Sparse_tot: {mean_sparse_total:7.2f} | "
                f"Len: {mean_len:6.1f} | "
                f"Delivery: {delivery_rate:5.1f}% | "
                f"Onion->Pot: {mean_onion_in_pot:4.2f} | "
                f"SoupPick: {mean_soup_pickup:4.2f} | "
                f"DelivEv: {mean_delivery_events:4.2f} | "
                f"MissReady: {mean_missed_ready:4.2f} | "
                f"{eps_per_s:.1f} ep/s"
            )

    np.savez(
        output_dir / f"{args.algorithm}_overcooked_returns.npz",
        r0=np.asarray(ep_r0, dtype=float),
        r1=np.asarray(ep_r1, dtype=float),
        sparse_r0=np.asarray(ep_sparse_r0, dtype=float),
        sparse_r1=np.asarray(ep_sparse_r1, dtype=float),
        lengths=np.asarray(ep_lengths, dtype=float),
    )

    agent_path = output_dir / _agent_filename(args)
    save_fn(agent, str(agent_path))

    elapsed = time.time() - t_start
    print(
        f"\n{args.algorithm.upper()} training done: {args.episodes} episodes in "
        f"{elapsed:.1f}s ({args.episodes / max(elapsed, 1e-6):.1f} ep/s)"
    )

    plot_training_curves(output_dir, args.algorithm)

    if not args.no_gif:
        generate_gif(
            agent=agent,
            env=env,
            algorithm=args.algorithm,
            output_dir=output_dir,
            num_episodes=args.gif_episodes,
            seed=args.seed,
            fps=args.gif_fps,
        )


def main():
    parser = argparse.ArgumentParser(description="Train LFA agents on Overcooked (JaxMARL)")
    parser.add_argument("--algorithm", type=str, choices=["qre", "rqe", "nqovi"], required=True)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--update-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="overcooked/results")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-episodes", type=int, default=1)
    parser.add_argument("--gif-fps", type=int, default=5)

    # QRE / RQE params
    parser.add_argument("--eps1", type=float, default=0.1)
    parser.add_argument("--eps2", type=float, default=0.1)
    parser.add_argument("--qre-max-iter", type=int, default=100)

    # RQE params
    parser.add_argument("--tau1", type=float, default=0.1)
    parser.add_argument("--tau2", type=float, default=0.1)

    # NQOVI params
    parser.add_argument("--nash-selection", type=str, choices=["welfare", "maximin", "random"], default="random")

    # Cooperative mode
    parser.add_argument("--cooperative", action="store_true", default=True)
    parser.add_argument("--no-cooperative", dest="cooperative", action="store_false")

    # Diagnostics
    parser.add_argument("--delivery-reward", type=float, default=20.0)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
