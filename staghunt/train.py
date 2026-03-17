import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staghunt.staghunt import reset, step
from staghunt.agents.qre import QREStagHunt, save_agent as qre_save
from staghunt.agents.rqe import RQEStagHunt, save_agent as rqe_save
from staghunt.agents.nqovi import (
    NQOVIStagHunt,
    save_agent as nqovi_save,
)
from staghunt.vis import animate_rollout


def make_agent(args):
    if args.algorithm == "qre":
        return QREStagHunt(
            horizon=args.horizon,
            lam=args.lam, beta=args.beta,
            eps1=args.eps1, eps2=args.eps2,
            qre_max_iter=args.qre_max_iter,
            buffer_size=args.buffer_size, reward_scale=args.reward_scale,
        )
    elif args.algorithm == "rqe":
        return RQEStagHunt(
            horizon=args.horizon,
            lam=args.lam, beta=args.beta,
            eps1=args.eps1, eps2=args.eps2,
            tau1=args.tau1, tau2=args.tau2,
            buffer_size=args.buffer_size, reward_scale=args.reward_scale,
        )
    elif args.algorithm == "nqovi":
        return NQOVIStagHunt(
            horizon=args.horizon,
            lam=args.lam, beta=args.beta,
            buffer_size=args.buffer_size, reward_scale=args.reward_scale,
        )
    raise ValueError(f"Unknown algorithm: {args.algorithm}")


def train(args):
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    agent = make_agent(args)
    save_fn = {"qre": qre_save, "rqe": rqe_save, "nqovi": nqovi_save}[args.algorithm]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_r0, ep_r1 = [], []
    ep_lens, ep_interactions = [], []
    ep_ss, ep_hh, ep_mixed = [], [], []
    t_start = time.time()

    for k in range(1, args.episodes + 1):
        key, ep_key = jax.random.split(key)
        state = reset(ep_key)
        total_r0, total_r1 = 0.0, 0.0
        interactions = 0
        ss_count, hh_count, mixed_count = 0, 0, 0
        steps = 0

        for h in range(agent.H):
            a0, a1 = agent.select_action(state, h)
            actions = jnp.array([a0, a1], dtype=jnp.int32)

            next_state = step(state, actions)

            r0 = float(next_state.last_rewards[0])
            r1 = float(next_state.last_rewards[1])
            done = bool(next_state.done)

            agent.store_transition(h, state, a0, a1, next_state, r0, r1, done)

            total_r0 += r0
            total_r1 += r1
            if bool(next_state.interaction_happened):
                s = np.asarray(next_state.last_interaction_strategy, dtype=np.int32)
                s0 = int(s[0])
                s1 = int(s[1])

                valid_outcome = (s0 in (0, 1)) and (s1 in (0, 1))
                if valid_outcome:
                    interactions += 1
                    if s0 == 0 and s1 == 0:
                        ss_count += 1
                    elif s0 == 1 and s1 == 1:
                        hh_count += 1
                    else:
                        mixed_count += 1

            state = next_state
            steps += 1

            if done:
                break

        ep_r0.append(total_r0)
        ep_r1.append(total_r1)
        ep_lens.append(steps)
        ep_interactions.append(interactions)
        ep_ss.append(ss_count)
        ep_hh.append(hh_count)
        ep_mixed.append(mixed_count)

        if args.update_freq > 0 and k % args.update_freq == 0:
            agent.update()

        if k % 50 == 0:
            recent = min(50, len(ep_r0))
            ret_tot = np.mean([r0 + r1 for r0, r1 in zip(ep_r0[-recent:], ep_r1[-recent:])])
            avg_int = np.mean(ep_interactions[-recent:])
            ss_recent = int(np.sum(ep_ss[-recent:]))
            hh_recent = int(np.sum(ep_hh[-recent:]))
            mixed_recent = int(np.sum(ep_mixed[-recent:]))
            total_outcomes = max(ss_recent + hh_recent + mixed_recent, 1)
            elapsed = time.time() - t_start
            eps_s = k / max(elapsed, 1e-6)
            print(f"Ep {k:5d} | Return: {ret_tot:7.2f} | Interactions: {avg_int:5.1f} | "
                  f"{eps_s:.1f} ep/s")
            print(
                f"         Outcomes SS/HH/Mixed: "
                f"{ss_recent}/{hh_recent}/{mixed_recent} "
                f"({ss_recent / total_outcomes:.2f}/{hh_recent / total_outcomes:.2f}/{mixed_recent / total_outcomes:.2f})"
            )

    np.savez(
        output_dir / f"{args.algorithm}_staghunt_returns.npz",
        r0=np.array(ep_r0), r1=np.array(ep_r1),
        lengths=np.array(ep_lens), interactions=np.array(ep_interactions),
        ss=np.array(ep_ss), hh=np.array(ep_hh), mixed=np.array(ep_mixed),
    )

    agent_path = output_dir / f"{args.algorithm}_staghunt_agent.pkl"
    save_fn(agent, str(agent_path))

    elapsed = time.time() - t_start
    print(f"\n{args.algorithm.upper()} training done: {args.episodes} episodes in "
          f"{elapsed:.1f}s ({args.episodes / max(elapsed, 1e-6):.1f} ep/s)")

    plot_training_curves(output_dir, args.algorithm)

    if not args.no_gif:
        generate_gif(agent, output_dir, args.algorithm, args.gif_episodes,
                     args.seed, fps=args.gif_fps)

    return agent


def plot_training_curves(output_dir, algorithm):
    data = np.load(output_dir / f"{algorithm}_staghunt_returns.npz")
    r0, r1 = data["r0"], data["r1"]
    total = r0 + r1

    window = min(100, len(total))
    if window <= 1:
        return
    ma = lambda x: np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ma(total), label="Total", linewidth=2)
    axes[0].plot(ma(r0), label="Agent 0", alpha=0.8)
    axes[0].plot(ma(r1), label="Agent 1", alpha=0.8)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title(f"{algorithm.upper()} Stag Hunt Returns")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ss, hh, mixed = data["ss"], data["hh"], data["mixed"]
    totals = np.maximum(ss + hh + mixed, 1).astype(np.float64)
    ss_pct = ma(ss / totals) * 100
    hh_pct = ma(hh / totals) * 100
    mixed_pct = ma(mixed / totals) * 100
    axes[1].plot(ss_pct, label="Stag-Stag", color="#2ecc71", linewidth=2)
    axes[1].plot(hh_pct, label="Hare-Hare", color="#e74c3c", linewidth=2)
    axes[1].plot(mixed_pct, label="Mixed", color="#f39c12", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Outcome %")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Outcome Distribution (rolling avg)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ma(data["interactions"]), color="red")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Interactions per Episode")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{algorithm}_staghunt_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_dir / f'{algorithm}_staghunt_curves.png'}")


def generate_gif(agent, output_dir, algorithm, num_episodes, seed,
                  max_steps_per_ep=300, fps=5):
    key = jax.random.PRNGKey(seed + 1000)
    state_seq = []

    for ep in range(num_episodes):
        key, ep_key = jax.random.split(key)
        state = reset(ep_key)
        for h in range(min(agent.H, max_steps_per_ep)):
            a0, a1 = agent.select_action(state, h)
            actions = jnp.array([a0, a1], dtype=jnp.int32)
            next_state = step(state, actions)
            r = np.array(next_state.last_rewards)
            state_seq.append((next_state, r))
            state = next_state
            if bool(state.done):
                break

    gif_path = str(output_dir / f"{algorithm}_staghunt_rollout.gif")
    animate_rollout(state_seq, filename=gif_path, fps=fps)
    print(f"Saved rollout GIF to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LFA agents on Stag Hunt")
    parser.add_argument("--algorithm", type=str, choices=["qre", "rqe", "nqovi"],
                        required=True)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--update-freq", type=int, default=20)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--reward-scale", type=float, default=4.0)
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="staghunt/results")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-episodes", type=int, default=1)
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--eps1", type=float, default=0.05)
    parser.add_argument("--eps2", type=float, default=0.05)
    parser.add_argument("--qre-max-iter", type=int, default=100)
    parser.add_argument("--tau1", type=float, default=0.01)
    parser.add_argument("--tau2", type=float, default=0.01)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
