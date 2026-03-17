"""Risk-averse QRE (RQRE) agent for Overcooked (JaxMARL) with linear function approximation.

Uses entropic certainty equivalent instead of expected value against opponent.
tau1, tau2: lower = more risk-averse; higher = more risk-neutral.
eps1, eps2: lower = more rational/deterministic; higher = more random.
"""

from collections import deque
from functools import partial
from typing import List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ..lfa import phi_single, phi_all_actions, phi_all_actions_batch, NUM_ACTIONS


# ---------------------------------------------------------------------------
# JAX Utilities
# ---------------------------------------------------------------------------

def softmax_precision(u: jnp.ndarray, eps: float) -> jnp.ndarray:
    z = u / eps
    z = z - jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def entropic_utility_vs_opponent_jax(Q_row: jnp.ndarray, opp_pi: jnp.ndarray, tau: float) -> jnp.ndarray:
    """Risk-adjusted utility: CE(Q) = -tau * log(E[exp(-Q/tau)])."""
    log_pi = jnp.log(jnp.maximum(opp_pi, 1e-300))
    z = log_pi - Q_row / tau
    return -tau * logsumexp(z, axis=-1)


# ---------------------------------------------------------------------------
# JAX RQRE Solvers
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(6, 7, 8))
def rqre_jax(
    Q1: jnp.ndarray,
    Q2: jnp.ndarray,
    eps1: float,
    eps2: float,
    tau1: float,
    tau2: float,
    max_iter: int = 100,
    num_actions: int = NUM_ACTIONS,
    tol: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched RQRE solver. Q1, Q2: [B, A, A]. Returns pi1 [B,A], pi2 [B,A], V1 [B], V2 [B]."""
    B = Q1.shape[0]
    A = num_actions

    pi1_init = jnp.ones((B, A), dtype=jnp.float32) / A
    pi2_init = jnp.ones((B, A), dtype=jnp.float32) / A

    def body_fn(carry, _):
        pi1, pi2, converged = carry

        def continue_iteration(args):
            pi1_curr, pi2_curr, _ = args
            u1 = entropic_utility_vs_opponent_jax(Q1, pi2_curr[:, None, :], tau1)
            Q2_T = jnp.transpose(Q2, (0, 2, 1))
            u2 = entropic_utility_vs_opponent_jax(Q2_T, pi1_curr[:, None, :], tau2)
            pi1_new = softmax_precision(u1, eps1)
            pi2_new = softmax_precision(u2, eps2)
            diff1 = jnp.max(jnp.abs(pi1_new - pi1_curr))
            diff2 = jnp.max(jnp.abs(pi2_new - pi2_curr))
            new_converged = (diff1 < tol) & (diff2 < tol)
            return (pi1_new, pi2_new, new_converged)

        def stop_iteration(args):
            return args

        return jax.lax.cond(
            ~converged, continue_iteration, stop_iteration,
            (pi1, pi2, converged)
        ), None

    init_converged = jnp.array(False)
    (pi1, pi2, _), _ = jax.lax.scan(body_fn, (pi1_init, pi2_init, init_converged), None, length=max_iter)

    u1_final = entropic_utility_vs_opponent_jax(Q1, pi2[:, None, :], tau1)
    Q2_T = jnp.transpose(Q2, (0, 2, 1))
    u2_final = entropic_utility_vs_opponent_jax(Q2_T, pi1[:, None, :], tau2)

    V1 = jnp.sum(pi1 * u1_final, axis=-1)
    V2 = jnp.sum(pi2 * u2_final, axis=-1)

    return pi1, pi2, V1, V2


def rqre_single(
    Q1: np.ndarray,
    Q2: np.ndarray,
    eps1: float,
    eps2: float,
    tau1: float,
    tau2: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Single-state RQRE solver."""
    Q1_b = jnp.array(Q1)[None, :, :]
    Q2_b = jnp.array(Q2)[None, :, :]
    pi1, pi2, V1, V2 = rqre_jax(Q1_b, Q2_b, eps1, eps2, tau1, tau2, 100, Q1.shape[0], 1e-6)
    return np.array(pi1[0]), np.array(pi2[0]), float(V1[0]), float(V2[0])


def rqre_batch(
    Q1: np.ndarray,
    Q2: np.ndarray,
    eps1: float,
    eps2: float,
    tau1: float,
    tau2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched RQRE solver."""
    pi1, pi2, V1, V2 = rqre_jax(
        jnp.array(Q1), jnp.array(Q2), eps1, eps2, tau1, tau2, 100, Q1.shape[1], 1e-6
    )
    return np.array(pi1), np.array(pi2), np.array(V1), np.array(V2)


# ---------------------------------------------------------------------------
# RQE Agent
# ---------------------------------------------------------------------------

class RQEOvercooked:
    """Risk-averse QRE with LSVI for Overcooked (JaxMARL)."""

    def __init__(
        self,
        *,
        feature_dim: int,
        horizon: int,
        num_actions: int = NUM_ACTIONS,
        lam: float = 1.0,
        beta: float = 1.0,
        eps1: float = 2.0,
        eps2: float = 2.0,
        tau1: float = 1.0,
        tau2: float = 1.0,
        buffer_size: int = 500,
        reward_scale: float = 20.0,
    ):
        self.H = int(horizon)
        self.lam = float(lam)
        self.beta = float(beta)
        self.num_actions = int(num_actions)
        self.buffer_size = int(buffer_size)
        self.reward_scale = float(reward_scale)
        self.eps1 = float(eps1)
        self.eps2 = float(eps2)
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)

        D = int(feature_dim)
        self._feature_dim = D

        self.w1: List[np.ndarray] = [np.zeros(D, dtype=np.float32) for _ in range(self.H)]
        self.w2: List[np.ndarray] = [np.zeros(D, dtype=np.float32) for _ in range(self.H)]
        self.Lambda: List[np.ndarray] = [self.lam * np.eye(D, dtype=np.float64) for _ in range(self.H)]
        self.Lambda_inv: List[np.ndarray] = [
            (1.0 / self.lam) * np.eye(D, dtype=np.float64) for _ in range(self.H)
        ]
        self.buffers: List[deque] = [deque(maxlen=None) for _ in range(self.H)]

    def _optimistic_Q(self, h: int, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimistic Q-matrices for a single state."""
        h_idx = min(max(int(h), 0), self.H - 1)
        A = self.num_actions

        Phi = phi_all_actions(obs)  # [A, A, D]
        phi_flat = Phi.reshape(A * A, -1)

        q1_flat = phi_flat @ self.w1[h_idx]
        q2_flat = phi_flat @ self.w2[h_idx]

        tmp = phi_flat @ self.Lambda_inv[h_idx]
        quad = np.sum(tmp * phi_flat, axis=1)
        bonus = self.beta * np.sqrt(np.maximum(quad, 0.0))

        cap = self.reward_scale * float(self.H - h + 1)
        Q1 = np.minimum(q1_flat + bonus, cap).reshape(A, A)
        Q2 = np.minimum(q2_flat + bonus, cap).reshape(A, A)
        return Q1, Q2

    def _optimistic_Q_batch(self, obs_batch: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimistic Q-matrices for a batch of states."""
        h_idx = min(max(int(h), 0), self.H - 1)
        A = self.num_actions
        B = obs_batch.shape[0]

        phi_batch = phi_all_actions_batch(obs_batch)  # [B, A, A, D]
        D = self._feature_dim
        phi_flat = phi_batch.reshape(B * A * A, D)

        q1_flat = phi_flat @ self.w1[h_idx]
        q2_flat = phi_flat @ self.w2[h_idx]

        tmp = phi_flat @ self.Lambda_inv[h_idx]
        quad = np.sum(tmp * phi_flat, axis=1)
        bonus = self.beta * np.sqrt(np.maximum(quad, 0.0))

        cap = self.reward_scale * float(self.H - h + 1)
        Q1 = np.minimum(q1_flat + bonus, cap).reshape(B, A, A)
        Q2 = np.minimum(q2_flat + bonus, cap).reshape(B, A, A)
        return Q1, Q2

    def select_action(self, obs: np.ndarray, h: int) -> Tuple[int, int]:
        Q1, Q2 = self._optimistic_Q(h, obs)

        pi1, pi2, _, _ = rqre_single(Q1, Q2, self.eps1, self.eps2, self.tau1, self.tau2)
        pi1_np = np.asarray(pi1, dtype=np.float64)
        pi2_np = np.asarray(pi2, dtype=np.float64)
        pi1_np = np.clip(pi1_np, 0.0, None)
        pi2_np = np.clip(pi2_np, 0.0, None)
        pi1_np /= pi1_np.sum()
        pi2_np /= pi2_np.sum()
        a0 = int(np.random.choice(self.num_actions, p=pi1_np))
        a1 = int(np.random.choice(self.num_actions, p=pi2_np))
        return a0, a1

    def store_transition(
        self,
        h: int,
        obs: np.ndarray,
        a0: int,
        a1: int,
        next_obs: np.ndarray,
        r0: float,
        r1: float,
        done: bool,
    ) -> None:
        h = int(h)
        phi = np.asarray(phi_single(obs, a0, a1), dtype=np.float64)

        if len(self.buffers[h]) >= self.buffer_size:
            oldest = self.buffers[h].popleft()
            self.Lambda[h] -= np.outer(oldest["phi"], oldest["phi"])

        self.buffers[h].append({
            "phi": phi,
            "next_obs": next_obs,
            "r0": float(r0),
            "r1": float(r1),
            "done": bool(done),
        })

        self.Lambda[h] += np.outer(phi, phi)

    def update(self) -> None:
        """Risk-averse LSVI update using incremental Lambda (float64 for stability)."""
        for h in range(self.H - 1, -1, -1):
            buf = self.buffers[h]
            if len(buf) == 0:
                continue

            D = self._feature_dim
            Lambda_h = self.Lambda[h]
            b0 = np.zeros(D, dtype=np.float64)
            b1 = np.zeros(D, dtype=np.float64)

            if h < self.H - 1:
                next_obs_list = []
                trans_indices = []
                for i, trans in enumerate(buf):
                    if not trans["done"]:
                        next_obs_list.append(trans["next_obs"])
                        trans_indices.append(i)

                if len(next_obs_list) > 0:
                    next_obs_batch = np.stack(next_obs_list, axis=0)
                    Q1_batch, Q2_batch = self._optimistic_Q_batch(next_obs_batch, h + 1)

                    _, _, V1_batch, V2_batch = rqre_batch(
                        Q1_batch, Q2_batch, self.eps1, self.eps2, self.tau1, self.tau2
                    )

                    V_next_map = {
                        trans_idx: (float(V1_batch[idx]), float(V2_batch[idx]))
                        for idx, trans_idx in enumerate(trans_indices)
                    }
                else:
                    V_next_map = {}
            else:
                V_next_map = {}

            for i, trans in enumerate(buf):
                phi = trans["phi"]
                if h == self.H - 1 or trans["done"]:
                    V0_next, V1_next = 0.0, 0.0
                else:
                    V0_next, V1_next = V_next_map.get(i, (0.0, 0.0))

                b0 = b0 + phi * (float(trans["r0"]) + float(V0_next))
                b1 = b1 + phi * (float(trans["r1"]) + float(V1_next))

            try:
                self.w1[h] = np.linalg.solve(Lambda_h, b0)
                self.w2[h] = np.linalg.solve(Lambda_h, b1)
                self.Lambda_inv[h] = np.linalg.solve(Lambda_h, np.eye(D, dtype=np.float64))
            except np.linalg.LinAlgError:
                Lambda_h = Lambda_h + 1e-3 * np.eye(D, dtype=np.float64)
                self.w1[h] = np.linalg.solve(Lambda_h, b0)
                self.w2[h] = np.linalg.solve(Lambda_h, b1)
                self.Lambda_inv[h] = np.linalg.solve(Lambda_h, np.eye(D, dtype=np.float64))
                self.Lambda[h] = Lambda_h


def save_agent(agent: RQEOvercooked, path: str) -> None:
    import pickle
    checkpoint = {
        "w1": agent.w1,
        "w2": agent.w2,
        "Lambda": agent.Lambda,
        "Lambda_inv": agent.Lambda_inv,
        "H": agent.H,
        "lam": agent.lam,
        "beta": agent.beta,
        "num_actions": agent.num_actions,
        "buffer_size": agent.buffer_size,
        "eps1": agent.eps1,
        "eps2": agent.eps2,
        "tau1": agent.tau1,
        "tau2": agent.tau2,
        "feature_dim": agent._feature_dim,
        "reward_scale": agent.reward_scale,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved RQE agent to {path}")


def load_agent(path: str) -> RQEOvercooked:
    import pickle
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    agent = RQEOvercooked(
        feature_dim=checkpoint["feature_dim"],
        horizon=checkpoint["H"],
        num_actions=checkpoint["num_actions"],
        lam=checkpoint["lam"],
        beta=checkpoint["beta"],
        eps1=checkpoint["eps1"],
        eps2=checkpoint["eps2"],
        tau1=checkpoint.get("tau1", 1.0),
        tau2=checkpoint.get("tau2", 1.0),
        buffer_size=checkpoint["buffer_size"],
        reward_scale=checkpoint.get("reward_scale", 20.0),
    )
    agent.w1 = checkpoint["w1"]
    agent.w2 = checkpoint["w2"]
    agent.Lambda = checkpoint["Lambda"]
    agent.Lambda_inv = checkpoint["Lambda_inv"]
    print(f"Loaded RQE agent from {path}")
    return agent
