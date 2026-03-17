import pickle
import numpy as np
from typing import List, Tuple
from collections import deque
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from staghunt.staghunt import State, NUM_ACTIONS
from staghunt.lfa import extract_features, phi_batch_jax, FEATURE_DIM


def softmax_precision(u: jnp.ndarray, eps: float) -> jnp.ndarray:
    z = u / eps
    z = z - jnp.max(z, axis=-1, keepdims=True)
    e = jnp.exp(z)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def entropic_utility_vs_opponent(Q_row: jnp.ndarray, opp_pi: jnp.ndarray,
                                  tau: float) -> jnp.ndarray:
    """CE = -tau * logsumexp(log(pi) - Q/tau)"""
    log_pi = jnp.log(jnp.maximum(opp_pi, 1e-300))
    z = log_pi - Q_row / tau
    return -tau * logsumexp(z, axis=-1)


@partial(jax.jit, static_argnums=(6, 7, 8))
def rqre_jax(
    Q1: jnp.ndarray, Q2: jnp.ndarray,
    eps1: float, eps2: float, tau1: float, tau2: float,
    max_iter: int = 100, num_actions: int = NUM_ACTIONS, tol: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched RQRE solver. Q1, Q2: [B, A, A]. Returns pi1, pi2, V1, V2."""
    B = Q1.shape[0]
    A = num_actions
    pi1_init = jnp.ones((B, A), dtype=jnp.float32) / A
    pi2_init = jnp.ones((B, A), dtype=jnp.float32) / A

    def body_fn(carry, _):
        pi1, pi2, converged = carry

        def continue_iteration(args):
            pi1_c, pi2_c, _ = args
            u1 = entropic_utility_vs_opponent(Q1, pi2_c[:, None, :], tau1)
            Q2_T = jnp.transpose(Q2, (0, 2, 1))
            u2 = entropic_utility_vs_opponent(Q2_T, pi1_c[:, None, :], tau2)
            pi1_new = softmax_precision(u1, eps1)
            pi2_new = softmax_precision(u2, eps2)
            diff1 = jnp.max(jnp.abs(pi1_new - pi1_c))
            diff2 = jnp.max(jnp.abs(pi2_new - pi2_c))
            return (pi1_new, pi2_new, (diff1 < tol) & (diff2 < tol))

        def stop_iteration(args):
            return args

        return jax.lax.cond(~converged, continue_iteration, stop_iteration,
                            (pi1, pi2, converged)), None

    (pi1, pi2, _), _ = jax.lax.scan(
        body_fn, (pi1_init, pi2_init, jnp.array(False)), None, length=max_iter)

    u1_final = entropic_utility_vs_opponent(Q1, pi2[:, None, :], tau1)
    Q2_T = jnp.transpose(Q2, (0, 2, 1))
    u2_final = entropic_utility_vs_opponent(Q2_T, pi1[:, None, :], tau2)
    V1 = jnp.sum(pi1 * u1_final, axis=-1)
    V2 = jnp.sum(pi2 * u2_final, axis=-1)
    return pi1, pi2, V1, V2


@jax.jit
def rqre_single(
    Q1: jnp.ndarray, Q2: jnp.ndarray,
    eps1: float, eps2: float, tau1: float, tau2: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """Single-state RQRE solver."""
    Q1_b = Q1[None, :, :]
    Q2_b = Q2[None, :, :]
    pi1, pi2, V1, V2 = rqre_jax(Q1_b, Q2_b, eps1, eps2, tau1, tau2,
                                  100, Q1.shape[0], 1e-6)
    return pi1[0], pi2[0], V1[0], V2[0]


class RQEStagHunt:
    """Risk-averse QRE agent for Stag Hunt."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        horizon: int,
        num_actions: int = NUM_ACTIONS,
        lam: float = 1.0,
        beta: float = 1.0,
        eps1: float = 2.0,
        eps2: float = 2.0,
        tau1: float = 1.0,
        tau2: float = 1.0,
        buffer_size: int = 2000,
        reward_scale: float = 4.0,
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
        self._feature_dim = int(feature_dim)

        D = self._feature_dim
        self.w1: List[np.ndarray] = [np.zeros(D, dtype=np.float32) for _ in range(self.H)]
        self.w2: List[np.ndarray] = [np.zeros(D, dtype=np.float32) for _ in range(self.H)]
        self.Lambda: List[np.ndarray] = [self.lam * np.eye(D, dtype=np.float64) for _ in range(self.H)]
        self.Lambda_inv: List[np.ndarray] = [
            (1.0 / self.lam) * np.eye(D, dtype=np.float64) for _ in range(self.H)
        ]
        self.buffers: List[deque] = [deque(maxlen=None) for _ in range(self.H)]

    def _optimistic_Q(self, h: int, state: State) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimistic Q-matrices for a single state."""
        h_idx = min(max(int(h), 0), self.H - 1)
        A = self.num_actions

        sf = np.asarray(extract_features(state), dtype=np.float32)
        sf_batch = np.broadcast_to(sf[None, :], (A * A, sf.shape[0]))
        a0_all = np.repeat(np.arange(A, dtype=np.int32), A)
        a1_all = np.tile(np.arange(A, dtype=np.int32), A)
        phi_flat = np.asarray(phi_batch_jax(sf_batch, a0_all, a1_all), dtype=np.float32)

        q1_flat = phi_flat @ self.w1[h_idx]
        q2_flat = phi_flat @ self.w2[h_idx]

        tmp = phi_flat @ self.Lambda_inv[h_idx]
        quad = np.sum(tmp * phi_flat, axis=1)
        bonus = self.beta * np.sqrt(np.maximum(quad, 0.0))

        cap = self.reward_scale * float(self.H - h + 1)
        Q1 = np.minimum(q1_flat + bonus, cap).reshape(A, A)
        Q2 = np.minimum(q2_flat + bonus, cap).reshape(A, A)
        return Q1, Q2

    def _optimistic_Q_batch(self, state_batch: List[State], h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimistic Q-matrices for a batch of states. Returns [B, A, A]."""
        h_idx = min(max(int(h), 0), self.H - 1)
        A = self.num_actions
        B = len(state_batch)

        state_feat_batch = np.stack(
            [np.asarray(extract_features(s), dtype=np.float32) for s in state_batch],
            axis=0,
        )
        sd = state_feat_batch.shape[1]
        sf_expanded = np.tile(state_feat_batch[:, None, :], (1, A * A, 1)).reshape(B * A * A, sd)
        a0_all = np.tile(np.repeat(np.arange(A, dtype=np.int32), A), B)
        a1_all = np.tile(np.tile(np.arange(A, dtype=np.int32), A), B)
        phi_flat = np.asarray(phi_batch_jax(sf_expanded, a0_all, a1_all), dtype=np.float32)

        q1_flat = phi_flat @ self.w1[h_idx]
        q2_flat = phi_flat @ self.w2[h_idx]
        tmp = phi_flat @ self.Lambda_inv[h_idx]
        quad = np.sum(tmp * phi_flat, axis=1)
        bonus = self.beta * np.sqrt(np.maximum(quad, 0.0))

        cap = self.reward_scale * float(self.H - h + 1)
        Q1_batch = np.minimum(q1_flat + bonus, cap).reshape(B, A, A)
        Q2_batch = np.minimum(q2_flat + bonus, cap).reshape(B, A, A)
        return Q1_batch, Q2_batch

    def select_action(self, state: State, h: int) -> Tuple[int, int]:
        Q1, Q2 = self._optimistic_Q(h, state)
        pi1, pi2, _, _ = rqre_single(jnp.array(Q1), jnp.array(Q2),
                                      self.eps1, self.eps2, self.tau1, self.tau2)
        pi1_np = np.asarray(pi1, dtype=np.float64)
        pi2_np = np.asarray(pi2, dtype=np.float64)
        pi1_np = np.clip(pi1_np, 0.0, None); pi1_np /= pi1_np.sum()
        pi2_np = np.clip(pi2_np, 0.0, None); pi2_np /= pi2_np.sum()
        a0 = int(np.random.choice(self.num_actions, p=pi1_np))
        a1 = int(np.random.choice(self.num_actions, p=pi2_np))
        return a0, a1

    def store_transition(self, h: int, state: State, a0: int, a1: int,
                         next_state: State, r0: float, r1: float,
                         done: bool) -> None:
        """Store transition. Only maintains Lambda[h] incrementally."""
        h = int(h)
        sf = np.asarray(extract_features(state), dtype=np.float32)
        phi = np.asarray(
            phi_batch_jax(sf[None, :], np.asarray([a0], dtype=np.int32),
                          np.asarray([a1], dtype=np.int32)),
            dtype=np.float64,
        )[0]

        if len(self.buffers[h]) >= self.buffer_size:
            oldest = self.buffers[h].popleft()
            self.Lambda[h] -= np.outer(oldest["phi"], oldest["phi"])

        self.buffers[h].append({
            "phi": phi,
            "next_state": next_state,
            "r0": float(r0),
            "r1": float(r1),
            "done": bool(done),
        })
        self.Lambda[h] += np.outer(phi, phi)

    def update(self) -> None:
        """LSVI update using incremental Lambda (float64 for stability)."""
        for h in range(self.H - 1, -1, -1):
            buf = self.buffers[h]
            if len(buf) == 0:
                continue

            D = self._feature_dim
            Lambda_h = self.Lambda[h]
            b0 = np.zeros(D, dtype=np.float64)
            b1 = np.zeros(D, dtype=np.float64)

            # Compute V_next for non-done transitions
            if h < self.H - 1:
                next_state_list = []
                trans_indices = []
                for i, trans in enumerate(buf):
                    if not trans["done"]:
                        next_state_list.append(trans["next_state"])
                        trans_indices.append(i)

                V_next_map = {}
                if len(next_state_list) > 0:
                    Q1_np, Q2_np = self._optimistic_Q_batch(next_state_list, h + 1)
                    _, _, V1_batch, V2_batch = rqre_jax(
                        jnp.array(Q1_np), jnp.array(Q2_np),
                        self.eps1, self.eps2, self.tau1, self.tau2,
                        max_iter=100, num_actions=self.num_actions, tol=1e-6,
                    )
                    V1_batch = np.array(V1_batch)
                    V2_batch = np.array(V2_batch)
                    for idx, trans_idx in enumerate(trans_indices):
                        V_next_map[trans_idx] = (V1_batch[idx], V2_batch[idx])
            else:
                V_next_map = {}

            # Accumulate targets
            for i, trans in enumerate(buf):
                phi = trans["phi"]
                if h == self.H - 1 or trans["done"]:
                    V0_next, V1_next = 0.0, 0.0
                else:
                    V0_next, V1_next = V_next_map.get(i, (0.0, 0.0))
                b0 += phi * (float(trans["r0"]) + float(V0_next))
                b1 += phi * (float(trans["r1"]) + float(V1_next))

            # Solve from scratch
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


def save_agent(agent: RQEStagHunt, path: str) -> None:
    checkpoint = {
        "w1": agent.w1, "w2": agent.w2,
        "Lambda": agent.Lambda, "Lambda_inv": agent.Lambda_inv,
        "H": agent.H,
        "lam": agent.lam, "beta": agent.beta,
        "num_actions": agent.num_actions, "buffer_size": agent.buffer_size,
        "eps1": agent.eps1, "eps2": agent.eps2,
        "tau1": agent.tau1, "tau2": agent.tau2,
        "feature_dim": agent._feature_dim, "reward_scale": agent.reward_scale,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_agent(path: str) -> RQEStagHunt:
    with open(path, "rb") as f:
        ck = pickle.load(f)
    agent = RQEStagHunt(
        feature_dim=ck["feature_dim"], num_actions=ck["num_actions"],
        horizon=ck.get("H", 100), lam=ck["lam"], beta=ck["beta"],
        eps1=ck["eps1"], eps2=ck["eps2"],
        tau1=ck.get("tau1", 1.0), tau2=ck.get("tau2", 1.0),
        buffer_size=ck["buffer_size"], reward_scale=ck["reward_scale"],
    )
    agent.w1 = ck["w1"]
    agent.w2 = ck["w2"]
    agent.Lambda = ck["Lambda"]
    agent.Lambda_inv = ck["Lambda_inv"]
    return agent
