"""NQOVI agent for Overcooked (JaxMARL) with linear function approximation."""

from __future__ import annotations

import pickle
import warnings
from collections import deque
from itertools import combinations
from typing import List, Tuple

import numpy as np

from ..lfa import NUM_ACTIONS, phi_all_actions, phi_all_actions_batch, phi_single


_NASH_UNIFORM_FALLBACK_COUNT = 0


# ---------------------------------------------------------------------------
# Nash equilibrium solvers
# ---------------------------------------------------------------------------

def _all_nash_enumeration(
    P1: np.ndarray,
    P2: np.ndarray,
    A: int,
    tol: float = 1e-6,
) -> list:
    """Find all Nash equilibria via support enumeration.

    Returns a list of (pi1, pi2, V1, V2) tuples.
    """
    def _solve_mixture(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        try:
            x = np.linalg.solve(np.asarray(M, dtype=np.float64),
                                np.asarray(b, dtype=np.float64)).reshape(-1)
        except np.linalg.LinAlgError:
            return None
        if np.any(~np.isfinite(x)) or np.any(x < -tol):
            return None
        s = float(np.sum(x))
        if s <= tol:
            return None
        return x / s

    best_r1 = np.zeros((A, A), dtype=bool)
    best_r2 = np.zeros((A, A), dtype=bool)
    for a1 in range(A):
        mx = np.max(P1[:, a1])
        best_r1[P1[:, a1] >= mx - tol, a1] = True
    for a0 in range(A):
        mx = np.max(P2[a0, :])
        best_r2[a0, P2[a0, :] >= mx - tol] = True

    pure_candidates = np.argwhere(best_r1 & best_r2)
    if pure_candidates.size > 0:
        pick = int(np.random.randint(len(pure_candidates)))
        a0, a1 = int(pure_candidates[pick][0]), int(pure_candidates[pick][1])
        pi1 = np.zeros(A, dtype=np.float64)
        pi2 = np.zeros(A, dtype=np.float64)
        pi1[a0] = 1.0
        pi2[a1] = 1.0
        return [(pi1, pi2, float(P1[a0, a1]), float(P2[a0, a1]))]

    supports_by_size = {k: [tuple(s) for s in combinations(range(A), k)] for k in range(1, A + 1)}
    mixed_solutions = []
    for k in range(2, min(2, A) + 1):
        supports = supports_by_size[k]
        for S in supports:
            S_list = list(S)
            a0_ref = S_list[0]
            for T in supports:
                T_list = list(T)
                a1_ref = T_list[0]

                rows, rhs = [], []
                for a in S_list[1:]:
                    rows.append(P1[a, T_list] - P1[a0_ref, T_list])
                    rhs.append(0.0)
                rows.append(np.ones(len(T_list), dtype=np.float64))
                rhs.append(1.0)
                pi2_T = _solve_mixture(np.vstack(rows), np.asarray(rhs, dtype=np.float64))
                if pi2_T is None:
                    continue

                rows, rhs = [], []
                for b_idx in T_list[1:]:
                    rows.append(P2[S_list, b_idx] - P2[S_list, a1_ref])
                    rhs.append(0.0)
                rows.append(np.ones(len(S_list), dtype=np.float64))
                rhs.append(1.0)
                pi1_S = _solve_mixture(np.vstack(rows), np.asarray(rhs, dtype=np.float64))
                if pi1_S is None:
                    continue

                pi1 = np.zeros(A, dtype=np.float64)
                pi2 = np.zeros(A, dtype=np.float64)
                for idx, a in enumerate(S_list):
                    pi1[a] = pi1_S[idx]
                for idx, b_idx in enumerate(T_list):
                    pi2[b_idx] = pi2_T[idx]

                V1 = float(pi1 @ P1 @ pi2)
                V2 = float(pi1 @ P2 @ pi2)
                if np.any(P1 @ pi2 > V1 + 1e-6):
                    continue
                if np.any(pi1 @ P2 > V2 + 1e-6):
                    continue
                mixed_solutions.append((pi1, pi2, V1, V2))

    return mixed_solutions


def solve_nash(
    P1: np.ndarray,
    P2: np.ndarray,
    A: int,
    selection: str = "welfare",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Find a Nash equilibrium and select among all equilibria by criterion.

    selection:
      "welfare" - maximise V1+V2 (default; best for cooperative games)
      "maximin" - maximise min(V1, V2)
      "random"  - uniformly at random from all equilibria
    """
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)

    equilibria = _all_nash_enumeration(P1, P2, A)

    global _NASH_UNIFORM_FALLBACK_COUNT

    if not equilibria:
        _NASH_UNIFORM_FALLBACK_COUNT += 1
        count = _NASH_UNIFORM_FALLBACK_COUNT
        if count <= 5 or count % 100 == 0:
            msg = (
                "NQOVI fallback: no equilibrium candidates found; "
                f"using uniform policy (count={count})."
            )
            print(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        # Fall back to uniform policy when support enumeration yields no candidates.
        pi_u = np.ones(A, dtype=np.float64) / A
        return pi_u, pi_u, float(pi_u @ P1 @ pi_u), float(pi_u @ P2 @ pi_u)

    if selection == "random":
        idx = int(np.random.randint(len(equilibria)))
    else:
        if selection == "welfare":
            scores = np.array([V1 + V2 for _, _, V1, V2 in equilibria])
        elif selection == "maximin":
            scores = np.array([min(V1, V2) for _, _, V1, V2 in equilibria])
        else:
            raise ValueError(f"Unknown Nash selection criterion: {selection!r}")
        # Break ties randomly so exploration isn't killed when Q values are uniform
        best = float(scores.max())
        tied = np.where(scores >= best - 1e-6)[0]
        idx = int(np.random.choice(tied))

    pi1, pi2, V1, V2 = equilibria[idx]
    pi1 = np.maximum(pi1, 0.0)
    pi2 = np.maximum(pi2, 0.0)
    pi1 /= pi1.sum()
    pi2 /= pi2.sum()
    return pi1, pi2, float(V1), float(V2)


# ---------------------------------------------------------------------------
# NQOVI agent
# ---------------------------------------------------------------------------

class NQOVIOvercooked:
    """Finite-horizon NQOVI with optimistic LSVI updates."""

    def __init__(
        self,
        *,
        feature_dim: int,
        horizon: int,
        num_actions: int = NUM_ACTIONS,
        lam: float = 1.0,
        beta: float = 1.0,
        buffer_size: int = 2000,
        reward_scale: float = 20.0,
        nash_selection: str = "welfare",
    ):
        self.H = int(horizon)
        self.lam = float(lam)
        self.beta = float(beta)
        self.num_actions = int(num_actions)
        self.buffer_size = int(buffer_size)
        self.reward_scale = float(reward_scale)
        self.nash_selection = str(nash_selection)
        self._feature_dim = int(feature_dim)

        D = self._feature_dim
        self.w1: List[np.ndarray] = [np.zeros(D, dtype=np.float32) for _ in range(self.H)]
        self.w2: List[np.ndarray] = [np.zeros(D, dtype=np.float32) for _ in range(self.H)]
        self.Lambda: List[np.ndarray] = [self.lam * np.eye(D, dtype=np.float32) for _ in range(self.H)]
        self.Lambda_inv: List[np.ndarray] = [
            (1.0 / self.lam) * np.eye(D, dtype=np.float32) for _ in range(self.H)
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

    def select_action(self, obs: np.ndarray, h: int) -> Tuple[int, int]:
        Q1, Q2 = self._optimistic_Q(h, obs)

        pi1, pi2, _, _ = solve_nash(Q1, Q2, self.num_actions, self.nash_selection)
        a0 = int(np.random.choice(self.num_actions, p=pi1.astype(np.float64)))
        a1 = int(np.random.choice(self.num_actions, p=pi2.astype(np.float64)))
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
        phi = phi_single(obs, int(a0), int(a1))

        if len(self.buffers[h]) >= self.buffer_size:
            old = self.buffers[h].popleft()
            self.Lambda[h] -= np.outer(old["phi"], old["phi"])

        self.buffers[h].append({
            "phi": phi,
            "next_obs": next_obs,
            "r0": float(r0),
            "r1": float(r1),
            "done": bool(done),
        })

        self.Lambda[h] += np.outer(phi, phi)

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

    def update(self) -> None:
        """LSVI backward pass; recomputes Lambda_inv from scratch each call."""
        for h in range(self.H - 1, -1, -1):
            buf = self.buffers[h]
            if len(buf) == 0:
                continue

            D = self._feature_dim

            b0 = np.zeros(D, dtype=np.float32)
            b1 = np.zeros(D, dtype=np.float32)

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

                    V_next_map = {}
                    for idx, trans_idx in enumerate(trans_indices):
                        _, _, V1, V2 = solve_nash(
                            Q1_batch[idx], Q2_batch[idx], self.num_actions, self.nash_selection
                        )
                        V_next_map[trans_idx] = (V1, V2)
                else:
                    V_next_map = {}
            else:
                V_next_map = {}

            # Recompute Lambda from scratch to prevent drift from incremental rank-1 updates
            Lambda_h = self.lam * np.eye(D, dtype=np.float32)
            for i, trans in enumerate(buf):
                phi = trans["phi"]
                Lambda_h += np.outer(phi, phi)
                if h == self.H - 1 or trans["done"]:
                    V0_next, V1_next = 0.0, 0.0
                else:
                    V0_next, V1_next = V_next_map.get(i, (0.0, 0.0))

                b0 = b0 + phi * (float(trans["r0"]) + float(V0_next))
                b1 = b1 + phi * (float(trans["r1"]) + float(V1_next))
            self.Lambda[h] = Lambda_h

            try:
                self.Lambda_inv[h] = np.linalg.solve(
                    self.Lambda[h], np.eye(D, dtype=np.float32)
                )
            except np.linalg.LinAlgError:
                self.Lambda_inv[h] = np.linalg.solve(
                    self.Lambda[h] + 1e-3 * np.eye(D, dtype=np.float32),
                    np.eye(D, dtype=np.float32),
                )

            Linv = self.Lambda_inv[h]
            self.w1[h] = Linv @ b0
            self.w2[h] = Linv @ b1


def save_agent(agent: NQOVIOvercooked, path: str) -> None:
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
        "feature_dim": agent._feature_dim,
        "reward_scale": agent.reward_scale,
        "nash_selection": agent.nash_selection,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved NQOVI agent to {path}")


def load_agent(path: str) -> NQOVIOvercooked:
    with open(path, "rb") as f:
        ck = pickle.load(f)

    agent = NQOVIOvercooked(
        feature_dim=ck["feature_dim"],
        horizon=ck.get("H", 120),
        num_actions=ck["num_actions"],
        lam=ck["lam"],
        beta=ck["beta"],
        buffer_size=ck["buffer_size"],
        reward_scale=ck.get("reward_scale", 20.0),
        nash_selection=ck.get("nash_selection", "welfare"),
    )
    agent.w1 = ck["w1"]
    agent.w2 = ck["w2"]
    agent.Lambda = ck["Lambda"]
    agent.Lambda_inv = ck["Lambda_inv"]
    print(f"Loaded NQOVI agent from {path}")
    return agent
