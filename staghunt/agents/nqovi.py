import pickle
import numpy as np
from typing import List, Tuple
from collections import deque
from itertools import combinations

from staghunt.staghunt import State, NUM_ACTIONS
from staghunt.lfa import extract_features, phi_batch_jax, FEATURE_DIM


_NE_SOLVE_CALLS = 0
_NE_FALLBACK_CALLS = 0


def get_ne_solver_stats() -> tuple[int, int, float]:
    calls = int(_NE_SOLVE_CALLS)
    fallback_calls = int(_NE_FALLBACK_CALLS)
    fallback_rate = 100.0 * fallback_calls / max(calls, 1)
    return calls, fallback_calls, fallback_rate


def mixed_nash_support_enumeration(
    payoffs1: np.ndarray,
    payoffs2: np.ndarray,
    num_actions: int,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute a mixed Nash equilibrium by support enumeration."""
    global _NE_SOLVE_CALLS, _NE_FALLBACK_CALLS

    _NE_SOLVE_CALLS += 1
    A = int(num_actions)
    P1 = np.asarray(payoffs1, dtype=np.float64)
    P2 = np.asarray(payoffs2, dtype=np.float64)

    # Try pure equilibria first.
    best_r1 = np.zeros((A, A), dtype=bool)
    best_r2 = np.zeros((A, A), dtype=bool)
    for a1 in range(A):
        mx = np.max(P1[:, a1])
        best_r1[P1[:, a1] >= mx - tol, a1] = True
    for a0 in range(A):
        mx = np.max(P2[a0, :])
        best_r2[a0, P2[a0, :] >= mx - tol] = True

    nash_mask = best_r1 & best_r2
    candidates = np.argwhere(nash_mask)
    if candidates.size > 0:
        pick = int(np.random.randint(len(candidates)))
        a0s, a1s = int(candidates[pick][0]), int(candidates[pick][1])
        pi1 = np.zeros(A, dtype=np.float64)
        pi2 = np.zeros(A, dtype=np.float64)
        pi1[a0s] = 1.0
        pi2[a1s] = 1.0
        return pi1, pi2, float(P1[a0s, a1s]), float(P2[a0s, a1s])

    def _solve_mixture(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        M = np.asarray(M, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        try:
            x = np.linalg.solve(M, b).reshape(-1)
        except np.linalg.LinAlgError:
            return None
        if np.any(~np.isfinite(x)) or np.any(x < -tol):
            return None
        s = float(np.sum(x))
        if s <= tol:
            return None
        return x / s

    supports_by_size = {
        k: [tuple(s) for s in combinations(range(A), k)] for k in range(1, A + 1)
    }

    def _collect_mixed(max_support_size: int) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
        sols = []
        for k in range(2, min(max_support_size, A) + 1):
            supports = supports_by_size[k]
            for S in supports:
                a0_ref = S[0]
                for T in supports:
                    a1_ref = T[0]

                    rows, rhs = [], []
                    for a in S[1:]:
                        rows.append(P1[a, list(T)] - P1[a0_ref, list(T)])
                        rhs.append(0.0)
                    rows.append(np.ones(len(T), dtype=np.float64))
                    rhs.append(1.0)
                    pi2_T = _solve_mixture(np.vstack(rows), np.asarray(rhs, dtype=np.float64))
                    if pi2_T is None:
                        continue

                    rows, rhs = [], []
                    for b_idx in T[1:]:
                        rows.append(P2[list(S), b_idx] - P2[list(S), a1_ref])
                        rhs.append(0.0)
                    rows.append(np.ones(len(S), dtype=np.float64))
                    rhs.append(1.0)
                    pi1_S = _solve_mixture(np.vstack(rows), np.asarray(rhs, dtype=np.float64))
                    if pi1_S is None:
                        continue

                    pi1 = np.zeros(A, dtype=np.float64)
                    pi2 = np.zeros(A, dtype=np.float64)
                    for idx, a in enumerate(S):
                        pi1[a] = pi1_S[idx]
                    for idx, b_idx in enumerate(T):
                        pi2[b_idx] = pi2_T[idx]

                    V1 = float(pi1 @ P1 @ pi2)
                    V2 = float(pi1 @ P2 @ pi2)

                    if np.any(P1 @ pi2 > V1 + 1e-6):
                        continue
                    if np.any(pi1 @ P2 > V2 + 1e-6):
                        continue

                    sols.append((pi1, pi2, V1, V2))
        return sols

    # No pure NE — enumerate mixed NE (start with small supports, fallback to full)
    all_solutions = _collect_mixed(max_support_size=2)
    if not all_solutions and A > 2:
        _NE_FALLBACK_CALLS += 1
        all_solutions = _collect_mixed(max_support_size=A)

    if not all_solutions:
        raise RuntimeError(
            "No Nash equilibrium found by support enumeration. "
            f"P1=\n{P1}\nP2=\n{P2}"
        )

    pi1, pi2, V1, V2 = all_solutions[int(np.random.randint(len(all_solutions)))]
    pi1 = np.maximum(pi1, 0.0)
    pi2 = np.maximum(pi2, 0.0)
    s1 = float(pi1.sum())
    s2 = float(pi2.sum())
    if not np.isfinite(s1) or not np.isfinite(s2) or s1 <= 0.0 or s2 <= 0.0:
        raise RuntimeError("Invalid mixed Nash probabilities from support enumeration")
    pi1 = pi1 / s1
    pi2 = pi2 / s2
    return pi1, pi2, float(V1), float(V2)


class NQOVIStagHunt:
    """Nash Q-value Optimistic VI for Stag Hunt."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        horizon: int,
        num_actions: int = NUM_ACTIONS,
        lam: float = 1.0,
        beta: float = 1.0,
        buffer_size: int = 2000,
        reward_scale: float = 4.0,
    ):
        self.H = int(horizon)
        self.lam = float(lam)
        self.beta = float(beta)
        self.num_actions = int(num_actions)
        self.buffer_size = int(buffer_size)
        self.reward_scale = float(reward_scale)
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
        pi1, pi2, _, _ = mixed_nash_support_enumeration(Q1, Q2, self.num_actions)
        a0 = int(np.random.choice(self.num_actions, p=pi1.astype(np.float64)))
        a1 = int(np.random.choice(self.num_actions, p=pi2.astype(np.float64)))
        return a0, a1

    def store_transition(self, h: int, state: State, a0: int, a1: int,
                         next_state: State, r0: float, r1: float,
                         done: bool) -> None:
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
                    Q1_batch, Q2_batch = self._optimistic_Q_batch(next_state_list, h + 1)
                    for local_idx, trans_idx in enumerate(trans_indices):
                        _, _, V1, V2 = mixed_nash_support_enumeration(
                            Q1_batch[local_idx], Q2_batch[local_idx], self.num_actions
                        )
                        V_next_map[trans_idx] = (float(V1), float(V2))
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


def save_agent(agent: NQOVIStagHunt, path: str) -> None:
    checkpoint = {
        "w1": agent.w1, "w2": agent.w2,
        "Lambda": agent.Lambda, "Lambda_inv": agent.Lambda_inv,
        "H": agent.H,
        "lam": agent.lam, "beta": agent.beta,
        "num_actions": agent.num_actions, "buffer_size": agent.buffer_size,
        "feature_dim": agent._feature_dim, "reward_scale": agent.reward_scale,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_agent(path: str) -> NQOVIStagHunt:
    with open(path, "rb") as f:
        ck = pickle.load(f)
    agent = NQOVIStagHunt(
        feature_dim=ck["feature_dim"], num_actions=ck["num_actions"],
        horizon=ck.get("H", 100), lam=ck["lam"], beta=ck["beta"],
        buffer_size=ck["buffer_size"], reward_scale=ck["reward_scale"],
    )
    agent.w1 = ck["w1"]
    agent.w2 = ck["w2"]
    agent.Lambda = ck["Lambda"]
    agent.Lambda_inv = ck["Lambda_inv"]
    return agent
