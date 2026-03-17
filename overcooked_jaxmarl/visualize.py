"""Matplotlib-based visualizer for the JaxMARL Overcooked environment.

Renders State objects using a clean, minimalist style inspired by
src/ppo/cooked_vis.py but extended for the full Overcooked layout
(pots, onion piles, plate piles, serving areas, loose items, etc.).

Usage (standalone):
    python -m overcooked_jaxmarl.visualize          # random rollout
    python -m overcooked_jaxmarl.visualize --layout counter_circuit

Usage (from code):
    from overcooked_jaxmarl.visualize import render_state, animate_rollout
    render_state(state, env)          # single frame
    animate_rollout(states, env, ...)  # gif / mp4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Ellipse, Arc

from .common import OBJECT_TO_INDEX, COLOR_TO_INDEX
from .overcooked import POT_EMPTY_STATUS, POT_FULL_STATUS, POT_READY_STATUS, MAX_ONIONS_IN_POT

# ---------------------------------------------------------------------------
# Theme (extended from src/ppo/cooked_vis.py)
# ---------------------------------------------------------------------------
THEME = {
    # Backgrounds
    "fig_bg":       "#F8F6F0",
    "floor_1":      "#F8F6F0",
    "floor_2":      "#ECEAE4",

    # Furniture
    "counter":      "#E5E7E9",
    "counter_edge": "#BDC3C7",
    "stove":        "#2C3E50",

    # Pot
    "pot_rim":      "#95A5A6",
    "pot_water":    "#D6EAF8",
    "pot_soup":     "#E74C3C",
    "pot_cooking":  "#F39C12",

    # Onion
    "onion_bulb":   "#F4D03F",
    "onion_layer":  "#D68910",
    "onion_spout":  "#58D68D",

    # Plate / Dish
    "plate":        "#FDFEFE",
    "plate_edge":   "#D5D8DC",
    "dish_soup":    "#E74C3C",

    # Serving / Goal
    "goal":         "#2ECC71",
    "goal_edge":    "#27AE60",

    # Onion pile / Plate pile
    "onion_pile":   "#F9E79F",
    "plate_pile":   "#F2F3F4",

    # Agents
    "skin":         "#F5CBA7",
    "hat_brim":     "#FFFFFF",
    "hat_fold":     "#D0D3D4",
    "agent_0":      "#5DADE2",
    "agent_1":      "#AF7AC5",

    # Text
    "text":         "#2C3E50",
}


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_counter(ax, x, y):
    rect = FancyBboxPatch(
        (x, y), 1, 1,
        boxstyle="round,pad=-0.05,rounding_size=0.1",
        fc=THEME["counter"], ec=THEME["counter_edge"], lw=1, zorder=1,
    )
    ax.add_patch(rect)


def _draw_floor(ax, x, y, parity):
    color = THEME["floor_1"] if parity else THEME["floor_2"]
    ax.add_patch(mpatches.Rectangle((x, y), 1, 1, color=color, zorder=0))


def _draw_goal(ax, x, y):
    _draw_counter(ax, x, y)
    inner = FancyBboxPatch(
        (x + 0.1, y + 0.1), 0.8, 0.8,
        boxstyle="round,pad=0,rounding_size=0.1",
        fc=THEME["goal"], ec=THEME["goal_edge"], lw=1.5, alpha=0.7, zorder=2,
    )
    ax.add_patch(inner)
    ax.text(x + 0.5, y + 0.5, "S", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=3)


def _draw_onion_pile(ax, x, y):
    _draw_counter(ax, x, y)
    for dx, dy, s in [(0.3, 0.35, 0.7), (0.7, 0.35, 0.7), (0.5, 0.65, 0.7)]:
        _draw_onion(ax, x + dx, y + dy, scale=s)


def _draw_plate_pile(ax, x, y):
    _draw_counter(ax, x, y)
    for dx, dy in [(0.35, 0.3), (0.65, 0.45), (0.45, 0.7)]:
        c = Circle((x + dx, y + dy), 0.13, fc=THEME["plate"], ec=THEME["plate_edge"], lw=0.8, zorder=5)
        ax.add_patch(c)


def _draw_onion(ax, cx, cy, scale=1.0, zorder=15):
    r = 0.14 * scale
    bulb = Ellipse((cx, cy), r * 2.2, r * 2.0, fc=THEME["onion_bulb"], zorder=zorder)
    ax.add_patch(bulb)
    arc1 = Arc((cx, cy), r * 1.5, r * 1.5, theta1=90, theta2=270,
               color=THEME["onion_layer"], lw=scale * 0.8, alpha=0.6, zorder=zorder + 1)
    arc2 = Arc((cx, cy), r * 1.5, r * 1.5, theta1=-90, theta2=90,
               color=THEME["onion_layer"], lw=scale * 0.8, alpha=0.6, zorder=zorder + 1)
    ax.add_patch(arc1)
    ax.add_patch(arc2)
    spout = Rectangle((cx - 0.02 * scale, cy + r * 0.7), 0.04 * scale, 0.08 * scale,
                       fc=THEME["onion_spout"], zorder=zorder - 1)
    ax.add_patch(spout)


def _draw_plate(ax, cx, cy, zorder=15):
    c = Circle((cx, cy), 0.16, fc=THEME["plate"], ec=THEME["plate_edge"], lw=1, zorder=zorder)
    ax.add_patch(c)


def _draw_dish(ax, cx, cy, zorder=15):
    """Plate with soup on it."""
    _draw_plate(ax, cx, cy, zorder=zorder)
    soup = Circle((cx, cy), 0.10, fc=THEME["dish_soup"], alpha=0.85, zorder=zorder + 1)
    ax.add_patch(soup)


def _draw_pot(ax, x, y, pot_status):
    """Draw pot on a stove tile. pot_status encodes the JaxMARL pot state."""
    # Stove background
    rect = FancyBboxPatch(
        (x, y), 1, 1,
        boxstyle="round,pad=-0.05,rounding_size=0.1",
        fc=THEME["stove"], ec="none", zorder=1,
    )
    ax.add_patch(rect)

    cx, cy = x + 0.5, y + 0.5

    n_onions = 0
    is_cooking = False
    is_ready = False

    if pot_status == POT_EMPTY_STATUS:
        n_onions = 0
    elif pot_status > POT_FULL_STATUS:
        n_onions = POT_EMPTY_STATUS - pot_status  # 1, 2, or 3
    elif pot_status == POT_READY_STATUS:
        is_ready = True
        n_onions = 3
    else:
        is_cooking = True
        n_onions = 3

    # Pot body
    if is_ready:
        liquid_color = THEME["pot_soup"]
    elif is_cooking:
        liquid_color = THEME["pot_cooking"]
    else:
        liquid_color = THEME["pot_water"]

    body = Circle((cx, cy), 0.35, fc=THEME["pot_rim"], ec="#7F8C8D", lw=1, zorder=10)
    ax.add_patch(body)
    contents = Circle((cx, cy), 0.28, fc=liquid_color, alpha=0.9, zorder=11)
    ax.add_patch(contents)

    # Handles
    ax.add_patch(Rectangle((cx - 0.42, cy - 0.08), 0.1, 0.16, fc=THEME["pot_rim"], zorder=9))
    ax.add_patch(Rectangle((cx + 0.32, cy - 0.08), 0.1, 0.16, fc=THEME["pot_rim"], zorder=9))

    # Onions inside
    onion_offsets = [(-0.1, 0.05), (0.1, 0.05), (0.0, -0.1)]
    for i in range(min(n_onions, 3)):
        dx, dy = onion_offsets[i]
        _draw_onion(ax, cx + dx, cy + dy, scale=0.55, zorder=12)

    # Steam if cooking
    if is_cooking:
        for dx, dy, r in [(-0.1, 0.2, 0.04), (0.1, 0.25, 0.03), (0.0, 0.18, 0.035)]:
            ax.add_patch(Circle((cx + dx, cy + dy), r, fc="white", alpha=0.5, zorder=13))

    # Cooking progress bar
    if is_cooking:
        bar_w = 0.6
        bar_h = 0.06
        bx = cx - bar_w / 2
        by = y + 0.08
        ax.add_patch(Rectangle((bx, by), bar_w, bar_h, fc="#555555", alpha=0.4, zorder=14))
        progress = 1.0 - pot_status / POT_FULL_STATUS
        ax.add_patch(Rectangle((bx, by), bar_w * progress, bar_h, fc="#2ECC71", zorder=15))

    # "READY" label
    if is_ready:
        ax.text(cx, y + 0.12, "READY", ha="center", va="center",
                fontsize=5, fontweight="bold", color="#2ECC71", zorder=15)


def _draw_chef(ax, cx, cy, agent_idx, holding_item=None):
    color = THEME[f"agent_{agent_idx}"]

    # Body
    body = Circle((cx, cy), 0.35, fc=color, ec="white", lw=0.5, zorder=20)
    ax.add_patch(body)

    # Hands
    hand_color = THEME["skin"]
    hand_r = 0.07

    if holding_item is not None:
        # Hands forward (below body)
        ax.add_patch(Circle((cx - 0.12, cy - 0.22), hand_r, fc=hand_color, zorder=25))
        ax.add_patch(Circle((cx + 0.12, cy - 0.22), hand_r, fc=hand_color, zorder=25))
        # Draw held item
        if holding_item == OBJECT_TO_INDEX["onion"]:
            _draw_onion(ax, cx, cy - 0.28, scale=0.8, zorder=26)
        elif holding_item == OBJECT_TO_INDEX["plate"]:
            _draw_plate(ax, cx, cy - 0.28, zorder=26)
        elif holding_item == OBJECT_TO_INDEX["dish"]:
            _draw_dish(ax, cx, cy - 0.28, zorder=26)
    else:
        # Hands at sides
        ax.add_patch(Circle((cx - 0.35, cy), hand_r, fc=hand_color, zorder=21))
        ax.add_patch(Circle((cx + 0.35, cy), hand_r, fc=hand_color, zorder=21))

    # Head
    head = Circle((cx, cy + 0.08), 0.22, fc=THEME["skin"], zorder=22)
    ax.add_patch(head)

    # Chef hat
    hat_z = 23
    ax.add_patch(Rectangle((cx - 0.17, cy + 0.13), 0.34, 0.12, fc=THEME["hat_brim"], zorder=hat_z))
    for i in range(4):
        offset = (i - 1.5) * 0.07
        pleat = Ellipse(
            (cx + offset, cy + 0.28), 0.10, 0.20,
            fc=THEME["hat_brim"], ec=THEME["hat_fold"], lw=0.4, zorder=hat_z + 1,
        )
        ax.add_patch(pleat)

    # Agent label
    ax.text(cx, cy - 0.02, str(agent_idx), ha="center", va="center",
            fontsize=7, fontweight="bold", color="white", zorder=24)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_state(state, env, ax=None, step_num=0, total_reward=0.0, title_extra=""):
    """Render a single JaxMARL Overcooked State to a matplotlib Axes.

    Args:
        state: overcooked_jaxmarl.overcooked.State (flax dataclass)
        env: Overcooked environment instance (for layout dimensions)
        ax: matplotlib Axes (created if None)
        step_num: current timestep (for HUD)
        total_reward: cumulative reward (for HUD)
        title_extra: extra string appended to title
    Returns:
        fig, ax
    """
    w = env.width
    h = env.height
    padding = (state.maze_map.shape[0] - h) // 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(w * 1.2, 5), max(h * 1.2, 4)))
        fig.patch.set_facecolor(THEME["fig_bg"])
    else:
        fig = ax.figure

    ax.clear()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Extract the unpadded maze_map: shape (h, w, 3)
    maze = np.array(state.maze_map[padding: padding + h, padding: padding + w])
    agent_pos = np.array(state.agent_pos)   # (2, 2) — (x, y)
    agent_inv = np.array(state.agent_inv)   # (2,)
    pot_pos = np.array(state.pot_pos)       # (n_pots, 2) — (x, y)

    # Build a set of pot positions for quick lookup
    pot_set = set()
    for p in pot_pos:
        pot_set.add((int(p[0]), int(p[1])))

    # Build agent position set
    agent_cells = {}
    for i in range(2):
        agent_cells[(int(agent_pos[i, 0]), int(agent_pos[i, 1]))] = i

    # Render grid cell by cell (y=0 is top row in maze, but bottom in matplotlib)
    for row in range(h):
        for col in range(w):
            draw_x = col
            draw_y = h - 1 - row  # flip y for matplotlib

            obj_type = int(maze[row, col, 0])
            obj_status = int(maze[row, col, 2])

            # Check if this is an agent cell (agents overlay floor)
            cell_key = (col, row)

            if cell_key in agent_cells:
                # Floor under agent
                _draw_floor(ax, draw_x, draw_y, (row + col) % 2 == 0)
                idx = agent_cells[cell_key]
                inv = int(agent_inv[idx])
                held = inv if inv != OBJECT_TO_INDEX["empty"] else None
                _draw_chef(ax, draw_x + 0.5, draw_y + 0.5, idx, holding_item=held)

            elif cell_key in pot_set:
                _draw_pot(ax, draw_x, draw_y, obj_status)

            elif obj_type == OBJECT_TO_INDEX["wall"]:
                _draw_counter(ax, draw_x, draw_y)

            elif obj_type == OBJECT_TO_INDEX["empty"]:
                _draw_floor(ax, draw_x, draw_y, (row + col) % 2 == 0)

            elif obj_type == OBJECT_TO_INDEX["goal"]:
                _draw_goal(ax, draw_x, draw_y)

            elif obj_type == OBJECT_TO_INDEX["onion_pile"]:
                _draw_onion_pile(ax, draw_x, draw_y)

            elif obj_type == OBJECT_TO_INDEX["plate_pile"]:
                _draw_plate_pile(ax, draw_x, draw_y)

            elif obj_type == OBJECT_TO_INDEX["onion"]:
                # Loose onion on counter
                _draw_counter(ax, draw_x, draw_y)
                _draw_onion(ax, draw_x + 0.5, draw_y + 0.5, scale=1.0)

            elif obj_type == OBJECT_TO_INDEX["plate"]:
                _draw_counter(ax, draw_x, draw_y)
                _draw_plate(ax, draw_x + 0.5, draw_y + 0.5)

            elif obj_type == OBJECT_TO_INDEX["dish"]:
                _draw_counter(ax, draw_x, draw_y)
                _draw_dish(ax, draw_x + 0.5, draw_y + 0.5)

            elif obj_type == OBJECT_TO_INDEX["agent"]:
                # Agent drawn from agent_cells above; this shouldn't happen
                # but handle gracefully
                _draw_floor(ax, draw_x, draw_y, (row + col) % 2 == 0)

            else:
                _draw_floor(ax, draw_x, draw_y, (row + col) % 2 == 0)

    # HUD
    title = f"Step {step_num}  |  Reward {total_reward:.0f}"
    if title_extra:
        title += f"  |  {title_extra}"
    ax.set_title(title, fontsize=11, fontweight="bold", color=THEME["text"], pad=8)

    return fig, ax


# ---------------------------------------------------------------------------
# Animate a sequence of states
# ---------------------------------------------------------------------------

def animate_rollout(state_seq, env, filename="overcooked.gif", fps=5):
    """Create a GIF from a list of State objects.

    Args:
        state_seq: list of (state, reward) tuples or just list of states
        env: Overcooked environment instance
        filename: output path (.gif or .mp4)
        fps: frames per second
    """
    fig, ax = plt.subplots(figsize=(max(env.width * 1.2, 5), max(env.height * 1.2, 4)))
    fig.patch.set_facecolor(THEME["fig_bg"])

    cum_reward = 0.0

    def _update(frame_idx):
        nonlocal cum_reward
        item = state_seq[frame_idx]
        if isinstance(item, tuple):
            state, reward = item
            cum_reward += float(reward)
        else:
            state = item
        render_state(state, env, ax=ax, step_num=frame_idx, total_reward=cum_reward)

    anim = animation.FuncAnimation(fig, _update, frames=len(state_seq), interval=1000 // fps)
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved animation to {filename}")


# ---------------------------------------------------------------------------
# CLI: run a random rollout and visualize
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize JaxMARL Overcooked")
    parser.add_argument("--layout", type=str, default="cramped_room",
                        choices=["cramped_room", "asymm_advantages", "coord_ring",
                                 "forced_coord", "counter_circuit", "mini_circuit"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="Save gif to path")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from .overcooked import Overcooked, Actions
    from .layouts import overcooked_layouts

    layout = overcooked_layouts[args.layout]
    env = Overcooked(layout=layout, max_steps=args.steps)

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)

    state_seq = [(state, 0.0)]

    for t in range(args.steps):
        key, k1, k2 = jax.random.split(key, 3)
        a0 = jax.random.randint(k1, (), 0, 6)
        a1 = jax.random.randint(k2, (), 0, 6)
        actions = {"agent_0": a0, "agent_1": a1}
        obs, state, rewards, dones, info = env.step(key, state, actions)
        state_seq.append((state, float(rewards["agent_0"])))
        if dones["__all__"]:
            break

    if args.save:
        animate_rollout(state_seq, env, filename=args.save, fps=args.fps)
    else:
        # Interactive: show frames one by one
        fig, ax = plt.subplots(figsize=(max(env.width * 1.2, 5), max(env.height * 1.2, 4)))
        fig.patch.set_facecolor(THEME["fig_bg"])
        plt.ion()
        cum_reward = 0.0
        for i, (s, r) in enumerate(state_seq):
            cum_reward += r
            render_state(s, env, ax=ax, step_num=i, total_reward=cum_reward)
            plt.draw()
            plt.pause(0.25)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
