"""
Animate the linear search simulation as a GIF.

This script mirrors `linear_search_main.py` (same default params), but produces an
animation of the drone moving smoothly along the planned trajectory, pausing to
scan at each search center, and recharging when required. A battery/energy bar
shows remaining energy during each sortie (resets to full on recharge).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

# We intentionally reuse the same internal helpers as the simulation so that:
# - the flight path is the same cubic profile
# - the energy bookkeeping matches the report/plots
from linear_search import (  # type: ignore
    _center_xy,
    _return_energy,
    _search_energy,
    _segment_energy_and_path,
    _segment_energy_only,
    _unit,
    max_distance_one_search,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"


EventType = Literal["move", "scan", "recharge"]


@dataclass(frozen=True)
class Event:
    type: EventType
    duration_s: float
    # Move-specific
    path_xy: Optional[np.ndarray] = None  # (N, 2)
    # Scan-specific
    center_xy: Optional[np.ndarray] = None  # (2,)
    scan_index: Optional[int] = None
    found_after: Optional[bool] = None
    # Recharge-specific
    recharge_count: Optional[int] = None
    # Energy bookkeeping (per-sortie; resets on recharge)
    e_used_start: float = 0.0
    e_used_end: float = 0.0


def _simulate_linear_search_with_events(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-run the linear-search logic while logging move/scan/recharge events.

    Returns a dict compatible with the plot in `plotting.plot_linear_search_area`,
    plus an `events` list and additional bookkeeping used for animation.
    """
    u = _unit(np.array(params["direction"], dtype=float))
    base = np.array([params["x0"], params["y0"]], dtype=float)
    R = float(params["scan_radius"])
    step = 2.0 * R

    # Centers along the vector; enforce tangency by fixed step and ensure end is included.
    s_centers = list(np.arange(0.0, float(params["vector_length"]) + 1e-12, step))
    if s_centers[-1] < float(params["vector_length"]) - 1e-12:
        s_centers.append(float(params["vector_length"]))
    all_centers_xy = np.stack([_center_xy(params["x0"], params["y0"], u, s) for s in s_centers], axis=0)

    rng = np.random.default_rng(params.get("seed"))
    vector_length = float(params["vector_length"])
    target_s_override = params.get("target_s", None)
    if target_s_override is not None:
        s_target = float(target_s_override)
        s_target = float(np.clip(s_target, 0.0, vector_length))
    else:
        # For the animation demo it’s nicer if the target is not immediately found at the base.
        # The original simulation samples s_target ~ Uniform(0, L); we only bias this when
        # `avoid_immediate_find` is True (default).
        avoid_immediate_find = bool(params.get("avoid_immediate_find", True))
        low = 0.0
        if avoid_immediate_find:
            low = min(vector_length, float(params["scan_radius"]) * 1.05)
        if vector_length <= low + 1e-12:
            s_target = float(rng.uniform(0.0, vector_length))
        else:
            s_target = float(rng.uniform(low, vector_length))
    target_xy_arr = _center_xy(params["x0"], params["y0"], u, s_target)
    target_xy: Tuple[float, float] = tuple(target_xy_arr.tolist())

    max_one_search = max_distance_one_search(params)
    e_scan = _search_energy(EH=params["EH"], search_time=params["search_time"])

    e_used = 0.0
    recharge_count = 0
    searched_centers_xy: List[np.ndarray] = []
    events: List[Event] = []
    sortie_paths: List[np.ndarray] = []

    found = False
    found_at_index: Optional[int] = None

    def log_move(p_from: np.ndarray, p_to: np.ndarray) -> None:
        nonlocal e_used
        de, path = _segment_energy_and_path(
            p_from,
            p_to,
            T=params["move_time"],
            m=params["mass"],
            EH=params["EH"],
            samples=70,
        )
        e0 = float(e_used)
        e_used += float(de)
        e1 = float(e_used)
        events.append(
            Event(
                type="move",
                duration_s=float(params["move_time"]),
                path_xy=path,
                e_used_start=e0,
                e_used_end=e1,
            )
        )
        sortie_paths.append(path)

    def can_do_next_from(p_now: np.ndarray, p_next: np.ndarray) -> bool:
        e_left = float(params["e_max"]) - float(e_used)
        e_move = _segment_energy_only(p_now, p_next, T=params["move_time"], m=params["mass"], EH=params["EH"])
        e_back = _return_energy(p_next, base=base, T=params["move_time"], m=params["mass"], EH=params["EH"])
        need = float(e_move) + float(e_scan) + float(e_back) + float(params["eps"])
        return e_left >= need

    def do_scan(p: np.ndarray, idx: int) -> None:
        nonlocal e_used, found, found_at_index
        e0 = float(e_used)
        e_used += float(e_scan)
        e1 = float(e_used)

        searched_centers_xy.append(p.copy())
        if np.hypot(p[0] - target_xy[0], p[1] - target_xy[1]) <= R + 1e-12:
            found = True
            found_at_index = idx

        events.append(
            Event(
                type="scan",
                duration_s=float(params["search_time"]),
                center_xy=p.copy(),
                scan_index=len(searched_centers_xy) - 1,
                found_after=bool(found),
                e_used_start=e0,
                e_used_end=e1,
            )
        )

    def do_recharge() -> None:
        nonlocal e_used, recharge_count
        recharge_count += 1
        events.append(
            Event(
                type="recharge",
                duration_s=float(params.get("recharge_pause_s", 0.6)),
                center_xy=base.copy(),
                recharge_count=recharge_count,
                e_used_start=float(e_used),
                e_used_end=0.0,
            )
        )
        e_used = 0.0

    # Start at base; initial scan.
    current_idx = 0
    current_pos = all_centers_xy[current_idx].copy()
    do_scan(current_pos, current_idx)

    while not found and current_idx < len(all_centers_xy) - 1:
        next_idx = current_idx + 1
        next_pos = all_centers_xy[next_idx]

        if not can_do_next_from(current_pos, next_pos):
            # Check if even a fresh battery (minus any reposition from base) can do the next step.
            e_reposition = 0.0 if current_idx == 0 else _segment_energy_only(
                base, current_pos, T=params["move_time"], m=params["mass"], EH=params["EH"]
            )
            e_need_from_current = (
                _segment_energy_only(current_pos, next_pos, T=params["move_time"], m=params["mass"], EH=params["EH"])
                + e_scan
                + _return_energy(next_pos, base=base, T=params["move_time"], m=params["mass"], EH=params["EH"])
                + float(params["eps"])
            )
            if float(params["e_max"]) - float(e_reposition) < float(e_need_from_current):
                break

            # Return to base, recharge, and reposition to the last scanned center (without re-scan).
            log_move(current_pos, base)
            do_recharge()
            if current_idx != 0:
                log_move(base, current_pos)
            continue

        log_move(current_pos, next_pos)
        current_idx = next_idx
        current_pos = next_pos.copy()
        do_scan(current_pos, current_idx)

    # If target is found away from base, return to base (charging station).
    # This is always feasible because each move+scan decision budgets energy for returning to base.
    if found and not np.allclose(current_pos, base):
        log_move(current_pos, base)

    return {
        "target_xy": target_xy,
        "found": found,
        "found_at_index": found_at_index,
        "searched_centers_xy": np.stack(searched_centers_xy, axis=0) if searched_centers_xy else np.zeros((0, 2)),
        "all_centers_xy": all_centers_xy,
        "sortie_paths": sortie_paths,
        "recharge_count": recharge_count,
        "max_one_search_distance": max_one_search,
        "events": events,
        "base_xy": tuple(base.tolist()),
    }


def _energy_color(frac: float) -> str:
    if frac <= 0.15:
        return "tab:red"
    if frac <= 0.4:
        return "tab:orange"
    return "tab:green"


def _make_drone_icon(
    ax: plt.Axes,
    *,
    size: float,
    color: str = "tab:purple",
    rotor_color: str = "0.25",
    zorder: int = 9,
) -> Dict[str, Any]:
    """
    Create a simple vector "drone" icon: body + 4 rotors + arms.

    Returns a dict of artists to update each frame.
    """
    s = float(size)
    # Geometry (in data units)
    body_r = 0.28 * s
    rotor_r = 0.20 * s
    rotor_d = 0.70 * s

    body = patches.Circle((0.0, 0.0), body_r, fc=color, ec="none", alpha=0.95, zorder=zorder)
    rotors = [
        patches.Circle((+rotor_d, +rotor_d), rotor_r, fc="none", ec=rotor_color, lw=1.6, alpha=0.95, zorder=zorder),
        patches.Circle((+rotor_d, -rotor_d), rotor_r, fc="none", ec=rotor_color, lw=1.6, alpha=0.95, zorder=zorder),
        patches.Circle((-rotor_d, +rotor_d), rotor_r, fc="none", ec=rotor_color, lw=1.6, alpha=0.95, zorder=zorder),
        patches.Circle((-rotor_d, -rotor_d), rotor_r, fc="none", ec=rotor_color, lw=1.6, alpha=0.95, zorder=zorder),
    ]
    arms = [
        Line2D([0.0, +rotor_d], [0.0, +rotor_d], color=rotor_color, lw=1.4, alpha=0.95, zorder=zorder),
        Line2D([0.0, +rotor_d], [0.0, -rotor_d], color=rotor_color, lw=1.4, alpha=0.95, zorder=zorder),
        Line2D([0.0, -rotor_d], [0.0, +rotor_d], color=rotor_color, lw=1.4, alpha=0.95, zorder=zorder),
        Line2D([0.0, -rotor_d], [0.0, -rotor_d], color=rotor_color, lw=1.4, alpha=0.95, zorder=zorder),
    ]

    ax.add_patch(body)
    for r in rotors:
        ax.add_patch(r)
    for a in arms:
        ax.add_line(a)

    return {"body": body, "rotors": rotors, "arms": arms, "rotor_d": rotor_d}


def _move_drone_icon(drone: Dict[str, Any], xy: Tuple[float, float]) -> None:
    x, y = float(xy[0]), float(xy[1])
    rotor_d = float(drone["rotor_d"])
    drone["body"].center = (x, y)
    centers = [(x + rotor_d, y + rotor_d), (x + rotor_d, y - rotor_d), (x - rotor_d, y + rotor_d), (x - rotor_d, y - rotor_d)]
    for r, c in zip(drone["rotors"], centers):
        r.center = c
    for a, c in zip(drone["arms"], centers):
        a.set_data([x, c[0]], [y, c[1]])


def _build_frames(
    *,
    events: List[Event],
    e_max: float,
    scan_radius: float,
    fps: int,
    min_move_frames: int = 6,
    min_scan_frames: int = 6,
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    scanned_count = 0
    recharge_count = 0
    current_xy = np.array([0.0, 0.0], dtype=float)
    t = 0.0

    for ev in events:
        n_frames = max(1, int(round(float(ev.duration_s) * fps)))
        if ev.type == "move":
            n_frames = max(n_frames, int(min_move_frames))
            assert ev.path_xy is not None
            path = ev.path_xy
            idxs = np.linspace(0, len(path) - 1, n_frames).round().astype(int)
            for k, i in enumerate(idxs):
                a = 0.0 if n_frames == 1 else (k / (n_frames - 1))
                e_used = (1.0 - a) * float(ev.e_used_start) + a * float(ev.e_used_end)
                current_xy = path[i].copy()
                frames.append(
                    {
                        "xy": current_xy.copy(),
                        "t": t + a * float(ev.duration_s),
                        "scanned_count": scanned_count,
                        "scan_progress": None,
                        "e_remaining": float(e_max) - float(e_used),
                        "recharge_count": recharge_count,
                        "stage": "move",
                    }
                )
            t += float(ev.duration_s)

        elif ev.type == "scan":
            n_frames = max(n_frames, int(min_scan_frames))
            assert ev.center_xy is not None
            center = ev.center_xy
            for k in range(n_frames):
                a = 0.0 if n_frames == 1 else (k / (n_frames - 1))
                e_used = (1.0 - a) * float(ev.e_used_start) + a * float(ev.e_used_end)
                current_xy = center.copy()
                frames.append(
                    {
                        "xy": current_xy.copy(),
                        "t": t + a * float(ev.duration_s),
                        "scanned_count": scanned_count,
                        "scan_progress": float(a),
                        "e_remaining": float(e_max) - float(e_used),
                        "recharge_count": recharge_count,
                        "stage": "scan",
                    }
                )
            scanned_count += 1
            t += float(ev.duration_s)

        elif ev.type == "recharge":
            assert ev.center_xy is not None
            center = ev.center_xy
            recharge_count = int(ev.recharge_count or (recharge_count + 1))
            for k in range(n_frames):
                a = 0.0 if n_frames == 1 else (k / (n_frames - 1))
                # Show energy ramping back up for visual clarity (even though recharge is instantaneous).
                e_used = (1.0 - a) * float(ev.e_used_start) + a * float(ev.e_used_end)
                current_xy = center.copy()
                frames.append(
                    {
                        "xy": current_xy.copy(),
                        "t": t + a * float(ev.duration_s),
                        "scanned_count": scanned_count,
                        "scan_progress": None,
                        "e_remaining": float(e_max) - float(e_used),
                        "recharge_count": recharge_count,
                        "stage": "recharge",
                    }
                )
            t += float(ev.duration_s)

        else:
            raise ValueError(f"unknown event type: {ev.type}")

    # Attach scan radius for convenience (used by scan indicator).
    for fr in frames:
        fr["scan_radius"] = float(scan_radius)
    return frames


def animate_linear_search(
    *,
    params: Dict[str, Any],
    savepath: str,
    fps: int = 30,
    dpi: int = 160,
    trail_max_points: int = 900,
) -> None:
    result = _simulate_linear_search_with_events(params)

    all_centers_xy = result["all_centers_xy"]
    searched_centers_xy = result["searched_centers_xy"]
    base_xy = tuple(result["base_xy"])
    vector_end_xy = tuple(all_centers_xy[-1].tolist())
    target_xy = result["target_xy"]
    found = bool(result["found"])
    events: List[Event] = result["events"]

    frames = _build_frames(
        events=events,
        e_max=float(params["e_max"]),
        scan_radius=float(params["scan_radius"]),
        fps=int(fps),
    )
    if len(frames) == 0:
        raise RuntimeError("No frames produced (empty simulation).")

    # --- Figure / axes ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=int(dpi))

    # Vector array (faint) and base/end/target markers.
    ax.plot(
        all_centers_xy[:, 0],
        all_centers_xy[:, 1],
        ".-",
        color="0.80",
        lw=1.2,
        ms=4,
        zorder=0,
        label="Vector array",
    )
    ax.scatter([base_xy[0]], [base_xy[1]], s=90, c="green", marker="*", zorder=6, label="Base")
    ax.scatter([vector_end_xy[0]], [vector_end_xy[1]], s=70, c="black", marker="x", zorder=6, label="Vector end")
    ax.scatter(
        [target_xy[0]],
        [target_xy[1]],
        s=65,
        c=("tab:green" if found else "tab:red"),
        marker="o",
        zorder=6,
        label=("Target (found)" if found else "Target (not found)"),
    )

    # Optional max one-search distance marker.
    max_one = float(result["max_one_search_distance"] or 0.0)
    if max_one > 0:
        base = np.array(base_xy, dtype=float)
        end = np.array(vector_end_xy, dtype=float)
        d = float(np.hypot(*(end - base)))
        if d > 0:
            u = (end - base) / d
            p = base + max_one * u
            ax.scatter([p[0]], [p[1]], s=55, c="tab:orange", marker="D", zorder=6, label="Max 1-search distance")

    # Pre-create scan circle patches (invisible until scanned).
    scan_radius = float(params["scan_radius"])
    scan_fill_patches: List[patches.Circle] = []
    scan_edge_patches: List[patches.Circle] = []
    for c in searched_centers_xy:
        fill = patches.Circle(
            (float(c[0]), float(c[1])),
            scan_radius,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.0,
            lw=1.0,
            zorder=2,
        )
        edge = patches.Circle(
            (float(c[0]), float(c[1])),
            scan_radius,
            facecolor="none",
            edgecolor="tab:blue",
            alpha=0.0,
            lw=1.1,
            zorder=3,
        )
        ax.add_patch(fill)
        ax.add_patch(edge)
        scan_fill_patches.append(fill)
        scan_edge_patches.append(edge)

    # Flight trail and drone icon.
    trail_line, = ax.plot([], [], color="0.55", lw=1.6, alpha=0.9, zorder=1, label="Flight path")
    # Icon size chosen relative to scan radius so it scales with the scene.
    drone_icon = _make_drone_icon(ax, size=0.42 * scan_radius, color="tab:purple", rotor_color="0.25", zorder=9)
    # Dummy handle for legend entry.
    ax.plot([], [], color="tab:purple", lw=0, marker="o", ms=6, label="Drone")

    # Scan "sweep" indicator: expanding ring at the current scan center.
    scan_indicator = patches.Circle((0.0, 0.0), 0.0, fill=False, edgecolor="tab:cyan", lw=2.0, alpha=0.85, zorder=8)
    ax.add_patch(scan_indicator)

    # Energy bar (axes coordinates).
    bar_x, bar_y = 0.03, 0.95
    bar_w, bar_h = 0.33, 0.028
    bar_bg = patches.Rectangle((bar_x, bar_y - bar_h), bar_w, bar_h, transform=ax.transAxes, fc="0.93", ec="0.55", lw=1.0)
    bar_fg = patches.Rectangle((bar_x, bar_y - bar_h), bar_w, bar_h, transform=ax.transAxes, fc="tab:green", ec="none", lw=0.0)
    ax.add_patch(bar_bg)
    ax.add_patch(bar_fg)
    hud_text = ax.text(
        0.03,
        0.03,
        "",
        transform=ax.transAxes,
        fontsize=10.5,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.85, boxstyle="round,pad=0.25"),
        zorder=10,
    )

    # Bounds with margin (match plotting.py logic).
    pts = [all_centers_xy]
    if len(searched_centers_xy) > 0:
        pts.append(searched_centers_xy)
    pts.append(np.array([[base_xy[0], base_xy[1]], [vector_end_xy[0], vector_end_xy[1]], [target_xy[0], target_xy[1]]]))
    all_pts = np.vstack(pts)
    x_min, y_min = np.min(all_pts, axis=0)
    x_max, y_max = np.max(all_pts, axis=0)
    margin = 0.15 * max(x_max - x_min, y_max - y_min, 1.0) + scan_radius
    ax.set_xlim(float(x_min - margin), float(x_max + margin))
    ax.set_ylim(float(y_min - margin), float(y_max + margin))

    title = "Linear search animation: smooth flight + scan pauses"
    subtitle = f"searches={len(searched_centers_xy)}, recharges={result['recharge_count']}, found={found}"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    # Trail buffers.
    trail_x: List[float] = []
    trail_y: List[float] = []

    def init() -> List[Any]:
        trail_line.set_data([], [])
        _move_drone_icon(drone_icon, (base_xy[0], base_xy[1]))
        scan_indicator.set_radius(0.0)
        scan_indicator.set_alpha(0.0)
        bar_fg.set_width(0.0)
        hud_text.set_text("")
        for p in scan_fill_patches + scan_edge_patches:
            p.set_alpha(0.0)
        icon_artists: List[Any] = [drone_icon["body"]] + drone_icon["rotors"] + drone_icon["arms"]
        return [trail_line, scan_indicator, bar_fg, hud_text] + icon_artists + scan_fill_patches + scan_edge_patches

    def update(frame_idx: int) -> List[Any]:
        fr = frames[frame_idx]
        x, y = float(fr["xy"][0]), float(fr["xy"][1])

        trail_x.append(x)
        trail_y.append(y)
        if len(trail_x) > int(trail_max_points):
            del trail_x[: len(trail_x) - int(trail_max_points)]
            del trail_y[: len(trail_y) - int(trail_max_points)]
        trail_line.set_data(trail_x, trail_y)

        _move_drone_icon(drone_icon, (x, y))

        # Reveal completed scan circles, and partially reveal the in-progress scan.
        scanned_count = int(fr["scanned_count"])
        scan_progress = fr["scan_progress"]
        for i in range(len(scan_fill_patches)):
            if i < scanned_count:
                scan_fill_patches[i].set_alpha(0.14)
                scan_edge_patches[i].set_alpha(0.55)
            elif fr["stage"] == "scan" and scan_progress is not None and i == scanned_count:
                a = float(scan_progress)
                scan_fill_patches[i].set_alpha(0.14 * a)
                scan_edge_patches[i].set_alpha(0.55 * a)
            else:
                scan_fill_patches[i].set_alpha(0.0)
                scan_edge_patches[i].set_alpha(0.0)

        # Scan indicator.
        if fr["stage"] == "scan" and fr["scan_progress"] is not None:
            a = float(fr["scan_progress"])
            scan_indicator.center = (x, y)
            scan_indicator.set_radius(a * float(fr["scan_radius"]))
            scan_indicator.set_alpha(0.9 * (1.0 - 0.35 * a))
        else:
            scan_indicator.set_alpha(0.0)
            scan_indicator.set_radius(0.0)

        # Energy bar.
        e_remaining = max(0.0, float(fr["e_remaining"]))
        frac = 0.0 if float(params["e_max"]) <= 0 else min(1.0, e_remaining / float(params["e_max"]))
        bar_fg.set_width(bar_w * frac)
        bar_fg.set_facecolor(_energy_color(frac))

        hud_text.set_text(
            "t={:.2f}s   stage={}   scans={}/{}   recharges={}\nE remaining: {:.2f} / {:.2f}".format(
                float(fr["t"]),
                fr["stage"],
                scanned_count,
                len(searched_centers_xy),
                int(fr.get("recharge_count", 0)),
                e_remaining,
                float(params["e_max"]),
            )
        )

        icon_artists = [drone_icon["body"]] + drone_icon["rotors"] + drone_icon["arms"]
        return [trail_line, scan_indicator, bar_fg, hud_text] + icon_artists + scan_fill_patches + scan_edge_patches

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=int(round(1000 / max(1, int(fps)))),
        blit=True,
    )

    savepath = str(savepath)
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    anim.save(savepath, writer=PillowWriter(fps=int(fps)))
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a GIF animation of the linear search demo.")
    p.add_argument("--out", type=str, default=str(PLOTS_DIR / "linear_search_animation.gif"), help="Output GIF path.")
    p.add_argument("--fps", type=int, default=20, help="Frames per second.")
    p.add_argument("--dpi", type=int, default=110, help="Figure DPI.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for target placement.")
    p.add_argument(
        "--target-s",
        type=float,
        default=None,
        help="Override target position along the vector (distance from base). If omitted, uses RNG.",
    )
    p.add_argument("--vector-length", type=float, default=6.0, help="Length of the search vector.")
    p.add_argument("--scan-radius", type=float, default=0.6, help="Scan radius.")
    p.add_argument("--e-max", type=float, default=20.0, help="Max energy per sortie.")
    p.add_argument("--eps", type=float, default=5e-2, help="Energy margin epsilon.")
    p.add_argument("--move-time", type=float, default=2.5, help="Time per move segment (s).")
    p.add_argument("--search-time", type=float, default=0.2, help="Time per scan (s).")
    p.add_argument("--recharge-pause", type=float, default=0.4, help="Pause at base when recharging (s).")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    params = {
        "x0": 0.0,
        "y0": 0.0,
        "direction": (1.0, 0.35),
        "vector_length": float(args.vector_length),
        "scan_radius": float(args.scan_radius),
        "e_max": float(args.e_max),
        "eps": float(args.eps),
        "move_time": float(args.move_time),
        "mass": 1.0,
        "EH": 1.0,
        "search_time": float(args.search_time),
        "seed": int(args.seed),
        "target_s": args.target_s,
        "recharge_pause_s": float(args.recharge_pause),
    }

    animate_linear_search(params=params, savepath=str(args.out), fps=int(args.fps), dpi=int(args.dpi))
    print("Saved:", str(args.out))


if __name__ == "__main__":
    main()

