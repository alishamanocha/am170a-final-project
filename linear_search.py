"""
Linear search simulation along a vector with tangential scan circles.

This module implements a "linear search" strategy:
- The drone searches at centers along a ray/segment, spaced by the scan diameter (2R),
  so consecutive scan circles meet tangentially.
- Before committing to the next move+search, the drone predicts whether it can:
    move to the next center, perform the search, and still return to base.
  If not, it returns to base, recharges, and resumes from the last searched center
  (returning there without re-searching).

Energy model:
- Uses the existing quadratic-profile flight model in `drone_sim.forward_odes`.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Kamran Hussain
"""

from typing import List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    if n == 0.0:
        raise ValueError("direction vector must be non-zero")
    return v / n


def _center_xy(x0: float, y0: float, u: np.ndarray, s: float) -> np.ndarray:
    return np.array([x0, y0], dtype=float) + s * u


def _segment_energy_only(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    T: float,
    m: float,
    EH: float,
) -> float:
    d = p1 - p0
    d2 = float(d[0] ** 2 + d[1] ** 2)
    T = float(T)
    return float(EH) * T + (9.0 * float(m) * d2) / (4.0 * (T**2))


def _segment_energy_and_path(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    T: float,
    m: float,
    EH: float,
    samples: int = 60,
) -> Tuple[float, np.ndarray]:
    if np.allclose(p0, p1):
        return 0.0, np.repeat(p0[None, :], 2, axis=0)

    T = float(T)
    tau = np.linspace(0.0, 1.0, int(samples))
    s = (3.0 * tau**2) - (2.0 * tau**3)
    path = p0[None, :] + s[:, None] * (p1 - p0)[None, :]
    e_used = _segment_energy_only(p0, p1, T=T, m=m, EH=EH)
    return float(e_used), path


def _return_energy(
    p: np.ndarray,
    *,
    base: np.ndarray,
    T: float,
    m: float,
    EH: float,
) -> float:
    return _segment_energy_only(p, base, T=T, m=m, EH=EH)


def _search_energy(*, EH: float, search_time: float) -> float:
    return float(EH) * float(search_time)


def _max_distance_one_search(
    params: dict,
) -> float:
    e_scan = _search_energy(EH=params["EH"], search_time=params["search_time"])
    T = float(params["move_time"])
    m = float(params["mass"])
    EH = float(params["EH"])

    budget = float(params["e_max"]) - float(params["eps"]) - e_scan - 2.0 * EH * T
    if budget <= 0:
        return 0.0

    s2 = budget * (2.0 * T**2) / (9.0 * m)
    return float(min(float(params["vector_length"]), np.sqrt(max(s2, 0.0))))


def simulate_linear_search(params: dict) -> dict:
    u = _unit(np.array(params["direction"], dtype=float))
    base = np.array([params["x0"], params["y0"]], dtype=float)
    R = float(params["scan_radius"])
    step = 2.0 * R

    # Search centers along the vector; enforce tangency by fixed step, and ensure the end is covered.
    s_centers = list(np.arange(0.0, float(params["vector_length"]) + 1e-12, step))
    if s_centers[-1] < float(params["vector_length"]) - 1e-12:
        s_centers.append(float(params["vector_length"]))

    all_centers_xy = np.stack([_center_xy(params["x0"], params["y0"], u, s) for s in s_centers], axis=0)

    rng = np.random.default_rng(params.get("seed"))
    # Target is on the vector array (1D), consistent with "array along a vector".
    s_target = float(rng.uniform(0.0, float(params["vector_length"])))
    target_xy = tuple(_center_xy(params["x0"], params["y0"], u, s_target).tolist())

    max_one_search = _max_distance_one_search(params)

    e_used = 0.0
    recharge_count = 0
    sortie_paths: List[np.ndarray] = []

    searched_centers_xy: List[np.ndarray] = []
    found = False
    found_at_index: Optional[int] = None

    e_scan = _search_energy(EH=params["EH"], search_time=params["search_time"])

    def fly(p_from: np.ndarray, p_to: np.ndarray) -> float:
        nonlocal e_used
        de, path = _segment_energy_and_path(
            p_from, p_to, T=params["move_time"], m=params["mass"], EH=params["EH"], samples=70
        )
        e_used += de
        sortie_paths.append(path)
        return de

    def can_do_next_from(p_now: np.ndarray, p_next: np.ndarray) -> bool:
        e_left = float(params["e_max"]) - float(e_used)
        e_move = _segment_energy_only(p_now, p_next, T=params["move_time"], m=params["mass"], EH=params["EH"])
        e_back = _return_energy(p_next, base=base, T=params["move_time"], m=params["mass"], EH=params["EH"])
        need = e_move + e_scan + e_back + float(params["eps"])
        return e_left >= need

    def do_search(p: np.ndarray, idx: int) -> None:
        nonlocal e_used, found, found_at_index
        e_used += e_scan
        searched_centers_xy.append(p.copy())
        if np.hypot(p[0] - target_xy[0], p[1] - target_xy[1]) <= R + 1e-12:
            found = True
            found_at_index = idx

    # Start at base; initial search.
    current_idx = 0
    current_pos = all_centers_xy[current_idx].copy()
    do_search(current_pos, current_idx)

    # Linear searching process.
    while not found and current_idx < len(all_centers_xy) - 1:
        next_idx = current_idx + 1
        next_pos = all_centers_xy[next_idx]

        if not can_do_next_from(current_pos, next_pos):
            e_reposition = 0.0 if current_idx == 0 else _segment_energy_only(
                base, current_pos, T=params["move_time"], m=params["mass"], EH=params["EH"]
            )
            e_need_from_current = (
                _segment_energy_only(current_pos, next_pos, T=params["move_time"], m=params["mass"], EH=params["EH"])
                + e_scan
                + _return_energy(next_pos, base=base, T=params["move_time"], m=params["mass"], EH=params["EH"])
                + float(params["eps"])
            )
            if float(params["e_max"]) - e_reposition < e_need_from_current:
                break

            fly(current_pos, base)
            e_used = 0.0
            recharge_count += 1

            if current_idx != 0:
                fly(base, current_pos)
            continue

        fly(current_pos, next_pos)
        current_idx = next_idx
        current_pos = next_pos.copy()
        do_search(current_pos, current_idx)

    return {
        "target_xy": target_xy,
        "found": found,
        "found_at_index": found_at_index,
        "searched_centers_xy": np.stack(searched_centers_xy, axis=0) if searched_centers_xy else np.zeros((0, 2)),
        "all_centers_xy": all_centers_xy,
        "sortie_paths": sortie_paths,
        "recharge_count": recharge_count,
        "max_one_search_distance": max_one_search,
    }

