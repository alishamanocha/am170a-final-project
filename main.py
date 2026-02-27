"""
Drone round-trip simulation entry point.

Runs the modified drone dynamics (quadratic velocity profile) from start to target
and back, with optional midway turn-around when energy margin is insufficient.
Uses ODE integration for forward flight, stopping, and return; then generates
all trajectory and energy plots via the plotting module.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Roberto Julian Campos, Kamran Hussain
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from drone_sim import (
    check_energy_turn,
    expected_return_energy,
    forward_odes,
    get_location_distance,
    stop_odes,
)

from search_figure import (
    simulate_search_vector
)

from params import Parameters

from plotting import (
    plot_energy_to_return,
    plot_energy_used_vs_time,
    plot_position_and_speed_vs_time,
    plot_trajectory_parametric,
    plot_energy_error_tolerance,
    plot_speed_vs_time,
)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"

def main():
    # Initialize params
    params = Parameters(
        X0 = 0.0,
        Y0 = 0.0,
        XL = 3.1,
        YL = 0.1,
        R_SCAN = 0.25,
        T = 1.0,
        M = 1.0,
        EH = 1.0,
        ES = 0.3,
        E_MAX = 35.0,
        DT = 1e-3,
        EPS = 5e-2,
    )
    print("Simulations for the 5 fixed angle search directions\n")
    results = []
    angles = [0]
    for angle in angles:
        print(f"Direction {angle}")
        full_trajectory, full_times, full_e_used_track, full_e_turn_track, full_e_turn_times, turned, located, turn_index, stopped_index, scan_indices = simulate_search_vector(angle, params)
        x = full_trajectory[:,0]
        print(x)
        y = full_trajectory[:,1]
        vx_traj = full_trajectory[:,2]
        vy_traj = full_trajectory[:,3]
        speed = np.hypot(vx_traj, vy_traj)
        e_traj = full_trajectory[:,4]
        print(full_times)
        for i, idx in enumerate(scan_indices):
            print(f"State before scan {i+1}: {full_trajectory[idx-1]}")
            print(f"State after scan {i+1}: {full_trajectory[idx]}")
        expected_e_turn = expected_return_energy(full_e_turn_times, params.EH, params.TS, params.T, params.M, params.X0, params.Y0, params.XL, params.YL)
        plot_trajectory_parametric(x, y, params.X0, params.Y0, params.XL, params.YL, params.R_SCAN, turn_index, stopped_index, turned, located)
        plot_speed_vs_time(full_times, speed, turn_index, stopped_index, turned)
        plot_energy_to_return(
        full_e_turn_times, full_e_turn_track + params.EPS, expected_e_turn + params.EPS, params.E_MAX - full_e_used_track, params.E_MAX)

if __name__ == "__main__":
    main()
