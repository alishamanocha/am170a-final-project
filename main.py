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

from params import Parameters

from plotting import (
    plot_energy_to_return,
    plot_energy_used_vs_time,
    plot_position_and_speed_vs_time,
    plot_trajectory_parametric,
    plot_energy_error_tolerance,
)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"

# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------
X0, Y0 = 0.0, 0.0  # Start
XT, YT = 1.0, 2.0  # Target
XL, YL = 0.8, 1.0 # Location of the desired person/thing to find
R = 0.5 # Radius of the "scanner"
T = 1.0  # Flight time (start ↔ target)
M = 1.0  # Drone mass
EH = 1.0  # Hovering energy
E_MAX = 35  # Max energy budget
TS = T / 20  # Time to come to stop when turning midway
EPS = 5e-2  # Threshold used to determine if the drone should return midway (if energy margin < eps)
DT = T / 1000  # Time interval at which to check if drone should return


def run_forward_phase(params, state, e_max, eps, ts):
    """
    Integrate forward from current state toward target; stop when turn condition met.
    Returns (times, trajectory, e_turn, e_turn_times, turned, turn_index).
    """
    t = 0.0

    # Keep track of drone's full trajectory over time
    times = [t]
    trajectory = [state.copy()]

    # Track energy used and energy to return over time
    e_used_tracker = []
    e_turn_tracker = []
    e_turn_times = np.array([])

    # Keep track of if drone turned, and if so, which index in trajectory represents that point in time
    turned = False
    turn_index = None
    located = False
    x0, y0, xT, yT, T, m, EH = params

    x_init = state[0]
    y_init = state[1]

    while t < T:
        # Solve system of ODEs over a small time span with the current state and parameters
        sol = solve_ivp(
            forward_odes,
            (t, t + DT),
            state,
            args=([x_init, y_init, xT, yT, T, m, EH],),
            method="RK45",
            max_step=DT,
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError("Forward flight integration failed")

        # Add all times and state arrays into trajectory tracker
        for i in range(1, sol.y.shape[1]):
            trajectory.append(sol.y[:, i])
            times.append(sol.t[i])

        # Update current state and time
        state = sol.y[:, -1]
        t = sol.t[-1]

        # Get difference between energy margin that would remain after returning and epsilon
        margin_minus_eps = check_energy_turn(
            t, state, e_max, eps, ts, T, m, EH, x0, y0, e_turn_tracker, e_used_tracker
        )
        e_turn_times = np.append(e_turn_times, t)

        # If negative or zero, margin is less than or equal to epsilon, so need to return
        if margin_minus_eps <= 0:
            print("Turning around just before there is insufficient energy to return")
            turned = True
            turn_index = len(trajectory) - 1
            break

    return times, trajectory, e_used_tracker, e_turn_tracker, e_turn_times, turned, turn_index


def run_stop_phase(turn_state, ts, m, EH):
    """Integrate from turn_state to full stop over time ts. Returns solution."""
    # Pass in velocity at the time of starting to stop, amount of time to stop, mass, hovering
    # energy as parameters
    params_stop = [turn_state[2], turn_state[3], ts, m, EH]
    # Come to a stop
    return solve_ivp(
        stop_odes,
        (0, ts),
        turn_state,
        args=(params_stop,),
        method="RK45",
        max_step=ts / 50,
        rtol=1e-8,
        atol=1e-10,
    )


def run_return_phase(stopped_state, x0, y0, T, m, EH):
    """Integrate return from stopped state to (x0, y0). Returns solution."""
    # Pass in stopped position as initial position, initial starting point as ending position, flight
    # time, mass, and hovering energy as parameters
    params_return = [stopped_state[0], stopped_state[1], x0, y0, T, m, EH]
    # Return to initial point
    return solve_ivp(
        forward_odes,
        (0, T),
        stopped_state,
        args=(params_return,),
        method="RK45",
        max_step=T / 200,
        rtol=1e-8,
        atol=1e-10,
    )

# Note: this is broken rn, don't use
def main():
    params = [X0, Y0, XT, YT, T, M, EH]
    state = [X0, Y0, 0.0, 0.0, 0.0]

    # ---- Forward phase ----
    times, trajectory, e_used_tracker, e_turn_tracker, e_turn_times, turned, turn_index = run_forward_phase(
        params, state, E_MAX, EPS, TS
    )

    # ---- Optional stop phase (if we turned midway) ----
    if turned:
        # Get the state at the point of deciding to return
        turn_state = trajectory[turn_index]
        sol_stop = run_stop_phase(turn_state, TS, M, EH)

        # Add all times and state arrays into trajectory tracker, offsetting the times because
        # we simulated from time 0 rather than the actual current time
        t_offset = times[-1]
        for i in range(1, sol_stop.y.shape[1]):
            trajectory.append(sol_stop.y[:, i])
            times.append(t_offset + sol_stop.t[i])
    else:
        print("Reached destination! Turning around now.")

    # ---- Return phase ----
    stopped_index = len(trajectory) - 1
    stopped_state = trajectory[stopped_index]
    print("Stopped state:", stopped_state)

    sol_return = run_return_phase(stopped_state, X0, Y0, T, M, EH)
    print("Ending state:", sol_return.y[:, -1])

    # Add all times and state arrays into trajectory tracker, offsetting the times because we
    # simulated from time 0 rather than the actual current time
    t_offset = times[-1]
    for i in range(1, sol_return.y.shape[1]):
        trajectory.append(sol_return.y[:, i])
        times.append(t_offset + sol_return.t[i])

    # ---- Convert to numpy arrays and derive quantities ----
    trajectory = np.array(trajectory)
    times = np.array(times)
    e_used_tracker = np.array(e_used_tracker)
    e_turn_tracker = np.array(e_turn_tracker)
    expected_e_turn = expected_return_energy(e_turn_times, EH, TS, T, M, X0, Y0, XT, YT)

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    speed = np.hypot(trajectory[:, 2], trajectory[:, 3]) # Speed = |v| = sqrt(vx^2 + vy^2)
    e = trajectory[:, 4]

    # ---- Plotting ----
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectory_parametric(
        x, y, X0, Y0, XT, YT, XL, YL, R, turn_index, stopped_index, turned, located,
        savepath=str(PLOTS_DIR / "parametric_trajectory.png"),
    )
    plot_position_and_speed_vs_time(
        times, x, y, speed, turn_index, stopped_index, turned,
        savepath=str(PLOTS_DIR / "position_and_speed_vs_time.png"),
    )
    plot_energy_used_vs_time(
        times,
        e,
        E_MAX,
        turn_index=turn_index,
        stopped_index=stopped_index,
        turned=turned,
        savepath=str(PLOTS_DIR / "energy_used_vs_time.png"),
    )
    plot_energy_to_return(
        e_turn_times, e_turn_tracker + EPS, expected_e_turn + EPS, E_MAX - e_used_tracker, E_MAX,
        savepath=str(PLOTS_DIR / "energy_to_return.png"),
    )
    plot_energy_error_tolerance(
        e_turn_times, e_turn_tracker + EPS, expected_e_turn + EPS,
        savepath=str(PLOTS_DIR / "energy_error_tolerance.png"),
    )

    # ---- Summary ----
    e_used = trajectory[-1, 4]
    e_left = E_MAX - e_used
    print("Final energy used:", e_used)
    print("Energy left:", e_left)


if __name__ == "__main__":
    main()
