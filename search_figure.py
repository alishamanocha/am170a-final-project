"""
Generate Multi-Directional Search Pattern Figure

Uses the real drone simulation (drone_sim.py) to fly each direction vector and
record the actual energy-constrained reach. LIDAR scan circles are placed every
2*R_LIDAR along each simulated trajectory.

Authors: Alisha Manocha, Reagan Ross, Aydin Khan, Kamran Hussain
"""

from math import e
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.radau import P

from drone_sim import (
    check_energy_turn,
    forward_odes,
    stop_odes,
    get_location_distance,
    expected_return_energy
)
from plotting import (
    plot_speed_vs_time,
    plot_trajectory_parametric,
    plot_energy_to_return,
    plot_energy_error_tolerance
)


def run_forward_phase(params, state, xT, yT):
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
    e_turn_times = []

    # Keep track of if drone turned, and if so, which index in trajectory represents that point in time
    turned = False
    turn_index = None
    located = False

    x_init = state[0]
    y_init = state[1]

    while t < params.T:
        # Solve system of ODEs over a small time span with the current state and parameters
        sol = solve_ivp(
            forward_odes,
            (t, t + params.DT),
            state,
            args=([x_init, y_init, xT, yT, params.T, params.M, params.EH],),
            method="RK45",
            max_step=params.DT,
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
            t, state, params.E_MAX, params.EPS, params.TS, params.T, params.M, params.EH, params.ES, params.X0, params.Y0, e_turn_tracker, e_used_tracker
        )
        e_turn_times.append(t)

        # If negative or zero, margin is less than or equal to epsilon, so need to return
        if margin_minus_eps <= 0:
            print("Turning around just before there is insufficient energy to return")
            turned = True
            turn_index = len(trajectory) - 1
            break

    return times, trajectory, e_used_tracker, e_turn_tracker, e_turn_times, turned, turn_index


def run_stop_phase(turn_state, params):
    """Integrate from turn_state to full stop over time ts. Returns solution."""
    # Pass in velocity at the time of starting to stop, amount of time to stop, mass, hovering
    # energy as parameters
    params_stop = [turn_state[2], turn_state[3], params.TS, params.M, params.EH]
    # Come to a stop
    return solve_ivp(
        stop_odes,
        (0, params.TS),
        turn_state,
        args=(params_stop,),
        method="RK45",
        max_step=params.TS / 50,
        rtol=1e-8,
        atol=1e-10,
    )


def run_return_phase(stopped_state, params):
    """Integrate return from stopped state to (x0, y0). Returns solution."""
    # Pass in stopped position as initial position, initial starting point as ending position, flight
    # time, mass, and hovering energy as parameters
    params_return = [stopped_state[0], stopped_state[1], params.X0, params.Y0, params.T, params.M, params.EH]
    # Return to initial point
    return solve_ivp(
        forward_odes,
        (0, params.T),
        stopped_state,
        args=(params_return,),
        method="RK45",
        max_step=params.T / 200,
        rtol=1e-8,
        atol=1e-10,
    )

# Search directions
# Start at  0°, then bisect: 180°, 90°, 270°, 45°
# ANGLES = [0]
# COLORS  = ['#1f77b4', '#ff7f0e', "#3aad3a", '#9467bd', '#d62728']

def simulate_search_vector(angle, params):
    """
    Simulate the drone generating targets at regular intervals along a given angle until the energy
    margin forces a turn-around or the destination is located. Returns the (x, y) trajectory array and
    the list of LIDAR scan stop positions.
    """
    direction = np.array([np.cos(angle), np.sin(angle)])

    state = [params.X0, params.Y0, 0.0, 0.0, 0.0]
    t = 0.0
    full_times = []
    full_trajectory = []
    full_e_turn_track = []
    full_e_used_track = []
    full_e_turn_times = []
    scan_indices = []
    
    turned = False
    turn_index = None
    located = False

    # Place first target 2 * radius of LIDAR scan circle away from initial position
    xT = params.X0 + 2 * params.R_SCAN * direction[0]
    yT = params.Y0 + 2 * params.R_SCAN * direction[1]

    while True:
        # Travel to the next target
        times, trajectory, e_turn_track, e_used_track, e_turn_times, turned, _ = run_forward_phase(params, state, xT, yT)
        if len(full_times) == 0:
            full_times.extend(times)
            full_trajectory.extend(trajectory)
            full_e_turn_times.extend(e_turn_times)
        else:
            t_offset = full_times[-1]
            full_times.extend([t_offset + t for t in times[1:]])
            full_trajectory.extend(trajectory[1:])
            full_e_turn_times.extend([t_offset + t for t in e_turn_times])
        full_e_turn_track.extend(e_turn_track)
        full_e_used_track.extend(e_used_track)
        state = full_trajectory[-1]

        # Need to turn back
        if turned:
            turn_index = len(full_trajectory)-1
            # Get the state at the point of deciding to return
            turn_state = full_trajectory[-1]
            sol_stop = run_stop_phase(turn_state, params)

            # Add all times and state arrays into trajectory tracker, offsetting the times because
            # we simulated from time 0 rather than the actual current time
            t_offset = full_times[-1]
            for i in range(1, sol_stop.y.shape[1]):
                full_trajectory.append(sol_stop.y[:, i])
                full_times.append(t_offset + sol_stop.t[i])
            break
        else:
            print(f"Reached target! Now scanning! At position ({state[0], state[1]}), velocity ({state[2], state[3]}), energy used {state[4]}")
            dist = get_location_distance(state, params.XL, params.YL)
            
            state_after_scan = state.copy()
            state_after_scan[4] += params.ES
            full_trajectory.append(state_after_scan)
            full_times.append(times[-1]) # Duplicating the existing last time, assuming scan is instantaneous
            scan_indices.append(len(full_trajectory)-1)

            if dist < params.R_SCAN:
                print(f"Found location at ({params.XL, params.YL})! Turning back!")
                located = True
                turned = True
                turn_index = len(full_trajectory) - 1
                break

            # Didn't find target, generate new target
            xT = state[0] + 2 * params.R_SCAN * direction[0]
            yT = state[1] + 2 * params.R_SCAN * direction[1]
    
    # ---- Return phase ----
    stopped_index = len(full_trajectory) - 1
    stopped_state = full_trajectory[stopped_index]
    print(f"Stopped state: {stopped_state}")
    print("Now returning back to origin")
    sol_return = run_return_phase(stopped_state, params)
        
    # Add all times and state arrays into trajectory tracker, offsetting the times because we
    # simulated from time 0 rather than the actual current time
    t_offset = full_times[-1]
    for i in range(1, sol_return.y.shape[1]):
        full_trajectory.append(sol_return.y[:, i])
        full_times.append(t_offset + sol_return.t[i])

    full_trajectory = np.array(full_trajectory)
    full_times = np.array(full_times)
    full_e_turn_times = np.array(full_e_turn_times)
    full_e_used_track = np.array(full_e_used_track)
    full_e_turn_track = np.array(full_e_turn_track)

    print(f"Final position: ({full_trajectory[-1][0]}, {full_trajectory[-1][1]})")
    if not located:
        print("Did not find missing person")
    
    return full_trajectory, full_times, full_e_used_track, full_e_turn_track, full_e_turn_times, turned, located, turn_index, stopped_index, scan_indices

# def plot_search_pattern(results, savepath="search_pattern_simulated.png"):
#     """
#     Plot all simulated search vectors with their LIDAR scan circles.
#     """
#     fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
#     ax.set_facecolor('#f9f9f9')
#     fig.patch.set_facecolor('white')

#     # Draw the energy-constrained reach circle (average of all directions)
#     reach_values = [r for *_, r in results]
#     avg_reach = np.mean(reach_values)
#     max_circle = patches.Circle(
#         (X0, Y0), avg_reach,
#         edgecolor='#888888', facecolor='none',
#         linestyle='--', linewidth=1.5, zorder=1
#     )
#     ax.add_patch(max_circle)

#     COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#FF77FF']
#     BLUE = '#4a90d9'

#     for i, (x_traj, y_traj, scan_stops, reach) in enumerate(results):
#         color = COLORS[i % len(COLORS)]
#         # Flight path colored per vector
#         ax.plot(x_traj, y_traj, color=color, lw=1.8, alpha=0.85, zorder=2)

#         # Scan circles same color as flight path
#         for (sx, sy) in scan_stops:
#             circle = patches.Circle(
#                 (sx, sy), R_LIDAR,
#                 edgecolor=color, facecolor=color,
#                 alpha=0.18, linewidth=1.0, zorder=3
#             )
#             ax.add_patch(circle)
#             ax.scatter(sx, sy, color=color, s=14, zorder=4)
            
#     # Center point
#     ax.scatter(X0, Y0, color='black', s=30, zorder=6, marker='o')
#     ax.annotate('Center', (X0, Y0),
#                 textcoords='offset points', xytext=(6, -16),
#                 fontsize=9, color='black')

#     # Target point
#     XT, YT = -1.2, 1.2
#     ax.scatter(XT, YT, color='red', s=30, zorder=6, marker='o')
#     ax.annotate('Target', (XT, YT),
#                 textcoords='offset points', xytext=(6, -16),
#                 fontsize=9, color='red')
    
#     # Annotate R_LIDAR on the first scan stop of Vector 1 (if exists)
#     if results[0][2]:
#         first_x, first_y = results[0][2][0]
#         ax.annotate(
#             '', xy=(first_x + R_LIDAR, first_y), xytext=(first_x, first_y),
#             arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2)
#         )
#         ax.text(first_x + R_LIDAR / 2 + 0.08, first_y + 0.12,
#                 r'$r_{\mathrm{scan}}$', ha='center', fontsize=12, color='black')
        
#     # Annotate R_max
#     ax.annotate(
#         '', xy=(avg_reach, 0), xytext=(X0, Y0),
#         arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2, linestyle='dashed')
#     )
#     ax.text(avg_reach / 2, -0.22,
#             r'$R_{\mathrm{max}}$', ha='center', fontsize=12, color='black')
    
#     # Legend
#     legend_handles = [
#         Line2D([0], [0], color=COLORS[0], lw=2),
#         Line2D([0], [0], color=COLORS[1], lw=2),
#         Line2D([0], [0], color=COLORS[2], lw=2),
#         Line2D([0], [0], color=COLORS[3], lw=2),
#         Line2D([0], [0], color=COLORS[4], lw=2),
#         patches.Patch(facecolor=BLUE, alpha=0.35, edgecolor=BLUE),
#         Line2D([0], [0], color='#888888', lw=1.5, linestyle='--'),
#         Line2D([0], [0], marker='o', color='black', markersize=5,
#                linestyle='None', markerfacecolor='black'),
#         Line2D([0], [0], marker='o', color='red', markersize=5,
#                linestyle='None', markerfacecolor='red'),
#     ]
#     legend_labels = [
#         '1st flight', '2nd flight', '3rd flight', '4th flight', '5th flight',
#         'Scan area', 'Max energy boundary', 'Center', 'Target',
#     ]
#     ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=7, framealpha=0.92)

#     ax.set_xlim(-3.2, 3.2)
#     ax.set_ylim(-3.2, 3.2)
#     ax.set_aspect('equal')
#     ax.set_xlabel('x (m)', fontsize=12)
#     ax.set_ylabel('y (m)', fontsize=12)
#     ax.set_title(
#         'Drone Trajectories with Scanning',
#         fontsize=13
#     )
#     ax.grid(True, alpha=0.3, color='lightgray')
#     ax.tick_params(labelsize=10)

#     plt.tight_layout()
#     plt.savefig(savepath, dpi=200, bbox_inches='tight')
#     plt.close()
    
#if __name__ == "__main__":
    # print("Simulations for the 5 fixed angle search directions\n")
    # results = []
    # for angle in ANGLES:
    #     print(f"Direction {angle}")
    #     full_trajectory, full_times, full_e_used_track, full_e_turn_track, full_e_turn_times, turned, located, turn_index, stopped_index = simulate_search_vector(angle)
    #     x = full_trajectory[:,0]
    #     print(x)
    #     y = full_trajectory[:,1]
    #     vx_traj = full_trajectory[:,2]
    #     vy_traj = full_trajectory[:,3]
    #     speed = np.hypot(vx_traj, vy_traj)
    #     e_traj = full_trajectory[:,4]
    #     print(full_times)
    #     expected_e_turn = expected_return_energy(full_e_turn_times, EH, TS, T, M, X0, Y0, XL, YL)
    #     plot_trajectory_parametric(x, y, X0, Y0, XL, YL, R_LIDAR, turn_index, stopped_index, turned, located)
    #     plot_speed_vs_time(full_times, speed, turn_index, stopped_index, turned)
    #     plot_energy_to_return(
    #     full_e_turn_times, full_e_turn_track + EPS, expected_e_turn + EPS, E_MAX - full_e_used_track, E_MAX)
    #     plt.show()
        #results.append(result)

    #plot_search_pattern(results, savepath="plots/search_with_scanning.png")
