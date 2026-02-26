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

from drone_sim import (
    check_energy_turn,
    forward_odes,
    stop_odes,
    get_location_distance
)
from main import (
    run_forward_phase,
    run_stop_phase,
    run_return_phase
)

# Simulation parameters
X0, Y0 = 0.0, 0.0
XL, YL = 3.0, 0.0 # Location of the desired person/thing to find
T  = 1.0 # Flight time budget per leg
M = 1.0 # Drone mass
EH = 1.0 # Hovering energy rate
E_MAX = 35.0 # Max energy budget
TS = T / 20 # Time to stop when turning
EPS = 5e-2 # Energy margin threshold
DT = T / 1000 # Integration step size

R_LIDAR = 0.25 # LIDAR scan radius (spacing 2*R_LIDAR gives about 5 stops per vector)
FAR = 3.0 # Target distance (actual reach is energy-constrained to about 2.3 m)

# Search directions
# Start at  0°, then bisect: 180°, 90°, 270°, 45°
ANGLES = [0, np.pi]
COLORS  = ['#1f77b4', '#ff7f0e', "#3aad3a", '#9467bd', '#d62728']

def travel_next_target(initial_state, xT, yT):
    # Get current position of this travel phase
    # Pass current position as initial position
    params = [X0, Y0, xT, yT, T, M, EH]

    # Run forward travel from initial position x_init, y_init to xT, yT
    # Get in return: times, trajectory, e_used_tracker, e_turn_tracker, e_turn_times, turned, turn_index
    return run_forward_phase(
        params, initial_state, E_MAX, EPS, TS
    )

def simulate_search_vector(angle):
    """
    Simulate the drone generating targets at regular intervals along a given angle until the energy
    margin forces a turn-around or the destination is located. Returns the (x, y) trajectory array and
    the list of LIDAR scan stop positions.
    """
    direction = np.array([np.cos(angle), np.sin(angle)])

    state = [X0, Y0, 0.0, 0.0, 0.0]
    t = 0.0
    full_times = []
    full_trajectory = []
    full_e_turn_track = []
    full_e_used_track = []
    full_e_turn_times = []
    
    turned = False
    turn_index = None
    located = False

    # Place first target 2 * radius of LIDAR scan circle away from initial position
    xT = X0 + 2 * R_LIDAR * direction[0]
    yT = Y0 + 2 * R_LIDAR * direction[1]

    while True:
        # Travel to the next target
        times, trajectory, e_turn_track, e_used_track, e_turn_times, turned, _ = travel_next_target(state, xT, yT)
        full_times.extend(times)
        full_trajectory.extend(trajectory)
        full_e_turn_times.extend(e_turn_times)
        full_e_turn_track.extend(e_turn_track)
        full_e_used_track.extend(e_used_track)
        state = full_trajectory[-1]

        if turned:
            turn_index = len(full_trajectory)-1
            # Get the state at the point of deciding to return
            turn_state = full_trajectory[-1]
            sol_stop = run_stop_phase(turn_state, TS, M, EH)

            # Add all times and state arrays into trajectory tracker, offsetting the times because
            # we simulated from time 0 rather than the actual current time
            t_offset = full_times[-1]
            for i in range(1, sol_stop.y.shape[1]):
                full_trajectory.append(sol_stop.y[:, i])
                full_times.append(t_offset + sol_stop.t[i])
            break
        else:
            print(f"Reached target! Now scanning! At position ({state[0], state[1]}), velocity ({state[2], state[3]}), energy used {state[4]}")
            dist = get_location_distance(state, XL, YL)

            if dist < R_LIDAR:
                print(f"Found location at ({XL, YL})! Turning back!")
                located = True
                turned = True
                turn_index = len(full_trajectory) - 1
                break

            # Didn't find target, generate new target
            xT = state[0] + 2 * R_LIDAR * direction[0]
            yT = state[1] + 2 * R_LIDAR * direction[1]
    
    # ---- Return phase ----
    stopped_index = len(full_trajectory) - 1
    stopped_state = full_trajectory[stopped_index]
    print(f"Stopped state: {stopped_state}")
    print("Now returning back to origin")
    sol_return = run_return_phase(stopped_state, X0, Y0, T, M, EH)
        
    # Add all times and state arrays into trajectory tracker, offsetting the times because we
    # simulated from time 0 rather than the actual current time
    t_offset = full_times[-1]
    for i in range(1, sol_return.y.shape[1]):
        full_trajectory.append(sol_return.y[:, i])
        full_times.append(t_offset + sol_return.t[i])

    full_trajectory = np.array(full_trajectory)
    full_times = np.array(full_times)
    full_e_used_track = np.array(full_e_used_track)
    full_e_turn_track = np.array(full_e_turn_track)

    x_traj = full_trajectory[:, 0]
    y_traj = full_trajectory[:, 1]

    print(f"Final position: ({x_traj[-1]}, {y_traj[-1]})")
    if not located:
        print("Did not find missing person")
    
    return x_traj, y_traj

def plot_search_pattern(results, savepath="search_pattern_simulated.png"):
    """
    Plot all simulated search vectors with their LIDAR scan circles.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')

    # Draw the energy-constrained reach circle (average of all directions)
    reach_values = [r for *_, r in results]
    avg_reach = np.mean(reach_values)
    max_circle = patches.Circle(
        (X0, Y0), avg_reach,
        edgecolor='#888888', facecolor='none',
        linestyle='--', linewidth=1.5, zorder=1
    )
    ax.add_patch(max_circle)

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#FF77FF']
    BLUE = '#4a90d9'

    for i, (x_traj, y_traj, scan_stops, reach) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        # Flight path colored per vector
        ax.plot(x_traj, y_traj, color=color, lw=1.8, alpha=0.85, zorder=2)

        # Scan circles same color as flight path
        for (sx, sy) in scan_stops:
            circle = patches.Circle(
                (sx, sy), R_LIDAR,
                edgecolor=color, facecolor=color,
                alpha=0.18, linewidth=1.0, zorder=3
            )
            ax.add_patch(circle)
            ax.scatter(sx, sy, color=color, s=14, zorder=4)
            
    # Center point
    ax.scatter(X0, Y0, color='black', s=30, zorder=6, marker='o')
    ax.annotate('Center', (X0, Y0),
                textcoords='offset points', xytext=(6, -16),
                fontsize=9, color='black')

    # Target point
    XT, YT = -1.2, 1.2
    ax.scatter(XT, YT, color='red', s=30, zorder=6, marker='o')
    ax.annotate('Target', (XT, YT),
                textcoords='offset points', xytext=(6, -16),
                fontsize=9, color='red')
    
    # Annotate R_LIDAR on the first scan stop of Vector 1 (if exists)
    if results[0][2]:
        first_x, first_y = results[0][2][0]
        ax.annotate(
            '', xy=(first_x + R_LIDAR, first_y), xytext=(first_x, first_y),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2)
        )
        ax.text(first_x + R_LIDAR / 2 + 0.08, first_y + 0.12,
                r'$r_{\mathrm{scan}}$', ha='center', fontsize=12, color='black')
        
    # Annotate R_max
    ax.annotate(
        '', xy=(avg_reach, 0), xytext=(X0, Y0),
        arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2, linestyle='dashed')
    )
    ax.text(avg_reach / 2, -0.22,
            r'$R_{\mathrm{max}}$', ha='center', fontsize=12, color='black')
    
    # Legend
    legend_handles = [
        Line2D([0], [0], color=COLORS[0], lw=2),
        Line2D([0], [0], color=COLORS[1], lw=2),
        Line2D([0], [0], color=COLORS[2], lw=2),
        Line2D([0], [0], color=COLORS[3], lw=2),
        Line2D([0], [0], color=COLORS[4], lw=2),
        patches.Patch(facecolor=BLUE, alpha=0.35, edgecolor=BLUE),
        Line2D([0], [0], color='#888888', lw=1.5, linestyle='--'),
        Line2D([0], [0], marker='o', color='black', markersize=5,
               linestyle='None', markerfacecolor='black'),
        Line2D([0], [0], marker='o', color='red', markersize=5,
               linestyle='None', markerfacecolor='red'),
    ]
    legend_labels = [
        '1st flight', '2nd flight', '3rd flight', '4th flight', '5th flight',
        'Scan area', 'Max energy boundary', 'Center', 'Target',
    ]
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=7, framealpha=0.92)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(
        'Drone Trajectories with Scanning',
        fontsize=13
    )
    ax.grid(True, alpha=0.3, color='lightgray')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    print("Simulations for the 5 fixed angle search directions\n")
    results = []
    for angle in ANGLES:
        print(f"Direction {angle}")
        x_traj, y_traj = simulate_search_vector(angle)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.plot(x_traj, y_traj)
        plt.show()
        #results.append(result)

    #plot_search_pattern(results, savepath="plots/search_with_scanning.png")
