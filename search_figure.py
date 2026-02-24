"""
Generate Multi-Directional Search Pattern Figure

Uses the real drone simulation (drone_sim.py) to fly each direction vector and
record the actual energy-constrained reach. LIDAR scan circles are placed every
2*R_LIDAR along each simulated trajectory.

Authors: Alisha Manocha, Reagan Ross, Aydin Khan, Kamran Hussain
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from drone_sim import (
    check_energy_turn,
    forward_odes,
    stop_odes,
)

# Simulation parameters
X0, Y0 = 0.0, 0.0
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
ANGLES_DEG = [0, 180, 90, 270, 45]
COLORS  = ['#1f77b4', '#ff7f0e', "#3aad3a", '#9467bd', '#d62728']

def simulate_search_vector(angle_deg):
    """
    Simulate the drone flying outward along a given angle until the energy
    margin forces a turn-around. Returns the (x, y) trajectory array and
    the list of LIDAR scan stop positions.
    """
    angle = np.radians(angle_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Place target some meters away so that the energy check actually stops the drone before arrival
    xT = X0 + FAR * direction[0]
    yT = Y0 + FAR * direction[1]

    # Place the location off to the side so its never actually triggered
    xL, yL = X0 + 9999.0, Y0 + 9999.0
    r_loc = 0.001

    state = [X0, Y0, 0.0, 0.0, 0.0]
    t = 0.0

    trajectory  = [np.array(state)]
    times = [t]
    e_turn_track = []
    e_used_track = []

    turned = False
    turn_index  = None

    while t < T:
        sol = solve_ivp(
            forward_odes,
            (t, t + DT),
            state,
            args=([X0, Y0, xT, yT, T, M, EH],),
            method="RK45",
            max_step=DT,
            rtol=1e-8,
            atol=1e-10,
        )
        
        for i in range(1, sol.y.shape[1]):
            trajectory.append(sol.y[:, i])
            times.append(sol.t[i])

        state = sol.y[:, -1]
        t = sol.t[-1]

        margin_minus_eps = check_energy_turn(
            t, state, E_MAX, EPS, TS, T, M, EH,
            X0, Y0, e_turn_track, e_used_track
        )

        if margin_minus_eps <= 0:
            turned  = True
            turn_index = len(trajectory) - 1
            break

    trajectory = np.array(trajectory)
    x_traj = trajectory[:, 0]
    y_traj = trajectory[:, 1]

    # Compute LIDAR scan positions
    
    # Place stops every 2*R_LIDAR of cumulative arc length along the trajectory
    # At each stop the drone halts and scans a circle of radius R_LIDAR
    cum_dist = np.zeros(len(x_traj))
    for i in range(1, len(x_traj)):
        cum_dist[i] = cum_dist[i-1] + np.hypot(x_traj[i] - x_traj[i-1], y_traj[i] - y_traj[i-1] )
        
    max_dist = cum_dist[-1]
    stop_distances = np.arange(R_LIDAR, max_dist, 2 * R_LIDAR)

    scan_stops = []
    for d in stop_distances:
        idx = np.searchsorted(cum_dist, d)
        idx = min(idx, len(x_traj) - 1)
        scan_stops.append((x_traj[idx], y_traj[idx]))

    print(f"Direction {angle_deg:>4} degrees: max reach = {max_dist:.3f} m, "f"{len(scan_stops)} scan stops")

    return x_traj, y_traj, scan_stops, max_dist

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
    for angle_deg in ANGLES_DEG:
        print(f"Direction {angle_deg}")
        result = simulate_search_vector(angle_deg)
        results.append(result)

    plot_search_pattern(results, savepath="plots/search_with_scanning.png")
