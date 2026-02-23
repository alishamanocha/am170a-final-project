"""
Plotting Functions

Script that visualizes the drone simulation results and produces the figures linked in the report.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Roberto Julian Campos, Kamran Hussain
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional


def plot_linear_search_area(
    *,
    all_centers_xy: np.ndarray,
    searched_centers_xy: np.ndarray,
    scan_radius: float,
    base_xy: tuple[float, float],
    vector_end_xy: tuple[float, float],
    target_xy: tuple[float, float],
    found: bool,
    sortie_paths: list[np.ndarray],
    recharge_count: int,
    max_one_search_distance: Optional[float] = None,
    savepath: str = "linear_search_area.png",
):
    """
    Visualize the searched area (union of scan circles) for the linear search algorithm.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=300)

    # Light path traces (sorties).
    for i, path in enumerate(sortie_paths):
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="0.65",
            lw=1.2,
            alpha=0.7,
            label="Flight path" if i == 0 else None,
            zorder=1,
        )

    # Scan circles (searched).
    for i, c in enumerate(searched_centers_xy):
        circ = patches.Circle(
            (float(c[0]), float(c[1])),
            float(scan_radius),
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.14,
            lw=1.0,
            zorder=2,
            label="Searched area" if i == 0 else None,
        )
        ax.add_patch(circ)

    # Circle outlines for visual crispness.
    for c in searched_centers_xy:
        circ = patches.Circle(
            (float(c[0]), float(c[1])),
            float(scan_radius),
            facecolor="none",
            edgecolor="tab:blue",
            alpha=0.55,
            lw=1.1,
            zorder=3,
        )
        ax.add_patch(circ)

    # Centers along the vector (all, faint).
    ax.plot(
        all_centers_xy[:, 0],
        all_centers_xy[:, 1],
        ".-",
        color="0.75",
        lw=1.2,
        ms=4,
        zorder=0,
        label="Vector array",
    )

    # Base and end markers.
    ax.scatter([base_xy[0]], [base_xy[1]], s=90, c="green", marker="*", zorder=5, label="Base")
    ax.scatter(
        [vector_end_xy[0]],
        [vector_end_xy[1]],
        s=70,
        c="black",
        marker="x",
        zorder=5,
        label="Vector end",
    )

    # Target marker.
    ax.scatter(
        [target_xy[0]],
        [target_xy[1]],
        s=65,
        c=("tab:green" if found else "tab:red"),
        marker="o",
        zorder=6,
        label=("Target (found)" if found else "Target (not found)"),
    )

    # Optional annotation for max one-search distance along the vector.
    if max_one_search_distance is not None and max_one_search_distance > 0:
        base = np.array(base_xy, dtype=float)
        end = np.array(vector_end_xy, dtype=float)
        d = np.hypot(*(end - base))
        if d > 0:
            u = (end - base) / d
            p = base + float(max_one_search_distance) * u
            ax.scatter([p[0]], [p[1]], s=55, c="tab:orange", marker="D", zorder=6, label="Max 1-search distance")

    title = "Linear search: tangential scan circles"
    subtitle = f"searches={len(searched_centers_xy)}, recharges={recharge_count}, found={found}"
    ax.set_title(f"{title}\n{subtitle}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    # Bounds with margin.
    pts = [all_centers_xy]
    if len(searched_centers_xy) > 0:
        pts.append(searched_centers_xy)
    pts.append(np.array([[base_xy[0], base_xy[1]], [vector_end_xy[0], vector_end_xy[1]], [target_xy[0], target_xy[1]]]))
    all_pts = np.vstack(pts)
    x_min, y_min = np.min(all_pts, axis=0)
    x_max, y_max = np.max(all_pts, axis=0)
    margin = 0.15 * max(x_max - x_min, y_max - y_min, 1.0) + float(scan_radius)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_multi_direction_lidar_coverage(
    *,
    base_xy: tuple[float, float],
    directions: list[tuple[float, float]],
    scan_radius: float,
    max_radius: float,
    step: Optional[float] = None,
    savepath: str = "multi_direction_lidar.png",
):
    """
    Visualize tangential scan-circle coverage along multiple directions from a common base.

    This is a visualization helper (not a full search simulation): for each direction vector,
    it places scan centers every `step` meters (defaults to 2*scan_radius) out to `max_radius`,
    draws the flight path through those centers, and overlays the max-energy boundary circle.
    """
    base = np.array([base_xy[0], base_xy[1]], dtype=float)
    scan_radius = float(scan_radius)
    max_radius = float(max_radius)
    if step is None:
        step = 2.0 * scan_radius
    step = float(step)
    if step <= 0:
        raise ValueError("step must be positive")

    fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=300)

    # Max-energy boundary (dashed circle centered at base).
    if max_radius > 0:
        boundary = patches.Circle(
            (float(base[0]), float(base[1])),
            max_radius,
            fill=False,
            edgecolor="0.5",
            linestyle="--",
            linewidth=1.6,
            zorder=0,
            label="Max energy boundary",
        )
        ax.add_patch(boundary)

    all_centers = []
    labeled_path = False
    labeled_area = False

    for v in directions:
        u = np.array([float(v[0]), float(v[1])], dtype=float)
        n = float(np.hypot(u[0], u[1]))
        if n == 0.0:
            continue
        u = u / n

        s_centers = list(np.arange(0.0, max_radius + 1e-12, step))
        if len(s_centers) == 0:
            s_centers = [0.0]
        if s_centers[-1] < max_radius - 1e-12:
            s_centers.append(max_radius)

        centers = np.stack([base + s * u for s in s_centers], axis=0)
        all_centers.append(centers)

        # Flight path through centers.
        ax.plot(
            centers[:, 0],
            centers[:, 1],
            "-o",
            color="tab:blue",
            lw=2.2,
            ms=3.8,
            alpha=0.9,
            zorder=3,
            label="Flight path" if not labeled_path else None,
        )
        labeled_path = True

        # Scan circles at each center (coverage area).
        for i, c in enumerate(centers):
            circ = patches.Circle(
                (float(c[0]), float(c[1])),
                scan_radius,
                facecolor="tab:blue",
                edgecolor="tab:blue",
                alpha=0.16,
                lw=0.8,
                zorder=2,
                label="LIDAR scan area" if (not labeled_area and i == 0) else None,
            )
            ax.add_patch(circ)
        labeled_area = True

    # Base / center marker.
    ax.scatter([base[0]], [base[1]], s=70, c="black", marker="o", zorder=5, label="Center")

    # Annotate R_max on +x axis for clarity.
    if max_radius > 0:
        ax.annotate(
            r"$R_{\max}$",
            xy=(float(base[0] + max_radius), float(base[1])),
            xytext=(float(base[0] + 0.55 * max_radius), float(base[1] - 0.12 * max_radius)),
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2),
            color="0.35",
            fontsize=11,
            zorder=6,
        )

    ax.set_title("Drone Trajectories with LIDAR Scan Coverage")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    # Bounds with margin.
    if len(all_centers) > 0:
        pts = np.vstack(all_centers + [base[None, :]])
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
    else:
        x_min = x_max = float(base[0])
        y_min = y_max = float(base[1])
    margin = 0.18 * max(x_max - x_min, y_max - y_min, 1.0) + scan_radius
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_trajectory_parametric(
    x, y, x0, y0, xT, yT, xL, yL, r, turn_index, stopped_index, turned, located, savepath="parametric_trajectory.png"
):
    """
    Plot parametric solution of x(t) vs. y(t) split into forward and return phases.

    Args:
        x (numpy.ndarray): The x-coordinates of the drone trajectory.
        y (numpy.ndarray): The y-coordinates of the drone trajectory.
        x0 (float): The x-coordinate of the starting point.
        y0 (float): The y-coordinate of the starting point.
        xT (float): The x-coordinate of the ending point.
        yT (float): The y-coordinate of the ending point.
        turn_index (int): The index of the time when the drone turns around.
        stopped_index (int): The index of the time when the drone stops.
        turned (bool): Whether the drone has turned around.
        savepath (str): The path to save the plot.

    Returns:
        None
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    if turned:
        ax1.plot(x[: turn_index + 1], y[: turn_index + 1], lw=2.5, label="Forward", color="tab:blue")
        ax1.plot(
            x[turn_index : stopped_index + 1],
            y[turn_index : stopped_index + 1],
            ":",
            lw=2,
            label="Stopping",
            color="tab:orange",
        )
        ax1.scatter(x[turn_index], y[turn_index], c="orange", s=100, marker="x", label="Turn decision", zorder=5)
        ax1.scatter(x[stopped_index], y[stopped_index], c="purple", s=100, marker="s", label="Stopped", zorder=5)
        if located:
            circle = patches.Circle((x[turn_index], y[turn_index]), r, edgecolor="r", facecolor="none", linewidth=2)
            ax1.add_patch(circle)
    else:
        ax1.plot(x[: stopped_index + 1], y[: stopped_index + 1], lw=2.5, label="Forward")

    ax1.scatter(x0, y0, c="green", s=80, label="Start", zorder=5)
    ax1.scatter(xT, yT, c="red", s=80, label="Target", zorder=5)
    ax1.scatter(xL, yL, c="orange", s=80, label="Location", zorder=5)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Forward Journey")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Pad bounds to include the target.
    x_forward = x[: stopped_index + 1]
    y_forward = y[: stopped_index + 1]
    x_min = min(np.min(x_forward), xT)
    x_max = max(np.max(x_forward), xT)
    y_min = min(np.min(y_forward), yT)
    y_max = max(np.max(y_forward), yT)
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin = 0.1 * max(x_range, y_range, 0.5)
    ax1.set_xlim(x_min - margin, x_max + margin)
    ax1.set_ylim(y_min - margin, y_max + margin)

    # Right subplot (return path)
    ax2.plot(x[stopped_index:], y[stopped_index:], "--", lw=2.5, label="Return", color="tab:green")
    ax2.scatter(x[stopped_index], y[stopped_index], c="purple", s=100, marker="s", label="Stopped", zorder=5)
    ax2.scatter(x0, y0, c="green", s=80, label="Start", zorder=5)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Return Journey")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    
    # Padding around return trajectory
    x_return = x[stopped_index:]
    y_return = y[stopped_index:]
    x_min = np.min(x_return)
    x_max = np.max(x_return)
    y_min = np.min(y_return)
    y_max = np.max(y_return)
    x_range = x_max - x_min
    y_range = y_max - y_min
    margin = 0.1 * max(x_range, y_range, 0.5)
    ax2.set_xlim(x_min - margin, x_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()



def plot_position_vs_time(times, x, y, turn_index, stopped_index, turned, savepath="position_vs_time.png"):
    """
    Plot x(t) and y(t) against time.

    Args:
        times (numpy.ndarray): The time values.
        x (numpy.ndarray): The x-coordinates of the drone trajectory.
        y (numpy.ndarray): The y-coordinates of the drone trajectory.
        turn_index (int): The index of the time when the drone turns around.
        stopped_index (int): The index of the time when the drone stops.
        turned (bool): Whether the drone has turned around.
        savepath (str): The path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(times, x, lw=2, label="x(t)")
    plt.plot(times, y, lw=2, label="y(t)")

    if turned:
        plt.axvline(times[turn_index], color="orange", ls="--", label="Turn decision")
        plt.axvline(times[stopped_index], color="purple", ls=":", label="Stopped")

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Position vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_speed_vs_time(times, speed, turn_index, stopped_index, turned, savepath="speed_vs_time.png"):
    """
    Plot speed of drone against time.

    Args:
        times (numpy.ndarray): The time values.
        speed (numpy.ndarray): The speed of the drone.
        turn_index (int): The index of the time when the drone turns around.
        stopped_index (int): The index of the time when the drone stops.
        turned (bool): Whether the drone has turned around.
        savepath (str): The path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(times, speed, lw=2, label="|v(t)|")

    if turned:
        plt.axvline(times[turn_index], color="orange", ls="--", label="Turn decision")
        plt.axvline(times[stopped_index], color="purple", ls=":", label="Stopped")

    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("Speed vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_position_and_speed_vs_time(
    times,
    x,
    y,
    speed,
    turn_index,
    stopped_index,
    turned,
    savepath="position_and_speed_vs_time.png",
):
    """
    Compound figure: Position vs time (top) and Speed vs time (bottom) with shared Time axis.

    Args:
        times (numpy.ndarray): The time values.
        x (numpy.ndarray): The x-coordinates of the drone trajectory.
        y (numpy.ndarray): The y-coordinates of the drone trajectory.
        speed (numpy.ndarray): The speed of the drone.
        turn_index (int): The index of the time when the drone turns around.
        stopped_index (int): The index of the time when the drone stops.
        turned (bool): Whether the drone has turned around.
        savepath (str): The path to save the plot.

    Returns:
        None
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, dpi=300)
    fig.subplots_adjust(hspace=0.08)

    # Top: Position vs time
    ax_top.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax_top.plot(times, x, color="blue", lw=2, label="x(t)")
    ax_top.plot(times, y, color="orange", lw=2, label="y(t)")
    if turned:
        ax_top.axvline(times[turn_index], color="orange", ls="--", lw=2, label="Turn decision")
        ax_top.axvline(times[stopped_index], color="purple", ls=":", lw=2, label="Stopped")
    ax_top.set_ylabel("Position")
    ax_top.set_title("Position vs time")
    ax_top.legend(loc="upper right")

    ax_bot.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax_bot.plot(times, speed, color="blue", lw=2, label="|v(t)|")
    if turned:
        ax_bot.axvline(times[turn_index], color="orange", ls="--", lw=2, label="Turn decision")
        ax_bot.axvline(times[stopped_index], color="purple", ls=":", lw=2, label="Stopped")
    ax_bot.set_xlabel("Time")
    ax_bot.set_ylabel("Speed")
    ax_bot.set_title("Speed vs time")
    ax_bot.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_energy_used_vs_time(
    times,
    e,
    e_max,
    turn_index=None,
    stopped_index=None,
    turned=False,
    savepath="energy_used_vs_time.png",
):
    """
    Plot energy used over time with max energy line, energy-remaining fill,
    and optional turn/stopped markers.

    Args:
        times (numpy.ndarray): The time values.
        e (numpy.ndarray): The energy used by the drone.
        e_max (float): Maximum energy budget (horizontal line and fill ceiling).
        turn_index (int, optional): Index for turn-decision vertical line.
        stopped_index (int, optional): Index for stopped vertical line.
        turned (bool): Whether to draw turn/stopped vertical lines.
        savepath (str): The path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(9, 4), dpi=300)
    plt.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

    plt.fill_between(
        times,
        e,
        e_max,
        color="green",
        alpha=0.25,
        label="Energy remaining",
    )
    plt.plot(times, e, color="blue", lw=2, label="e(t)")
    plt.axhline(y=e_max, color="red", linestyle="--", lw=2, label=f"Max energy ({e_max})")

    if turned and turn_index is not None and stopped_index is not None:
        plt.axvline(
            times[turn_index],
            color="orange",
            ls="--",
            lw=2,
            label="Turn decision",
        )
        plt.axvline(
            times[stopped_index],
            color="purple",
            ls=":",
            lw=2,
            label="Stopped",
        )

    plt.xlabel("Time")
    plt.ylabel("Energy used")
    plt.title("Energy used vs time")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = ["e(t)", f"Max energy ({e_max})", "Energy remaining"]
    if turned and turn_index is not None and stopped_index is not None:
        order.extend(["Stopped", "Turn decision"])
    reordered = [(h, l) for l in order for h, lab in zip(handles, labels) if lab == l]
    plt.legend(
        [h for h, _ in reordered],
        [l for _, l in reordered],
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_energy_to_return(e_turn_times, e_turn_tracker, expected_e_turn, e_used_tracker, e_max, savepath="energy_to_return.png"):
    """
    Plot expected energy to return over time.

    Args:
        e_turn_times (numpy.ndarray): The time values when the drone turns around.
        e_turn_tracker (numpy.ndarray): The energy used by the drone when turning around.
        expected_e_turn (numpy.ndarray): The expected energy used by the drone when turning around.
        e_used_tracker (numpy.ndarray): The energy used by the drone over time.
        e_max (float): Maximum energy budget (horizontal line and fill ceiling).
        savepath (str): The path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(e_turn_times, e_turn_tracker, lw=2, label="Actual e_{turn}(t) + epsilon")
    plt.plot(e_turn_times, expected_e_turn, "--", lw=2, label="Expected e_turn(t) + epsilon")
    plt.plot(e_turn_times, e_used_tracker, lw=2, label="Available energy e_max(t) - e_used(t)")
    plt.axhline(y=e_max, color="red", linestyle="--", lw=2, label=f"Max energy ({e_max})")

    plt.xlabel("Time")
    plt.ylabel("Energy to return")
    plt.title("Energy to return vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_energy_error_tolerance(e_turn_times, e_turn_tracker, expected_e_turn, savepath="energy_error_tolerance.png"):
    """
    Plot the error energy needed to return within the scale of an acceptable error range
        
    Args:
        e_turn_times (numpy.ndarray): The time values when the drone turns around.
        e_turn_tracker (numpy.ndarray): The energy used by the drone when turning around.
        expected_e_turn (numpy.ndarray): The expected energy used by the drone when turning around.
        savepath (str): The path to save the plot.

    Returns:
        None
    """
    tau1 = 1e-6
    tau2 = 1e-5
    error = np.abs(e_turn_tracker - expected_e_turn)
    scale = 1e6

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(e_turn_times, error * scale, lw=2, label=r"$|e_{\mathrm{actual}}(t) - e_{\mathrm{expected}}(t)|$")
    plt.axhline(y=tau1 * scale, color="red",  linestyle="--", lw=2, label=rf"$\tau = {tau1:g}$")
    plt.axhline(y=tau2 * scale, linestyle="--", lw=2, label=rf"$\tau = {tau2:g}$")
    plt.fill_between(e_turn_times, 0, tau2 * scale, alpha=0.2, label="Acceptable Error")
    plt.xlabel("Time")
    plt.ylabel("Error Range $\t *(10^{-6})$ ")
    plt.title("Error Calculated Within Energy to Return Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
