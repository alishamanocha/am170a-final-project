"""
Plotting Functions

Script that visualizes the drone simulation results and produces the figures linked in the report.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Roberto Julian Campos, Kamran Hussain
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_parametric(
    x, y, x0, y0, xT, yT, turn_index, stopped_index, turned, savepath="parametric_trajectory.png"
):
    """
    Plot parametric solution of x(t) vs. y(t).

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
    plt.figure(figsize=(7, 7), dpi=300)

    if turned:
        plt.plot(x[: turn_index + 1], y[: turn_index + 1], lw=2.5, label="Forward")
        plt.plot(
            x[turn_index : stopped_index + 1],
            y[turn_index : stopped_index + 1],
            ":",
            lw=2,
            label="Stopping",
        )
    else:
        plt.plot(x, y, lw=2.5, label="Forward")

    plt.plot(x[stopped_index:], y[stopped_index:], "--", lw=2.5, label="Return")

    plt.scatter(x0, y0, c="green", s=80, label="Start")
    plt.scatter(xT, yT, c="red", s=80, label="Target")
    plt.scatter(x[-1], y[-1], c="black", s=120, marker="*", label="Final")

    if turned:
        plt.scatter(
            x[turn_index],
            y[turn_index],
            c="orange",
            s=100,
            marker="x",
            label="Turn decision",
        )
        plt.scatter(
            x[stopped_index],
            y[stopped_index],
            c="purple",
            s=100,
            marker="s",
            label="Stopped",
        )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Drone trajectory (parametric)")
    plt.legend()
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
    if turned and turn_index is not None and stopped_index is not None:
        ax_top.axvline(times[turn_index], color="orange", ls="--", lw=2, label="Turn decision")
        ax_top.axvline(times[stopped_index], color="purple", ls=":", lw=2, label="Stopped")
    ax_top.set_ylabel("Position")
    ax_top.set_title("Position vs time")
    ax_top.legend(loc="upper right")

    # Bottom: Speed vs time
    ax_bot.grid(True, color="lightgray", linestyle="-", linewidth=0.5)
    ax_bot.plot(times, speed, color="blue", lw=2, label="|v(t)|")
    if turned and turn_index is not None and stopped_index is not None:
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
