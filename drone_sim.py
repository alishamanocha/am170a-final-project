"""
Drone Simulation Model

A script to simulate the flight of a drone from a starting point to an ending point and returning back to the
starting point. It takes into account the energy used by the drone during the flight and provides a simulation
to check available energy and come to a stop if needed.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Roberto Julian Campos, Kamran Hussain
"""

import numpy as np
from scipy.integrate import solve_ivp

def forward_odes(t, state, params):
    """
    This is the existing model for moving with a quadratic velocity from a starting point to an
    ending point. We solve five ODEs: dx/dt = vx, dy/dt = vy, dvx/dt = Fx/m, dvy/dt=Fy/m, and
    de/dt = EH + |F * v|. This function, to be passed into solve_ivp, takes the current time,
    state [x, y, vx, vy, e], and simulation parameters as inputs and returns the derivatives of each
    element in the state.

    Args:
        t (float): The current time.
        state (list): The current state of the drone.
        params (list): The simulation parameters.

    Returns:
        list: The derivatives of each element in the state.
    """
    # Get simulation parameters - starting and ending point, flight time, mass, hovering energy
    x0, y0, xT, yT, T, m, EH = params
    # Get current state - position, velocity, and energy used
    x, y, vx, vy, e = state

    # Compute acceleration in x- and y- directions
    ax = (6 * (T - 2*t) / T**3) * (xT - x0)
    ay = (6 * (T - 2*t) / T**3) * (yT - y0)

    # Compute force by multiplying acceleration by mass
    Fx = m * ax
    Fy = m * ay

    # Compute derivative of energy
    dedt = EH + abs(Fx * vx + Fy * vy)

    return [vx, vy, ax, ay, dedt]

"""This is the model for coming to a stop with constant acceleration over a fixed amount of time.
We solve five ODESs: dx/dt = vx, dy/dt = vy, dvx/dt = Fx/m, dvy/dt=Fy/m, and de/dt = EH + |F * v|.
This function, to be passed into solve_ivp, takes the current time, state [x, y, vx, vy, e], and
simulation parameters as inputs and returns the derivatives of each element in the state.
"""
def stop_odes(t, state, params):
    # Get simulation parameters - velocity at the time of starting to stop, amount of time to stop,
    # mass, hovering energy
    vx0, vy0, ts, m, EH = params
    # Get current state - position, velocity, and energy used
    x, y, vx, vy, e = state

    # Compute acceleration in x- and y- directions
    ax = -vx0 / ts
    ay = -vy0 / ts

    # Compute force by multiplying acceleration by mass
    Fx = m * ax
    Fy = m * ay

    # Compute derivative of energy
    dedt = EH + abs(Fx * vx + Fy * vy)

    return [vx, vy, ax, ay, dedt]

"""Estimate the amount of energy needed to come to a stop given a current state of the drone by
using solve_ivp to simulate stopping."""
def check_stop_energy(state, ts, m, EH):
    # Get the current state of the drone in its flight
    x, y, vx, vy, e = state
    # Pass in current velocity, time to stop, mass, and hovering energy as parameters
    params = [vx, vy, ts, m, EH]

    # Use the current state as the initial state to simulate stopping from, but with energy = 0
    # as to not include energy already used in stopping energy estimate
    initial_state = [x, y, vx, vy, 0]
    # Simulate from time 0 to time needed to stop
    t0 = 0.0
    tspan = np.linspace(t0, ts, 100)

    sol = solve_ivp(stop_odes, (t0, ts), initial_state, args=(params,), t_eval=tspan, method="RK45", rtol=1e-8, atol=1e-10)

    if sol.success:
        # Return the amount of energy used to stop
        return sol.y[:, -1]
    else:
        raise RuntimeError("Stopping integration failed")

"""Estimate the amount of energy needed to return back to the starting point given a current state
of the drone by using solve_ivp to simulate flight forward."""
def check_return_energy(state, m, EH, x0, y0):
    # Get the current state of the drone after coming to a stop
    x, y, vx, vy, e = state

    # Compute the distance to the starting point
    dist_return = np.sqrt((x0 - x)**2 + (y0 - y)**2)
    # Compute the optimal return time
    t_r_star = (9 * m / (2 * EH)) ** (1/3) * dist_return ** (2/3)

    # Pass in current position as starting point, initial starting point of the whole journey as
    # ending point, optimal return time, mass, hovering energy
    params = [x, y, x0, y0, t_r_star, m, EH]

    # Use the current position, velocity = 0, and energy = 0 as to not include energy already used
    # in returning energy estimate
    initial_state = [x, y, 0, 0, 0]
    # Simulate from time 0 to flight time
    t0 = 0.0
    tspan = np.linspace(t0, t_r_star, 100)

    sol = solve_ivp(forward_odes, (t0, t_r_star), initial_state, args=(params,), t_eval=tspan, method="RK45", rtol=1e-8, atol=1e-10)

    if sol.success:
        # Return the amount of energy used to return
        return sol.y[:, -1]
    else:
        raise RuntimeError("Returning integration failed")

"""Check whether, at the current time, the drone has enough energy remaining to come to a stop in a
fixed amount of time and then return from that point back to the starting point, and if it has enough
energy to perform one more scan. The energy margin that would be left after performing this return is
compared to a given epsilon, below which the drone should turn back. Returns the difference between
the margin and epsilon."""
def check_energy_turn(t, state, e_max, eps, ts, m, EH, ES, x0, y0, e_turn_tracker, e_used_tracker):
    # Get the amount of energy used thus far
    e_used = state[4]

    # Compute the amount of energy the drone has left
    e_left = e_max - e_used

    # Compute the amount of energy needed to stop
    stopped_state = check_stop_energy(state, ts, m, EH)
    e_stop = stopped_state[4]

    # Compute the amount of energy needed to return from the state after stopping
    return_state = check_return_energy(stopped_state, m, EH, x0, y0)
    e_return = return_state[4]

    e_turn = e_stop + e_return

    # Add energy used and to return to trackers for later plotting
    e_used_tracker.append(e_used)
    e_turn_tracker.append(e_turn)

    # Compute energy margin after performing the return and one more scan
    margin = e_left - e_turn - ES
    
    # print(
    #     f"t={t:.3f}, "
    #     f"e_used={e_used:.3f}, "
    #     f"e_stop={e_stop:.3f}, "
    #     f"e_return={e_return:.3f}, "
    #     f"margin={margin:.3f}"
    # )

    if margin<eps:
        print(f"Need to return! Energy to return = {e_turn}, Energy to scan = {ES}, Energy available = {e_left}")

    return margin - eps

def get_location_distance(state, xL, yL):
    # Get the current position
    x = state[0]
    y = state[1]

    # Get distance from the location of the desired person/thing to find
    return np.hypot(xL - x, yL - y)

"""Compute the expected return energy using the analytical solution at all timestamps."""
def expected_return_energy(times, EH, ts, T, m, x0, y0, xT, yT):
    # Predicted velocity
    vx = ((6 * times * (T - times)) / (T**3)) * (xT - x0)
    vy = ((6 * times * (T - times)) / (T**3)) * (yT - y0)

    # Predicted trajectory
    x = x0 + ((3 * (times**2) / (T**2)) - (2 * (times**3) / (T**3))) * (xT - x0)
    y = y0 + ((3 * (times**2) / (T**2)) - (2 * (times**3) / (T**3))) * (yT - y0)

    stop_x = x + vx * ts / 2
    stop_y = y + vy * ts / 2
    dist_return = np.sqrt((x0 - stop_x)**2 + (y0 - stop_y)**2)
    t_r_star = (9 * m / (2 * EH)) ** (1/3) * dist_return ** (2/3)
    expected_e_turn = EH * (t_r_star + ts) + (m * (vx**2 + vy**2) / 2) + ((9 * m) / (4 * t_r_star**2)) * ((x0 - x - ((vx * ts) / 2))**2 + (y0 - y - ((vy * ts) / 2))**2)

    return expected_e_turn
