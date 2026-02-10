import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

"""This is the existing model for moving with a quadratic velocity from a starting point to an
ending point. We solve five ODEs: dx/dt = vx, dy/dt = vy, dvx/dt = Fx/m, dvy/dt=Fy/m, and
de/dt = EH + |F * v|. This function, to be passed into solve_ivp, takes the current time,
state [x, y, vx, vy, e], and simulation parameters as inputs and returns the derivatives of each
element in the state.
"""
def forward_odes(t, state, params):
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
def check_return_energy(state, T, m, EH, x0, y0):
    # Get the current state of the drone after coming to a stop
    x, y, vx, vy, e = state
    # Pass in current position as starting point, initial starting point of the whole journey as
    # ending point, flight time, mass, hovering energy
    params = [x, y, x0, y0, T, m, EH]

    # Use the current position, velocity = 0, and energy = 0 as to not include energy already used
    # in returning energy estimate
    initial_state = [x, y, 0, 0, 0]
    # Simulate from time 0 to flight time
    t0 = 0.0
    tspan = np.linspace(t0, T, 100)

    sol = solve_ivp(forward_odes, (t0, T), initial_state, args=(params,), t_eval=tspan, method="RK45", rtol=1e-8, atol=1e-10)

    if sol.success:
        # Return the amount of energy used to return
        return sol.y[:, -1]
    else:
        raise RuntimeError("Returning integration failed")

"""Check whether, at the current time, the drone has enough energy remaining to come to a stop in a
fixed amount of time and then return from that point back to the starting point. The energy margin
that would be left after performing this return is compared to a given epsilon, below which the drone
should turn back. Returns the difference between the margin and epsilon."""
def check_turn(t, state, e_max, eps, ts, T, m, EH, x0, y0, e_turn):
    # Get the amount of energy used thus far
    e_used = state[4]
    # Compute the amount of energy the drone has left
    e_left = e_max - e_used

    # Compute the amount of energy needed to stop
    stopped_state = check_stop_energy(state, ts, m, EH)
    e_stop = stopped_state[4]

    # Compute the amount of energy needed to return from the state after stopping
    return_state = check_return_energy(stopped_state, T, m, EH, x0, y0)
    e_return = return_state[4]

    turn_energy = e_stop + e_return
    e_turn.append(turn_energy)

    # Compute energy margin after performing the return
    margin = e_left - (turn_energy)
    
    print(
        f"t={t:.3f}, "
        f"e_used={e_used:.3f}, "
        f"e_stop={e_stop:.3f}, "
        f"e_return={e_return:.3f}, "
        f"margin={margin:.3f}"
    )

    return margin - eps

def expected_return_energy(times, EH, ts, T, m, x0, y0, xT, yT):
    # Predicted velocity
    vx = ((6 * times * (T - times)) / (T**3)) * (xT - x0)
    vy = ((6 * times * (T - times)) / (T**3)) * (yT - y0)

    # Predicted trajectory
    x = x0 + ((3 * (times**2) / (T**2)) - (2 * (times**3) / (T**3))) * (xT - x0)
    y = y0 + ((3 * (times**2) / (T**2)) - (2 * (times**3) / (T**3))) * (yT - y0)

    expected_e_turn = EH * (T + ts) + (m * (vx**2 + vy**2) / 2) + ((9 * m) / (4 * T**2)) * ((x0 - x - ((vx * ts) / 2))**2 + (y0 - y - ((vy * ts) / 2))**2)

    return expected_e_turn

def main():
    (x0, y0) = (0, 0) # Starting point
    (xT, yT) = (1, 2) # Destination point
    T = 1 # Flight time from start to destination or vice versa
    m = 1 # Drone mass
    EH = 1 # Hovering energy
    e_max = 12.5 # Maximum energy the drone can use
    ts = T / 20 # Time for the drone to come to a stop midway, if necessary
    eps = 5e-2 # Threshold used to determine if the drone should return midway (if energy margin < eps)

    params = [x0, y0, xT, yT, T, m, EH] # Pack parameters into a list

    state = [x0, y0, 0, 0, 0] # Initial state

    dt = T / 1000 # Time interval at which to check if drone should return
    t = 0

    # Keep track of drone's full trajectory over time
    times = [t]
    trajectory = [state.copy()]
    e_turn = []
    e_turn_times = np.array([])

    # Keep track of if drone turned, and if so, which index in trajectory represents that point in time
    turned = False
    turn_index = None
    stopped_index = None

    while t < T:
        # Solve system of ODEs over a small time span with the current state and parameters
        sol_forward = solve_ivp(
            forward_odes,
            (t, t+dt),
            state,
            args=(params,),
            method="RK45",
            max_step=dt,
            rtol=1e-8,
            atol=1e-10
        )

        if not sol_forward.success:
            raise RuntimeError("Forward flight integration failed")

        # Add all times and state arrays into trajectory tracker
        for i in range(1, sol_forward.y.shape[1]):
            trajectory.append(sol_forward.y[:, i])
            times.append(sol_forward.t[i])
 
        # Update current state and time
        state = sol_forward.y[:, -1]
        t = sol_forward.t[-1]

        # Get difference between energy margin that would remain after returning and epsilon
        turn = check_turn(t, state, e_max, eps, ts, T, m, EH, x0, y0, e_turn)
        e_turn_times = np.append(e_turn_times, t)
        # If negative or zero, margin is less than or equal to epsilon, so need to return
        if turn <= 0:
            print("Turning around just before there is insufficient energy to return")
            turned = True
            turn_index = len(trajectory) - 1
            break

    # The loop was broken out of because the drone needed to return midway
    if turned:
        # Get the state at the point of deciding to return
        turn_state = trajectory[turn_index]

        # Pass in velocity at the time of starting to stop, amount of time to stop, mass, hovering
        # energy as parameters
        params = [turn_state[2], turn_state[3], ts, m, EH]

        # Come to a stop
        sol_stop = solve_ivp(
            stop_odes,
            (0, ts),
            turn_state,
            args=(params,),
            method="RK45",
            max_step=ts/50,
            rtol=1e-8,
            atol=1e-10
        )

        # Add all times and state arrays into trajectory tracker, offsetting the times because
        # we simulated from time 0 rather than the actual current time
        t_offset = times[-1]
        for i in range(1, sol_stop.y.shape[1]):
            trajectory.append(sol_stop.y[:, i])
            times.append(t_offset + sol_stop.t[i])
    else:
        print("Reached destination! turning around now")

    # Get state from which the drone stopped, whether that's midway or at the destination
    stopped_index = len(trajectory) - 1
    stopped_state = trajectory[stopped_index]

    print("stopped state:", stopped_state)

    # Pass in stopped position as initial position, initial starting point as ending position, flight
    # time, mass, and hovering energy as parameters
    params = [stopped_state[0], stopped_state[1], x0, y0, T, m, EH]

    # Return to initial point
    sol_return = solve_ivp(
        forward_odes,
        (0, T),
        stopped_state,
        args=(params,),
        method="RK45",
        max_step=T/200,
        rtol=1e-8,
        atol=1e-10
    )

    print("ending state:", sol_return.y[:, -1])

    # Add all times and state arrays into trajectory tracker, offsetting the times because we
    # simulated from time 0 rather than the actual current time
    t_offset = times[-1]
    for i in range(1, sol_return.y.shape[1]):
        trajectory.append(sol_return.y[:, i])
        times.append(t_offset + sol_return.t[i])

    # Convert the full trajectory and times of the whole journey to a numpy array and get each state component
    trajectory = np.array(trajectory)
    times = np.array(times)
    e_turn = np.array(e_turn)

    x, y = trajectory[:, 0], trajectory[:, 1]
    vx, vy = trajectory[:, 2], trajectory[:, 3]
    speed = np.hypot(vx, vy) # Speed = |v| = sqrt(vx^2 + vy^2)
    e = trajectory[:, 4]
    expected_e_turn = expected_return_energy(e_turn_times, EH, ts, T, m, x0, y0, xT, yT)

    # Plot parametric solution of x(t) vs. y(t)
    plt.figure(figsize=(7, 7))

    if turned:
        # If drone turned around midway, split up the forward trajectory and stopping phase in graph
        plt.plot(x[:turn_index + 1], y[:turn_index + 1], lw=2.5, label="Forward")
        plt.plot(x[turn_index:stopped_index + 1],
                 y[turn_index:stopped_index + 1],
                 ":", lw=2, label="Stopping")
    else:
        # Drone made it to destination, plot the forward trajectory
        plt.plot(x, y, lw=2.5, label="Forward")

    # Plot the return trajectory from the point where the drone stopped, whether midway or not
    plt.plot(x[stopped_index:], y[stopped_index:], "--", lw=2.5, label="Return")

    # Mark starting, destination, and final points
    plt.scatter(x0, y0, c="green", s=80, label="Start")
    plt.scatter(xT, yT, c="red", s=80, label="Target")
    plt.scatter(x[-1], y[-1], c="black", s=120, marker="*", label="Final")

    if turned:
        # If drone turned around midway, mark point where drone decided to turn around and stopped point
        plt.scatter(x[turn_index], y[turn_index],
                    c="orange", s=100, marker="x", label="Turn decision")
        plt.scatter(x[stopped_index], y[stopped_index],
                    c="purple", s=100, marker="s", label="Stopped")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Drone trajectory (parametric)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot x(t) and y(t) against time
    plt.figure(figsize=(9, 4))
    plt.plot(times, x, lw=2, label="x(t)")
    plt.plot(times, y, lw=2, label="y(t)")

    if turned:
        # If drone turned around midway, mark point where drone decided to turn around and stopped point
        plt.axvline(times[turn_index], color="orange", ls="--", label="Turn decision")
        plt.axvline(times[stopped_index], color="purple", ls=":", label="Stopped")

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Position vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot speed of drone against time
    plt.figure(figsize=(9, 4))
    plt.plot(times, speed, lw=2, label="|v(t)|")

    if turned:
        # If drone turned around midway, mark point where drone decided to turn around and stopped point
        plt.axvline(times[turn_index], color="orange", ls="--", label="Turn decision")
        plt.axvline(times[stopped_index], color="purple", ls=":", label="Stopped")

    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("Speed vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot energy used over time
    plt.figure(figsize=(9, 4))
    plt.plot(times, e, lw=2, label="e(t)")

    plt.xlabel("Time")
    plt.ylabel("Energy used")
    plt.title("Energy used vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot expected energy to return over time
    plt.figure(figsize=(9, 4))
    plt.plot(e_turn_times, e_turn, lw=2, label="Actual e_{turn}(t)")
    plt.plot(e_turn_times, expected_e_turn, "--", lw=2, label="Expected e_turn(t)")

    plt.xlabel("Time")
    plt.ylabel("Energy to return")
    plt.title("Energy to return vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the amount of energy used and energy left
    e_used = trajectory[-1, 4]
    e_left = e_max - e_used
    print("Final energy used:", e_used)
    print("Energy left:", e_left)


if __name__ == "__main__":
    main()
