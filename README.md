# Drone Method and Validation Code

**Group:** Alisha Manocha, Reagan Ross, Aydin Khan, Roberto Julian Campos, Kamran Hussain

**Summary:** The drone must always keep enough energy to stop and return. A turn-around is triggered when the energy margin (remaining minus energy needed to stop and return) drops below ε.

---

## How it works

1. **Run:** `python main.py`
2. **Flow:** Start at (X_0,Y_0) → integrate **forward** toward target (X_T,Y_T) with small time steps. At each step, **check** whether there is still enough energy to stop and return; if not, **turn**, **stop**, then **return** to start. If the drone reaches the target, it stops there and returns.
3. **Output:** All plots are written to the `plots/` directory.

---

## Simulation Model (`drone_sim.py`)

| Function | Purpose |
|----------|--------|
| `forward_odes(t, state, params)` | ODEs for flight segment (position, velocity, energy). |
| `stop_odes(t, state, params)` | ODEs for deceleration to stop. |
| `check_stop_energy(state, ts, m, EH)` | Energy needed to stop from current state. |
| `check_return_energy(state, T, m, EH, x0, y0)` | Energy needed to return from (stopped) state to start. |
| `check_turn(t, state, e_max, eps, ...)` | Margin vs ε; turn when margin ≤ 0. |
| `expected_return_energy(times, EH, ts, T, m, x0, y0, xT, yT)` | Predicted energy-to-return profile. |

---

## Plots generated (`plots/`)

- **parametric_trajectory.png** — (x, y) path; forward / stop / return; start, target, turn/stopped markers.
- **position_and_speed_vs_time.png** — Position (x, y) and speed vs time (shared axis).
- **energy_used_vs_time.png** — Energy used, max energy line, remaining (shaded), turn/stopped lines.
- **energy_to_return.png** — Actual vs expected energy to return.
- **energy_error_tolerance.png** — Prediction error vs time with ±ε tolerance band.

---

## Contributions

- Main driver / IVP solver: Alisha  
- Readability / documentation: Kamran  
- Graphs: Kamran, Alisha, Aydin, Reagan  
