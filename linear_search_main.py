"""
Entry point for the linear search simulation.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Kamran Hussain
"""

from pathlib import Path

from linear_search import simulate_linear_search
from plotting import plot_linear_search_area


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"


def main() -> None:
    params = {
        "x0": 0.0,
        "y0": 0.0,
        "direction": (1.0, 0.35),
        "vector_length": 10.0,
        "scan_radius": 0.6,
        "e_max": 35.0,
        "eps": 5e-2,
        "move_time": 4.0,
        "mass": 1.0,
        "EH": 1.0,
        "search_time": 0.25,
        "seed": 0,
    }

    result = simulate_linear_search(params)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_linear_search_area(
        all_centers_xy=result["all_centers_xy"],
        searched_centers_xy=result["searched_centers_xy"],
        scan_radius=params["scan_radius"],
        base_xy=(params["x0"], params["y0"]),
        vector_end_xy=tuple(result["all_centers_xy"][-1].tolist()),
        target_xy=result["target_xy"],
        found=result["found"],
        sortie_paths=result["sortie_paths"],
        recharge_count=result["recharge_count"],
        max_one_search_distance=result["max_one_search_distance"],
        savepath=str(PLOTS_DIR / "linear_search_area.png"),
    )

    print("Saved:", str(PLOTS_DIR / "linear_search_area.png"))
    print("Found:", result["found"])
    print("Searches:", len(result["searched_centers_xy"]))
    print("Recharges:", result["recharge_count"])


if __name__ == "__main__":
    main()

