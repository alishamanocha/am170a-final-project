"""
Entry point for the linear search simulation.

Author: Alisha Manocha, Reagan Ross, Aydin Khan, Kamran Hussain
"""

from pathlib import Path

from linear_search import max_distance_one_search, simulate_linear_search
from plotting import plot_linear_search_area, plot_multi_direction_lidar_coverage


SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"


def run_linear_search_demo(*, savepath: str) -> None:
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
        savepath=savepath,
    )

    print("Saved:", savepath)
    print("Found:", result["found"])
    print("Searches:", len(result["searched_centers_xy"]))
    print("Recharges:", result["recharge_count"])


def run_multi_direction_coverage_demo(*, savepath: str) -> None:
    params = {
        "x0": 0.0,
        "y0": 0.0,
        "direction": (1.0, 0.0),  # unused by the coverage plot
        "vector_length": 1e9,  # avoid capping R_max by a short vector length
        "scan_radius": 0.35,
        "e_max": 35.0,
        "eps": 5e-2,
        "move_time": 4.0,
        "mass": 1.0,
        "EH": 1.0,
        "search_time": 0.25,
    }

    r_max = max_distance_one_search(params)
    directions = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.75),
    ]

    plot_multi_direction_lidar_coverage(
        base_xy=(params["x0"], params["y0"]),
        directions=directions,
        scan_radius=params["scan_radius"],
        max_radius=r_max,
        savepath=savepath,
    )

    print("Saved:", savepath)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    run_linear_search_demo(savepath=str(PLOTS_DIR / "linear_search_area.png"))
    run_multi_direction_coverage_demo(savepath=str(PLOTS_DIR / "multi_direction_lidar_coverage.png"))


if __name__ == "__main__":
    main()

