"""
Coverage Density Heatmap for Adaptive Angular Bisection Validation

Runs the full adaptive_model without a reachable target so the algorithm
exhausts all search directions. Collects every scan circle center from every
linear search pass, then renders a 2D Cartesian coverage density map showing
how many scan circles overlap each point in the search area.

Expected appearance:
- Bright/hot core at the origin (every search vector passes through it)
- Hot blobs along each search vector direction (0°, 180°, 90°, 270°, 45°, ...)
- Overlap count drops as radius increases since circles spread apart angularly
- Cool/dark regions in angular gaps between vectors that tighten as bisection adds passes
- Everything clipped to the max search radius circle

Usage:
    python coverage_heatmap.py

Authors: Alisha Manocha, Aydin Khan, Kamran Hussain, Reagan Ross
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from params import Parameters
from adaptive_bisection import adaptive_model


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_scan_centers(all_results: list, params) -> np.ndarray:
    """
    Pull every (x, y) scan center out of the adaptive_model result list.

    Each element of all_results is the tuple returned by simulate_search_vector:
        (full_trajectory, full_times, e_used, e_turn, e_turn_times,
         turned, located, turn_index, stopped_index, scan_indices)

    scan_indices contains the trajectory row indices where a scan occurred.
    """
    centers = [(float(params.X0), float(params.Y0))] # the drone starts here
    for result in all_results:
        traj = result[0] # shape (N, 5): [x, y, vx, vy, e]
        scan_idxs = result[9] # list of int indices into traj
        for idx in scan_idxs:
            x, y = float(traj[idx, 0]), float(traj[idx, 1])
            centers.append((x, y))
    return np.array(centers, dtype=float) # shape (M, 2)


def build_coverage_grid(
    scan_centers: np.ndarray,
    scan_radius: float,
    max_radius: float,
    grid_resolution: int = 4800,
) -> tuple:
    """
    For every point on a fine 2D Cartesian grid, count how many scan circles
    (each of radius scan_radius) contain that point.

    Returns:
        xx, yy -- meshgrid arrays (grid_resolution × grid_resolution)
        counts -- integer array of the same shape, NaN outside max_radius
    """
    
    # Build a square grid that covers the search circle
    lin = np.linspace(-max_radius, max_radius, grid_resolution)
    xx, yy = np.meshgrid(lin, lin)

    # Mask out points outside the search circle
    inside = (xx ** 2 + yy ** 2) <= max_radius ** 2

    counts = np.zeros_like(xx, dtype=float)

    # For each scan center, add 1 to every grid cell within scan_radius
    # Vectorised: shape (M, G, G) would be too large for M big, so we batch
    # by chunking scan_centers if there are many.
    r2 = scan_radius ** 2
    chunk = 200  # process this many scan circles at once
    for start in range(0, len(scan_centers), chunk):
        batch = scan_centers[start : start + chunk] # (B, 2)
        # dx[b, i, j] = xx[i,j] - cx[b]
        dx = xx[None, :, :] - batch[:, 0, None, None] # (B, G, G)
        dy = yy[None, :, :] - batch[:, 1, None, None]
        dist2 = dx * dx + dy * dy # (B, G, G)
        counts += np.sum(dist2 <= r2, axis=0).astype(float)

    # Mask outside the search area with NaN so it renders as white
    counts[~inside] = np.nan

    return xx, yy, counts


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_coverage_heatmap(
    all_results: list,
    params,
    max_dist_rad: float,
    savepath: str = "coverage_heatmap.png",
    grid_resolution: int = 4800,
) -> None:
    """
    Build and save the coverage density heatmap.

    Parameters
    ----------
    all_results   : output of adaptive_model
    params        : Parameters dataclass instance
    max_dist_rad  : maximum radius the drone can reach (from adaptive_model)
    savepath      : where to save the PNG
    grid_resolution : number of grid cells along each axis (higher = sharper)
    """
    scan_centers = extract_scan_centers(all_results, params)
    print(f" Total scans across all passes: {len(scan_centers)}")

    print("Making heatmap")
    xx, yy, counts = build_coverage_grid(
        scan_centers,
        scan_radius=params.R_SCAN,
        max_radius=max_dist_rad,
        grid_resolution=grid_resolution,
    )

    # ------------------------------------------------------------------
    # Colormap: dark blue (0 overlap) -> cyan -> yellow -> red (higher overlap)
    # ------------------------------------------------------------------
    cmap = LinearSegmentedColormap.from_list(
        "coverage",
        [
            (0.05, 0.05, 0.35), # deep navy -> 0 (gaps)
            (0.00, 0.50, 0.80), # steel blue -> low
            (0.00, 0.85, 0.85), # cyan
            (0.20, 0.90, 0.20), # green
            (1.00, 0.90, 0.00), # yellow
            (1.00, 0.45, 0.00), # orange
            (0.85, 0.00, 0.00), # red -> high overlap
        ],
    )
    cmap.set_bad(color="white") # NaN (outside of the circle) is white
    cmap.set_under(color="white") # count = 0 gaps render white

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    valid_counts = counts[~np.isnan(counts)]
    vmin = 0.5  # anything below 0.5 (like count = 0) triggers set_under to make it white
    vmax = float(np.percentile(valid_counts, 98)) if len(valid_counts) > 0 else 1
    if vmax < 1: vmax = float(np.max(valid_counts)) if len(valid_counts) > 0 else 1

    im = ax.pcolormesh(
        xx, yy, counts,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of overlapping scan circles", fontsize=11)

    # Search boundary circle
    boundary = patches.Circle(
        (params.X0, params.Y0),
        max_dist_rad,
        fill=False,
        edgecolor="0.4",
        linestyle="--",
        linewidth=1.8,
        zorder=5,
        label=f"Max search radius ({max_dist_rad:.2f} m)",
    )
    ax.add_patch(boundary)

    # Scan center dots (small might make bigger later)
    if len(scan_centers) > 0:
        ax.scatter(
            scan_centers[:, 0],
            scan_centers[:, 1],
            s=6,
            c="white",
            edgecolors="gray",
            linewidths=0.3,
            alpha=0.55,
            zorder=6,
            label=f"Scan centers (n={len(scan_centers)})",
        )

    # Origin marker
    ax.scatter(
        [params.X0], [params.Y0],
        s=80, c="white", marker="*",
        edgecolors="black", linewidths=0.8,
        zorder=7, label="Origin",
    )

    # Target marker (only shown if within reachable area)
    if np.hypot(params.XL - params.X0, params.YL - params.Y0) <= max_dist_rad * 1.5:
        ax.scatter(
            [params.XL], [params.YL],
            s=120, c="purple", marker="o",
            edgecolors="white", linewidths=0.8,
            zorder=8, label="Target",
        )

    ax.set_xlim(-max_dist_rad * 1.12, max_dist_rad * 1.12)
    ax.set_ylim(-max_dist_rad * 1.12, max_dist_rad * 1.12)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(
        "Coverage Density Heatmap: Angular Bisection\n"
        f"(n={len(scan_centers)} scans, "
        f"$r_{{\\mathrm{{scan}}}}$={params.R_SCAN} m, "
        f"$R_{{\\mathrm{{max}}}}$={max_dist_rad:.2f} m)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(False)

    plt.tight_layout()
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {savepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Place the target outside the reachable area so the algorithm
    # this fully exhausts all search directions so it doesnt stop early
    params = Parameters(
        X0=0.0,
        Y0=0.0,
        XL=999.0,
        YL=999.0,
        R_SCAN=0.25,
        T=1.0,
        M=1.0,
        EH=1.0,
        ES=0.3,
        E_MAX=35.0,
        DT=1e-3,
        EPS=5e-2,
    )

    all_results = adaptive_model(
        params,
        rad_search=params.R_SCAN,
        max_dist_rad=None,
        point_list=None,
        max_arclength=None,
    )
    print(f"Model finished. Total search passes: {len(all_results)}")

    # Recover max_dist_rad from the first result's trajectory
    first_traj = all_results[0][0]
    r_vals = np.hypot(
        first_traj[:, 0] - params.X0,
        first_traj[:, 1] - params.Y0,
    )
    max_dist_rad = float(np.max(r_vals))
    print(f"max_dist_rad recovered: {max_dist_rad:.4f} m")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_coverage_heatmap(
        all_results=all_results,
        params=params,
        max_dist_rad=max_dist_rad,
        savepath=str(PLOTS_DIR / "coverage_heatmap.png"),
        grid_resolution=4800,
    )


if __name__ == "__main__":
    main()
