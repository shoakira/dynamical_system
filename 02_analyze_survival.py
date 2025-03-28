# -*- coding: utf-8 -*-
"""
02_analyze_survival.py

Analyzes simulation results to calculate and plot survival probability.
Reads data from the .npz file generated by 01_run_simulation.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ==============================================================================
# Configuration
# ==============================================================================

# --- Input Data ---
# Automatically determine filename based on parameters (or set manually)
E = 1.05             # Must match the simulation E
GAMMA = 0.5           # Must match the simulation GAMMA
DATA_FOLDER = "data_3dof_organized"
INPUT_FILENAME = f"simulation_results_E{E:.3f}_gamma{GAMMA}.npz"
INPUT_FILEPATH = os.path.join(DATA_FOLDER, INPUT_FILENAME)

# --- Plotting ---
SAVE_PLOTS = True
PLOT_FOLDER = DATA_FOLDER # Save plots in the same data folder

# ==============================================================================
# Analysis Functions
# ==============================================================================

def load_simulation_data(filepath):
    """Loads simulation results from a .npz file."""
    print(f"Loading simulation data from: {filepath}")
    try:
        data = np.load(filepath)
        print("Data loaded successfully. Contents:", list(data.keys()))
        # Extract necessary data
        rec_times = data['recross_times']
        n_total = data['config_N_TRAJ'].item() # Use actual simulated number
        t_max_sim = data['config_T_MAX'].item()
        dt_sim = data['config_DT'].item()
        # Store config for plot titles etc.
        config = {
            'E': data['config_E'].item(),
            'GAMMA': data['config_GAMMA'].item(),
            'N_TRAJ': n_total,
            'T_MAX': t_max_sim,
            'DT': dt_sim
        }
        return rec_times, n_total, t_max_sim, dt_sim, config
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

def calculate_survival(recross_times, t_max_, dt_, n_total_traj):
    """Calculates the survival fraction P(t) over a log-spaced time axis."""
    if recross_times is None or len(recross_times) == 0:
        return None, None

    recrossed_mask = recross_times < t_max_
    n_recrossed = np.sum(recrossed_mask)
    print(f"  Recrossing trajectories found: {n_recrossed}")

    if n_recrossed == 0:
        print("  No recrossing trajectories found for survival analysis.")
        return None, None

    recrossed_times_only = recross_times[recrossed_mask]
    # Ensure min_time_plot is positive and less than max_time_plot
    min_t = np.min(recrossed_times_only[recrossed_times_only > 0]) if np.any(recrossed_times_only > 0) else dt_
    min_time_plot = max(dt_ * 5, min_t * 0.5)
    max_time_plot = np.max(recrossed_times_only)

    if max_time_plot <= min_time_plot:
        print(f"  Warning: Recrossing time range too narrow [{min_time_plot:.2e}, {max_time_plot:.2e}]. Adjusting max.")
        max_time_plot = min_time_plot * 100
        if max_time_plot > t_max_: max_time_plot = t_max_

    print(f"  Plotting time range: [{min_time_plot:.2e}, {max_time_plot:.2e}]")
    t_plot = np.logspace(np.log10(min_time_plot), np.log10(max_time_plot), 100)

    # Calculate survival fraction P(t) = N(time > t) / N_total
    survival_fraction = np.array([np.sum(recross_times > t) / n_total_traj for t in t_plot])

    return t_plot, survival_fraction

def plot_survival(t_plot, survival_fraction, config, plot_folder_):
    """Plots the survival fraction P(t) on semi-log and log-log scales."""
    if t_plot is None or survival_fraction is None:
        print("Skipping survival plot generation.")
        return

    energy = config['E']
    gamma_ = config['GAMMA']
    n_total_traj = config['N_TRAJ']

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Semi-log plot
    axs[0].semilogx(t_plot, survival_fraction, marker='.', linestyle='-')
    axs[0].set_xlabel("Time (log scale)")
    axs[0].set_ylabel("Survival Fraction P(t)")
    axs[0].set_title("Survival Probability (Semi-log)")
    axs[0].grid(True, which="both", ls="--", alpha=0.6)
    axs[0].set_ylim(0, max(1.05, np.max(survival_fraction)*1.05))

    # Log-log plot
    axs[1].loglog(t_plot, survival_fraction, marker='.', linestyle='-')
    axs[1].set_xlabel("Time (log scale)")
    axs[1].set_ylabel("Survival Fraction P(t) (log scale)")
    axs[1].set_title("Survival Probability (Log-log)")
    axs[1].grid(True, which="both", ls="--", alpha=0.6)
    min_survival_plot = max(1e-6, 0.5 / n_total_traj)
    # Adjust y-axis limits based on data range
    max_y = max(1.05, np.max(survival_fraction) * 1.1) if len(survival_fraction) > 0 else 1.05
    min_y = min(min_survival_plot, np.min(survival_fraction[survival_fraction > 0]) * 0.5) if np.any(survival_fraction > 0) else min_survival_plot
    axs[1].set_ylim(bottom=min_y, top=max_y)


    plt.suptitle(f"3DoF Recrossing Survival (E={energy:.3f}, gamma={gamma_}, N={n_total_traj})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"survival_3dof_E{energy:.3f}_gamma{gamma_}.pdf"
    filepath = os.path.join(plot_folder_, filename)
    try:
        plt.savefig(filepath)
        print(f"Survival plot saved to '{filepath}'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Loads data and generates survival plots."""
    start_time = time.time()
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    # 1. Load Data
    rec_times, n_total, t_max_sim, dt_sim, config = load_simulation_data(INPUT_FILEPATH)
    if rec_times is None:
        return # Stop if loading failed

    # 2. Calculate Survival
    print("\nCalculating survival probability...")
    t_plot, survival_frac = calculate_survival(rec_times, t_max_sim, dt_sim, n_total)

    # 3. Plot Survival
    if SAVE_PLOTS:
        print("\nPlotting survival probability...")
        plot_survival(t_plot, survival_frac, config, PLOT_FOLDER)

    print(f"\nSurvival analysis finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
