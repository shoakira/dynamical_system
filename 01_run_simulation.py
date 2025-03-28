# -*- coding: utf-8 -*-
"""
01_run_simulation.py

3自由度非線形ダブルウェルモデルのシミュレーション実行スクリプト

- パラメータ設定
- 物理モデル定義 (Numba)
- 積分関数定義 (Numba)
- 初期条件生成
- 並列シミュレーション実行
- 結果 (初期パラメータと再交差時間) をファイルに保存
"""

import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import numba as nb
import os
import time

# ==============================================================================
# 0. Configuration Settings
# ==============================================================================

# --- Simulation Parameters ---
E = 1.05             # Total energy (must be > 1.0)
N_TRAJ = 100000       # Number of trajectories
T_MAX = 100000.0       # Maximum simulation time
DT = 0.002            # Time step for integration

# --- Potential Parameters ---
GAMMA = 0.5           # Non-linear interaction strength
OMEGA_Y = 1.0         # Frequency in y-direction at saddle
OMEGA_Z = 1.0         # Frequency in z-direction at saddle

# --- Initial Condition Parameters (Approximate NHIM) ---
USE_FIXED_PX = False  # Use a fixed small px (True) or determine from energy (False)
PX_INIT_VAL = 1e-5    # Fixed value if USE_FIXED_PX is True

# --- Data Storage ---
DATA_FOLDER = "data_3dof_organized"
OUTPUT_FILENAME = f"simulation_results_E{E:.3f}_gamma{GAMMA}.npz"

# --- Parallelization ---
N_CORES = mp.cpu_count() # Use all available CPU cores

# ==============================================================================
# 1. Physics Model (Potential, Force, Hamiltonian) - Numba Accelerated
# ==============================================================================
print("Defining physics model...")

@nb.njit
def V(x, y, z, gamma_):
    """Potential energy function V(x, y, z)."""
    term_dw = (x**2 - 1)**2
    term_harm = 0.5 * (OMEGA_Y**2 * y**2 + OMEGA_Z**2 * z**2)
    term_int = gamma_ * x**2 * (y**2 + z**2)
    return term_dw + term_harm + term_int

@nb.njit
def force(x, y, z, gamma_):
    """Calculates the force F = -dV/dq."""
    fx = -(4 * x * (x**2 - 1) + 2 * gamma_ * x * (y**2 + z**2))
    fy = -(OMEGA_Y**2 * y + 2 * gamma_ * x**2 * y)
    fz = -(OMEGA_Z**2 * z + 2 * gamma_ * x**2 * z)
    return fx, fy, fz

@nb.njit
def hamiltonian(state, gamma_):
    """Calculates the Hamiltonian H(q, p)."""
    x, y, z, px, py, pz = state
    kinetic = 0.5 * (px**2 + py**2 + pz**2)
    potential = V(x, y, z, gamma_)
    return kinetic + potential

# ==============================================================================
# 2. Numerical Integration - Numba Accelerated
# ==============================================================================
print("Defining numerical integrator...")

@nb.njit
def velocity_verlet_step(state, dt_, gamma_):
    """Performs one step of Velocity Verlet integration."""
    x, y, z, px, py, pz = state
    fx, fy, fz = force(x, y, z, gamma_)
    px_half = px + fx * dt_ / 2
    py_half = py + fy * dt_ / 2
    pz_half = pz + fz * dt_ / 2
    x_new = x + px_half * dt_
    y_new = y + py_half * dt_
    z_new = z + pz_half * dt_
    fx_new, fy_new, fz_new = force(x_new, y_new, z_new, gamma_)
    px_new = px_half + fx_new * dt_ / 2
    py_new = py_half + fy_new * dt_ / 2
    pz_new = pz_half + fz_new * dt_ / 2
    return np.array([x_new, y_new, z_new, px_new, py_new, pz_new], dtype=np.float64)

@nb.njit
def integrate_trajectory_recross(state0, t_max_, dt_, gamma_):
    """
    Integrates a trajectory and detects the first recrossing event (x=0, px<0).
    Returns: (final_state, recrossing_time or t_max).
    """
    state = state0.copy()
    t_curr = 0.0
    t_recross = t_max_
    entered_well = False
    max_steps = int(t_max_ / dt_) + 1

    for _ in range(max_steps):
        x_prev = state[0]
        state = velocity_verlet_step(state, dt_, gamma_)
        x_curr, px_curr = state[0], state[3]
        t_curr += dt_

        if not entered_well and x_curr > 1e-4:
            entered_well = True

        if entered_well and x_prev > 0 and x_curr <= 0 and px_curr < 0:
            if x_prev != x_curr:
                 t_event_ratio = x_prev / (x_prev - x_curr)
                 t_recross = (t_curr - dt_) + t_event_ratio * dt_
            else:
                 t_recross = t_curr - dt_ / 2
            break

        if t_curr >= t_max_:
            break
    # Note: final_state is returned but not used in the main parallel loop
    return state, t_recross

# ==============================================================================
# 3. Initial Condition Generation
# ==============================================================================
print("Defining initial condition generator...")

def generate_initial_conditions(n_samples, energy, gamma_, px_fixed=None):
    """
    Generates initial conditions on the approximate NHIM at x=0 with constant energy E.
    Returns: list of initial states, list of initial parameters (dict).
    """
    initial_conditions = []
    params_list = []
    generated_count = 0
    attempts = 0
    max_attempts = n_samples * 100

    print(f"Generating {n_samples} initial conditions on NHIM (E={energy})...")

    while generated_count < n_samples and attempts < max_attempts:
        attempts += 1
        x0 = 0.0
        potential_at_saddle = V(x0, 0.0, 0.0, gamma_) # Should be 1.0

        # Approx. energy available for y-z motion
        if px_fixed is not None:
            px0_val = px_fixed
            e_center_approx = energy - potential_at_saddle - 0.5 * px0_val**2
        else:
            e_center_approx = energy - potential_at_saddle

        if e_center_approx < 0:
            if attempts % 10000 == 0: print(f"Warning: E_center_approx < 0 ({e_center_approx:.2e})")
            continue

        # Distribute energy randomly between y and z oscillators
        e_y = np.random.uniform(0, e_center_approx)
        e_z = e_center_approx - e_y
        if e_z < 0: e_z = 0; e_y = e_center_approx

        # Choose random phases
        phi_y = np.random.uniform(0, 2 * np.pi)
        phi_z = np.random.uniform(0, 2 * np.pi)

        # Calculate y, py, z, pz from energy and phase
        y0 = np.sqrt(2 * e_y / OMEGA_Y**2) * np.cos(phi_y) if OMEGA_Y > 0 else 0
        py0 = -OMEGA_Y * np.sqrt(2 * e_y) * np.sin(phi_y) if OMEGA_Y > 0 else 0
        z0 = np.sqrt(2 * e_z / OMEGA_Z**2) * np.cos(phi_z) if OMEGA_Z > 0 else 0
        pz0 = -OMEGA_Z * np.sqrt(2 * e_z) * np.sin(phi_z) if OMEGA_Z > 0 else 0

        # Determine px from energy conservation (or use fixed value)
        if px_fixed is None:
            potential_yz = V(x0, y0, z0, gamma_)
            kinetic_yz = 0.5 * (py0**2 + pz0**2)
            px_squared_needed = 2 * (energy - potential_yz - kinetic_yz)
            if px_squared_needed < 0:
                if px_squared_needed < -1e-9: # Allow small negative due to numerics
                    if attempts % 10000 == 0: print(f"Warning: px^2 negative ({px_squared_needed:.2e})")
                    continue
                px_squared_needed = 0.0
            px0_val = np.sqrt(px_squared_needed)
            # Ensure px is slightly positive if it's zero
            if px0_val < 1e-7: px0_val = 1e-7
        else:
             px0_val = px_fixed # Use the provided fixed value

        # Construct state and check final energy
        state0 = np.array([x0, y0, z0, px0_val, py0, pz0])
        final_H = hamiltonian(state0, gamma_)
        if abs(final_H - energy) > 1e-7:
            if attempts % 10000 == 0: print(f"Warning: Energy mismatch {abs(final_H - energy):.2e}")
            continue

        initial_conditions.append(state0)
        # Store parameters needed for analysis (use more compact types)
        params = {'phi_y': np.float32(phi_y), 'phi_z': np.float32(phi_z),
                  'E_y': np.float32(e_y), 'E_z': np.float32(e_z)}
        params_list.append(params)
        generated_count += 1

        if generated_count % (n_samples // 10) == 0 and generated_count > 0:
            print(f"  Generated {generated_count}/{n_samples} conditions...")

    if generated_count < n_samples:
        print(f"Warning: Could only generate {generated_count} conditions after {max_attempts} attempts.")

    print(f"Successfully generated {generated_count} initial conditions.")
    return initial_conditions, params_list

# ==============================================================================
# 4. Simulation Execution
# ==============================================================================
print("Defining simulation execution functions...")

def _simulate_single_wrapper(args):
    """Internal wrapper for joblib parallel execution."""
    idx, state0, t_max_, dt_, gamma_ = args
    try:
        _, t_recross = integrate_trajectory_recross(state0, t_max_, dt_, gamma_)
        # Return index and recrossing time (use float32 for time to save space)
        return idx, np.float32(t_recross)
    except Exception as e:
        print(f"Error in trajectory {idx}: {e}")
        return idx, np.float32(t_max_) # Return max time on error

def run_parallel_simulations(initial_states, t_max_, dt_, gamma_, n_cores_):
    """Runs trajectory simulations in parallel using joblib."""
    actual_n_traj = len(initial_states)
    if actual_n_traj == 0:
        print("No initial states to simulate.")
        return np.array([], dtype=np.float32)

    print(f"\nStarting {actual_n_traj} trajectory simulations using {n_cores_} cores...")
    print(f"Max simulation time: {t_max_:.1e}, Time step: {dt_}")

    simulation_args = [(i, initial_states[i], t_max_, dt_, gamma_) for i in range(actual_n_traj)]

    # Use joblib for parallel execution with progress display
    results = Parallel(n_jobs=n_cores_, verbose=11)(
        delayed(_simulate_single_wrapper)(args) for args in simulation_args
    )

    # Sort results by index to match initial conditions order
    results.sort(key=lambda x: x[0])
    recross_times = np.array([res[1] for res in results], dtype=np.float32)

    return recross_times

# ==============================================================================
# 5. Main Execution Block
# ==============================================================================

def main():
    """Main execution function for running simulation and saving results."""
    start_time_main = time.time()
    os.makedirs(DATA_FOLDER, exist_ok=True)
    output_filepath = os.path.join(DATA_FOLDER, OUTPUT_FILENAME)

    print(f"--- 3DoF Recrossing Simulation ---")
    print(f"Parameters: E={E:.3f}, gamma={GAMMA}, N_traj={N_TRAJ}, T_max={T_MAX:.1e}, dt={DT}")
    print(f"Output will be saved to: {output_filepath}")

    # 1. Generate Initial Conditions
    initial_states, initial_params_list = generate_initial_conditions(
        N_TRAJ, E, GAMMA,
        px_fixed=PX_INIT_VAL if USE_FIXED_PX else None
    )
    actual_n_traj = len(initial_states)
    if actual_n_traj == 0:
        print("Stopping: No initial conditions generated.")
        return

    # 2. Run Simulations
    recross_times = run_parallel_simulations(initial_states, T_MAX, DT, GAMMA, N_CORES)
    simulation_time = time.time() - start_time_main
    print(f"\nTotal Simulation Time: {simulation_time:.2f} seconds")

    # 3. Prepare Data for Saving
    # Convert list of dicts to NumPy arrays for efficient saving
    initial_params_rec = np.rec.fromrecords(initial_params_list,
                                             names=['phi_y', 'phi_z', 'E_y', 'E_z'])

    # 4. Save Results
    print(f"\nSaving results to {output_filepath}...")
    try:
        np.savez_compressed(
            output_filepath,
            recross_times=recross_times,        # Shape: (actual_n_traj,) dtype: float32
            initial_params=initial_params_rec,  # Structured array
            config_E=E,
            config_GAMMA=GAMMA,
            config_N_TRAJ=actual_n_traj, # Save actual number simulated
            config_T_MAX=T_MAX,
            config_DT=DT
        )
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    total_time = time.time() - start_time_main
    print(f"\n--- Total Execution Time: {total_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
