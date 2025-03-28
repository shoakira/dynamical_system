# -*- coding: utf-8 -*-
"""
ダブルウェルモデルにおける再帰渡（recrossing）現象のシミュレーション
シンプレクティック積分法（ベロシティ・ベルレ法）を使用
NHIM（法双曲不変多様体）の厳密計算と初期点設定
高速化：並列処理とNumbaによるJITコンパイル

変更点：
1. 結果ファイル (PDF) を ./data/ フォルダに保存
2. 初期条件生成関数 (generate_unstable_manifold_point) のロジックを改善
3. 再交差した軌道のみを選択して生存確率を計算・プロット
4. 再交差までの滞在時間に応じて、初期条件がNHIM上でどのように分布するかを可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed
import numba as nb
from scipy.optimize import root
import os
import time
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm # 対数カラースケールのため

# --- データ保存用フォルダ ---
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)
print(f"データは '{data_folder}/' フォルダに保存されます。")

# パフォーマンス計測のための時間計測開始
start_time = time.time()

# --- シミュレーションパラメータ ---
E = 1.01           # 全エネルギー（サドルエネルギー1より上）
n_traj = 200000    # 初期状態の数（再交差軌道を十分に得るために増やすことを推奨）
T_max = 100000     # 最長シミュレーション時間
dt = 0.001           # 時間刻み幅
manifold_distance = 0.0001# NHIMからの不安定多様体上の距離（小さめに変更）

# --- 非線形相互作用パラメータ ---
gamma = 0.1  # 相互作用の強さ

# --- Numbaで高速化された関数 ---
@nb.njit
def V(x, y):
    """ポテンシャルエネルギー関数"""
    return (x**2 - 1)**2 + 0.5 * y**2 + y**4 + gamma * x**2 * y**2

@nb.njit
def dVdx(x, y):
    """ポテンシャルのxに関する偏導関数"""
    return 4 * x * (x**2 - 1) + 2 * gamma * x * y**2

@nb.njit
def dVdy(x, y):
    """ポテンシャルのyに関する偏導関数"""
    return y + 4 * y**3 + 2 * gamma * x**2 * y

@nb.njit
def hamiltonian(state):
    """ハミルトニアン（全エネルギー）を計算"""
    x, y, px, py = state
    return 0.5 * (px**2 + py**2) + V(x, y)

@nb.njit
def adjust_energy(state, target_energy):
    """
    与えられた状態のエネルギーが target_energy になるように px を調整する。
    状態を直接変更する。成功すれば True、不可能なら False を返す。
    """
    x, y, _, py = state
    potential = V(x, y)
    py_energy = 0.5 * py**2

    # 目標のpx^2を計算
    px_squared_target = 2 * (target_energy - potential - py_energy)

    if px_squared_target < -1e-12: # 許容誤差より小さい場合
        # print(f"警告: エネルギー調整で px^2 < 0 ({px_squared_target:.2e}) at (x,y)=({x:.3f},{y:.3f})")
        return False # エネルギー調整不可能
    elif px_squared_target < 0: # 許容誤差内の負の値は0とする
        state[2] = 0.0
    else:
        # 必ず左向き (px < 0) になるように調整
        state[2] = -np.sqrt(px_squared_target)

    # 最終確認 (デバッグ用)
    # final_H = hamiltonian(state)
    # if abs(final_H - target_energy) > 1e-9:
    #     print(f"警告: エネルギー調整後の誤差が大きい {abs(final_H - target_energy):.2e}")
    return True


@nb.njit
def symplectic_integrate(state0, t_start, t_end, dt=0.01, max_steps=None):
    """
    ベロシティ・ベルレ法によるシンプレクティック積分（Numba最適化版）
    指定された時間範囲 [t_start, t_end] を積分し、最終状態とイベント時刻を返す。
    イベントは x=0 を px<0 で横切った時刻。
    """
    if max_steps is None:
        max_steps = int(abs(t_end - t_start) / abs(dt)) + 1

    state = np.empty(4, dtype=np.float64)
    state[:] = state0[:]

    t_current = t_start
    t_event = None
    actual_dt = dt if t_end >= t_start else -dt

    for _ in range(max_steps):
        if (actual_dt > 0 and t_current >= t_end) or \
           (actual_dt < 0 and t_current <= t_end):
            break

        x, y, px, py = state

        x_half = x + px * actual_dt / 2
        y_half = y + py * actual_dt / 2

        force_x = -dVdx(x_half, y_half)
        force_y = -dVdy(x_half, y_half)
        px_new = px + force_x * actual_dt
        py_new = py + force_y * actual_dt

        x_new = x_half + px_new * actual_dt / 2
        y_new = y_half + py_new * actual_dt / 2

        # イベント検出: x=0 の負方向への横切り (サドル再交差)
        if actual_dt > 0 and t_current > t_start + 1e-6 and x > 0 and x_new <= 0 and px_new < 0:
            if x != x_new:
                dt_event_ratio = x / (x - x_new)
                t_event = t_current + dt_event_ratio * actual_dt
            else:
                t_event = t_current + actual_dt / 2
            # イベント発生時の状態を記録して break する場合
            # state[0] = x_half + px_new * dt_event_ratio * actual_dt / 2 # 補間位置
            # state[1] = y_half + py_new * dt_event_ratio * actual_dt / 2
            # state[2] = px_new
            # state[3] = py_new
            # break # 今回は積分終了後に判定するので break しない

        state[0] = x_new
        state[1] = y_new
        state[2] = px_new
        state[3] = py_new
        t_current += actual_dt

    return state, t_event


@nb.njit
def symplectic_integrate_enhanced(state0, t_start, t_end, dt=0.01, max_steps=None):
    """
    ベロシティ・ベルレ法によるシンプレクティック積分（イベント検出強化版）
    サドル再交差、井戸からの離脱（右側）、最大時間到達を検出し、
    最終状態、イベント時刻、イベントの種類（フラグ）を返す。
    """
    if max_steps is None:
        max_steps = int((t_end - t_start) / dt) + 1

    state = np.empty(4, dtype=np.float64)
    state[:] = state0[:]

    t_current = t_start
    t_event = T_max # デフォルト

    crossed_saddle = False
    escaped_right = False
    stayed_in_well = False

    initial_x = state0[0]
    right_well_limit = 2.0

    for step in range(max_steps):
        if t_current >= t_end:
            stayed_in_well = True
            t_event = T_max
            break

        x, y, px, py = state

        x_half = x + px * dt / 2
        y_half = y + py * dt / 2

        force_x = -dVdx(x_half, y_half)
        force_y = -dVdy(x_half, y_half)
        px_new = px + force_x * dt
        py_new = py + force_y * dt

        x_new = x_half + px_new * dt / 2
        y_new = y_half + py_new * dt / 2

        # イベント検出 1: サドル再交差 (x=0, px<0)
        if t_current > t_start + 1e-6 and x > 0 and x_new <= 0 and px_new < 0:
            crossed_saddle = True
            if x != x_new:
                dt_event_ratio = x / (x - x_new)
                t_event = t_current + dt_event_ratio * dt
            else:
                t_event = t_current + dt / 2
            state[0] = x_new # イベント発生時の状態を更新しておく
            state[1] = y_new
            state[2] = px_new
            state[3] = py_new
            break

        # イベント検出 2: 右側離脱 (x > limit)
        if x_new > right_well_limit:
            escaped_right = True
            t_event = t_current + dt
            state[0] = x_new # 状態更新
            state[1] = y_new
            state[2] = px_new
            state[3] = py_new
            break

        # イベント検出 3: 左側離脱 (x < -limit) - 通常は起こりにくい
        # if x_new < -right_well_limit:
        #     # escaped_left = True # 必要ならフラグ追加
        #     t_event = t_current + dt
        #     state[:] = [x_new, y_new, px_new, py_new]
        #     break

        state[0] = x_new
        state[1] = y_new
        state[2] = px_new
        state[3] = py_new
        t_current += dt

    # ループが break せずに終了した場合
    if not crossed_saddle and not escaped_right and t_current >= t_end:
        stayed_in_well = True
        t_event = T_max

    return state, t_event, crossed_saddle, escaped_right, stayed_in_well


def integrate_trajectory(state0, t_span, dt=0.01):
    """
    軌道全体を記録する積分関数（主に可視化やNHIM計算用）
    """
    t_start, t_end = t_span
    n_steps = int(abs(t_end - t_start) / abs(dt)) + 1
    t_vals = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((n_steps, 4))
    states[0] = state0

    current_state = state0.copy()
    actual_dt = dt if t_end >= t_start else -dt

    for i in range(1, n_steps):
        temp_state, _ = symplectic_integrate(current_state, t_vals[i-1], t_vals[i], actual_dt)
        states[i] = temp_state
        current_state = temp_state

    return t_vals, states


# ===== NHIM（不安定周期軌道）計算のための関数群 =====

def saddle_linearization():
    """
    サドル点(0,0)近傍での線形化ハミルトン系のヤコビ行列の固有値・固有ベクトルを計算
    """
    d2Vdx2 = -4.0
    d2Vdy2 = 1.0
    d2Vdxdy = 0.0

    A = np.array([
        [0,       0,       1, 0],
        [0,       0,       0, 1],
        [-d2Vdx2, -d2Vdxdy, 0, 0],
        [-d2Vdxdy,-d2Vdy2,  0, 0]
    ])

    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argsort(np.real(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def shooting_function(params, period, energy):
    """シューティング法のための残差関数"""
    y0, py0 = params
    x0 = 0.0
    V_init = V(x0, y0)
    px0_squared = 2 * (energy - V_init - 0.5 * py0**2)
    if px0_squared < -1e-12: # 許容誤差を追加
        return [1e10, 1e10]
    px0 = -np.sqrt(max(0, px0_squared)) # 負の値にする
    initial_state = np.array([x0, y0, px0, py0])
    t_vals_shoot, states_shoot = integrate_trajectory(initial_state, [0, period], dt=0.001)
    final_state = states_shoot[-1]
    residuals = [final_state[1] - y0, final_state[3] - py0]
    return residuals


def find_unstable_periodic_orbit(energy, initial_guess=None):
    """シューティング法を用いて不安定周期軌道(NHIM)を計算する"""
    print(f"\nエネルギー E={energy} の NHIM（不安定周期軌道）を計算中...")
    eigvals, eigvecs = saddle_linearization()
    center_eigenvals = eigvals[np.isclose(np.real(eigvals), 0) & ~np.isclose(np.imag(eigvals), 0)]
    if len(center_eigenvals) >= 2:
        omega = np.abs(np.imag(center_eigenvals[0]))
        estimated_period = 2 * np.pi / omega
        print(f"  線形化からの推定周期: {estimated_period:.4f} (ω={omega:.4f})")
    else:
        estimated_period = 2 * np.pi
        print(f"  警告: 線形化で中心方向が見つからず。推定周期: {estimated_period:.4f} を使用。")

    if initial_guess is None:
        energy_above_saddle = energy - 1.0
        if energy_above_saddle < 0:
             print(f"  エラー: エネルギー E={energy} はサドル点エネルギー(1.0)より低いです。")
             return None
        y0_guess = np.sqrt(energy_above_saddle * 0.5)
        py0_guess = 0.0
        initial_guess = [y0_guess, py0_guess]
        print(f"  初期推定値 [y0, py0] を生成: {initial_guess}")

    sol = root(
        shooting_function, initial_guess, args=(estimated_period, energy),
        method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 1000}
    )

    if not sol.success:
        print(f"  警告: 周期軌道の計算が収束しませんでした。Message: {sol.message}")
        # return None # 収束しなくても進む場合あり

    y0_opt, py0_opt = sol.x
    print(f"  最適化結果 [y0, py0]: [{y0_opt:.6f}, {py0_opt:.6f}]")
    print(f"  残差 (y, py): [{sol.fun[0]:.2e}, {sol.fun[1]:.2e}]")

    x0 = 0.0
    V_init_opt = V(x0, y0_opt)
    px0_opt_squared = 2 * (energy - V_init_opt - 0.5 * py0_opt**2)
    if px0_opt_squared < -1e-12:
        print("  エラー: 最適化解でエネルギー保存を満たせません (px^2 < 0)。")
        # 収束しなかった場合、ここに来ることがある
        # エネルギーを満たせる近い解を探すか、エラーとするか
        # ここでは警告だけ出して進めてみる (px=0になる)
        px0_opt_squared = 0

    px0_opt = -np.sqrt(px0_opt_squared)
    nhim_initial_state = np.array([x0, y0_opt, px0_opt, py0_opt])
    nhim_t_vals, nhim_orbit_states = integrate_trajectory(nhim_initial_state, [0, estimated_period], dt=0.0005)

    nhim_energies = np.array([hamiltonian(s) for s in nhim_orbit_states])
    print(f"  NHIM上のエネルギー: 平均={np.mean(nhim_energies):.8f}, 標準偏差={np.std(nhim_energies):.2e} (目標 E={energy})")
    final_state_check = nhim_orbit_states[-1]
    print(f"  周期性確認: y(T)={final_state_check[1]:.6f}, py(T)={final_state_check[3]:.6f}")

    # 軌道データが空でないかチェック
    if len(nhim_orbit_states) == 0:
        print("  エラー: NHIM軌道データの生成に失敗しました。")
        return None

    print(f"NHIM計算完了。周期 ≈ {estimated_period:.6f}")
    return nhim_t_vals, nhim_orbit_states, estimated_period

def sample_nhim(nhim_data, n_samples):
    """
    NHIM上から完全に均等に点をサンプリングする
    
    Parameters:
    -----------
    nhim_data : tuple
        (t_vals, orbit_states, period) のタプル
    n_samples : int
        サンプリングする点の数
        
    Returns:
    --------
    nhim_samples : ndarray
        サンプリングされた点の座標
    sampled_orbit_indices : ndarray
        元のNHIM軌道内でのインデックス（位相情報を保持するため）
    """
    if nhim_data is None: 
        return None, None
        
    t_vals, orbit_states, period = nhim_data
    if len(orbit_states) == 0: 
        return None, None
        
    n_orbit_points = len(orbit_states)
    
    # 完全に均等分布になるように修正
    # 方法1: 単純な均等確率でランダムに選択
    probabilities = np.ones(n_orbit_points) / n_orbit_points
    
    # 方法2（オプション）: さらに確実な均等分布のため、位相に基づく線形補間
    # positions = np.linspace(0, 1, n_samples, endpoint=False)  # 0～1の範囲で均等に
    # indices = (positions * n_orbit_points).astype(int)
    # return orbit_states[indices], indices
    
    try:
        # サンプリングされた点が、元の orbit_states のどのインデックスに対応するかを取得
        sampled_orbit_indices = np.random.choice(
            n_orbit_points,
            size=n_samples,
            replace=True,  # 大量のサンプルが必要な場合は True
            p=probabilities
        )
    except ValueError as e:
        print(f"NHIMサンプリング中にエラー: {e}. 単純均等サンプリングを試みます。")
        sampled_orbit_indices = np.random.choice(n_orbit_points, size=n_samples, replace=True)

    # サンプリングされた点の座標を取得
    nhim_samples = orbit_states[sampled_orbit_indices]
    
    # サンプリングされた点の座標と、元の軌道内でのインデックスを返す
    # （インデックスは後で位相情報の生成に使用可能）
    return nhim_samples, sampled_orbit_indices

# ===== 初期条件生成関数の改善 =====
# @nb.njit # Numba化するとデバッグが難しくなるため、一旦外す
def generate_unstable_manifold_point(nhim_state, manifold_distance=0.005, target_energy=E):
    """
    【改善版】NHIM上の点から出発して、不安定多様体上の点を生成する。
    エネルギー保存と方向(x>0, px<0)をより厳密に扱う。
    失敗した場合は None を返す。
    """
    small_perturbation = 1e-7 # 初期摂動の大きさ (小さめにする)
    integration_dt = 0.001   # 短時間積分用のdt
    integration_time_max = 2.0 # 最大積分時間

    # 1. 不安定方向の取得と摂動
    eigvals, eigvecs = saddle_linearization()
    unstable_idx = np.argmax(np.real(eigvals))
    unstable_dir = eigvecs[:, unstable_idx].real
    if unstable_dir[0] > 0: unstable_dir = -unstable_dir # x<0方向へ
    unstable_dir /= np.linalg.norm(unstable_dir) # 正規化

    perturbed_state = nhim_state.copy()
    perturbed_state += small_perturbation * unstable_dir

    # 2. 摂動直後のエネルギー調整と妥当性チェック
    if not adjust_energy(perturbed_state, target_energy):
        # print(f"デバッグ: 摂動直後のエネルギー調整失敗 @ NHIM state {nhim_state[:2]}")
        return None # エネルギー的に不可能

    # x>0, px<0 をチェック
    if perturbed_state[0] <= 1e-9: # xがほぼゼロまたは負
        # print(f"デバッグ: 摂動直後に x<=0 ({perturbed_state[0]:.2e}) @ NHIM state {nhim_state[:2]}")
        return None
    if perturbed_state[2] >= -1e-9: # pxがほぼゼロまたは正
         # adjust_energy で px<0 になっているはずだが念のため
        # print(f"デバッグ: 摂動直後に px>=0 ({perturbed_state[2]:.2e}) @ NHIM state {nhim_state[:2]}")
        return None

    # デバッグ用: 摂動直後のエネルギー確認
    # H_pert = hamiltonian(perturbed_state)
    # print(f"デバッグ: 摂動直後 H={H_pert:.8f} (Error={abs(H_pert-target_energy):.2e})")

    # 3. 短時間積分による多様体上の点探索
    current_state_int = perturbed_state.copy()
    manifold_state = None
    n_steps_integrate = int(integration_time_max / integration_dt)

    for i in range(n_steps_integrate):
        # 1ステップ積分 (Numba版を使う)
        next_state_int_nb, _ = symplectic_integrate(current_state_int, 0, integration_dt, integration_dt)
        next_state_int = np.array(next_state_int_nb) # Numba配列をNumpy配列に変換

        # NHIM上の元の点からの位相空間距離
        distance_from_nhim = np.linalg.norm(next_state_int - nhim_state)

        # エネルギー保存チェック (デバッグ用)
        # H_int = hamiltonian(next_state_int)
        # if abs(H_int - target_energy) > 1e-6:
        #     print(f"警告@generate_unstable: 積分ステップ {i} でエネルギー逸脱 {abs(H_int - target_energy):.2e}")
            # 必要ならここでエネルギーを再調整するか、積分を打ち切る
            # if not adjust_energy(next_state_int, target_energy):
            #     print("  積分中のエネルギー再調整失敗。中断。")
            #     return None

        if distance_from_nhim >= manifold_distance:
            manifold_state = next_state_int.copy()
            # print(f"デバッグ: 目標距離 {manifold_distance:.2e} 到達 @ ステップ {i} (距離 {distance_from_nhim:.3e})")
            break # 目標距離に到達

        current_state_int = next_state_int.copy()

        # x=0 を超えてしまったら中断 (多様体から外れた可能性)
        if current_state_int[0] <= 0:
            # print(f"デバッグ: 積分中に x<=0 ({current_state_int[0]:.2e}) になったため中断 @ ステップ {i}")
            return None # この初期条件は使わない

    if manifold_state is None:
        # print(f"警告@generate_unstable: 最大時間積分しても目標距離 {manifold_distance:.2e} に達せず。最後の点を使用 (距離 {distance_from_nhim:.3e})。")
        manifold_state = current_state_int # 最後の状態を使うが、適切でない可能性

    # 4. 最終チェックと調整
    # エネルギー再調整
    if not adjust_energy(manifold_state, target_energy):
        # print(f"デバッグ: 最終エネルギー調整失敗。")
        return None

    # x>0, px<0 の最終確認
    if manifold_state[0] <= 1e-9:
        # print(f"デバッグ: 最終状態で x<=0 ({manifold_state[0]:.2e})")
        return None
    if manifold_state[2] >= -1e-9:
        # adjust_energy で px<0 になっているはず
        # print(f"デバッグ: 最終状態で px>=0 ({manifold_state[2]:.2e})")
        return None

    # 最終エネルギー確認 (デバッグ用)
    # H_final = hamiltonian(manifold_state)
    # final_error = abs(H_final - target_energy)
    # if final_error > 1e-7:
    #      print(f"警告@generate_unstable: 最終エネルギー誤差が大きい {final_error:.2e}")

    # print(f"デバッグ: 生成成功 x={manifold_state[0]:.4f}, px={manifold_state[2]:.4f}, E_err={final_error:.2e}")
    return manifold_state


# --- シミュレーション関数 (ラッパー) ---
def simulate_single_trajectory_enhanced_wrapper(args):
    """並列処理のためのラッパー関数。引数をタプルで受け取る。"""
    idx, nhim_samples_local, manifold_distance_local, T_max_local, dt_local, target_energy_local = args
    sample_idx = idx % len(nhim_samples_local)
    nhim_state = nhim_samples_local[sample_idx].copy()

    # 不安定多様体上の初期点を生成 (改善版を使用)
    state0 = generate_unstable_manifold_point(nhim_state, manifold_distance_local, target_energy_local)

    # 初期状態生成が成功したかチェック
    if state0 is None:
        # print(f"デバッグ: 初期状態生成失敗 (idx={idx})")
        return T_max_local, False, False, False, idx, None # state0 は None

    # state0 は numpy array のはず
    state0_np = np.array(state0) # 念のため変換

    # シミュレーション実行 (Numba 関数を直接呼び出す)
    try:
        final_state, t_event, crossed_saddle, escaped_right, stayed_in_well = \
            symplectic_integrate_enhanced(state0_np, 0, T_max_local, dt_local)
        # 初期状態 state0_np も返すように変更
        return t_event, crossed_saddle, escaped_right, stayed_in_well, idx, state0_np
    except Exception as e:
        print(f"警告: 積分中にエラー (idx={idx}): {e}")
        return T_max_local, False, False, False, idx, state0_np # エラー発生時は無効な結果


# --- 可視化関数 ---
def visualize_unstable_manifold(nhim_orbit_states, n_points=20, manifold_distance=0.01, target_energy=E):
    """NHIM軌道と、そこから生成された不安定多様体上の点を可視化する"""
    print(f"\n不安定多様体の可視化 (manifold_distance={manifold_distance})...")
    plt.figure(figsize=(10, 8))
    plt.plot(nhim_orbit_states[:, 0], nhim_orbit_states[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, label='NHIM Orbit')

    manifold_points_x = []
    manifold_points_y = []
    successful_points = 0
    step = max(1, len(nhim_orbit_states) // n_points)

    for i in range(0, len(nhim_orbit_states), step):
        nhim_state = nhim_orbit_states[i].copy()
        manifold_state = generate_unstable_manifold_point(nhim_state, manifold_distance, target_energy)
        if manifold_state is not None:
            successful_points += 1
            manifold_points_x.append(manifold_state[0])
            manifold_points_y.append(manifold_state[1])
            t_vis, states_vis = integrate_trajectory(manifold_state, [0, 1.0], dt=0.01)
            plt.plot(states_vis[:, 0], states_vis[:, 1], '-', color='red', alpha=0.3, linewidth=0.8)

    print(f"  {successful_points}/{n_points} 個の多様体上の点の生成に成功。")
    plt.scatter(manifold_points_x, manifold_points_y, c='red', s=15, label=f'Unstable Manifold Points (dist={manifold_distance})')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8, label='Saddle Line (x=0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('NHIM and Sampled Unstable Manifold Points')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    max_y_abs = np.max(np.abs(nhim_orbit_states[:, 1])) * 1.5
    plt.xlim(-0.2, 0.2)
    plt.ylim(-max_y_abs, max_y_abs)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(data_folder, 'nhim_and_unstable_manifold.pdf')) # 保存先変更
    plt.close()
    print(f"  可視化結果を '{os.path.join(data_folder, 'nhim_and_unstable_manifold.pdf')}' に保存しました。")


# --- メイン実行ブロック ---
if __name__ == "__main__":
    n_cores = mp.cpu_count()
    print(f"利用可能なCPUコア数: {n_cores}")

    # --- 1. NHIM 計算 ---
    nhim_data = find_unstable_periodic_orbit(E)
    if nhim_data is None: exit()
    nhim_t_vals, nhim_orbit_states, nhim_period = nhim_data

    # --- 2. NHIM サンプリング ---
    n_nhim_samples = 500
    nhim_samples, nhim_sampled_indices = sample_nhim(nhim_data, n_nhim_samples)
    if nhim_samples is None: exit()

    # 既存のプロットコードを調整して、均等サンプリングであることを明示
    plt.figure(figsize=(8, 8))
    plt.plot(nhim_orbit_states[:, 0], nhim_orbit_states[:, 1], '-', color='blue', alpha=0.7, linewidth=1, label='NHIM Orbit (x, y)')
    plt.scatter(nhim_samples[:, 0], nhim_samples[:, 1], c='red', s=10, alpha=0.8, label=f'{n_nhim_samples} Uniform Sampled Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Calculated NHIM and Uniformly Sampled Points')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(data_folder, 'nhim_orbit_and_samples.pdf')) # 保存先変更
    plt.close()
    print(f"\nNHIM軌道とサンプル点を '{os.path.join(data_folder, 'nhim_orbit_and_samples.pdf')}' に保存しました。")

    # 不安定多様体の可視化
    visualize_unstable_manifold(nhim_orbit_states, n_points=30, manifold_distance=manifold_distance, target_energy=E)

    # --- 3. 並列シミュレーションの実行 ---
    print(f"\n{n_traj}個のトラジェクトリ計算を開始 (manifold_distance={manifold_distance})...")
    # 引数リストに target_energy も追加
    simulation_args = [(i, nhim_samples, manifold_distance, T_max, dt, E) for i in range(n_traj)]

    results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(simulate_single_trajectory_enhanced_wrapper)(args) for args in simulation_args
    )

    # --- 4. 結果の集計と分析 ---
    simulation_end_time = time.time()
    print(f"\nシミュレーション完了。計算時間: {simulation_end_time - start_time:.2f}秒")

    recrossing_results = []
    escaped_results = []
    stayed_results = []
    invalid_initial_count = 0
    generated_initial_states = {} # 生成された初期状態を保存 (idx -> state0)

    for result in results:
        if result is None:
             invalid_initial_count += 1
             continue

        t_event, crossed, escaped, stayed, idx, state0 = result

        if state0 is None:
            invalid_initial_count += 1
            continue
        else:
            # 正常に生成された初期状態を記録
            generated_initial_states[idx] = state0

        if crossed:
            recrossing_results.append({'time': t_event, 'idx': idx}) # 初期状態は generated_initial_states から引く
        elif escaped:
            escaped_results.append({'time': t_event, 'idx': idx})
        elif stayed:
            stayed_results.append({'time': t_event, 'idx': idx})

    n_generated = len(generated_initial_states)
    n_recrossing = len(recrossing_results)
    n_escaped = len(escaped_results)
    n_stayed = len(stayed_results)
    n_simulated = n_recrossing + n_escaped + n_stayed # 実際にシミュレーションされた数

    print("\n--- シミュレーション結果概要 ---")
    print(f"総試行回数 (n_traj): {n_traj}")
    print(f"初期状態生成 試行回数: {n_traj}")
    print(f"初期状態生成 成功数: {n_generated} ({n_generated/n_traj*100:.1f}%)")
    print(f"初期状態生成 失敗数: {n_traj - n_generated}")
    if n_generated == 0:
        print("\n有効な初期状態が生成できませんでした。パラメータやコードを確認してください。")
        exit()

    print(f"\n--- 生成成功した初期状態からの軌道追跡結果 (N={n_generated}) ---")
    print(f"  サドルを再交差 (Recrossed): {n_recrossing} ({n_recrossing/n_generated*100:.1f}%)")
    print(f"  右側に離脱 (Escaped Right): {n_escaped} ({n_escaped/n_generated*100:.1f}%)")
    print(f"  井戸内に滞在 (Stayed in Well): {n_stayed} ({n_stayed/n_generated*100:.1f}%)")
    # 合計が n_generated になるはず (確認)
    # print(f"  (確認) 分類合計: {n_recrossing+n_escaped+n_stayed}")


    # --- 5. 再交差軌道の生存確率プロット ---
    if n_recrossing > 1:
        print("\n再交差軌道の生存確率を計算・プロット中...")
        recrossing_times = np.array([r['time'] for r in recrossing_results])

        min_plot_time = max(dt * 10, np.min(recrossing_times[recrossing_times > 0]))
        max_plot_time = np.max(recrossing_times)

        if min_plot_time >= max_plot_time or np.isinf(min_plot_time) or np.isinf(max_plot_time):
             print("警告: 再交差時間の範囲が無効なため、生存確率プロットをスキップします。")
        else:
            t_plot = np.logspace(np.log10(min_plot_time), np.log10(max_plot_time), 50)
            survival_fraction = np.array([np.sum(recrossing_times > t) / n_recrossing for t in t_plot])

            plt.figure(figsize=(8, 6))
            plt.loglog(t_plot, survival_fraction, marker='o', linestyle='-', markersize=5)
            plt.xlabel("Time (log scale)")
            plt.ylabel("Recrossing Survival Fraction (log scale)")
            plt.title(f"Survival Probability of Recrossing Trajectories (N={n_recrossing})")
            plt.grid(True, which="both", linestyle=':', alpha=0.7)
            plt.ylim(bottom=max(1e-5, 0.5 / n_recrossing))
            filename = f"survival_recrossing_gamma_{gamma}_E_{E:.3f}.pdf"
            plt.savefig(os.path.join(data_folder, filename)) # 保存先変更
            plt.close()
            print(f"  生存確率プロットを '{os.path.join(data_folder, filename)}' に保存しました。")
            print(f"  再交差時間: Min={np.min(recrossing_times):.3e}, Max={np.max(recrossing_times):.3e}, Mean={np.mean(recrossing_times):.3e}")
    else:
        print("\n再交差した軌道が1つ以下しか見つからなかったため、生存確率プロットは生成されません。")
        if n_generated > 0:
            print("  考えられる原因: 再交差自体が稀, n_traj不足, manifold_distance/エネルギー不適切, T_maxなど。")

    # --- 6. 滞在時間ごとの初期値分布 (NHIM y-py 平面) ---
    if n_recrossing > 1:
        print("\n滞在時間ごとの初期値分布をNHIM上で可視化中...")
        # 再交差した軌道の初期状態に対応するNHIM上の点を取得
        recrossing_indices = [r['idx'] for r in recrossing_results]
        # 初期状態生成時の元になったNHIMサンプルインデックス
        nhim_sample_indices = [idx % len(nhim_samples) for idx in recrossing_indices]

        # NHIM上の点の y座標 と py座標 を取得
        nhim_y_coords = nhim_samples[nhim_sample_indices, 1]
        nhim_py_coords = nhim_samples[nhim_sample_indices, 3]
        # 対応する再交差時間 (対数)
        recrossing_times_log = np.log10([r['time'] for r in recrossing_results])

        plt.figure(figsize=(10, 8))
        plt.plot(nhim_orbit_states[:, 1], nhim_orbit_states[:, 3], color='grey', linestyle='--', alpha=0.5, linewidth=1, label='NHIM Path (y, py)')

        scatter = plt.scatter(nhim_y_coords, nhim_py_coords,
                              c=recrossing_times_log,
                              cmap='viridis',
                              s=20,
                              alpha=0.7)

        plt.xlabel("NHIM y-coordinate")
        plt.ylabel("NHIM py-coordinate")
        plt.title(f"Initial Position on NHIM vs. Recrossing Time (N={n_recrossing})")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box')

        cbar = plt.colorbar(scatter)
        cbar.set_label("Log10(Recrossing Time)")
        plt.legend()
        filename = f"nhim_distribution_recrossing_time_gamma_{gamma}_E_{E:.3f}.pdf"
        plt.savefig(os.path.join(data_folder, filename)) # 保存先変更
        plt.close()
        print(f"  初期値分布プロットを '{os.path.join(data_folder, filename)}' に保存しました。")

    elif n_recrossing == 1:
        print("\n再交差軌道が1つしかないため、初期値分布プロットは生成されません。")
    # n_recrossing == 0 の場合は、生存確率プロットのところでメッセージ表示済み

    print("\n全ての処理が完了しました。")