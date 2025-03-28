#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ダブルウェルモデルにおける再帰渡（recrossing）現象のシミュレーション
シンプレクティック積分法（ベロシティ・ベルレ法）を使用
NHIM（法双曲不変多様体）の厳密計算と初期点設定
高速化：並列処理とNumbaによるJITコンパイル
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

# パフォーマンス計測のための時間計測開始
start_time = time.time()

# シミュレーションパラメータ
E = 1.001           # 全エネルギー（サドルエネルギー1より上）
n_traj = 1000000     # 初期状態の数（大規模計算時は増やす）
T_max = 100000.0     # 最長シミュレーション時間
dt = 0.001           # 時間刻み幅

# 非線形相互作用パラメータ
gamma = 0.2  # 相互作用の強さ

# Numbaで高速化されたポテンシャル関数と導関数
@nb.njit
def V(x, y):
    # 元のポテンシャル + 非線形相互作用項（x^2 * y^2）
    return (x**2 - 1)**2 + 0.5 * y**2 + y**4 + gamma * x**2 * y**2

@nb.njit
def dVdx(x, y):
    # 元の導関数 + 相互作用項の導関数
    return 4 * x * (x**2 - 1) + 2 * gamma * x * y**2

@nb.njit
def dVdy(x, y):
    # 元の導関数 + 相互作用項の導関数
    return y + 4 * y**3 + 2 * gamma * x**2 * y

@nb.njit
def hamiltonian(state):
    x, y, px, py = state
    return 0.5 * (px**2 + py**2) + V(x, y)

# シンプレクティック積分法（ベロシティ・ベルレ法）をNumbaで高速化
@nb.njit
def symplectic_integrate(state0, t_start, t_end, dt=0.01, max_steps=None):
    """
    ベロシティ・ベルレ法によるシンプレクティック積分（Numba最適化版）
    """
    if max_steps is None:
        max_steps = int((t_end - t_start) / dt) + 1
    
    # 必要最小限の情報だけを記録（最適化）- Numba互換の方法で
    state = np.empty(4, dtype=np.float64)
    state[0] = state0[0]  # x
    state[1] = state0[1]  # y
    state[2] = state0[2]  # px
    state[3] = state0[3]  # py
    
    t_current = t_start
    t_event = None
    
    for _ in range(max_steps):
        if t_current >= t_end:
            break
            
        x, y, px, py = state
        
        # 1. 位置の半ステップ更新
        x_half = x + px * dt/2
        y_half = y + py * dt/2
        
        # 2. 運動量の全ステップ更新
        px_new = px - dVdx(x_half, y_half) * dt
        py_new = py - dVdy(x_half, y_half) * dt
        
        # 3. 位置の残り半ステップ更新
        x_new = x_half + px_new * dt/2
        y_new = y_half + py_new * dt/2
        
        # イベント検出: x=0の負方向への横切り
        if t_current > 1e-6 and x > 0 and x_new <= 0 and px_new < 0:
            # 線形補間でイベント時刻を推定
            if x != x_new:  # 0除算防止
                dt_event = dt * x / (x - x_new)
                t_event = t_current + dt_event
            else:
                t_event = t_current + dt/2
            break
        
        # 状態更新
        state[0] = x_new
        state[1] = y_new
        state[2] = px_new
        state[3] = py_new
        t_current += dt
    
    # 最終状態と イベント時刻のみを返す（最適化）
    return state, t_event if t_event is not None else t_end

# より詳細な積分関数（NHIMの計算用、軌道全体を保存）
def integrate_trajectory(state0, t_span, dt=0.01):
    """
    軌道全体を記録する積分関数（NHIM計算用）
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # 結果の保存配列
    t_vals = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((n_steps, 4))
    states[0] = state0
    
    for i in range(1, n_steps):
        state, _ = symplectic_integrate(states[i-1], t_vals[i-1], t_vals[i], dt)
        states[i] = state
    
    return t_vals, states

# ===== NHIM（不安定周期軌道）計算のための関数群 =====

def saddle_linearization():
    """
    サドル点近傍の線形化解析
    固有値と固有ベクトルを返す
    """
    # サドル点 (0,0) での線形化ヤコビアン行列を計算
    # 状態変数の順序: [x, y, px, py]
    # 運動方程式: dx/dt = px, dy/dt = py, dpx/dt = -dV/dx, dpy/dt = -dV/dy
    
    # dV/dx, dV/dy の2階微分を計算（解析的に）
    d2Vdx2 = 12 * 0**2 - 4  # x=0 での値
    d2Vdy2 = 1 + 12 * 0**2  # y=0 での値
    d2Vdxdy = 0  # x=y=0 での値（ガンマを考慮すると変わる可能性あり）
    
    # 線形化行列
    A = np.array([
        [0, 0, 1, 0],  # dx/dt = px
        [0, 0, 0, 1],  # dy/dt = py
        [d2Vdx2, d2Vdxdy, 0, 0],  # dpx/dt = -d2V/dx2 * x - d2V/dxdy * y
        [d2Vdxdy, d2Vdy2, 0, 0]   # dpy/dt = -d2V/dxdy * x - d2V/dy2 * y
    ])
    
    # 固有値と固有ベクトルを計算
    eigvals, eigvecs = np.linalg.eig(A)
    
    # 固有値の並び替え（実部の大きさでソート）
    idx = np.argsort(np.abs(np.real(eigvals)))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs

def shooting_function(params, period, energy):
    """
    シューティング法のための残差関数
    """
    # パラメータから初期状態を構築
    y0, py0 = params
    x0 = 0.0  # サドル点での x=0 に固定
    
    # エネルギー保存則からpxを決定（ここでは左の井戸に向かう負の値）
    V_init = V(x0, y0)
    px0_squared = 2 * (energy - V_init - 0.5 * py0**2)
    if px0_squared < 0:
        # エネルギー的に不可能な初期条件
        return [1e10, 1e10]  # 大きな値を返して最適化アルゴリズムに「不適切」と伝える
    
    px0 = -np.sqrt(px0_squared)  # 左の井戸に向かう
    
    initial_state = np.array([x0, y0, px0, py0])
    
    # 期間 [0, period] で積分
    _, states = integrate_trajectory(initial_state, [0, period], dt=0.01)
    final_state = states[-1]
    
    # 周期条件の残差: y, py が元に戻り、x, px は符号が反転（対称性を考慮）
    residuals = [
        final_state[1] - y0,      # y が元に戻る
        final_state[3] - py0       # py が元に戻る
    ]
    
    return residuals

def find_unstable_periodic_orbit(energy, initial_guess=None):
    """
    シューティング法を使用して不安定周期軌道（NHIM）を計算
    """
    print("NHIM（不安定周期軌道）を計算中...")
    
    # 線形化解析から固有値を取得
    eigvals, _ = saddle_linearization()
    
    # 中心方向（純虚数の固有値）の角振動数を推定
    center_eigenvals = eigvals[np.abs(np.real(eigvals)) < 1e-10]
    if len(center_eigenvals) >= 2:
        omega = np.abs(np.imag(center_eigenvals[0]))
        estimated_period = 2 * np.pi / omega
    else:
        # 中心方向が見つからない場合のフォールバック
        estimated_period = 2 * np.pi
    
    # 初期推定値
    if initial_guess is None:
        # 線形近似から小振幅の初期推定値を計算
        amplitude = np.sqrt(0.1 * (energy - 1.0))  # エネルギー差の一部を振幅に割り当て
        initial_guess = [amplitude, 0.0]  # [y0, py0]
    
    # シューティング法による周期軌道の計算
    result = root(
        lambda params: shooting_function(params, estimated_period, energy),
        initial_guess,
        method='lm',
        options={'ftol': 1e-10, 'xtol': 1e-10}
    )
    
    if not result.success:
        print("警告: 周期軌道の計算が収束しませんでした。近似解を使用します。")
    
    y0, py0 = result.x
    x0 = 0.0
    V_init = V(x0, y0)
    px0_squared = 2 * (energy - V_init - 0.5 * py0**2)
    if px0_squared < 0:
        print("警告: エネルギー的に不可能な解が返されました。初期推定値を調整します。")
        # 再帰的に別の初期推定値で試す
        return find_unstable_periodic_orbit(energy, [amplitude * 0.5, 0.0])
    
    px0 = -np.sqrt(px0_squared)
    
    # 周期軌道を計算
    initial_state = np.array([x0, y0, px0, py0])
    t_vals, orbit_states = integrate_trajectory(initial_state, [0, estimated_period], dt=0.005)
    
    print(f"NHIM計算完了: 周期 = {estimated_period:.4f}, 残差 = {np.sum(np.abs(result.fun)):.1e}")
    
    return t_vals, orbit_states, estimated_period

def add_unstable_perturbation(nhim_state, magnitude=0.0001): 
    """
    NHIM上の点に井戸に向かう方向の摂動を加える（修正版）
    """
    # 摂動の大きさを調整
    magnitude = magnitude * 0.1  
    
    # コピーを作成
    perturbed_state = nhim_state.copy()
    
    # NHIMの点がx≤0の場合は正の方向に少し移動させる
    if perturbed_state[0] <= 0:
        perturbed_state[0] = 0.001  # x>0の領域に設定
    
    # 不安定方向への摂動を加える（線形化から取得）
    eigvals, eigvecs = saddle_linearization()
    unstable_idx = np.argmax(np.real(eigvals))
    unstable_dir = eigvecs[:, unstable_idx].real
    
    # 左の井戸に向かう方向に調整
    if unstable_dir[0] > 0:
        unstable_dir = -unstable_dir
    
    # 運動量を井戸に向かう方向に設定
    perturbed_state[2] = -abs(perturbed_state[2])  # px < 0 を確保
    
    # 摂動を適用
    for i in range(4):
        perturbed_state[i] += magnitude * unstable_dir[i]
    
    return perturbed_state

# NHIM上でサンプリングする関数も修正
def sample_nhim(nhim_data, n_samples):
    """
    NHIM（周期軌道）上から効果的に点をサンプリング（修正版）
    """
    t_vals, orbit_states, period = nhim_data
    
    # サドル点付近の点を優先してサンプリング
    # 時間ではなく、x=0（サドル点）に近い場所を中心にクラスタリング
    # より多くのサンプルをx=0付近に集中させる
    bias = np.exp(-10 * np.abs(orbit_states[:, 0]))  # x=0に近いほど重みが大きい
    bias = bias / np.sum(bias)  # 正規化
    
    # 重み付きサンプリング
    sample_indices = np.random.choice(
        len(t_vals), 
        size=n_samples,
        replace=True,
        p=bias
    )
    
    # サンプリング
    nhim_samples = orbit_states[sample_indices]
    
    return nhim_samples

# シミュレーション関数も修正
def simulate_single_trajectory(idx, nhim_samples, perturbation_magnitude=0.0001, debug=False):
    """
    NHIM上のサンプル点から始まる単一トラジェクトリのシミュレーション（修正版）
    """
    # NHIM上の点を取得（一様サンプリング）
    sample_idx = idx % len(nhim_samples)
    nhim_state = nhim_samples[sample_idx].copy()  # コピーを作成
    
    # 不安定方向への摂動を追加（井戸方向）
    state0 = add_unstable_perturbation(nhim_state, perturbation_magnitude)
    
    # エネルギー調整（厳密に同じエネルギーになるよう調整）
    target_energy = E
    current_energy = hamiltonian(state0)
    
    # エネルギーを微調整（px方向で調整）- より慎重な調整
    if abs(current_energy - target_energy) > 1e-10:
        # pxは必ず負の値（左の井戸方向）に設定
        px_squared = 2 * (target_energy - V(state0[0], state0[1]) - 0.5 * state0[3]**2)
        if px_squared >= 0:
            state0[2] = -np.sqrt(px_squared)  # 常に負の値（左井戸方向）
        else:
            # エネルギー的に不可能な場合は、y方向の運動エネルギーを調整
            py_squared = 2 * (target_energy - V(state0[0], state0[1])) - state0[2]**2
            if py_squared >= 0:
                state0[3] = np.sqrt(py_squared) * np.sign(state0[3])
    
    # 初期状態のx座標を確認
    if state0[0] <= 0:
        # すでに左側にいる場合は、代わりに別の粒子を使用
        print(f"警告: 粒子{idx}は初期状態ですでにx≤0です。")
        return 0.0  # 即時横切りとして記録
    
    # シミュレーション実行
    final_state, t_event = symplectic_integrate(state0, 0, T_max, dt)
    
    # デバッグ情報（最初のトラジェクトリのみ）
    if debug:
        initial_energy = hamiltonian(state0)
        final_energy = hamiltonian(final_state)
        energy_error = abs(final_energy - initial_energy)
        return t_event, initial_energy, final_energy, energy_error
    
    return t_event

# 代表的な軌道を追跡して可視化（デバッグ用）
def visualize_sample_trajectories(nhim_samples, n_vis=5):
    plt.figure(figsize=(10, 8))
    
    # 複数の軌道を追跡
    for i in range(n_vis):
        # 初期状態を取得
        nhim_state = nhim_samples[i*10].copy()
        state0 = add_unstable_perturbation(nhim_state, 0.0001)
        
        # 詳細なタイムステップで軌道を計算
        t_vals, states = integrate_trajectory(state0, [0, T_max], dt=0.01)
        
        # 軌道をプロット
        plt.plot(states[:, 0], states[:, 1], '-', label=f'Traj {i+1}')
        plt.plot(states[0, 0], states[0, 1], 'o', markersize=8)  # 開始点
    
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig('sample_trajectories.pdf')
    plt.close()

if __name__ == "__main__":
    # CPUコア数の取得
    n_cores = mp.cpu_count()
    print(f"利用可能なCPUコア数: {n_cores}")
    
    # NHIM（不安定周期軌道）の計算
    nhim_data = find_unstable_periodic_orbit(E)
    
    # NHIM上から均等にサンプリング（修正した関数で）
    n_nhim_samples = 500  # サンプル数を増加
    nhim_samples = sample_nhim(nhim_data, n_nhim_samples)
    
    # サンプリングしたNHIM点の表示（確認用）
    plt.figure(figsize=(8, 8))
    plt.plot(nhim_samples[:, 0], nhim_samples[:, 1], 'o-', label='NHIM (x, y)')
    plt.plot(nhim_samples[:, 2], nhim_samples[:, 3], 's-', label='NHIM (px, py)')
    plt.xlabel('x / px')
    plt.ylabel('y / py')
    plt.title('Calculated NHIM (Unstable Periodic Orbit)')
    plt.legend()
    plt.grid(True)
    plt.savefig('nhim_orbit.pdf')
    plt.close()
    
    # 不安定方向への摂動の大きさを小さく設定
    perturbation_magnitude = 0.0001  # 1/10に減少
    
    # 一つの軌道をデバッグ用に実行
    first_result = simulate_single_trajectory(0, nhim_samples, perturbation_magnitude, debug=True)
    print(f"初期エネルギー: {first_result[1]:.8f}")
    print(f"最終エネルギー: {first_result[2]:.8f}")
    print(f"誤差: {first_result[3]:.8e}")
    
    # 並列処理で複数のトラジェクトリを計算
    print(f"{n_traj}個のトラジェクトリを計算中...")
    
    # バッチ処理を使用してメモリ消費を抑える
    batch_size = min(n_traj, 10000)  # 一度に処理するトラジェクトリ数を制限
    recrossing_times = []
    
    # 進捗表示のための変数
    last_percentage = -1
    
    for batch_start in range(0, n_traj, batch_size):
        batch_end = min(batch_start + batch_size, n_traj)
        batch_count = batch_end - batch_start
        
        # バッチ実行（NHIMサンプルとインデックスを渡す）
        batch_results = Parallel(n_jobs=n_cores)(
            delayed(simulate_single_trajectory)(i + batch_start, nhim_samples, perturbation_magnitude) 
            for i in range(batch_count)
        )
        
        recrossing_times.extend(batch_results)
        
        # 現在の進捗を計算（パーセンテージ）
        current_percentage = int(batch_end / n_traj * 100)
        
        # 10%単位でのみ進捗を表示
        if current_percentage % 10 == 0 and current_percentage != last_percentage:
            print(f"進捗: {current_percentage}% 完了 ({batch_end}/{n_traj})")
            last_percentage = current_percentage
    
    # 100%に達していない場合は最後に表示
    if last_percentage < 100:
        print(f"進捗: 100% 完了 ({n_traj}/{n_traj})")
    
    recrossing_times = np.array(recrossing_times)
    
    # 計算時間表示
    calc_time = time.time() - start_time
    print(f"計算時間: {calc_time:.2f}秒")
    
    # 生存確率計算
    t_vals = np.logspace(np.log10(1e-2), np.log10(T_max), 100)
    survival = np.array([np.mean(recrossing_times > t) for t in t_vals])
    
    # 片対数と両対数プロットを並べて表示
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # 片対数プロット（左側）
    axs[0].semilogy(t_vals, survival, marker='o', linestyle='-')
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Population fraction")
    axs[0].set_title("Semi-log Plot: Survival Function")
    axs[0].grid(True, which="both", ls="--")
    
    # 両対数プロット（右側）
    axs[1].loglog(t_vals, survival, marker='o', linestyle='-')
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Population fraction")
    axs[1].set_title("Log-log Plot: Survival Function")
    axs[1].grid(True, which="both", ls="--")
    
    # グラフ全体の調整
    plt.suptitle(f"Recrossing Analysis with NHIM (γ={gamma}, {calc_time:.1f}s)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 結果を保存
    plt.savefig(f"survival_gamma_{gamma}_nhim.pdf")
    plt.show()


