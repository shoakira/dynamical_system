#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ダブルウェルモデルにおける再帰渡（recrossing）現象のシミュレーション
シンプレクティック積分法（ベロシティ・ベルレ法）を使用
高速化：並列処理とNumbaによるJITコンパイル
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed
import numba as nb
import os
import time

# パフォーマンス計測のための時間計測開始
start_time = time.time()

# シミュレーションパラメータ
E = 1.01           # 全エネルギー（サドルエネルギー1より上）
n_traj = 50000000     # 初期状態の数
T_max = 100000.0     # 最長シミュレーション時間
dt = 0.01         # 時間刻み幅

# 非線形相互作用パラメータ
gamma = 0.3  # 相互作用の強さ

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

# 単一トラジェクトリのシミュレーション（並列処理用）
def simulate_single_trajectory(seed, debug=False):
    # 乱数シードを設定（並列処理で再現性を確保）
    np.random.seed(seed)
    
    # V(0,0)= (0-1)^2 + 0 + 0 = 1 なので，運動エネルギーは E-1
    p0_magnitude = np.sqrt(2 * (E - V(0, 0)))
    
    # 第一象限（p_x > 0, p_y > 0）から一様に角度を選ぶ
    theta = np.random.uniform(0, np.pi / 2)
    px0 = p0_magnitude * np.cos(theta)
    py0 = p0_magnitude * np.sin(theta)
    
    # 初期状態: サドル上の NHIM として x = 0, y = 0 に設定
    # Numba互換の方法で配列を作成
    state0 = np.empty(4, dtype=np.float64)
    state0[0] = 0.0
    state0[1] = 0.0
    state0[2] = px0
    state0[3] = py0
    
    # シミュレーション実行
    final_state, t_event = symplectic_integrate(state0, 0, T_max, dt)
    
    # デバッグ情報（最初のトラジェクトリのみ）
    if debug:
        initial_energy = hamiltonian(state0)
        final_energy = hamiltonian(final_state)
        energy_error = abs(final_energy - initial_energy)
        return t_event, initial_energy, final_energy, energy_error
    
    return t_event

if __name__ == "__main__":
    # CPUコア数の取得
    n_cores = mp.cpu_count()
    print(f"利用可能なCPUコア数: {n_cores}")
    
    # デバッグ用に最初のトラジェクトリを実行
    first_result = simulate_single_trajectory(0, debug=True)
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
        
        # シード値を変えてバッチ実行
        batch_results = Parallel(n_jobs=n_cores)(
            delayed(simulate_single_trajectory)(i + batch_start) 
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
    plt.suptitle(f"Recrossing Analysis (γ={gamma}, {calc_time:.1f}s)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 結果を保存
    plt.savefig(f"survival_gamma_{gamma}.pdf")
    plt.show()


