#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3自由度ダブルウェルポテンシャルの再交差解析

特徴:
- 井戸は x = 1, -1 にあり、x = 0 がサドル
- 井戸内で x, y, z の非線形相互作用によるミキシング
- 初期状態はサドル上のNHIM上に等エネルギーかつ一様分布
- vx は微小に正の値
- 生存時間分布の両対数プロットと片対数プロット
- 滞在時間ごとのNHIM上の分布の可視化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from joblib import Parallel, delayed
import multiprocessing as mp
import numba as nb
import os
from tqdm import tqdm
import datetime

# シミュレーションパラメータ
E_target = 1.05        # 全エネルギー（サドルエネルギー1.0より上）
n_traj = 5000          # 初期状態の数
t_max = 1000.0         # 最長シミュレーション時間
vx_init = 0.01         # 初期vxの値（微小に正）

# 非線形相互作用パラメータ
gamma = 0.1            # x-y 相互作用の強さ
beta = 0.15            # x-z 相互作用の強さ
alpha = 0.2            # y-z 相互作用の強さ

# ポテンシャル関数
def V(x, y, z):
    """
    3自由度ダブルウェルポテンシャル
    井戸は x = 1, -1 にあり、x = 0 がサドル
    井戸内で x, y, z の非線形相互作用によるミキシング
    """
    # 基本的なダブルウェルポテンシャル (x^4 - 2x^2)
    v_double_well = (x**2 - 1)**2
    
    # 横方向の調和ポテンシャル
    v_harmonic = 0.5 * (y**2 + z**2)
    
    # 非線形相互作用項（ミキシングを引き起こす）
    v_interaction = gamma * x**2 * y**2 + beta * x**2 * z**2 + alpha * y**2 * z**2
    
    return v_double_well + v_harmonic + v_interaction

@nb.njit
def V_fast(x, y, z):
    """NumbaでJIT最適化されたポテンシャル関数"""
    v_double_well = (x**2 - 1)**2
    v_harmonic = 0.5 * (y**2 + z**2)
    v_interaction = gamma * x**2 * y**2 + beta * x**2 * z**2 + alpha * y**2 * z**2
    return v_double_well + v_harmonic + v_interaction

# 力の計算（ポテンシャルの勾配）
def dVdx(x, y, z):
    return 4 * x**3 - 4 * x + 2 * gamma * x * y**2 + 2 * beta * x * z**2

def dVdy(x, y, z):
    return y + 2 * gamma * x**2 * y + 2 * alpha * y * z**2

def dVdz(x, y, z):
    return z + 2 * beta * x**2 * z + 2 * alpha * y**2 * z

@nb.njit
def dVdx_fast(x, y, z):
    return 4 * x**3 - 4 * x + 2 * gamma * x * y**2 + 2 * beta * x * z**2

@nb.njit
def dVdy_fast(x, y, z):
    return y + 2 * gamma * x**2 * y + 2 * alpha * y * z**2

@nb.njit
def dVdz_fast(x, y, z):
    return z + 2 * beta * x**2 * z + 2 * alpha * y**2 * z

# 全エネルギー計算
def hamiltonian(state):
    x, y, z, vx, vy, vz = state
    kinetic = 0.5 * (vx**2 + vy**2 + vz**2)
    potential = V(x, y, z)
    return kinetic + potential

# 運動方程式
def equations_of_motion(t, state):
    x, y, z, vx, vy, vz = state
    
    # 加速度 = -grad(V)
    ax = -dVdx(x, y, z)
    ay = -dVdy(x, y, z)
    az = -dVdz(x, y, z)
    
    return [vx, vy, vz, ax, ay, az]

# イベント関数: x = 0 (サドル)の交差を検出
def crossing_event(t, state):
    x = state[0]
    return x

crossing_event.terminal = False  # イベント検出時にも積分を続ける
crossing_event.direction = 1     # 正の方向への交差のみ検出

# サドル点上の2自由度NHIM計算
def linearize_at_saddle():
    """
    サドル点でのポテンシャルの線形化解析
    固有値と固有ベクトルを返す
    """
    # サドル点 (0,0,0) でのポテンシャルのヘッセ行列
    H = np.array([
        [-4 + 2*gamma*0**2 + 2*beta*0**2, 2*gamma*0*0, 2*beta*0*0],
        [2*gamma*0*0, 1 + 2*gamma*0**2 + 2*alpha*0**2, 2*alpha*0*0],
        [2*beta*0*0, 2*alpha*0*0, 1 + 2*beta*0**2 + 2*alpha*0**2]
    ])
    
    # 簡略化（0での評価）
    H = np.array([
        [-4, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # 固有値と固有ベクトルを計算
    eigvals, eigvecs = np.linalg.eigh(H)
    
    return eigvals, eigvecs

# NHIM上の初期条件を生成
def generate_nhim_initial_conditions(n_samples):
    """
    サドル点のNHIM上で等エネルギーとなる初期条件を生成
    
    Parameters:
    -----------
    n_samples : int
        生成するサンプル数
        
    Returns:
    --------
    initial_states : ndarray
        初期状態の配列 [x, y, z, vx, vy, vz]
    """
    initial_states = []
    
    # サドル点での線形化解析
    eigvals, eigvecs = linearize_at_saddle()
    
    # x方向の固有値は負（不安定方向）
    # その他の方向は安定（ポジティブ）
    
    # NHIM上のサンプリング（y, z空間でランダムに）
    while len(initial_states) < n_samples:
        # x = 0 (サドル点)
        x = 0.0
        
        # y, z をランダムにサンプリング (範囲は調整可能)
        r = np.sqrt(np.random.uniform(0, 0.5))  # 極座標でサンプリング（一様分布のため）
        theta = np.random.uniform(0, 2*np.pi)
        
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        
        # vx は微小に正
        vx = vx_init
        
        # 現在のポテンシャルエネルギー
        V_current = V(x, y, z)
        
        # 残りのエネルギーから vy, vz を決定
        K_remain = E_target - V_current - 0.5 * vx**2
        
        if K_remain <= 0:
            continue  # エネルギー不足なのでスキップ
        
        # vy, vz 方向にランダムなエネルギー分配
        phi = np.random.uniform(0, 2*np.pi)
        vy = np.sqrt(2 * K_remain) * np.cos(phi)
        vz = np.sqrt(2 * K_remain) * np.sin(phi)
        
        # 初期状態を追加
        state = [x, y, z, vx, vy, vz]
        
        # エネルギーチェック
        energy = hamiltonian(state)
        if abs(energy - E_target) < 1e-6:
            initial_states.append(state)
    
    return np.array(initial_states)

# 単一のトラジェクトリをシミュレーション
def simulate_trajectory(initial_state):
    """
    1つのトラジェクトリをシミュレーションし、再交差時間を返す
    """
    # 積分設定
    t_span = (0, t_max)
    
    # イベント検出を設定して積分
    solution = solve_ivp(
        equations_of_motion, 
        t_span, 
        initial_state, 
        method='RK45',
        events=crossing_event,
        rtol=1e-8,
        atol=1e-8
    )
    
    # エネルギー保存のチェック
    initial_energy = hamiltonian(initial_state)
    final_energy = hamiltonian(solution.y[:, -1])
    energy_error = abs(final_energy - initial_energy)
    
    # イベント時間（再交差時間）を取得
    if len(solution.t_events[0]) > 0:
        # 最初の再交差イベント（x = 0 を正方向に横切る時点）
        recrossing_time = solution.t_events[0][0]
        recrossed = True
    else:
        # 再交差なし
        recrossing_time = t_max
        recrossed = False
    
    # 初期位置も返す（後の分析用）
    initial_position = initial_state[:3]
    
    return {
        'time': recrossing_time,
        'recrossed': recrossed,
        'initial_pos': initial_position,
        'energy_error': energy_error
    }

@nb.njit
def symplectic_integrate(state0, t_start, t_end, dt=0.01):
    """
    ベロシティ・ベルレ法によるシンプレクティック積分（高速版）
    """
    steps = int((t_end - t_start) / dt) + 1
    
    # 初期状態
    x, y, z, vx, vy, vz = state0
    
    # イベント検出用変数
    t_crossing = None
    
    for step in range(steps):
        t = t_start + step * dt
        
        # 1. 位置の半ステップ更新
        x += vx * dt/2
        y += vy * dt/2
        z += vz * dt/2
        
        # 2. 速度の全ステップ更新
        vx -= dVdx(x, y, z) * dt
        vy -= dVdy(x, y, z) * dt
        vz -= dVdz(x, y, z) * dt
        
        # 3. 位置の残り半ステップ更新
        x += vx * dt/2
        y += vy * dt/2
        z += vz * dt/2
        
        # サドル再交差イベント検出（x=0の正方向への交差）
        if t > 0.1 and x > 0 and vx < 0:  # サドルの右側から左向きの運動量
            # 前の時間ステップでx>0だったことを確認する必要がある（ここでは簡略化）
            t_crossing = t
            break
    
    return [x, y, z, vx, vy, vz], t_crossing

def simulate_trajectory_fast(initial_state):
    """
    高速化されたトラジェクトリシミュレーション
    """
    # シンプレクティック積分を使用
    final_state, recrossing_time = symplectic_integrate(
        initial_state, 0, t_max, dt=0.01
    )
    
    # 再交差の判定
    recrossed = recrossing_time is not None
    
    # 再交差しなかった場合は最大時間
    if not recrossed:
        recrossing_time = t_max
    
    # エネルギー保存のチェック（オプション）
    initial_energy = hamiltonian(initial_state)
    final_energy = hamiltonian(final_state)
    energy_error = abs(final_energy - initial_energy)
    
    # 初期位置も返す（後の分析用）
    initial_position = initial_state[:3]
    
    return {
        'time': recrossing_time,
        'recrossed': recrossed,
        'initial_pos': initial_position,
        'energy_error': energy_error
    }

# 生存確率の計算
def calculate_survival_probability(results, time_points):
    """
    生存確率を計算
    
    Parameters:
    -----------
    results : list
        シミュレーション結果のリスト
    time_points : ndarray
        評価する時間点
        
    Returns:
    --------
    survival_prob : ndarray
        各時間点での生存確率
    """
    recrossing_times = [result['time'] for result in results]
    
    survival_prob = []
    for t in time_points:
        # 時間 t より長く生存している割合
        survived = sum(1 for time in recrossing_times if time > t)
        survival_prob.append(survived / len(recrossing_times))
    
    return np.array(survival_prob)

# 滞在時間ごとのNHIM上の分布を分析
def analyze_residence_time_distribution(results):
    """
    滞在時間と初期位置の関係を分析
    """
    # 滞在時間をビンに分割
    bin_edges = np.logspace(np.log10(0.1), np.log10(t_max), 5)  # 4つのビン
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    
    # 各ビンに属するトラジェクトリのインデックスを取得
    binned_indices = []
    for i in range(len(bin_edges)-1):
        indices = [j for j, result in enumerate(results) 
                  if bin_edges[i] <= result['time'] < bin_edges[i+1]]
        binned_indices.append(indices)
    
    # 各ビンの初期位置を取得
    binned_positions = []
    for indices in binned_indices:
        positions = [results[i]['initial_pos'] for i in indices]
        binned_positions.append(positions)
    
    return bin_edges, bin_labels, binned_positions

def analyze_residence_time_distribution_fast(results):
    """
    メモリ効率が良い滞在時間分析
    """
    # 対数スケールでビン分割（より効率的）
    log_bins = np.linspace(np.log10(0.1), np.log10(t_max), 5)
    bin_edges = 10**log_bins
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    
    # プリアロケーションで効率化
    binned_positions = [[] for _ in range(len(bin_edges)-1)]
    
    # 一度のループで分類（より効率的）
    for result in results:
        time = result['time']
        for i in range(len(bin_edges)-1):
            if bin_edges[i] <= time < bin_edges[i+1]:
                binned_positions[i].append(result['initial_pos'])
                break
    
    return bin_edges, bin_labels, binned_positions

# 滞在時間ごとの分布を可視化
def visualize_residence_time_bins(bin_edges, bin_labels, binned_positions):
    """
    滞在時間ビンごとの初期位置の分布を可視化
    """
    # 2D可視化（y-z平面）
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, (positions, label) in enumerate(zip(binned_positions, bin_labels)):
        if not positions:
            continue
            
        positions = np.array(positions)
        y_coords = positions[:, 1]  # y座標
        z_coords = positions[:, 2]  # z座標
        
        axs[i].scatter(y_coords, z_coords, alpha=0.6)
        axs[i].set_title(f'滞在時間: {label}')
        axs[i].set_xlabel('y')
        axs[i].set_ylabel('z')
        axs[i].grid(True)
        axs[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('residence_time_bins_2d.pdf')
    plt.close()
    
    # 3D可視化（極座標変換して可視化）
    fig = plt.figure(figsize=(15, 12))
    
    for i, (positions, label) in enumerate(zip(binned_positions, bin_labels)):
        if not positions:
            continue
            
        positions = np.array(positions)
        y_coords = positions[:, 1]
        z_coords = positions[:, 2]
        
        # 極座標に変換（r, theta）
        r = np.sqrt(y_coords**2 + z_coords**2)
        theta = np.arctan2(z_coords, y_coords)
        
        ax = fig.add_subplot(2, 2, i+1, projection='polar')
        scatter = ax.scatter(theta, r, c=r, cmap='viridis', alpha=0.6)
        ax.set_title(f'滞在時間: {label}')
        plt.colorbar(scatter, ax=ax, label='r = sqrt(y^2 + z^2)')
    
    plt.tight_layout()
    plt.savefig('residence_time_bins_polar.pdf')
    plt.close()

# ポテンシャルを可視化する関数
def visualize_potential():
    """
    断面でのポテンシャルを可視化
    """
    # x-y平面での断面（z=0）
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = V(X[j, i], Y[j, i], 0)
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(label='Potential Energy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Potential (z=0 Cross Section)')
    plt.axvline(x=0, color='r', linestyle='--', label='Saddle (x=0)')
    plt.legend()
    plt.savefig('potential_xy_section.pdf')
    plt.close()
    
    # x-z平面での断面（y=0）
    z = np.linspace(-2, 2, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(z)):
            Y[j, i] = V(X[j, i], 0, Z[j, i])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Z, Y, 50, cmap='viridis')
    plt.colorbar(label='Potential Energy')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Potential (y=0 Cross Section)')
    plt.axvline(x=0, color='r', linestyle='--', label='Saddle (x=0)')
    plt.legend()
    plt.savefig('potential_xz_section.pdf')
    plt.close()
    
    # サドル面でのy-z平面（x=0、NHIM上）
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    Y, Z = np.meshgrid(y, z)
    X = np.zeros_like(Y)
    
    for i in range(len(y)):
        for j in range(len(z)):
            X[j, i] = V(0, Y[j, i], Z[j, i])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(Y, Z, X, 50, cmap='viridis')
    plt.colorbar(label='Potential Energy at Saddle (x=0)')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('NHIM Potential Surface (x=0)')
    plt.savefig('saddle_yz_section.pdf')
    plt.close()

# メイン関数
def main():
    start_time = time.time()
    
    # ポテンシャルの可視化
    print("ポテンシャルを可視化中...")
    visualize_potential()
    
    # NHIM上の初期条件を生成
    print(f"{n_traj}個の初期条件を生成中...")
    initial_states = generate_nhim_initial_conditions(n_traj)
    
    # 並列シミュレーション
    print("軌道をシミュレーション中...")
    num_cores = mp.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(simulate_trajectory)(state) for state in initial_states
    )
    
    # エネルギー誤差の確認
    energy_errors = [result['energy_error'] for result in results]
    max_energy_error = max(energy_errors)
    avg_energy_error = sum(energy_errors) / len(energy_errors)
    print(f"エネルギー保存: 最大誤差 = {max_energy_error:.2e}, 平均誤差 = {avg_energy_error:.2e}")
    
    # 再交差の統計
    recrossed_count = sum(1 for result in results if result['recrossed'])
    print(f"再交差率: {recrossed_count}/{len(results)} ({recrossed_count/len(results)*100:.1f}%)")
    
    # 生存確率の計算と可視化
    time_points = np.logspace(-1, np.log10(t_max), 100)
    survival_prob = calculate_survival_probability(results, time_points)
    
    # 両対数プロット
    plt.figure(figsize=(10, 8))
    plt.loglog(time_points, survival_prob, 'o-', linewidth=2)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Survival Probability (log scale)')
    plt.title('Recrossing Time Distribution (Log-Log Plot)')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('survival_probability_loglog.pdf')
    plt.close()
    
    # 片対数プロット（分布が線形、時間が対数）
    plt.figure(figsize=(10, 8))
    plt.semilogx(time_points, survival_prob, 'o-', linewidth=2)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Survival Probability')
    plt.title('Recrossing Time Distribution (Semi-Log Plot)')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('survival_probability_semilog.pdf')
    plt.close()
    
    # 滞在時間と初期位置の関係を分析
    print("滞在時間分布を分析中...")
    bin_edges, bin_labels, binned_positions = analyze_residence_time_distribution_fast(results)
    
    # 滞在時間ごとの分布を可視化
    print("滞在時間ビンごとの分布を可視化中...")
    visualize_residence_time_bins(bin_edges, bin_labels, binned_positions)
    
    # 実行時間の表示
    elapsed_time = time.time() - start_time
    print(f"実行時間: {elapsed_time:.2f}秒")

def main_optimized():
    global_start_time = time.time()
    log_progress("プログラム実行開始")
    
    # データ保存ディレクトリの確認/作成
    if not os.path.exists('data'):
        os.makedirs('data')
        log_progress("データ保存用ディレクトリ 'data' を作成しました")
    
    # ポテンシャルの可視化
    log_progress("ポテンシャルを可視化中...")
    visualize_potential()
    vis_time = time.time() - global_start_time
    log_progress("ポテンシャル可視化完了", vis_time)
    
    # NHIM上の初期条件を生成
    gen_start_time = time.time()
    log_progress(f"{n_traj}個の初期条件を生成中...")
    initial_states = generate_nhim_initial_conditions(n_traj)
    gen_time = time.time() - gen_start_time
    log_progress(f"初期条件生成完了: {len(initial_states)}個", gen_time)
    
    # 進捗表示付きシミュレーション実行
    sim_start_time = time.time()
    num_cores = mp.cpu_count()
    results = run_simulations_with_progress(initial_states, num_cores, batch_size=1000)
    sim_time = time.time() - sim_start_time
    
    # エネルギー誤差の確認
    analysis_start_time = time.time()
    log_progress("シミュレーション結果を分析中...")
    
    energy_errors = [result['energy_error'] for result in results]
    max_energy_error = max(energy_errors)
    avg_energy_error = sum(energy_errors) / len(energy_errors)
    log_progress(f"エネルギー保存: 最大誤差 = {max_energy_error:.2e}, 平均誤差 = {avg_energy_error:.2e}")
    
    # 再交差の統計
    recrossed_count = sum(1 for result in results if result['recrossed'])
    recross_rate = recrossed_count/len(results)*100
    log_progress(f"再交差率: {recrossed_count}/{len(results)} ({recross_rate:.1f}%)")
    
    # 生存確率の計算と可視化
    log_progress("再交差軌道の生存確率を計算・プロット中...")
    time_points = np.logspace(-1, np.log10(t_max), 100)
    survival_prob = calculate_survival_probability(results, time_points)
    
    # 両対数プロット
    plt.figure(figsize=(10, 8))
    plt.loglog(time_points, survival_prob, 'o-', linewidth=2)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Survival Probability (log scale)')
    plt.title('Recrossing Time Distribution (Log-Log Plot)')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('data/survival_probability_loglog.pdf')
    plt.close()
    
    # 片対数プロット（分布が線形、時間が対数）
    plt.figure(figsize=(10, 8))
    plt.semilogx(time_points, survival_prob, 'o-', linewidth=2)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Survival Probability')
    plt.title('Recrossing Time Distribution (Semi-Log Plot)')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig('data/survival_probability_semilog.pdf')
    plt.close()
    log_progress("生存確率プロット完了")
    
    # 滞在時間と初期位置の関係を分析
    log_progress("滞在時間分布を分析中...")
    bin_edges, bin_labels, binned_positions = analyze_residence_time_distribution_fast(results)
    
    # 滞在時間ごとの分布を可視化
    log_progress("滞在時間ビンごとの分布を可視化中...")
    visualize_residence_time_bins(bin_edges, bin_labels, binned_positions)
    analysis_time = time.time() - analysis_start_time
    log_progress("分析と可視化完了", analysis_time)
    
    # 実行時間の表示
    total_time = time.time() - global_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒" if hours > 0 else f"{int(minutes)}分 {seconds:.2f}秒"
    
    log_progress(f"実行完了! 合計時間: {time_str}")
    log_progress(f"内訳: 初期条件生成={gen_time:.2f}秒, シミュレーション={sim_time:.2f}秒, 分析={analysis_time:.2f}秒")

# 進捗表示用の関数
def log_progress(message, elapsed_time=None):
    """進捗状況を時刻付きでログ出力"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    if elapsed_time:
        print(f"[{timestamp}] {message} (経過時間: {elapsed_time:.2f}秒)")
    else:
        print(f"[{timestamp}] {message}")

# シミュレーション進捗表示関数（main_optimized関数内で使用）
def run_simulations_with_progress(initial_states, n_cores, batch_size=1000):
    """
    進捗表示付きシミュレーション実行
    """
    total_batches = (len(initial_states) + batch_size - 1) // batch_size
    results = []
    
    log_progress(f"シミュレーション開始: {len(initial_states)}個のトラジェクトリ, {n_cores}コア使用")
    batch_start_time = time.time()
    
    for batch_idx, batch_start in enumerate(range(0, len(initial_states), batch_size)):
        batch_end = min(batch_start + batch_size, len(initial_states))
        batch_states = initial_states[batch_start:batch_end]
        batch_size_actual = len(batch_states)
        
        # 経過時間と推定残り時間を計算
        if batch_idx > 0:
            avg_time_per_batch = (time.time() - batch_start_time) / batch_idx
            est_remaining = avg_time_per_batch * (total_batches - batch_idx)
            remaining_str = f", 残り約{est_remaining:.1f}秒"
        else:
            remaining_str = ""
        
        log_progress(f"バッチ {batch_idx+1}/{total_batches} 処理中 ({batch_start+1}-{batch_end}/{len(initial_states)}){remaining_str}")
        
        # バッチごとに並列処理（verboseを1に設定して進捗表示）
        batch_results = Parallel(n_jobs=n_cores, verbose=1)(
            delayed(simulate_trajectory_fast)(state) for state in batch_states
        )
        
        # 結果を蓄積
        results.extend(batch_results)
        
        # バッチ完了情報
        batch_time = time.time() - batch_start_time
        completion_percent = (batch_end / len(initial_states)) * 100
        recrossed_in_batch = sum(1 for r in batch_results if r['recrossed'])
        
        log_progress(f"バッチ {batch_idx+1} 完了: {recrossed_in_batch}/{batch_size_actual}個が再交差 "
                    f"(合計進捗: {completion_percent:.1f}%)")
    
    total_time = time.time() - batch_start_time
    log_progress(f"全{len(initial_states)}個のシミュレーション完了", total_time)
    
    return results

if __name__ == "__main__":
    main()
