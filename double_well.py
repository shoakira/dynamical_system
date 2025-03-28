#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ダブルウェルモデルにおける再帰渡（recrossing）現象のシミュレーション
シンプレクティック積分法（ベロシティ・ベルレ法）を使用
"""

import numpy as np
import matplotlib.pyplot as plt

# シミュレーションパラメータ
E = 1.01           # 全エネルギー（サドルエネルギー1より上）
n_traj = 10000     # 初期状態の数
T_max = 10000.0     # 最長シミュレーション時間
dt = 0.01         # 時間刻み幅

# 非線形相互作用パラメータ
gamma = 0.3  # 相互作用の強さ

# ポテンシャルとその微分（非線形相互作用項を追加）
def V(x, y):
    # 元のポテンシャル + 非線形相互作用項（x^2 * y^2）
    return (x**2 - 1)**2 + 0.5 * y**2 + y**4 + gamma * x**2 * y**2

def dVdx(x, y):
    # 元の導関数 + 相互作用項の導関数
    # d/dx (gamma * x^2 * y^2) = 2 * gamma * x * y^2
    return 4 * x * (x**2 - 1) + 2 * gamma * x * y**2

def dVdy(x, y):
    # 元の導関数 + 相互作用項の導関数
    # d/dy (gamma * x^2 * y^2) = 2 * gamma * x^2 * y
    return y + 4 * y**3 + 2 * gamma * x**2 * y

# ハミルトニアン（エネルギー）計算関数
def hamiltonian(state):
    x, y, px, py = state
    return 0.5 * (px**2 + py**2) + V(x, y)

# シンプレクティック積分法（ベロシティ・ベルレ法）による数値解法
def symplectic_integrate(state0, t_span, dt=0.01):
    """
    ベロシティ・ベルレ法によるシンプレクティック積分
    
    Parameters:
    -----------
    state0 : array-like
        初期状態 [x, y, px, py]
    t_span : tuple
        (t_start, t_end) のシミュレーション時間範囲
    dt : float
        時間刻み幅
    
    Returns:
    --------
    t_vals : array
        時間点
    states : array
        各時間点での状態
    t_events : array
        サドル横切り時刻
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    
    # 結果の保存配列
    t_vals = np.linspace(t_start, t_end, n_steps)
    states = np.zeros((n_steps, 4))
    states[0] = state0
    
    t_current = t_start
    state = np.array(state0, dtype=float)
    
    step = 0
    t_event = None
    
    while t_current < t_end and step < n_steps - 1:
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
            
            # 次のステップを保存して終了
            step += 1
            t_current += dt
            state = np.array([x_new, y_new, px_new, py_new])
            states[step] = state
            break
        
        # 状態更新
        state = np.array([x_new, y_new, px_new, py_new])
        t_current += dt
        step += 1
        states[step] = state
    
    # 使用したステップ数でトリミング
    t_vals = t_vals[:step+1]
    states = states[:step+1]
    
    return t_vals, states, t_event

# 各軌道の再帰渡時刻を記録
recrossing_times = []

# V(0,0)= (0-1)^2 + 0 + 0 = 1 なので，運動エネルギーは E-1
p0_magnitude = np.sqrt(2 * (E - V(0, 0)))

for i in range(n_traj):
    # 第一象限（p_x > 0, p_y > 0）から一様に角度を選ぶ
    theta = np.random.uniform(0, np.pi / 2)
    px0 = p0_magnitude * np.cos(theta)
    py0 = p0_magnitude * np.sin(theta)
    
    # 初期状態: サドル上の NHIM として x = 0, y = 0 に設定
    state0 = [0.0, 0.0, px0, py0]
    
    # シンプレクティック積分法によるシミュレーション
    t_vals, states, t_event = symplectic_integrate(
        state0,
        [0, T_max],
        dt=dt
    )
    
    # エネルギー保存の確認（デバッグ用）
    if i == 0:
        initial_energy = hamiltonian(state0)
        final_energy = hamiltonian(states[-1])
        print(f"初期エネルギー: {initial_energy:.8f}")
        print(f"最終エネルギー: {final_energy:.8f}")
        print(f"誤差: {abs(final_energy-initial_energy):.8e}")
    
    # イベントが検出されたらその時刻，なければ T_max を記録
    if t_event is not None:
        recrossing_times.append(t_event)
    else:
        recrossing_times.append(T_max)

# 生存確率の計算部分はそのまま
recrossing_times = np.array(recrossing_times)
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
plt.suptitle("Recrossing of the Saddle: Survival Analysis", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # タイトル用のスペースを確保
plt.show()


