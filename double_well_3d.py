# -*- coding: utf-8 -*-
"""
3自由度非線形ダブルウェルモデルにおけるNHIMからの再交差ダイナミクス

- 近似NHIM上の初期条件を使用
- シンプレクティック積分 (Velocity Verlet)
- 再交差時間の分布とNHIM上の初期位置依存性を可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed
import numba as nb
import os
import time
from mpl_toolkits.mplot3d import Axes3D # 3Dプロット用 (今回は使用しないが参考)
from matplotlib.colors import LogNorm

# --- データ保存用フォルダ ---
data_folder = "data_3dof"
os.makedirs(data_folder, exist_ok=True)
print(f"データは '{data_folder}/' フォルダに保存されます。")

# --- パラメータ ---
E = 1.05             # 全エネルギー (サドルエネルギー 1.0 より少し上)
n_traj = 100000       # 初期条件の数
T_max = 1000000.0       # 最大シミュレーション時間
dt = 0.002            # 時間刻み幅 (少し大きめにして計算速度を優先)

# ポテンシャルパラメータ
gamma = 0.5           # 非線形相互作用の強さ (ミキシング)
omega_y = 1.0         # y方向の振動数 (サドル点での線形化)
omega_z = 1.0         # z方向の振動数 (サドル点での線形化) -> 簡単のため omega_y = omega_z = 1 とする

# NHIM近似パラメータ
px_init_val = 1e-5    # 初期pxの微小正値 (エネルギー決定に使う場合はNone)
use_fixed_px = False   # pxを固定するか、エネルギーから決定するか

# --- Numba 高速化関数 ---
@nb.njit
def V(x, y, z, gamma_):
    """ポテンシャルエネルギー関数 V(x, y, z)"""
    # x方向ダブルウェル + y,z方向調和振動子 + 非線形相互作用
    term_dw = (x**2 - 1)**2
    term_harm = 0.5 * (omega_y**2 * y**2 + omega_z**2 * z**2)
    term_int = gamma_ * x**2 * (y**2 + z**2)
    return term_dw + term_harm + term_int

@nb.njit
def force(x, y, z, gamma_):
    """力 (-dV/dq) を計算"""
    # -dV/dx
    fx = -(4 * x * (x**2 - 1) + 2 * gamma_ * x * (y**2 + z**2))
    # -dV/dy
    fy = -(omega_y**2 * y + 2 * gamma_ * x**2 * y)
    # -dV/dz
    fz = -(omega_z**2 * z + 2 * gamma_ * x**2 * z)
    return fx, fy, fz

@nb.njit
def hamiltonian(state, gamma_):
    """ハミルトニアン H(q, p)"""
    x, y, z, px, py, pz = state
    kinetic = 0.5 * (px**2 + py**2 + pz**2)
    potential = V(x, y, z, gamma_)
    return kinetic + potential

@nb.njit
def velocity_verlet_step(state, dt_, gamma_):
    """Velocity Verlet法による1ステップ積分"""
    x, y, z, px, py, pz = state

    # 1. 運動量の半ステップ更新 (p(t+dt/2))
    fx, fy, fz = force(x, y, z, gamma_)
    px_half = px + fx * dt_ / 2
    py_half = py + fy * dt_ / 2
    pz_half = pz + fz * dt_ / 2

    # 2. 位置の全ステップ更新 (q(t+dt))
    x_new = x + px_half * dt_
    y_new = y + py_half * dt_
    z_new = z + pz_half * dt_

    # 3. 新しい位置での力を計算
    fx_new, fy_new, fz_new = force(x_new, y_new, z_new, gamma_)

    # 4. 運動量の残り半ステップ更新 (p(t+dt))
    px_new = px_half + fx_new * dt_ / 2
    py_new = py_half + fy_new * dt_ / 2
    pz_new = pz_half + fz_new * dt_ / 2

    return np.array([x_new, y_new, z_new, px_new, py_new, pz_new], dtype=np.float64)

@nb.njit
def integrate_trajectory_recross(state0, t_max_, dt_, gamma_):
    """
    軌道を積分し、最初の再交差イベント (x=0, px<0) を検出する。
    戻り値: (最終状態, 再交差時刻 or t_max)
    """
    state = state0.copy()
    t_curr = 0.0
    t_recross = t_max_ # デフォルトは最大時間

    # 初期状態は x=0, px>0 を想定
    entered_well = False # x>0 の井戸領域に入ったか

    max_steps = int(t_max_ / dt_) + 1
    for step in range(max_steps):
        x_prev = state[0]
        state = velocity_verlet_step(state, dt_, gamma_)
        x_curr, px_curr = state[0], state[3]
        t_curr += dt_

        # 一度 x>0 の領域に入ったことを確認
        if not entered_well and x_curr > 1e-4: # 少し正の領域に入ったらフラグON
            entered_well = True

        # 再交差イベント検出: x>0 から x<=0 になり、かつ px<0
        if entered_well and x_prev > 0 and x_curr <= 0 and px_curr < 0:
            # 線形補間でより正確な時刻を計算
            if x_prev != x_curr:
                 t_event_ratio = x_prev / (x_prev - x_curr)
                 t_recross = (t_curr - dt_) + t_event_ratio * dt_
            else:
                 t_recross = t_curr - dt_ / 2
            # イベント発生時の状態でループを抜ける
            # state はステップ終了時の状態なので、イベント発生点ではないが、
            # ここでは時刻だけが重要なので、このままで良い
            break

        if t_curr >= t_max_:
            break

    return state, t_recross

# --- 初期条件生成 (近似NHIMサンプリング) ---
def generate_initial_conditions(n_samples, energy, gamma_, px_fixed=None):
    """
    x=0 上の近似NHIMからエネルギーEを持つ初期条件を一様ランダムに生成
    戻り値: initial_conditions (list of states), params (list of dicts for analysis)
    """
    initial_conditions = []
    params_list = []
    generated_count = 0
    attempts = 0
    max_attempts = n_samples * 100 # 失敗しすぎたら諦める

    print(f"Generating {n_samples} initial conditions on approximate NHIM (E={energy})...")

    while generated_count < n_samples and attempts < max_attempts:
        attempts += 1

        # 1. y-z 平面内のエネルギー E_center を計算
        x0 = 0.0
        potential_at_saddle = V(x0, 0.0, 0.0, gamma_) # = 1.0

        if px_fixed is not None:
            # px を固定する場合
            px0 = px_fixed
            kinetic_center_target = energy - potential_at_saddle - 0.5 * px0**2
        else:
            # px もエネルギーから決定する場合 (後で計算)
            kinetic_center_target = -1 # 仮の値

        # 2. y-z 平面内のエネルギーを E_y と E_z にランダムに分配
        #    H_y + H_z = E_yz_target = E - V(0,y,z) - 0.5*px^2
        #    近似: V(0,y,z) ≈ 1 + 0.5*(y^2+z^2) を使う
        #    => H_y + H_z ≈ E - 1 - 0.5*px^2 = E_center
        #    E_center は y,z 方向の振動エネルギーの上限に対応
        if px_fixed is not None:
            e_center_approx = energy - potential_at_saddle - 0.5 * px0**2
        else:
            # pxが決まってないので、E_centerの正確な値は不明
            # とりあえずエネルギーの上限として E-1 を使う (px=0の場合)
            e_center_approx = energy - potential_at_saddle
            if e_center_approx < 0: # エネルギーが低すぎる
                if attempts % 1000 == 0: print("Warning: E_center_approx < 0")
                continue


        # E_y を [0, E_center] から一様ランダムに選ぶ (Action変数に比例)
        e_y = np.random.uniform(0, e_center_approx)
        e_z = e_center_approx - e_y # 残りを E_z に (これは近似)
        if e_z < 0: # 分配がうまくいかなかった場合 (近似のため起こりうる)
             e_z = 0
             e_y = e_center_approx # E_y が全エネルギーを持つ


        # 3. 各振動子の位相 phi_y, phi_z をランダムに選ぶ
        phi_y = np.random.uniform(0, 2 * np.pi)
        phi_z = np.random.uniform(0, 2 * np.pi)

        # 4. (y, py) と (z, pz) を計算 (Action-Angle 変数から)
        #    H = 0.5*p^2 + 0.5*omega^2*q^2 = E_osc
        #    q = sqrt(2*E_osc / omega^2) * cos(phi)
        #    p = -sqrt(2*E_osc) * sin(phi) * omega (間違いやすい！)
        #    p = -sqrt(2*E_osc*omega^2) * sin(phi)
        #    => p = -omega * sqrt(2*E_osc) * sin(phi)
        #    omega=1 の場合: q = sqrt(2*E_osc)*cos(phi), p = -sqrt(2*E_osc)*sin(phi)
        y0 = np.sqrt(2 * e_y / omega_y**2) * np.cos(phi_y) if omega_y > 0 else 0
        py0 = -omega_y * np.sqrt(2 * e_y) * np.sin(phi_y) if omega_y > 0 else 0 # omega掛けるの忘れない
        #py0 = -np.sqrt(2 * e_y) * np.sin(phi_y) * omega_y # omega=1なので同じ

        z0 = np.sqrt(2 * e_z / omega_z**2) * np.cos(phi_z) if omega_z > 0 else 0
        pz0 = -omega_z * np.sqrt(2 * e_z) * np.sin(phi_z) if omega_z > 0 else 0 # omega掛けるの忘れない
        #pz0 = -np.sqrt(2 * e_z) * np.sin(phi_z) * omega_z # omega=1なので同じ


        # 5. px をエネルギー保存から決定 (px_fixed is None の場合)
        if px_fixed is None:
            potential_yz = V(x0, y0, z0, gamma_)
            kinetic_yz = 0.5 * (py0**2 + pz0**2)
            px_squared_needed = 2 * (energy - potential_yz - kinetic_yz)

            # 修正: 数値誤差による小さな負の値は許容して0に設定
            if px_squared_needed < 0:
                if px_squared_needed < -1e-8:  # 真に問題がある場合は警告
                    if attempts % 100000 == 0:  # 警告表示頻度を大幅に下げる
                        print(f"Warning: px^2 significantly negative ({px_squared_needed:.2e})")
                    continue  # このサンプルは棄却
                else:
                    # 許容できる数値誤差の場合はゼロに設定して続行
                    px_squared_needed = 0.0
            
            # 非常に小さい正の値もチェック
            if px_squared_needed < 1e-10:
                px0 = 1e-5  # 最小の正の値を設定
            else:
                px0 = np.sqrt(px_squared_needed)  # 正の値を取る
        else:
            # px_fixed を使う場合は、エネルギーが合うかチェック (近似のためずれる可能性)
             current_H = 0.5 * (px0**2 + py0**2 + pz0**2) + V(x0, y0, z0, gamma_)
             if abs(current_H - energy) > 1e-6: # 許容誤差
                 # print(f"Warning: Energy mismatch {abs(current_H - energy):.2e}")
                 # エネルギーを再調整するか、棄却するか
                 # ここでは棄却
                 continue


        # 状態ベクトルを作成
        state0 = np.array([x0, y0, z0, px0, py0, pz0])

        # 最終エネルギーチェック
        final_H = hamiltonian(state0, gamma_)
        if abs(final_H - energy) > 1e-7:
            print(f"Fatal Warning: Final energy mismatch! H={final_H:.8f}, E={energy:.8f}")
            continue # 棄却

        initial_conditions.append(state0)
        # 解析用にパラメータを保存
        params = {'E_y': e_y, 'E_z': e_z, 'phi_y': phi_y, 'phi_z': phi_z,
                  'y0': y0, 'py0': py0, 'z0': z0, 'pz0': pz0, 'px0': px0}
        params_list.append(params)
        generated_count += 1

        if generated_count % (n_samples // 10) == 0 and generated_count > 0:
            print(f"  Generated {generated_count}/{n_samples} conditions...")

    if generated_count < n_samples:
        print(f"Warning: Could only generate {generated_count} initial conditions after {max_attempts} attempts.")

    print(f"Successfully generated {generated_count} initial conditions.")
    return initial_conditions, params_list


# --- 並列計算用ラッパー ---
def simulate_single_wrapper(args):
    """joblib用のラッパー関数"""
    idx, state0, t_max_, dt_, gamma_ = args
    try:
        final_state, t_recross = integrate_trajectory_recross(state0, t_max_, dt_, gamma_)
        return idx, t_recross
    except Exception as e:
        print(f"Error in trajectory {idx}: {e}")
        return idx, t_max_ # エラー時は最大時間を返す


def extract_nhim_structure_with_tsne(phi_y, phi_z, recross_times):
    """t-SNEを使った非線形構造抽出"""
    from sklearn.manifold import TSNE
    
    # 特徴ベクトルを作成（位相角と再交差時間を組み合わせる）
    features = np.column_stack([phi_y, phi_z, np.log10(recross_times)])
    
    # t-SNEで2次元に圧縮
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=np.log10(recross_times), cmap='cividis', 
                s=30, alpha=0.8, edgecolors='w', linewidths=0.2)
    plt.colorbar(label='Log10(Recrossing Time)')
    plt.title('t-SNE Visualization of NHIM Structure')
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'nhim_tsne_structure.pdf'))
    plt.close()


def extract_clusters_on_nhim(phi_y, phi_z, recross_times):
    """再交差時間に基づくクラスタリングで構造を抽出"""
    from sklearn.cluster import KMeans, DBSCAN
    
    # 特徴量の準備（位相と再交差時間）
    features = np.column_stack([phi_y, phi_z, np.log10(recross_times)])
    
    # K-means クラスタリング
    n_clusters = 5  # クラスタ数は調整可能
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # クラスタごとに可視化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(phi_y, phi_z, c=clusters, cmap='tab10', 
                         s=30, alpha=0.8, edgecolors='w', linewidths=0.2)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Initial Phase φy / (2π)')
    plt.ylabel('Initial Phase φz / (2π)')
    plt.title('Clusters in NHIM Structure Based on Recrossing Behavior')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'nhim_clusters.pdf'))
    plt.close()
    
    # 各クラスタの特性分析
    for i in range(n_clusters):
        mask = (clusters == i)
        print(f"Cluster {i}: {np.sum(mask)} points")
        print(f"  Mean recrossing time: {10**np.mean(np.log10(recross_times[mask])):.2e}")
        print(f"  Min/Max recrossing time: {np.min(recross_times[mask]):.2e}/{np.max(recross_times[mask]):.2e}")
        
        
def analyze_time_slices(phi_y, phi_z, recross_times):
    """再交差時間のスライスごとに構造を可視化"""
    # 対数スケールでスライスを作成
    log_times = np.log10(recross_times)
    min_log, max_log = np.min(log_times), np.max(log_times)
    n_slices = 5
    slice_edges = np.linspace(min_log, max_log, n_slices+1)
    
    plt.figure(figsize=(15, 12))
    for i in range(n_slices):
        plt.subplot(2, 3, i+1)
        mask = (log_times >= slice_edges[i]) & (log_times < slice_edges[i+1])
        
        if np.sum(mask) > 0:
            # 各スライス内のデータを可視化
            plt.scatter(phi_y[mask], phi_z[mask], 
                        c='darkblue', s=15, alpha=0.7, edgecolors='w', linewidths=0.1)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, linestyle=':', alpha=0.4)
            plt.title(f'Time Slice: 10^{slice_edges[i]:.1f} - 10^{slice_edges[i+1]:.1f}')
            plt.xlabel('φy / (2π)') if i >= 3 else None
            plt.ylabel('φz / (2π)') if i % 3 == 0 else None
            plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'nhim_time_slices.pdf'))
    plt.close()


def visualize_density_and_gradients(phi_y, phi_z, recross_times):
    """カーネル密度推定と勾配を用いた詳細構造の可視化"""
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_gradient_magnitude
    
    # カーネル密度推定で滑らかな分布を作成
    xy = np.vstack([phi_y, phi_z])
    kde = gaussian_kde(xy, bw_method=0.05)  # バンド幅は調整可能
    
    # グリッド上で密度を評価
    grid_size = 100
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    density = kde(positions).reshape(grid_size, grid_size)
    
    # 勾配（密度変化が急な場所）を計算
    gradient_mag = gaussian_gradient_magnitude(density, sigma=1)
    
    # 可視化 (1): 密度マップ
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(X, Y, np.log10(density + 1e-10), cmap='cividis', shading='auto')
    plt.colorbar(label='Log10(Density)')
    plt.contour(X, Y, density, colors='white', alpha=0.3, levels=10, linewidths=0.5)
    plt.xlabel('Initial Phase φy / (2π)')
    plt.ylabel('Initial Phase φz / (2π)')
    plt.title('Kernel Density Estimation of Recrossing Trajectories')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'nhim_kde_density.pdf'))
    plt.close()
    
    # 可視化 (2): 勾配（構造の境界）
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(X, Y, gradient_mag, cmap='plasma', shading='auto')
    plt.colorbar(label='Density Gradient Magnitude')
    plt.contour(X, Y, gradient_mag, colors='white', alpha=0.3, levels=5, linewidths=0.5)
    plt.xlabel('Initial Phase φy / (2π)')
    plt.ylabel('Initial Phase φz / (2π)')
    plt.title('Structural Boundaries in NHIM (Density Gradients)')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'nhim_structure_boundaries.pdf'))
    plt.close()
    

def create_ensemble_visualization(phi_y, phi_z, recross_times):
    """複数の可視化手法を組み合わせた高度な可視化"""
    # デュアルプロット: 散布図と等高線の組み合わせ
    from scipy.interpolate import griddata
    
    # グリッドデータの作成（補間）
    grid_size = 100
    xi = np.linspace(0, 1, grid_size)
    yi = np.linspace(0, 1, grid_size)
    zi = griddata((phi_y, phi_z), np.log10(recross_times), (xi[None,:], yi[:,None]), method='cubic')
    
    # シャープな構造を強調するためのフィルタリング
    # (オプション) シャープネスフィルター
    from scipy.ndimage import gaussian_filter
    zi_smooth = gaussian_filter(zi, sigma=1)
    zi_sharp = zi + 2.0 * (zi - zi_smooth)
    
    plt.figure(figsize=(14, 12))
    
    # 左: 補間された連続的なカラーマップ
    plt.subplot(121, aspect='equal')
    plt.pcolormesh(xi, yi, zi_sharp, cmap='cividis', shading='auto')
    plt.colorbar(label='Log10(Recrossing Time)')
    plt.contour(xi, yi, zi_sharp, colors='white', alpha=0.3, levels=10, linewidths=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Enhanced Structure Visualization\n(Sharpened Interpolation)')
    plt.xlabel('Initial Phase φy / (2π)')
    plt.ylabel('Initial Phase φz / (2π)')
    plt.grid(False)
    
    # 右: 生データとエッジ強調の組み合わせ
    plt.subplot(122, aspect='equal')
    # エッジ検出のための勾配計算
    gradient_y, gradient_x = np.gradient(zi_sharp)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # 散布図とエッジを重ねる
    plt.scatter(phi_y, phi_z, c=np.log10(recross_times), 
               cmap='cividis', s=15, alpha=0.6, edgecolors='w', linewidths=0.1)
    plt.contour(xi, yi, gradient_mag, colors='red', alpha=0.5, levels=5, linewidths=1.0)
    plt.colorbar(label='Log10(Recrossing Time)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Structural Boundaries Detection\n(Raw Data + Edge Detection)')
    plt.xlabel('Initial Phase φy / (2π)')
    plt.ylabel('Initial Phase φz / (2π)')
    plt.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'nhim_enhanced_structure.pdf'))
    plt.close()


# --- メイン実行ブロック ---
if __name__ == "__main__":
    start_time_main = time.time()

    # --- 1. 初期条件生成 ---
    initial_states, initial_params = generate_initial_conditions(
        n_traj, E, gamma, px_fixed=px_init_val if use_fixed_px else None
    )
    actual_n_traj = len(initial_states) # 実際に生成できた数

    if actual_n_traj == 0:
        print("初期条件を生成できませんでした。パラメータを確認してください。")
        exit()

    # --- 2. 並列シミュレーション ---
    n_cores = mp.cpu_count()
    print(f"\nStarting {actual_n_traj} trajectory simulations using {n_cores} cores...")
    print(f"Estimated time: {T_max*actual_n_traj/(n_cores*1000):.1f}s (based on trajectory length)")
    print(f"Progress will be displayed by joblib...")

    # 進捗表示を改善するため、ベルベットを使用
    simulation_args = [(i, initial_states[i], T_max, dt, gamma) for i in range(actual_n_traj)]

    # バッチサイズを設定して進捗管理を改善
    batch_size = min(1000, actual_n_traj // 10)
    results = []

    for batch_start in range(0, actual_n_traj, batch_size):
        batch_end = min(batch_start + batch_size, actual_n_traj)
        batch_args = simulation_args[batch_start:batch_end]
        
        # 進捗表示
        print(f"Processing batch {batch_start//batch_size + 1}/{(actual_n_traj+batch_size-1)//batch_size}: "
              f"trajectories {batch_start+1}-{batch_end}/{actual_n_traj}")
        
        # このバッチの並列計算
        batch_results = Parallel(n_jobs=n_cores, verbose=10)(
            delayed(simulate_single_wrapper)(args) for args in batch_args
        )
        
        results.extend(batch_results)
        
        # バッチ完了報告
        recrossed_in_batch = sum(1 for res in batch_results if res[1] < T_max)
        print(f"Batch complete: {recrossed_in_batch}/{len(batch_results)} trajectories recrossed")

    # --- 3. 結果集計 ---
    # 結果をソート (インデックス順に戻す)
    results.sort(key=lambda x: x[0])
    recross_times = np.array([res[1] for res in results])

    # 再交差した軌道と、しなかった軌道を区別
    recrossed_mask = recross_times < T_max
    n_recrossed = np.sum(recrossed_mask)
    recrossed_times_only = recross_times[recrossed_mask]

    print(f"\nSimulation finished. Calculation time: {time.time() - start_time_main:.2f} seconds")
    print(f"Number of trajectories simulated: {actual_n_traj}")
    print(f"Number of recrossing trajectories: {n_recrossed} ({n_recrossed/actual_n_traj*100:.1f}%)")

    if n_recrossed == 0:
        print("再交差した軌道がありませんでした。プロットをスキップします。")
        exit()

    # --- 4. 生存確率プロット ---
    print("Plotting survival probability...")
    # 時間軸 (対数スケール)
    min_time_plot = max(dt * 5, np.min(recrossed_times_only) * 0.5) if n_recrossed > 0 else dt
    max_time_plot = np.max(recrossed_times_only) if n_recrossed > 0 else T_max
    if max_time_plot <= min_time_plot: max_time_plot = min_time_plot * 10 # 念のため

    t_plot = np.logspace(np.log10(min_time_plot), np.log10(max_time_plot), 100)

    # 各時刻 t で再交差していない割合 (全軌道に対する割合)
    survival_fraction = np.array([np.sum(recross_times > t) / actual_n_traj for t in t_plot])

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # 片対数プロット (時間軸対数)
    axs[0].semilogx(t_plot, survival_fraction, marker='.', linestyle='-')
    axs[0].set_xlabel("Time (log scale)")
    axs[0].set_ylabel("Survival Fraction P(t)")
    axs[0].set_title("Survival Probability (Semi-log)")
    axs[0].grid(True, which="both", ls="--", alpha=0.6)
    axs[0].set_ylim(0, 1.05)

    # 両対数プロット
    axs[1].loglog(t_plot, survival_fraction, marker='.', linestyle='-')
    axs[1].set_xlabel("Time (log scale)")
    axs[1].set_ylabel("Survival Fraction P(t) (log scale)")
    axs[1].set_title("Survival Probability (Log-log)")
    axs[1].grid(True, which="both", ls="--", alpha=0.6)
    # y軸下限を調整
    min_survival_plot = max(1e-5, 0.5 / actual_n_traj)
    axs[1].set_ylim(bottom=min_survival_plot)


    plt.suptitle(f"3DoF Recrossing Survival (E={E:.3f}, gamma={gamma}, N={actual_n_traj})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # タイトルとの重なり調整
    filename_survival = f"survival_3dof_E{E:.3f}_gamma{gamma}.pdf"
    plt.savefig(os.path.join(data_folder, filename_survival))
    plt.close(fig)
    print(f"  Survival plot saved to '{os.path.join(data_folder, filename_survival)}'")

    # --- 5. NHIM上の分布可視化 (初期位相 vs 再交差時間) ---
    if n_recrossed > 0:
        print("Plotting NHIM initial phase vs Recrossing time...")
        # 再交差した軌道の初期パラメータを取得
        recrossed_params = [initial_params[i] for i in np.where(recrossed_mask)[0]]

        phi_y_recrossed = np.array([p['phi_y'] for p in recrossed_params]) / (2 * np.pi) # 正規化
        phi_z_recrossed = np.array([p['phi_z'] for p in recrossed_params]) / (2 * np.pi) # 正規化
        times_log_recrossed = np.log10(recrossed_times_only)

        # 色覚多様性に配慮したプロットを複数作成
        for cmap_name, cmap_label in [
            ('cividis', 'CVD-friendly'),  # 色覚多様性に特化したカラーマップ
            ('plasma', 'Blue-Yellow'),     # 青-黄色のグラデーション
            ('inferno', 'Dark-Yellow'),    # 暗い色から黄色へのグラデーション
        ]:
            plt.figure(figsize=(10, 8))
            
            # 散布図のプロット - マーカーサイズを大きくし、エッジを追加
            scatter = plt.scatter(
                phi_y_recrossed, phi_z_recrossed,
                c=times_log_recrossed,
                cmap=cmap_name,
                s=25,  # マーカーサイズを大きく
                alpha=0.8,
                edgecolors='w',  # 白い輪郭線を追加
                linewidths=0.2   # 輪郭線を薄く
            )

            plt.xlabel("Initial Phase φy / (2π)", fontsize=12)
            plt.ylabel("Initial Phase φz / (2π)", fontsize=12)
            plt.title(f"Initial Phase on NHIM vs. Recrossing Time ({cmap_label})\nE={E:.3f}, γ={gamma}, N={n_recrossed}", 
                      fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を1:1に

            # カラーバーをわかりやすく
            cbar = plt.colorbar(scatter, extend='both')
            cbar.set_label("Log10(Recrossing Time)", fontsize=12)
            
            # ティックラベルを見やすく
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            cbar.ax.tick_params(labelsize=10)

            filename_dist = f"nhim_phase_dist_3dof_E{E:.3f}_gamma{gamma}_{cmap_name}.pdf"
            plt.savefig(os.path.join(data_folder, filename_dist))
            plt.close()
            print(f"  NHIM phase distribution plot saved to '{os.path.join(data_folder, filename_dist)}'")
        
        # 分布の特徴をより詳細に分析するための2Dヒストグラム
        plt.figure(figsize=(10, 8))
        
        # 2Dヒストグラムで密度を可視化
        hist, xedges, yedges = np.histogram2d(
            phi_y_recrossed, phi_z_recrossed, 
            bins=20, 
            range=[[0, 1], [0, 1]]
        )
        
        # 対数スケールでプロット
        plt.pcolormesh(xedges, yedges, hist.T, 
                      norm=LogNorm(vmin=1, vmax=hist.max()),
                      cmap='cividis')
        
        plt.xlabel("Initial Phase φy / (2π)", fontsize=12)
        plt.ylabel("Initial Phase φz / (2π)", fontsize=12)
        plt.title(f"Density of Recrossing Trajectories on NHIM\nE={E:.3f}, γ={gamma}, N={n_recrossed}", 
                  fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.colorbar(label="Count (log scale)")
        plt.gca().set_aspect('equal', adjustable='box')
        
        filename_hist = f"nhim_phase_density_3dof_E{E:.3f}_gamma{gamma}.pdf"
        plt.savefig(os.path.join(data_folder, filename_hist))
        plt.close()
        print(f"  NHIM phase density plot saved to '{os.path.join(data_folder, filename_hist)}'")

    # --- 6. 微妙な構造の抽出と可視化 ---
    if n_recrossed > 100:  # 十分なデータがある場合のみ実行
        print("Extracting and visualizing subtle structures in NHIM...")
        
        # t-SNEによる非線形次元削減
        try:
            extract_nhim_structure_with_tsne(phi_y_recrossed, phi_z_recrossed, recrossed_times_only)
            print("  t-SNE visualization completed")
        except ImportError:
            print("  Warning: sklearn not available, skipping t-SNE visualization")
        
        # クラスタリング分析
        try:
            extract_clusters_on_nhim(phi_y_recrossed, phi_z_recrossed, recrossed_times_only)
            print("  Clustering analysis completed")
        except ImportError:
            print("  Warning: sklearn not available, skipping clustering analysis")
        
        # 時間スライスによる分析
        analyze_time_slices(phi_y_recrossed, phi_z_recrossed, recrossed_times_only)
        print("  Time slice analysis completed")
        
        # 密度と勾配の可視化
        visualize_density_and_gradients(phi_y_recrossed, phi_z_recrossed, recrossed_times_only)
        print("  Density and gradient visualization completed")
        
        # 高度なアンサンブル可視化
        create_ensemble_visualization(phi_y_recrossed, phi_z_recrossed, recrossed_times_only)
        print("  Enhanced structure visualization completed")
        
        print("All structure extraction and visualization completed")

    print("\nAll processing finished.")
