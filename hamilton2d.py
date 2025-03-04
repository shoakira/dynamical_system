import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pywt
import os
from datetime import datetime
import pathlib

# --- best_transitions.txtから初期パラメータを読み込む ---
def load_transition_params_from_file(filepath='best_transitions.txt'):
    """best_transitions.txtから遷移軌道のパラメータを読み込む"""
    print(f"ファイル '{filepath}' からパラメータを読み込み中...")
    
    params = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            if lines[i].startswith('# 遷移軌道'):
                if i + 2 < len(lines):
                    try:
                        eps_line = lines[i+1].strip()
                        ic_line = lines[i+2].strip()
                        
                        eps = float(eps_line.split('=')[1].strip())
                        ic_str = ic_line.split('=')[1].strip()
                        ic = eval(ic_str)  # 安全な環境で使用する前提
                        
                        # DTの読み込み（存在する場合）
                        dt_value = 0.05  # デフォルト値
                        if i + 3 < len(lines) and lines[i+3].strip().startswith('DT ='):
                            dt_line = lines[i+3].strip()
                            dt_value = float(dt_line.split('=')[1].strip())
                        
                        params.append({
                            'epsilon': eps,
                            'initial_condition': ic,
                            'trajectory_id': int(lines[i].split()[2]),
                            'dt': dt_value  # DTを追加
                        })
                        
                        print(f"Parameter loaded: Trajectory ID {params[-1]['trajectory_id']}, "
                              f"EPSILON={eps}, Initial Condition={ic}, DT={dt_value}")
                    except Exception as e:
                        print(f"Parameter parsing error: {e}")
                
                # 次の軌道データセットへ
                i += 4  # DTも含めて4行に
            else:
                i += 1
        
        if not params:
            print("No valid parameters found. Using default values.")
            params.append({
                'epsilon': 0.9,
                'initial_condition': [1.0, 0.01, 0.1, 0.05],
                'trajectory_id': 0,
                'dt': 0.05  # デフォルトDT
            })
    
    except FileNotFoundError:
        print(f"File '{filepath}' not found. Using default values.")
        params.append({
            'epsilon': 0.9,
            'initial_condition': [1.0, 0.01, 0.1, 0.05],
            'trajectory_id': 0,
            'dt': 0.05  # デフォルトDT
        })
    
    return params

# --- シンプレクティック積分器 ---
def symplectic_integrator(hamiltonian_force, t_span, y0, dt):
    """Symplectic integration method (Störmer-Verlet)"""
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt/2, dt)
    steps = len(t)
    
    y = np.zeros((4, steps))
    y[:, 0] = y0
    
    q = np.array([y0[0], y0[1]])
    p = np.array([y0[2], y0[3]])
    
    for i in range(1, steps):
        # Step 1: Half-step momentum update
        forces = hamiltonian_force(q)
        p = p - dt/2 * forces
        
        # Step 2: Full-step position update
        q = q + dt * p
        
        # Step 3: Half-step momentum update
        forces = hamiltonian_force(q)
        p = p - dt/2 * forces
        
        y[0, i] = q[0]
        y[1, i] = q[1]
        y[2, i] = p[0]
        y[3, i] = p[1]
    
    return t, y

# --- アクション変数の計算 ---
def calculate_actions(q1, p1, q2, p2, eps):
    """
    Calculate action variables I1, I2
    
    In this system, actions are approximated using energy values:
    I1: Action variable for first mode ≈ E1 = 0.5 * (p1^2 + q1^2)
    I2: Action variable for second mode ≈ E2 = 0.5 * (p2^2 + 4*q2^2)
    """
    # Energy for each mode (action approximation)
    I1 = 0.5 * (p1**2 + q1**2)
    I2 = 0.5 * (p2**2 + 4*q2**2)
    
    return I1, I2

# --- CWT計算関数 ---
def calculate_cwt(signal, dt, max_freq=5.0):
    """CWT calculation function (no plotting)"""
    # Enhanced frequency resolution
    scales = np.logspace(0, 8, 1500, base=2)  
    wavelet = 'cmor1.5-1.0'
    
    frequencies = pywt.scale2frequency(wavelet, scales) / dt
    
    mask = frequencies <= max_freq
    scales_filtered = scales[mask]
    frequencies_filtered = frequencies[mask]
    
    coefficients, _ = pywt.cwt(signal, scales_filtered, wavelet, dt)
    
    return coefficients, frequencies_filtered

# --- 統合プロット関数 ---
def plot_trajectory_analysis(t, q1, q2, p1, p2, eps, traj_id, DT, T_MAX, save_path=None):
    """Integrated display of trajectory, action space, and CWT, with option to save as PDF"""
    max_freq = 0.5  # CWTの最大周波数
    
    # CWT計算
    q1_coeffs, q1_freqs = calculate_cwt(q1, DT, max_freq)
    q2_coeffs, q2_freqs = calculate_cwt(q2, DT, max_freq)
    
    # アクション変数の計算
    I1, I2 = calculate_actions(q1, p1, q2, p2, eps)
    
    # カラーマップ作成
    jet_modified = LinearSegmentedColormap.from_list(
        'jet_modified', plt.cm.jet(np.linspace(0, 1, 1000)), N=1000)
    
    # プロット作成 - 2x3レイアウト
    fig = plt.figure(figsize=(20, 12))
    
    # 1. アクション空間での軌道（時間10ステップごと）
    ax1 = plt.subplot(2, 3, 1)
    
    # 時間による色分け
    n_points = len(t)
    step_size = 10  # 10ステップごと
    indices = np.arange(0, n_points, step_size)
    
    if len(indices) > 0:
        # 時間による色分け - 点のサイズを小さくし、線を削除
        norm_times = (t[indices] - t[0]) / (t[-1] - t[0])
        scatter = ax1.scatter(I1[indices], I2[indices], c=norm_times, cmap='viridis', 
                        s=2, alpha=0.7)  # s=5から2に縮小
        plt.colorbar(scatter, ax=ax1, label='Normalized Time')
        
    ax1.set_title('Action Space (I1-I2) Every 10 Steps')
    ax1.set_xlabel('I1 ≈ E1')
    ax1.set_ylabel('I2 ≈ E2')
    ax1.grid(True, alpha=0.3)
    
    # 2. I1の時系列 (q1から変更)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t, I1)
    ax2.set_title('I1 vs Time')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('I1')
    ax2.grid(True)
    
    # 3. I2の時系列 (q2から変更)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t, I2)
    ax3.set_title('I2 vs Time')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('I2')
    ax3.grid(True)
    
    # 4. q1のCWT
    ax4 = plt.subplot(2, 3, 5)
    img1 = ax4.imshow(
        np.abs(q1_coeffs),
        extent=[t[0], t[-1], q1_freqs[0], q1_freqs[-1]],
        aspect='auto',
        cmap=jet_modified,
        interpolation='bilinear',
        origin='lower'
    )
    ax4.set_title('Wavelet Analysis for q1')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Frequency [Hz]')
    fig.colorbar(img1, ax=ax4, label='Magnitude')
    
    # 5. q2のCWT
    ax5 = plt.subplot(2, 3, 6)
    img2 = ax5.imshow(
        np.abs(q2_coeffs),
        extent=[t[0], t[-1], q2_freqs[0], q2_freqs[-1]],
        aspect='auto',
        cmap=jet_modified,
        interpolation='bilinear',
        origin='lower'
    )
    ax5.set_title('Wavelet Analysis for q2')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Frequency [Hz]')
    fig.colorbar(img2, ax=ax5, label='Magnitude')
    
    # 6. 位相空間 (q1-p1)
    ax6 = plt.subplot(2, 3, 4)
    ax6.plot(q1, p1, linewidth=0.8)
    ax6.set_title('Phase Space (q1, p1)')
    ax6.set_xlabel('q1')
    ax6.set_ylabel('p1')
    ax6.grid(True)
    
    plt.suptitle(f'Integrated Analysis for Trajectory {traj_id}: EPSILON = {eps}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 結果をPDFに保存またはプロット表示
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)  # メモリリークを防ぐためにfigureを閉じる
        print(f"Analysis saved to {save_path}")
    else:
        plt.show()

# --- メイン関数 - 軌道生成とCWT解析 ---
def run_cwt_analysis():
    """best_transitions.txtから読み込んだパラメータで軌道を生成し、CWT解析を実行してPDFに保存"""
    # パラメータ読み込み
    transition_params = load_transition_params_from_file()
    
    # 出力ディレクトリの作成
    output_dir = "orbit_analysis_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to directory: {output_dir}")
    
    # シミュレーションパラメータ（デフォルト値）
    T_MAX = 3000.0  # シミュレーション時間
    
    # 各パラメータでシミュレーション実行
    for param in transition_params:
        eps = param['epsilon']
        y0 = param['initial_condition']
        traj_id = param.get('trajectory_id', 0)
        DT = param.get('dt', 0.05)  # パラメータからDTを取得、なければデフォルト値
        
        print(f"\n=== Simulating and analyzing trajectory {traj_id} (EPSILON = {eps}, DT = {DT}) ===")
        print(f"Initial condition: {y0}")
        
        # ハミルトン系の運動方程式定義
        def hamiltonian_force(q):
            q1, q2 = q
            force1 = q1 + 2 * eps * q1 * q2
            force2 = 4 * q2 + eps * q1**2
            return np.array([force1, force2])
        
        # シミュレーション実行
        t, y = symplectic_integrator(hamiltonian_force, [0, T_MAX], y0, DT)
        q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
        
        # エネルギー計算（エネルギー保存の確認用）
        energy = 0.5 * (p1**2 + p2**2 + q1**2 + 4*q2**2) + eps * q1**2 * q2
        energy_drift = np.max(np.abs((energy - energy[0]) / energy[0]))
        print(f"Energy conservation accuracy: {energy_drift*100:.8f}%")
        
        # PDF保存用のファイル名生成
        pdf_filename = os.path.join(output_dir, f"trajectory_{traj_id}_eps_{eps:.3f}.pdf")
        
        # CWT解析の実行と結果の保存
        print(f"Running wavelet analysis and saving to {pdf_filename}...")
        plot_trajectory_analysis(t, q1, q2, p1, p2, eps, traj_id, DT, T_MAX, save_path=pdf_filename)
    
    print(f"\nAnalysis complete. All results saved to directory: {output_dir}")

# メイン実行
if __name__ == "__main__":
    print("Loading parameters from best_transitions.txt, generating trajectories and saving analysis to PDF...")
    run_cwt_analysis()
