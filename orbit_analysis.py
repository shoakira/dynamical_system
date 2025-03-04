import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pywt
import time
import os

# --- シンプレクティック積分器 (Störmer-Verlet法) ---
def symplectic_integrator(hamiltonian_force, t_span, y0, dt):
    """シンプレクティック積分法（ストーマー・ヴェルレ法）"""
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt/2, dt)
    steps = len(t)
    
    y = np.zeros((4, steps))
    y[:, 0] = y0
    
    q = np.array([y0[0], y0[1]])
    p = np.array([y0[2], y0[3]])
    
    for i in range(1, steps):
        # ステップ1: 運動量の半ステップ更新
        forces = hamiltonian_force(q)
        p = p - dt/2 * forces
        
        # ステップ2: 位置の1ステップ更新
        q = q + dt * p
        
        # ステップ3: 運動量の残り半ステップ更新
        forces = hamiltonian_force(q)
        p = p - dt/2 * forces
        
        y[0, i] = q[0]
        y[1, i] = q[1]
        y[2, i] = p[0]
        y[3, i] = p[1]
    
    return t, y

# --- ポアンカレ断面図を作成する関数 ---
def create_poincare_section(q1, p1, q2, p2, threshold=0.0, direction='positive'):
    """
    q2 = threshold の断面を通過する点での (q1, p1) 座標を取得
    direction: 'positive' → p2 > 0 のみ, 'both' → 両方向
    """
    points_q1 = []
    points_p1 = []
    points_t_idx = []  # 時間インデックスも記録
    
    for i in range(1, len(q2)):
        # q2がしきい値を跨いで通過する点を検出
        if (q2[i-1] - threshold) * (q2[i] - threshold) <= 0:
            # 方向の条件チェック
            if direction == 'positive' and p2[i] <= 0:
                continue
            
            # 断面との交差点での補間
            frac = abs(q2[i-1] - threshold) / abs(q2[i] - q2[i-1])
            q1_interp = q1[i-1] + frac * (q1[i] - q1[i-1])
            p1_interp = p1[i-1] + frac * (p1[i] - p1[i-1])
            
            points_q1.append(q1_interp)
            points_p1.append(p1_interp)
            points_t_idx.append(i)
    
    return np.array(points_q1), np.array(points_p1), np.array(points_t_idx)

# --- トーラスからカオスへの遷移を検出する関数 ---
def detect_torus_chaos_transition(q1, p1, q2, p2, window_size=1000, overlap=500, threshold=0.0):
    """
    Detect transitions from torus to chaos by calculating Poincare sections in time windows
    Returns: torus-like scores (lower means more chaotic) and transition flag
    """
    # 全時系列を窓に分割して分析
    n_points = len(q1)
    windows = []
    
    for start_idx in range(0, n_points - window_size, window_size - overlap):
        end_idx = start_idx + window_size
        if end_idx > n_points:
            break
            
        # 各窓でのポアンカレ断面
        win_q1 = q1[start_idx:end_idx]
        win_p1 = p1[start_idx:end_idx]
        win_q2 = q2[start_idx:end_idx]
        win_p2 = p2[start_idx:end_idx]
        
        poincare_q1, poincare_p1, _ = create_poincare_section(win_q1, win_p1, win_q2, win_p2, threshold)
        
        if len(poincare_q1) < 5:  # 十分な交差点がない場合はスキップ
            continue
            
        # トーラス性の評価: 点の分布が曲線に乗っているか
        # 単純な指標: 点の分散方向の比率（楕円度）- トーラスでは特定方向に集中
        if len(poincare_q1) >= 8:  # 閾値を下げて探索範囲を広げる
            try:
                # 共分散行列からトーラス性を評価
                points = np.vstack([poincare_q1, poincare_p1]).T
                cov = np.cov(points.T)
                eigenvals, _ = np.linalg.eig(cov)
                
                # 小さい固有値/大きい固有値の比 (0に近いほどトーラス的、1に近いほどカオス的)
                ratio = min(eigenvals) / max(eigenvals) if max(eigenvals) > 0 else 1.0
                
                windows.append({
                    'start_idx': start_idx,
                    'torus_score': 1.0 - ratio,  # 1に近いほどトーラス的
                    'points': points
                })
            except:
                pass
    
    if not windows:  # 分析できる窓がない
        return [], False
    
    # トーラス性スコアの推移
    scores = [w['torus_score'] for w in windows]
    
    # トーラスからカオスへの遷移を検出 (スコアが高い→低いへ)
    transition = False
    if len(scores) >= 3:
        for i in range(len(scores) - 2):
            # 条件を緩和: トーラス的な状態(>0.6)から、カオス的な状態(<0.5)に移行
            if scores[i] > 0.6 and scores[i+2] < 0.5:
                transition = True
                break
    
    return windows, transition

# --- カオス・トーラス遷移軌道の探索実験 ---
def search_transition_orbits(eps_range, initial_conditions_grid, T, DT, save_dir='transitions'):
    """
    トーラス→カオス遷移軌道を系統的に探索
    """
    # 結果保存用ディレクトリ
    os.makedirs(save_dir, exist_ok=True)
    
    # 結果のリスト
    found_transitions = []
    
    # 全パラメータ組み合わせ数
    total_combinations = len(eps_range) * len(initial_conditions_grid)
    
    # 進捗表示用のカウンター
    counter = 0
    start_time = time.time()
    
    for eps in eps_range:
        # ハミルトン系の運動方程式の力学項を定義
        def hamiltonian_force(q):
            q1, q2 = q
            force1 = q1 + 2 * eps * q1 * q2
            force2 = 4 * q2 + eps * q1**2
            return np.array([force1, force2])
        
        for idx, y0 in enumerate(initial_conditions_grid):
            counter += 1
            
            # 進捗表示
            if counter % 10 == 0 or counter == total_combinations:
                elapsed = time.time() - start_time
                eta = (elapsed / counter) * (total_combinations - counter)
                print(f"進捗: {counter}/{total_combinations} ({counter/total_combinations*100:.1f}%) - "
                      f"経過時間: {elapsed:.1f}秒 - 残り: {eta:.1f}秒")
            
            # シミュレーション実行
            t, y = symplectic_integrator(hamiltonian_force, [0, T], y0, DT)
            
            # 結果抽出
            q1, q2 = y[0], y[1]
            p1, p2 = y[2], y[3]
            
            # エネルギー計算と保存則のチェック
            energy = 0.5 * (p1**2 + p2**2 + q1**2 + 4*q2**2) + eps * q1**2 * q2
            energy_drift = np.max(np.abs((energy - energy[0]) / energy[0]))
            
            if energy_drift > 0.01:  # エネルギー誤差が大きい場合は警告
                print(f"警告: EPSILON={eps}, 初期条件={y0} でエネルギー誤差が大きい: {energy_drift*100:.2f}%")
                continue
            
            # トーラス→カオス遷移検出
            windows, has_transition = detect_torus_chaos_transition(q1, p1, q2, p2)
            
            if has_transition:
                print(f"遷移検出! EPSILON={eps}, 初期条件={y0}")
                
                # 結果を保存
                transition_data = {
                    'epsilon': eps,
                    'initial_condition': y0,
                    'trajectory': y,
                    'time': t,
                    'windows': windows,
                    'energy_drift': energy_drift
                }
                found_transitions.append(transition_data)
                
                # 図の保存
                fig_path = os.path.join(save_dir, f"transition_eps{eps:.3f}_ic{idx}.png")
                plot_transition_analysis(transition_data, fig_path)
    
    return found_transitions

# --- アクション変数の計算 ---
def calculate_actions(q1, p1, q2, p2, eps):
    """
    Calculate action variables I1, I2
    
    In this system, actions are approximated using energy values
    I1: Action variable for first mode ≈ E1 = 0.5 * (p1^2 + q1^2)
    I2: Action variable for second mode ≈ E2 = 0.5 * (p2^2 + 4*q2^2)
    """
    # Energy for each mode (action approximation)
    I1 = 0.5 * (p1**2 + q1**2)
    I2 = 0.5 * (p2**2 + 4*q2**2)
    
    return I1, I2

# --- 遷移軌道の詳細分析と可視化 ---
def plot_transition_analysis(transition_data, save_path=None):
    """Detailed analysis and visualization of transition orbit"""
    eps = transition_data['epsilon']
    y0 = transition_data['initial_condition']
    t = transition_data['time']
    y = transition_data['trajectory']
    windows = transition_data['windows']
    
    q1, q2 = y[0], y[1]
    p1, p2 = y[2], y[3]
    
    # アクション変数の計算
    I1, I2 = calculate_actions(q1, p1, q2, p2, eps)
    
    # ウィンドウごとのトーラス性スコア
    window_times = [t[w['start_idx']] for w in windows]
    torus_scores = [w['torus_score'] for w in windows]
    
    # 図の作成
    fig = plt.figure(figsize=(15, 12))
    
    # 1. アクション空間のプロット（点のみ、線は繋げない、10ステップごと）
    ax1 = plt.subplot(231)
    
    # 10ステップごとのポイントを表示
    step_size = 10
    indices = np.arange(0, len(t), step_size)
    
    # 時間による色分け - 点のサイズを小さく
    norm_times = (t[indices] - t[0]) / (t[-1] - t[0])
    scatter = ax1.scatter(I1[indices], I2[indices], c=norm_times, cmap='viridis', 
                         s=1.5, alpha=0.7)  # 点のサイズを小さく
    
    ax1.set_title('Action Space (I1-I2) Every 10 Steps')
    ax1.set_xlabel('I1 ≈ E1')
    ax1.set_ylabel('I2 ≈ E2')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Normalized Time')
    
    # 2. I1の時系列 (q1からI1に変更)
    ax2 = plt.subplot(232)
    ax2.plot(t, I1)
    ax2.set_title('I1 vs Time')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('I1')
    ax2.grid(True)
    
    # 3. I2の時系列 (q2からI2に変更)
    ax3 = plt.subplot(233)
    ax3.plot(t, I2)
    ax3.set_title('I2 vs Time')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('I2')
    ax3.grid(True)
    
    # 4. 位相空間 (q1-p1)
    ax4 = plt.subplot(234)
    ax4.plot(q1, p1)
    ax4.set_title('Phase Space (q1, p1)')
    ax4.set_xlabel('q1')
    ax4.set_ylabel('p1')
    ax4.grid(True)
    
    # 5. 位相空間 (q2-p2)
    ax5 = plt.subplot(235)
    ax5.plot(q2, p2)
    ax5.set_title('Phase Space (q2, p2)')
    ax5.set_xlabel('q2')
    ax5.set_ylabel('p2')
    ax5.grid(True)
    
    # 6. トーラス性スコアの時間変化
    ax6 = plt.subplot(236)
    ax6.plot(window_times, torus_scores, 'o-')
    ax6.set_title('Torus Score vs Time')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Torus Score')
    ax6.set_ylim(-0.1, 1.1)
    ax6.grid(True)
    ax6.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Torus Threshold')
    ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chaos Threshold')
    ax6.legend()
    
    plt.suptitle(f'Torus-Chaos Transition Analysis: EPSILON={eps}, Initial Condition={y0}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

# --- 新しい高速審査モード: 見込みある軌道だけを詳細分析 ---
def fast_orbit_screening(eps_range, initial_conditions_grid, T_screen, DT, num_to_analyze=10):
    """
    高速審査モード: まず短い時間で多くの軌道を分析し、有望そうな軌道のみ詳細分析
    """
    print("=== 高速スクリーニングモード ===")
    print(f"まず短時間シミュレーション(T={T_screen})で有望な軌道を選別...")
    
    # 結果記録用
    candidates = []
    
    # 全組み合わせ
    total_combs = len(eps_range) * len(initial_conditions_grid)
    counter = 0
    start_time = time.time()
    
    # パラメータの組み合わせをすべて試す
    for eps in eps_range:
        def hamiltonian_force(q):
            q1, q2 = q
            force1 = q1 + 2 * eps * q1 * q2
            force2 = 4 * q2 + eps * q1**2
            return np.array([force1, force2])
        
        for idx, y0 in enumerate(initial_conditions_grid):
            counter += 1
            if counter % 20 == 0:
                elapsed = time.time() - start_time
                print(f"スクリーニング進捗: {counter}/{total_combs} ({counter/total_combs*100:.1f}%)")
            
            # 短時間シミュレーション
            t, y = symplectic_integrator(hamiltonian_force, [0, T_screen], y0, DT)
            q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
            
            # 簡易評価: ポアンカレ断面の性質変化
            first_half_q1, first_half_p1, _ = create_poincare_section(
                q1[:len(q1)//2], p1[:len(p1)//2], 
                q2[:len(q2)//2], p2[:len(p2)//2]
            )
            
            second_half_q1, second_half_p1, _ = create_poincare_section(
                q1[len(q1)//2:], p1[len(p1)//2:], 
                q2[len(q2)//2:], p2[len(p2)//2:]
            )
            
            # スコア計算（両方のセクションで点が十分ある場合）
            if len(first_half_q1) >= 10 and len(second_half_q1) >= 10:
                try:
                    # 前半部分の評価
                    first_points = np.vstack([first_half_q1, first_half_p1]).T
                    first_cov = np.cov(first_points.T)
                    first_evals, _ = np.linalg.eig(first_cov)
                    first_ratio = min(first_evals) / max(first_evals) if max(first_evals) > 0 else 1.0
                    first_score = 1.0 - first_ratio
                    
                    # 後半部分の評価
                    second_points = np.vstack([second_half_q1, second_half_p1]).T
                    second_cov = np.cov(second_points.T)
                    second_evals, _ = np.linalg.eig(second_cov)
                    second_ratio = min(second_evals) / max(second_evals) if max(second_evals) > 0 else 1.0
                    second_score = 1.0 - second_ratio
                    
                    # スコアの差（大きいほど変化大）
                    score_diff = abs(first_score - second_score)
                    
                    # 候補として記録
                    candidates.append({
                        'eps': eps,
                        'y0': y0,
                        'score_diff': score_diff,
                        'first_score': first_score,
                        'second_score': second_score
                    })
                except:
                    pass
    
    # スコア差で降順ソート（変化が大きい順）
    candidates.sort(key=lambda x: x['score_diff'], reverse=True)
    
    # スコア変化のパターンでフィルタリング（トーラス→カオス遷移に合致するもの）
    filtered_candidates = [
        c for c in candidates 
        if (c['first_score'] - c['second_score']) > 0.2  # トーラス→カオス
        and c['first_score'] > 0.5                       # 最初はある程度トーラス的
        and c['second_score'] < 0.5                      # 後にはある程度カオス的
    ]
    
    print(f"スクリーニング完了: {len(filtered_candidates)}個の有望候補を特定")
    
    # 上位N個の候補のみを返す
    return filtered_candidates[:num_to_analyze]

# --- メイン実行コード ---
def main():
    # より広範囲なパラメータ探索
    eps_range = np.linspace(0.5, 1.0, 11)  # 0.5から1.0までより広く探索
    
    # 初期条件の多様化
    q1_range = np.linspace(0.7, 1.3, 7)
    q2_range = np.linspace(0.01, 0.2, 5)
    p1_range = np.linspace(-0.1, 0.2, 4)
    p2_range = np.linspace(-0.1, 0.1, 3)
    
    # 既知の有望な初期条件（広げた）
    seed_conditions = [
        [1.0, 0.05, 0.0, 0.0],     # 標準的な初期条件
        [0.9, 0.1, 0.05, 0.0],     # より広い初期分布
        [1.0, 0.01, 0.1, 0.05],    # 細かい変動
        [1.0, 0.01, -0.1, 0.05],   # p1に負の値
        [0.7, 0.15, 0.1, -0.05],   # q1をより小さく
        [1.2, 0.08, 0.15, 0.0],    # q1をより大きく
        [1.0, 0.2, 0.0, 0.0],      # q2をより大きく
        [0.9, 0.15, 0.15, -0.05],  # 混合パターン1
        [1.1, 0.12, -0.05, 0.08],  # 混合パターン2
    ]
    
    initial_conditions = []
    initial_conditions.extend(seed_conditions)
    
    # グリッド探索用の初期条件をサンプリング（全グリッドは計算量が大きすぎるため）
    for q1 in q1_range:
        for q2 in q2_range[::2]:  # 間引く
            for p1 in p1_range[::2]:  # 間引く
                for p2 in p2_range[::2]:  # 間引く
                    if np.random.random() < 0.3:  # 30%の確率でサンプル
                        initial_conditions.append([q1, q2, p1, p2])
    
    # シミュレーション設定
    T_SCREEN = 200.0  # 高速スクリーニング用の短い時間
    T_FULL = 3000.0   # 詳細分析用の長い時間（2倍に延長）
    DT = 0.03
    
    print(f"探索する初期条件の数: {len(initial_conditions)}")
    print(f"探索するEPSILON値の数: {len(eps_range)}")
    print(f"合計組み合わせ数: {len(initial_conditions)*len(eps_range)}")
    
    # ======= 新しいアプローチ: 2段階探索 =======
    # ステップ1: 高速スクリーニング
    candidates = fast_orbit_screening(
        eps_range, initial_conditions, T_SCREEN, DT, num_to_analyze=15
    )
    
    if not candidates:
        print("スクリーニングで有望な候補が見つかりませんでした。")
        return
    
    print("\n=== 有望候補の詳細分析 ===")
    
    # ステップ2: 有望な軌道のみ長時間シミュレーション
    transitions = []
    
    for idx, candidate in enumerate(candidates):
        eps = candidate['eps']
        y0 = candidate['y0']
        
        print(f"候補 {idx+1}/{len(candidates)}: EPSILON={eps}, 初期条件={y0} の詳細分析...")
        
        def hamiltonian_force(q):
            q1, q2 = q
            force1 = q1 + 2 * eps * q1 * q2
            force2 = 4 * q2 + eps * q1**2
            return np.array([force1, force2])
        
        # 長時間シミュレーション
        t, y = symplectic_integrator(hamiltonian_force, [0, T_FULL], y0, DT)
        q1, q2 = y[0], y[1]
        p1, p2 = y[2], y[3]
        
        # エネルギー計算
        energy = 0.5 * (p1**2 + p2**2 + q1**2 + 4*q2**2) + eps * q1**2 * q2
        energy_drift = np.max(np.abs((energy - energy[0]) / energy[0]))
        
        if energy_drift > 0.01:
            print(f"  警告: エネルギー誤差が大きい: {energy_drift*100:.2f}%")
            continue
        
        # トーラス→カオス遷移の詳細検出
        windows, has_transition = detect_torus_chaos_transition(q1, p1, q2, p2, 
                                                             window_size=1000, 
                                                             overlap=800)
        
        if has_transition:
            print(f"  ✓ 遷移検出成功!")
            
            transition_data = {
                'epsilon': eps,
                'initial_condition': y0,
                'trajectory': y,
                'time': t,
                'windows': windows,
                'energy_drift': energy_drift
            }
            transitions.append(transition_data)
            
            # 解析結果を表示
            plot_transition_analysis(transition_data)
        else:
            print(f"  × 詳細分析の結果、遷移なしと判定")
            
            # 参考のため、ポアンカレ図だけ表示
            plt.figure(figsize=(10, 8))
            poincare_q1, poincare_p1, time_indices = create_poincare_section(q1, p1, q2, p2)
            
            norm_times = (t[time_indices] - t[0]) / (t[-1] - t[0])
            scatter = plt.scatter(poincare_q1, poincare_p1, c=norm_times, cmap='viridis', 
                             s=3, alpha=0.7)
            plt.colorbar(scatter, label='Normalized Time')
            plt.title(f'ポアンカレ断面 - EPSILON={eps}, 初期条件={y0} (遷移なし)')
            plt.xlabel('q1')
            plt.ylabel('p1')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    if transitions:
        print(f"\n合計{len(transitions)}個の遷移軌道を発見!")
        
        # 結果をファイルに保存
        import pickle
        with open('transition_results.pkl', 'wb') as f:
            # DTも含めたデータを保存
            for tr in transitions:
                if 'dt' not in tr:
                    tr['dt'] = DT  # 現在のDT値を追加
            pickle.dump(transitions, f)
        
        # 簡易テキスト出力（hamilton2dで使いやすいように）
        with open('best_transitions.txt', 'w') as f:
            for i, tr in enumerate(transitions):
                f.write(f"# 遷移軌道 {i+1}\n")
                f.write(f"EPSILON = {tr['epsilon']}\n")
                f.write(f"INITIAL_CONDITION = {tr['initial_condition']}\n")
                # DTも追加で書き込む
                f.write(f"DT = {tr.get('dt', DT)}\n\n")  # dt情報がない場合は現在のDTを使用
    else:
        print("\n遷移軌道は見つかりませんでした。")
        print("手動解析のための提案:")
        print("1. EPSILON値をさらに高く（0.7～1.5）設定して試す")
        print("2. 初期条件を q1=1.0, q2=0.1付近でより細かく探索する")
        print("3. 遷移検出の閾値（スコア）をさらに緩和する")

if __name__ == "__main__":
    main()