import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pywt

# ----- パラメータ設定 -----
w1 = 1.0           # 第1振動子の固有振動数 (rad/s)
w2 = 5.0           # 第2振動子の固有振動数を 5 rad/s に設定
epsilon = 0.0      # 非線形結合を無効化
T = 200.0          # シミュレーション総時間
dt = 0.01
t_eval = np.arange(0, T+dt, dt)

# ----- ハミルトン系の運動方程式 -----
def hamiltonian_odes(t, y):
    # y = [q1, q2, p1, p2]
    q1, q2, p1, p2 = y
    dq1 = p1
    dq2 = p2
    dp1 = - q1              # 調和振動: q1'' + q1 = 0  → 周波数 = 1 rad/s
    dp2 = - (w2**2) * q2    # 調和振動: q2'' + (5^2)q2 = 0 → 周波数 = 5 rad/s
    return [dq1, dq2, dp1, dp2]

# ----- 初期条件 -----
# 両振動子を励起
# q1(0)=1.0, q2(0)=1.0 かつ p1(0)=p2(0)=0 なら
# q1(t)=cos(t) (周波数1) と q2(t)=cos(5t) (周波数5) が得られる
y0 = [1.0, 1.0, 0, 0]

# ----- 数値積分 -----
sol = solve_ivp(hamiltonian_odes, [0, T], y0, t_eval=t_eval,
                method='RK45', rtol=1e-9, atol=1e-9)
q1 = sol.y[0]
q2 = sol.y[1]
p1 = sol.y[2]
p2 = sol.y[3]

# ----- 合成信号の定義 -----
# ２つの異なる周波数（1 rad/s と 5 rad/s）が混在した信号とする
signal = q1 + 0.5 * q2



# ----- ステップ2: 合成信号に対するウェーブレット変換とスケールグラムのプロット -----
# scale の範囲を 0.1 から 128 にして高周波部もカバー
scales = np.linspace(0.1, 128, 500)
wavelet = 'cmor'  # 複素モレーレ wavelet（デフォルトパラメータ）

# dt を用いて各 scale に対応する中心周波数を計算（Hz 単位）
frequencies = pywt.scale2frequency(wavelet, scales) / dt
coefficients, _ = pywt.cwt(signal, scales, wavelet, dt)

plt.figure(figsize=(10, 6))
# extent の設定：横軸は時間、縦軸は周波数（高い周波数が上になるよう反転）
plt.imshow(np.abs(coefficients),
           extent=[t_eval[0], t_eval[-1], frequencies[-1], frequencies[0]],
           aspect='auto', cmap='jet')
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.title("合成信号に対するウェーブレット・スケールグラム")
plt.colorbar(label="Coefficient magnitude")
plt.tight_layout()
plt.show()