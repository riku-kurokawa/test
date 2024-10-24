import numpy as np
import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
mplstyle.use('fast')

# パラメータの設定
Ro, C = 1.21, 343
dx = 5.0e-5
dt = 5.0e-8
waveL = int(0.008575 / dx)
tmin, tmax = 0, int(50*(waveL*dx/C)/dt)*dt
print(tmax)
xmin, xmax = 0, 0.5
K = Ro * C * C

# アニメーションの設定
#frame_interval, dpi, file_name = 20, 400, "ex1_highres"
frame_interval, dpi, file_name = int(waveL*dx/C/dt), 100, "prex2"

# シミュレーションオブジェクトの作成
nx = int((xmax - xmin) / dx) + 1
M = int((tmax - tmin) / dt) + 1

filename1 = f"fix_P_pre1.npy"
field1 = np.memmap(filename1, dtype='float64', mode='r', shape=(int(M/frame_interval)+1, waveL +100))
filename2 = f"fix_P_pre2.npy"
field2 = np.memmap(filename2, dtype='float64', mode='r', shape=(int(M/frame_interval)+1, waveL +100))

def animate1dx2(u1, u2, xlabel='x', ylabel='P'):
    fig, ax = plt.subplots()
    min_value = min(np.min(u1), np.min(u2))
    max_value = max(np.max(u1), np.max(u2))
    ax.set_xlim(0, max(u1.shape[1], u2.shape[1]))
    ax.set_ylim(min_value, max_value)
    ax.set_xlabel(xlabel + f"[m / ({dx})]")
    ax.set_ylabel(ylabel + "[Pa]")

    line1, = ax.plot(u1[0, :], color='red', label='Non-linear')
    line2, = ax.plot(u2[0, :], color='blue', label='Linear')
    ax.legend()

    def update(frame):
        line1.set_ydata(u1[frame, :])
        line2.set_ydata(u2[frame, :])
        return line1, line2

    ani = FuncAnimation(fig, update, frames=np.arange(0, min(u1.shape[0], u2.shape[0]), 1), interval=10, repeat=True)

    output_file = file_name + '_1Dx2.gif'
    ani.save(output_file, writer='pillow', dpi=dpi)

animate1dx2(field1, field2)
