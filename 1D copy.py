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
K = Ro * C * C
#Ro2, C2 = 40, 800
Ro2, C2 = Ro, C

K2 = Ro2 * C2 * C2
dx = 0.5
dt = 0.0005
tmin, tmax = 0, 0.9
xmin, xmax = 0, 400
frequency = 5000

# アニメーションの設定
frame_interval, dpi, file_name = 20, 400, "ex2_highres"
#frame_interval, dpi, file_name = 20, 100, "ex2"

# シミュレーションオブジェクトの作成
nx = int((xmax - xmin) / dx) + 1
M = int((tmax - tmin) / dt) + 1

#create memmap
u_field_filename = f"P_{file_name}.npy"
u_field = np.memmap(u_field_filename, dtype='float64', mode='w+', shape=(M, nx))
v_field_filename = f"V_{file_name}.npy"
v_field = np.memmap(v_field_filename, dtype='float64', mode='w+', shape=(M, nx+1))
a_field_filename = f"A_{file_name}.npy"
a_field = np.memmap(a_field_filename, dtype='float64', mode='w+', shape=(M, nx))

#sound source
Q =0.5 + 0.5 * cp.cos(cp.arange(-cp.pi, cp.pi, 2 * cp.pi / 200))

#difine array
P1, P2 = cp.zeros((nx), "float64"), cp.zeros((nx), "float64")
Ux1, Ux2 = cp.zeros((nx + 1), "float64"), cp.zeros((nx + 1), "float64")

#mic = []

for n in tqdm(range(M - 1)):
    if n < len(Q):
        P1[1] += Q[n]
        P1[-1] += Q[n]

    #mic.append(P1[0, 0, 0])
    Ux2[1:int(nx/2)] = Ux1[1:int(nx/2)] - dt / Ro / dx * (P1[1:int(nx/2)] - P1[0:int(nx/2) - 1])
    Ux2[int(nx/2):nx] = Ux1[int(nx/2):nx] - dt / Ro2 / dx * (P1[int(nx/2):nx] - P1[int(nx/2)-1:nx - 1])

    P2[:int(nx/2)] = P1[:int(nx/2)] - K * dt / dx * (Ux2[1:int(nx/2) + 1] - Ux2[0:int(nx/2)])
    P2[int(nx/2):] = P1[int(nx/2):] - K2 * dt / dx * (Ux2[int(nx/2) + 1:nx + 1] - Ux2[int(nx/2):nx])


    u_field[n + 1] = P2.get()
    v_field[n + 1] = Ux2.get()

    P1, P2 = P2, P1
    Ux1, Ux2 = Ux2, Ux1

a_field[:, 1:] = -(u_field[:, 1:] - u_field[:, 0:-1]) / dx

def animate1d(u, frame_interval, dpi, file_name):
        fig, ax = plt.subplots()
        min_value = np.min(u)
        max_value = np.max(u)
        ax.set_xlim(0, u.shape[1] - 1)
        ax.set_ylim(min_value, max_value*1.1)

        line, = ax.plot(u[0, :], color='blue')

        def update(frame):
            line.set_ydata(u[frame, :])
            ax.set_title(f'Frame {frame}')
            return line,

        ani = FuncAnimation(fig, update, frames=np.arange(0, u.shape[0], frame_interval), interval=200, repeat=True)

        output_file = file_name + '_1D.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)

def animate1dx2(u1, u2, frame_interval, dpi, file_name, xlabel='z', ylabel='u'):
        fig, ax = plt.subplots()
        min_value = min(np.min(u1), np.min(u2))
        max_value = max(np.max(u1), np.max(u2))
        ax.set_xlim(0, max(u1.shape[1], u2.shape[1]))
        ax.set_ylim(min_value, max_value*1.1)
        ax.set_xlabel(xlabel + "[m]")
        ax.set_ylabel(ylabel)

        line1, = ax.plot(u1[0, :], color='red', label='Affected')
        line2, = ax.plot(u2[0, :], color='blue', label='Not affected')
        ax.legend()

        def update(frame):
            line1.set_ydata(u1[frame, :])
            line2.set_ydata(u2[frame, :])
            decimal_places = len(str(dt).split('.')[-1])
            k = round(frame * dt, decimal_places)
            formatted_k = "{:.{}f}".format(k, decimal_places)
            ax.set_title('Time: ' + formatted_k + ' s', size=10)
            return line1, line2

        ani = FuncAnimation(fig, update, frames=np.arange(0, min(u1.shape[0], u2.shape[0]), frame_interval), interval=200, repeat=True)

        output_file = file_name + '_1Dx2.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)

def vectorAni(u, v, interval = 1, frame_interval = 1, dpi = 400, file_name = "temp"):
    leap_x = np.arange(0, len(u[0]), interval)
    fig, ax = plt.subplots()
    quiver = ax.quiver(leap_x, u[0, leap_x], v[0, leap_x], u[0, leap_x], scale=1, width=0.01)

    line, = ax.plot(u[0, :], color='blue')

    # グラフの表示範囲を設定
    ax.set_xlim(0, len(u[0]))
    ax.set_ylim(np.min(u), np.max(u)*1.1)

    def update(frame):
        # フレームごとにベクトルを更新
        quiver.set_offsets(np.column_stack((leap_x, u[frame, leap_x])))
        quiver.set_UVC(v[frame, leap_x], 0)
        line.set_ydata(u[frame, :])
        return quiver, line,

    ani = FuncAnimation(fig, update, frames=np.arange(0, u.shape[0], frame_interval), interval=200, repeat=True)
    
    output_file = file_name + '_vector.gif'
    ani.save(output_file, writer='pillow', dpi=dpi)

    plt.show()

def densityAni(u, frame_interval, dpi, file_name):
    # プロットの初期化
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], s = 1)  # 空の散布図を作成
    ax.set_xlim(0, len(u[0]))
    ax.set_ylim(0, 1)
    # y軸の目盛りを非表示にする
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    # アニメーションの更新関数
    def update(frame):
        # データをランダムに更新
        new_x = []
        new_y = []
        for i in range(int(len(u[frame])/2)):
            new_x.extend(i + np.random.rand(int(u[frame, i]) + 10))
            new_y.extend(np.random.rand(int(u[frame, i]) + 10))
            new_x.extend(i + int(len(u[frame])/2) + np.random.rand(int(u[frame, i + int(len(u[frame])/2)]) + 40))
            new_y.extend(np.random.rand(int(u[frame, i  + int(len(u[frame])/2)]) + 40))

        # プロットに新しいデータを追加
        scatter.set_offsets(np.column_stack((new_x, new_y)))

        # タイトルにフレーム番号を表示
        ax.set_title(f'Frame {frame}')

    # アニメーションの設定
    ani = FuncAnimation(fig, update, frames=np.arange(0, u.shape[0], frame_interval), interval=200, repeat=True)

    output_file = file_name + '_density.gif'
    ani.save(output_file, writer='pillow', dpi=dpi)

    # 表示
    plt.show()

animate1d(u_field[int(M/4):,int(nx/4):int(3*nx/4)] / np.max(u_field[int(M/4):,int(nx/4):int(3*nx/4)]), frame_interval, dpi, file_name)
vectorAni(u_field[int(M/4):,int(nx/4):int(3*nx/4)] / np.max(u_field[int(M/4):,int(nx/4):int(3*nx/4)]), 0.5 * v_field[int(M/4):,int(nx/4):int(3*nx/4)] / np.max(v_field[int(M/4):,int(nx/4):int(3*nx/4)]), 5, frame_interval, dpi, file_name + "v")
densityAni(u_field[int(M/4):,int(nx/4):int(3*nx/4)] * 20, frame_interval, dpi, file_name)
vectorAni(u_field[int(M/4):,int(nx/4):int(3*nx/4)] / np.max(u_field[int(M/4):,int(nx/4):int(3*nx/4)]), 0.2* a_field[int(M/4):,int(nx/4):int(3*nx/4)] / np.max(a_field[int(M/4):,int(nx/4):int(3*nx/4)]), 5, frame_interval, dpi, file_name + "a")
