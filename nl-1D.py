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

# アニメーションの設定
#frame_interval, dpi, file_name = 20, 400, "ex1_highres"
frame_interval, dpi, file_name = int(waveL*dx/C/dt), 100, "pre1"

# シミュレーションオブジェクトの作成
nx = int((xmax - xmin) / dx) + 1
M = int((tmax - tmin) / dt) + 1

#create memmap
u_field_filename = f"P_{file_name}.npy"
u_field = np.memmap(u_field_filename, dtype='float64', mode='w+', shape=(M, nx+1))
v_field_filename = f"V_{file_name}.npy"
v_field = np.memmap(v_field_filename, dtype='float64', mode='w+', shape=(M, nx+1))
a_field_filename = f"A_{file_name}.npy"
a_field = np.memmap(a_field_filename, dtype='float64', mode='w+', shape=(M, nx+1))


fix_field_filename = f"fix_P_{file_name}.npy"
fix_field = np.memmap(fix_field_filename, dtype='float64', mode='w+', shape=(int(M/frame_interval)+1, waveL +100))

#sound source
Q = cp.sin(cp.arange(0, 2 * cp.pi, 2 * cp.pi / 400))

#difine array
P1, P2 = cp.zeros((nx + 1), "float64"), cp.zeros((nx + 1), "float64")
Ux1, Ux2 = cp.zeros((nx + 1), "float64"), cp.zeros((nx + 1), "float64")

#mic = []

P1[50:(50 + waveL)] = -1000 * cp.sin(2 * cp.pi * cp.arange(waveL) / waveL)
Ux1[50:(50 + waveL)] = -1000 *cp.sin(2 * cp.pi * cp.arange(waveL)/ waveL) / Ro /C

for n in tqdm(range(M - 1)):
    #if n < len(Q):
        #P1[1000] += Q[n]*2000

    #mic.append(P1[0, 0, 0])
    Ux2[0:nx] = Ux1[0:nx] - dt * (P1[0:nx] - cp.roll(P1[0:nx], 1)) / (dx * (Ro + (P1[0:nx] + cp.roll(P1[0:nx], 1)) / (2 * C**2))) - dt * Ux1[0:nx] * (Ux1[1:nx+1] - cp.roll(Ux1[0:nx], 1)) / (2 * dx)
    P2[0:nx] = P1[0:nx] - dt * C**2 * ((Ro + (P1[1:nx+1] + P1[0:nx]) / (2 * C**2)) * Ux2[1:nx+1] - (Ro + (P1[0:nx] + cp.roll(P1[0:nx], 1)) / (2 * C**2)) * Ux2[0:nx]) / dx

    P2[nx]  = P2[0]
    Ux2[nx] = Ux2[0]

    if (n % frame_interval == 0 ):
        fix_field[int(n / frame_interval),:] = P2[int(dt*n*C/dx):100 + waveL + int(dt*n*C/dx)].get()

    #u_field[n + 1] = P2.get()
    #v_field[n + 1] = Ux2.get()

    P1, P2 = P2, P1
    Ux1, Ux2 = Ux2, Ux1

#a_field[:, 1:] = -(u_field[:, 1:] - u_field[:, 0:-1]) / dx

def animate1d(u, frame_interval, dpi, file_name):
        fig, ax = plt.subplots()
        min_value = np.min(u)
        max_value = np.max(u)
        ax.set_xlim(0, u.shape[1] - 1)
        ax.set_ylim(min_value*2, max_value*2)

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
        for i in range(len(u[frame])):
            new_x.extend(i + np.random.rand(int(u[frame, i]) + 10))
            new_y.extend(np.random.rand(int(u[frame, i]) + 10))

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


animate1d(fix_field[:,:] , 1, dpi, file_name)