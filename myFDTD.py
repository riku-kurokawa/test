import numpy as np
import cupy as cp
from tqdm import tqdm
from wave_animation_3d import WaveAnimation3D, print_memory_usage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
mplstyle.use('fast')


# パラメータの設定
Ro, C = 1.21, 343
K = Ro * C * C
dx = dy = dz = 0.0075
dt = 0.000005
tmin, tmax = 0, 0.005
xmin, xmax, ymin, ymax, zmin, zmax = 0, 0.2, 0, 0.2, 0, 0.2
frequency = 5000

# アニメーションの設定
frame_interval, dpi, file_name = 20, 400, "fdtd_STL_test4"

# シミュレーションオブジェクトの作成
wave = WaveAnimation3D(dx, dy, dz, dt, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, C)
t, M, nx, ny, nz, limit = wave.initialize()

#loading STL
enclosure, stl_data1 =  wave.loadSTL("3Dmodels/2",0.5)
M5N, stl_data2 =  wave.loadSTL("3Dmodels/1",0.5)
#Q3, stl_data3 =  wave.loadSTL("3Dmodels/Q32",0.5)
enclosure = np.round(np.array(enclosure)).astype(int)
M5N = np.round(np.array(M5N)).astype(int)
#Q3 = np.round(np.array(Q3)).astype(int)

#create memmap
u_field_filename = f"{file_name}.npy"
u_field = np.memmap(u_field_filename, dtype='float64', mode='w+', shape=(M, nx, ny, nz))

#sound source
Q =( 0.5 + 0.5 * cp.cos(cp.arange(-40*cp.pi, 40*cp.pi, 2 * cp.pi / 80)))*100

#difine array
P1, P2 = cp.zeros((nx, ny, nz), "float64"), cp.zeros((nx, ny, nz), "float64")
Ux1, Ux2 = cp.zeros((nx + 1, ny, nz), "float64"), cp.zeros((nx + 1, ny, nz), "float64")
Uy1, Uy2 = cp.zeros((nx, ny + 1, nz), "float64"), cp.zeros((nx, ny + 1, nz), "float64")
Uz1, Uz2 = cp.zeros((nx, ny, nz + 1), "float64"), cp.zeros((nx, ny, nz + 1), "float64")

#mic = []

for n in tqdm(range(M - 1)):
    #if n < len(Q):
        #P1[20, 15, 15] += Q[n]

    #mic.append(P1[0, 0, 0])

    Ux2[1:nx, :, :] = Ux1[1:nx, :, :] - dt / Ro / dx * (P1[1:nx, :, :] - P1[0:nx - 1, :, :])
    Uy2[:, 1:ny, :] = Uy1[:, 1:ny, :] - dt / Ro / dy * (P1[:, 1:ny, :] - P1[:, 0:ny - 1, :])
    Uz2[:, :, 1:nz] = Uz1[:, :, 1:nz] - dt / Ro / dz * (P1[:, :, 1:nz] - P1[:, :, 0:nz - 1])

    P2[:, :, :] = P1[:, :, :] - K * dt / dx * (Ux2[1:nx + 1, :, :] - Ux2[0:nx, :, :]) \
        - K * dt / dy * (Uy2[:, 1:ny + 1, :] - Uy2[:, 0:ny, :]) - K * dt / dz * (Uz2[:, :, 1:nz + 1] - Uz2[:, :, 0:nz])
    
    P2[np.array(enclosure)[:, 0], np.array(enclosure)[:, 1], np.array(enclosure)[:, 2]] = 0
    
    #if n < len(Q):
    P2[np.array(M5N)[:, 0], np.array(M5N)[:, 1], np.array(M5N)[:, 2]] += cp.sin(2*cp.pi*(n+1)*dt*frequency) *2
    #P2[np.array(Q3)[:, 0], np.array(Q3)[:, 1], np.array(Q3)[:, 2]] += cp.sin(2*cp.pi*(n+1)*dt*frequency*4)*5
        

    u_field[n + 1] = P2.get()

    P1, P2 = P2, P1
    Ux1, Ux2 = Ux2, Ux1
    Uy1, Uy2 = Uy2, Uy1
    Uz1, Uz2 = Uz2, Uz1

    # メモリ管理
    if n % limit == limit - 1:
        u_field.flush()
        del u_field
        u_field = np.memmap(u_field_filename, dtype='float64', mode='r+', shape=(M, nx, ny, nz))

max_value =  np.max(u_field)
u_field[:,0,0,0] = max_value

#wave.animate(u_field, frame_interval, dpi, file_name, 0, M, plane='xy', plane_index=int(nz / 2))    
def plot3d(u, ax, colors, alpha, cutoff):
        # メッシュグリッドを作成
        x = np.arange(u.shape[0])
        y = np.arange(u.shape[1])
        z = np.arange(u.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # プロットする点の座標を取得
        x_points = X.flatten()
        y_points = Y.flatten()
        z_points = Z.flatten()

        # プロットする点の強度を取得
        intensities = u.flatten()

        # 0でない点のみを描画
        mask = np.abs(intensities) >= cutoff
        if mask.any():  # 0でない点が存在する場合のみプロット
            ax.scatter(x_points[mask], y_points[mask], z_points[mask], c=colors[mask], alpha=alpha[mask], s=1)
            # STLモデルをプロット
            mesh_data_faces1 = Poly3DCollection(stl_data1.vectors / dx, facecolors='gray', alpha = 0.4)
            ax.add_collection3d(mesh_data_faces1)
            mesh_data_faces2 = Poly3DCollection(stl_data2.vectors / dx, facecolors='red')
            ax.add_collection3d(mesh_data_faces2)
            #mesh_data_faces3 = Poly3DCollection(stl_data3.vectors / dx, facecolors='blue')
            #ax.add_collection3d(mesh_data_faces3)

        # 軸ラベルを設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


def animate3d(u, M, frame_interval, dpi, file_name, cutoff=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))

    vmin = np.min(u)
    vmax = np.max(u)

    def update(frame):
        ax.cla()

        cmap = plt.colormaps['jet']
        colors_frame = cmap((u[frame].flatten() - vmin) / (vmax - vmin))

        alpha_frame = np.abs(u[frame].flatten()) / (vmax*5)
        plot3d(u[frame], ax, colors_frame, alpha_frame, cutoff)

        decimal_places = len(str(dt).split('.')[-1])
        k = round(frame * dt, decimal_places)
        formatted_k = "{:.{}f}".format(k, decimal_places)
        ax.set_title('Time: ' + formatted_k + ' s', size=10)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

    update(0)

    with tqdm(total=(M - 1) // frame_interval + 1) as pbar:
        def update_with_progress(frame):
            update(frame)
            pbar.update(frame_interval)

        ani = FuncAnimation(fig, update_with_progress, frames=np.arange(0, M, frame_interval), interval=10)

    output_file = file_name + '_3d.gif'
    try:
        cmap = plt.colormaps['jet']
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # ax パラメータを指定して Colorbar を追加
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Intensity')

        ani.save(output_file, writer='pillow', dpi=dpi)

    except IndexError:
        pass

animate3d(u_field, M, frame_interval, dpi, file_name + "3D", cutoff=0.15)