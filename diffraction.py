import numpy as np
import cupy as cp
from tqdm import tqdm
from wave_animation_3d import WaveAnimation3D, print_memory_usage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
mplstyle.use('fast')


# パラメータの設定
Ro, C = 1.21, 343
K = Ro * C * C
Ro2, C2 = 2.42, 400
K2 = Ro2 * C2 * C2
dx = dy = dz = 0.01
dt = 0.000005
tmin, tmax = 0, 0.01
xmin, xmax, ymin, ymax, zmin, zmax = 0, 2, 0, 2, 0, 2
frequency = 5000

# アニメーションの設定
frame_interval, dpi, file_name = 20, 400, "diffraction"

# シミュレーションオブジェクトの作成
wave = WaveAnimation3D(dx, dy, dz, dt, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, C)
t, M, nx, ny, nz, limit = wave.initialize()
 
#音源
x_sp, y_sp, length, angle = int(nx/2), int(nx/8), 10, np.pi
x = np.arange(x_sp*10,(x_sp + length)*10)/10
y = np.tan(angle) * (x - x_sp) + y_sp

 
# 格子上の点として座標を格納
lattice_points = np.round(x).astype(int), np.round(y).astype(int)
# 同じ座標が格納されないように修正
unique_points = np.unique(lattice_points, axis=1)
lattice_points = unique_points[0], unique_points[1]

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
    #    P1[int(ny/2), int(ny/2), 0] += Q[n]
    Ro2, C2 = Ro + 5 * np.sin(n/500), C + 5 * np.sin(n/500)
    K2 = Ro2 * C2 * C2

    
    for (i_lattice, j_lattice) in zip(*unique_points):
        P1[i_lattice, int(ny/2-10):int(ny/2+10), j_lattice] += np.sin(n/50)
        P1[i_lattice, int(ny/2-10):int(ny/2+10), j_lattice -1 ] = 0      
    #mic.append(P1[0, 0, 0])
    border = int(nx / 2) * 0 + 1
    
    Ux2[1:nx, :, :border] = Ux1[1:nx, :, :border] - dt / Ro / dx * (P1[1:nx, :, :border] - P1[0:nx - 1, :, :border])
    Uy2[:, 1:ny, :border] = Uy1[:, 1:ny, :border] - dt / Ro / dy * (P1[:, 1:ny, :border] - P1[:, 0:ny - 1, :border])
    Uz2[:, :, 1:border] = Uz1[:, :, 1:border] - dt / Ro / dz * (P1[:, :, 1:border] - P1[:, :, 0:border - 1])
        
    Ux2[1:nx, :, border:] = Ux1[1:nx, :, border:] - dt / Ro2 / dx * (P1[1:nx, :, border:] - P1[0:nx - 1, :, border:])
    Uy2[:, 1:ny, border:] = Uy1[:, 1:ny, border:] - dt / Ro2 / dy * (P1[:, 1:ny, border:] - P1[:, 0:ny - 1, border:])
    Uz2[:, :, border:nz] = Uz1[:, :, border:nz] - dt / Ro2 / dz * (P1[:, :, border:nz] - P1[:, :, border -1 :nz - 1])

    Ro2, C2 = Ro + 5 * np.sin((n + 0.5)/500), C + 5 * np.sin((n + 0.5)/500)
    K2 = Ro2 * C2 * C2

    P2[:, :, :border] = P1[:, :, :border] - K * dt / dx * (Ux2[1:nx + 1, :, :border] - Ux2[0:nx, :, :border]) \
        - K * dt / dy * (Uy2[:, 1:ny + 1, :border] - Uy2[:, 0:ny, :border]) - K * dt / dz * (Uz2[:, :, 1:border + 1] - Uz2[:, :, 0:border])  
    
    P2[:, :, border:] = P1[:, :, border:] - K2 * dt / dx * (Ux2[1:nx + 1, :, border:] - Ux2[0:nx, :, border:]) \
        - K2 * dt / dy * (Uy2[:, 1:ny + 1, border:] - Uy2[:, 0:ny, border:]) - K2 * dt / dz * (Uz2[:, :, border + 1:nz + 1] - Uz2[:, :, border:nz])
    
    
    u_field[n + 1, :, int(ny / 2), :] = P2[:, int(ny / 2), :].get()

    P1, P2 = P2, P1
    Ux1, Ux2 = Ux2, Ux1
    Uy1, Uy2 = Uy2, Uy1
    Uz1, Uz2 = Uz2, Uz1

    # メモリ管理
    #if n % limit == limit - 1:
        #u_field.flush()
        #del u_field
        #u_field = np.memmap(u_field_filename, dtype='float64', mode='r+', shape=(M, nx, ny, nz))

wave.animate(u_field, frame_interval, dpi, file_name, 0, M, plane='xz', plane_index = int(ny / 2))    
wave.animate1d(u_field[:,int(nx/2), int(ny/2), :], frame_interval, dpi, file_name, xlabel='z', ylabel='u')
