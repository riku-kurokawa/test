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
dx = dy = dz = 0.02
dt = 0.00001
tmin, tmax = 0, 0.05
xmin, xmax, ymin, ymax, zmin, zmax = 0, 100, 0, 100, 0, 100

# アニメーションの設定
frame_interval, dpi, file_name = 50, 400, "fdtd_test"

# シミュレーションオブジェクトの作成
wave = WaveAnimation3D(dx, dy, dz, dt, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, C)
t, M, nx, ny, nz, limit = wave.initialize()

enclosure, stl_data1 =  wave.loadSTL("3Dmodels/block2",0.5)
M5N, stl_data2 =  wave.loadSTL("3Dmodels/M5N2",0.5)
Q3, stl_data3 =  wave.loadSTL("3Dmodels/Q32",0.5)

# プロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')  # 正射投影を指定
#ax.set_box_aspect([1, 1, 1])  # 各軸のアスペクト比を指定
# アスペクト比を設定して数値の幅を正確に表示
limits = np.array(((np.min(enclosure[:, 0]), np.max(enclosure[:, 0])),
        (np.min(enclosure[:, 1]), np.max(enclosure[:, 1])),
        (np.min(enclosure[:, 2]), np.max(enclosure[:, 2]))))
limits *= 1.2

ax.set_xlim(*limits[0])
ax.set_ylim(*limits[1])
ax.set_zlim(*limits[2])
ax.set_box_aspect([range_extent / 2.0 for range_extent in (limits[0][1] - limits[0][0],
                                                            limits[1][1] - limits[1][0],
                                                            limits[2][1] - limits[2][0])])


# enclosureの点をプロット
# Plot the points with gradient colors based on distances

mesh_data_faces1 = Poly3DCollection(stl_data1.vectors / dx, facecolors='gray', alpha = 0.4)
ax.add_collection3d(mesh_data_faces1)
mesh_data_faces2 = Poly3DCollection(stl_data2.vectors / dx, facecolors='red')
ax.add_collection3d(mesh_data_faces2)
mesh_data_faces3 = Poly3DCollection(stl_data3.vectors / dx, facecolors='blue')
ax.add_collection3d(mesh_data_faces3)

 # Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()