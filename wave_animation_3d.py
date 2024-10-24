# import libraries
import numpy as np
from stl import mesh
import numpy as np
from tqdm import tqdm
from scipy.signal import hamming, find_peaks
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
mplstyle.use('fast')

class WaveAnimation3D:
    def __init__(self, dx, dy, dz, dt, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, c):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.tmin = tmin
        self.tmax = tmax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.c = c

        self.Lx = xmax - xmin
        self.Ly = ymax - ymin
        self.Lz = zmax - zmin
        self.Lt = tmax - tmin

    def initialize(self):
        self.nx = int((self.xmax - self.xmin) / self.dx) + 1
        self.ny = int((self.ymax - self.ymin) / self.dy) + 1
        self.nz = int((self.zmax - self.zmin) / self.dz) + 1
        self.M = int((self.tmax - self.tmin) / self.dt) + 1

        t = np.linspace(0, (self.tmax - self.tmin), self.M)

        # メモリ管理
        limit = int(25 * 1024 * 1024 * 1024 / (self.nx*self.ny*self.nz*8))
        print(f"cycle_limit_num: {limit}")

        return t, self.M, self.nx, self.ny, self.nz, limit

    def animate(self, u, frame_interval, dpi, file_name, start, end, plane='xy', plane_index=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if plane == 'xy':
            data = u[:, :, :, plane_index]
            data = np.transpose(data,(0,2,1))  # データの軸を変更
            extent = [self.ymin, self.ymax, self.xmin, self.xmax]  
            ax.set_xlabel('x[m]')  
            ax.set_ylabel('y[m]')
        elif plane == 'xz':
            data = u[:, :, plane_index, :]
            data = np.transpose(data,(0,2,1))  # データの軸を変更
            extent = [self.zmin, self.zmax, self.xmin, self.xmax]  
            ax.set_xlabel('x[m]')  
            ax.set_ylabel('z[m]')
        elif plane == 'yz':
            data = u[ :, plane_index, :, :]
            data = np.transpose(data,(0,2,1))  # データの軸を変更
            extent = [self.zmin, self.zmax, self.ymin, self.ymax]  
            ax.set_xlabel('x[m]')  
            ax.set_ylabel('y[m]')

        # ヒートマップの表示
        heatmap = ax.imshow(data[0, :, :], cmap='jet', extent=extent, origin='lower')   

        vmin = np.min(data)
        vmax = np.max(data)
        heatmap.set_clim(vmin, vmax)
        fig.colorbar(heatmap)

        def update(frame):
            heatmap.set_array(data[frame, :, :])

            decimal_places = len(str(self.dt).split('.')[-1])
            k = round(frame * self.dt, decimal_places)
            formatted_k = "{:.{}f}".format(k, decimal_places)
            ax.set_title('Time: ' + formatted_k + ' s', size=10)
            return heatmap,

        ani = FuncAnimation(fig, update, frames=np.arange(start, end, frame_interval), interval=10, repeat=True)

        output_file = file_name + '.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)
        #plt.show()


    def save_plot(self,u, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = np.transpose(u)  # データの軸を変更

        heatmap = ax.imshow(data, cmap='jet', extent=[0, self.Lx, 0, self.Ly], origin='lower')
        ax.set_xlabel('x[m]')
        ax.set_ylabel('y[m]')

        vmin = np.min(data)
        vmax = np.max(data)
        heatmap.set_clim(vmin, vmax)
        fig.colorbar(heatmap)

        output_file = file_name + '_intensity.png'
        plt.savefig(output_file, dpi=300)  # ファイルをPNGとして保存します
        plt.close(fig)   
        
    def plot3d(self, u, ax, colors, alpha, cutoff):
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

        # 軸ラベルを設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


    def animate3d(self, u, M, frame_interval, dpi, file_name, cutoff = 0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1, 1, 1))  # 2D版と若干違う

        vmin = np.min(u)  # 最小値を計算
        vmax = np.max(u)  # 最大値を計算

        def update(frame):
            ax.cla()  # プロットをクリア

            # colorsとalphaをフレームごとに再計算
            cmap = plt.cm.get_cmap('jet')
            colors_frame = cmap((u[frame].flatten() - vmin) / (vmax - vmin))  # カラーマップを計算

            alpha_frame = np.abs(u[frame].flatten()) / (vmax)  # 透明度を計算
            self.plot3d(u[frame], ax, colors_frame, alpha_frame, cutoff)

            decimal_places = len(str(self.dt).split('.')[-1])
            k = round(frame * self.dt, decimal_places)
            formatted_k = "{:.{}f}".format(k, decimal_places)
            ax.set_title('Time: ' + formatted_k + ' s', size=10)
            ax.set_xlim(0, self.nx)  # X軸の表示範囲を設定
            ax.set_ylim(0, self.ny)  # Y軸の表示範囲を設定
            ax.set_zlim(0, self.nz)  # Z軸の表示範囲を設定

        update(0)  # 最初のフレームをプロット

        with tqdm(total=(M - 1) // frame_interval + 1) as pbar:  # tqdmプログレスバーを作成
            def update_with_progress(frame):
                update(frame)  # フレームを更新
                pbar.update(frame_interval)  # プログレスバーを進行させる

            ani = FuncAnimation(fig, update_with_progress, frames=np.arange(0, M, frame_interval), interval=10)

        output_file = file_name + '_3d.gif'
        try:
            # カラーバーを表示する
            cmap = plt.cm.get_cmap('jet')
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm)
            ani.save(output_file, writer='pillow', dpi=dpi)
            
        except IndexError:
            pass
    
        
    def animate1d(self, u, frame_interval, dpi, file_name, xlabel='z', ylabel='u'):
        fig, ax = plt.subplots()
        min_value = np.min(u)
        max_value = np.max(u)
        ax.set_xlim(0, u.shape[1] - 1)
        ax.set_ylim(min_value, max_value)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('1D Animation')

        line, = ax.plot(u[0, :], color='blue')

        def update(frame):
            line.set_ydata(u[frame, :])
            ax.set_title(f'Frame {frame}')
            return line,

        ani = FuncAnimation(fig, update, frames=np.arange(0, u.shape[0], frame_interval), interval=10, repeat=True)

        output_file = file_name + '_1D.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)

    def animate1dx2(self, u1, u2, frame_interval, dpi, file_name, xlabel='z', ylabel='u'):
        fig, ax = plt.subplots()
        min_value = min(np.min(u1), np.min(u2))
        max_value = max(np.max(u1), np.max(u2))
        ax.set_xlim(25, max(u1.shape[1], u2.shape[1]) - 20)
        ax.set_ylim(min_value, max_value)
        ax.set_xlabel(xlabel + "[m]")
        ax.set_ylabel(ylabel)

        line1, = ax.plot(u1[0, :], color='red', label='Affected')
        line2, = ax.plot(u2[0, :], color='blue', label='Not affected')
        ax.legend()

        def update(frame):
            line1.set_ydata(u1[frame, :])
            line2.set_ydata(u2[frame, :])
            decimal_places = len(str(self.dt).split('.')[-1])
            k = round(frame * self.dt, decimal_places)
            formatted_k = "{:.{}f}".format(k, decimal_places)
            ax.set_title('Time: ' + formatted_k + ' s', size=10)
            return line1, line2

        ani = FuncAnimation(fig, update, frames=np.arange(0, min(u1.shape[0], u2.shape[0]), frame_interval), interval=10, repeat=True)

        output_file = file_name + '_1Dx2.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)

    def rms_plot(self,data,freq):
        rate = 1/self.dt
        # グラフの初期化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # 波形を表示
        time_values = np.arange(len(data)) / rate
        ax1.plot(time_values, data, label='Waveform')  # labelを追加

        ax1.set_ylabel('Amplitude')
        ax1.legend()

        # RMSを計算して表示
        window_size = int(rate /freq)
        squared_values = data ** 2
        rms = np.sqrt(np.convolve(squared_values, np.ones(window_size)/window_size, mode='same'))
        ax2.plot(time_values, rms, label='RMS')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RMS')
        ax2.legend()

        # グラフを表示
        plt.tight_layout()
        #plt.show()

    def rms_plotx2(self, data, data2, freq):
        rate = 1 / self.dt
        # グラフの初期化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # 波形を表示
        time_values = np.arange(len(data)) / rate
        ax1.plot(time_values, data, label='Waveform 1')  # labelを追加
        ax1.plot(time_values, data2, label='Waveform 2')  # 第2のデータもプロット
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        # RMSを計算して表示
        window_size = int(rate / freq)
        squared_values = data ** 2
        rms = np.sqrt(np.convolve(squared_values, np.ones(window_size) / window_size, mode='same'))
        ax2.plot(time_values, rms, label='RMS 1')  # labelを追加
        squared_values2 = data2 ** 2
        rms2 = np.sqrt(np.convolve(squared_values2, np.ones(window_size) / window_size, mode='same'))
        ax2.plot(time_values, rms2, label='RMS 2')  # 第2のデータのRMSもプロット
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RMS')
        ax2.legend()

        # グラフを表示
        plt.tight_layout()
        #plt.show()

    def fft(self,data,desired_frequency):
        rate = 1 / self.dt

        # データの長さ
        data_length = len(data)

        # フレームサイズとオーバーラップを自動設定
        frame_duration = 10/desired_frequency  # フレームの時間（秒）
        overlap_duration = frame_duration*0.5  # オーバーラップの時間（秒）

        frame_size = int(frame_duration * rate)
        overlap = int(overlap_duration * rate)

        # ウィンドウ関数（ハミング窓）
        window = hamming(frame_size)


        # ウィンドウ関数を適用しながらFFTを計算する関数
        def calculate_fft(index):
            start = index * overlap
            end = start + frame_size
            frame = data[start:end] * window
            spectrum = np.abs(np.fft.fft(frame))
            spectrum = spectrum[:frame_size // 2]  # ナイキスト周波数までの範囲に制限
            spectrum = np.log10(spectrum + 1)  # 対数表示
            line.set_ydata(spectrum)
    
            # 時間を表示
            time_label.set_text(f'Time: {index * overlap / rate:.2f} s')
    
            # FFTピーク情報を計算して表示
            peaks, _ = find_peaks(spectrum, height=0.5, distance=50)  # ピークを検出
            peak_amplitudes = spectrum[peaks]
            peak_indices_sorted = np.argsort(peak_amplitudes)[::-1]  # ピークを降順でソート
            peak_frequencies = frequencies[peaks][peak_indices_sorted]
            peak_amplitudes = peak_amplitudes[peak_indices_sorted]
    
            peak_info_text = "\n".join([f"{f:.0f} Hz: {a:.2f}" for f, a in zip(peak_frequencies, peak_amplitudes)])
            peak_info_text = "FFT Peaks:\n" + peak_info_text
            peak_info.set_text(peak_info_text)
    
            return line, time_label, peak_info

        # FFT結果の最大値を計算
        max_spectrum = 0
        for i in range(0, len(data) - frame_size, overlap):
            frame = data[i:i + frame_size] * window
            spectrum = np.abs(np.fft.fft(frame))
            max_spectrum = max(max_spectrum, np.max(spectrum[:frame_size // 2]))

        # グラフの初期化
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        frequencies = np.fft.fftfreq(frame_size, 1.0 / rate)[:frame_size // 2]
        line, = ax.plot(frequencies, np.zeros(frame_size // 2))
        ax.set_xlim(0, frequencies[-1])
        ax.set_ylim(0, np.log10(max_spectrum + 1) + 1)  # 最大値に合わせてy軸の範囲を設定
        ax.set_ylabel('Amplitude')

        # y軸に単位を設定
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(10 ** y)))

        # x軸に単位を設定
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x) + ' Hz'))
        ax.set_xlabel('Frequency')

        # 時間を表示するテキスト
        time_label = ax.text(0.75, 0.9, '', transform=ax.transAxes, fontsize=12)

        # FFTピーク情報を表示するテキスト
        peak_info = ax.text(0.02, 0.75, '', transform=ax.transAxes, fontsize=12, color='red')

        # アニメーションの設定
        ani = FuncAnimation(fig, calculate_fft, frames=(data_length - frame_size) // overlap, blit=True, interval=1)
       
        # 静止グラフを別ウィンドウで表示
        fig_static, ax_static = plt.subplots(1, 1, figsize=(8, 6))
       
        # 静止グラフの作成
        desired_frequency_index = np.abs(frequencies - desired_frequency).argmin()
        scatter_x = []
        scatter_y = []
        time_values = []

        for i in range(0, data_length - frame_size, overlap):
            frame = data[i:i + frame_size] * window
            spectrum = np.abs(np.fft.fft(frame))
            spectrum = spectrum[:frame_size // 2]
            scatter_x.append(i / rate)
            scatter_y.append(spectrum[desired_frequency_index])
            time_values.append(i / rate)
    
        # 静止グラフをプロット
        ax_static.plot(time_values, scatter_y)

        # y軸の範囲を設定（最小値から最大値を自動で決定）
        ax_static.set_ylim(np.min(scatter_y), np.max(scatter_y))

        # y軸に単位を設定
        ax_static.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

        # x軸に単位を設定
        ax_static.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))

        ax_static.set_xlabel('Time (s)')
        ax_static.set_ylabel(f'Amplitude at {desired_frequency} Hz')

        # グラフを表示
        plt.tight_layout()
        plt.show()
    
    def loadSTL(self,STL_file_name,fineness = 1.):
        # set parameters
        #Standard_Area = 0.5 * (self.dx**2) * 0.1

        # Load STL file
        stl_data = mesh.Mesh.from_file(f'{STL_file_name}.stl')
        points = stl_data.points.reshape([-1, 3, 3])

        SurfaceDots = []  # Initialize as a list

        for i in tqdm(range(len(points))):
            triangles = points[i:i+1]

            while triangles.any():
                new_triangles = []  # Initialize as a list
                for triangle in triangles:
                    A, B, C = triangle

                    # Calculate the area of the triangle using its vertices
                    #S = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
                    #if S > Standard_Area:
                    ab = np.linalg.norm(B-A)
                    bc = np.linalg.norm(C-B)
                    ca = np.linalg.norm(A-C)
                    
                    
                    if np.max([ab,bc,ca]) > self.dx*fineness:
                        # サブディビジョン
                        AB = (A + B) / 2
                        BC = (B + C) / 2
                        CA = (C + A) / 2

                        new_triangles.append([A, AB, CA])
                        new_triangles.append([AB, B, BC])
                        new_triangles.append([CA, C, BC])
                        new_triangles.append([AB, BC, CA])

                    else:
                        SurfaceDots.append([A, B, C])

                triangles = np.array(new_triangles)

        # 最終的にNumPy配列に変換
        SurfaceDots = np.array(SurfaceDots)
        Round_SurfaceDots = np.round(SurfaceDots / self.dx)
        print(Round_SurfaceDots.shape)
        Round_SurfaceDots = np.unique(Round_SurfaceDots, axis=0)

        np.set_printoptions(threshold=np.inf)  # 全ての要素を表示
        print(Round_SurfaceDots.shape)
        
        return  Round_SurfaceDots.reshape([-1, 3]), stl_data
    
    def emty(Round_SurfaceDots):
        # プロットの設定
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')  # 正射投影を指定
        #ax.set_box_aspect([1, 1, 1])  # 各軸のアスペクト比を指定
        # アスペクト比を設定して数値の幅を正確に表示
        limits = ((np.min(Round_SurfaceDots[:, :, 0]), np.max(Round_SurfaceDots[:, :, 0])),
                (np.min(Round_SurfaceDots[:, :, 1]), np.max(Round_SurfaceDots[:, :, 1])),
                (np.min(Round_SurfaceDots[:, :, 2]), np.max(Round_SurfaceDots[:, :, 2])))

        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_box_aspect([range_extent / 2.0 for range_extent in (limits[0][1] - limits[0][0],
                                                                    limits[1][1] - limits[1][0],
                                                                    limits[2][1] - limits[2][0])])



        # SurfaceDotsの点をプロット
        # Calculate distances from the origin for each point
        distances = np.linalg.norm(Round_SurfaceDots, axis=2)

        # Plot the points with gradient colors based on distances
        scatter = ax.scatter(
            Round_SurfaceDots[:, :, 0],
            Round_SurfaceDots[:, :, 1],
            Round_SurfaceDots[:, :, 2],
            c=distances.flatten(),  # Use distances as colors
            marker='.',
            cmap='jet',  # Choose a colormap (you can use other colormaps)
        )

        # Add colorbar for reference
        cbar = plt.colorbar(scatter)
        cbar.set_label('Distance from Origin')

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()



def print_memory_usage(arr):
    memory_usage = arr.nbytes
    units = ["B", "KB", "MB", "GB", "TB"]

    size = memory_usage
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    print(f"Memory Usage: {size:.2f} {units[unit_index]}")

def find_closest_extrema(arr):
    extrema = []

    # 極小値の検出
    for i in range(1, len(arr) - 1):
        if arr[i - 1] > arr[i] < arr[i + 1]:
            extrema.append((i, arr[i]))

    # 極大値の検出
    for i in range(1, len(arr) - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:
            extrema.append((i, arr[i]))

    min_difference = float('inf')
    result = None

    # 隣り合う極大値同士のペアと隣り合う極小値同士のペアから最小差のペアを検索
    for i in range(len(extrema) - 1):
        current_difference = abs(extrema[i][1] - extrema[i + 1][1])
        if current_difference < min_difference:
            min_difference = current_difference
            result = (extrema[i][0], extrema[i + 1][0], extrema[i][1], extrema[i + 1][1])

    # 結果の表示
    if result:
        print(f"最小差のペア: インデックス {result[0]} と {result[1]}, 値 {result[2]} と {result[3]}, 差 {min_difference}")
    else:
        print("極値が見つかりませんでした。")

    return result