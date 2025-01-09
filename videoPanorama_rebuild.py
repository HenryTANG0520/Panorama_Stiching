# videoPanorama.py

import sys
import cv2
import numpy as np
import math
import gc
import psutil
import time
import os
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

# ------------------------ 核心算法与函数 ------------------------

def get_memory_usage():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class Panorama:
    def __init__(self, initial_frame):
        """初始化全景图系统"""
        self.frame_h, self.frame_w = initial_frame.shape[:2]
        
        # 设置初始画布尺寸
        margin = 100
        self.canvas = np.zeros((self.frame_h + 2*margin, self.frame_w + 2*margin, 3), dtype=np.uint8)
        
        # 初始化位置信息
        self.current_x = margin
        self.current_y = margin
        self.min_x = margin
        self.max_x = margin + self.frame_w
        self.min_y = margin
        self.max_y = margin + self.frame_h
        
        # 放置第一帧
        self.canvas[margin:margin+self.frame_h, margin:margin+self.frame_w] = initial_frame
        
        # 设置最大画布尺寸（例如，宽度和高度不超过8000像素）
        self.max_width = 8000
        self.max_height = 8000

    def create_alpha_mask(self, dx, dy):
        """创建带有最小混合区域的alpha遮罩"""
        # 设置最小混合宽度为图像尺寸的5%
        min_blend_width = min(self.frame_w, self.frame_h) // 100
        
        alpha = np.ones((self.frame_h, self.frame_w), dtype=np.float16)
        
        if abs(dx) > abs(dy):  # 主要是水平移动
            blend_width = max(int(abs(dx)), min_blend_width)  # 确保最小宽度
            if dx > 0:  # 向右移动
                alpha[:, :blend_width] = np.linspace(0, 1, blend_width, dtype=np.float16)
            else:  # 向左移动
                alpha[:, -blend_width:] = np.linspace(1, 0, blend_width, dtype=np.float16)
        else:  # 主要是垂直移动
            blend_width = max(int(abs(dy)), min_blend_width)  # 确保最小宽度
            if dy > 0:  # 向下移动
                alpha[:blend_width, :] = np.linspace(0, 1, blend_width, dtype=np.float16)[:, np.newaxis]
            else:  # 向上移动
                alpha[-blend_width:, :] = np.linspace(1, 0, blend_width, dtype=np.float16)[:, np.newaxis]
        
        return np.stack([alpha] * 3, axis=2).astype(np.float16)
    
    def expand_canvas_if_needed(self, new_x, new_y):
        """在需要时扩展画布"""
        need_expand = False
        pad_left = pad_right = pad_top = pad_bottom = 0
        
        if new_x < 0:
            pad_left = abs(new_x)
            need_expand = True
        if new_x + self.frame_w > self.canvas.shape[1]:
            pad_right = new_x + self.frame_w - self.canvas.shape[1]
            need_expand = True
        if new_y < 0:
            pad_top = abs(new_y)
            need_expand = True
        if new_y + self.frame_h > self.canvas.shape[0]:
            pad_bottom = new_y + self.frame_h - self.canvas.shape[0]
            need_expand = True
            
        if need_expand:
            new_h = self.canvas.shape[0] + pad_top + pad_bottom
            new_w = self.canvas.shape[1] + pad_left + pad_right
            new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            
            y_start = pad_top
            x_start = pad_left
            new_canvas[y_start:y_start+self.canvas.shape[0],
                       x_start:x_start+self.canvas.shape[1]] = self.canvas
            
            self.current_x += pad_left
            self.current_y += pad_top
            self.min_x += pad_left
            self.max_x += pad_left
            self.min_y += pad_top
            self.max_y += pad_top
            
            self.canvas = new_canvas
            return pad_left, pad_top
            
        return 0, 0
    
    def add_frame(self, frame, angle, magnitude):
        """添加新帧到全景图"""
        dx = magnitude * math.cos(math.radians(angle + 90))
        dy = magnitude * math.sin(math.radians(angle - 90))
        
        new_x = int(self.current_x + dx)
        new_y = int(self.current_y + dy)
        
        offset_x, offset_y = self.expand_canvas_if_needed(new_x, new_y)
        new_x += offset_x
        new_y += offset_y
        
        roi = self.canvas[new_y:new_y+self.frame_h, new_x:new_x+self.frame_w]
        
        alpha = self.create_alpha_mask(dx, dy)
        
        result = (1 - alpha) * roi + alpha * frame
        
        self.canvas[new_y:new_y+self.frame_h, new_x:new_x+self.frame_w] = result.astype(np.uint8)
        
        self.current_x = new_x
        self.current_y = new_y
        self.min_x = min(self.min_x, new_x)
        self.max_x = max(self.max_x, new_x + self.frame_w)
        self.min_y = min(self.min_y, new_y)
        self.max_y = max(self.max_y, new_y + self.frame_h)
        
        current_width = self.max_x - self.min_x
        current_height = self.max_y - self.min_y
        if current_width > self.max_width or current_height > self.max_height:
            scale = min(self.max_width / current_width, self.max_height / current_height)
            self.canvas = cv2.resize(self.get_result(), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            self.frame_h, self.frame_w = self.canvas.shape[:2]
            self.current_x = self.canvas.shape[1] // 2
            self.current_y = self.canvas.shape[0] // 2
            self.min_x = 0
            self.max_x = self.canvas.shape[1]
            self.min_y = 0
            self.max_y = self.canvas.shape[0]
    
    def get_result(self):
        """获取最终结果"""
        return self.canvas[self.min_y:self.max_y, self.min_x:self.max_x]


def calculate_movement(img1, img2):
    """计算两帧之间的运动方向和幅度"""
    h, w = img1.shape[:2]
    
    orb = cv2.ORB_create(nfeatures=500)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    print(f"\n== 调试信息 ==")
    print(f"图像尺寸: {w}x{h}")
    print(f"检测到的特征点数量: 帧1={len(keypoints1)}, 帧2={len(keypoints2)}")
    
    if descriptors1 is None or descriptors2 is None:
        print("未检测到特征点")
        return None, None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"良好匹配点数量: {len(good_matches)}")
    
    if len(good_matches) < 10:
        print("匹配点数量不足")
        return None, None
    
    movements = []
    for match in good_matches:
        pt1 = np.array(keypoints1[match.queryIdx].pt)
        pt2 = np.array(keypoints2[match.trainIdx].pt)
        movement = pt2 - pt1
        movements.append(movement)
    
    mean_movement = np.mean(movements, axis=0)
    dx, dy = mean_movement
    
    std_dev = np.std(movements, axis=0)
    print(f"平均移动: dx={dx:.2f}, dy={dy:.2f}")
    print(f"移动标准差: dx_std={std_dev[0]:.2f}, dy_std={std_dev[1]:.2f}")
    
    angle = math.degrees(math.atan2(-dy, dx))  # 使用 -dy，因为图像坐标系 y 轴向下
    angle = (angle + 90) % 360
    
    magnitude = math.sqrt(dx*dx + dy*dy)
    
    is_horizontal = abs(dx) > abs(dy)
    if is_horizontal:
        magnitude = magnitude * 1.5  # 横向移动时增加magnitude
    
    move_direction = "横向" if is_horizontal else "纵向"
    print(f"主要移动方向: {move_direction}")
    print(f"移动角度: {angle:.2f}°")
    print(f"移动幅度: {magnitude:.2f}")
    print("================\n")
    
    return angle, magnitude


def frame_generator(cap, frame_interval=1, resize_scale=1.0):
    """生成器函数，按指定间隔生成帧"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
        yield frame
        for _ in range(frame_interval - 1):
            if not cap.grab():
                return


def resize_to_fit(image, max_width, max_height):
    """Resize image to fit within max_width and max_height, keeping aspect ratio"""
    height, width = image.shape[:2]
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale = min(width_ratio, height_ratio, 1.0)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    return image

# ------------------------ 播放视频的Worker ------------------------

class PlaybackWorker(QObject):
    """
    后台线程使用的Worker，用于实时播放原始视频到小窗口。
    """
    frame_signal = pyqtSignal(np.ndarray)  # 发送视频帧到小窗口
    error_signal = pyqtSignal(str)         # 发送错误信息

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit("无法打开视频文件进行播放")
            return

        # 获取视频的实际FPS，如果可用
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # 默认FPS
        frame_interval = 1 / video_fps

        while self.is_running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕，停止播放
                break

            self.frame_signal.emit(frame)

            # 控制播放速度
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

# ------------------------ 处理视频的Worker ------------------------

class ProcessingWorker(QObject):
    """
    后台线程使用的Worker，用于执行视频处理逻辑，并生成全景图。
    """
    progress_signal = pyqtSignal(str)                # 用于输出进度/日志
    result_signal = pyqtSignal(np.ndarray)           # 中间结果（用于实时更新大窗口）
    error_signal = pyqtSignal(str)                   # 出错时的信号

    # finished_signal 包含:
    #   - result: 最终全景图
    #   - frame_count: 总处理帧数
    #   - total_time: 总处理时间
    #   - max_memory: 最大内存使用
    #   - avg_memory: 平均内存使用
    finished_signal = pyqtSignal(np.ndarray, int, float, float, float)

    def __init__(self, video_path, start_frame=0, frame_interval=1, i=1, resize_scale=0.5, display_every=10):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.frame_interval = frame_interval
        self.i = i
        self.resize_scale = resize_scale
        self.display_every = display_every
        self.is_running = True

    def run(self):
        """
        执行视频处理逻辑，生成全景图，并发送进度和结果信号。
        """
        start_time = time.time()
        max_memory = 0
        memory_readings = []

        # 创建输出目录
        Path('output').mkdir(exist_ok=True)

        # 打开视频
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit("无法打开视频文件进行处理")
            return

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            frames = frame_generator(cap, self.frame_interval, self.resize_scale)

            # 读取第一帧
            try:
                prev_frame = next(frames)
            except StopIteration:
                self.error_signal.emit("无法读取第一帧")
                cap.release()
                return

            # 初始化全景图
            panorama = Panorama(prev_frame)
            frame_count = 1

            for curr_frame in frames:
                if not self.is_running:
                    self.progress_signal.emit("用户终止处理")
                    break

                # 计算运动
                angle, magnitude = calculate_movement(prev_frame, curr_frame)
                if angle is None or magnitude is None:
                    self.progress_signal.emit(f"无法计算帧间运动，跳过第 {frame_count} 帧")
                    prev_frame = curr_frame
                    frame_count += 1
                    continue

                # 添加到全景图
                try:
                    panorama.add_frame(curr_frame, angle, magnitude)
                    self.progress_signal.emit(
                        f"处理第 {frame_count} 帧 - 方向: {angle:.1f}°, 幅度: {magnitude:.1f}"
                    )

                    # 每隔 display_every 帧，发送当前拼接结果
                    if frame_count % self.display_every == 0:
                        current_result = panorama.get_result()
                        self.result_signal.emit(current_result)

                except Exception as e:
                    self.error_signal.emit(f"处理帧时出错: {str(e)}")
                    break

                # 监控内存使用
                current_memory = get_memory_usage()
                memory_readings.append(current_memory)
                max_memory = max(max_memory, current_memory)

                prev_frame = curr_frame
                frame_count += 1

                # 定期清理内存
                if frame_count % 100 == 0:
                    gc.collect()

            end_time = time.time()
            result = panorama.get_result()
            output_path = f'output/panorama_{self.i}.jpg'
            cv2.imwrite(output_path, result)
            self.progress_signal.emit(f"处理完成，共处理 {frame_count} 帧，结果保存至 {output_path}")

            # 性能统计
            total_time = end_time - start_time
            avg_memory = sum(memory_readings) / len(memory_readings) if memory_readings else 0

            # 发送完成信号
            self.finished_signal.emit(result, frame_count, total_time, max_memory, avg_memory)

            cap.release()
            gc.collect()

        except Exception as e:
            self.error_signal.emit(f"处理视频时发生异常: {str(e)}")

# ------------------------ PyQt6 主窗口 ------------------------

class EmittingStream(QtCore.QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(str(text))
    def flush(self):
        pass

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # 加载 UI
        uic.loadUi('videoPanorama_rebuild.ui', self)

        # 获取UI元素
        self.select_button = self.findChild(QtWidgets.QPushButton, 'select_button')
        self.start_button = self.findChild(QtWidgets.QPushButton, 'start_button')
        self.switch_button = self.findChild(QtWidgets.QPushButton, 'switch_button')
        self.console_window = self.findChild(QtWidgets.QTextBrowser, 'console_window')
        self.small_graphic_window = self.findChild(QtWidgets.QGraphicsView, 'small_graphic_window')
        self.big_graphic_window = self.findChild(QtWidgets.QGraphicsView, 'big_graphic_window')

        # 初始化场景
        self.small_scene = QtWidgets.QGraphicsScene()
        self.big_scene = QtWidgets.QGraphicsScene()
        self.small_graphic_window.setScene(self.small_scene)
        self.big_graphic_window.setScene(self.big_scene)

        # 连接按钮
        self.select_button.clicked.connect(self.select_video)
        self.start_button.clicked.connect(self.start_processing)
        self.switch_button.clicked.connect(self.switch_windows)

        # 重定向print到console_window
        sys.stdout = EmittingStream()
        sys.stdout.text_written.connect(self.write_console)
        sys.stderr = EmittingStream()
        sys.stderr.text_written.connect(self.write_console)

        self.video_path = None
        self.current_display = 'big'  # 初始显示状态
        self.switch_count = 0        # 切换计数

        # 线程和工作者
        self.playback_thread = None
        self.playback_worker = None
        self.processing_thread = None
        self.processing_worker = None

    def write_console(self, text):
        self.console_window.append(text)

    def select_video(self):
        try:
            options = QFileDialog.Option.ReadOnly
            file_name, _ = QFileDialog.getOpenFileName(
                self, 
                "选择视频文件", 
                "", 
                "视频文件 (*.mp4 *.avi *.mov *.mkv)", 
                options=options
            )
            if file_name:
                # 如果之前有播放或处理线程，先停止它们
                self.stop_playback()
                self.stop_processing()

                self.video_path = file_name
                self.console_window.append(f"已选择视频: {self.video_path}")

                # 显示第一帧在 small_graphic_window
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame_rgb.shape
                        qimg = QtGui.QImage(frame_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888).copy()
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        self.small_scene.clear()
                        self.small_scene.addPixmap(pixmap)
                    else:
                        self.console_window.append("无法读取视频的第一帧。")
                        QMessageBox.warning(self, "警告", "无法读取视频的第一帧。")
                    cap.release()
                else:
                    QMessageBox.critical(self, "错误", "无法打开选择的视频文件")
        except Exception as e:
            err = f"选择视频时出错: {str(e)}"
            self.console_window.append(err)
            QMessageBox.critical(self, "错误", err)

    def start_processing(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先选择一个视频文件")
            return

        # 禁用按钮以防止重复点击
        self.select_button.setEnabled(False)
        self.start_button.setEnabled(False)

        # 启动处理线程
        self.processing_worker = ProcessingWorker(
            video_path=self.video_path,
            start_frame=1,       # 可根据需求修改
            frame_interval=2,   # 可根据需求修改
            i=8,                 # 可根据需求修改
            resize_scale=0.5,    # 可根据需求修改
            display_every=1     # 可根据需求修改
        )
        self.processing_thread = QThread()
        self.processing_worker.moveToThread(self.processing_thread)

        # 连接信号
        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.progress_signal.connect(self.update_console)
        self.processing_worker.result_signal.connect(self.display_big_image)
        self.processing_worker.error_signal.connect(self.processing_error)
        self.processing_worker.finished_signal.connect(self.processing_finished)

        # 线程结束后自动清理
        self.processing_worker.finished_signal.connect(self.processing_thread.quit)
        self.processing_worker.finished_signal.connect(self.processing_worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)

        self.processing_thread.start()
        self.console_window.append("已启动视频处理任务")

        # 启动播放线程
        self.playback_worker = PlaybackWorker(
            video_path=self.video_path
        )
        self.playback_thread = QThread()
        self.playback_worker.moveToThread(self.playback_thread)

        # 连接信号
        self.playback_thread.started.connect(self.playback_worker.run)
        self.playback_worker.frame_signal.connect(self.display_small_image)
        self.playback_worker.error_signal.connect(self.playback_error)

        # 线程结束后自动清理
        self.playback_thread.finished.connect(self.playback_worker.deleteLater)
        self.playback_thread.finished.connect(self.playback_thread.deleteLater)

        self.playback_thread.start()
        self.console_window.append("已启动原始视频播放任务")

    def display_small_image(self, frame):
        """在小窗口显示原视频帧，适应窗口大小"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 获取 QGraphicsView 的大小
            view_size = self.small_graphic_window.viewport().size()
            # 调整图像以适应窗口
            scaled_frame = resize_to_fit(frame_rgb, view_size.width(), view_size.height())
            h, w, c = scaled_frame.shape
            qimg = QtGui.QImage(scaled_frame.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888).copy()
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.small_scene.clear()
            self.small_scene.addPixmap(pixmap)
        except Exception as e:
            self.console_window.append(f"显示小窗口时出错: {str(e)}")

    def display_big_image(self, image):
        """在大窗口显示拼接结果，适应窗口大小"""
        try:
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 获取 QGraphicsView 的大小
            view_size = self.big_graphic_window.viewport().size()
            # 调整图像以适应窗口
            scaled_frame = resize_to_fit(frame_rgb, view_size.width(), view_size.height())
            h, w, c = scaled_frame.shape
            qimg = QtGui.QImage(scaled_frame.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888).copy()
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.big_scene.clear()
            self.big_scene.addPixmap(pixmap)
        except Exception as e:
            self.console_window.append(f"显示大窗口时出错: {str(e)}")

    def update_console(self, message):
        self.console_window.append(message)

    def processing_finished(self, result, frame_count, total_time, max_memory, avg_memory):
        """
        处理完成后，将性能统计与最终结果打印到 console_window，
        并重新启用按钮。
        """
        self.console_window.append("\n=== 处理完成 ===")
        self.console_window.append(f"输出全景图尺寸: {result.shape[1]}x{result.shape[0]}")
        self.console_window.append(f"共处理 {frame_count} 帧")
        
        # 性能统计（保持您原先的输出格式）
        self.console_window.append("\n=== 性能统计 ===")
        self.console_window.append(f"总处理时间: {total_time:.2f} 秒")
        if frame_count > 0:
            self.console_window.append(f"平均每帧处理时间: {total_time/frame_count:.2f} 秒")
        else:
            self.console_window.append("平均每帧处理时间: N/A")

        self.console_window.append(f"最大内存使用: {max_memory:.2f} MB")
        if avg_memory > 0:
            self.console_window.append(f"平均内存使用: {avg_memory:.2f} MB")
        else:
            self.console_window.append("平均内存使用: N/A")

        self.console_window.append(f"总处理帧数: {frame_count}")
        self.console_window.append("================")

        # 在大窗显示最终结果
        self.display_big_image(result)

        # 重新启用按钮
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def processing_error(self, error_message):
        self.console_window.append(f"\n处理错误: {error_message}")
        QMessageBox.critical(self, "处理错误", error_message)
        self.stop_processing()
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def playback_error(self, error_message):
        self.console_window.append(f"\n播放错误: {error_message}")
        QMessageBox.critical(self, "播放错误", error_message)
        self.stop_playback()
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def switch_windows(self):
        """切换小窗口和大窗口的显示内容"""
        try:
            self.switch_count += 1
            if self.switch_count % 2 == 1:
                # 奇数次切换：大窗口显示原始视频，小窗口显示生成的视频
                # 断开当前连接
                if self.playback_worker:
                    try:
                        self.playback_worker.frame_signal.disconnect(self.display_small_image)
                        self.playback_worker.frame_signal.connect(self.display_big_image)
                    except:
                        pass
                if self.processing_worker:
                    try:
                        self.processing_worker.result_signal.disconnect(self.display_big_image)
                        self.processing_worker.result_signal.connect(self.display_small_image)
                    except:
                        pass
                self.current_display = 'small_big'
                self.console_window.append("已切换：大窗口显示原始视频，小窗口显示生成视频")
            else:
                # 偶数次切换：大窗口显示生成视频，小窗口显示原始视频
                if self.playback_worker:
                    try:
                        self.playback_worker.frame_signal.disconnect(self.display_big_image)
                        self.playback_worker.frame_signal.connect(self.display_small_image)
                    except:
                        pass
                if self.processing_worker:
                    try:
                        self.processing_worker.result_signal.disconnect(self.display_small_image)
                        self.processing_worker.result_signal.connect(self.display_big_image)
                    except:
                        pass
                self.current_display = 'big_small'
                self.console_window.append("已切换：大窗口显示生成视频，小窗口显示原始视频")
        except Exception as e:
            self.console_window.append(f"切换窗口时出错: {str(e)}")

    def stop_playback(self):
        """停止播放线程"""
        if self.playback_worker and self.playback_thread:
            self.playback_worker.is_running = False
            self.playback_thread.quit()
            self.playback_thread.wait()
            self.playback_worker = None
            self.playback_thread = None
            self.console_window.append("已停止原始视频播放任务")

    def stop_processing(self):
        """停止处理线程"""
        if self.processing_worker and self.processing_thread:
            self.processing_worker.is_running = False
            self.processing_thread.quit()
            self.processing_thread.wait()
            self.processing_worker = None
            self.processing_thread = None
            self.console_window.append("已停止视频处理任务")

    def closeEvent(self, event):
        """关闭窗口时，确保所有线程安全退出"""
        self.stop_playback()
        self.stop_processing()
        event.accept()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
