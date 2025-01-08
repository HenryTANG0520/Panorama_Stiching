# videoPanorama.py

import sys
import cv2
import numpy as np
from pathlib import Path
import time
import psutil
import os
import gc
import math
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

# 定义一个类来重定向print到QTextBrowser
class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

# 定义视频处理的Worker
class VideoProcessor(QObject):
    # 定义信号
    progress = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray, float, float, int)
    error = pyqtSignal(str)
    update_display = pyqtSignal(np.ndarray)

    def __init__(self, video_path, start_frame, frame_interval, resize_scale, display_every):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.frame_interval = frame_interval
        self.resize_scale = resize_scale
        self.display_every = display_every
        self.is_running = True

    def process_video(self):
        try:
            # 创建输出目录
            Path('output').mkdir(exist_ok=True)

            # 初始化性能监控变量
            start_time = time.time()
            max_memory = 0
            memory_readings = []

            # 打开视频
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("无法打开视频文件")
                return

            # 跳到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            # 创建生成器
            frames = self.frame_generator(cap, self.frame_interval, self.resize_scale)

            # 读取第一帧
            try:
                prev_frame = next(frames)
            except StopIteration:
                self.error.emit("无法读取第一帧")
                cap.release()
                return

            # 初始化全景图
            panorama = Panorama(prev_frame)
            frame_count = 1

            # 发送初始状态到显示
            self.update_display.emit(panorama.get_result())

            for curr_frame in frames:
                if not self.is_running:
                    self.progress.emit("用户终止处理")
                    break

                # 计算运动
                angle, magnitude = calculate_movement(prev_frame, curr_frame)

                # 检查计算结果是否有效
                if angle is None or magnitude is None:
                    self.progress.emit(f"无法计算帧间运动，跳过第 {frame_count} 帧")
                    prev_frame = curr_frame
                    frame_count += 1
                    continue

                # 添加到全景图
                try:
                    panorama.add_frame(curr_frame, angle, magnitude)
                    self.progress.emit(f"处理第 {frame_count} 帧 - 方向: {angle:.1f}°, 幅度: {magnitude:.1f}")

                    # 获取当前结果并发送更新信号（每 display_every 帧）
                    if frame_count % self.display_every == 0:
                        current_result = panorama.get_result()
                        self.update_display.emit(current_result)

                except Exception as e:
                    self.error.emit(f"处理帧时出错: {str(e)}")
                    break

                # 监控内存使用
                current_memory = self.get_memory_usage()
                memory_readings.append(current_memory)
                max_memory = max(max_memory, current_memory)

                prev_frame = curr_frame
                frame_count += 1

                # 定期清理内存
                if frame_count % 100 == 0:
                    gc.collect()

            # 保存结果
            end_time = time.time()
            result = panorama.get_result()
            output_path = f'output/panorama_output.jpg'
            cv2.imwrite(output_path, result)
            self.progress.emit(f"处理完成，共处理 {frame_count} 帧，结果保存至 {output_path}")

            # 计算性能统计
            total_time = end_time - start_time
            avg_memory = sum(memory_readings) / len(memory_readings) if memory_readings else 0

            # 发送完成信号
            self.finished.emit(result, total_time, avg_memory, frame_count)

            cap.release()
            gc.collect()

        except Exception as e:
            self.error.emit(f"处理视频时发生异常: {str(e)}")

    def stop(self):
        self.is_running = False

    def frame_generator(self, cap, frame_interval=1, resize_scale=1.0):
        """生成器函数，按指定间隔生成帧"""
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
            yield frame
            for _ in range(frame_interval - 1):
                if not cap.grab():
                    return

    def get_memory_usage(self):
        """获取当前进程的内存使用量（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

# 您现有的Panorama类和其他辅助函数
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
        
        # 检查是否需要扩展
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
            # 创建新画布
            new_h = self.canvas.shape[0] + pad_top + pad_bottom
            new_w = self.canvas.shape[1] + pad_left + pad_right
            new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            
            # 复制原画布内容到新位置
            y_start = pad_top
            x_start = pad_left
            new_canvas[y_start:y_start+self.canvas.shape[0], 
                      x_start:x_start+self.canvas.shape[1]] = self.canvas
            
            # 更新坐标
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
        # 计算新位置
        dx = magnitude * math.cos(math.radians(angle + 90))
        dy = magnitude * math.sin(math.radians(angle - 90))
        
        new_x = int(self.current_x + dx)
        new_y = int(self.current_y + dy)
        
        # 如果需要则扩展画布
        offset_x, offset_y = self.expand_canvas_if_needed(new_x, new_y)
        new_x += offset_x
        new_y += offset_y
        
        # 计算重叠区域
        roi = self.canvas[new_y:new_y+self.frame_h, new_x:new_x+self.frame_w]
        
        # 创建alpha遮罩
        alpha = self.create_alpha_mask(dx, dy)
        
        # 混合帧
        result = (1 - alpha) * roi + alpha * frame
        
        # 更新画布
        self.canvas[new_y:new_y+self.frame_h, new_x:new_x+self.frame_w] = result.astype(np.uint8)
        
        # 更新位置信息
        self.current_x = new_x
        self.current_y = new_y
        self.min_x = min(self.min_x, new_x)
        self.max_x = max(self.max_x, new_x + self.frame_w)
        self.min_y = min(self.min_y, new_y)
        self.max_y = max(self.max_y, new_y + self.frame_h)
        
        # 检查画布尺寸并进行缩放
        current_width = self.max_x - self.min_x
        current_height = self.max_y - self.min_y
        if current_width > self.max_width or current_height > self.max_height:
            scale = min(self.max_width / current_width, self.max_height / current_height)
            self.canvas = cv2.resize(self.get_result(), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            # 更新位置信息
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
    # 获取图像尺寸用于调试信息
    h, w = img1.shape[:2]
    
    # 使用ORB
    orb = cv2.ORB_create(nfeatures=500)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    debug_info = f"\n== 调试信息 ==\n图像尺寸: {w}x{h}\n检测到的特征点数量: 帧1={len(keypoints1)}, 帧2={len(keypoints2)}\n"
    
    if descriptors1 is None or descriptors2 is None:
        debug_info += "未检测到特征点\n"
        print(debug_info)
        return None, None
    
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # 过滤匹配点
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    debug_info += f"良好匹配点数量: {len(good_matches)}\n"
    
    if len(good_matches) < 10:
        debug_info += "匹配点数量不足\n"
        print(debug_info)
        return None, None
    
    # 计算移动向量
    movements = []
    for match in good_matches:
        pt1 = np.array(keypoints1[match.queryIdx].pt)
        pt2 = np.array(keypoints2[match.trainIdx].pt)
        movement = pt2 - pt1
        movements.append(movement)
    
    mean_movement = np.mean(movements, axis=0)
    dx, dy = mean_movement
    
    # 计算移动统计信息
    std_dev = np.std(movements, axis=0)
    debug_info += f"平均移动: dx={dx:.2f}, dy={dy:.2f}\n移动标准差: dx_std={std_dev[0]:.2f}, dy_std={std_dev[1]:.2f}\n"
    
    # 计算角度（0度为正上方，顺时针旋转）
    angle = math.degrees(math.atan2(-dy, dx))  # 使用-dy是因为图像坐标系y轴向下
    angle = (angle + 90) % 360
    
    # 计算移动幅度
    magnitude = math.sqrt(dx*dx + dy*dy)
    
    # 判断主要移动方向并调整幅度
    is_horizontal = abs(dx) > abs(dy)
    if is_horizontal:
        magnitude = magnitude * 1.5  # 横向移动时增加magnitude以匹配纵向效果
    
    # 输出移动方向相关信息
    move_direction = "横向" if is_horizontal else "纵向"
    debug_info += f"主要移动方向: {move_direction}\n移动角度: {angle:.2f}°\n移动幅度: {magnitude:.2f}\n================\n"
    
    print(debug_info)
    
    return angle, magnitude

def get_memory_usage():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def resize_to_screen(image, max_width=1920, max_height=1080):
    """调整图像大小以适应屏幕，保持宽高比"""
    height, width = image.shape[:2]
    
    # 计算缩放比例
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale = min(width_ratio, height_ratio, 1.0)  # 不放大，只缩小
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, new_width, new_height
    return image, width, height

# 主窗口类
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('videoPanorama.ui', self)

        # 调试：列出所有子控件及其 objectName
        print("Loaded UI controls:")
        for widget in self.findChildren(QtWidgets.QWidget):
            print(f"{widget.__class__.__name__}: {widget.objectName()}")

        # 获取UI元素
        self.select_button = self.findChild(QtWidgets.QPushButton, 'select_button')
        self.start_button = self.findChild(QtWidgets.QPushButton, 'start_button')
        self.switch_button = self.findChild(QtWidgets.QPushButton, 'switch_button')
        self.console_window = self.findChild(QtWidgets.QTextBrowser, 'console_window')
        self.small_graphic_window = self.findChild(QtWidgets.QGraphicsView, 'small_graphic_window')
        self.big_graphic_window = self.findChild(QtWidgets.QGraphicsView, 'big_graphic_window')

        # 检查是否成功找到所有控件
        if not all([self.select_button, self.start_button, self.switch_button, self.console_window, self.small_graphic_window, self.big_graphic_window]):
            QMessageBox.critical(self, "错误", "无法找到所有UI控件。请检查UI文件中的objectName属性。")
            sys.exit(1)

        # 初始化图形场景
        self.small_scene = QtWidgets.QGraphicsScene()
        self.big_scene = QtWidgets.QGraphicsScene()
        self.small_graphic_window.setScene(self.small_scene)
        self.big_graphic_window.setScene(self.big_scene)

        # 设置初始视频路径
        self.video_path = None

        # 连接按钮信号
        self.select_button.clicked.connect(self.select_video)
        self.start_button.clicked.connect(self.start_processing)
        self.switch_button.clicked.connect(self.switch_windows)

        # 重定向print到console_window
        sys.stdout = EmittingStream()
        sys.stdout.text_written.connect(self.write_console)
        sys.stderr = EmittingStream()
        sys.stderr.text_written.connect(self.write_console)

        self.current_display = 'big'  # 当前显示的窗口

    def write_console(self, text):
        self.console_window.append(text)

    def select_video(self):
        try:
            # 设置文件对话框选项
            options = QFileDialog.Option.ReadOnly
            # 如果需要其他选项，可以组合使用，例如：
            # options |= QFileDialog.Option.DontUseNativeDialog

            # 打开文件对话框
            file_name, _ = QFileDialog.getOpenFileName(
                self, 
                "选择视频文件", 
                "", 
                "视频文件 (*.mp4 *.avi *.mov *.mkv)", 
                options=options
            )
            
            if file_name:
                self.video_path = file_name
                self.console_window.append(f"已选择视频: {self.video_path}")
                print(f"已选择视频: {self.video_path}")  # 调试输出

                # 显示原视频的第一帧
                cap = cv2.VideoCapture(self.video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print("成功读取第一帧")  # 调试输出
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = frame.shape
                        bytes_per_line = 3 * width
                        qimg = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qimg)
                        self.small_scene.clear()
                        self.small_scene.addPixmap(pixmap)
                        self.console_window.append("已显示视频的第一帧。")
                    else:
                        self.console_window.append("无法读取视频的第一帧。")
                        QMessageBox.warning(self, "警告", "无法读取视频的第一帧。")
                    cap.release()
                else:
                    QMessageBox.critical(self, "错误", "无法打开选择的视频文件")
        except Exception as e:
            # 将异常信息输出到控制台窗口
            error_message = f"选择视频时出错: {str(e)}"
            self.console_window.append(error_message)
            QMessageBox.critical(self, "错误", error_message)
            print(error_message)  # 同时打印到终端以便进一步调试

    def start_processing(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先选择一个视频文件")
            return

        # 禁用按钮以防止多次启动
        self.select_button.setEnabled(False)
        self.start_button.setEnabled(False)

        # 创建线程和处理对象
        self.thread = QThread()
        self.worker = VideoProcessor(
            video_path=self.video_path,
            start_frame=0,
            frame_interval=1,
            resize_scale=0.5,
            display_every=10
        )
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.thread.started.connect(self.worker.process_video)
        self.worker.progress.connect(self.update_console)
        self.worker.finished.connect(self.processing_finished)
        self.worker.error.connect(self.processing_error)
        self.worker.update_display.connect(self.update_big_graphic)

        # 连接线程结束后清理
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def update_console(self, message):
        self.console_window.append(message)

    def processing_finished(self, result, total_time, avg_memory, frame_count):
        self.console_window.append("视频处理完成。")
        self.display_image(result, self.big_scene)
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def processing_error(self, error_message):
        self.console_window.append(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)

    def update_big_graphic(self, image):
        self.display_image(image, self.big_scene)

    def display_image(self, image, scene):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimg = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scene.clear()
        scene.addPixmap(pixmap)

    def switch_windows(self):
        if self.current_display == 'big':
            # 将大窗口内容切换到小窗口
            self.small_scene.clear()
            items = self.big_scene.items()
            if items:
                pixmap = items[0].pixmap()
                self.small_scene.addPixmap(pixmap)
            self.big_scene.clear()
            self.current_display = 'small'
            self.console_window.append("已将大窗口内容切换到小窗口")
        else:
            # 将小窗口内容切换到大窗口
            self.big_scene.clear()
            items = self.small_scene.items()
            if items:
                pixmap = items[0].pixmap()
                self.big_scene.addPixmap(pixmap)
            self.small_scene.clear()
            self.current_display = 'big'
            self.console_window.append("已将小窗口内容切换到大窗口")

    def closeEvent(self, event):
        # 确保线程在关闭时停止
        try:
            if hasattr(self, 'worker') and self.worker.is_running:
                self.worker.stop()
                self.thread.quit()
                self.thread.wait()
        except:
            pass
        event.accept()

if __name__ == "__main__":
    # 忽略DeprecationWarning（如果需要）
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
