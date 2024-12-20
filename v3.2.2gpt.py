import cv2
import numpy as np
from pathlib import Path
import time
import psutil
import os
import gc
import math

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
    
    # 使用ORB替代SIFT
    orb = cv2.ORB_create(nfeatures=500)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    print(f"\n== 调试信息 ==")
    print(f"图像尺寸: {w}x{h}")
    print(f"检测到的特征点数量: 帧1={len(keypoints1)}, 帧2={len(keypoints2)}")
    
    if descriptors1 is None or descriptors2 is None:
        print("未检测到特征点")
        return None, None
    
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # 过滤匹配点
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    print(f"良好匹配点数量: {len(good_matches)}")
    
    if len(good_matches) < 10:
        print("匹配点数量不足")
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
    print(f"平均移动: dx={dx:.2f}, dy={dy:.2f}")
    print(f"移动标准差: dx_std={std_dev[0]:.2f}, dy_std={std_dev[1]:.2f}")
    
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
    print(f"主要移动方向: {move_direction}")
    print(f"移动角度: {angle:.2f}°")
    print(f"移动幅度: {magnitude:.2f}")
    print("================\n")
    
    return angle, magnitude

def get_memory_usage():
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

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

def process_video(video_path, start_frame=0, frame_interval=1, i=1, resize_scale=0.5, display_every=10):
    """处理视频"""
    # 创建输出目录
    Path('output').mkdir(exist_ok=True)
    
    # 初始化性能监控变量
    start_time = time.time()
    max_memory = 0
    memory_readings = []
    
    # 创建窗口
    cv2.namedWindow('Panorama Progress', cv2.WINDOW_NORMAL)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    try:
        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 创建生成器
        frames = frame_generator(cap, frame_interval, resize_scale)
        
        # 读取第一帧
        try:
            prev_frame = next(frames)
        except StopIteration:
            raise ValueError("无法读取第一帧")
        
        # 初始化全景图
        panorama = Panorama(prev_frame)
        frame_count = 1
        
        # 显示初始状态
        initial_result = panorama.get_result()
        display_result, win_width, win_height = resize_to_screen(initial_result)
        cv2.resizeWindow('Panorama Progress', win_width, win_height)
        cv2.imshow('Panorama Progress', display_result)
        cv2.waitKey(1)
        
        for curr_frame in frames:
            # 计算运动
            angle, magnitude = calculate_movement(prev_frame, curr_frame)
            
            # 检查计算结果是否有效
            if angle is None or magnitude is None:
                print("\n无法计算帧间运动，跳过当前帧")
                prev_frame = curr_frame
                frame_count += 1
                continue
                
            # 添加到全景图
            try:
                panorama.add_frame(curr_frame, angle, magnitude)
                print(f"\r处理第 {frame_count} 帧 - 方向: {angle:.1f}°, 幅度: {magnitude:.1f}", end="")
                
                # 获取当前结果并显示（每 display_every 帧）
                if frame_count % display_every == 0:
                    current_result = panorama.get_result()
                    display_result, win_width, win_height = resize_to_screen(current_result)
                    cv2.resizeWindow('Panorama Progress', win_width, win_height)
                    cv2.imshow('Panorama Progress', display_result)
                    
                    # 删除临时变量
                    del current_result, display_result, win_width, win_height
                
            except Exception as e:
                print(f"\n处理帧时出错: {str(e)}")
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
            
            # 检查是否按下 'q' 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户终止处理")
                break
        
        # 保存结果
        end_time = time.time()
        result = panorama.get_result()
        cv2.imwrite(f'output/panorama_{i}.jpg', result)
        print(f"\n处理完成，共处理 {frame_count} 帧")
        
        # 计算并显示性能统计
        total_time = end_time - start_time
        avg_memory = sum(memory_readings) / len(memory_readings) if memory_readings else 0

        # 显示最终结果
        cv2.namedWindow('Final Panorama', cv2.WINDOW_NORMAL)
        final_display, win_width, win_height = resize_to_screen(result)
        cv2.resizeWindow('Final Panorama', win_width, win_height)
        cv2.imshow('Final Panorama', final_display)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        
        # 性能统计
        print(f"\n\n=== 性能统计 ===")
        print(f"总处理时间: {total_time:.2f} 秒")
        print(f"平均每帧处理时间: {total_time/frame_count:.2f} 秒" if frame_count else "平均每帧处理时间: N/A")
        print(f"最大内存使用: {max_memory:.2f} MB")
        print(f"平均内存使用: {avg_memory:.2f} MB" if memory_readings else "平均内存使用: N/A")
        print(f"总处理帧数: {frame_count}")
        print("================")
        
    finally:
        cap.release()
        gc.collect()

if __name__ == "__main__":
    cv2.setNumThreads(16)  # 根据CPU核心数调整
    i = 8
    video_path = f"video/drone/{i}.mp4"  
    process_video(video_path, start_frame=1, frame_interval=10, i=i, resize_scale=0.5, display_every=10)
