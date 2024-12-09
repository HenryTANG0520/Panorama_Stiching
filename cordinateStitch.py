import cv2
import numpy as np
from pathlib import Path
import shutil
import gc
import math

class SimplePanorama:
    def __init__(self, initial_frame):
        """初始化全景图系统"""
        self.frame_h, self.frame_w = initial_frame.shape[:2]
        
        # 创建初始画布（给一些边距以便扩展）
        margin = 100
        self.canvas = np.zeros((self.frame_h + 2*margin, self.frame_w + 2*margin, 3), dtype=np.uint8)
        
        # 当前位置（从中心开始）
        self.current_x = margin
        self.current_y = margin
        
        # 记录已使用区域的边界
        self.min_x = margin
        self.max_x = margin + self.frame_w
        self.min_y = margin
        self.max_y = margin + self.frame_h
        
        # 放置第一帧
        self.canvas[margin:margin+self.frame_h, margin:margin+self.frame_w] = initial_frame
    
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
        dx = magnitude * math.cos(math.radians(angle + 90))  # 0度对应向上
        dy = -magnitude * math.sin(math.radians(angle + 90))  # 图像坐标系y轴向下为正
        
        new_x = int(self.current_x + dx)
        new_y = int(self.current_y + dy)
        
        # 如果需要则扩展画布
        offset_x, offset_y = self.expand_canvas_if_needed(new_x, new_y)
        new_x += offset_x
        new_y += offset_y
        
        # 直接放置新帧
        self.canvas[new_y:new_y+self.frame_h, new_x:new_x+self.frame_w] = frame
        
        # 更新当前位置和边界
        self.current_x = new_x
        self.current_y = new_y
        self.min_x = min(self.min_x, new_x)
        self.max_x = max(self.max_x, new_x + self.frame_w)
        self.min_y = min(self.min_y, new_y)
        self.max_y = max(self.max_y, new_y + self.frame_h)
    
    def get_result(self):
        """获取最终结果"""
        # 只返回包含图像的区域
        return self.canvas[self.min_y:self.max_y, self.min_x:self.max_x]

def calculate_movement(img1, img2):
    """计算两帧之间的运动方向和幅度"""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    if descriptors1 is None or descriptors2 is None:
        return None, None
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 10:
        return None, None
        
    # 计算平均移动向量
    movements = []
    for match in good_matches:
        pt1 = np.array(keypoints1[match.queryIdx].pt)
        pt2 = np.array(keypoints2[match.trainIdx].pt)
        movement = pt2 - pt1
        movements.append(movement)
    
    mean_movement = np.mean(movements, axis=0)
    dx, dy = mean_movement
    
    # 计算角度（0度为正上方，顺时针旋转）
    angle = math.degrees(math.atan2(-dy, dx))  # 使用-dy是因为图像坐标系y轴向下
    angle = (angle + 90) % 360
    
    # 计算移动幅度
    magnitude = math.sqrt(dx*dx + dy*dy)
    
    return angle, magnitude

def process_video(video_path, start_frame=0, frame_interval=1):
    """处理视频"""
    # 创建输出目录
    Path('output').mkdir(exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    try:
        # 跳到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 读取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("无法读取第一帧")
            
        # 初始化全景图
        panorama = SimplePanorama(prev_frame)
        frame_count = 1
        
        while True:
            # 跳过指定数量的帧
            for _ in range(frame_interval - 1):
                ret = cap.grab()
                if not ret:
                    break
            
            # 读取当前帧
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            # 计算运动
            angle, magnitude = calculate_movement(prev_frame, curr_frame)
            
            if angle is not None and magnitude > 5:  # 只在有显著运动时处理
                # 添加到全景图
                panorama.add_frame(curr_frame, angle, magnitude)
                print(f"\r处理第 {frame_count} 帧 - 方向: {angle:.1f}°, 幅度: {magnitude:.1f}", end="")
                
            prev_frame = curr_frame.copy()
            frame_count += 1
            
        # 保存结果
        result = panorama.get_result()
        cv2.imwrite('output/panorama.jpg', result)
        print(f"\n处理完成，共处理 {frame_count} 帧")
        
    finally:
        cap.release()
        gc.collect()

if __name__ == "__main__":
    video_path = "video/4.mp4"  # 替换为实际的视频路径
    process_video(video_path, start_frame=1, frame_interval=5)