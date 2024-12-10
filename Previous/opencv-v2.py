import cv2
import numpy as np
from pathlib import Path
import shutil
import gc
import math

def cleanup():
    """清理临时文件"""
    try:
        shutil.rmtree('temp_frames', ignore_errors=True)
    except Exception as e:
        print(f"清理临时文件时出错: {str(e)}")
        
def create_folders():
    """创建必要的文件夹结构"""
    folders = [
        'temp_frames',
        'output'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
def calculate_movement_angle(img1, img2):
    """
    计算两帧之间的运动方向角度
    返回：(angle, magnitude, confidence)
    angle: 0-360度，0度为正上方，顺时针旋转
    magnitude: 移动幅度
    confidence: 置信度（好的特征点匹配数量）
    """
    # 创建SIFT检测器
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    if descriptors1 is None or descriptors2 is None:
        return None, None, 0
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 10:
        return None, None, 0
    
    # 计算所有匹配点的移动向量
    movements = []
    for match in good_matches:
        pt1 = np.array(keypoints1[match.queryIdx].pt)
        pt2 = np.array(keypoints2[match.trainIdx].pt)
        movement = pt2 - pt1
        movements.append(movement)
    
    movements = np.array(movements)
    
    # 计算平均移动向量
    mean_movement = np.mean(movements, axis=0)
    dx, dy = mean_movement
    
    # 计算角度（arctan2返回-π到π的弧度）
    angle = math.degrees(math.atan2(-dy, dx))  # 使用-dy是因为图像坐标系y轴向下
    # 将角度转换到0-360度范围，并使0度对应正上方
    angle = (angle + 90) % 360
    
    # 计算移动幅度
    magnitude = math.sqrt(dx*dx + dy*dy)
    
    # 清理内存
    del keypoints1, keypoints2, descriptors1, descriptors2, matches, good_matches
    gc.collect()
    
    return angle, magnitude, len(movements)

def get_sector_mask(img_shape, angle, sector_width=60):
    """
    创建扇形区域掩码
    angle: 运动方向角度
    sector_width: 扇区宽度（度）
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w//2, h//2)
    
    # 计算扇区的起始和结束角度
    start_angle = (angle - sector_width/2) % 360
    end_angle = (angle + sector_width/2) % 360
    
    # 创建从中心到边缘的渐变
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    gradient = dist_from_center / max_dist
    
    # 为每个像素计算角度
    angles = (np.degrees(np.arctan2(-(Y - center[1]), X - center[0])) + 90) % 360
    
    # 处理跨越0度的情况
    if start_angle > end_angle:
        mask[(angles >= start_angle) | (angles <= end_angle)] = 1
    else:
        mask[(angles >= start_angle) & (angles <= end_angle)] = 1
    
    # 应用渐变
    mask *= gradient
    
    return mask

def blend_with_direction(base_img, new_img, angle, magnitude):
    """根据运动方向角度混合图像"""
    if magnitude < 10:  # 移动太小，不处理
        return base_img
    
    # 创建扇形掩码
    mask = get_sector_mask(base_img.shape, angle)
    
    # 扩展掩码到3通道
    mask = np.stack([mask] * 3, axis=2)
    
    # 混合图像
    result = base_img.copy()
    result = (1 - mask) * result + mask * new_img
    
    return result.astype(np.uint8)

def stitch_frames_with_direction(total_frames):
    """基于方向的帧拼接"""
    panorama = None
    try:
        # 读取第一帧作为基准
        panorama = cv2.imread(f'temp_frames/frame_0.jpg')
        if panorama is None:
            raise ValueError("无法读取第一帧")
        
        print("\n开始拼接处理...")
        
        for i in range(1, total_frames):
            try:
                current_frame = cv2.imread(f'temp_frames/frame_{i}.jpg')
                if current_frame is None:
                    print(f"\nWarning: Could not read frame {i}")
                    continue
                
                print(f"\r拼接进度: {i}/{total_frames-1}", end="", flush=True)
                
                # 计算运动方向角度
                angle, magnitude, confidence = calculate_movement_angle(panorama, current_frame)
                
                if angle is not None and confidence >= 10:
                    # 根据运动方向混合图像
                    panorama = blend_with_direction(panorama, current_frame, angle, magnitude)
                    print(f"\n帧 {i} - 方向: {angle:.1f}°, 幅度: {magnitude:.1f}, 置信度: {confidence}")
                else:
                    print(f"\nWarning: Movement detection failed for frame {i}")
                
                del current_frame
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing frame {i}: {str(e)}")
                continue
        
        if panorama is not None:
            print("\n拼接处理完成")
            cv2.imwrite('output/panorama.jpg', panorama)
        
    except Exception as e:
        print(f"\nError in stitching: {str(e)}")
        if panorama is not None:
            cv2.imwrite('output/panorama_error.jpg', panorama)
    
    finally:
        del panorama
        gc.collect()

def extract_frames(video_path, start_frame, frame_interval):
    """从指定帧开始提取帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        frame_count = 0
        saved_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"总帧数: {total_frames}")
        print(f"从第 {start_frame} 帧开始，间隔 {frame_interval} 帧")
        print(f"预计处理帧数: {(total_frames - start_frame ) // frame_interval}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                cv2.imwrite(f'temp_frames/frame_{saved_count}.jpg', frame)
                print(f"\r提取帧进度: {saved_count + 1}/{(total_frames - start_frame) // frame_interval}", end="", flush=True)
                saved_count += 1
                
            frame_count += 1
            del frame
            gc.collect()
        
        print("\n帧提取完成")
        return saved_count
        
    finally:
        cap.release()
        gc.collect()

def main(video_path, start_frame, frame_interval):
    """主函数"""
    try:
        create_folders()
        
        print("正在提取帧...")
        total_frames = extract_frames(video_path, start_frame, frame_interval)
        
        print("正在拼接全景图...")
        stitch_frames_with_direction(total_frames)
        
        print("处理完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        cleanup()
        gc.collect()

if __name__ == "__main__":
    video_path = "video/7.mp4"  
    start_frame = 30  # 从第30帧开始
    frame_interval = 20  # 每20帧取一帧
    main(video_path, start_frame, frame_interval)