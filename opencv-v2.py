import cv2
import numpy as np
from pathlib import Path
import shutil
import gc

def create_folders():
    """创建必要的文件夹结构"""
    folders = ['temp_frames', 'output']
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

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

def detect_movement_direction(img1, img2):
    """检测两帧之间的运动方向"""
    # 使用SIFT特征检测
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    if descriptors1 is None or descriptors2 is None:
        return None, None
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 10:  # 确保有足够的匹配点
        return None, None
    
    # 计算匹配点的平均移动方向
    movements = []
    for match in good_matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        movements.append((pt2[0] - pt1[0], pt2[1] - pt1[1]))
    
    avg_movement = np.mean(movements, axis=0)
    
    # 清理内存
    del keypoints1, keypoints2, descriptors1, descriptors2, matches, good_matches
    gc.collect()
    
    # 返回主要运动方向和平均位移
    return avg_movement, len(movements)

def blend_images(base_img, new_img, movement):
    """根据运动方向混合图像"""
    h, w = base_img.shape[:2]
    dx, dy = movement
    
    # 创建结果图像（与基准图像相同大小）
    result = base_img.copy()
    
    # 计算混合区域
    blend_width = w // 4  # 混合区域宽度
    blend_height = h // 4  # 混合区域高度
    
    # 根据主要运动方向决定如何混合
    if abs(dy) > abs(dx):  # 垂直运动为主
        if dy < 0:  # 向上运动
            # 取新图像的上半部分
            new_region = new_img[:h//2, :]
            # 创建渐变权重矩阵
            weight = np.linspace(1, 0, h//2)[:, np.newaxis]
            weight = np.repeat(weight, w, axis=1)
            weight = np.stack([weight] * 3, axis=2)
            
            # 混合图像
            result[:h//2, :] = (new_region * weight + 
                               result[:h//2, :] * (1 - weight)).astype(np.uint8)
        else:  # 向下运动
            new_region = new_img[h//2:, :]
            weight = np.linspace(0, 1, h//2)[:, np.newaxis]
            weight = np.repeat(weight, w, axis=1)
            weight = np.stack([weight] * 3, axis=2)
            
            result[h//2:, :] = (new_region * weight + 
                               result[h//2:, :] * (1 - weight)).astype(np.uint8)
    else:  # 水平运动为主
        if dx < 0:  # 向左运动
            new_region = new_img[:, :w//2]
            weight = np.linspace(1, 0, w//2)[np.newaxis, :]
            weight = np.repeat(weight, h, axis=0)
            weight = np.stack([weight] * 3, axis=2)
            
            result[:, :w//2] = (new_region * weight + 
                               result[:, :w//2] * (1 - weight)).astype(np.uint8)
        else:  # 向右运动
            new_region = new_img[:, w//2:]
            weight = np.linspace(0, 1, w//2)[np.newaxis, :]
            weight = np.repeat(weight, h, axis=0)
            weight = np.stack([weight] * 3, axis=2)
            
            result[:, w//2:] = (new_region * weight + 
                               result[:, w//2:] * (1 - weight)).astype(np.uint8)
    
    return result

def stitch_frames_with_checkpoints(total_frames, checkpoint_interval=5):
    """分批处理所有帧"""
    result = None
    try:
        result = cv2.imread(f'temp_frames/frame_0.jpg')
        if result is None:
            raise ValueError("无法读取第一帧")
        
        print("\n开始拼接处理...")
        last_checkpoint = 0
        checkpoint_count = 0
        
        cv2.imwrite(f'temp_frames/checkpoint_{checkpoint_count}.jpg', result)
        
        for i in range(1, total_frames):
            try:
                next_frame = cv2.imread(f'temp_frames/frame_{i}.jpg')
                if next_frame is None:
                    print(f"\nWarning: Could not read frame {i}")
                    continue
                
                print(f"\r拼接进度: {i}/{total_frames-1}", end="", flush=True)
                
                # 检测运动方向
                movement, match_count = detect_movement_direction(result, next_frame)
                
                if movement is not None and match_count >= 10:
                    # 根据运动方向混合图像
                    result = blend_images(result, next_frame, movement)
                else:
                    print(f"\nWarning: Failed to detect movement for frame {i}")
                
                # 保存检查点
                if i - last_checkpoint >= checkpoint_interval:
                    cv2.imwrite(f'temp_frames/checkpoint_{checkpoint_count + 1}.jpg', result)
                    checkpoint_count += 1
                    last_checkpoint = i
                
                del next_frame
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing frame {i}: {str(e)}")
                continue
        
        if result is not None:
            print("\n拼接处理完成")
            cv2.imwrite('output/panorama.jpg', result)
        
    except Exception as e:
        print(f"\nError in stitching: {str(e)}")
        if result is not None:
            cv2.imwrite('output/panorama_error.jpg', result)
    
    finally:
        # 清理中间文件和内存
        try:
            for i in range(checkpoint_count + 1):
                checkpoint_file = f'temp_frames/checkpoint_{i}.jpg'
                Path(checkpoint_file).unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup checkpoints: {e}")
        
        del result
        gc.collect()

def cleanup():
    """清理临时文件"""
    try:
        shutil.rmtree('temp_frames', ignore_errors=True)
    except Exception as e:
        print(f"清理临时文件时出错: {str(e)}")

def main(video_path, start_frame, frame_interval):
    """主函数"""
    try:
        create_folders()
        
        print("正在提取帧...")
        total_frames = extract_frames(video_path, start_frame, frame_interval)
        
        print("正在拼接全景图...")
        stitch_frames_with_checkpoints(total_frames)
        
        print("处理完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        cleanup()
        gc.collect()

if __name__ == "__main__":
    video_path = "video/7.mp4"  # 替换为你的视频路径
    start_frame = 30  # 从第30帧开始
    frame_interval = 20  # 每20帧取一帧
    main(video_path, start_frame, frame_interval)