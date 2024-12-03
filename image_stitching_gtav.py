import cv2
import numpy as np
from pathlib import Path
import shutil

def create_folders():
    """创建必要的文件夹结构"""
    folders = [
        'temp_frames/up',
        'temp_frames/down',
        'temp_frames/left',
        'temp_frames/right',
        'output/split',
        'output'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def extract_and_split_frames(video_path, frame_interval):
    """提取帧并分割存储"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # 分割帧
            height, width = frame.shape[:2]
            # 上下区域 (2586*363)
            up = frame[0:363, :]
            down = frame[height-363:height, :]
            # 左右区域 (1080*1080)
            left = frame[363:1443, 0:1080]
            right = frame[363:1443, width-1080:width]
            
            # 保存分割后的图片
            cv2.imwrite(f'temp_frames/up/frame_{saved_count}_up.jpg', up)
            cv2.imwrite(f'temp_frames/down/frame_{saved_count}_down.jpg', down)
            cv2.imwrite(f'temp_frames/left/frame_{saved_count}_left.jpg', left)
            cv2.imwrite(f'temp_frames/right/frame_{saved_count}_right.jpg', right)
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count

def match_features(gray1, gray2):
    """特征匹配"""
    # SIFT特征检测
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # 特征匹配
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    
    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    return kp1, kp2, good_matches

def stitch_images(img1, img2):
    """将两张图片进行拼接"""
    try:
        # 直接使用原图转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 特征匹配
        kp1, kp2, good_matches = match_features(gray1, gray2)
        
        if len(good_matches) < 4:
            print("Warning: 没有足够的特征点匹配")
            return None
        
        # 获取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Warning: 无法计算单应性矩阵")
            return None
        
        # 进行图像变换
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算变换后的图像范围
        pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        # 计算输出图像的大小
        xmin = min(dst[:, 0, 0].min(), 0)
        ymin = min(dst[:, 0, 1].min(), 0)
        xmax = max(dst[:, 0, 0].max(), w2)
        ymax = max(dst[:, 0, 1].max(), h2)
        
        # 调整变换矩阵
        translation_matrix = np.array([
            [1, 0, -xmin],
            [0, 1, -ymin],
            [0, 0, 1]
        ])
        H = translation_matrix.dot(H)
        
        # 创建输出图像
        output_img = cv2.warpPerspective(img1, H, (int(xmax-xmin), int(ymax-ymin)))
        output_img[-int(ymin):h2-int(ymin), -int(xmin):w2-int(xmin)] = img2
        
        # 创建并应用渐变混合
        mask = np.zeros((output_img.shape[0], output_img.shape[1]), dtype=np.float32)
        mask[-int(ymin):h2-int(ymin), -int(xmin):w2-int(xmin)] = 1
        mask = cv2.GaussianBlur(mask, (41, 41), 11)
        mask = np.dstack((mask, mask, mask))
        
        warped_img = cv2.warpPerspective(img1, H, (int(xmax-xmin), int(ymax-ymin)))
        img2_placed = np.zeros_like(output_img)
        img2_placed[-int(ymin):h2-int(ymin), -int(xmin):w2-int(xmin)] = img2
        
        # 混合图像
        output_img = img2_placed * mask + warped_img * (1 - mask)
        
        return output_img.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in stitch_images: {str(e)}")
        return None

def stitch_region(region_name, total_frames):
    """拼接指定区域的所有帧"""
    # 读取第一帧作为基准图片
    result = cv2.imread(f'temp_frames/{region_name}/frame_0_{region_name}.jpg')
    if result is None:
        raise ValueError(f"无法读取{region_name}区域的第一帧")
    
    # 逐帧拼接
    for i in range(1, total_frames):
        next_frame = cv2.imread(f'temp_frames/{region_name}/frame_{i}_{region_name}.jpg')
        if next_frame is None:
            print(f"Warning: Could not read frame {i} for {region_name}")
            continue
            
        stitched = stitch_images(result, next_frame)
        if stitched is not None:
            result = stitched
        else:
            print(f"Warning: Failed to stitch frame {i} for {region_name}")
    
    # 保存结果
    cv2.imwrite(f'output/split/{region_name}_stitched.jpg', result)

def merge_final_map():
    """合并四个区域图片"""
    # 读取各个区域的拼接结果
    up = cv2.imread('output/split/up_stitched.jpg')
    down = cv2.imread('output/split/down_stitched.jpg')
    left = cv2.imread('output/split/left_stitched.jpg')
    right = cv2.imread('output/split/right_stitched.jpg')
    
    if any(img is None for img in [up, down, left, right]):
        raise ValueError("无法读取某个区域的拼接结果")
    
    # 创建最终地图
    central_height = left.shape[0]  # 802
    total_height = central_height + up.shape[0] + down.shape[0]  # 802 + 270 + 270
    total_width = max(up.shape[1], left.shape[1] + right.shape[1], down.shape[1])
    
    final_map = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # 放置上部分
    final_map[0:up.shape[0], :up.shape[1]] = up
    
    # 放置左右部分
    start_y = up.shape[0]
    final_map[start_y:start_y+left.shape[0], :left.shape[1]] = left
    final_map[start_y:start_y+right.shape[0], total_width-right.shape[1]:] = right
    
    # 放置下部分
    start_y = start_y + left.shape[0]
    final_map[start_y:start_y+down.shape[0], :down.shape[1]] = down
    
    # 保存最终结果
    cv2.imwrite('output/final_map.jpg', final_map)

def cleanup():
    """清理临时文件"""
    shutil.rmtree('temp_frames', ignore_errors=True)

def main(video_path, frame_interval=30):
    try:
        # 创建文件夹
        create_folders()
        
        # 提取和分割帧
        print("正在提取和分割帧...")
        total_frames = extract_and_split_frames(video_path, frame_interval)
        
        # 拼接各个区域
        print("正在拼接各个区域...")
        for region in ['up', 'down', 'left', 'right']:
            print(f"处理{region}区域...")
            stitch_region(region, total_frames)
        
        # 合并最终地图
        print("正在合成最终地图...")
        merge_final_map()
        
        print("处理完成！")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 是否要清理临时文件可以根据需要决定
        # cleanup()
        pass

if __name__ == "__main__":
    video_path = "video\\GTAV-v1.MP4"  # 替换为实际的视频路径
    frame_interval = 30  # 每30帧取一帧
    main(video_path, frame_interval)