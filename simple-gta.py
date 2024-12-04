
import cv2
import numpy as np
from pathlib import Path
import shutil
import gc

def create_folders():
    """创建必要的文件夹结构"""
    folders = [
        'temp_frames',
        'output'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, start_frame, frame_interval):
    """
    从指定帧开始提取帧
    Args:
        video_path: 视频路径
        start_frame: 开始帧的索引
        frame_interval: 帧间隔
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        frame_count = 0
        saved_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"总帧数: {total_frames}")
        print(f"从第 {start_frame} 帧开始，间隔 {frame_interval} 帧")
        print(f"预计处理帧数: {(total_frames - start_frame) // frame_interval}")
        
        # 跳到指定的开始帧
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

def stitch_images(img1, img2):
    """图像拼接函数，增加了变换范围的约束"""
    try:
        # 1. 特征点检测与匹配
        sift = cv2.SIFT_create()

        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        if descriptors1 is None or descriptors2 is None:
            return None

        # 创建特征匹配器
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # 立即释放描述符内存
        del descriptors1, descriptors2

        # 2. 筛选好的匹配点（使用更严格的比率测试）
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # 更严格的比率阈值
                good_matches.append(m)

        del matches
        gc.collect()

        if len(good_matches) >= 4:
            # 获取匹配点的坐标
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 释放关键点内存
            del keypoints1, keypoints2, good_matches

            # 计算单应性矩阵（增加更严格的RANSAC参数）
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000)

            del src_pts, dst_pts
            gc.collect()

            if H is not None:
                # 计算变换后图像的范围
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                pts2 = cv2.perspectiveTransform(pts1, H)

                # 添加范围约束检查
                max_expected_width = w1 + w2 * 2  # 最大允许宽度
                max_expected_height = (h1 + h2) * 2  # 最大允许高度

                pts = np.concatenate((pts2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
                [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

                # 检查变换后的尺寸是否合理
                if (xmax - xmin > max_expected_width) or (ymax - ymin > max_expected_height):
                    print(f"\n警告：检测到异常的变换范围 {xmax-xmin}x{ymax-ymin}，跳过当前帧")
                    return None

                t = [-xmin, -ymin]

                del pts, pts1, pts2

                # 创建平移矩阵
                Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
                H_final = Ht.dot(H)
                del H, Ht

                # 对第一张图片进行变换
                result = cv2.warpPerspective(img1, H_final, (xmax-xmin, ymax-ymin))

                # 将第二张图片复制到结果图像中
                result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2

                # 处理重叠区域
                mask1 = np.zeros((ymax-ymin, xmax-xmin), dtype=np.float32)
                mask1[t[1]:h2+t[1], t[0]:w2+t[0]] = 1
                mask2 = cv2.warpPerspective(np.ones_like(img1[:,:,0], dtype=np.float32), H_final, (xmax-xmin, ymax-ymin))

                # 创建权重矩阵
                weight1 = cv2.GaussianBlur(mask1, (21, 21), 11)
                weight2 = cv2.GaussianBlur(mask2, (21, 21), 11)

                del mask1, mask2

                # 归一化权重
                weight_sum = weight1 + weight2 + 1e-6
                weight1 /= weight_sum
                weight2 /= weight_sum
                del weight_sum

                # 扩展维度
                weight1 = np.expand_dims(weight1, axis=2)
                weight2 = np.expand_dims(weight2, axis=2)

                # 最终混合
                warped_img1 = cv2.warpPerspective(img1, H_final, (xmax-xmin, ymax-ymin))
                result = (weight1 * result + weight2 * warped_img1).astype(np.uint8)

                # 清理最后的中间变量
                del weight1, weight2, warped_img1, H_final
                gc.collect()

                # 检查最终结果的非黑色区域
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                non_black = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
                x, y, w, h = cv2.boundingRect(non_black)

                # 只保留有内容的部分
                result = result[y:y+h, x:x+w]

                return result

        return None

    except Exception as e:
        print(f"拼接过程中出错: {str(e)}")
        return None

    finally:
        gc.collect()

def stitch_frames_with_checkpoints(total_frames, checkpoint_interval=5):
    """分批拼接所有帧"""
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
                
                if i - last_checkpoint >= checkpoint_interval:
                    cv2.imwrite(f'temp_frames/checkpoint_{checkpoint_count + 1}.jpg', result)
                    del result
                    gc.collect()
                    checkpoint_count += 1
                    last_checkpoint = i
                    result = cv2.imread(f'temp_frames/checkpoint_{checkpoint_count}.jpg')
                
                stitched = stitch_images(result, next_frame)
                if stitched is not None:
                    del result
                    result = stitched
                    gc.collect()
                else:
                    print(f"\nWarning: Failed to stitch frame {i}")
                
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
        # cleanup()
        gc.collect()

if __name__ == "__main__":
    video_path = "video/2287_raw.mp4"  # 替换为你的视频路径
    start_frame = 30  # 从第30帧开始
    frame_interval = 60  # 每60帧取一帧
    main(video_path, start_frame, frame_interval)