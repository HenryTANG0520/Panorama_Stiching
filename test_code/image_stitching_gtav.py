import cv2
import numpy as np
from pathlib import Path
import shutil
import gc

def check_gpu():
    """检查CUDA是否可用"""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            print(f"找到 {count} 个支持CUDA的设备")
            device = cv2.cuda.getDevice()
            print(f"使用设备 {device}")
            props = cv2.cuda.getDeviceInfo(device)
            print(f"设备名称: {props.name()}")
            print(f"可用内存: {props.totalMemory() / (1024*1024):.2f} MB")
            return True
        else:
            print("未找到支持CUDA的设备，将使用CPU")
            return False
    except Exception as e:
        print(f"检查CUDA设备时出错: {e}")
        return False

def create_folders():
    """创建必要的文件夹结构"""
    folders = [
        'temp_frames/up',
        'temp_frames/down',
        'temp_frames/left',
        'temp_frames/right',
        'temp_frames/stitching',  # 用于存储拼接过程的中间结果
        'output/split',
        'output'
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
def extract_and_split_frames(video_path, frame_interval, helicopter_ratio):
    """
    提取帧并分割存储
    Args:
        video_path: 视频路径
        frame_interval: 帧间隔
        helicopter_ratio: 直升机区域占高度的比例
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        frame_count = 0
        saved_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"总帧数: {total_frames}")
        print(f"预计处理帧数: {total_frames // frame_interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 计算分割区域
                height, width = frame.shape[:2]
                helicopter_height = int(height * helicopter_ratio)
                center_height = height - 2 * helicopter_height  # 中间区域的高度
                
                # 计算各个区域的范围
                up = frame[0:helicopter_height, :]
                down = frame[height-helicopter_height:height, :]
                left = frame[helicopter_height:height-helicopter_height, 0:width//2]
                right = frame[helicopter_height:height-helicopter_height, width//2:width]
                
                # 保存分割后的图片
                cv2.imwrite(f'temp_frames/up/frame_{saved_count}_up.jpg', up)
                cv2.imwrite(f'temp_frames/down/frame_{saved_count}_down.jpg', down)
                cv2.imwrite(f'temp_frames/left/frame_{saved_count}_left.jpg', left)
                cv2.imwrite(f'temp_frames/right/frame_{saved_count}_right.jpg', right)
                
                print(f"\r提取帧进度: {saved_count + 1}/{total_frames // frame_interval}", end="", flush=True)
                saved_count += 1
                
                # 释放内存
                del up, down, left, right
                gc.collect()
            
            frame_count += 1
            del frame
            gc.collect()
        
        print("\n帧提取完成")
        return saved_count
        
    finally:
        cap.release()
        gc.collect()

def release_gpu_memory(*gpu_mats):
    """释放GPU内存"""
    for mat in gpu_mats:
        if mat is not None:
            mat.release()

def stitch_images(img1, img2):
    """CPU版本的图像拼接函数"""
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

        # 2. 筛选好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        del matches
        gc.collect()

        if len(good_matches) >= 4:
            # 获取匹配点的坐标
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 释放关键点内存
            del keypoints1, keypoints2, good_matches

            # 计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            del src_pts, dst_pts
            gc.collect()

            if H is not None:
                # 计算变换后图像的范围
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                pts2 = cv2.perspectiveTransform(pts1, H)

                pts = np.concatenate((pts2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
                [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
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

                return result

        return None

    except Exception as e:
        print(f"拼接过程中出错: {str(e)}")
        return None

    finally:
        # 强制垃圾回收
        gc.collect()

def stitch_images_gpu(img1, img2):
    """使用GPU加速的图像拼接"""
    # 初始化所有GPU变量为None
    gpu_vars = {
        'gpu_img1': None, 'gpu_img2': None,
        'gpu_gray1': None, 'gpu_gray2': None,
        'gpu_warped': None, 'gpu_mask1': None,
        'gpu_mask2': None, 'gpu_weight1': None,
        'gpu_weight2': None, 'gpu_ones': None
    }
    
    try:
        # 创建并上传GPU图像
        gpu_vars['gpu_img1'] = cv2.cuda_GpuMat()
        gpu_vars['gpu_img2'] = cv2.cuda_GpuMat()
        gpu_vars['gpu_img1'].upload(img1)
        gpu_vars['gpu_img2'].upload(img2)

        # 转换为灰度图
        gpu_vars['gpu_gray1'] = cv2.cuda.cvtColor(gpu_vars['gpu_img1'], cv2.COLOR_BGR2GRAY)
        gpu_vars['gpu_gray2'] = cv2.cuda.cvtColor(gpu_vars['gpu_img2'], cv2.COLOR_BGR2GRAY)

        # 使用CUDA SIFT
        cuda_sift = cv2.cuda.SIFT_create()
        
        # 检测关键点和描述符
        keypoints1, descriptors1 = cuda_sift.detectAndComputeAsync(gpu_vars['gpu_gray1'], None)
        keypoints2, descriptors2 = cuda_sift.detectAndComputeAsync(gpu_vars['gpu_gray2'], None)
        
        # 释放灰度图的GPU内存
        release_gpu_memory(gpu_vars['gpu_gray1'], gpu_vars['gpu_gray2'])
        gpu_vars['gpu_gray1'] = gpu_vars['gpu_gray2'] = None

        # 下载描述符到CPU进行匹配
        descriptors1_cpu = descriptors1.download()
        descriptors2_cpu = descriptors2.download()
        descriptors1.release()
        descriptors2.release()

        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1_cpu, descriptors2_cpu, k=2)
        
        # 清理CPU内存
        del descriptors1_cpu, descriptors2_cpu
        gc.collect()

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return None

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            pts2 = cv2.perspectiveTransform(pts1, H)

            pts = np.concatenate((pts2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
            t = [-xmin, -ymin]

            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
            H_final = Ht.dot(H)

            # 使用GPU进行透视变换
            gpu_vars['gpu_warped'] = cv2.cuda.warpPerspective(gpu_vars['gpu_img1'], H_final, (xmax-xmin, ymax-ymin))
            result = gpu_vars['gpu_warped'].download()
            release_gpu_memory(gpu_vars['gpu_img1'])
            gpu_vars['gpu_img1'] = None

            # 将第二张图片复制到结果图像中
            result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2

            # 在GPU上创建和处理掩码
            gpu_vars['gpu_mask1'] = cv2.cuda_GpuMat()
            mask_cpu = np.zeros((ymax-ymin, xmax-xmin), dtype=np.float32)
            mask_cpu[t[1]:h2+t[1], t[0]:w2+t[0]] = 1
            gpu_vars['gpu_mask1'].upload(mask_cpu)
            del mask_cpu

            # 在GPU上进行高斯模糊
            gpu_vars['gpu_weight1'] = cv2.cuda.GaussianBlur(gpu_vars['gpu_mask1'], (21, 21), 11)
            release_gpu_memory(gpu_vars['gpu_mask1'])
            gpu_vars['gpu_mask1'] = None

            gpu_vars['gpu_ones'] = cv2.cuda_GpuMat()
            gpu_vars['gpu_ones'].upload(np.ones_like(img1[:,:,0], dtype=np.float32))
            gpu_vars['gpu_mask2'] = cv2.cuda.warpPerspective(gpu_vars['gpu_ones'], H_final, (xmax-xmin, ymax-ymin))
            release_gpu_memory(gpu_vars['gpu_ones'])
            gpu_vars['gpu_ones'] = None

            gpu_vars['gpu_weight2'] = cv2.cuda.GaussianBlur(gpu_vars['gpu_mask2'], (21, 21), 11)
            release_gpu_memory(gpu_vars['gpu_mask2'])
            gpu_vars['gpu_mask2'] = None

            # 下载权重
            weight1 = gpu_vars['gpu_weight1'].download()
            weight2 = gpu_vars['gpu_weight2'].download()
            release_gpu_memory(gpu_vars['gpu_weight1'], gpu_vars['gpu_weight2'])
            gpu_vars['gpu_weight1'] = gpu_vars['gpu_weight2'] = None

            # 权重归一化和维度扩展
            weight_sum = weight1 + weight2 + 1e-6
            weight1 = np.expand_dims(weight1 / weight_sum, axis=2)
            weight2 = np.expand_dims(weight2 / weight_sum, axis=2)
            del weight_sum

            # 最终混合
            warped_img2 = cv2.cuda.warpPerspective(gpu_vars['gpu_img2'], H_final, (xmax-xmin, ymax-ymin)).download()
            release_gpu_memory(gpu_vars['gpu_img2'])
            gpu_vars['gpu_img2'] = None

            final_result = (weight1 * result + weight2 * warped_img2).astype(np.uint8)
            del result, warped_img2, weight1, weight2
            gc.collect()

            return final_result

        return None

    except Exception as e:
        print(f"GPU拼接过程中出错: {str(e)}")
        return None

    finally:
        # 确保所有GPU资源都被释放
        for gpu_mat in gpu_vars.values():
            release_gpu_memory(gpu_mat)
        cv2.cuda.deviceReset()
        gc.collect()

def stitch_region_with_checkpoints(region_name, total_frames, checkpoint_interval=5, use_gpu=True):
    """分批拼接指定区域的所有帧"""
    result = None
    try:
        result = cv2.imread(f'temp_frames/{region_name}/frame_0_{region_name}.jpg')
        if result is None:
            raise ValueError(f"无法读取{region_name}区域的第一帧")
        
        print(f"\n开始处理{region_name}区域...")
        last_checkpoint = 0
        checkpoint_count = 0
        
        cv2.imwrite(f'temp_frames/stitching/{region_name}_checkpoint_{checkpoint_count}.jpg', result)
        
        for i in range(1, total_frames):
            try:
                next_frame = cv2.imread(f'temp_frames/{region_name}/frame_{i}_{region_name}.jpg')
                if next_frame is None:
                    print(f"\nWarning: Could not read frame {i} for {region_name}")
                    continue
                
                print(f"\r{region_name}区域进度: {i}/{total_frames-1}", end="", flush=True)
                
                if i - last_checkpoint >= checkpoint_interval:
                    cv2.imwrite(f'temp_frames/stitching/{region_name}_checkpoint_{checkpoint_count + 1}.jpg', result)
                    del result
                    gc.collect()
                    checkpoint_count += 1
                    last_checkpoint = i
                    result = cv2.imread(f'temp_frames/stitching/{region_name}_checkpoint_{checkpoint_count}.jpg')
                
                stitched = stitch_images_gpu(result, next_frame) if use_gpu else stitch_images(result, next_frame)
                if stitched is not None:
                    del result
                    result = stitched
                    gc.collect()
                else:
                    print(f"\nWarning: Failed to stitch frame {i} for {region_name}")
                
                del next_frame
                gc.collect()
                
            except Exception as e:
                print(f"\nError processing frame {i} for {region_name}: {str(e)}")
                continue
        
        if result is not None:
            print(f"\n{region_name}区域处理完成")
            cv2.imwrite(f'output/split/{region_name}_stitched.jpg', result)
        
    except Exception as e:
        print(f"\nError in {region_name} region: {str(e)}")
        if result is not None:
            cv2.imwrite(f'output/split/{region_name}_stitched_error.jpg', result)
    
    finally:
        # 清理中间文件和内存
        try:
            for i in range(checkpoint_count + 1):
                checkpoint_file = f'temp_frames/stitching/{region_name}_checkpoint_{i}.jpg'
                Path(checkpoint_file).unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup checkpoints for {region_name}: {e}")
        
        del result
        gc.collect()

def merge_final_map():
    """合并四个区域图片，保持自然探索尺寸"""
    # 读取各个区域的拼接结果
    up = cv2.imread('output/split/up_stitched.jpg')
    down = cv2.imread('output/split/down_stitched.jpg')
    left = cv2.imread('output/split/left_stitched.jpg')
    right = cv2.imread('output/split/right_stitched.jpg')
    
    if any(img is None for img in [up, down, left, right]):
        raise ValueError("无法读取某个区域的拼接结果")
    
    try:
        # 计算中间区域的总宽度和位置
        central_width = left.shape[1] + right.shape[1]
        central_height = max(left.shape[0], right.shape[0])
        
        # 计算最终地图的尺寸
        total_width = max(up.shape[1], central_width, down.shape[1])
        total_height = up.shape[0] + central_height + down.shape[0]
        
        # 创建最终地图画布
        final_map = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        # 计算各部分的起始位置
        up_start_x = (total_width - up.shape[1]) // 2
        down_start_x = (total_width - down.shape[1]) // 2
        left_start_x = (total_width - central_width) // 2
        right_start_x = left_start_x + left.shape[1]
        
        # 放置上部分
        final_map[0:up.shape[0], 
                 up_start_x:up_start_x+up.shape[1]] = up
        
        # 放置中间左右部分
        middle_y = up.shape[0]
        final_map[middle_y:middle_y+left.shape[0], 
                 left_start_x:left_start_x+left.shape[1]] = left
        final_map[middle_y:middle_y+right.shape[0], 
                 right_start_x:right_start_x+right.shape[1]] = right
        
        # 放置下部分
        down_y = middle_y + central_height
        final_map[down_y:down_y+down.shape[0], 
                 down_start_x:down_start_x+down.shape[1]] = down
        
        # 保存最终结果
        cv2.imwrite('output/final_map_natural.jpg', final_map)
        print(f"最终地图尺寸: {final_map.shape[1]}x{final_map.shape[0]}")
        
    finally:
        # 清理内存
        del up, down, left, right, final_map
        gc.collect()

def cleanup():
    """清理临时文件"""
    try:
        shutil.rmtree('temp_frames', ignore_errors=True)
    except Exception as e:
        print(f"清理临时文件时出错: {str(e)}")

def main(video_path, frame_interval=30, helicopter_ratio=0.1):
    """主函数"""
    try:
        use_gpu = check_gpu()
        create_folders()
        
        print("正在提取和分割帧...")
        total_frames = extract_and_split_frames(video_path, frame_interval, helicopter_ratio)
        
        print("正在拼接各个区域...")
        for region in ['up', 'down', 'left', 'right']:
            print(f"处理{region}区域...")
            stitch_region_with_checkpoints(region, total_frames, use_gpu=use_gpu)
            gc.collect()  # 每个区域处理完后强制垃圾回收
        
        print("正在合成最终地图...")
        merge_final_map(helicopter_ratio)
        
        print("处理完成！")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        cv2.cuda.deviceReset()  # 重置GPU设备
        gc.collect()

if __name__ == "__main__":
    video_path = "video/GTAV-v1.mp4"
    frame_interval = 90
    helicopter_ratio = 0.32
    main(video_path, frame_interval, helicopter_ratio)