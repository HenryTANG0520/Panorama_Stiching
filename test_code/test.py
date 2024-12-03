import cv2
import numpy as np

class ImageStitcher:
    def __init__(self):
        # 初始化SIFT检测器
        self.sift = cv2.SIFT_create()
        # 初始化FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def resize_images(self, img1, img2, target_height=800):
        """调整图片大小，保持比例"""
        # 计算调整比例
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        ratio1 = target_height / float(h1)
        ratio2 = target_height / float(h2)
        
        # 调整图片大小
        img1_resized = cv2.resize(img1, (int(w1 * ratio1), target_height))
        img2_resized = cv2.resize(img2, (int(w2 * ratio2), target_height))
        
        return img1_resized, img2_resized, (ratio1, ratio2)

    def match_features(self, img1, img2):
        """特征匹配"""
        # 检测SIFT特征点和描述符
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        # FLANN特征匹配
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # 应用Lowe比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return kp1, kp2, good_matches

    def stitch_images(self, img1_path, img2_path):
        """拼接两张图片"""
        # 读取图片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("无法读取图片")
        
        # 调整图片大小
        img1_resized, img2_resized, ratios = self.resize_images(img1, img2)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # 特征匹配
        kp1, kp2, good_matches = self.match_features(gray1, gray2)
        
        if len(good_matches) < 4:
            raise ValueError("没有足够的特征点匹配")
        
        # 获取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 进行图像变换
        h1, w1 = img1_resized.shape[:2]
        h2, w2 = img2_resized.shape[:2]
        
        # 计算变换后的图像范围
        pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        dst = np.int32(dst)
        
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
        output_img = cv2.warpPerspective(
            img1_resized,
            H,
            (xmax - xmin, ymax - ymin)
        )
        
        # 复制第二张图片到输出图像
        output_img[-ymin:h2-ymin, -xmin:w2-xmin] = img2_resized
        
        # 创建渐变混合区域
        mask = np.zeros((output_img.shape[0], output_img.shape[1]), dtype=np.float32)
        mask[-ymin:h2-ymin, -xmin:w2-xmin] = 1
        
        # 应用渐变混合
        mask = cv2.GaussianBlur(mask, (41, 41), 11)
        mask = np.dstack((mask, mask, mask))
        
        warped_img = cv2.warpPerspective(
            img1_resized,
            H,
            (xmax - xmin, ymax - ymin)
        )
        
        img2_placed = np.zeros_like(output_img)
        img2_placed[-ymin:h2-ymin, -xmin:w2-xmin] = img2_resized
        
        # 混合图像
        output_img = img2_placed * mask + warped_img * (1 - mask)
        
        return output_img.astype(np.uint8)

def main():
    stitcher = ImageStitcher()
    try:
        # 拼接图片
        result = stitcher.stitch_images('image1.jpg', 'image2.jpg')
        
        # 保存结果
        cv2.imwrite('stitched_result.jpg', result)
        print("图片拼接完成！")
        
    except Exception as e:
        print(f"拼接过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()