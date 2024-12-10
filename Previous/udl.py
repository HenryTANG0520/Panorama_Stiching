import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from tqdm import tqdm
import time
import numpy as np
from pathlib import Path
import shutil
import gc
import traceback

def create_folders():
    """创建必要的文件夹结构"""
    folders = ['temp_frames', 'output', 'checkpoints']
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, start_frame, frame_interval):
    """从视频中提取关键帧"""
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
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                cv2.imwrite(f'temp_frames/frame_{saved_count}.jpg', frame)
                print(f"\r提取帧进度: {saved_count + 1}/{(total_frames - start_frame) // frame_interval}", 
                      end="", flush=True)
                saved_count += 1
                
            frame_count += 1
            del frame
            gc.collect()
        
        print("\n帧提取完成")
        return saved_count
    finally:
        cap.release()
        gc.collect()

class FeatureExtractor(nn.Module):
    """特征提取网络"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.relu(self.bn1(self.conv1(x)))
        p1 = self.pool(f1)
        
        f2 = self.relu(self.bn2(self.conv2(p1)))
        p2 = self.pool(f2)
        
        f3 = self.relu(self.bn3(self.conv3(p2)))
        
        return [f1, f2, f3]

class CorrelationLayer(nn.Module):
    """特征相关性计算层"""
    def __init__(self, chunk_size=1000):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.size()
        feat1_flat = feat1.view(b, c, -1)
        feat2_flat = feat2.view(b, c, -1)
        total_positions = h * w
        correlation = torch.zeros(b, total_positions, h, w, device=feat1.device)

        for i in range(0, total_positions, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_positions)
            curr_correlation = torch.bmm(
                feat1_flat[:, :, i:end_idx].permute(0, 2, 1),
                feat2_flat
            )
            correlation[:, i:end_idx] = curr_correlation.view(b, end_idx-i, h, w)
            del curr_correlation
            torch.cuda.empty_cache()

        return F.softmax(correlation, dim=1)

class StitchingNet(nn.Module):
    """全景图拼接网络"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.correlation = CorrelationLayer()
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.blend_weights = nn.Conv2d(64, 1, 1)
        
    def forward(self, img1, img2):
        # 特征提取
        feats1 = self.feature_extractor(img1)
        feats2 = self.feature_extractor(img2)
        
        # 多尺度特征匹配
        correlations = []
        for f1, f2 in zip(feats1, feats2):
            corr = self.correlation(f1, f2)
            correlations.append(corr)
        
        # 特征融合与权重预测
        fusion_feats = torch.cat([feats1[-1], feats2[-1]], dim=1)
        fused = self.fusion(fusion_feats)
        weights = torch.sigmoid(self.blend_weights(fused))
        
        # 上采样权重并生成结果
        weights = F.interpolate(weights, size=img1.shape[2:], mode='bilinear', align_corners=False)
        result = weights * img1 + (1 - weights) * img2
        
        return result, weights, correlations

class SelfSupervisedLoss(nn.Module):
    """自监督学习损失函数"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, result, img1, img2, weights, correlations):
        # 重建损失
        reconstruct_loss = self.l1_loss(result, img2)
        
        # 特征相关性一致性损失
        correlation_loss = sum(self.mse_loss(corr, torch.ones_like(corr)/corr.shape[1]) 
                             for corr in correlations)
        
        # 权重平滑度损失
        smoothness_loss = (
            self.l1_loss(weights[:,:,1:,:], weights[:,:,:-1,:]) +
            self.l1_loss(weights[:,:,:,1:], weights[:,:,:,:-1])
        )
        
        # 总损失
        return reconstruct_loss + 0.1 * correlation_loss + 0.01 * smoothness_loss

class FramePairsDataset(Dataset):
    """帧对数据集"""
    def __init__(self, frames_dir, size=(128, 128)):
        self.frames_dir = Path(frames_dir)
        self.frame_pairs = self._get_frame_pairs()
        self.size = size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3] if x.size(0) > 3 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __getitem__(self, idx):
        try:
            frame1_path, frame2_path = self.frame_pairs[idx]
            frame1 = self._safe_read_image(frame1_path)
            frame2 = self._safe_read_image(frame2_path)
            
            if self.transform:
                try:
                    frame1 = self.transform(frame1)
                    frame2 = self.transform(frame2)
                except Exception as e:
                    print(f"Transform error for index {idx}: {str(e)}")
                    frame1 = torch.zeros((3, *self.size))
                    frame2 = torch.zeros((3, *self.size))
            
            return frame1, frame2
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            return torch.zeros((3, *self.size)), torch.zeros((3, *self.size))
    
    def _get_frame_pairs(self):
        """获取相邻帧对"""
        frames = []
        for frame in sorted(list(self.frames_dir.glob('frame_*.jpg'))):
            if frame.exists() and frame.stat().st_size > 0:
                frames.append(frame)
                
        if not frames:
            raise RuntimeError(f"No valid frames found in {self.frames_dir}")
            
        return [(frames[i], frames[i+1]) for i in range(len(frames)-1)]
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def _safe_read_image(self, path):
        """安全读取图像"""
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.uint8)
            
        except Exception as e:
            print(f"Error reading image {path}: {str(e)}")
            return np.zeros((*self.size, 3), dtype=np.uint8)

def train_model(model, train_loader, num_epochs, device):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = SelfSupervisedLoss()
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    epoch_pbar = tqdm(total=num_epochs, desc="Training Progress")

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            # 梯度累积
            accumulation_steps = 4
            optimizer.zero_grad()

            batch_pbar = tqdm(total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{num_epochs}",
                            leave=False)

            start_time = time.time()

            for batch_idx, (img1, img2) in enumerate(train_loader):
                try:
                    img1, img2 = img1.to(device), img2.to(device)

                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()

                    result, weights, correlations = model(img1, img2)
                    loss = criterion(result, img1, img2, weights, correlations)
                    loss = loss / accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += loss.item()
                    batch_count += 1

                    batch_pbar.update(1)
                    batch_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss/batch_count:.4f}'
                    })

                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    continue

            batch_pbar.close()

            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            epoch_time = time.time() - start_time

            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'time': f'{epoch_time:.1f}s'
            })

            # 模型保存和早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, 'checkpoints/best_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {str(e)}")
    finally:
        epoch_pbar.close()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, 'checkpoints/final_model.pt')

def stitch_with_model(model, img1, img2, device, target_size=(256, 256)):
    """使用训练好的模型进行图像拼接"""
    model.eval()
    try:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 预处理图像
                if isinstance(img1, np.ndarray):
                    img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
                if isinstance(img2, np.ndarray):
                    img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0

                # 降采样到目标大小
                img1_small = F.interpolate(img1.unsqueeze(0), size=target_size, mode='bilinear')
                img2_small = F.interpolate(img2.unsqueeze(0), size=target_size, mode='bilinear')

                # 转移到设备
                img1_small = img1_small.to(device)
                img2_small = img2_small.to(device)
                del img1, img2
                torch.cuda.empty_cache()

                # 执行拼接
                result_small, _, _ = model(img1_small, img2_small)
                del img1_small, img2_small
                torch.cuda.empty_cache()

                # 上采样到原始分辨率
                result = F.interpolate(result_small, size=(1080, 1920), mode='bilinear')
                del result_small
                torch.cuda.empty_cache()

                # 后处理
                result = result.squeeze(0).cpu().numpy()
                result = (result * 255).astype(np.uint8)
                result = np.clip(result, 0, 255)
                result = np.transpose(result, (1, 2, 0))
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                return result

    except Exception as e:
        print(f"Stitching error: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return None

def process_panorama(model, device, total_frames):
    """渐进式全景图拼接"""
    print("开始拼接全景图...")
    panorama = None

    try:
        for i in range(total_frames - 1):
            torch.cuda.empty_cache()

            try:
                # 读取帧
                if panorama is None:
                    img1 = cv2.imread(f'temp_frames/frame_{i}.jpg')
                    if img1 is None:
                        raise ValueError(f"无法读取帧 {i}")
                else:
                    img1 = panorama

                img2 = cv2.imread(f'temp_frames/frame_{i+1}.jpg')
                if img2 is None:
                    raise ValueError(f"无法读取帧 {i+1}")

                # 转换颜色空间
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                print(f"\nProcessing frames {i} and {i+1}")
                print(f"Shapes - img1: {img1_rgb.shape}, img2: {img2_rgb.shape}")

                # 执行拼接
                result = stitch_with_model(model, img1_rgb, img2_rgb, device)

                if result is not None:
                    panorama = result
                    cv2.imwrite(f'temp_frames/intermediate_{i}.jpg', result)

                print(f"Progress: {i+1}/{total_frames-1}")

            except Exception as e:
                print(f"\nError in frame {i}: {str(e)}")
                if panorama is None:
                    panorama = img1
                continue

            gc.collect()
            torch.cuda.empty_cache()

        # 保存结果
        if panorama is not None:
            print("\nStitching completed")
            cv2.imwrite('output/panorama_final.jpg', panorama)
        else:
            print("\nNo panorama generated")

    except Exception as e:
        print(f"\nPanorama processing error: {str(e)}")
        if panorama is not None:
            cv2.imwrite('output/panorama_error.jpg', panorama)

def main():
    """主函数"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 创建文件夹
        create_folders()
        
        # 提取视频帧
        video_path = "video/4.mp4"
        start_frame = 3
        frame_interval = 60
        print("正在提取帧...")
        total_frames = extract_frames(video_path, start_frame, frame_interval)
        
        # 初始化数据集和加载器
        try:
            print("初始化数据集...")
            dataset = FramePairsDataset('temp_frames', size=(128, 128))
            train_loader = DataLoader(
                dataset, 
                batch_size=2,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            print(f"数据集初始化完成，共有 {len(dataset)} 对图像")
            
        except Exception as e:
            print(f"创建数据加载器时出错: {str(e)}")
            raise
        
        # 训练模型
        model = StitchingNet()
        print("开始训练模型...")
        train_model(model, train_loader, num_epochs=100, device=device)
        
        # 生成全景图
        print("开始拼接全景图...")
        process_panorama(model, device, len(dataset) + 1)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()