# 航拍图像拼接系统
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

import cv2
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, List, Tuple
import wandb
from tqdm import tqdm
import random

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import kornia
from torchvision.transforms.functional import to_tensor
from sklearn.metrics import precision_recall_fscore_support

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================ 第1部分: 模型定义 ================

class PatchEmbed(nn.Module):
    """
    图像patch嵌入模块
    将输入图像划分为固定大小的patch，并进行特征提取
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层进行patch划分和特征提取
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        
        # 添加分类token和位置编码
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    用于捕捉图像patch之间的长程依赖关系
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    包含多头自注意力和前馈神经网络
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer特征提取器
    用于提取图像的层级特征表示
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

class CrossAttentionMatcher(nn.Module):
    """
    交叉注意力匹配模块
    用于计算两张图像特征之间的相似度
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        # 计算查询、键和值
        q = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

class ImageStitchingTransformer(nn.Module):
    """
    完整的图像拼接Transformer模型
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        # 特征提取器
        self.feature_extractor = ViTFeatureExtractor(
            img_size, patch_size, in_channels, embed_dim, depth, 
            num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate
        )
        # 特征匹配器
        self.matcher = CrossAttentionMatcher(embed_dim, num_heads, attn_drop_rate, drop_rate)
        
        # 匹配得分头
        self.match_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, img1, img2):
        # 提取特征
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        
        # 交叉注意力匹配
        matched_features, attention_weights = self.matcher(feat1, feat2)
        
        # 生成匹配分数
        matching_scores = self.match_head(matched_features).squeeze(-1)
        
        return {
            'features1': feat1,
            'features2': feat2,
            'matched_features': matched_features,
            'attention_weights': attention_weights,
            'matching_scores': matching_scores
        }

# ================ 第2部分: 数据处理 ================

class VideoFrameExtractor:
    """
    视频帧提取器
    从输入视频中提取帧用于训练
    """
    def __init__(self, overlap_ratio=0.3, frame_interval=5):
        self.overlap_ratio = overlap_ratio
        self.frame_interval = frame_interval
        self.last_frame_path = None
        
    def extract_frames(self, video_path, output_dir):
        """提取视频帧并保存"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_pairs = []
        frame_count = 0
        last_saved_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_interval == 0:
                frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                if self.last_frame_path is not None:
                    frame_pairs.append((self.last_frame_path, frame_path))
                self.last_frame_path = frame_path
                
            frame_count += 1
            
        cap.release()
        return frame_pairs
class StitchingDataset(Dataset):
    """
    图像拼接数据集
    用于训练图像拼接模型
    """
    def __init__(self, frame_pairs, img_size=224, is_train=True):
        self.frame_pairs = frame_pairs
        self.img_size = img_size
        self.is_train = is_train
        
        # 基础图像变换
        self.basic_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # 训练时的数据增强
        if is_train:
            self.train_transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.RandomRotate90(p=0.2),
                A.HorizontalFlip(p=0.5),
            ])
            
    def __len__(self):
        return len(self.frame_pairs)
        
    def __getitem__(self, idx):
        img1_path, img2_path = self.frame_pairs[idx]
        
        # 读取图像
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        # BGR转RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # 训练时进行数据增强
        if self.is_train:
            seed = random.randint(0, 2**32)
            
            # 对两张图像应用相同的随机变换
            random.seed(seed)
            img1 = self.train_transform(image=img1)["image"]
            random.seed(seed)
            img2 = self.train_transform(image=img2)["image"]
        
        # 应用基础变换
        img1 = self.basic_transform(image=img1)["image"]
        img2 = self.basic_transform(image=img2)["image"]
        
        return {
            'image1': img1,
            'image2': img2,
            'path1': str(img1_path),
            'path2': str(img2_path)
        }

def create_dataloaders(video_path, output_dir, batch_size=8, img_size=224, 
                      num_workers=4, frame_interval=5):
    """
    创建训练和验证数据加载器
    
    参数:
        video_path: 输入视频路径
        output_dir: 帧保存目录
        batch_size: 批次大小
        img_size: 图像大小
        num_workers: 数据加载进程数
        frame_interval: 帧采样间隔
    """
    # 提取视频帧
    extractor = VideoFrameExtractor(frame_interval=frame_interval)
    frame_pairs = extractor.extract_frames(video_path, output_dir)
    
    # 划分训练集和验证集
    train_pairs = frame_pairs[:-len(frame_pairs)//5]  # 后20%用于验证
    val_pairs = frame_pairs[-len(frame_pairs)//5:]
    
    # 创建数据集
    train_dataset = StitchingDataset(train_pairs, img_size=img_size, is_train=True)
    val_dataset = StitchingDataset(val_pairs, img_size=img_size, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class Trainer:
    """
    模型训练器
    管理整个训练过程，包括日志记录、检查点保存等
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.last_frame_path = None

        # 设置设备
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 设置学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 设置混合精度训练
        self.scaler = GradScaler('cuda')
        
        # 设置日志
        self.setup_logging()
        
        # 设置wandb
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'image-stitching'),
                name=config.get('wandb_run_name', time.strftime('%Y%m%d_%H%M%S')),
                config=config
            )
    def setup_logging(self):
        """设置日志配置"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Dict,
        best_metric: float,
        is_best: bool
    ):
        """
        保存模型检查点
        
        参数:
            epoch: 当前训练轮数
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            scheduler_state: 学习率调度器状态字典
            best_metric: 最佳指标值
            is_best: 是否为最佳模型
        """
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            'best_metric': best_metric,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(
            checkpoint,
            checkpoint_dir / 'latest_checkpoint.pth'
        )
        
        # 保存最佳检查点
        if is_best:
            torch.save(
                checkpoint,
                checkpoint_dir / 'best_checkpoint.pth'
            )
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # 将数据移至设备
            img1 = batch['image1'].to(self.device)
            img2 = batch['image2'].to(self.device)
            
            # 使用混合精度训练
            with autocast('cuda'):
                outputs = self.model(img1, img2)
                loss = self.compute_loss(outputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 计算指标
            accuracy = self.compute_accuracy(outputs)
            
            # 更新进度条
            total_loss += loss.item()
            total_accuracy += accuracy
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': total_accuracy / (pbar.n + 1)
            })
        
        return total_loss / len(self.train_loader), total_accuracy / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for batch in pbar:
            # 将数据移至设备
            img1 = batch['image1'].to(self.device)
            img2 = batch['image2'].to(self.device)
            
            # 前向传播
            outputs = self.model(img1, img2)
            loss = self.compute_loss(outputs)
            
            # 计算指标
            accuracy = self.compute_accuracy(outputs)
            
            # 更新指标
            total_loss += loss.item()
            total_accuracy += accuracy
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': total_accuracy / (pbar.n + 1)
            })
        
        return total_loss / len(self.val_loader), total_accuracy / len(self.val_loader)
    
    def compute_loss(self, outputs: Dict) -> torch.Tensor:
        """
        计算训练损失
        
        参数:
            outputs: 模型输出字典
        """
        # 匹配分数
        matching_scores = outputs['matching_scores']
        attention_weights = outputs['attention_weights']
        
        # 匹配损失
        matching_loss = nn.functional.binary_cross_entropy_with_logits(
            matching_scores,
            torch.ones_like(matching_scores)  # 假设所有对都应匹配
        )
        
        # 注意力正则化
        attention_loss = -torch.mean(
            torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        )
        
        return matching_loss + 0.1 * attention_loss
    
    def compute_accuracy(self, outputs: Dict) -> float:
        """
        计算准确率指标
        
        参数:
            outputs: 模型输出字典
        """
        matching_scores = outputs['matching_scores']
        predictions = (torch.sigmoid(matching_scores) > 0.5).float()
        return predictions.mean().item()
class StitchingEvaluator:
    """
    拼接评估器
    用于评估模型的拼接性能
    """
    def __init__(self, device='cuda'):
        self.device = device

    def compute_metrics(self, outputs: Dict, batch: Dict) -> Dict[str, float]:
        """计算多个评估指标"""
        metrics = {}
        
        # 1. 匹配准确率指标
        matching_metrics = self._compute_matching_metrics(
            outputs['matching_scores'],
            batch['gt_matches']
        )
        metrics.update(matching_metrics)
        
        # 2. 几何一致性指标
        geometric_metrics = self._compute_geometric_consistency(
            outputs['matched_features'],
            outputs['attention_weights']
        )
        metrics.update(geometric_metrics)
        
        # 3. 重叠区域质量指标
        if 'warped_image' in outputs:
            quality_metrics = self._compute_image_quality_metrics(
                outputs['warped_image'],
                batch['image2']
            )
            metrics.update(quality_metrics)
            
        return metrics
    
    def _compute_matching_metrics(self, pred_scores: torch.Tensor, 
                                gt_matches: torch.Tensor) -> Dict[str, float]:
        """计算特征匹配的准确率指标"""
        pred_matches = (torch.sigmoid(pred_scores) > 0.5).cpu().numpy()
        gt_matches = gt_matches.cpu().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_matches.ravel(),
            pred_matches.ravel(),
            average='binary'
        )
        
        return {
            'matching_precision': precision,
            'matching_recall': recall,
            'matching_f1': f1
        }
    
    def _compute_geometric_consistency(self, matched_features: torch.Tensor,
                                    attention_weights: torch.Tensor) -> Dict[str, float]:
        """计算几何一致性指标"""
        # 1. 局部一致性评分
        local_consistency = self._compute_local_consistency(matched_features)
        
        # 2. 全局变换评分
        global_consistency = self._compute_global_consistency(matched_features)
        
        return {
            'local_geometric_consistency': local_consistency,
            'global_geometric_consistency': global_consistency
        }
    
    def _compute_image_quality_metrics(self, warped_image: torch.Tensor,
                                     target_image: torch.Tensor) -> Dict[str, float]:
        """计算图像质量指标"""
        # 计算PSNR
        psnr = kornia.metrics.psnr(warped_image, target_image, max_val=1.0)
        
        # 计算SSIM
        ssim = kornia.metrics.ssim(warped_image, target_image, window_size=11)
        
        return {
            'psnr': psnr.item(),
            'ssim': ssim.mean().item()
        }

class ImageStitcher:
    """
    图像拼接器
    用于实际执行图像拼接任务
    """
    def __init__(self, model: torch.nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理输入图像"""
        # 调整图像大小
        image = cv2.resize(image, (self.config['img_size'], self.config['img_size']))
        
        # 转换为RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 归一化和转换为tensor
        image = to_tensor(image)
        image = F.normalize(image, 
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        
        return image
    
    @torch.no_grad()
    def stitch_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """拼接两张图像"""
        # 预处理
        tensor1 = self.preprocess_image(img1).unsqueeze(0).to(self.device)
        tensor2 = self.preprocess_image(img2).unsqueeze(0).to(self.device)
        
        # 模型推理
        outputs = self.model(tensor1, tensor2)
        
        # 获取匹配点
        matched_points = self._get_matched_points(outputs)
        
        # 估计变换矩阵
        H = self._estimate_transform(matched_points)
        
        # 图像融合
        stitched_image = self._blend_images(img1, img2, H)
        
        return stitched_image, {
            'matched_points': matched_points,
            'homography': H,
            'confidence_scores': outputs['matching_scores'].cpu().numpy()
        }
    
    def _get_matched_points(self, outputs: Dict) -> np.ndarray:
        """从模型输出中获取匹配点对"""
        scores = torch.sigmoid(outputs['matching_scores'])
        matches = scores > self.config.get('matching_threshold', 0.5)
        
        # 转换为图像坐标
        matched_indices = torch.nonzero(matches).cpu().numpy()
        matched_points = []
        
        for idx1, idx2 in matched_indices:
            pt1 = self._index_to_coordinate(idx1)
            pt2 = self._index_to_coordinate(idx2)
            matched_points.append((pt1, pt2))
            
        return np.array(matched_points)

    def _estimate_transform(self, matched_points: np.ndarray) -> np.ndarray:
        """估计变换矩阵"""
        if len(matched_points) < 4:
            raise ValueError("匹配点数量不足，无法估计单应性矩阵")
            
        H, mask = cv2.findHomography(
            matched_points[:, 0],
            matched_points[:, 1],
            cv2.RANSAC,
            5.0
        )
        
        return H

    def _blend_images(self, img1: np.ndarray, img2: np.ndarray, 
                     H: np.ndarray) -> np.ndarray:
        """融合图像"""
        # 计算变换后图像的大小
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        corners2_warped = cv2.perspectiveTransform(corners2, H)
        corners = np.concatenate([corners1, corners2_warped], axis=0)
        
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        
        # 创建输出图像
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        img1_warped = cv2.warpPerspective(img1, Ht, (xmax-xmin, ymax-ymin))
        img2_warped = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
        
        # 简单的加权融合
        mask1 = (img1_warped != 0).all(axis=2)
        mask2 = (img2_warped != 0).all(axis=2)
        overlap = mask1 & mask2
        
        # 在重叠区域进行加权融合
        result = img1_warped.copy()
        result[overlap] = (img1_warped[overlap] * 0.5 + img2_warped[overlap] * 0.5).astype(np.uint8)
        result[~mask1 & mask2] = img2_warped[~mask1 & mask2]
        
        return result

def stitch_video_sequence(video_path: str, output_path: str, model: nn.Module, config: Dict):
        """
        拼接完整的视频序列
        
        参数:
            video_path: 输入视频路径
            output_path: 输出全景图保存路径
            model: 训练好的拼接模型
            config: 配置参数
        """
        # 创建视频帧提取器
        extractor = VideoFrameExtractor(
            overlap_ratio=0.3,  # 相邻帧30%的重叠率
            frame_interval=5    # 每隔5帧采样一帧
        )
        
        # 提取视频帧
        print("正在提取视频帧...")
        frame_pairs = extractor.extract_frames(
            video_path=video_path,
            output_dir="temp_frames"
        )
        
        # 创建拼接器
        stitcher = ImageStitcher(model, config)
        
        # 初始化全景图为第一帧
        panorama = cv2.imread(str(frame_pairs[0][0]))
        
        # 逐帧拼接
        print("开始拼接过程...")
        for i, (frame1_path, frame2_path) in enumerate(frame_pairs):
            print(f"正在处理第 {i+1}/{len(frame_pairs)} 对帧...")
            
            # 读取当前帧
            current_frame = cv2.imread(str(frame2_path))
            
            # 将当前帧与全景图拼接
            panorama, info = stitcher.stitch_images(panorama, current_frame)
            
            # 可选：保存中间结果
            if (i + 1) % 10 == 0:  # 每处理10帧保存一次
                cv2.imwrite(f"panorama_checkpoint_{i+1}.jpg", panorama)
        
        # 保存最终全景图
        cv2.imwrite(output_path, panorama)
        print(f"拼接完成! 全景图已保存至: {output_path}")
        
        # 清理临时文件
        import shutil
        shutil.rmtree("temp_frames")

if __name__ == "__main__":
    # 配置参数
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'img_size': 224,
        'matching_threshold': 0.5,
        'learning_rate': 1e-4,
        'epochs': 100
    }
    
    # 训练阶段
    print("===== 开始训练阶段 =====")
    model = ImageStitchingTransformer()
    train_loader, val_loader = create_dataloaders(
        video_path="video\\2287_raw.MP4",
        output_dir="train_frames",
        batch_size=8,
        img_size=config['img_size']
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()
    
    # 拼接阶段
    print("\n===== 开始拼接阶段 =====")
    stitch_video_sequence(
        video_path="video\\2287_raw.MP4",  # 测试视频
        output_path="final_panorama.jpg",     # 输出全景图
        model=model,
        config=config
    )