"""
图像超分辨率模型：SRCNN baseline + SRResNet 改进版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN: Image Super-Resolution Using Deep Convolutional Networks
    结构：Patch extraction → Non-linear mapping → Reconstruction
    参数量：~57K
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale = scale_factor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    """残差块：Conv → BN → ReLU → Conv → BN → +skip"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class SRResNet(nn.Module):
    """
    改进版超分辨率网络，基于残差学习
    结构与 SRCNN 相同输入（bicubic 放大后的 LR），但更深、带残差连接
    结构：Conv → 8×ResBlock → Conv → +GlobalSkip
    参数量：~560K
    """
    def __init__(self, scale_factor=2, num_blocks=8, channels=64):
        super().__init__()
        self.scale = scale_factor

        # 输入层
        self.conv_input = nn.Conv2d(3, channels, 3, padding=1)

        # 残差块堆叠
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # 中间过渡
        self.conv_mid = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn_mid = nn.BatchNorm2d(channels)

        # 输出层（不改变空间尺寸，专注细化 bicubic 结果）
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        residual = self.conv_input(x)
        out = self.res_blocks(residual)
        out = self.bn_mid(self.conv_mid(out))
        out = out + residual  # 全局残差连接
        out = self.conv_out(out)
        return out
