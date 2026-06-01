# 图像超分辨率 (Image Super-Resolution)

深度学习课内实践项目。对比 SRCNN 基线模型与 SRResNet 改进模型，实现 2× 图像超分辨率重建。

## 项目结构

```
super_resolution/
├── model.py           # SRCNN + SRResNet 模型定义
├── dataset.py         # 数据加载（COCO 397 张 + Set14）
├── train.py           # 训练脚本（自动对比双模型）
├── eval.py            # 评估脚本（生成 6 子图对比）
├── infer.py           # 命令行推理工具
├── train.py           # 训练脚本
├── super_resolution_final.pptx   # 答辩 PPT (McKinsey+Block 风格, 8页)
├── checkpoints/       # 模型权重
│   ├── srcnn_best.pth      # SRCNN 最佳 (47.99 dB, 81KB)
│   └── srresnet_best.pth   # SRResNet 最佳 (47.75 dB, 2.5MB)
├── results/           # 可视化输出
│   ├── comparison_01~04.png   # 4组完整对比图
│   └── sr_*.png               # 单独输出
├── dist/              # 打包发布
│   └── SR/            # 可执行文件夹（SR.exe + 依赖）
└── data/              # 数据集缓存
```

## 快速开始

### 环境

```bash
pip install torch torchvision pillow matplotlib numpy
```

### 训练

```bash
python train.py --epochs 30 --batch_size 16
```

### 推理

```bash
python infer.py input.jpg --model srresnet --output result.png
```

### 评估

```bash
python eval.py --num_demo 4
```

## 模型结构

| 模型 | 层数 | 参数量 | PSNR | 训练用时 |
|------|------|--------|------|----------|
| Bicubic (基准) | - | 0 | - | - |
| SRCNN (基线) | 3 | 57K | 47.99 dB | 11.6 min |
| SRResNet (改进) | 20+ | 560K | 47.75 dB | 53.3 min |

**SRCNN**: Conv(3→64, 9×9) → Conv(64→32, 1×1) → Conv(32→3, 5×5)

**SRResNet**: Conv(3→64) → 8×ResBlock → GlobalSkip → Conv(64→32→3)

## 核心改进

- **残差连接**：8 个 ResidualBlock + 全局跳跃连接，解决深层网络梯度消失
- **批归一化**：加速收敛，稳定训练
- **数据增强**：随机翻转 + 90° 旋转，提升泛化
- **数据复用**：使用已有 COCO 图片，零额外下载
