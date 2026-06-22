# 图像超分辨率重建

基于深度学习的图像超分辨率重建，将低分辨率图像恢复为高分辨率清晰图像。

## 项目概述

超分辨率重建是计算机视觉中的经典任务，目标是从低分辨率输入中恢复高频细节。本项目实现了完整的训练、评估、推理流程，并包含答辩 PPT 和演讲脚本。

## 模型架构

输入低分辨率图像，通过深度卷积网络学习高分辨率映射，输出放大后的清晰图像。模型设计重点：

- **残差学习**：网络学习的是高分辨率与低分辨率上采样之间的残差，而非完整图像，降低了学习难度
- **亚像素卷积**：在特征空间完成上采样，比直接插值保留更多高频信息
- **感知损失**：结合像素级损失与感知损失，确保恢复图像在视觉上自然

## 技术栈

- **框架**：PyTorch
- **数据处理**：自定义 Dataset，图像增强
- **模型**：深度残差网络 + 亚像素卷积
- **评估指标**：PSNR、SSIM
- **可视化**：训练曲线、对比效果图

## 项目结构

```
super_resolution/
├── train.py                  # 训练脚本
├── model.py                  # 模型定义
├── dataset.py                # 数据加载
├── eval.py                   # 评估脚本
├── infer.py                  # 推理脚本
├── make_ppt_final.py         # PPT 自动生成脚本
├── requirements.txt          # 依赖
├── 提词器_演讲脚本.md        # 答辩演讲脚本
├── super_resolution_final.pptx # 答辩 PPT
├── checkpoints/              # 模型检查点
├── results/                  # 输出结果
└── submission/               # 提交文件
```

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python train.py

# 评估
python eval.py

# 推理单张图片
python infer.py --input low_res.png --output high_res.png

# 生成答辩 PPT
python make_ppt_final.py
```

## 评估指标

| 指标 | 说明 |
|------|------|
| PSNR | 峰值信噪比，衡量像素级重建质量 |
| SSIM | 结构相似性，衡量感知重建质量 |

## 答辩材料

包含完整的答辩 PPT（Python 自动生成）和演讲脚本（提词器格式），覆盖：
- 问题定义与背景
- 技术路线与模型设计
- 实验结果与分析
- 不足与改进方向
