"""
图像超分辨率推理脚本
用法：python infer.py input.jpg [--model srresnet|srcnn] [--scale 2] [--output output.jpg]
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import SRCNN, SRResNet


def main():
    parser = argparse.ArgumentParser(description="图像超分辨率推理")
    parser.add_argument("input", type=str, help="输入图片路径（低分辨率）")
    parser.add_argument("--model", type=str, default="srresnet", choices=["srcnn", "srresnet"],
                        help="使用哪个模型")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4],
                        help="放大倍数")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认：同目录下 _sr.png）")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="模型权重路径（默认：./checkpoints/{model}_best.pth）")
    args = parser.parse_args()

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 加载模型 ----
    ckpt = args.ckpt or os.path.join(
        os.path.dirname(__file__), "checkpoints", f"{args.model}_best.pth"
    )
    if not os.path.exists(ckpt):
        print(f"错误: 模型权重未找到: {ckpt}")
        print("请先运行 train.py 训练模型，或用 --ckpt 指定权重路径")
        return

    if args.model == "srcnn":
        model = SRCNN(scale_factor=args.scale)
    else:
        model = SRResNet(scale_factor=args.scale)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型: {args.model.upper()} | 参数: {param_count/1000:.0f}K | 放大: {args.scale}x")

    # ---- 读取图片 ----
    if not os.path.exists(args.input):
        print(f"错误: 输入图片不存在: {args.input}")
        return

    img = Image.open(args.input).convert("RGB")
    w, h = img.size
    print(f"输入: {w}x{h}")

    # ---- 预处理 ----
    # 1. 对齐到 scale 的倍数（避免输出尺寸不对）
    new_w = (w // args.scale) * args.scale * args.scale
    new_h = (h // args.scale) * args.scale * args.scale
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"  对齐后: {new_w}x{new_h}")

    # 2. Bicubic 放大到目标尺寸（作为模型输入）
    target_w = new_w * args.scale
    target_h = new_h * args.scale
    lr_img = img.resize((new_w // args.scale, new_h // args.scale), Image.BICUBIC)
    lr_upsampled = lr_img.resize((target_w, target_h), Image.BICUBIC)

    # 3. 转 tensor
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_upsampled).unsqueeze(0).to(device)

    # ---- 推理 ----
    print(f"推理中... 目标分辨率: {target_w}x{target_h}")
    t_start = time.time()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    t_elapsed = time.time() - t_start
    print(f"推理耗时: {t_elapsed*1000:.0f} ms")

    # ---- 保存 ----
    sr_img = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())

    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input).stem
        out_path = str(Path(args.input).parent / f"{stem}_sr.png")

    sr_img.save(out_path)
    file_size = os.path.getsize(out_path) / 1024
    print(f"输出: {out_path} ({sr_img.size[0]}x{sr_img.size[1]}, {file_size:.0f} KB)")


if __name__ == "__main__":
    main()
