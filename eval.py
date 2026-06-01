"""
评估脚本：在测试图片上运行推理，生成对比可视化
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from model import SRCNN, SRResNet


def bicubic_upscale(img, scale):
    """双三次插值上采样（传统方法，作为对比基准）"""
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)


def upsample_with_model(model, img_tensor, device):
    """使用深度学习模型进行超分辨率"""
    with torch.no_grad():
        model.eval()
        sr = model(img_tensor.unsqueeze(0).to(device))
        sr = torch.clamp(sr, 0, 1)
    return sr.squeeze(0).cpu()


def tensor_to_pil(tensor):
    """tensor → PIL Image"""
    return transforms.ToPILImage()(tensor)


def plot_result_grid(save_dir, lr_img, bicubic_img, srcnn_img, srresnet_img, hr_img, scale, idx):
    """绘制四栏对比图：LR / Bicubic / SRCNN / SRResNet / HR"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    images = [
        (lr_img, f"低分辨率输入\n({lr_img.size[0]}x{lr_img.size[1]})"),
        (bicubic_img, f"Bicubic 插值\n(传统方法)"),
        (srcnn_img, "SRCNN\n(基线模型)"),
        (srresnet_img, "SRResNet\n(改进模型)"),
        (hr_img, f"原始高分辨率\n({hr_img.size[0]}x{hr_img.size[1]})"),
        (None, "局部细节放大"),
    ]

    for ax, (img, title) in zip(axes.flat, images):
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # 在最后一格放局部放大对比（Bicubic vs SRResNet 的细节区域）
    ax_detail = axes[1, 2]
    # 取中间区域放大
    bw, bh = bicubic_img.size
    box = (bw // 4, bh // 4, bw * 3 // 4, bh * 3 // 4)
    detail_bic = bicubic_img.crop(box)
    detail_res = srresnet_img.crop(box)
    detail_hr = hr_img.crop(box)

    # 拼接细节对比
    w_d = box[2] - box[0]
    h_d = box[3] - box[1]
    detail_canvas = Image.new("RGB", (w_d * 3, h_d))
    detail_canvas.paste(detail_bic, (0, 0))
    detail_canvas.paste(detail_res, (w_d, 0))
    detail_canvas.paste(detail_hr, (w_d * 2, 0))
    ax_detail.imshow(detail_canvas)
    ax_detail.set_title("细节对比 (Bicubic | SRResNet | HR)", fontsize=11)
    ax_detail.axis("off")

    plt.suptitle(f"图像超分辨率对比 (x{scale})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = save_dir / f"comparison_{idx:02d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  对比图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="图像超分辨率评估")
    parser.add_argument("--srcnn_ckpt", type=str, default="./checkpoints/srcnn_best.pth")
    parser.add_argument("--srresnet_ckpt", type=str, default="./checkpoints/srresnet_best.pth")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num_demo", type=int, default=4, help="演示图片数量")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    srcnn = SRCNN(scale_factor=args.scale).to(device)
    srresnet = SRResNet(scale_factor=args.scale, num_blocks=8).to(device)

    if os.path.exists(args.srcnn_ckpt):
        srcnn.load_state_dict(torch.load(args.srcnn_ckpt, map_location=device))
        print(f"SRCNN 权重已加载: {args.srcnn_ckpt}")
    else:
        print(f"警告: SRCNN 权重未找到 ({args.srcnn_ckpt})")

    if os.path.exists(args.srresnet_ckpt):
        srresnet.load_state_dict(torch.load(args.srresnet_ckpt, map_location=device))
        print(f"SRResNet 权重已加载: {args.srresnet_ckpt}")
    else:
        print(f"警告: SRResNet 权重未找到 ({args.srresnet_ckpt})")

    # 找测试图片（优先 COCO 真实图片）
    data_root = Path(args.data_root)
    coco_dirs = [
        Path.home() / ".cache" / "coco_full" / "train2017_kb",
        Path("F:/openhanako/yolov8-seg/datasets/kb_mouse_full/images/train"),
        Path("F:/openhanako/yolov8-seg/datasets/kb_final/images/train"),
    ]
    val_dirs = [
        data_root / "Set14",
        data_root / "DIV2K_valid_HR",
        data_root / "valid_HR",
    ]
    hr_files = []
    # 优先用 COCO 真实图片
    for d in coco_dirs:
        if d.exists():
            found = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if len(found) >= 4:
                hr_files = found
                break
    # 回退到 Set14
    if not hr_files:
        for d in val_dirs:
            if d.exists():
                hr_files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.bmp")))
                if hr_files:
                    break

    if not hr_files:
        print("错误: 未找到验证图片，请先运行 train.py 下载数据")
        return

    # 随机选几张做演示
    import random
    random.seed(42)
    hr_files = random.sample(hr_files, min(args.num_demo, len(hr_files)))
    to_tensor = transforms.ToTensor()

    print(f"\n对 {len(hr_files)} 张图片进行推理演示...")

    for i, hr_path in enumerate(hr_files):
        print(f"\n[图片 {i+1}/{len(hr_files)}] {hr_path.name}")

        hr_img = Image.open(hr_path).convert("RGB")
        w, h = hr_img.size
        new_w = (w // args.scale) * args.scale
        new_h = (h // args.scale) * args.scale
        hr_img = hr_img.crop((0, 0, new_w, new_h))

        # 生成低分辨率
        lr_img = hr_img.resize(
            (new_w // args.scale, new_h // args.scale), Image.BICUBIC
        )

        # 上采样到目标尺寸
        lr_upsampled = lr_img.resize((new_w, new_h), Image.BICUBIC)
        bicubic_img = bicubic_upscale(lr_img, args.scale)

        lr_tensor = to_tensor(lr_upsampled)
        srcnn_img = tensor_to_pil(upsample_with_model(srcnn, lr_tensor, device))
        srresnet_img = tensor_to_pil(upsample_with_model(srresnet, lr_tensor, device))

        # 保存单独的输出图
        srresnet_img.save(save_dir / f"sr_{i+1:02d}_srresnet.png")
        srcnn_img.save(save_dir / f"sr_{i+1:02d}_srcnn.png")
        bicubic_img.save(save_dir / f"sr_{i+1:02d}_bicubic.png")
        hr_img.save(save_dir / f"sr_{i+1:02d}_original.png")

        # 绘制对比图
        plot_result_grid(
            save_dir, lr_img, bicubic_img, srcnn_img, srresnet_img, hr_img,
            args.scale, i + 1
        )

    print(f"\n全部推理完成，结果保存在: {save_dir}")


if __name__ == "__main__":
    main()
