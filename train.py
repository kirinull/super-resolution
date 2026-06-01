"""
图像超分辨率训练脚本
训练 SRCNN (baseline) 和 SRResNet (改进版)，对比 PSNR/SSIM
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# 设置 matplotlib 中文字体
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from model import SRCNN, SRResNet
from dataset import prepare_data, get_dataloaders


# ============================================================
# 评估指标
# ============================================================
def psnr(pred, target):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = nn.functional.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ============================================================
# 训练一个 epoch
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ============================================================
# 验证
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        total_loss += criterion(sr, hr).item()
        total_psnr += psnr(sr, hr).item()
    n = len(loader)
    return total_loss / n, total_psnr / n


# ============================================================
# 训练主循环
# ============================================================
def train_model(model, train_loader, val_loader, device, args, model_name):
    """训练一个模型并记录指标"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "val_psnr": []}
    best_psnr = 0.0
    best_path = Path(args.save_dir) / f"{model_name}_best.pth"

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), best_path)
            print(f"  >>> 最佳模型已保存 (PSNR={best_psnr:.2f} dB)")

    total_time = time.time() - total_start
    print(f"\n{model_name} 训练完成！")
    print(f"最佳 PSNR: {best_psnr:.2f} dB")
    print(f"总用时: {total_time / 60:.1f} 分钟")

    return history, best_psnr


# ============================================================
# 绘制对比曲线
# ============================================================
def plot_comparison(hist_srcnn, hist_srresnet, save_path):
    """绘制 SRCNN vs SRResNet 对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax = axes[0]
    ax.plot(hist_srcnn["train_loss"], label="SRCNN Train", alpha=0.6)
    ax.plot(hist_srcnn["val_loss"], label="SRCNN Val", linewidth=2)
    ax.plot(hist_srresnet["train_loss"], label="SRResNet Train", alpha=0.6)
    ax.plot(hist_srresnet["val_loss"], label="SRResNet Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PSNR 曲线
    ax = axes[1]
    ax.plot(hist_srcnn["val_psnr"], label="SRCNN", linewidth=2, marker="o", markersize=3)
    ax.plot(hist_srresnet["val_psnr"], label="SRResNet", linewidth=2, marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比曲线已保存: {save_path}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="图像超分辨率训练")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip_srcnn", action="store_true", help="跳过 SRCNN 训练")
    parser.add_argument("--skip_srresnet", action="store_true", help="跳过 SRResNet 训练")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    # 准备数据
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n=== 准备数据集 ===")
    train_dir, val_dir = prepare_data(args.data_root)
    train_loader, val_loader = get_dataloaders(
        train_dir, val_dir, scale=args.scale, patch_size=args.patch_size,
        batch_size=args.batch_size, num_workers=args.workers
    )

    results = {}

    # --- 训练 SRCNN (baseline) ---
    if not args.skip_srcnn:
        model_srcnn = SRCNN(scale_factor=args.scale)
        hist_srcnn, best_psnr_srcnn = train_model(
            model_srcnn, train_loader, val_loader, device, args, "srcnn"
        )
        results["srcnn"] = {"hist": hist_srcnn, "psnr": best_psnr_srcnn}
    else:
        print("\n跳过 SRCNN 训练")

    # --- 训练 SRResNet (改进版) ---
    if not args.skip_srresnet:
        model_srresnet = SRResNet(scale_factor=args.scale, num_blocks=8, channels=64)
        hist_srresnet, best_psnr_srresnet = train_model(
            model_srresnet, train_loader, val_loader, device, args, "srresnet"
        )
        results["srresnet"] = {"hist": hist_srresnet, "psnr": best_psnr_srresnet}
    else:
        print("\n跳过 SRResNet 训练")

    # --- 绘制对比图 ---
    if "srcnn" in results and "srresnet" in results:
        plot_comparison(
            results["srcnn"]["hist"],
            results["srresnet"]["hist"],
            str(Path(args.results_dir) / "comparison_curves.png")
        )

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print("训练汇总")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name.upper():10s}  最佳 PSNR: {r['psnr']:.2f} dB")
    if len(results) == 2:
        delta = results["srresnet"]["psnr"] - results["srcnn"]["psnr"]
        print(f"  SRResNet 相比 SRCNN 提升: {delta:+.2f} dB")

    # 保存汇总信息
    summary_path = Path(args.results_dir) / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("图像超分辨率训练结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"放大倍数: {args.scale}x\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n\n")
        for name, r in results.items():
            f.write(f"{name.upper()}: PSNR = {r['psnr']:.2f} dB\n")
        if len(results) == 2:
            f.write(f"\nSRResNet 提升: {delta:+.2f} dB\n")
    print(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
