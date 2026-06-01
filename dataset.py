"""
数据集加载：图像超分辨率
训练集：用户已有的 COCO 图片（397张，无需额外下载）
验证集：Set14 经典测试集（自动下载，~10MB）
"""

import os
import random
import zipfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def prepare_data(data_root="./data"):
    """
    准备数据集：
    - 训练集：复用已有 COCO 图片
    - 验证集：下载 Set14
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # ---- 训练集：COCO ----
    coco_candidates = [
        Path.home() / ".cache" / "coco_full" / "train2017_kb",
        Path("F:/openhanako/yolov8-seg/datasets/kb_mouse_full/images/train"),
        Path("F:/openhanako/yolov8-seg/datasets/kb_final/images/train"),
    ]
    train_dir = None
    for d in coco_candidates:
        if d.exists():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if len(imgs) >= 20:
                train_dir = d
                print(f"训练集: {d} ({len(imgs)} 张)")
                break

    if train_dir is None:
        # 最后回退：自动生成合成训练数据
        print("未找到现有图片，生成合成训练数据...")
        train_dir = data_root / "synth_train"
        train_dir.mkdir(parents=True, exist_ok=True)
        _generate_synth(train_dir, 200)

    # ---- 验证集：Set14 ----
    set14_dir = data_root / "Set14"
    if not set14_dir.exists() or len(list(set14_dir.glob("*.png"))) < 10:
        _download_set14(set14_dir, data_root)

    val_files = list(set14_dir.glob("*.png")) + list(set14_dir.glob("*.bmp"))
    print(f"验证集: Set14 ({len(val_files)} 张)")

    return str(train_dir), str(set14_dir)


def _generate_synth(out_dir, count):
    """生成合成训练图片（带纹理的随机图）"""
    import numpy as np
    for i in range(count):
        arr = np.random.randint(0, 256, (256 + random.randint(0, 256), 256 + random.randint(0, 256), 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 4)))
        img.save(out_dir / f"synth_{i:04d}.png")
    print(f"  合成数据: {count} 张")


def _download_set14(set14_dir, data_root):
    """下载 Set14 经典超分测试集"""
    set14_dir.mkdir(parents=True, exist_ok=True)
    try:
        url = "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14_SR.zip"
        zip_path = data_root / "Set14_SR.zip"
        print(f"下载 Set14 (~10MB)...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Set14_SR.zip 内部结构可能是 Set14_SR/image_SRF_2/HR/*.png
            for member in zf.namelist():
                if member.endswith(".png") and "HR" in member:
                    zf.extract(member, data_root)
                    src = data_root / member
                    dst = set14_dir / Path(member).name
                    if src.exists():
                        src.rename(dst)
            # 清理
            for d in data_root.glob("Set14_SR*"):
                if d.is_dir():
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
        os.remove(zip_path)
        print(f"  Set14 下载完成 ({len(list(set14_dir.glob('*.png')))} 张)")
    except Exception as e:
        print(f"  Set14 下载失败: {e}。使用合成验证数据。")
        _generate_synth(set14_dir, 14)


# ============================================================
# 训练数据集
# ============================================================
class SuperResTrainDataset(Dataset):
    """从 HR 图片随机裁剪 patch，生成 LR-HR 训练对"""

    def __init__(self, img_dir, scale=2, patch_size=96, augment=True):
        self.img_dir = Path(img_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.lr_patch = patch_size
        self.hr_patch = patch_size * scale

        self.images = sorted(
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.png")) +
            list(self.img_dir.glob("*.bmp"))
        )
        if not self.images:
            raise RuntimeError(f"未在 {img_dir} 找到图片")
        print(f"  训练图片文件: {len(self.images)} 张")

    def __len__(self):
        return len(self.images) * 30  # 每张图取 30 个随机块

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        try:
            hr_img = Image.open(self.images[img_idx]).convert("RGB")
        except Exception:
            hr_img = Image.new("RGB", (256, 256), (128, 128, 128))

        w, h = hr_img.size
        if w < self.hr_patch + 1 or h < self.hr_patch + 1:
            ratio = max(self.hr_patch / w, self.hr_patch / h) * 1.1
            hr_img = hr_img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)
            w, h = hr_img.size

        x = random.randint(0, max(0, w - self.hr_patch - 1))
        y = random.randint(0, max(0, h - self.hr_patch - 1))
        hr_patch = hr_img.crop((x, y, x + self.hr_patch, y + self.hr_patch))

        # 生成 LR：下采样再插值放大
        lr_patch = hr_patch.resize((self.lr_patch, self.lr_patch), Image.BICUBIC)
        lr_patch = lr_patch.resize((self.hr_patch, self.hr_patch), Image.BICUBIC)

        # 数据增强
        if self.augment:
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
                lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                k = random.randint(0, 3)
                hr_patch = hr_patch.rotate(90 * k, expand=False)
                lr_patch = lr_patch.rotate(90 * k, expand=False)

        to_tensor = transforms.ToTensor()
        return to_tensor(lr_patch), to_tensor(hr_patch)


# ============================================================
# 验证数据集
# ============================================================
class SuperResValDataset(Dataset):
    """验证集：整张图片推理"""

    def __init__(self, img_dir, scale=2):
        self.img_dir = Path(img_dir)
        self.scale = scale
        self.images = sorted(
            list(self.img_dir.glob("*.png")) +
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.bmp"))
        )
        if not self.images:
            raise RuntimeError(f"未在 {img_dir} 找到验证图片")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        hr_img = Image.open(self.images[idx]).convert("RGB")
        w, h = hr_img.size
        new_w = (w // self.scale) * self.scale
        new_h = (h // self.scale) * self.scale
        hr_img = hr_img.crop((0, 0, max(new_w, self.scale), max(new_h, self.scale)))

        lr_size = (hr_img.size[0] // self.scale, hr_img.size[1] // self.scale)
        lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

        return self.to_tensor(lr_img), self.to_tensor(hr_img)


def get_dataloaders(train_dir, val_dir, scale=2, patch_size=96, batch_size=16, num_workers=2):
    """返回 train_loader 和 val_loader"""
    train_set = SuperResTrainDataset(train_dir, scale, patch_size, augment=True)
    val_set = SuperResValDataset(val_dir, scale)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"每 epoch: {len(train_loader)} batch (batch_size={batch_size})")
    return train_loader, val_loader
