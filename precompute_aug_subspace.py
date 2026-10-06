"""算法 C 前置: 扰动视觉子空间 B_aug 预计算 (每数据集一次, 串行)

对源域图像采样, 池化 7 种 PIL 级扰动的特征位移 (f(T(x))−f(x)),
SVD 取同秩 32 维 → cache/aug_subspace_{ds}.pt (B_aug: 768×32, 列正交)

用法: python precompute_aug_subspace.py CUB AWA2 ... (默认全部七数据集串行)
"""
import os
import sys

import torch
import clip
import torchvision.transforms as T
from PIL import Image, ImageOps

from args import get_args
from reliability import compute_domain_diffs_and_scores  # noqa: F401 (确保 import 链可用)

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
model, preprocess = clip.load(args.CLIP_type, device=device)
model.eval()

PERTURBS_PIL = {
    "solarize":     lambda im: ImageOps.solarize(im, threshold=128),
    "equalize":     ImageOps.equalize,
    "posterize":    lambda im: ImageOps.posterize(im, 3),
    "autocontrast": ImageOps.autocontrast,
    "grayscale":    lambda im: im.convert("L").convert("RGB"),
    "blur":         T.GaussianBlur(kernel_size=7, sigma=(3.0, 5.0)),
    "colorjitter":  T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
}

# 源域图片根目录与采样密度 (不依赖类名, 直接遍历子目录)
SRC_DIRS = {
    "CUB":        ("data/CUB/CUB_200_2011/images", 120, 6),
    "AwA2":       ("data/awa2/AwA2-data/JPEGImages", 50, 8),
    "LADA":       ("data/LAD_animal/animals", 50, 8),
    "LADV":       ("data/LAD_vehicle/vehicles", 40, 8),
    "PACS":       ("data/pacs/PACS/photo", 7, 20),
    "OfficeHome": ("data/pacs/OfficeHome/Real_World", 65, 8),
    "DomainNet":  ("data/domainnet/images/real", 200, 4),
}
RANK = 32


@torch.no_grad()
def collect_deltas(img_root, max_dirs, per_class):
    deltas = []
    subs = sorted([d for d in os.listdir(img_root)
                   if os.path.isdir(os.path.join(img_root, d))])[:max_dirs]
    for sub in subs:
        d = os.path.join(img_root, sub)
        files = sorted([f for f in os.listdir(d)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:per_class]
        for fn in files:
            try:
                pil = Image.open(os.path.join(d, fn)).convert("RGB")
                img = preprocess(pil).unsqueeze(0).to(device)
                f0 = model.encode_image(img).float()[0]
            except Exception:
                continue
            for name, tf in PERTURBS_PIL.items():
                try:
                    img_k = preprocess(tf(pil)).unsqueeze(0).to(device)
                    fk = model.encode_image(img_k).float()[0]
                    dv = fk - f0
                    deltas.append(dv / (dv.norm() + 1e-8))
                except Exception:
                    continue
    return torch.stack(deltas) if deltas else None


def main():
    datasets = os.environ.get("PRECOMP_DS", ",").split(",") if os.environ.get("PRECOMP_DS") else list(SRC_DIRS.keys())
    for ds in datasets:
        out_path = f"cache/aug_subspace_{ds}.pt"
        if os.path.exists(out_path):
            print(f"[skip] {out_path} 已存在")
            continue
        img_root, max_dirs, per_class = SRC_DIRS[ds]
        if not os.path.isdir(img_root):
            print(f"[err] {ds}: 源目录不存在 {img_root}")
            continue
        print(f"[{ds}] 采样扰动位移 ({img_root})...", flush=True)
        D = collect_deltas(img_root, max_dirs, per_class)
        if D is None or len(D) < 64:
            print(f"[err] {ds}: 位移样本不足 ({0 if D is None else len(D)})")
            continue
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        ev = S ** 2
        cum = torch.cumsum(ev, 0) / ev.sum()
        m = int(torch.searchsorted(cum, torch.tensor(0.95, device=cum.device)).item()) + 1
        m = min(m, RANK)
        B_aug = Vh[:m].T.contiguous().cpu()      # (768, m)
        torch.save({"B_aug": B_aug, "m": m, "n_deltas": len(D)}, out_path)
        print(f"[done] {ds}: {len(D)} 个位移 → B_aug {tuple(B_aug.shape)} → {out_path}", flush=True)


if __name__ == "__main__":
    main()
