"""P0 第二阶段: 逐种合成扰动对真实域偏移的解释力 (R_aug→real)

问题: solarize/posterize 等常规增强产生的特征变化, 能否近似真实 domain shift?
判据: R_augk→real = ‖P_augk D_real‖_F^2 / ‖D_real‖_F^2  (对每类扰动 k 单独算)
参照: R_text→real (语言子空间) 与 E_base (随机方向基准) 同表给出。

若 R_aug→real ≈ E_base 量级 → 增强方向与真实偏移无关 → 不可作为 L_visual 监督
若 R_aug→real >> E_base 且接近 R_text → 该增强可信
"""
import os
import torch
import clip
import torchvision.transforms as T
from PIL import Image
from args import get_args
from modality_misalignment2 import (DATASETS, parse_classes, class_conditional_real,
                                    svd_subspace)
from reliability import compute_domain_diffs_and_scores
from prompts.prompt200new import source_text_prompts, target_text_prompts

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
model, preprocess = clip.load(args.CLIP_type, device=device)
model.eval()

# PIL 层变换 (transform 作用在 PIL 图上, 再统一 preprocess)
from PIL import ImageOps
PERTURBS_PIL = {
    "solarize":     lambda im: ImageOps.solarize(im, threshold=128),
    "equalize":     ImageOps.equalize,
    "posterize":    lambda im: ImageOps.posterize(im, 3),
    "autocontrast": ImageOps.autocontrast,
    "grayscale":    lambda im: im.convert("L").convert("RGB"),
    "blur":         T.GaussianBlur(kernel_size=7, sigma=(3.0, 5.0)),
    "colorjitter":  T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
}

IMG_DIRS = {
    "CUB":        ("data/CUB/CUB_200_2011/images", 8),
    "AwA2":       ("data/awa2/AwA2-data/JPEGImages", 8),
    "LADA":       ("data/LAD_animal/animals", 8),
    "LADV":       ("data/LAD_vehicle/vehicles", 8),
    "PACS":       ("data/pacs/PACS/photo", 20),
    "OfficeHome": ("data/pacs/OfficeHome/Real_World", 8),
}


@torch.no_grad()
def collect_aug_deltas(img_dir, classes, per_class, max_classes=120):
    """逐图: 每种扰动的 f(T(x)) − f(x), 归一化后按扰动类型堆叠
    不依赖类名匹配: 直接取 img_dir 的前 max_classes 个子目录采样"""
    deltas = {k: [] for k in PERTURBS_PIL}
    subs = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])[:max_classes]
    for cls in subs:
        d = os.path.join(img_dir, cls)
        files = sorted([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:per_class]
        for fn in files:
            try:
                pil = Image.open(os.path.join(d, fn)).convert("RGB")
                img = preprocess(pil).unsqueeze(0).to(device)
            except Exception:
                continue
            f0 = model.encode_image(img).float()[0]
            for name, tf in PERTURBS_PIL.items():
                try:
                    img_k = preprocess(tf(pil)).unsqueeze(0).to(device)
                    fk = model.encode_image(img_k).float()[0]
                except Exception:
                    continue
                dv = fk - f0
                deltas[name].append(dv / (dv.norm() + 1e-8))
    return deltas


def proj_ratio(B, D):
    """‖P D‖²_F / ‖D‖²_F, P = B Bᵀ"""
    proj = (B @ B.T) @ D.T
    return (torch.norm(proj, p="fro") ** 2 / torch.norm(D.T, p="fro") ** 2).item()


import csv
rows = []
print(f'{"数据集":12s}' + "".join(f'{k:>13s}' for k in PERTURBS_PIL) + f'{"R_text":>9s}{"E_base":>8s}')
for ds, (clsfile, style, src, tgt) in DATASETS.items():
    if ds not in IMG_DIRS:
        continue
    try:
        D_real, _ = class_conditional_real(src, tgt)
        if D_real is None:
            continue
        # 参照: 语言子空间
        names = parse_classes(clsfile, style)
        diffs, _ = compute_domain_diffs_and_scores(
            model, source_text_prompts, target_text_prompts, names, device, verbose=False)
        dirs = diffs.mean(dim=1).float()
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        B_text, _ = svd_subspace(dirs)
        r_text = proj_ratio(B_text, D_real)
        z = torch.load(src, map_location="cpu")["feats"].float().to(device)
        z = z / z.norm(dim=-1, keepdim=True)
        e_base = (((z @ B_text) @ B_text.T * z).sum(-1)).mean().item()

        img_dir, per_class = IMG_DIRS[ds]
        deltas = collect_aug_deltas(img_dir, names, per_class)
        vals = {}
        for k, lst in deltas.items():
            if len(lst) < 32:
                vals[k] = float("nan")
                continue
            D_aug = torch.stack(lst).to(device)
            B_aug, m_aug = svd_subspace(D_aug, tau=0.95, cap=32)   # 同秩公平对比
            vals[k] = proj_ratio(B_aug, D_real)
        line = f"{ds:12s}" + "".join(f"{vals[k]:13.3f}" for k in PERTURBS_PIL) + f"{r_text:9.3f}{e_base:8.3f}"
        print(line)
        rows.append([ds] + [f"{vals[k]:.4f}" for k in PERTURBS] + [f"{r_text:.4f}", f"{e_base:.4f}"])
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{ds}: 失败 {e}")

with open("downloads/aug_vs_real.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset"] + list(PERTURBS_PIL) + ["R_text", "E_base"])
    w.writerows(rows)
print("完成 → downloads/aug_vs_real.csv")
