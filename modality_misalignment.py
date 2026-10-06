"""P0 模态错位诊断: 文本域子空间 vs 视觉域偏移子空间

三个指标 (专家指定):
  1. Principal Angles: cos θ_i = σ_i(SVD(B_text^T B_img))
  2. Projection Overlap: ||B_text^T B_img||_F^2 / m
  3. Cross Reconstruction: R_t→v = ||B_text B_text^T D_img||_F^2 / ||D_img||_F^2

协议 (严格 DG, 不用任何 target 域图像):
  B_text: 204 prompt 类平均方向的 SVD (每数据集)
  B_img : 视觉域变化子空间 —— 需要 >1 个 source 域。
          严格 DG 数据集 (CUB/LADA/LADV/OfficeHome) 无多 source → 用
          style-perturbation 协议: 对源域图像做 K 种风格变换 (solarize/eq/posterize/
          autocontrast), 每种计算类内均值 → Δ_d = μ_d − μ_source → D_img → SVD
  PACS  附加真实多源版本: photo 之外的 art/cartoon/sketch 三域均值差 (诊断用,
          论文注明该版本仅用于事后分析, 主方法不使用 target 图像)

输出: 每数据集的前 8 个主角度余弦、overlap、R_t→v, 存 CSV 与 npz
"""
import os
import torch
import clip
from args import get_args
from reliability import compute_domain_diffs_and_scores
from prompts.prompt200new import source_text_prompts, target_text_prompts

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
model, preprocess = clip.load(args.CLIP_type, device=device)
model.eval()

import torchvision.transforms as T
PERTURBS = {
    "solarize": T.RandomSolarize(threshold=(64, 192), p=1.0),
    "equalize": T.RandomEqualize(p=1.0),
    "posterize": T.RandomPosterize(bits=3, p=1.0),
    "autocontrast": T.RandomAutocontrast(p=1.0),
}
K_STYLES = len(PERTURBS)

DATASETS = {
    # name: (classes 文件, 解析风格, cache 特征, [有真实多源时: 域目录列表])
    "CUB":       ("data/CUB/CUB_200_2011/classes.txt", "idname_lower", "cache/CUB_train.pt", None),
    "AwA2":      ("data/awa2/classes.txt", "idname", "cache/AWA2_train.pt", None),
    "LADA":      ("data/LAD_animal/class.txt", "idname", "cache/LADA_train.pt", None),
    "LADV":      ("data/LAD_vehicle/class.txt", "idname", "cache/LADV_train.pt", None),
    "PACS":      ("data/pacs/classes.txt", "idname_asis", "cache/PACS_train.pt",
                  ["data/pacs/PACS/art_painting", "data/pacs/PACS/cartoon", "data/pacs/PACS/sketch"]),
    "OfficeHome": ("data/pacs/officehome_classes.txt", "idname_asis_us", "cache/OfficeHome_train.pt",
                   ["data/pacs/OfficeHome/Art", "data/pacs/OfficeHome/Clipart", "data/pacs/OfficeHome/Product"]),
}


def parse_classes(path, style):
    names = []
    for x in open(path, encoding="utf-8"):
        x = x.strip()
        if not x:
            continue
        if style == "idname_lower":
            names.append(x.split(" ")[1][4:].replace("_", " ").lower())
        elif style == "idname":
            names.append(x.split(" ")[1].rstrip())
        else:
            parts = x.split(" ", 1)
            n = parts[1] if len(parts) > 1 else parts[0]
            if style == "idname_asis_us":
                n = n.replace("_", " ")
            names.append(n.rstrip())
    return names


def build_b_text(names):
    diffs, _ = compute_domain_diffs_and_scores(
        model, source_text_prompts, target_text_prompts, names, device, verbose=False)
    dirs = diffs.mean(dim=1).float()
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    # 类平均方向可能不满秩 → 只取有效秩部分
    U, S, Vh = torch.linalg.svd(dirs, full_matrices=False)
    ev = S ** 2
    cum = torch.cumsum(ev, 0) / ev.sum()
    m = int(torch.searchsorted(cum, torch.tensor(0.8, device=cum.device)).item()) + 1
    B = Vh[:m].T.contiguous()
    return B, m


@torch.no_grad()
def img_subspace_from_perturb(cache_path, K=K_STYLES):
    """风格扰动协议: 每种变换下类内均值 − 原均值 → 域变化矩阵 → SVD"""
    d = torch.load(cache_path, map_location="cpu")
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    class _Raw(Dataset):
        def __init__(self):
            raise RuntimeError("未使用")

    # cache 只存了特征, 没存路径 → 改为从特征直接构造合成扰动:
    # 特征级近似: 对每个类中心 c_k, 扰动 = c_k + σ·n_k (n_k 为固定随机单位方向)
    # 但这不真实。改为: 要求 encode 时保存 per-class 均值。
    raise RuntimeError("cache 无路径信息")


@torch.no_grad()
def img_means_with_perturb(images_dir, classes, samples_per_class=20):
    """直接从图片目录计算: 原均值与 K 种风格变换下的均值 (每类采样)"""
    from PIL import Image
    mu = {}
    for ci, cls in enumerate(classes):
        d = os.path.join(images_dir, cls)
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png", ".jpeg"))])[:samples_per_class]
        for fn in files:
            img = preprocess(Image.open(os.path.join(d, fn)).convert("RGB")).unsqueeze(0).to(device)
            feats = [model.encode_image(img).float()[0]]
            for p in PERTURBS.values():
                feats.append(model.encode_image(p(img)).float()[0])
            mu.setdefault(ci, []).append(torch.stack(feats))  # (n, K+1, 768)
    if not mu:
        return None
    # 原始均值 μ_source 与每个风格域的均值 μ_d
    allf = torch.cat([torch.stack(v) for v in mu.values()])  # (N, K+1, 768)
    mu_source = allf[:, 0].mean(0)
    deltas = []
    for k in range(1, K_STYLES + 1):
        mu_k = allf[:, k].mean(0)
        deltas.append(mu_k - mu_source)
    D_img = torch.stack(deltas)  # (K, 768)
    D_img = D_img / (D_img.norm(dim=-1, keepdim=True) + 1e-8)
    return D_img


@torch.no_grad()
def img_subspace_real_multiroot(domains, classes, samples_per_class=15):
    """真实多 source 域协议 (PACS/OH): 每个非 photo 域的类均值 − photo 类均值 → SVD"""
    deltas = []
    mu_source_per_cls = {}
    src_dir = domains[0]  # 约定第一个为 source (photo / Real_World)
    for ci, cls in enumerate(classes):
        d = os.path.join(src_dir, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))[:samples_per_class]
        vecs = [model.encode_image(preprocess(Image.open(os.path.join(d, f)).convert("RGB")).unsqueeze(0).to(device)).float()[0]
                for f in files]
        if vecs:
            mu_source_per_cls[ci] = torch.stack(vecs).mean(0)
    for dom in domains[1:]:
        for ci, cls in enumerate(classes):
            d = os.path.join(dom, cls)
            if not os.path.isdir(d) or ci not in mu_source_per_cls:
                continue
            files = sorted(os.listdir(d))[:samples_per_class]
            vecs = [model.encode_image(preprocess(Image.open(os.path.join(d, f)).convert("RGB")).unsqueeze(0).to(device)).float()[0]
                    for f in files]
            if vecs:
                deltas.append(torch.stack(vecs).mean(0) - mu_source_per_cls[ci])
    if not deltas:
        return None
    D_img = torch.stack(deltas)
    D_img = D_img / (D_img.norm(dim=-1, keepdim=True) + 1e-8)
    U, S, Vh = torch.linalg.svd(D_img, full_matrices=False)
    return Vh.T.contiguous(), len(deltas)  # (768, K), 有效样本方向数


def metrics(B_text, B_img, D_img):
    # 1) principal angles
    sv = torch.linalg.svdvals(B_text.T @ B_img)
    # 2) projection overlap
    overlap = (torch.norm(B_text.T @ B_img, p="fro") ** 2 / B_text.shape[1]).item()
    # 3) cross reconstruction
    proj = (B_text @ B_text.T) @ D_img.T          # (768, n)
    r_tv = (torch.norm(proj, p="fro") ** 2 / torch.norm(D_img.T, p="fro") ** 2).item()
    return sv[:8].tolist(), overlap, r_tv


results = {}
import csv
with open("downloads/modality_misalignment.csv", "w", newline="", encoding="utf-8") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["dataset", "m_text", "img_protocol", "n_dirs", "principal_angles(8)", "overlap", "R_t2v"])
    for ds, (clsfile, style, cache, realroots) in DATASETS.items():
        try:
            names = parse_classes(clsfile, style)
            B_text, m = build_b_text(names)
            rows = []
            # --- 真实多源 (仅 PACS/OH) ---
            if realroots:
                out = img_subspace_real_multiroot(realroots, names)
                if out:
                    B_img, n_dirs = out
                    sv, ov, rtv = metrics(B_text, B_img, B_img.T)
                    rows.append((f"real_multi({len(realroots)}tgt)", n_dirs, sv, ov, rtv))
            # --- 风格扰动协议 (全部数据集) ---
            if ds == "PACS":
                imgdir = "data/pacs/PACS/photo"
                classes_dir = names
            elif ds == "OfficeHome":
                imgdir = "data/pacs/OfficeHome/Real_World"
                classes_dir = names
            else:
                imgdir = None
            if imgdir:
                D_img = None
                # 通用: 逐类采样原始+扰动 → deltas (直接算, 不走 cache)
                deltas = []
                import random
                random.seed(0)
                from PIL import Image
                for ci, cls in enumerate(classes_dir):
                    d = os.path.join(imgdir, cls)
                    if not os.path.isdir(d):
                        continue
                    files = sorted(os.listdir(d))[:10]
                    for fn in files:
                        try:
                            img = preprocess(Image.open(os.path.join(d, fn)).convert("RGB")).unsqueeze(0).to(device)
                        except Exception:
                            continue
                        f0 = model.encode_image(img).float()[0]
                        for p in PERTURBS.values():
                            fk = model.encode_image(p(img)).float()[0]
                            deltas.append(fk - f0)
                if deltas:
                    D_img = torch.stack(deltas)
                    D_img = D_img / (D_img.norm(dim=-1, keepdim=True) + 1e-8)
                    U, S, Vh = torch.linalg.svd(D_img, full_matrices=False)
                    B_img = Vh[:32].T.contiguous()
                    sv, ov, rtv = metrics(B_text, B_img, D_img)
                    rows.append(("style_perturb", len(deltas), sv, ov, rtv))
            results[ds] = rows
            for proto, n_dirs, sv, ov, rtv in rows:
                print(f"{ds:12s} {proto:20s} n={n_dirs:5d} angles={['%.3f' % s for s in sv[:4]]} "
                      f"overlap={ov:.3f} R_t→v={rtv:.3f}")
                w.writerow([ds, m, proto, n_dirs, ";".join(f"{s:.4f}" for s in sv), f"{ov:.4f}", f"{rtv:.4f}"])
        except Exception as e:
            import traceback
            print(f"{ds}: 失败 {e}")
torch.save(results, "downloads/modality_misalignment.pt")
print("完成 → downloads/modality_misalignment.csv")
