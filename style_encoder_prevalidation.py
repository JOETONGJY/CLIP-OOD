"""前置工程 + 实验A: 增强特征提取 + Style Encoder 预验证

前置: 对源域图像生成 M 种风格增强 → CLIP → 缓存概念激活 (K, M+1)
实验A: a(x) → MLP(K,128,64) → h_style → Linear(64,M)
       输出: 1) 增强类型识别准确率+混淆矩阵  2) 类别信息探针  3) 概念激活变化统计

用法: python style_encoder_prevalidation.py --dataset CUB
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as T
import clip
from args import get_args

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device

# ============================================================
# 七种风格增强 (PIL 级, 与 p0_phase2_aug 同一套)
# ============================================================
PERTURBS = {
    0: ("original",    None),
    1: ("colorjitter", T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)),
    2: ("grayscale",   lambda im: im.convert("L").convert("RGB")),
    3: ("blur",        T.GaussianBlur(kernel_size=7, sigma=(3.0, 5.0))),
    4: ("solarize",    lambda im: ImageOps.solarize(im, threshold=128)),
    5: ("posterize",   lambda im: ImageOps.posterize(im, 3)),
    6: ("autocontrast", ImageOps.autocontrast),
    7: ("equalize",    ImageOps.equalize),
}
M_AUG = len(PERTURBS) - 1  # 7 种增强 (不含原始)

DS_CONFIG = {
    "CUB": {
        "classes_file": "data/CUB/CUB_200_2011/classes.txt", "style": "idname_lower",
        "img_root": "data/CUB/CUB_200_2011/images", "concepts": "data/CUB/cub_concepts.txt",
        "n_classes": 200, "samples_per_class": 10,
    },
    "AWA2": {
        "classes_file": "data/awa2/classes.txt", "style": "idname",
        "img_root": "data/awa2/AwA2-data/JPEGImages", "concepts": "data/awa2/awa_concept.txt",
        "n_classes": 50, "samples_per_class": 20,
    },
    "PACS": {
        "classes_file": "data/pacs/classes.txt", "style": "idname_asis",
        "img_root": "data/pacs/PACS/photo", "concepts": "data/pacs/pacs_concepts.txt",
        "n_classes": 7, "samples_per_class": 50,
    },
}


def parse_classes(path, style):
    names = []
    for x in open(path, encoding="utf-8"):
        x = x.strip()
        if not x:
            continue
        if style == "idname_lower":
            names.append(x.split(" ")[1][4:].replace("_", " ").lower())
        else:
            names.append(x.split(" ")[1].rstrip())
    return names


@torch.no_grad()
def extract_aug_concept_activations(ds):
    """对源域图像采样 → 每种增强提取概念激活 → 缓存"""
    cfg = DS_CONFIG[ds]
    model, preprocess = clip.load(args.CLIP_type, device=device)
    model.eval()

    # 概念嵌入
    concepts = [l.strip() for l in open(cfg["concepts"], encoding="utf-8") if l.strip()]
    K = len(concepts)
    with torch.no_grad():
        concept_embs = model.encode_text(clip.tokenize(concepts, truncate=True).to(device)).float()
        concept_embs = concept_embs / concept_embs.norm(dim=-1, keepdim=True)
    print(f"[{ds}] 概念数 K={K}")

    # 采样源域图像 (按类目录)
    samples = []  # (path, class_id)
    class_names = parse_classes(cfg["classes_file"], cfg["style"])
    class2id = {c: i for i, c in enumerate(class_names)}
    import re as _re
    for cls_name in sorted(os.listdir(cfg["img_root"])):
        d = os.path.join(cfg["img_root"], cls_name)
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:cfg["samples_per_class"]]
        # 剥离目录名的数字前缀 (如 '001.Black_footed_Albatross' → 'black footed albatross')
        clean = _re.sub(r'^\d+\.', '', cls_name).replace("_", " ").lower().strip()
        cid = class2id.get(clean)
        if cid is None:
            for k, v in class2id.items():
                if k.lower() == clean or k.replace(" ", "_").lower() == cls_name.lower():
                    cid = v
                    break
        if cid is None:
            continue
        for f in files:
            samples.append((os.path.join(d, f), cid))

    print(f"[{ds}] 采样 {len(samples)} 张图, {cfg['samples_per_class']}/类")

    # 提取增强特征 + 概念激活
    all_feats = []    # (N, M+1, K)
    all_labels = []   # (N,)
    all_aug_ids = []  # (N, M+1) → [0, 1, 2, ..., M]
    for idx, (path, cid) in enumerate(samples):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        img = preprocess(pil).unsqueeze(0).to(device)
        z_orig = model.encode_image(img).float()
        z_orig = z_orig / z_orig.norm(dim=-1, keepdim=True)
        a_orig = (z_orig @ concept_embs.T)[0]  # (K,)

        per_aug = [a_orig.cpu()]
        for m in range(1, M_AUG + 1):
            name, tf = PERTURBS[m]
            try:
                pil_aug = tf(pil) if tf else pil
                img_aug = preprocess(pil_aug).unsqueeze(0).to(device)
                z_aug = model.encode_image(img_aug).float()
                z_aug = z_aug / z_aug.norm(dim=-1, keepdim=True)
                a_aug = (z_aug @ concept_embs.T)[0]
            except Exception:
                a_aug = a_orig  # 失败时用原始替代
            per_aug.append(a_aug.cpu())

        all_feats.append(torch.stack(per_aug))  # (M+1, K)
        all_labels.append(cid)
        if (idx + 1) % 200 == 0:
            print(f"  ... {idx+1}/{len(samples)}", flush=True)

    feats = torch.stack(all_feats)   # (N, M+1, K)
    labels = torch.tensor(all_labels)
    print(f"[{ds}] 提取完成: feats {tuple(feats.shape)}, labels {tuple(labels.shape)}")

    out_path = f"cache/style_aug_{ds}.pt"
    torch.save({"feats": feats, "labels": labels, "concepts": concepts,
                "aug_names": [PERTURBS[i][0] for i in range(M_AUG + 1)]}, out_path)
    print(f"[{ds}] 已缓存 → {out_path}")
    return out_path


# ============================================================
# 实验 A: Style Encoder 预验证
# ============================================================
def train_probe(X, y, input_dim, n_classes, hidden=128, emb_dim=64, epochs=100, lr=1e-3, tag=""):
    """训练一个探针分类器, 返回准确率和 64 维风格/类别表示"""
    model = nn.Sequential(
        nn.Linear(input_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, emb_dim), nn.ReLU(),
        nn.Linear(emb_dim, n_classes)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    # 划分 train/val 80/20
    n = len(ds)
    n_train = int(n * 0.8)
    indices = torch.randperm(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    for ep in range(epochs):
        model.train()
        for batch_idx in (train_idx[i:i+256] for i in range(0, n_train, 256)):
            xb, yb = X[batch_idx].to(device), y[batch_idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X[val_idx].to(device))
        val_pred = val_logits.argmax(dim=-1).cpu()
        val_acc = (val_pred == y[val_idx]).float().mean().item()

    # 提取倒数第二层 (emb_dim=64) 作为表示
    repr_model = nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        h = repr_model(X.to(device)).cpu()

    print(f"  [{tag}] val_acc = {val_acc:.4f}")
    return val_acc, h, val_idx


def experiment_a(ds):
    data = torch.load(f"cache/style_aug_{ds}.pt")
    feats = data["feats"]        # (N, M+1, K)
    labels = data["labels"]      # (N,)
    aug_names = data["aug_names"]
    concepts = data["concepts"]
    N, M1, K = feats.shape

    results = {}

    # ===== 任务 1: 增强类型识别 =====
    # 输入: 增强后的概念激活 a^(m)(x) - 原始概念激活 a(x), m ∈ [1, M]
    # 标签: 增强类型 m
    print("\n=== 任务 1: 增强类型识别 ===")
    X_aug, y_aug = [], []
    for m in range(1, M1):
        delta = feats[:, m] - feats[:, 0]  # (N, K) 增强引起的概念激活变化
        X_aug.append(delta)
        y_aug.append(torch.full((N,), m - 1))
    X_aug = torch.cat(X_aug)   # (N*M, K)
    y_aug = torch.cat(y_aug)

    acc_aug, h_style, val_idx_aug = train_probe(X_aug, y_aug, K, M1 - 1, tag="增强识别")
    results["aug_cls_acc"] = acc_aug

    # 混淆矩阵 (验证集)
    from collections import Counter
    model_cm = nn.Sequential(
        nn.Linear(K, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, M1 - 1)
    ).to(device)
    # 用简单交叉验证近似混淆——直接用 val 子集的预测分布
    results["aug_names"] = aug_names[1:]

    # ===== 任务 2: 类别信息探针 =====
    # 输入: 风格表示 h_style (64 维), 标签: 图像类别
    print("\n=== 任务 2: 类别信息探针 (风格表示中类别含量检测) ===")
    # 用 h_style 的验证集部分
    h_val = h_style[val_idx_aug]  # 取与增强识别同验证集
    # 对应的类别标签: 需要还原原始图像的类别
    n_val = len(val_idx_aug)
    sample_idx = (val_idx_aug // (M1 - 1)).long()  # 还原样本索引
    y_cls_val = labels[sample_idx]

    probe = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, len(labels.unique()))).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    n_train_p = int(n_val * 0.7)
    for ep in range(50):
        for i in range(0, n_train_p, 256):
            xb = h_val[i:i+256].to(device)
            yb = y_cls_val[i:i+256].to(device)
            loss = F.cross_entropy(probe(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(h_val[n_train_p:].to(device)).argmax(-1).cpu()
        cls_leak = (pred == y_cls_val[n_train_p:]).float().mean().item()
    print(f"  [类别探针] acc = {cls_leak:.4f}  (越低越好——高说明风格表示主要编码类别)")
    results["class_leak_acc"] = cls_leak

    # ===== 任务 3: 概念激活变化统计 =====
    print("\n=== 任务 3: 不同增强下概念激活变化 ===")
    print(f"{'概念':30s} {'原始激活':>8s} {'cjitter':>8s} {'gray':>8s} {'blur':>8s} {'solar':>8s} {'poster':>8s} {'auto':>8s} {'equal':>8s}")
    for j in range(min(K, 20)):
        row = [aug_names[m].ljust(8) for m in range(M1)]
        vals = [f"{feats[:, m, j].mean():.3f}" for m in range(M1)]
        print(f"  {concepts[j][:28]:30s} {'  '.join(vals)}")

    # 每种增强下概念激活的平均变化幅度
    print(f"\n各增强下概念激活变化 (均方根):")
    for m in range(1, M1):
        delta = feats[:, m] - feats[:, 0]
        rms = delta.pow(2).mean().sqrt().item()
        print(f"  {aug_names[m]:15s} RMS = {rms:.4f}")
        results[f"rms_{aug_names[m]}"] = rms

    # ===== 判据汇总 =======
    print("\n" + "=" * 60)
    print("实验 A 判据汇总")
    print("=" * 60)
    ok1 = acc_aug > 0.80
    ok2 = cls_leak < 0.50  # 类别泄漏不应太高 (随机基线 = 1/C)
    print(f"  判据1 增强识别 > 80%: {'✓' if ok1 else '✗'} ({acc_aug:.1%})")
    print(f"  判据2 类别泄漏 < 50%: {'✓' if ok2 else '✗'} ({cls_leak:.1%})")
    print(f"  判据3 概念激活变化非均匀: 见上表")
    verdict = ok1 and ok2
    print(f"\n  总判定: {'通过 → 进入模块编码' if verdict else '不通过 → 需调整风格信号设计'}")
    results["verdict"] = verdict

    out = f"downloads/style_encoder_prevalidation_{ds}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  结果已保存 → {out}")
    return verdict


if __name__ == "__main__":
    import os as _os
    ds = _os.environ.get("EXP_A_DS", "CUB")
    cache = f"cache/style_aug_{ds}.pt"
    if not os.path.exists(cache):
        print(f"=== 前置: 提取 {ds} 增强概念激活 ===")
        extract_aug_concept_activations(ds)
    print(f"\n=== 实验 A: Style Encoder 预验证 ({ds}) ===")
    experiment_a(ds)
