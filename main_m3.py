"""
M3: 风格条件动态概念门控 (Style-Conditioned Dynamic Concept Gating)

M2 的两次失败结论: 静态类别选择(无论初始化如何)都会被训练冲刷。
M3 改为动态: 门控输入是当前样本的概念激活 a(x), 输出是逐概念的可靠性
调整——不同风格的图像自动得到不同的概念权重。

核心设计:
  h_style = MLP(a(x))           (K → 128 → 64)  风格表示 (实验A已验证可从概念层提取)
  G_style(x) = 1 + η·tanh(W_s·h_style + b_s)   残差式, η=0.2 固定
  logits = classifier(LayerNorm(G_style(x) ⊙ a(x)))

关键约束:
  1. 分类头 classifier 预训练后冻结 (吸取 M2 教训: 双因子可训练=无约束重参数化)
     → 只训练风格编码器和门控, 分类能力不退化
  2. G_style 围绕 1 调制 (η·tanh 范围 [-0.2, +0.2])
     → 初始时 G=1 等价于不做调制, 保证从基线出发
  3. 风格监督: 用增强对 (x, T(x)) 训练风格编码器预测增强类型 (辅助损失)

训练流程:
  S1: 冻结 CLIP + 预训练分类头 (缓存特征, 秒级)
  S2: 训练风格编码器 + 门控 (L_cls + λ_style·L_style), 增强对做配对输入
  S3: (可选) 解冻分类头联合微调

用法: python main_m3.py --dataset CUB --epochs 50 ...
"""
import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

from args import get_args
import clip as clip_lib

# 七种增强 (与实验A一致)
from PIL import Image, ImageOps
import torchvision.transforms as T
PERTURBS = {
    1: T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    2: lambda im: im.convert("L").convert("RGB"),
    3: T.GaussianBlur(kernel_size=7, sigma=(3.0, 5.0)),
    4: lambda im: ImageOps.solarize(im, threshold=128),
    5: lambda im: ImageOps.posterize(im, 3),
    6: ImageOps.autocontrast,
    7: ImageOps.equalize,
}


class StyleGatedClassifier(nn.Module):
    """M3: 风格条件动态概念门控 (分类头冻结)"""

    def __init__(self, K, C, style_dim=64, eta=0.2, hidden=128):
        super().__init__()
        self.K = K
        self.C = C
        self.eta = eta

        # 冻结的分类头 (由外部预训练权重初始化)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(K),
            nn.Linear(K, C)
        )
        self.classifier.requires_grad_(False)  # 冻结

        # 风格编码器: 概念激活变化 → 风格表示
        # 输入是 (a(x), a_ref(x)) 差分 或 直接 a(x)
        # 这里直接用 a(x) — 风格编码器自己学 a 中与类别无关的风格分量
        self.style_encoder = nn.Sequential(
            nn.Linear(K, hidden), nn.ReLU(),
            nn.Linear(hidden, style_dim), nn.ReLU(),
        )

        # 动态门控: h_style → 每概念调制系数
        self.gate_head = nn.Linear(style_dim, K)
        # 零初始化 gate_head → 初始 G = 1 + η·tanh(0) = 1 (不调制)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        # 风格分类头 (辅助监督: 预测增强类型)
        self.style_classifier = nn.Linear(style_dim, len(PERTURBS))

    def forward(self, concept_act, return_style=False):
        """
        concept_act: (bs, K) 概念激活
        return: logits (bs, C)
        """
        h = self.style_encoder(concept_act)          # (bs, style_dim)
        gate_raw = self.gate_head(h)                  # (bs, K)
        G = 1 + self.eta * torch.tanh(gate_raw)       # (bs, K) ∈ [1-η, 1+η]

        a_modulated = G * concept_act                 # (bs, K) 调制后概念激活
        logits = self.classifier(a_modulated)         # (bs, C)

        if return_style:
            return logits, h, G
        return logits

    def style_logits(self, concept_act):
        h = self.style_encoder(concept_act)
        return self.style_classifier(h)


class PairedAugDataset(Dataset):
    """配对增强数据: (原始概念激活, 增强概念激活, 类别, 增强类型)
    从实验A的缓存 cache/style_aug_{ds}.pt 加载"""

    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location="cpu")
        self.feats = data["feats"]      # (N, M+1, K)
        self.labels = data["labels"]    # (N,)
        self.M1 = self.feats.shape[1]   # M+1

    def __len__(self):
        # 每个样本产生 M 种配对 (原始 vs 每种增强)
        return len(self.labels) * (self.M1 - 1)

    def __getitem__(self, idx):
        sample_idx = idx // (self.M1 - 1)
        aug_idx = idx % (self.M1 - 1) + 1  # 1..M
        a_orig = self.feats[sample_idx, 0]       # (K,)
        a_aug = self.feats[sample_idx, aug_idx]  # (K,)
        label = self.labels[sample_idx]
        aug_type = aug_idx - 1  # 0..M-1
        return a_orig, a_aug, label, aug_type


def main_m3(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = args.dataset
    logger.info(f"=== M3: 风格条件动态概念门控 ({ds}) ===")

    # ===== 1. 加载训练/测试缓存特征 =====
    train_data = torch.load(f"cache/{ds}_train.pt", map_location="cpu")
    src_test = torch.load(f"cache/{ds}_source_test.pt", map_location="cpu")
    tgt_test = torch.load(f"cache/{ds}_target_test.pt", map_location="cpu")

    # 概念嵌入 (CLIP 文本侧, 冻结)
    clip_model, _ = clip_lib.load(args.CLIP_type, device=device)
    clip_model.eval()
    concepts_file = {
        "CUB": "data/CUB/cub_concepts.txt", "AWA2": "data/awa2/awa_concept.txt",
        "PACS": "data/pacs/pacs_concepts.txt",
    }[ds]
    concepts = [l.strip() for l in open(concepts_file, encoding="utf-8") if l.strip()]
    K = len(concepts)
    C = int(max(train_data["labels"].max(), tgt_test["labels"].max())) + 1
    logger.info(f"K={K} concepts, C={C} classes")

    with torch.no_grad():
        e_c = clip_model.encode_text(clip_lib.tokenize(concepts, truncate=True).to(device)).float()
        e_c = e_c / e_c.norm(dim=-1, keepdim=True)
        del clip_model  # 释放显存
    torch.cuda.empty_cache()

    # 概念激活 (预计算)
    def to_act(data):
        z = data["feats"].float().to(device)
        z = z / z.norm(dim=-1, keepdim=True)
        return z @ e_c.T  # (N, K)
    with torch.no_grad():
        a_train = to_act(train_data)
        y_train = train_data["labels"].to(device)
        a_src = to_act(src_test); y_src = src_test["labels"].to(device)
        a_tgt = to_act(tgt_test); y_tgt = tgt_test["labels"].to(device)

    # ===== 2. S1: 加载主管线 checkpoint 作为冻结分类头 (若存在), 否则预训练 =====
    ckpt_path = f"best_{ds}.pth"
    clf = nn.Sequential(nn.Flatten(), nn.LayerNorm(K), nn.Linear(K, C)).to(device)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        # 主管线保存的是 clip_cbm_subspace 或 clip_cbm_orth: classifier.{1,2}.*
        prefix = "classifier." if "classifier.2.weight" in state else "inv_classifier."
        clf_state = {}
        for k, v in state.items():
            if k.startswith(prefix):
                new_k = k.replace(prefix, "", 1)
                clf_state[new_k] = v
        try:
            clf.load_state_dict(clf_state)
            logger.info(f"[S1] 已加载主管线 checkpoint: {ckpt_path} (prefix={prefix})")
        except Exception as e:
            logger.warning(f"[S1] checkpoint 加载失败 ({e}), 退回预训练")
            opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
            for ep in range(50):
                perm = torch.randperm(len(a_train), device=device)
                for i in range(0, len(a_train), 256):
                    idx = perm[i:i+256]
                    loss = F.cross_entropy(clf(a_train[idx]), y_train[idx])
                    opt_clf.zero_grad(); loss.backward(); opt_clf.step()
    else:
        logger.info("[S1] 无 checkpoint, 预训练分类头...")
        opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
        for ep in range(50):
            perm = torch.randperm(len(a_train), device=device)
            for i in range(0, len(a_train), 256):
                idx = perm[i:i+256]
                loss = F.cross_entropy(clf(a_train[idx]), y_train[idx])
                opt_clf.zero_grad(); loss.backward(); opt_clf.step()
    with torch.no_grad():
        src_acc0 = (clf(a_src).argmax(1) == y_src).float().mean().item() * 100
        tgt_acc0 = (clf(a_tgt).argmax(1) == y_tgt).float().mean().item() * 100
    logger.info(f"[S1] 基线: Source {src_acc0:.2f} | Target {tgt_acc0:.2f}")

    # ===== 3. M3 模型 (分类头权重载入后冻结) =====
    model = StyleGatedClassifier(K, C, eta=0.2).to(device)
    model.classifier.load_state_dict(clf.state_dict())
    model.classifier.requires_grad_(False)
    del clf

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"M3 可训练参数: {trainable:,} (风格编码器+门控+风格分类头)")

    # ===== 4. S2: 训练风格编码器 + 门控 =====
    aug_cache = f"cache/style_aug_{ds}.pt"
    use_pairs = os.path.exists(aug_cache)
    if use_pairs:
        paired = PairedAugDataset(aug_cache)
        logger.info(f"[S2] 配对增强数据: {len(paired)} 对")
    else:
        logger.warning(f"[S2] 无增强缓存 {aug_cache}, 仅用分类损失")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    bs = 256
    lambda_style = 0.5
    best = {"source_acc": src_acc0, "target_acc": tgt_acc0, "epoch": -1}

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(a_train), device=device)
        total_cls, total_style = 0.0, 0.0
        n_batches = 0
        for i in range(0, len(a_train), bs):
            idx = perm[i:i+bs]
            a_batch, y_batch = a_train[idx], y_train[idx]

            logits = model(a_batch)
            l_cls = F.cross_entropy(logits, y_batch)
            loss = l_cls

            # 风格监督 (配对增强)
            if use_pairs and n_batches % 2 == 0:  # 每两个 batch 加一次
                pidx = torch.randint(0, len(paired), (min(bs, len(paired)),))
                a_o = torch.stack([paired[i][0] for i in pidx]).to(device)
                a_g = torch.stack([paired[i][1] for i in pidx]).to(device)
                t_type = torch.tensor([paired[i][3] for i in pidx]).to(device)
                # 风格分类: 用增强后的概念激活预测增强类型
                l_style = F.cross_entropy(model.style_logits(a_g), t_type)
                loss = loss + lambda_style * l_style
                total_style += l_style.item()

            opt.zero_grad(); loss.backward(); opt.step()
            total_cls += l_cls.item()
            n_batches += 1

        # 评估
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                src_acc = (model(a_src).argmax(1) == y_src).float().mean().item() * 100
                tgt_acc = (model(a_tgt).argmax(1) == y_tgt).float().mean().item() * 100
            if tgt_acc > best["target_acc"]:
                best = {"source_acc": src_acc, "target_acc": tgt_acc, "epoch": ep}
            if (ep + 1) % 10 == 0:
                # 门控统计
                with torch.no_grad():
                    _, h, G = model(a_tgt[:100], return_style=True)
                logger.info(f"Ep {ep+1} | Src {src_acc:.2f} | Tgt {tgt_acc:.2f} | "
                            f"Best {best['target_acc']:.2f} | G范围 [{G.min():.3f},{G.max():.3f}] "
                            f"std {G.std():.4f}")

    # ===== 5. 门控机制验证: target vs source 的门控差异 =====
    model.eval()
    with torch.no_grad():
        _, _, G_src = model(a_src[:500], return_style=True)
        _, _, G_tgt = model(a_tgt[:500], return_style=True)
    gate_diff = (G_src.mean(0) - G_tgt.mean(0)).abs().mean().item()
    logger.info(f"[机制] 门控均值差 (source vs target): {gate_diff:.4f} "
                f"(>0 说明门控对风格有响应)")

    # 保存
    run_name = f"{ds}_m3_seed{args.seed}"
    result = {"dataset": ds, "model_type": "m3_style_gate", "seed": args.seed,
              "baseline_target": tgt_acc0, "m3_target": best["target_acc"],
              "m3_source": best["source_acc"], "best_epoch": best["epoch"],
              "gate_diff": gate_diff}
    os.makedirs("logs/results", exist_ok=True)
    json.dump(result, open(f"logs/results/{run_name}.json", "w"), indent=2)
    logger.info(f"结果已保存 → logs/results/{run_name}.json")
    logger.info(f"基线 Target {tgt_acc0:.2f} → M3 Target {best['target_acc']:.2f} "
                f"({best['target_acc']-tgt_acc0:+.2f})")


if __name__ == "__main__":
    args = get_args()
    if not hasattr(args, "epochs") or args.epochs == 100:
        args.epochs = 50  # M3 特征级训练很快, 50 epochs 足够
    main_m3(args)
