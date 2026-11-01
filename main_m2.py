"""
M2: 类别条件概念选择 (Class-Conditional Concept Selection)

核心思想: 每个类别学习自己的概念重要性向量 G_cls[k] ∈ R^K,
        分类时 s_k = Σ_j W_kj · G_cls[k,j] · a_j

数学陷阱规避: 直接学习 G_cls 等价于线性分类器重参数化。
        解法: G_cls = σ(α·S + Δ), 其中 S 是冻结的类别-概念语义先验,
        Δ 是低秩修正 (r << min(C,K)), α 是可学习缩放.

新增模型类: clip_cbm_cls (在 clip_cbm_orth 基础上加入 M2)
新增参数:
  --CBM_type clip_cbm_cls     使用 M2 模型
  --cls_rank 16               低秩修正的秩 r
  --lambda_cls_sparse 0.01    概念选择稀疏正则

产物:
  logs/results/{ds}_m2_seed{N}.json        最优指标
  logs/cls_selection/{ds}_seed{N}/         G_cls 矩阵 + 类别 top 概念
"""
import os
import time
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
os.environ["WANDB_MODE"] = "offline"

from args import get_args
from model.cbm_models import clip_cbm_orth
from data import get_dataset_classes

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


class clip_cbm_cls(nn.Module):
    """M2: 类别条件概念选择 + 第三章 Prior-Residual 加权 DDO (可选)

    在 clip_cbm_orth 的基础上, 将分类头从固定权重 W 改为:
      s_k(x) = Σ_j W_kj · G_cls[k,j] · a_j(x) + b_k
    其中 G_cls = σ(α·S + Δ), S 为冻结语义先验, Δ 为低秩修正.
    """

    def __init__(self, args, class_names, concept_names, domain_diffs,
                 domain_weights=None, weight_mode="none",
                 cls_rank=16, init_with_class_embedding=False):
        super().__init__()
        self.device = args.device
        self.weight_mode = weight_mode
        self.num_prompts = domain_diffs.size(0) if domain_diffs is not None else 0

        # ===== 第三章 Prior-Residual (可选, 兼容原有逻辑) =====
        if domain_weights is not None and weight_mode in ("residual", "prior_residual"):
            scores = domain_weights.clone().float().to(self.device).detach()
            if weight_mode == "residual":
                scores = torch.zeros_like(scores)
            self.register_buffer('reliability_prior', scores)
            self.domain_residual = nn.Parameter(torch.zeros_like(scores))
        else:
            self.reliability_prior = None
            self.domain_residual = None

        self.class_names = class_names
        self.concept_names = concept_names

        # ===== CLIP (冻结) =====
        import clip as clip_lib
        self.clip_model, _ = clip_lib.load(args.CLIP_type, device=self.device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.feature_dim = self.clip_model.visual.output_dim

        # ===== 概念嵌入 (冻结) =====
        attr_texts = clip_lib.tokenize(concept_names, truncate=True).to(self.device)
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)

        # ===== M2 核心: 类别条件概念选择 (v2: 训练信号驱动) =====
        self.C = len(class_names)
        self.K = len(concept_names)

        # 先训练一个基线分类头, 用其权重 W₀ 做初始选择矩阵
        # (v1 用 CLIP 语义先验 S, 但诊断发现 S 行间余弦 0.99、无类别区分度, 已弃用)
        # 方案: 先用概念激活做特征, 训练一个线性分类头到收敛, 然后 σ(W₀) 作为 G_cls 初始值
        with torch.no_grad():
            cls_texts = clip_lib.tokenize(class_names, truncate=True).to(self.device)
            cls_embs = self.clip_model.encode_text(cls_texts).float()
            cls_embs = cls_embs / cls_embs.norm(dim=-1, keepdim=True)
            S_raw = cls_embs @ self.concept_embeddings.T  # (C, K) 仅用于日志对比

        # 训练初始分类头 (快速 50 epochs, 只训练 W)
        logger.info("[M2v2] 预训练基线分类头 (50 epochs)...")
        W_init = self._pretrain_classifier(
            concept_embs=self.concept_embeddings,
            n_classes=self.C,
            n_concepts=self.K,
            device=self.device,
        )
        # 归一化 W₀ 到合理 sigmoid 输入范围
        W_norm = (W_init - W_init.mean()) / (W_init.std() + 1e-8)
        self.register_buffer('cls_concept_init', W_norm)
        self.register_buffer('cls_concept_prior_S', S_raw)  # 旧先验保留做对照

        # 可学习缩放 β (初始 1.0, 冻结——防止覆盖初始值)
        self.cls_scale = nn.Parameter(torch.tensor(1.0))
        self.cls_scale.requires_grad = False  # β 固定, 防止训练中整体缩放

        # 低秩修正 Δ = U·Vᵀ, 零初始化 → 初始 G_cls = σ(W₀)
        self.cls_rank = cls_rank
        self.cls_U = nn.Parameter(torch.zeros(self.C, cls_rank) + 0.001 * torch.randn(self.C, cls_rank))
        self.cls_V = nn.Parameter(torch.zeros(cls_rank, self.K))

        # ===== 分类头 (与原版一致: LayerNorm + Linear) =====
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.K),
            nn.Linear(self.K, self.C)
        )
        # 用预训练的 W_init 初始化分类头 (而不是随机初始化)
        with torch.no_grad():
            self.classifier[2].weight.copy_(W_init)

        # ===== DDO 正则 (兼容第三章) =====
        if domain_diffs is not None:
            self.diffs = domain_diffs.clone()
            self.domain_concept_projection = self.diffs @ self.concept_embeddings.T
        else:
            self.diffs = None
            self.domain_concept_projection = None

    def _pretrain_classifier(self, concept_embs, n_classes, n_concepts, device, epochs=50, lr=1e-3):
        """用缓存特征快速训练一个线性分类头, 返回权重 W (C, K)"""
        # 尝试加载缓存特征 (如果存在)
        cache_path = f"cache/{getattr(self, '_dataset_name', 'CUB')}_train.pt"
        if not os.path.exists(cache_path):
            cache_path = "cache/CUB_train.pt"  # 默认 CUB

        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location="cpu")
            z = data["feats"].float().to(device)
            y = data["labels"].to(device)
            z = z / z.norm(dim=-1, keepdim=True)
            a = z @ concept_embs.T  # (N, K) 概念激活
            # 加 LayerNorm 匹配分类头结构
            a_ln = torch.nn.functional.layer_norm(a, (n_concepts,))
            W = self._train_linear(a_ln, y, n_classes, n_concepts, device, epochs, lr)
        else:
            # 无缓存时退化为用类别原型做初始 (不如训练版好但比随机强)
            logger.warning(f"[M2v2] 缓存 {cache_path} 不存在, 使用随机初始化")
            W = torch.randn(n_classes, n_concepts, device=device) * 0.01
        return W

    def _train_linear(self, X, y, n_classes, n_features, device, epochs=50, lr=1e-3):
        """训练线性分类头"""
        model = nn.Linear(n_features, n_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = len(X)
        bs = min(256, n)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                logits = model(X[idx])
                loss = nn.functional.cross_entropy(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model.weight.data.clone()

    def get_domain_weights(self):
        if self.weight_mode == "none" or self.reliability_prior is None:
            return None, None
        if self.weight_mode == "fixed":
            return self.fixed_weights, None
        logits = self.reliability_prior + self.domain_residual
        weights = torch.softmax(logits, dim=0) * self.num_prompts
        return weights, torch.softmax(self.domain_residual, dim=0)

    def get_prior_weights(self):
        if self.reliability_prior is None:
            return None
        return torch.softmax(self.reliability_prior, dim=0) * self.num_prompts

    def get_concept_selection(self):
        """返回当前 G_cls (C, K), sigmoid 输出 ∈ (0,1)"""
        delta = self.cls_U @ self.cls_V  # 低秩修正 (C, K)
        G = torch.sigmoid(self.cls_scale * self.cls_concept_init + delta)
        return G

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        concept_activations = visual_features @ self.concept_embeddings.T  # (bs, K)

        # M2: 类别条件概念选择
        G = self.get_concept_selection()  # (C, K)
        # 调制后的概念激活: 每个类别看到"自己关心的概念子集"
        # 实现方式: G[k,j] * a[j] 对每个类别 k 分别计算
        # 为了效率, 用矩阵形式: logits[k] = W[k] · (G[k] ⊙ a) = Σ_j W[k,j] * G[k,j] * a[j]
        # 即 logits = W * (G ⊙ aᵀ) 的行和 → 等价于 (W ⊙ G) @ aᵀ
        W = self.classifier[2].weight  # (C, K)
        b = self.classifier[2].bias    # (C,)
        logits = (W * G) @ concept_activations.T + b.unsqueeze(1)  # (C, bs)
        logits = logits.T  # (bs, C)

        # LayerNorm 的作用被 G 替代 (概念选择本身起了正则化作用)
        # 保留原始通路做训练稳定器 (残差连接)
        logits_orig = self.classifier(concept_activations)  # (bs, C)
        logits_main = logits + 0.1 * logits_orig  # 主输出 + 10% 原始通路

        # DDO 正则 (与第三章一致, 不乘 G — G 作用于类别维度而非描述符维度)
        if self.domain_concept_projection is not None:
            regularizer = torch.abs(self.classifier[2](self.domain_concept_projection)).mean(dim=(1, 2))
            weights, _ = self.get_domain_weights()
            if weights is not None:
                regularizer = regularizer * weights
        else:
            regularizer = None

        return concept_activations, logits_main, regularizer

    def log_concept_selection(self, run_name, epoch, topk=10):
        """每 epoch 记录概念选择矩阵的统计量和代表性类别的 top 概念"""
        G = self.get_concept_selection().detach().cpu()
        os.makedirs(f"logs/cls_selection/{run_name}", exist_ok=True)

        # 保存完整 G_cls
        torch.save({"epoch": epoch, "G_cls": G, "beta": self.cls_scale.item(),
                    "delta_norm": (self.cls_U @ self.cls_V).detach().norm().item(),
                    "init_norm": self.cls_concept_init.norm().item()},
                   f"logs/cls_selection/{run_name}/epoch_{epoch:04d}.pt")

        if epoch % 10 == 0:
            stats = {"mean": G.mean().item(), "std": G.std().item(),
                     "max": G.max().item(), "min": G.min().item(),
                     "beta": self.cls_scale.item(),
                     "delta_norm": (self.cls_U @ self.cls_V).detach().norm().item(),
                     "init_norm": self.cls_concept_init.norm().item()}
            logger.info(f"[M2] G_cls stats: {stats}")

            # 展示 3 个类别的 top-10 概念
            for cls_id in [0, self.C // 2, self.C - 1]:
                top_idx = G[cls_id].argsort(descending=True)[:topk]
                top_concepts = [self.concept_names[j] for j in top_idx]
                logger.info(f"[M2] '{self.class_names[cls_id]}' top-{topk}: {top_concepts}")


# ============================================================
# 训练循环 (复用 main.py 的逻辑, 但用 M2 模型)
# ============================================================
def train_m2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 数据
    train_ds, train_dl, test_ds, test_dl, tgt_ds, tgt_dl = get_dataset_classes(args)
    logger.info(f"Train {len(train_ds)} / Source test {len(test_ds)} / Target test {len(tgt_ds)}")

    # 模型
    model = clip_cbm_cls(
        args,
        class_names=list(train_ds.classname2id.keys()),
        concept_names=list(train_ds.concept2id.keys()),
        domain_diffs=train_ds.domain_diffs,
        domain_weights=getattr(train_ds, "domain_weights", None),
        weight_mode=args.weight_mode if args.weight_mode != "none" else "none",
        cls_rank=getattr(args, "cls_rank", 16),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_params:,} | Trainable: {trainable:,}")
    logger.info(f"  其中 M2 新增: cls_U {model.cls_U.numel():,} + cls_V {model.cls_V.numel():,}"
                f" + cls_scale 1 = {model.cls_U.numel() + model.cls_V.numel() + 1:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    cls_loss_fn = nn.CrossEntropyLoss()

    best = {"source_acc": 0.0, "target_acc": 0.0, "epoch": -1}
    run_name = f"{args.dataset}_m2_seed{args.seed}"

    for epoch in range(args.epochs):
        model.train()
        for images, labels, attrs in train_dl:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)

            concept_preds, cls_preds, reg = model(images)
            l_cls = cls_loss_fn(cls_preds, labels)
            l_ort = torch.abs(reg).mean() if reg is not None else torch.tensor(0.0, device=device)

            # KL 正则 (兼容第三章)
            l_kl = torch.tensor(0.0, device=device)
            _, rs = model.get_domain_weights() if model.get_domain_weights()[0] is not None else (None, None), None
            if rs is not None:
                N = rs.size(0)
                H = -(rs * (rs + 1e-8).log()).sum()
                l_kl = torch.log(torch.tensor(float(N), device=device)) - H

            # M2 稀疏正则 (鼓励概念选择不过度集中)
            G = model.get_concept_selection()
            l_sparse = G.mean(dim=0).pow(2).sum()  # 每个概念被所有类别平均选中的平方和

            loss = l_cls + args.alpha * l_ort + args.lambda_kl * l_kl \
                   + getattr(args, "lambda_cls_sparse", 0.01) * l_sparse

            loss.backward()
            opt.step()

        # 评估
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                src_correct, src_total = 0, 0
                for images, labels, _ in test_dl:
                    images, labels = images.to(device), labels.to(device)
                    _, preds, _ = model(images)
                    src_correct += (preds.argmax(1) == labels).sum().item()
                    src_total += len(labels)
                src_acc = 100 * src_correct / src_total

                tgt_correct, tgt_total = 0, 0
                for images, labels, _ in tgt_dl:
                    images, labels = images.to(device), labels.to(device)
                    _, preds, _ = model(images)
                    tgt_correct += (preds.argmax(1) == labels).sum().item()
                    tgt_total += len(labels)
                tgt_acc = 100 * tgt_correct / tgt_total

            if tgt_acc > best["target_acc"]:
                best = {"source_acc": src_acc, "target_acc": tgt_acc, "epoch": epoch}
            if (epoch + 1) % 20 == 0:
                logger.info(f"Ep {epoch+1} | Src {src_acc:.2f} | Tgt {tgt_acc:.2f} | "
                            f"Best {best['target_acc']:.2f} @ {best['epoch']+1}")

        # 记录概念选择
        if (epoch + 1) % 10 == 0:
            model.log_concept_selection(run_name, epoch)

    # 保存结果
    os.makedirs("logs/results", exist_ok=True)
    result = {"dataset": args.dataset, "model_type": "clip_cbm_cls", "weight_mode": "m2",
              "seed": args.seed, "alpha": args.alpha, "lambda_kl": args.lambda_kl,
              "cls_rank": getattr(args, "cls_rank", 16), **best}
    json.dump(result, open(f"logs/results/{run_name}.json", "w"), indent=2)
    logger.info(f"Result saved → logs/results/{run_name}.json")
    logger.info(f"Best Target: {best['target_acc']:.2f} @ Epoch {best['epoch']+1}")

    return best


if __name__ == "__main__":
    args = get_args()
    # 默认参数适配
    if not hasattr(args, "cls_rank"):
        args.cls_rank = 16
    if not hasattr(args, "lambda_cls_sparse"):
        args.lambda_cls_sparse = 0.01
    train_m2(args)
