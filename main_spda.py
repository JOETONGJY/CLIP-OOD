"""P2: SPDA — 语义保护的域子空间对齐 (Semantic-Preserving Domain Alignment), 缓存特征管线

针对 full_align 的 PACS 失败机制（域方向与类别判别方向纠缠）:
  抑制域方向前, 先剔除其中与类别判别子空间重叠的分量。

方法链 (在 full_align 基础上新增一步):
  ① B_text = SVD(域方向)                          [已有]
  ② B_adapt = orth(B_text + ΔB)                   [已有, 视觉校准]
  ③ Q = SVD(源域类别原型中心化矩阵) 前 k 维          【新增: 类别判别子空间】
  ④ B_safe = orth((I − QQᵀ) B_adapt)               【新增: 剔除类别重叠】
  ⑤ z_inv = z − γ·B_safe B_safeᵀ z                 [软抑制]
对齐正则 (anchor/visual) 仍锚定 B_adapt —— 视觉校准目标不变。

实现: 子类化 clip_cbm_subspace 并在 CachedTrainingSession 内原地升级
(__class__ 替换), 训练循环/评估/断点/结果落盘全部复用 main.py 原管线;
特征缓存 (cache/*.pt) 使 100-epoch run 从数小时降到数分钟。

结果文件名: {DS}_spda_seed{n}.json (ch4_mode 在构建后改写为 spda 仅用于命名,
模型内部保持 full_align 以复用 get_alignment_losses / 双头结构)。

用法: python main_spda.py --dataset PACS --CBM_type clip_cbm_subspace \
        --ch4_mode full_align --tau 0.8 --protect_k 0 --seed 1 \
        --CLIP_type "F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
"""
import os
import torch

from main import logger
from main_cached import CachedTrainingSession
from model.cbm_models import clip_cbm_subspace


class clip_cbm_spda(clip_cbm_subspace):
    """full_align + 类别语义保护: 抑制基由 B_adapt 换为 B_safe = orth((I−QQᵀ)B_adapt)"""

    def spda_setup(self, class_feats, class_labels, tau_c=0.9, protect_k=0):
        """由源域缓存特征计算类别判别子空间 Q 并注册为 buffer (不进优化器)"""
        with torch.no_grad():
            C = int(class_labels.max().item()) + 1
            protos = torch.stack([class_feats[class_labels == c].mean(0) for c in range(C)])
            protos = protos / protos.norm(dim=-1, keepdim=True)
            M = protos - protos.mean(0, keepdim=True)          # (C, 768) 中心化原型
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            ev = S ** 2
            cum = torch.cumsum(ev, 0) / ev.sum()
            if protect_k and protect_k > 0:
                k = min(int(protect_k), Vh.shape[0])
            else:
                k = int(torch.searchsorted(cum, torch.tensor(tau_c)).item()) + 1
            self.register_buffer("cls_basis",
                                 Vh[:k].T.contiguous().to(self.device))   # (768, k)
            self.protect_dim = k
        logger.info(f"[SPDA] Q: {k} 维 (τ_c={tau_c}, C={C})")

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()
        z = visual_features / visual_features.norm(dim=-1, keepdim=True)

        B_adapt = torch.linalg.qr(self.domain_basis + self.delta_B)[0]   # (768, m)
        # SPDA: 抑制前剔除类别判别子空间重叠 (等价于 (I−QQᵀ)B_adapt, 免 768² 投影)
        B_filt = B_adapt - self.cls_basis @ (self.cls_basis.T @ B_adapt)
        B_safe = torch.linalg.qr(B_filt)[0]

        gamma = self.get_gamma()
        z_inv = z - gamma * (z @ B_safe) @ B_safe.T

        # 对齐正则仍锚定 B_adapt (视觉校准目标不变, 与 full_align 一致)
        P_a = B_adapt @ B_adapt.T
        with torch.no_grad():
            P_t = self.domain_basis @ self.domain_basis.T
            P_v = self.aug_basis @ self.aug_basis.T
            self._align_anchor_ref = P_t
            self._align_visual_ref = P_v
        self._anchor_loss = torch.norm(P_a - self._align_anchor_ref, p="fro") ** 2
        self._visual_loss = torch.norm(P_a - self._align_visual_ref, p="fro") ** 2

        # concept bottleneck + 双头 (full_align 必有 inv_classifier; P4 校准走 E')
        E_eff = self._effective_E()
        concept_activations = z @ E_eff.T
        concept_activations_inv = z_inv @ E_eff.T
        logits_ori = self.classifier(concept_activations)
        logits_inv = self.inv_classifier(concept_activations_inv)

        ddo_proj = self._domain_projection(E_eff)
        regularizer = torch.abs(self.classifier[1:](ddo_proj))
        reg_inv = torch.abs(self.inv_classifier[1:](ddo_proj))
        regularizer = regularizer + self.eta_ddo_inv * reg_inv
        weights, _ = self.get_domain_weights()
        if weights is not None:
            regularizer = regularizer * weights.view(-1, 1, 1)

        return concept_activations_inv, (logits_ori, logits_inv), regularizer


class SPDACachedSession(CachedTrainingSession):
    def _init_model(self):
        super()._init_model()      # 构建 clip_cbm_subspace(full_align) + 恒等编码器
        # 源域缓存特征 → 类别原型
        d = torch.load(os.path.join("cache", f"{self.args.dataset}_train.pt"),
                       map_location="cpu")
        feats = d['feats'].float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        labels = d['labels'].long()
        # 原地升级为 SPDA 子类: 保留全部参数/buffer/方法, 优化器不受影响 (cls_basis 是 buffer)
        self.model.__class__ = clip_cbm_spda
        self.model.spda_setup(feats, labels, tau_c=0.9,
                              protect_k=getattr(self.args, 'protect_k', 0))
        # 命名: 结果落 {DS}_spda_seed{n}.json, 避免覆盖 full_align 基线
        self.args.ch4_mode = 'spda'


if __name__ == "__main__":
    import random
    import numpy as np
    from args import get_args

    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    session = SPDACachedSession(args)
    session.run()
