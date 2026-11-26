import os
import time
import json
import hashlib
import logging
import random
import shutil
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
os.environ["WANDB_MODE"] = "offline"

# Local imports
from args import get_args
from model.cbm_models import clip_cbm_orth, clip_mlp, clip_cbm_subspace
from data import get_dataset_classes
from run_status import RunLock, git_info, jsonable
from ld_dro import effective_response, ld_dro_loss, pgd_worst_loss          # LD 臂的梯度诊断用
from ld_training import MODES as LD_MODES, build_ld_generator, pen_stats as ld_pen_stats, training_loss as ld_training_loss
from language_ellipsoid import domain_directions, second_moments

# --grad_diag 时测量 CE/DDO 梯度统计的 epoch (1 起算, 在该 epoch 训练之前测), 另加训练结束后一次
GRAD_DIAG_EPOCHS = (1, 5, 10, 20, 30, 50, 75)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingSession:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.args.device = self.device
        self._init_components()

        self.best_metrics = {
            'source_acc': 0.0,
            'target_acc': 0.0,
            'epoch': -1
        }
        # F1 严格 DG 选模: 训练结束时模型的指标 (目标域不参与选模)
        self.selected_metrics = None

    @staticmethod
    def _selection_epoch(epoch, total_epochs):
        """model_selection=final 时, 仅在最后一个 epoch 触发选模保存"""
        return epoch == total_epochs - 1

    def _setup_device(self):
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()  else "cpu"
        )
        logger.info(f"Using device: {device}")
        return device

    def _init_components(self):
        self._prepare_datasets()
        self._init_model()
        self._init_optimizer()
        self._init_loss_fns()
        self._init_ld_dro()

    def _init_ld_dro(self):
        """LD 臂（外审第十轮批准；预注册 RESULTS §3z）：训练损失 = LD-DRO / ISDA / PGD 的鲁棒交叉熵，
        **不叠加**原始 DDO 正则（强度完全由 eps / lam 决定）。默认 ld_mode='none' 时直接返回，全部行为不变。
        配置守卫：LD 臂要求 alpha=0 / beta=0 / weight_mode=none / ddo_dirs=full / lambda_kl=0 / lambda_graph=0。
        评估始终用干净 logits（鲁棒项只进训练目标）；椭球生成元由训练 seed 按候选规则构造并记录 sha256。"""
        self.ld_mode = getattr(self.args, "ld_mode", "none") or "none"
        if self.ld_mode == "none":
            return
        if self.ld_mode not in LD_MODES:
            raise ValueError(f"未知 ld_mode='{self.ld_mode}' (可选: {LD_MODES})")
        a = self.args
        if getattr(a, "CBM_type", "clip_cbm") != "clip_cbm":
            raise ValueError("ld_mode 只在第三章入口 (clip_cbm) 下定义")
        if getattr(a, "weight_mode", "none") != "none" or getattr(a, "ddo_dirs", "full") != "full" \
                or float(a.beta) != 0.0 or float(a.alpha) != 0.0 \
                or float(a.lambda_kl) != 0.0 or float(a.lambda_graph) != 0.0:
            raise ValueError("ld_mode != none 要求 alpha=0 / beta=0 / weight_mode=none / ddo_dirs=full / "
                             f"lambda_kl=0 / lambda_graph=0 (got alpha={a.alpha}, weight_mode={a.weight_mode}, "
                             f"ddo_dirs={getattr(a, 'ddo_dirs', 'full')}); LD 臂不叠加原始 DDO 正则")
        if self.ld_mode in ("ld", "pgd") and not float(a.ld_eps) > 0:
            raise ValueError(f"ld_mode={self.ld_mode} 需要 --ld_eps > 0")
        if self.ld_mode == "isda" and not float(a.ld_lam) > 0:
            raise ValueError("ld_mode=isda 需要 --ld_lam > 0")
        # 椭球: 与 Stage 1C 半径诊断同一构造 (G = E[uuᵀ] 非中心化二阶矩, u = LN0(E·d)); 随机椭球的种子候选
        # 规则一致 (s/+10/+20/+30)。生成元 L 是 buffer (不参与梯度); M = A·L 里变化的只有 A (分类头权重)。
        E = self.model.concept_embeddings.detach().double().cpu()
        U = domain_directions(self.train_dataset.domain_diffs.double().cpu(), E)
        G = second_moments(U)[0]
        L, gmeta = build_ld_generator(G, a.ld_ell, int(a.seed))
        self.ld_L = L.to(self.device)
        calib = None
        if getattr(a, "ld_calib", ""):
            with open(a.ld_calib, "rb") as f:
                calib = {"path": a.ld_calib.replace(os.sep, "/"), "sha256": hashlib.sha256(f.read()).hexdigest()}
        self.ld_meta = {"mode": self.ld_mode, "eps": float(a.ld_eps), "lam": float(a.ld_lam),
                        "pgd_steps": int(getattr(a, "pgd_steps", 5)), "pgd_restarts": int(getattr(a, "pgd_restarts", 1)),
                        "pgd_seed": int(getattr(a, "pgd_seed", 0)), "K": int(E.size(0)),
                        "E_sha256": hashlib.sha256(E.numpy().tobytes()).hexdigest(),
                        "G_sha256": hashlib.sha256(G.numpy().tobytes()).hexdigest(),
                        "generator": gmeta, "calibration": calib,
                        "protocol": "训练损失 = 鲁棒 CE (不叠加原始 DDO 正则, alpha=0); 评估用干净 logits"}
        logger.info(f"[LD] mode={self.ld_mode} ell={a.ld_ell} (r={gmeta['r']}, seed_used={gmeta['ell_seed_used']}, "
                    f"sha {gmeta['sha256'][:16]}) eps={a.ld_eps:g} lam={a.ld_lam:g}")

    def _ld_M(self):
        """LD 臂的 M = A·L。A = W⊙γ 是分类头的有效权重（随训练变化，每个 batch 重算）；
        L 是固定的椭球生成元（见 _init_ld_dro）。h(u + Lq) = h(u) + M q。"""
        A, _ = effective_response(self.model.classifier)
        return A @ self.ld_L

    def _prepare_datasets(self):
        """CUB, AWA2, LADA, LADV"""
        self.train_dataset, self.train_loader, self.source_test_dataset, self.source_test_loader, self.target_test_dataset, self.target_test_loader = get_dataset_classes(self.args)

        logger.info(f"Dataset {self.args.dataset} loaded")
        logger.info(f"Train samples: {len(self.train_dataset):,}")
        logger.info(f"Source test samples: {len(self.source_test_dataset):,}")
        logger.info(f"Target test samples: {len(self.target_test_dataset):,}")

        self._apply_redundancy_args()

    def _apply_redundancy_args(self):
        """C-1 冗余参数接线 (2026-11-22 外审修复): 旧版 --redundancy_topk 从未传到
        数据集 (attach 时的 getattr(dataset,'redundancy_topk',0) 恒为 0)。此处按 args
        重算挂载, 使 topk 真正生效; main_cached 复用同一方法。"""
        if getattr(self.args, "redundancy", False):
            topk = int(getattr(self.args, "redundancy_topk", 0))
            if topk > 0 and getattr(self.train_dataset, "domain_diffs", None) is not None:
                from reliability import compute_descriptor_redundancy
                self.train_dataset.redundancy_topk = topk
                self.train_dataset.descriptor_redundancy = compute_descriptor_redundancy(
                    self.train_dataset.domain_diffs, topk=topk)
                logger.info(f"[C-1] redundancy_topk={topk} 已生效 (数据集侧冗余度重算)")

    def _init_model(self):
        """clip_cbm, cliplp, clip_cbm_subspace"""
        model_factory = {
            'clip_cbm': clip_cbm_orth,
            'cliplp': clip_mlp,
            'clip_cbm_subspace': clip_cbm_subspace
        }

        try:
            model_class = model_factory[self.args.CBM_type]
        except KeyError:
            raise ValueError(f"Unsupported model type: {self.args.CBM_type}")

        if self.args.CBM_type == 'clip_cbm_subspace':
            # 第四章模型: SVD basis -> soft projection -> invariant head
            # --use_c3_weights 时叠加第三章 importance 加权 (2×2 联合实验 C3+C4)
            self.model = model_class(
                args=self.args,
                class_names=list(self.train_dataset.classname2id.keys()),
                concept_names=list(self.train_dataset.concept2id.keys()),
                domain_diffs=self.train_dataset.domain_diffs,
                tau=getattr(self.args, 'tau', 0.8),
                ch4_mode=getattr(self.args, 'ch4_mode', 'full'),
                eta_ddo_inv=getattr(self.args, 'eta_ddo_inv', 1.0),
                domain_weights=(getattr(self.train_dataset, "domain_weights", None)
                                if getattr(self.args, 'use_c3_weights', False) else None),
                weight_mode=(getattr(self.args, 'weight_mode', 'prior_residual')
                             if getattr(self.args, 'use_c3_weights', False) else "none"),
                cal_rank=getattr(self.args, 'cal_rank', 0),
                descriptor_redundancy=getattr(self.train_dataset, "descriptor_redundancy", None),
                use_redundancy=(getattr(self.args, 'redundancy', False)
                                and getattr(self.args, 'use_c3_weights', False)),
            ).to(self.device)
        else:
            # 改动时间26.4.29-----------------------------------
            # E1 干预 (2026-11-21): 可选加载 δ 冻结掩码 (可信-困难象限), 默认空=关闭
            freeze_file = getattr(self.args, "freeze_delta_file", "")
            freeze_mask = None
            if freeze_file:
                _fm = torch.load(freeze_file, map_location="cpu", weights_only=False)
                if isinstance(_fm, dict):
                    _fm = _fm["indices"]
                freeze_mask = _fm.long().clone()
                logger.info(f"[E1] freeze_delta_file={freeze_file} "
                            f"-> 冻结 {freeze_mask.numel()} 个描述符的 δ")
            # CA-Fixed (2026-11-24): 类别感知固定权重先验文件
            ca_prior = None
            ca_file = getattr(self.args, "ca_prior_file", "")
            if ca_file:
                _ca = torch.load(ca_file, map_location="cpu", weights_only=False)
                if isinstance(_ca, dict):
                    _ca = _ca["interaction"]
                ca_prior = _ca.float().clone()
                logger.info(f"[CA] ca_prior_file={ca_file} -> 交互矩阵 {tuple(ca_prior.shape)}, "
                            f"β={getattr(self.args, 'ca_beta', 1.0)}")
            self.model = model_class(
                args=self.args,
                class_names=list(self.train_dataset.classname2id.keys()),
                concept_names=list(self.train_dataset.concept2id.keys()),
                domain_diffs=self.train_dataset.domain_diffs,
                domain_weights=getattr(self.train_dataset, "domain_weights", None),
                weight_mode=getattr(self.args, "weight_mode", "prior_residual"),
                descriptor_redundancy=getattr(self.train_dataset, "descriptor_redundancy", None),
                use_redundancy=getattr(self.args, "redundancy", False),
                freeze_delta_mask=freeze_mask,
                grouped_weights=getattr(self.args, "grouped_weights", False),
                descriptor_clusters=(
                    __import__('reliability').compute_descriptor_clusters(
                        self.train_dataset.domain_diffs,
                        threshold=float(getattr(self.args, 'cluster_threshold', 0.9)))
                    if getattr(self.args, 'grouped_weights', False) else None),
                ca_prior=ca_prior,
                ca_beta=getattr(self.args, "ca_beta", 1.0),
                # DDO-Subspace (2026-11-25): 训练时约束的方向集 (默认 full = 原始行为)
                ddo_dirs=getattr(self.args, "ddo_dirs", "full"),
                ddo_k=getattr(self.args, "ddo_k", 32),
                ddo_pca_src=getattr(self.args, "ddo_pca_src", "ln0"),
                ddo_dirs_seed=getattr(self.args, "ddo_dirs_seed", 0),
            ).to(self.device)
        # print("Model domain_weights:", getattr(self.model, "domain_weights", None), flush=True)
        # if hasattr(self.model, "domain_weights") and self.model.domain_weights is not None:
        #     print("domain_weights shape:", self.model.domain_weights.shape, flush=True)
        #     print("domain_weights sample:", self.model.domain_weights[:5], flush=True)
        # 改动时间26.4.29-----------------------------------
        # self.model = model_class(
        #     args=self.args,
        #     class_names=list(self.train_dataset.classname2id.keys()),
        #     concept_names=list(self.train_dataset.concept2id.keys()),
        #     domain_diffs=self.train_dataset.domain_diffs
        # ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {self.args.CBM_type}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # A1 语义锚定: cal_rank>0 且 lambda_rel>0 时挂载 (无新参数, 纯损失约束)
        if getattr(self.args, 'cal_rank', 0) > 0 and getattr(self.args, 'lambda_rel', 0.0) > 0:
            from model.semantic_anchor import SemanticAnchor
            self.sem_anchor = SemanticAnchor(self.model.concept_embeddings,
                                             topk=getattr(self.args, 'anchor_topk', 5)
                                             ).to(self.model.device)
            logger.info(f"[A1] SemanticAnchor 挂载: topk={getattr(self.args, 'anchor_topk', 5)}, "
                        f"lambda_rel={self.args.lambda_rel}")

    def _init_optimizer(self):
        """init optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        logger.info(f"Optimizer initialized with lr={self.args.lr:.1e}")

    def _init_loss_fns(self):
        """init loss"""
        self.loss_fns = {
            'cls': nn.CrossEntropyLoss(),
            'concept': nn.BCEWithLogitsLoss()
        }
        logger.info("Loss functions initialized")

    def _save_concept_logs(self):
        """save concept logs"""
        os.makedirs("logs", exist_ok=True)
        base_path = f"logs/{self.args.dataset}_{self.args.CBM_type}"

        last_concepts, best_concepts = self.model.extract_cls_concept()

        with open(f"{base_path}_last.json", 'w') as f:
            json.dump(last_concepts, f, indent=2)

        with open(f"{base_path}_best.json", 'w') as f:
            json.dump(best_concepts, f, indent=2)

        logger.info(f"Concept analysis saved to {base_path}_*.json")

    def _log_domain_weights(self, epoch):
        """
        每个 epoch 保存/记录 domain 权重诊断量 (专家要求的机制证据链):
          - reliability prior r_i, residual δ_i, prior weight w_i^prior, final weight w_i^final
          - weight entropy H(q), max/min/std of final weights
        用途: (1) Δw 分析找 "高 reliability 被下调 / 低 reliability 被上调" 案例
              (2) KL 防退化的实证 (entropy 是否维持)
        """
        weights, residual_softmax = (
            self.model.get_domain_weights()
            if hasattr(self.model, 'get_domain_weights') else (None, None)
        )
        if weights is None:
            # 纯第四章 (无 C3 权重) 或无权重机制: 记录 gamma (机制证据: 学到的抑制强度)
            # 注: clip_cbm_subspace 也定义了 get_domain_weights (纯C4时返回 None,None),
            #     因此不能只靠 hasattr 判断, 否则 gamma 永远不会被记录
            diagnostics = {}
            if hasattr(self.model, 'get_gamma'):
                gamma = self.model.get_gamma()
                g = gamma.item() if torch.is_tensor(gamma) else float(gamma)
                diagnostics['gamma'] = g
                logger.info(f"[Ch4] gamma={g:.4f} (subspace_dim={getattr(self.model, 'subspace_dim', '?')})")
            return diagnostics

        diagnostics = {}
        with torch.no_grad():
            prior_w = self.model.get_prior_weights()
            final_w = weights
            N = final_w.size(0)
            if residual_softmax is not None:
                entropy = -(residual_softmax * (residual_softmax + 1e-8).log()).sum().item()
            else:
                entropy = float('nan')
            diagnostics = {
                'weight_entropy': entropy,
                'weight_max': final_w.max().item(),
                'weight_min': final_w.min().item(),
                'weight_std': final_w.std().item(),
            }

            # 保存完整向量 (torch 格式, 每 epoch 一个文件)
            run_name = getattr(self, 'run_name', f"{self.args.dataset}_{self.args.CBM_type}")
            os.makedirs(f"logs/weights/{run_name}", exist_ok=True)
            save_dict = {
                'epoch': epoch,
                'reliability_prior': self.model.reliability_prior.detach().cpu(),
                'residual': self.model.domain_residual.detach().cpu() if self.model.domain_residual is not None else None,
                'prior_weights': prior_w.detach().cpu() if prior_w is not None else None,
                'final_weights': final_w.detach().cpu(),
                'weight_entropy': entropy,
            }
            # E3 grouped 模式日志 (2026-11-22 外审修复: 簇级残差/结构不落盘导致
            # "分组结构 vs 簇级残差"无法归因) —— 分组结构与两级权重分开记录
            # 2026-11-23 修正: grouped_prior_weights 改用 get_grouped_prior_weights()
            # (Δ_g=0 完整 w^(0), 含簇内份额与校准); 旧实现混入残差且仅展开簇级, 名实不符。
            # 归因分解: Δw_structure = w^(0) − prior_weights; Δw_learn = final_weights − w^(0)
            if getattr(self.model, 'grouped_weights', False):
                gpw = self.model.get_grouped_prior_weights()
                save_dict.update({
                    'grouped': True,
                    'cluster_assignment': self.model.cluster_assignment.detach().cpu(),
                    'cluster_prior': self.model.cluster_prior.detach().cpu(),
                    'cluster_residual': self.model.cluster_residual.detach().cpu(),
                    'grouped_prior_weights': gpw.detach().cpu(),
                })
            torch.save(save_dict, f"logs/weights/{run_name}/epoch_{epoch:04d}.pt")

            # Δw 摘要 (final - prior), 只在日志里打 Top 修正
            # (2026-11-24 修复: CA-Fixed 的 final_w 是 (204, C), 与 (204,) prior 形状不同 → 跳过 Δw)
            if prior_w is not None and final_w.shape == prior_w.shape:
                delta_w = (final_w - prior_w)
                top_up = delta_w.argmax().item()
                top_down = delta_w.argmin().item()
                diagnostics['delta_w_max'] = delta_w[top_up].item()   # 被上调最多的 prompt
                diagnostics['delta_w_min'] = delta_w[top_down].item()  # 被下调最多的 prompt

            logger.info(
                f"[Weights] H(q)={diagnostics['weight_entropy']:.4f} "
                f"max={diagnostics['weight_max']:.4f} min={diagnostics['weight_min']:.4f} "
                f"std={diagnostics['weight_std']:.4f}"
                + (f" Δw∈[{diagnostics.get('delta_w_min', 0):.4f}, {diagnostics.get('delta_w_max', 0):.4f}]"
                   if 'delta_w_max' in diagnostics else "")
            )
        return diagnostics

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        # 2026-11-25: 按样本数加权的 epoch 累计量 —— 原先返回的是最后一个 batch 的损失,
        # 不能用于机制分析 (外审指出)。ddo_own_unw = 本臂训练方向集上的未加权 DDO 损失 (不跨方法可比);
        # 跨方法可比的统一口径是每个 epoch 结束后在【原始完整方向集】上另算的 ddo_full_eval (见 epoch_metrics)。
        sums = {'total': 0.0, 'cls': 0.0, 'concept': 0.0, 'orth': 0.0, 'ddo_own_unw': 0.0, 'kl': 0.0,
                'clean_ce': 0.0, 'pen_mean': 0.0, 'pen_std': 0.0, 'wcpen': 0.0, 'mrob': 0.0}
        ld_on = self.ld_mode != "none"

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.args.epochs}",
            dynamic_ncols=True
        )

        for batch_idx, (images, labels, attr_labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            attr_labels = attr_labels.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)

            concept_preds, cls_preds, reg_loss = self.model(images)

            # 双头支持 (第四章 clip_cbm_subspace): (logits_ori, logits_inv)
            # 训练: L = CE(logits_ori) + λ_inv·CE(logits_inv); 评估/acc 用 logits_inv
            if isinstance(cls_preds, tuple):
                logits_ori, logits_inv = cls_preds
                cls_loss = (
                    self.loss_fns['cls'](logits_ori, labels)
                    + self.args.lambda_inv * self.loss_fns['cls'](logits_inv, labels)
                )
                main_logits = logits_inv
            else:
                main_logits = cls_preds
                if ld_on:
                    # LD 臂 (预注册 §3z): 训练损失 = 鲁棒 CE (LD-DRO / ISDA / PGD), 不叠加原始 DDO 正则 ——
                    # alpha=0 已由 _init_ld_dro 守卫强制, 下面的 alpha*orth 项自然为 0。监控统计用 detach 的副本。
                    M = self._ld_M()
                    cls_loss = ld_training_loss(self.ld_mode, cls_preds, M, labels, self.args.ld_eps,
                                                self.args.ld_lam, self.args.pgd_steps, self.args.pgd_restarts,
                                                self.args.pgd_seed)
                    ps = ld_pen_stats(cls_preds.detach(), M.detach(), labels,
                                      self.args.ld_eps if self.ld_mode != "isda" else self.args.ld_lam,
                                      kind="ld" if self.ld_mode != "isda" else "isda")
                    bs = images.size(0)
                    sums['clean_ce'] += ps['clean_ce'] * bs
                    sums['pen_mean'] += ps['pen_maxpair_mean'] * bs
                    sums['pen_std'] += ps['pen_maxpair_std'] * bs
                    sums['wcpen'] += ps['pen_worstcomp_mean'] * bs
                    sums['mrob'] += ps['robust_margin_mean'] * bs
                else:
                    cls_loss = self.loss_fns['cls'](cls_preds, labels)
            # concept_loss = self.loss_fns['concept'](concept_preds, attr_labels)
            if self.args.beta > 0:
                concept_loss = self.loss_fns['concept'](concept_preds, attr_labels.float())
            else:
                concept_loss = torch.tensor(0.0, device=images.device)

            orth_loss = torch.abs(reg_loss).mean() if reg_loss is not None else 0.0

            # KL 正则: 约束 residual 的 softmax 不偏离 uniform 太远 (防退化, 替代 clamp)
            # D_KL(q || U) = log(N) - H(q), 其中 q = softmax(residual), U = uniform
            # 注意方向: 正向 KL (q 在前), 衡量 q 到均匀分布的距离
            kl_loss = torch.tensor(0.0, device=images.device)
            residual_softmax = None
            if hasattr(self.model, 'get_domain_weights'):
                _, residual_softmax = self.model.get_domain_weights()
            if residual_softmax is not None:
                N = residual_softmax.size(0)
                entropy = -(residual_softmax * (residual_softmax + 1e-8).log()).sum()
                kl_loss = torch.log(torch.tensor(float(N), device=images.device)) - entropy

            # Graph Laplacian: 相关描述符的残差修正应一致 (pr_graph)
            graph_loss = torch.tensor(0.0, device=images.device)
            if getattr(self.model, 'desc_laplacian', None) is not None and self.args.lambda_graph > 0:
                d_ = self.model.domain_residual
                graph_loss = d_ @ self.model.desc_laplacian @ d_

            # 算法 C (full_align): 投影差锚定 + 视觉校准
            align_terms = self.model.get_alignment_losses() if hasattr(self.model, 'get_alignment_losses') else None
            align_loss = torch.tensor(0.0, device=images.device)
            if align_terms is not None:
                anchor_l, visual_l = align_terms
                align_loss = self.args.lambda_anchor * anchor_l + self.args.lambda_visual * visual_l

            # P4 低秩概念校准锚定: lambda_cal * ||ΔE||_F² (概念语义不漂移约束)
            cal_loss = torch.tensor(0.0, device=images.device)
            if getattr(self.model, 'cal_rank', 0) > 0:
                cal_loss = self.args.lambda_cal * self.model._cal_norm ** 2

            # A1 语义锚定: 局部语义关系矩阵约束 (top-k 邻域 Gram 结构保持)
            rel_loss = torch.tensor(0.0, device=images.device)
            if getattr(self, 'sem_anchor', None) is not None:
                rel_loss = self.sem_anchor(self.model._effective_E())

            total_loss = (
                    cls_loss
                    + self.args.alpha * orth_loss
                    + self.args.beta * concept_loss
                    + self.args.lambda_kl * kl_loss
                    + self.args.lambda_graph * graph_loss
                    + align_loss
                    + cal_loss
                    + getattr(self.args, 'lambda_rel', 0.0) * rel_loss
            )

            # 双头一致性: KL(p_inv || p_orig), full_mc 默认 0.1, 亦可 --cons_weight 指定
            cons_w = self.args.cons_weight
            if cons_w > 0 and isinstance(cls_preds, tuple):
                logits_ori, logits_inv = cls_preds
                p_o = torch.softmax(logits_ori, dim=-1)
                log_p_i = torch.log_softmax(logits_inv, dim=-1)
                cons = (p_o * (p_o.log() - log_p_i)).sum(-1).mean()
                total_loss = total_loss + cons_w * cons

            total_loss.backward()
            self.optimizer.step()
            # E1 干预 (2026-11-21): 冻结象限的 δ 每个 step 后回零 (无掩码时为 no-op)
            if hasattr(self.model, 'apply_delta_freeze'):
                self.model.apply_delta_freeze()

            batch_size = images.size(0)
            total += batch_size
            correct += (main_logits.argmax(dim=1) == labels).sum().item()
            _reg_u = getattr(self.model, '_reg_unweighted', None)   # 加权之前、在【本臂训练方向集】上的 DDO 响应
            sums['total'] += float(total_loss) * batch_size
            sums['cls'] += float(cls_loss) * batch_size
            sums['concept'] += float(concept_loss) * batch_size
            sums['orth'] += float(orth_loss) * batch_size
            # 自身训练方向集上的未加权损失 (不跨方法可比: 方向集随实验臂变化)
            sums['ddo_own_unw'] += (float(_reg_u.abs().mean()) if _reg_u is not None else float(orth_loss)) * batch_size
            sums['kl'] += float(kl_loss) * batch_size

            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%",
                'CLS': f"{cls_loss.item():.4f}",
                'CON': f"{concept_loss.item():.4f}",
                'ORT': f"{orth_loss.item():.4f}",
                'KL': f"{kl_loss.item():.4f}"
            })

        out = {
            'train_acc': 100 * correct / total,
            'train_loss': sums['total'] / total,          # 以下均为 epoch 内按样本加权平均
            'cls_loss': sums['cls'] / total,
            'concept_loss': sums['concept'] / total,
            'orth_loss': sums['orth'] / total,
            'ddo_own_unweighted_loss': sums['ddo_own_unw'] / total,
            'kl_loss': sums['kl'] / total,
        }
        if ld_on:
            # LD 臂 (预注册 §3z) 的逐 epoch 监控量: 干净 CE / 鲁棒损失 (= cls_loss 本身) / 惩罚分布 /
            # 实际最坏竞争类的惩罚 / 鲁棒最坏间隔 (按样本数加权)
            out.update({'clean_ce': sums['clean_ce'] / total, 'robust_loss': sums['cls'] / total,
                        'pen_maxpair_mean': sums['pen_mean'] / total, 'pen_maxpair_std': sums['pen_std'] / total,
                        'worstcomp_pen_mean': sums['wcpen'] / total, 'robust_margin_mean': sums['mrob'] / total})
        return out

    def _evaluate(self, data_loader, mode='val'):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        # 2026-11-25: 按样本数加权的平均 (原先返回最后一个 batch 的损失); cls 即固定验证集上的 CE
        sums = {'total': 0.0, 'cls': 0.0, 'concept': 0.0, 'orth': 0.0, 'kl': 0.0}

        with torch.no_grad():
            for images, labels, attr_labels in tqdm(data_loader, desc=f"{mode.capitalize()} Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                attr_labels = attr_labels.to(self.device).float()

                concept_preds, cls_preds, reg_loss = self.model(images)

                # 双头支持: 评估只用 invariant head (主输出), 避免 ensemble 质疑
                if isinstance(cls_preds, tuple):
                    logits_ori, logits_inv = cls_preds
                    main_logits = logits_inv
                else:
                    main_logits = cls_preds

                cls_loss = self.loss_fns['cls'](main_logits, labels)
                # concept_loss = self.loss_fns['concept'](concept_preds, attr_labels)
                if self.args.beta > 0:
                    concept_loss = self.loss_fns['concept'](concept_preds, attr_labels.float())
                else:
                    concept_loss = torch.tensor(0.0, device=images.device)
                orth_loss = torch.abs(reg_loss).mean() if reg_loss is not None else 0.0

                # KL 正则 (与 train 一致, 评估时也记录): D_KL(softmax(residual) || uniform)
                kl_loss = torch.tensor(0.0, device=images.device)
                residual_softmax = None
                if hasattr(self.model, 'get_domain_weights'):
                    _, residual_softmax = self.model.get_domain_weights()
                if residual_softmax is not None:
                    N = residual_softmax.size(0)
                    entropy = -(residual_softmax * (residual_softmax + 1e-8).log()).sum()
                    kl_loss = torch.log(torch.tensor(float(N), device=images.device)) - entropy

                graph_loss = torch.tensor(0.0, device=images.device)
                if getattr(self.model, 'desc_laplacian', None) is not None and self.args.lambda_graph > 0:
                    d_ = self.model.domain_residual
                    graph_loss = d_ @ self.model.desc_laplacian @ d_
                total_loss = cls_loss + self.args.alpha * orth_loss + self.args.beta * concept_loss + self.args.lambda_kl * kl_loss + self.args.lambda_graph * graph_loss

                batch_size = images.size(0)
                total += batch_size
                correct += (main_logits.argmax(dim=1) == labels).sum().item()
                sums['total'] += float(total_loss) * batch_size
                sums['cls'] += float(cls_loss) * batch_size
                sums['concept'] += float(concept_loss) * batch_size
                sums['orth'] += float(orth_loss) * batch_size
                sums['kl'] += float(kl_loss) * batch_size

        return {
            f'{mode}_acc': 100 * correct / total,
            f'{mode}_loss': sums['total'] / total,
            f'{mode}_cls_loss': sums['cls'] / total,       # 该数据集上的平均 CE
            f'{mode}_concept_loss': sums['concept'] / total,
            f'{mode}_orth_loss': sums['orth'] / total,
            f'{mode}_kl_loss': sums['kl'] / total,
        }

    def _check_run_conflicts(self):
        """覆盖保护 (2026-11-25)。同名实验的产物已存在时默认拒绝启动:
          - 结果 json  logs/results/{run_name}.json
          - 权重日志目录 logs/weights/{run_name}/ (非空)
          - 检查点 best_<dataset>*.pth (仅 save_model 时; 按数据集命名, 不区分 run)
        显式 --overwrite_run 才覆盖: 先删除该 run 的旧结果 json 与权重日志目录。
        (背景: 11-20 的 1-epoch smoke 覆盖了收敛检查点; 11-25 的回归检查覆盖了同名权重日志。)
        续训 (--resume) 是另一件事, 本函数不处理。"""
        result_path = f"logs/results/{self.run_name}.json"
        wdir = f"logs/weights/{self.run_name}"
        conflicts = []
        if os.path.exists(result_path):
            conflicts.append(result_path)
        if os.path.isdir(wdir) and os.listdir(wdir):
            conflicts.append(wdir + "/")
        if self.args.save_model:
            for ck in (f"best_{self.args.dataset}.pth",
                       f"best_{self.args.dataset}_sel{getattr(self.args, 'model_selection', 'final')}.pth"):
                if os.path.exists(ck):
                    conflicts.append(ck)
        if not conflicts:
            return
        if not getattr(self.args, "overwrite_run", False):
            raise RuntimeError(
                f"拒绝启动: run_name={self.run_name} 的产物已存在 -> {conflicts}。"
                f"换一个 --run_tag; 不需要保存模型请加 --no_save_model; 确需覆盖请加 --overwrite_run")
        logger.warning(f"[overwrite_run] 按要求覆盖已存在的产物: {conflicts}")
        if os.path.exists(result_path):
            os.remove(result_path)
        if os.path.isdir(wdir):
            shutil.rmtree(wdir)

    def _ddo_loss_equal_weight(self):
        """统一的 DDO 评价指标: 在模型的【原始完整方向集】上不加权地计算 |h(P)| 的均值,
        与训练时的加权方式、方向集变体、α 都无关 (外审: 横向比较不能用各方法自己的训练损失)。
        模型不支持 DDO 时返回 None。"""
        fn = getattr(self.model, "ddo_loss_equal_weight", None)
        return fn() if fn is not None else None

    def _diag_batch(self):
        """梯度诊断用的固定批次: 训练集 (缓存特征) 的前 batch_size 个样本。
        不经过打乱的 DataLoader, 不消耗全局随机数, 因此不影响训练。"""
        if getattr(self, "_diag_batch_cache", None) is not None:
            return self._diag_batch_cache
        ds = self.train_loader.dataset
        n = min(self.args.batch_size, len(ds))
        if getattr(ds, "feats", None) is not None:          # main_cached 的 FeatureDataset
            x, y = ds.feats[:n].to(self.device), ds.labels[:n].to(self.device)
        else:                                                # 原始图像管线: 用独立 generator, 不动全局随机数
            loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            xb, yb = next(iter(loader))[:2]
            x, y = xb.to(self.device), yb.to(self.device)
        self._diag_batch_cache = (x, y)
        return x, y

    def _diag_param_names(self):
        """梯度诊断所取参数的名字 (与 self._diag_params 同序), 用于记录哪些参数不参与正则项"""
        mods = [("", self.model.classifier)] if not hasattr(self.model, "inv_classifier") or self.model.inv_classifier is None \
            else [("", self.model.classifier), ("inv.", self.model.inv_classifier)]
        return [f"{pre}{n}" for pre, m in mods for n, _ in m.named_parameters() if _.requires_grad]

    def _grad_diag(self, tag):
        """在固定批次上测量: 分类损失 CE 与【等权】DDO 损失对分类头参数的梯度范数、
        范数比与夹角余弦。只读取梯度 (autograd.grad), 不写入 .grad, 不改变优化器状态与随机数。"""
        x, y = self._diag_batch()
        self.model.eval()
        params = [p for p in self.model.classifier.parameters() if p.requires_grad]
        inv = getattr(self.model, "inv_classifier", None)
        if inv is not None:
            params += [p for p in inv.parameters() if p.requires_grad]
        self._diag_params = params
        _, cls_preds, reg_loss = self.model(x)
        if isinstance(cls_preds, tuple):                     # 与训练目标一致的双头 CE
            logits_ori, logits_inv = cls_preds
            ce = (self.loss_fns['cls'](logits_ori, y)
                  + self.args.lambda_inv * self.loss_fns['cls'](logits_inv, y))
        else:
            ce = self.loss_fns['cls'](cls_preds, y)
        g_ce = torch.autograd.grad(ce, params, retain_graph=True)
        g_ce = torch.cat([g.reshape(-1) for g in g_ce])
        rec = {'tag': tag, 'ce': float(ce), 'grad_ce_norm': float(g_ce.norm())}
        # (a) 本模型训练时实际优化的正则项 (含其加权方式 / 方向集变体; 不含 alpha) —— 强度匹配用。
        # LD 臂训练优化的不是 DDO 正则 (见 (c)), 这里跳过以免误读。
        if reg_loss is not None and self.ld_mode == "none":
            alpha = float(self.args.alpha)
            rt = reg_loss.abs().mean()
            # bias-free 的正则 (full_bf / split_bf) 对 LayerNorm.beta 与 Linear.bias 没有梯度 (不在计算图里),
            # autograd.grad 需 allow_unused, 未参与的参数梯度记为零向量并如实记录是哪些
            raw = torch.autograd.grad(rt, params, retain_graph=True, allow_unused=True)
            unused = [n for n, g in zip(self._diag_param_names(), raw) if g is None]
            g_rt = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                              for g, p in zip(raw, params)])
            ratio = float(g_rt.norm() / g_ce.norm().clamp_min(1e-12))
            # 外审: 真正作用于优化的是 alpha * 正则梯度, 所以同时记录乘以 alpha 之后的实际范数比
            # (外审: 该比值只描述固定诊断批次、所记录时刻、分类头参数上的测量, 不等于两个目标之间没有冲突)
            rec.update({'reg_train': float(rt), 'grad_reg_train_norm': float(g_rt.norm()),
                        'grad_norm_ratio_train_over_ce': ratio,
                        'alpha': alpha, 'grad_norm_ratio_train_alpha_over_ce': alpha * ratio,
                        'grad_cos_ce_regtrain': float((g_ce @ g_rt) / (g_ce.norm() * g_rt.norm()).clamp_min(1e-12))})
            if unused:
                rec['reg_params_without_grad'] = unused
        # (b) 统一评价口径: 原始完整方向集上的等权 DDO 损失 —— 横向比较用
        ddo = self._ddo_loss_equal_weight()
        if ddo is not None:
            g_dd = torch.autograd.grad(ddo, params)
            g_dd = torch.cat([g.reshape(-1) for g in g_dd])
            denom = (g_ce.norm() * g_dd.norm()).clamp_min(1e-12)
            rec.update({'ddo_eq': float(ddo), 'grad_ddo_norm': float(g_dd.norm()),
                        'grad_norm_ratio_ddo_over_ce': float(g_dd.norm() / g_ce.norm().clamp_min(1e-12)),
                        'grad_cos_ce_ddo': float((g_ce @ g_dd) / denom)})
        # (c) LD 臂 (预注册 §3z): 鲁棒损失的正则部分 (鲁棒 CE − 干净 CE) 对分类头的梯度比与夹角;
        # 与 §3x/§3y 配平用的"正则部分梯度比"同一口径。PGD 臂另抽查训练搜索的质量:
        # 训练用的 PGD(steps×restarts) 下界 对 更强的 PGD-50×4+闭式起点, 与 LD 闭式上界构成链
        # CE ≤ PGD-训练 ≤ PGD-强 ≤ LD —— 差值大说明训练搜索没追上最坏情况, 不能读作"PGD 训练温和"。
        if self.ld_mode != "none":
            M = self._ld_M()
            full = ld_training_loss(self.ld_mode, cls_preds, M, y, self.args.ld_eps, self.args.ld_lam,
                                    self.args.pgd_steps, self.args.pgd_restarts, self.args.pgd_seed)
            raw = torch.autograd.grad(full - ce, params, retain_graph=True)
            g_reg = torch.cat([t.reshape(-1) for t in raw])
            strength = float(self.args.ld_eps if self.ld_mode != "isda" else self.args.ld_lam)
            rec.update({'ld_mode': self.ld_mode, 'ld_strength': strength,
                        'grad_regpart_ratio_over_ce': float(g_reg.norm() / g_ce.norm().clamp_min(1e-12)),
                        'grad_cos_ce_regpart': float((g_ce @ g_reg) / (g_ce.norm() * g_reg.norm()).clamp_min(1e-12))})
            if self.ld_mode == "pgd":
                with torch.no_grad():
                    lgd, Md = cls_preds.detach(), M.detach()
                    p_tr, _ = pgd_worst_loss(lgd, Md, y, self.args.ld_eps, steps=self.args.pgd_steps,
                                             restarts=self.args.pgd_restarts, seed=self.args.pgd_seed)
                    p_st, _ = pgd_worst_loss(lgd, Md, y, self.args.ld_eps, steps=50, restarts=4,
                                             seed=self.args.pgd_seed + 1, closed_form_starts=True)
                    ldc = float(ld_dro_loss(lgd, Md, y, self.args.ld_eps))
                rec['pgd_search_quality'] = {'pgd_train_bound': float(p_tr), 'pgd_strong_bound': float(p_st),
                                             'ld_closed': ldc, 'strong_minus_train': float(p_st - p_tr),
                                             'chain_ok': bool(rec['ce'] <= float(p_tr) + 1e-5 <= float(p_st) + 1e-5 <= ldc + 1e-5)}
        self.grad_diag_records.append(rec)
        logger.info("[grad-diag %s] %s" % (tag, json.dumps(
            {k: round(v, 6) for k, v in rec.items() if isinstance(v, float)})))

    def _evaluate_source_val(self):
        """P1: 训练结束后在源域留出集 (干净 + 扰动) 上的评估。只有 main_cached 入口实现 (需要增强特征缓存)。"""
        raise NotImplementedError("--source_val_cache 只在 main_cached 入口可用")

    def run(self):
        """训练入口。同名 run 的运行锁 (2026-11-25, 外审要求): 另一个进程正在跑同一个 run_name 时
        拒绝启动, 防止两个进程互相覆盖结果与权重日志; 正常结束、异常、中断都会释放锁。"""
        self.run_name = self._make_run_name()
        self._lock = RunLock(f"logs/locks/{self.run_name}.lock", what=f"run_name={self.run_name}")
        with self._lock:
            self._run()

    def _make_run_name(self):
        # run_name: 数据集_方法tag_seed (多 seed 汇总脚本按此命名约定找文件)
        if self.args.CBM_type == 'clip_cbm_subspace':
            method_tag = getattr(self.args, 'ch4_mode', 'full')
            if getattr(self.args, 'use_c3_weights', False):
                method_tag = f"c3+{method_tag}"      # 2×2 联合实验: C3+C4
            if getattr(self.args, 'cal_rank', 0) > 0:
                method_tag += f"_cal{self.args.cal_rank}"   # P4 概念校准变体
        else:
            method_tag = getattr(self.args, 'weight_mode', 'prior_residual')
        if getattr(self.args, 'run_tag', ''):
            method_tag = self.args.run_tag            # tau 扫描等: 同模式多取值需区分文件
        # F1: 严格 DG 协议 (final) 的 run 命名加 _sel, 与历史 target-选模结果文件隔离
        if getattr(self.args, 'model_selection', 'final') == 'final' \
                and not getattr(self.args, 'run_tag', '') and not method_tag.endswith('_sel'):
            method_tag += '_sel'
        return f"{self.args.dataset}_{method_tag}_seed{self.args.seed}"

    def _run(self):
        self._t0 = time.time()                              # 训练成本 (预注册 §3z 要求记录; 含最终评估与存头)
        # 覆盖保护放在最前面: 产物冲突且未指定 --overwrite_run 时在此抛出 ——
        # 不打印训练头部、不进入训练 (运行锁已在 run() 中取得, 先于覆盖检查)
        self._check_run_conflicts()

        logger.info("\n" + "=" * 60)
        logger.info(f"Starting Training for {self.args.epochs} epochs")
        logger.info(f"Model selection protocol: {getattr(self.args, 'model_selection', 'final')} "
                    f"({'严格DG: 目标域不参与选模' if getattr(self.args, 'model_selection', 'final') == 'final' else '历史对照: 按target选模'})")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Model: {self.args.CBM_type}")
        logger.info(f"Alpha: {self.args.alpha}, Beta: {self.args.beta}, Lambda_KL: {self.args.lambda_kl}")
        if self.args.CBM_type == 'clip_cbm_subspace':
            logger.info(f"Ch4 mode: {getattr(self.args, 'ch4_mode', 'full')}, "
                        f"tau: {getattr(self.args, 'tau', 0.8)}, "
                        f"lambda_inv: {getattr(self.args, 'lambda_inv', 1.0)}, Seed: {self.args.seed}")
        else:
            logger.info(f"Weight mode: {getattr(self.args, 'weight_mode', 'prior_residual')}, Seed: {self.args.seed}")
        logger.info("=" * 60 + "\n")

        if self.args.wandb:
            wandb.init(
                project="LanCE",
                name=f"{self.run_name}-{time.strftime('%m%d%H%M')}",
                config=vars(self.args)
            )
            wandb.watch(self.model, log_freq=100)

        # 2026-11-25: 逐 epoch 指标与诊断记录 (写入结果 json, 供机制分析)
        self.epoch_metrics = []
        self.grad_diag_records = []
        self.final_metrics = None
        self._completed_epochs = 0
        # --target_eval final_only: 目标域不参与训练与选模, 仅在最后一个 epoch 评估一次 (默认 per_epoch, 行为不变)。
        # 措辞上的区分 (外审): 本开关保证的是"不评估"; main_cached 入口在此模式下把目标域特征缓存
        # 延迟到最终评估时才读入内存 (见 main_cached._LazyTargetLoader)。
        eval_target_every_epoch = getattr(self.args, 'target_eval', 'per_epoch') == 'per_epoch'
        # P1 (2026-11-26): --target_eval none —— 训练期间完全不评估目标域 (目标域结果只在源域选择锁定后,
        # 用已保存的分类头做事后评估, 见 eval_heads_targets.py)
        eval_target_none = getattr(self.args, 'target_eval', 'per_epoch') == 'none'
        self.source_val = None

        try:
            for epoch in range(self.args.epochs):
                self._lock.heartbeat()                     # 运行锁心跳: 每个 epoch 一次
                epoch_start = time.time()

                if getattr(self.args, 'grad_diag', False) and (epoch + 1) in GRAD_DIAG_EPOCHS:
                    self._grad_diag(f"before_epoch{epoch + 1}")

                train_metrics = self._train_epoch(epoch)

                source_metrics = self._evaluate(self.source_test_loader, 'source')
                is_last = self._selection_epoch(epoch, self.args.epochs)
                target_metrics = (self._evaluate(self.target_test_loader, 'target')
                                  if ((eval_target_every_epoch or is_last) and not eval_target_none) else None)

                # domain 权重诊断 (每 epoch): r/δ/prior_w/final_w/H(q)/max/min/std/Δw
                weight_diagnostics = self._log_domain_weights(epoch)

                if target_metrics is not None and \
                        target_metrics['target_acc'] > self.best_metrics['target_acc']:
                    self.best_metrics.update({
                        'source_acc': source_metrics['source_acc'],
                        'target_acc': target_metrics['target_acc'],
                        'epoch': epoch
                    })
                    # self._save_concept_logs()
                    if self.args.save_model:
                        torch.save(self.model.state_dict(), f"best_{self.args.dataset}.pth")

                # ===== F1 选模协议 (2026-11-07, 专家意见): 严格 DG 下目标域不参与选模 =====
                # model_selection=final  : 固定 epoch (训练结束时的模型), 目标域仅测试 —— 默认
                # model_selection=target : 旧行为 (按 target_acc 选 best), 仅历史对照用
                if is_last:
                    self.selected_metrics = {
                        'source_acc': source_metrics['source_acc'],
                        'target_acc': target_metrics['target_acc'] if target_metrics is not None else None,
                        'epoch': epoch,
                        'protocol': self.args.model_selection,
                    }
                    if self.args.save_model:
                        torch.save(self.model.state_dict(),
                                   f"best_{self.args.dataset}_sel{self.args.model_selection}.pth")
                    self.final_metrics = {'source': dict(source_metrics),
                                          'target': (dict(target_metrics) if target_metrics is not None else None)}

                # 2026-11-25: 逐 epoch 的平均损失与源域验证 CE (目标域精度仅在评估过的 epoch 记录)
                # 两个 DDO 字段口径不同 (外审第四轮指出旧字段 ddo_eq 名不副实, 已改名):
                #   ddo                  训练时实际优化的 DDO 项 (本臂方向集, 含加权)
                #   ddo_own_unweighted   本臂训练方向集上的未加权损失 —— 方向集随实验臂变化, 不跨方法可比
                #   ddo_full_eval        epoch 结束后在【原始完整方向集】上的等权损失 —— 统一评价口径, 可跨方法比较
                with torch.no_grad():
                    ddo_full = self._ddo_loss_equal_weight()
                self.epoch_metrics.append({
                    'epoch': epoch + 1,
                    'cls': train_metrics['cls_loss'],
                    'ddo': train_metrics['orth_loss'],
                    'ddo_own_unweighted': train_metrics['ddo_own_unweighted_loss'],
                    'ddo_full_eval': float(ddo_full) if ddo_full is not None else None,
                    'kl': train_metrics['kl_loss'],
                    'source_acc': source_metrics['source_acc'],
                    'source_ce': source_metrics['source_cls_loss'],
                    'target_acc': target_metrics['target_acc'] if target_metrics is not None else None,
                    # LD 臂 (预注册 §3z) 的逐 epoch 监控量 (来自 _train_epoch 的返回 dict; 非 LD 臂为 None):
                    # 干净 CE / 鲁棒损失 / 惩罚分布 / 实际最坏竞争类的惩罚 / 鲁棒最坏间隔
                    'clean_ce': train_metrics.get('clean_ce'),
                    'robust_loss': train_metrics.get('robust_loss'),
                    'pen_maxpair_mean': train_metrics.get('pen_maxpair_mean'),
                    'pen_maxpair_std': train_metrics.get('pen_maxpair_std'),
                    'worstcomp_pen_mean': train_metrics.get('worstcomp_pen_mean'),
                    'robust_margin_mean': train_metrics.get('robust_margin_mean'),
                })
                self._completed_epochs = epoch + 1

                epoch_time = time.time() - epoch_start
                log_data = {
                    'epoch': epoch + 1,
                    'epoch_time': epoch_time,
                    **train_metrics,
                    **source_metrics,
                    **(target_metrics or {}),
                    **weight_diagnostics
                }

                logger.info("\n" + "-" * 50)
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"Time: {epoch_time:.2f}s")
                logger.info(f"Train Acc: {train_metrics['train_acc']:.2f}%")
                logger.info(f"Source Acc: {source_metrics['source_acc']:.2f}%")
                if target_metrics is not None:
                    logger.info(f"Target Acc: {target_metrics['target_acc']:.2f}%")
                else:
                    logger.info("Target Acc: (not evaluated; --target_eval final_only)")
                if eval_target_every_epoch:
                    logger.info(
                        f"Best Target Acc: {self.best_metrics['target_acc']:.2f}% @ Epoch {self.best_metrics['epoch'] + 1}")
                logger.info("-" * 50)

                if self.args.wandb:
                    wandb.log(log_data)

            if getattr(self.args, 'grad_diag', False):
                self._grad_diag("final")

            # P1 (2026-11-26): 源域增强验证 —— heldout_val 上的干净 + 扰动 Acc / CE。放在训练与梯度诊断之后;
            # 与目标域评估是互不相干的两步: --target_eval none 时上面从未读过目标域特征缓存
            if getattr(self.args, "source_val_cache", ""):
                self.source_val = self._evaluate_source_val()

        except KeyboardInterrupt:
            # 2026-11-25: 中断的 run 不得留下"部分训练"的结果 json; 原样上抛, 由外层处理
            self._training_failed = True
            raise
        except Exception as e:
            # 2026-11-25: 标记失败 —— 原先 finally 无条件写结果 json, 失败的 run 会留下
            # 0.0 的"结果"并被汇总/断点续跑脚本当作已完成 (11-24 的 12 个静默零结果即此因)
            self._training_failed = True
            logger.error(f"Training interrupted: {str(e)}", exc_info=True)
            if self.args.wandb:
                wandb.alert(
                    title="Training Failed",
                    text=f"Error at epoch {epoch + 1}: {str(e)}"
                )
            # 2026-11-25: 继续上抛, 使进程以非零状态退出 (原先异常被吞掉、退出码为 0,
            # 批处理脚本无法发现失败)。结果 json 由 finally 中的 _training_failed 标记拦住。
            raise
        finally:
            if self.args.wandb:
                wandb.summary.update(self.best_metrics)
                wandb.finish()

            # --save_head (2026-11-25, Stage 0 机制诊断用): 只在训练完整结束时保存**分类头** (W, gamma, beta, b,
            # 几 KB), 按 run 命名, 不触碰任何按数据集命名的检查点。正式矩阵都用 --no_save_model, 没有收敛的分类头,
            # Stage 0A/0B 需要它 —— 另跑少量配置并带此开关 (doc/主线B实施规格.md §1、§2)。
            if getattr(self.args, "save_head", False) and not getattr(self, "_training_failed", False) \
                    and self._completed_epochs == self.args.epochs:
                os.makedirs("logs/heads", exist_ok=True)
                head = {n: p.detach().cpu() for n, p in self.model.classifier.named_parameters()}
                torch.save({"classifier": head, "run_name": self.run_name,
                            "run_config": {k: jsonable(v) for k, v in sorted(vars(self.args).items())},
                            "git": git_info(), "ddo_dirs_info": getattr(self.model, "ddo_dirs_info", None)},
                           f"logs/heads/{self.run_name}.pt")
                logger.info(f"Head saved to logs/heads/{self.run_name}.pt")

            # 保存结果 json (多 seed 汇总脚本 aggregate_results.py 按命名约定读取)
            # 文件名: logs/results/{dataset}_{method_tag}_seed{seed}.json
            os.makedirs("logs/results", exist_ok=True)
            if self.args.CBM_type == 'clip_cbm_subspace':
                method_tag = getattr(self.args, 'ch4_mode', 'full')
                if getattr(self.args, 'use_c3_weights', False):
                    method_tag = f"c3+{method_tag}"
                if getattr(self.args, 'cal_rank', 0) > 0:
                    method_tag += f"_cal{self.args.cal_rank}"
            else:
                method_tag = getattr(self.args, 'weight_mode', 'prior_residual')
            if getattr(self.args, 'run_tag', ''):
                method_tag = self.args.run_tag
            # F1: 严格 DG 协议 (final) 时结果文件加 sel 后缀, 与历史 target-选模结果隔离
            sel_mode = getattr(self.args, 'model_selection', 'final')
            if sel_mode == 'final' and not getattr(self.args, 'run_tag', '') \
                    and not method_tag.endswith('_sel'):
                method_tag += '_sel'
            # 2026-11-25: 仅当训练完整跑满 epochs 且未失败时才算完成、才写结果 json
            completed = (not getattr(self, "_training_failed", False)
                         and self._completed_epochs == self.args.epochs)
            result_path = f"logs/results/{self.run_name}.json"
            final_diag = None
            if completed:
                # 最终模型的统一诊断: 固定源域验证集上的 CE / 目标域精度 / 等权 DDO 损失
                with torch.no_grad():
                    ddo_eq_final = self._ddo_loss_equal_weight()
                fm = self.final_metrics or {}
                ft = fm.get('target') or {}                 # --target_eval none 时为 None
                final_diag = {
                    'source_acc': fm.get('source', {}).get('source_acc'),
                    'source_ce': fm.get('source', {}).get('source_cls_loss'),
                    'target_acc': ft.get('target_acc'),
                    'target_ce': ft.get('target_cls_loss'),
                    'ddo_eq': float(ddo_eq_final) if ddo_eq_final is not None else None,
                }
            sel = self.selected_metrics or {}
            result = {
                'dataset': self.args.dataset,
                'model_type': self.args.CBM_type,
                'weight_mode': method_tag,
                'seed': self.args.seed,
                'alpha': self.args.alpha,
                'beta': self.args.beta,
                'lambda_kl': self.args.lambda_kl,
                'model_selection': sel_mode,
                # 外审要求: 记录实际生效的损失配置 (graph 泄漏事件的防护)
                'loss_config': {
                    'lambda_graph': getattr(self.args, 'lambda_graph', 0.0),
                    'graph_active': bool(getattr(self.model, 'desc_laplacian', None) is not None
                                         and getattr(self.args, 'lambda_graph', 0) > 0),
                    'cal_rank': getattr(self.args, 'cal_rank', 0),
                    'lambda_cal': getattr(self.args, 'lambda_cal', 0.01),
                    'protect_k': getattr(self.args, 'protect_k', 0),
                },
                # 2026-11-25: 完整有效配置 (全部命令行参数, 含 lr / weight_decay / batch_size /
                # epochs / seed / alpha / 各损失系数) 与代码版本, 外审要求不能再靠文件名反推
                'run_config': {k: jsonable(v) for k, v in sorted(vars(self.args).items())},
                'git': git_info(),
                'status': 'completed' if completed else 'incomplete',
                'completed_epochs': self._completed_epochs,
                # 2026-11-25: 机制分析用 —— 逐 epoch 平均损失 (ddo 为训练实际优化的项, ddo_eq 为等权口径)、
                # 最终模型的统一诊断、梯度诊断 (--grad_diag 时)
                'epoch_metrics': self.epoch_metrics,
                'final_diag': final_diag,
                'grad_diag': (self.grad_diag_records if getattr(self.args, 'grad_diag', False) else None),
                # 2026-11-25 (外审): 实际构造的训练方向集 (类型 / 行数 / k / 谱能量占比 / 随机标架种子)
                # 与方向张量的 sha256 —— 不能只凭命令行参数反推模型里约束的是哪一组方向
                'ddo_dirs_info': getattr(self.model, 'ddo_dirs_info', None),
                # P1 (2026-11-26): 源域增强验证 (heldout_val 上干净 + 每种扰动的 Acc / CE) 与训练子集的来源记录
                # (两者都只含 float / int / str / None, 可直接写 json; run_status.jsonable 只处理标量, 不能用来包字典)
                'source_val': getattr(self, "source_val", None),
                'train_subset': getattr(self, "train_subset_info", None),
                # 描述符库身份 (2026-11-26, 外审第九轮 P4): 这个 run 用的是哪一份词库 (完整文本的 sha256)
                'descriptor_bank': getattr(self.train_dataset, "descriptor_bank_info", None),
                # LD 臂 (2026-11-26, 外审第十轮; 预注册 §3z): 鲁棒损失配置、椭球生成元身份与标定来源;
                # 非 LD 臂为 None。训练成本 (秒, 含最终评估与存头) 一并记录
                'ld_config': getattr(self, "ld_meta", None),
                'elapsed_seconds': round(time.time() - self._t0, 1),
                **self.best_metrics,
                # 严格 DG 口径: 训练结束模型的指标 (final 协议下与 best 不同)
                **{f'sel_{k}': v for k, v in sel.items()},
                'target_acc': (sel.get('target_acc', self.best_metrics['target_acc'])
                               if sel_mode == 'final' else self.best_metrics['target_acc']),
            }
            if not completed:
                logger.error(f"[main] 训练未完成 (完成 {self._completed_epochs}/{self.args.epochs} epochs): "
                             f"不写 {result_path} (避免被汇总/续跑脚本当作已完成; 同名旧结果如有则保持不动)")
            else:
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Result saved to {result_path}")
                if final_diag:
                    logger.info("[final-diag] " + json.dumps(
                        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in final_diag.items()}))

            logger.info("\n" + "=" * 60)
            logger.info("Training Completed")
            logger.info(f"Protocol: {sel_mode}")
            if sel_mode == 'final' and sel:
                if sel.get('target_acc') is not None:
                    logger.info(f"[selected@final] Source {sel['source_acc']:.2f}% | "
                                f"Target {sel['target_acc']:.2f}% (严格DG主指标)")
                else:
                    # --target_eval none: 目标域不在这里评估; 源域选择锁定后用 logs/heads/ 的分类头做事后评估
                    logger.info(f"[selected@final] Source {sel['source_acc']:.2f}% | "
                                f"Target 未评估 (--target_eval none; 事后评估用 eval_heads_targets.py)")
            logger.info(f"Best(Source-or-target监视) Source {self.best_metrics['source_acc']:.2f}% | "
                        f"Target {self.best_metrics['target_acc']:.2f}% (仅对照)")
            logger.info("=" * 60)


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    try:
        session = TrainingSession(args)
        session.run()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        # 2026-11-25: 以非零状态退出 (原先只记录日志、退出码为 0, 批处理脚本无法发现失败)
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)
        sys.exit(1)