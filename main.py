import os
import time
import json
import logging
import random
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

    def _prepare_datasets(self):
        """CUB, AWA2, LADA, LADV"""
        self.train_dataset, self.train_loader, self.source_test_dataset, self.source_test_loader, self.target_test_dataset, self.target_test_loader = get_dataset_classes(self.args)

        logger.info(f"Dataset {self.args.dataset} loaded")
        logger.info(f"Train samples: {len(self.train_dataset):,}")
        logger.info(f"Source test samples: {len(self.source_test_dataset):,}")
        logger.info(f"Target test samples: {len(self.target_test_dataset):,}")

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
            ).to(self.device)
        else:
            # 改动时间26.4.29-----------------------------------
            self.model = model_class(
                args=self.args,
                class_names=list(self.train_dataset.classname2id.keys()),
                concept_names=list(self.train_dataset.concept2id.keys()),
                domain_diffs=self.train_dataset.domain_diffs,
                domain_weights=getattr(self.train_dataset, "domain_weights", None),
                weight_mode=getattr(self.args, "weight_mode", "prior_residual")
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
            torch.save(save_dict, f"logs/weights/{run_name}/epoch_{epoch:04d}.pt")

            # Δw 摘要 (final - prior), 只在日志里打 Top 修正
            if prior_w is not None:
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
                cls_loss = self.loss_fns['cls'](cls_preds, labels)
                main_logits = cls_preds
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

            total_loss = (
                    cls_loss
                    + self.args.alpha * orth_loss
                    + self.args.beta * concept_loss
                    + self.args.lambda_kl * kl_loss
            )

            total_loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total += batch_size
            correct += (main_logits.argmax(dim=1) == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%",
                'CLS': f"{cls_loss.item():.4f}",
                'CON': f"{concept_loss.item():.4f}",
                'ORT': f"{orth_loss.item():.4f}",
                'KL': f"{kl_loss.item():.4f}"
            })

        return {
            'train_acc': 100 * correct / total,
            'train_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'concept_loss': concept_loss.item(),
            'orth_loss': orth_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def _evaluate(self, data_loader, mode='val'):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

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

                total_loss = cls_loss + self.args.alpha * orth_loss + self.args.beta * concept_loss + self.args.lambda_kl * kl_loss

                batch_size = images.size(0)
                total += batch_size
                correct += (main_logits.argmax(dim=1) == labels).sum().item()

        return {
            f'{mode}_acc': 100 * correct / total,
            f'{mode}_loss': total_loss.item(),
            f'{mode}_cls_loss': cls_loss.item(),
            f'{mode}_concept_loss': concept_loss.item(),
            f'{mode}_orth_loss': orth_loss.item(),
            f'{mode}_kl_loss': kl_loss.item()
        }

    def run(self):
        logger.info("\n" + "=" * 60)
        logger.info(f"Starting Training for {self.args.epochs} epochs")
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

        # run_name: 数据集_方法tag_seed (多 seed 汇总脚本按此命名约定找文件)
        if self.args.CBM_type == 'clip_cbm_subspace':
            method_tag = getattr(self.args, 'ch4_mode', 'full')
            if getattr(self.args, 'use_c3_weights', False):
                method_tag = f"c3+{method_tag}"      # 2×2 联合实验: C3+C4
        else:
            method_tag = getattr(self.args, 'weight_mode', 'prior_residual')
        self.run_name = f"{self.args.dataset}_{method_tag}_seed{self.args.seed}"

        if self.args.wandb:
            wandb.init(
                project="LanCE",
                name=f"{self.run_name}-{time.strftime('%m%d%H%M')}",
                config=vars(self.args)
            )
            wandb.watch(self.model, log_freq=100)

        try:
            for epoch in range(self.args.epochs):
                epoch_start = time.time()

                train_metrics = self._train_epoch(epoch)

                source_metrics = self._evaluate(self.source_test_loader, 'source')
                target_metrics = self._evaluate(self.target_test_loader, 'target')

                # domain 权重诊断 (每 epoch): r/δ/prior_w/final_w/H(q)/max/min/std/Δw
                weight_diagnostics = self._log_domain_weights(epoch)

                if target_metrics['target_acc'] > self.best_metrics['target_acc']:
                    self.best_metrics.update({
                        'source_acc': source_metrics['source_acc'],
                        'target_acc': target_metrics['target_acc'],
                        'epoch': epoch
                    })
                    # self._save_concept_logs()
                    if self.args.save_model:
                        torch.save(self.model.state_dict(), f"best_{self.args.dataset}.pth")

                epoch_time = time.time() - epoch_start
                log_data = {
                    'epoch': epoch + 1,
                    'epoch_time': epoch_time,
                    **train_metrics,
                    **source_metrics,
                    **target_metrics,
                    **weight_diagnostics
                }

                logger.info("\n" + "-" * 50)
                logger.info(f"Epoch {epoch + 1} Summary:")
                logger.info(f"Time: {epoch_time:.2f}s")
                logger.info(f"Train Acc: {train_metrics['train_acc']:.2f}%")
                logger.info(f"Source Acc: {source_metrics['source_acc']:.2f}%")
                logger.info(f"Target Acc: {target_metrics['target_acc']:.2f}%")
                logger.info(
                    f"Best Target Acc: {self.best_metrics['target_acc']:.2f}% @ Epoch {self.best_metrics['epoch'] + 1}")
                logger.info("-" * 50)

                if self.args.wandb:
                    wandb.log(log_data)

        except Exception as e:
            logger.error(f"Training interrupted: {str(e)}", exc_info=True)
            if self.args.wandb:
                wandb.alert(
                    title="Training Failed",
                    text=f"Error at epoch {epoch + 1}: {str(e)}"
                )
        finally:
            if self.args.wandb:
                wandb.summary.update(self.best_metrics)
                wandb.finish()

            # 保存结果 json (多 seed 汇总脚本 aggregate_results.py 按命名约定读取)
            # 文件名: logs/results/{dataset}_{method_tag}_seed{seed}.json
            os.makedirs("logs/results", exist_ok=True)
            if self.args.CBM_type == 'clip_cbm_subspace':
                method_tag = getattr(self.args, 'ch4_mode', 'full')
                if getattr(self.args, 'use_c3_weights', False):
                    method_tag = f"c3+{method_tag}"
            else:
                method_tag = getattr(self.args, 'weight_mode', 'prior_residual')
            result = {
                'dataset': self.args.dataset,
                'model_type': self.args.CBM_type,
                'weight_mode': method_tag,
                'seed': self.args.seed,
                'alpha': self.args.alpha,
                'beta': self.args.beta,
                'lambda_kl': self.args.lambda_kl,
                **self.best_metrics,
            }
            result_path = f"logs/results/{self.run_name}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to {result_path}")

            logger.info("\n" + "=" * 60)
            logger.info("Training Completed")
            logger.info(f"Best Source Accuracy: {self.best_metrics['source_acc']:.2f}%")
            logger.info(f"Best Target Accuracy: {self.best_metrics['target_acc']:.2f}%")
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
    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)