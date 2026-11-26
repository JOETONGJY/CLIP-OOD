"""
缓存特征上的等价训练入口(与 main.py 同一条路径, 仅两处替换):

  1. _prepare_datasets: 仍按原管线构建数据集对象(train_dataset 提供
     domain_diffs/domain_weights/classname2id —— 与 main.py 完全一致),
     但三个 loader 换成读 cache/*.pt 的 FeatureDataset。
  2. _init_model: 调原逻辑建模型(init 内用真 CLIP 编码文本),
     随后把 model.clip_model 替换为恒等桩 —— forward 里的
     encode_image(images) 直接返回缓存特征, 其余计算一行不改。

数学等价性: CLIP 冻结 + preprocess 确定性 => 缓存特征 == 现场 encode_image 输出。
等价性验证: 与 main.py 同参跑 2 epoch, 对比逐 epoch 指标(见 experiments 记录)。

P1 (2026-11-26, 外审第九轮; 预注册见 RESULTS.md §3v) 的三个开关 —— 默认全部关闭, 已有实验的行为不变:
  --train_subset_cache <增强缓存>   只用其 split["train_core"] 的样本训练 (逐项核对划分与标签, 见 _make_train_subset)
  --source_val_cache   <增强缓存>   训练结束后在 split["heldout_val"] 上评估干净 + 每种扰动的 Acc / CE, 写入结果 json
  --target_eval none                 训练期间完全不评估目标域、不读取目标域特征缓存; 目标域结果在源域选择锁定之后
                                    用 --save_head 保存的分类头做事后评估 (eval_heads_targets.py)

两份缓存的归一化方式不同: cache/<ds>_train.pt 是编码器原始输出 (未归一化, 归一化在模型 forward 里按 float32 做),
cache/<ds>_train_aug.pt 是 L2 归一化后的 fp16。两者走的是同一条 forward 路径, 差别只有 fp16 舍入
(2026-11-26 实测: 逐样本相对误差中位数 1.3e-3, 对 27 个分类头在 heldout_val 上的 Acc 影响不超过 0.08 个百分点)。

用法(与 main.py 参数完全一致):
  python main_cached.py --dataset CUB --CBM_type clip_cbm --weight_mode none \
      --epochs 100 --batch_size 256 --alpha 1.0 --beta 0 --seed 1 \
      --CLIP_type "F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
"""
import hashlib
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from main import TrainingSession


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class _IdentityEncoder(nn.Module):
    """encode_image 恒等桩: 输入已是缓存特征(必须是 nn.Module 才能替换注册的子模块)"""
    def encode_image(self, x):
        return x


class FeatureDataset(Dataset):
    def __init__(self, cache_path, index=None):
        d = torch.load(cache_path, map_location="cpu")
        if index is not None:                                    # P1: 只取子集 (按给定下标物化为新张量)
            index = torch.as_tensor(index, dtype=torch.long)
            d = {k: v[index] for k, v in d.items() if torch.is_tensor(v)}
        self.feats = d['feats']    # fp16 (N,768)
        self.labels = d['labels']  # long (N,)
        self.attrs = d['attrs']    # float (N,C)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], int(self.labels[idx]), self.attrs[idx]


class _LazyTargetLoader:
    """--target_eval final_only 下, 目标域特征缓存延迟到第一次 (也是唯一一次) 评估时才读入内存
    (2026-11-25, 外审: 此前缓存在初始化阶段就已加载, "训练期间不访问目标域"的表述不准确)。
    对外接口与 DataLoader 相同 (__iter__ / __len__ / dataset); torch.load 不消耗随机数, 不影响训练结果。"""
    def __init__(self, cache_path, batch_size):
        self._path, self._bs, self._loader = cache_path, batch_size, None

    def _get(self):
        if self._loader is None:
            self._loader = DataLoader(FeatureDataset(self._path), batch_size=self._bs,
                                      shuffle=False, num_workers=0)
            from main import logger
            logger.info(f"[cached] 目标域特征缓存此刻才读入 (final_only): {self._path}")
        return self._loader

    @property
    def dataset(self):
        return self._get().dataset

    def __len__(self):
        return len(self._get())

    def __iter__(self):
        return iter(self._get())


class CachedTrainingSession(TrainingSession):
    def _prepare_datasets(self):
        # 原管线构建(文本侧: reliability/domain_diffs 计算, 不读图片)
        (self.train_dataset, _, self.source_test_dataset, _,
         self.target_test_dataset, _) = __import__('data').get_dataset_classes(self.args)

        # 释放数据集持有的 CLIP(model init 会再加载一份)
        for ds in (self.train_dataset, self.source_test_dataset, self.target_test_dataset):
            if getattr(ds, "clip_model", None) is not None:
                ds.clip_model = None
        torch.cuda.empty_cache()

        # C-1 冗余参数接线 (与 main.py 同一逻辑, 2026-11-22 修复 topk 断链)
        self._apply_redundancy_args()

        # 缓存特征 loader
        name = self.args.dataset
        train_cache = os.path.join("cache", f"{name}_train.pt")
        self.train_subset_info = None
        train_ds = FeatureDataset(train_cache)
        subset_path = getattr(self.args, "train_subset_cache", "")
        if subset_path:                                           # P1: 只用 train_core 子集训练
            train_ds, self.train_subset_info = self._make_train_subset(train_ds, train_cache, subset_path)
        self.train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        self.source_test_loader = DataLoader(
            FeatureDataset(os.path.join("cache", f"{name}_source_test.pt")),
            batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        target_cache = os.path.join("cache", f"{name}_target_test.pt")
        tev = getattr(self.args, 'target_eval', 'per_epoch')
        if tev == 'final_only':
            # 目标域不参与训练与选模: 特征缓存也等到最终评估时才读 (见 _LazyTargetLoader)。
            # 目标域数据集对象 (划分表与属性标签) 仍由上面 get_dataset_classes 构建。
            self.target_test_loader = _LazyTargetLoader(target_cache, self.args.batch_size)
            target_note = "延迟读取 (final_only, 仅最终评估时载入)"
        elif tev == 'none':
            # P1: 训练期间完全不评估目标域, 目标域特征缓存全程不读取 (上面的目标域数据集对象只用文本侧的划分表)。
            # 目标域结果在源域选择规则与结果锁定之后, 用 --save_head 保存的分类头做事后评估 (eval_heads_targets.py)。
            self.target_test_loader = None
            target_note = "不评估 (--target_eval none)"
        else:
            self.target_test_loader = DataLoader(
                FeatureDataset(target_cache), batch_size=self.args.batch_size, shuffle=False, num_workers=0)
            target_note = f"{len(self.target_test_loader.dataset):,}"

        from main import logger
        logger.info(f"[cached] Train samples: {len(self.train_loader.dataset):,}"
                    + (f" (train_core 子集, 来自 {subset_path})" if subset_path else "")
                    + f" | Source test: {len(self.source_test_loader.dataset):,} | Target test: {target_note}")

    def _make_train_subset(self, full_ds, train_cache, subset_path):
        """P1 (2026-11-26): 只用划分文件 split["train_core"] 的样本训练。
        划分来自 cache/CUB_train_aug.pt (seed 42, 分层 80/20), 样本顺序与训练缓存一致 —— 这里逐项核对:
        下标合法且无重复、与 heldout_val 不相交、两者合起来正好是完整训练集、子集样本的标签与训练缓存一致。
        返回 (子集 FeatureDataset, 写入结果 json 的来源记录)。"""
        aug = torch.load(subset_path, map_location="cpu", weights_only=False)
        idx = torch.as_tensor(aug["split"]["train_core"], dtype=torch.long)
        val_idx = torch.as_tensor(aug["split"]["heldout_val"], dtype=torch.long)
        n = len(full_ds.labels)
        assert idx.numel() > 0 and int(idx.min()) >= 0 and int(idx.max()) < n, "train_core 下标越界"
        assert idx.unique().numel() == idx.numel() and val_idx.unique().numel() == val_idx.numel(), "划分内有重复下标"
        assert not np.intersect1d(idx.numpy(), val_idx.numpy()).size, "train_core 与 heldout_val 有交集"
        assert idx.numel() + val_idx.numel() == n, "train_core + heldout_val 不等于完整训练集"
        assert torch.equal(torch.as_tensor(aug["labels"])[idx], full_ds.labels[idx]), \
            "子集缓存与训练缓存的标签不一致: 两份缓存的样本顺序不同, 不能用下标取子集"
        info = {"split_cache": subset_path, "split_cache_sha256": _sha256(subset_path), "split_name": "train_core",
                "n_train": int(idx.numel()), "n_heldout": int(val_idx.numel()), "n_full": n,
                "subset_sha256": hashlib.sha256(idx.numpy().tobytes()).hexdigest(),
                "train_cache": train_cache, "train_cache_sha256": _sha256(train_cache),
                "labels_match_train_cache": True}
        return FeatureDataset(train_cache, index=idx), info

    def _evaluate_source_val(self):
        """P1 (2026-11-26): 训练结束后在 heldout_val (干净 + 每种扰动) 上的评估。
        走与目标域评估完全相同的 _evaluate: 缓存特征 → 模型 forward (float32 下 L2 归一化) → logits → Acc / CE。
        属性标签填零 (只允许 beta=0 的配置, 此时属性不参与损失)。"""
        assert self.args.beta == 0, "--source_val_cache 只支持 beta=0 (属性标签填零, 不参与损失)"
        path = self.args.source_val_cache
        d = torch.load(path, map_location="cpu", weights_only=False)
        sha = _sha256(path)
        subset = getattr(self, "train_subset_info", None)
        if subset is not None:
            # 训练子集与验证集必须来自同一份划分文件
            assert os.path.abspath(subset["split_cache"]) == os.path.abspath(path) and subset["split_cache_sha256"] == sha, \
                "--train_subset_cache 与 --source_val_cache 必须是同一份缓存 (否则不能保证训练集与验证集不相交)"
        idx = torch.as_tensor(d["split"]["heldout_val"], dtype=torch.long)
        labels = torch.as_tensor(d["labels"])[idx]
        n_concept = self.train_loader.dataset.attrs.size(1)

        def run(feats):
            loader = DataLoader(TensorDataset(feats, labels, torch.zeros(feats.size(0), n_concept)),
                                batch_size=self.args.batch_size, shuffle=False, num_workers=0)
            m = self._evaluate(loader, "val")
            return {"acc": m["val_acc"], "ce": m["val_cls_loss"]}

        out = {"cache": path, "cache_sha256": sha, "split_name": "heldout_val", "n": int(idx.numel()),
               "clean": run(d["clean_feats"][idx]), "aug": {}}
        for aug_name, t in d["aug_feats"].items():
            out["aug"][aug_name] = run(t[idx])
        out["mean_aug_acc"] = float(np.mean([v["acc"] for v in out["aug"].values()]))
        out["mean_aug_ce"] = float(np.mean([v["ce"] for v in out["aug"].values()]))
        from main import logger
        logger.info(f"[source-val] heldout_val n={out['n']}: clean {out['clean']['acc']:.2f}% / CE {out['clean']['ce']:.4f} | "
                    f"{len(out['aug'])} 种扰动均值 {out['mean_aug_acc']:.2f}% / CE {out['mean_aug_ce']:.4f}")
        return out

    def _init_model(self):
        super()._init_model()          # 原逻辑: 真 CLIP 编码文本/concept/SVD
        self.model.clip_model = _IdentityEncoder()   # 替换后显存释放, 训练零 GPU 负载


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

    session = CachedTrainingSession(args)
    session.run()
