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

用法(与 main.py 参数完全一致):
  python main_cached.py --dataset CUB --CBM_type clip_cbm --weight_mode none \
      --epochs 100 --batch_size 256 --alpha 1.0 --beta 0 --seed 1 \
      --CLIP_type "F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from main import TrainingSession


class _IdentityEncoder(nn.Module):
    """encode_image 恒等桩: 输入已是缓存特征(必须是 nn.Module 才能替换注册的子模块)"""
    def encode_image(self, x):
        return x


class FeatureDataset(Dataset):
    def __init__(self, cache_path):
        d = torch.load(cache_path, map_location="cpu")
        self.feats = d['feats']    # fp16 (N,768)
        self.labels = d['labels']  # long (N,)
        self.attrs = d['attrs']    # float (N,C)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], int(self.labels[idx]), self.attrs[idx]


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

        # 缓存特征 loader
        name = self.args.dataset
        self.train_loader = DataLoader(
            FeatureDataset(os.path.join("cache", f"{name}_train.pt")),
            batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        self.source_test_loader = DataLoader(
            FeatureDataset(os.path.join("cache", f"{name}_source_test.pt")),
            batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        self.target_test_loader = DataLoader(
            FeatureDataset(os.path.join("cache", f"{name}_target_test.pt")),
            batch_size=self.args.batch_size, shuffle=False, num_workers=0)

        from main import logger
        logger.info(f"[cached] Train samples: {len(self.train_loader.dataset):,} | "
                    f"Source test: {len(self.source_test_loader.dataset):,} | "
                    f"Target test: {len(self.target_test_loader.dataset):,}")

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
