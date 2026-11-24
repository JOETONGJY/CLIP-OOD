"""
CA-Fixed / CA-Shuffled 训练入口 (2026-11-24, 严格 DG final, save_model=False)
用法:
  python ca_fixed_run.py --dataset CUB --seed 1 --ca_prior_file .../ca_prior_cub.pt \
      --run_tag ca_fixed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import random

import numpy as np
import torch

from args import get_args
from main_cached import CachedTrainingSession

if __name__ == "__main__":
    args = get_args()
    args.save_model = False
    args.wandb = False
    args.model_selection = "final"
    args.CBM_type = "clip_cbm"
    # 2026-11-24 支持 --weight_mode fixed 作为 RW-Fixed 对照 (固定先验, 无残差无交互)
    args.weight_mode = getattr(args, "weight_mode", "ca_fixed")
    if args.weight_mode == "ca_fixed" and not getattr(args, "ca_prior_file", ""):
        raise ValueError("ca_fixed 模式必须提供 --ca_prior_file")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    session = CachedTrainingSession(args)
    session.run()
