"""
修复 CUB_ddo_cal16 受损记录 (2026-11-21): 同口径重跑 seed 1/2/3。

背景: 原始 3-seed 集合受损——seed1 丢失, seed2 被 11-07 的一次 sel 协议运行误存覆盖
(文件内 weight_mode=ddo_cal16_sel), 仅 seed3 幸存 (61.11, 10-27 原件已备份至
logs/results/_backup_20261121_CUB_ddo_cal16/)。RESULTS.md §2e 记录 61.56±0.35。

本脚本以与原始记录完全相同的协议 (旧口径 target 选模、缓存管线、batch 256、
100 epochs、ch4_mode=ddo、cal_rank=16、lambda_cal=0.01) 重跑, restore
logs/results/CUB_ddo_cal16_seed{1,2,3}.json。save_model=False, 不触碰任何检查点。
复现性检验: 新 seed3 应与备份 61.11 一致 (同 seed 同管线应逐位复现)。
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
    args.save_model = False   # 不覆盖任何 best_*.pth
    args.wandb = False
    args.model_selection = "target"   # 与 2026-10-27 原始记录同口径 (无 _sel 后缀)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    session = CachedTrainingSession(args)
    session.run()
