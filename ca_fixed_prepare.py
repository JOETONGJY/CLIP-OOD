"""
CA-Fixed 先验生成 (2026-11-24): 从 CA-Diagnostic 的交互矩阵生成训练用先验文件。
含尺度匹配前的原始交互矩阵 (β 与尺度匹配在模型内做) 与 CA-Shuffled 对照版
(类别对应关系随机置换, 验证有效的是交互信息而非随机权重扰动)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

for ds in ["cub", "awa2"]:
    src = os.path.join("experiment_results", "CA-Diagnostic", f"ca_matrices_{ds}.npz")
    if not os.path.exists(src):
        print(f"[skip] {src} 不存在 (先跑 ca_diagnostic.py --dataset {ds.upper()})")
        continue
    m = np.load(src)
    I = torch.from_numpy(m["I"]).float()
    torch.save({"interaction": I, "source": f"CA-Diagnostic/{ds}"},
               os.path.join("experiment_results", "CA-Diagnostic", f"ca_prior_{ds}.pt"))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(I.size(1), generator=g)      # 类别对应关系置换 (列置换)
    torch.save({"interaction": I[:, perm].contiguous(), "source": f"CA-Diagnostic/{ds} shuffled(seed0)"},
               os.path.join("experiment_results", "CA-Diagnostic", f"ca_prior_{ds}_shuffled.pt"))
    print(f"[prepare] {ds}: interaction {tuple(I.shape)} -> ca_prior_{ds}.pt / ca_prior_{ds}_shuffled.pt")
