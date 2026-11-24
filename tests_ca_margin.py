"""
CA-Diagnostic 语义 margin 单元测试 (2026-11-24, 外审发现实现错误后补)
手工构造正交原型与带偏置的风格化文本, 验证:
  m_{i,c} = cos(t_{i,c}, S_c) - max_{k≠c} cos(t_{i,c}, S_k)
逐元素与手算值精确一致 (容差 1e-6)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F

def main():
    C, D = 3, 3
    S = F.normalize(torch.eye(D), dim=-1)                 # 三个正交原型 e0/e1/e2
    T = torch.zeros(2, C, D)
    T[0, 0] = F.normalize(torch.tensor([1.0, 0.4, 0.0]), dim=-1)   # 类0 风格化, 偏向类1
    T[0, 1] = F.normalize(torch.tensor([0.4, 1.0, 0.0]), dim=-1)   # 类1 风格化, 偏向类0
    T[0, 2] = torch.tensor([0.0, 0.0, 1.0])                        # 类2 干净
    T[1, 0] = F.normalize(torch.tensor([1.0, 0.0, 0.3]), dim=-1)   # 类0, 轻微偏向类2
    T[1, 1] = torch.tensor([0.0, 1.0, 0.0])
    T[1, 2] = F.normalize(torch.tensor([0.0, 0.3, 1.0]), dim=-1)

    sims3 = torch.einsum("icd,kd->ick", T, S)
    gt = sims3.diagonal(dim1=1, dim2=2)
    eye = torch.eye(C, dtype=torch.bool).unsqueeze(0)
    runner = sims3.masked_fill(eye, float("-inf")).max(dim=-1).values
    m = gt - runner

    # 手算期望 (精确表达式; 2026-11-24 修正 exp[1,2] 的竞争者含 k=1 分量)
    exp = torch.zeros(2, C)
    exp[0, 0] = 0.6 / (1.16 ** 0.5)                  # 类0 vs 最近竞争者类1
    exp[0, 1] = 0.6 / (1.16 ** 0.5)
    exp[0, 2] = 1.0
    exp[1, 0] = 0.7 / (1.09 ** 0.5)                  # 类0 vs 最近竞争者类2
    exp[1, 1] = 1.0
    exp[1, 2] = 0.7 / (1.09 ** 0.5)                  # 类2 vs 最近竞争者类1
    ok = torch.allclose(m, exp, atol=1e-6)
    print("m =", m.numpy().round(6).tolist())
    print("期望 =", exp.numpy().round(6).tolist())
    print("逐元素一致:", ok)
    # 语义排序检查: 偏置越重 margin 越小
    assert m[0, 0] < m[1, 0] < m[1, 1], "margin 单调性不符合预期"
    assert ok, "语义 margin 实现与手算不一致!"
    print("[PASS] CA-Diagnostic 语义 margin 实现正确")

if __name__ == "__main__":
    main()
