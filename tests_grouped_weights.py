"""
E3 grouped RW-DDO 复制不变性单元测试 (2026-11-22, 外审复查驱动)

目的: 精确界定 "复制不变" 的成立条件 (外审指出原表述过强):
  场景 A: 复制【单元素簇】中的描述符 A -> 簇均值 r 不变, 簇内均分, 簇总话语权不变
          => 加权 DDO 贡献严格不变 (数学预期: 逐位相等)
  场景 B: 复制【多元素簇】中的描述符 A (簇内已有 A,B) -> R_g 从 (rA+rB)/2 变为
          (2rA+rB)/3 => 簇级先验与簇内份额均变 (数学预期: 有偏差)

权重公式与 model/cbm_models.py get_domain_weights 的 grouped 分支同构
(全局 shift 版簇内 softmax + mean(w)=1 校准)。
结果落盘 experiment_results/E3_分组去冗余/invariance_test.json。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import json

import torch

from reliability import compute_descriptor_clusters

OUT = os.path.join("experiment_results", "E3_分组去冗余", "invariance_test.json")


def grouped_weights(scores, diffs, threshold=0.9):
    """与 cbm_models.get_domain_weights grouped 分支同构的纯函数版"""
    assignment, g = compute_descriptor_clusters(diffs, threshold=threshold)
    a = assignment.to(scores.device)
    counts = torch.bincount(a, minlength=g)
    cp = torch.zeros(g, device=scores.device)
    cp.scatter_add_(0, a, scores)
    cp = cp / counts.clamp_min(1).to(scores.device).float()
    Q = torch.softmax(cp, 0) * g
    e = torch.exp(scores - scores.max())
    den = torch.zeros(g, device=scores.device)
    den.scatter_add_(0, a, e)
    q = e / den[a]
    w = Q[a] * q
    return w * (len(scores) / w.sum()), assignment, g


def ddo_contrib(w, ell):
    """加权 DDO 损失 (ell_i 为逐描述符正交损失占位)"""
    return float((w * ell).sum())


def make_setup(seed=0):
    """构造受控描述符集: 2 个双成员近重复簇 + 6 个单元素簇 (方向互相远离)"""
    g = torch.Generator().manual_seed(seed)
    dirs = []
    base = torch.nn.functional.normalize(torch.randn(10, 64, generator=g), dim=1)
    dirs.append(base[0])
    dirs.append(base[0] + 0.05 * torch.randn(64, generator=g))   # 与 0 近重复 (簇)
    dirs.append(base[1])
    dirs.append(base[1] + 0.05 * torch.randn(64, generator=g))   # 与 2 近重复 (簇)
    for k in range(2, 8):
        dirs.append(base[k])
    diffs = torch.stack(dirs).unsqueeze(1).expand(-1, 3, -1).contiguous()  # (N, C, D)
    scores = torch.rand(10, generator=g) * 0.3 + 0.6
    ell = torch.rand(10, generator=g) + 0.5     # 占位逐描述符正交损失
    return diffs, scores, ell


def clone_desc(diffs, scores, ell, idx):
    """在末尾追加 diffs[idx] 的精确复制 (cos=1.0, 与 E3 注入协议一致)"""
    return (torch.cat([diffs, diffs[idx:idx + 1]]),
            torch.cat([scores, scores[idx:idx + 1]]),
            torch.cat([ell, ell[idx:idx + 1]]))


def main():
    results = {}
    for scene, idx in [("A_单元素簇复制", 4), ("B_多元素簇内复制", 0)]:
        diffs, scores, ell = make_setup()
        w0, a0, g0 = grouped_weights(scores, diffs)
        c0 = torch.bincount(a0, minlength=g0)
        base_w = w0[idx].item()
        base_contrib = ddo_contrib(w0, ell)

        diffs2, scores2, ell2 = clone_desc(diffs, scores, ell, idx)
        w1, a1, g1 = grouped_weights(scores2, diffs2)
        c1 = torch.bincount(a1, minlength=g1)
        # 复制后: 原成员新权重 / 新成员权重 / 簇总话语权 (原簇成员权重之和)
        new_w_orig = w1[idx].item()
        new_w_clone = w1[-1].item()
        grp_sum0 = float(w0[a0 == a0[idx]].sum())
        grp_sum1 = float(w1[a1 == a1[idx]].sum())
        contrib1 = ddo_contrib(w1[:len(ell)], ell)   # 原 10 个方向的加权贡献

        strict = torch.equal(w0, w1[:10])
        results[scene] = {
            "copied_cluster_size_before": int(c0[a0[idx]]),
            "clusters_G": [int(g0), int(g1)],
            "w_copied_before": round(base_w, 6),
            "w_orig_after": round(new_w_orig, 6),
            "w_clone_after": round(new_w_clone, 6),
            "cluster_weight_share_before": round(grp_sum0, 6),
            "cluster_weight_share_after": round(grp_sum1, 6),
            "weighted_ddo_contrib_before": round(base_contrib, 6),
            "weighted_ddo_contrib_after_orig10": round(contrib1, 6),
            "max_weight_change": round(float((w1[:10] - w0).abs().max()), 6),
            "strict_invariance": bool(strict),
        }
        print(f"[{scene}] 复制前簇规模={c0[a0[idx]]} G:{g0}->{g1}")
        print(f"  原 w={base_w:.4f} -> 原 {new_w_orig:.4f} / 复制体 {new_w_clone:.4f}")
        print(f"  簇份额 {grp_sum0:.4f} -> {grp_sum1:.4f}   DDO贡献 {base_contrib:.4f} -> {contrib1:.4f}")
        print(f"  max|Δw|={results[scene]['max_weight_change']:.6f}   严格不变={strict}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
