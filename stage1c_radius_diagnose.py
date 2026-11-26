"""Stage 1C 半径诊断 (LD-DRO): 扰动半径 ε 的量纲与范围。预注册见 RESULTS.md §3x —— 规则与阈值写在那里, 本脚本只执行。

只用源域数据与文本, 不使用任何目标域样本或特征 (会话以 --target_eval none 构建, 目标域特征缓存全程不读取;
目标域数据集对象只为保持类别 / 概念表一致而构建, 不参与任何计算)。
  收敛分类头: logs/heads/CUB_s0_*.pt (27 个, 完整训练集上训练) —— 只做事后量级诊断
  初始化分类头: 与训练相同的随机种子下、尚未训练的分类头 (3 个) —— 用于初始强度配平
  评估样本: cache/CUB_train_aug.pt 的 heldout_val (1200 个干净特征); 梯度用的固定批次: train_core 的前 256 个
  PGD 下界: 只在 heldout_val 的前 300 个样本上做 (50 步 × 零起点 + 闭式起点 + 5 次随机重启, 固定种子)

要回答的问题:
  1. 量纲: 语言域椭球 {L q : ‖q‖ ≤ ε} 里, 多大的 ε 对应"实际的域方向"与"实际的光度扰动"? (ε_dom / ε_vis)
  2. 配平: 五个对照椭球在相同 ε 下惩罚差多少? 怎样把它们配到同一个初始强度? (同半径 / 同初始惩罚)
  3. 上界有多松: LD 损失 (闭式上界) 与多起点 PGD (下界) 的差距随 ε 怎样变化? PGD 收敛了吗?
  4. ISDA 的 λ 与 LD 的 ε 不是同一个量纲, 怎样标定?
  5. LD 损失对所有类别的公共 logit 平移是否严格不变? (原始 DDO 不具备这个性质)

椭球 (迹归一化为 1, 限制在零均值超平面 H 内; language_ellipsoid.build_ellipsoids):
  lang_full  Language-Full    G 的全部 K−1 个非零特征对, 先截断再迹归一化
  lang32     Language-32      截断到 r = 32
  rand_full  Random-Full      与 lang_full 同谱 (Q Λ Qᵀ), 特征向量换成 H 内的随机正交基
  rand32     Random-32        与 lang32 同谱
  identity   Identity-H       H / (K−1)
随机椭球的种子与分类头的训练种子配对 (1 / 2 / 3), 不只依赖一个偶然的随机矩阵。
每个分类头的有效权重 (A, c) 在加载时单独存一份, 并核对它能复现该头自己的 logits (2026-11-26 的首次正式运行
共用了会话里的分类头, 收敛头部分的数字无效, 已重算)。

用法: python stage1c_radius_diagnose.py [--selftest] [--smoke]
  --conv_extra "1/128,1/64": 事后补充的 PGD 收敛检查 (不在预注册范围内), 结果写到单独的文件
  --smoke: 缩小规模的真实数据冒烟 (1 个 seed、3 个 ε、不做收敛检查、不写产物), 只用来在正式运行前检查流程能否走通
产物: experiment_results/Stage1C-Radius/stage1c_radius_cub.{json,md} (md 由同一份结果生成)
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

from language_ellipsoid import build_ellipsoids, domain_directions, ln0, second_moments
from ld_dro import effective_response, isda_loss, ld_dro_loss, pair_distances, pgd_worst_loss
from run_status import git_info

OUT_DIR = os.path.join("experiment_results", "Stage1C-Radius")
SEEDS = (1, 2, 3)
ARMS = ("ce_a0", "full_a1", "full_a2", "mean_a1", "mean_a2", "sym_a1", "sym_a2", "bf_a1", "bf_a2")
MAIN_ARMS = ("ce_a0", "full_a1", "full_a2")         # 做 PGD 与逐项报告的收敛臂; 其余 6 个臂只算闭式量
ELL_ORDER = ("lang_full", "rand_full", "lang32", "rand32", "identity")
ELL_LABEL = {"lang_full": "Language-Full", "rand_full": "Random-Full", "lang32": "Language-32",
             "rand32": "Random-32", "identity": "Identity-H"}
SPECTRUM_KEYS = ("energy_captured_by_r", "effective_rank_entropy_trunc", "lambda_max_over_lambda_min_trunc",
                 "cos_first_axis_language_vs_random")
AUG_CACHE = "cache/CUB_train_aug.pt"
# ---- 预注册的设置 (RESULTS §3x); 不改 ----
GRID_MULT = (1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)   # ε 网格 = ε_dom × 这些倍数
N_PGD = 300                                            # PGD 下界只在 heldout_val 的前 300 个样本上算
PGD_STEPS, PGD_RESTARTS, PGD_SEED = 50, 5, 20261126     # 50 步 × (零起点 + 闭式起点 + 5 次随机重启)
CONV_HEADS = (("init", 1), ("ce_a0", 1), ("full_a1", 1))  # 收敛检查用的 3 个分类头
CONV_STEPS, CONV_RESTARTS, N_CONV = 200, 20, 100         # 更强的 PGD, 在前 100 个样本上
CONV_MULT = (1 / 16, 1 / 4, 1.0)                         # 收敛检查的 ε (× ε_dom)
CONV_MEAN_TOL, CONV_MAX_TOL = 1e-3, 1e-2                 # 收敛判据: 平均增量 < 1e-3 且最大增量 < 1e-2
BISECT_LO, BISECT_HI, BISECT_IT = 1e-2, 1e4, 60


# ---------------- 纯函数 (自测覆盖) ----------------
def _q(v, p):
    return float(torch.quantile(v.detach().double().reshape(-1), p))


def whiten_radius_sq(L, V):
    """行向量 v 在椭球 {L q} 内的白化半径平方 ‖L⁺ v‖², 以及 v 落在 L 值域之外的能量占比。
    用生成元 L 的 SVD (L = U diag(S) Vh), 只取显著奇异值 —— 不对协方差求逆。L: (K, r); V: (n, K)。"""
    U, S, _ = torch.linalg.svd(L.double(), full_matrices=False)
    keep = S > S.max() * 1e-10
    coef = V.double() @ U[:, keep]                             # (n, rank): 在列空间正交基上的坐标
    sq = ((coef / S[keep]) ** 2).sum(-1)
    tot = (V.double() ** 2).sum(-1).clamp_min(1e-300)
    resid = (tot - (coef ** 2).sum(-1)).clamp_min(0) / tot
    return sq, resid


def pair_radius(M, y):
    """r_yk = ‖M_k − M_{y_b}‖ (B, C), 真实类位置为 0 —— 与 ld_dro.pair_distances 同一定义的向量化写法"""
    r = torch.cdist(M, M).pow(2).clamp_min(0)[y].sqrt()
    r[torch.arange(y.numel(), device=y.device), y] = 0.0
    return r


def clean_margins(logits, y):
    """m_y = ℓ_y − max_{k≠y} ℓ_k (正确类相对最坏竞争类的干净间隔), 以及是否分类正确"""
    rows = torch.arange(y.numel(), device=y.device)
    comp = logits.clone()
    comp[rows, y] = float("-inf")
    m_y = logits[rows, y] - comp.max(1).values
    return m_y, m_y > 0


def penalized_metrics(logits, y, M, strength, kind="ld"):
    """闭式量与损失。LD: 竞争类 k 的 logit 加 ε·r_yk —— "椭球内存在能改变预测的扰动" ⟺
    max_k (ℓ_k − ℓ_y + ε r_yk) > 0, 对每个竞争类单独取最坏是精确的; LD 交叉熵是共享扰动下最坏交叉熵的上界。
    ISDA: 竞争类的 logit 加 (λ/2)·r_yk² (高斯扰动的 Jensen 上界), 这里的"可翻转"只是对该上界而言。
    惩罚的两套统计 (外审第十轮的命名更正, RESULTS §3y): pen_bind_* 是每个样本在**全部竞争类上的最大惩罚**
    的均值/分位 (平均最大类别对惩罚; 历史键名保留, 它当初被叫作"绑定惩罚", 名不副实 —— 它不一定是决定
    鲁棒间隔的那个竞争类); pen_worstcomp_* 是**实际最坏竞争类** (鲁棒间隔 m_yk − pen 最小的那个) 上的惩罚。"""
    rows = torch.arange(y.numel(), device=y.device)
    r = pair_radius(M, y)
    m_y, correct = clean_margins(logits, y)
    r_max = r.max(1).values
    m_yk = logits[rows, y][:, None] - logits                   # (B, C): 每个竞争类各自的干净间隔 m_yk
    pen = (strength * r) if kind == "ld" else (0.5 * strength * r * r)
    rob = m_yk - pen                                            # m_yk − 惩罚 (逐竞争类)
    rob[rows, y] = float("inf")
    m_rob = rob.min(1).values
    wc = rob.argmin(1)                                          # 实际最坏竞争类 = 鲁棒间隔最小的竞争类
    pen_wc = pen[rows, wc]
    off = r > 0
    loss = ld_dro_loss(logits, M, y, strength) if kind == "ld" else isda_loss(logits, M, y, strength)
    return {"strength": float(strength), "kind": kind, "acc": float(correct.double().mean()),
            "ce": float(F.cross_entropy(logits, y)), "loss": float(loss),
            "pen_bind_mean": float(pen.max(1).values.mean()), "pen_bind_q50": _q(pen.max(1).values, 0.5),
            "pen_bind_q90": _q(pen.max(1).values, 0.9), "pen_all_q50": _q(pen[off], 0.5), "pen_all_q90": _q(pen[off], 0.9),
            "pen_worstcomp_mean": float(pen_wc.mean()), "pen_worstcomp_q50": _q(pen_wc, 0.5), "pen_worstcomp_q90": _q(pen_wc, 0.9),
            "clean_worst_margin_mean": float(m_y.mean()), "clean_worst_margin_q10": _q(m_y, 0.1),
            "robust_worst_margin_mean": float(m_rob.mean()),
            "flip_rate_all": float((m_rob < 0).double().mean()),
            "flip_rate_correct": float(((m_rob < 0) & correct).sum() / correct.sum().clamp_min(1)),
            "r_max_mean": float(r_max.mean()), "r_max_sq_mean": float((r_max ** 2).mean()),
            "r_all_q50": _q(r[off], 0.5), "r_all_q90": _q(r[off], 0.9)}


def reg_ratio(logits, params, A, L, y, strength, kind, ddo_fn=None):
    """固定批次上: 正则部分对分类头参数的梯度范数与 CE 梯度范数之比, 以及两个梯度的夹角余弦。
    LD / ISDA 的正则部分 = 损失 − CE (ε = 0 时为 0); 等权 DDO 的正则部分 = 它本身。"""
    ce = F.cross_entropy(logits, y)
    g_ce = torch.autograd.grad(ce, params, retain_graph=True)
    g_ce = torch.cat([g.reshape(-1) for g in g_ce])
    if kind == "ddo":
        raw = torch.autograd.grad(ddo_fn(), params, retain_graph=True, allow_unused=True)
    else:
        loss = ld_dro_loss(logits, A @ L, y, strength) if kind == "ld" else isda_loss(logits, A @ L, y, strength)
        raw = torch.autograd.grad(loss - ce, params, retain_graph=True)
    g = torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(raw, params)])
    return (float(g.norm() / g_ce.norm().clamp_min(1e-12)),
            float((g @ g_ce) / (g.norm() * g_ce.norm()).clamp_min(1e-12)))


def bisect(fn, target, lo=BISECT_LO, hi=BISECT_HI, it=BISECT_IT):
    """在 [lo, hi] 上对 log(参数) 二分, 求 fn(x) = target (要求 fn 随 x 单调不减)。返回 (x, 两端是否夹住目标)。"""
    flo, fhi = fn(lo) - target, fn(hi) - target
    if not (flo <= 0 <= fhi):
        return (lo if flo > 0 else hi), False
    for _ in range(it):
        mid = float(np.sqrt(lo * hi))
        if fn(mid) - target < 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi)), True


def invariance_check(logits, u, A, L, y, eps, a, b0):
    """公共 logit 平移: 所有类的有效权重同加向量 a、偏置同加常数 b0。返回 (LD 损失之差, DDO 损失之差)。
    LD: 竞争类之差 (A_k − A_y) 不变 => r_yk 不变 => 损失严格不变。原始 DDO 的 |A·u| 一般会变。"""
    shift = u @ a + b0                                          # 每个样本一个公共常数, 所有类相同
    ld_gap = float((ld_dro_loss(logits, A @ L, y, eps) - ld_dro_loss(logits + shift[:, None], (A + a) @ L, y, eps)).abs())
    ddo, ddo2 = (A @ u.T).abs().mean(), ((A + a) @ u.T).abs().mean()
    return ld_gap, float((ddo2 - ddo).abs())


def pgd_worst_per_sample(logits, M, y, eps, steps, restarts, seed):
    """多起点 PGD 的逐样本最坏 CE (下界)。eps = 0 时等于 CE。"""
    if eps == 0:
        return F.cross_entropy(logits, y, reduction="none")
    _, q = pgd_worst_loss(logits, M, y, eps, steps=steps, restarts=restarts, seed=seed, closed_form_starts=True)
    return F.cross_entropy(logits + torch.einsum("cr,br->bc", M, q), y, reduction="none")


# ---------------- 自测 (合成数据, 不需要任何数据文件) ----------------
def self_test():
    torch.manual_seed(0)
    np.random.seed(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    K, C, B = 40, 12, 300
    U = ln0(torch.randn(600, K, dtype=torch.float64))          # 零均值、单位方差: 与真实域方向同一构造
    G = U.T @ U / U.size(0)
    full = build_ellipsoids(G, r=K - 1, seed=3, with_flat=False)
    r6 = build_ellipsoids(G, r=6, seed=3, with_flat=False)
    sq, resid = whiten_radius_sq(full["language"], U)
    check("W1 Language-Full: 域方向白化半径平方的均值 = tr(G)·(K−1), 值域外能量为 0",
          abs(float(sq.mean()) - float(G.trace()) * (K - 1)) < 1e-6 and float(resid.max()) < 1e-10,
          f"{float(sq.mean()):.4f} vs {float(G.trace()) * (K - 1):.4f}")
    sq_id, _ = whiten_radius_sq(full["identity"], U)
    check("W2 Identity-H: ‖L⁺u‖² = (K−1)·‖u‖²", bool(torch.allclose(sq_id, (K - 1) * (U ** 2).sum(-1), rtol=1e-8)))
    sq6, res6 = whiten_radius_sq(r6["language"], U)
    check("W3 截断椭球: 值域外能量占比的均值 = 1 − 截断捕获的能量占比",
          abs(float(res6.mean()) - (1 - r6["spectrum"]["energy_captured_by_r"])) < 1e-6, f"{float(res6.mean()):.4f}")
    Lp = torch.linalg.pinv(full["language"].double())             # (r, K): ‖L⁺u‖² = ‖(L⁺)ᵀ u‖²
    check("W4 白化半径与显式伪逆一致 (满秩良态情形)", bool(torch.allclose(sq, ((U @ Lp.T) ** 2).sum(-1), rtol=1e-6)))

    A = torch.randn(C, K) * 0.5
    logits = ln0(torch.randn(B, K)) @ A.T + torch.randn(C)
    y = torch.randint(0, C, (B,))
    M = A @ full["language"].float()
    r_ref, _ = pair_distances(M, y)
    check("W5 向量化的 r_yk 与 ld_dro.pair_distances 一致", torch.allclose(pair_radius(M, y), r_ref, atol=1e-5))
    for eps in (0.0, 0.5, 3.0):
        m = penalized_metrics(logits, y, M, eps)
        lk = logits + eps * pair_radius(M, y)
        lk[torch.arange(B), y] = float("-inf")
        exists = lk.max(1).values > logits[torch.arange(B), y]   # max_k (ℓ_k − ℓ_y + ε r_yk) > 0
        check(f"W6 ε={eps}: 翻转率与 max_k(ℓ_k−ℓ_y+ε r_yk)>0 等价; ε=0 时损失 = CE 且正确样本的翻转率为 0",
              abs(m["flip_rate_all"] - float(exists.double().mean())) < 1e-12
              and (eps > 0 or (abs(m["loss"] - m["ce"]) < 1e-6 and m["flip_rate_correct"] == 0.0)))
    m6 = penalized_metrics(logits, y, M, 2.0)                    # 命名更正 (外审第十轮): pen_bind 是最大类别对惩罚
    pen2 = 2.0 * pair_radius(M, y)
    rob2 = (logits[torch.arange(B), y][:, None] - logits) - pen2
    rob2[torch.arange(B), y] = float("inf")
    wc2 = rob2.argmin(1)
    manual_wc = float(pen2[torch.arange(B), wc2].mean())
    check("W6b 最坏竞争类惩罚与逐样本显式计算一致, 且 ≤ 最大类别对惩罚 (两套统计不可混用)",
          abs(m6["pen_worstcomp_mean"] - manual_wc) < 1e-9 and m6["pen_worstcomp_mean"] <= m6["pen_bind_mean"] + 1e-12,
          f"worstcomp {m6['pen_worstcomp_mean']:.4f} vs bind {m6['pen_bind_mean']:.4f}")
    _, q1 = pgd_worst_loss(logits, M, y, 2.0, steps=200, restarts=20, seed=1, closed_form_starts=True)
    lq = logits + torch.einsum("cr,br->bc", M, q1)                # PGD 找到的最坏扰动下的 logits
    lq_k = lq.clone()
    lq_k[torch.arange(B), y] = float("-inf")
    worst_num = lq_k.max(1).values - lq[torch.arange(B), y]       # 数值得到的 max_k ℓ_k(q) − ℓ_y(q)
    lk = logits + 2.0 * pair_radius(M, y)                          # 闭式: 对每个竞争类单独取最坏
    lk[torch.arange(B), y] = float("-inf")
    worst_closed = lk.max(1).values - logits[torch.arange(B), y]
    check("W7 逐竞争类闭式最坏值 ≥ 数值 PGD 的最坏值 (PGD 是最大化, 不保证达到全局最优)",
          bool((worst_closed >= worst_num - 1e-4).all()), f"闭式 − 数值 的中位差 {float((worst_closed - worst_num).median()):.4f}")

    head = torch.nn.Linear(K, C)
    params = list(head.parameters())
    xs, ys = ln0(torch.randn(64, K)), torch.randint(0, C, (64,))

    def ratio_of(strength, kind, L, ddo_fn=None):
        return reg_ratio(head(xs), params, head.weight, L, ys, strength, kind, ddo_fn=ddo_fn)[0]

    Lf = full["language"].float()
    grid = [0.0, 0.25, 1.0, 4.0, 16.0]
    rg = [ratio_of(e, "ld", Lf) for e in grid]
    check("W8 LD 正则部分的梯度比随 ε 单调不减; ε=0 时 LD 退化为 CE, 梯度比为 0", all(b >= a - 1e-9 for a, b in zip(rg, rg[1:])) and rg[0] == 0.0,
          " / ".join("%.4f" % v for v in rg))
    cos = reg_ratio(head(xs), params, head.weight, Lf, ys, 1.0, "ld")[1]
    check("W8b 正则梯度与 CE 梯度的夹角余弦 ∈ [−1, 1]", -1.0 <= cos <= 1.0, f"cos = {cos:.4f}")
    x2, sol = bisect(lambda e: ratio_of(e, "ld", Lf), 0.5)
    check("W9 二分求出的 ε 使梯度比达到目标 (误差 < 1%), 且两端夹住",
          sol and abs(ratio_of(x2, "ld", Lf) - 0.5) / 0.5 < 0.01, f"ε = {x2:.4f}, 比值 {ratio_of(x2, 'ld', Lf):.4f}")
    check("W10 不夹住时如实返回 False 并取端点", bisect(lambda e: 2.0 * e, 5e5)[1] is False
          and bisect(lambda e: 2.0 * e + 1e4, 5e3)[1] is False)
    check("W11 ISDA: 同一数值的 λ 与 ε 下梯度比不同 (不能机械地取同值)",
          abs(ratio_of(1.0, "isda", Lf) - ratio_of(1.0, "ld", Lf)) > 1e-6)
    check("W12 等权 DDO 的梯度比可计算且为正", ratio_of(0.0, "ddo", Lf, ddo_fn=lambda: (head.weight @ Lf).abs().mean()) > 0)

    g = torch.Generator().manual_seed(7)
    a, b0 = torch.randn(K, generator=g), 0.7
    gap, dgap = invariance_check(logits[:60].double(), U[:60], A.double(), full["language"], y[:60], 2.0, a.double(), b0)
    check("W13 公共 logit 平移: LD 损失严格不变 (< 1e-10), 原始 DDO 的损失发生变化", gap < 1e-10 and dgap > 1e-9,
          f"LD 差 {gap:.1e}, DDO 差 {dgap:.2e}")
    b1 = pgd_worst_per_sample(logits[:30], M, y[:30], 2.0, steps=5, restarts=2, seed=0)
    b2 = pgd_worst_per_sample(logits[:30], M, y[:30], 2.0, steps=100, restarts=20, seed=0)
    ce0 = F.cross_entropy(logits[:30], y[:30], reduction="none")
    check("W14 链 CE(0) ≤ 评估用 PGD ≤ 强 PGD (逐样本), 强 PGD 的增量非负",
          bool((b2 >= b1 - 1e-5).all() and (b1 >= ce0 - 1e-5).all()), f"平均增量 {float((b2 - b1).mean()):.4f}")
    check("W15 ε=0 时 PGD 退化为 CE", torch.equal(pgd_worst_per_sample(logits[:30], M, y[:30], 0.0, 5, 2, 0), ce0))

    # ---- 共享会话: 每个头存下的 (A, c) 必须是它自己的 (2026-11-26 的首次正式运行在这里出过错) ----
    import types
    sess = types.SimpleNamespace(model=torch.nn.Module())
    sess.model.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LayerNorm(K), torch.nn.Linear(K, C))
    stored = []
    for _ in range(3):                                            # 同一个会话先后加载 3 个不同的头
        with torch.no_grad():
            for p_ in sess.model.classifier.parameters():
                p_.copy_(torch.randn_like(p_))
        stored.append(head_state(sess))
    own = all(torch.equal(stored[i]["A"], stored[i]["A"].clone()) and all(
        not torch.equal(stored[i]["A"], stored[j]["A"]) for j in range(3) if j != i) for i in range(3))
    x_ = ln0(torch.randn(20, K))
    repro = all(float((x_ @ H["A"].T + H["c"] - H["sess"].model.classifier(x_)).abs().max()) > 0.0 for H in stored[:2])  # 会话里已是最后一个头
    rec2 = x_ @ stored[2]["A"].T + stored[2]["c"]
    last_ok = float((rec2 - sess.model.classifier(x_)).abs().max()) < 1e-5 * (1 + float(rec2.abs().max()))   # 相对容差
    check("W16 同一会话先后加载多个头: 每个头存下的 (A, c) 互不相同, 且最后一个能复现会话当前的输出 (之前的头不再能)",
          own and repro and last_ok)

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------- 真实数据 ----------------
class _NoTargetSession(__import__("main_cached").CachedTrainingSession):
    """只重建 loader 与模型; 以 --target_eval none 构建, 目标域特征缓存全程不读取。"""
    def _init_components(self):
        self._prepare_datasets()
        self._init_model()
        self._init_loss_fns()

    def _prepare_datasets(self):
        self.args.target_eval = "none"
        super()._prepare_datasets()


def model_key(cfg):
    skip = {"run_tag", "seed", "alpha", "epochs", "lr", "weight_decay", "target_eval", "save_head", "grad_diag",
            "save_model", "overwrite_run", "train_subset_cache", "source_val_cache", "device", "wandb"}
    return json.dumps(sorted((k, str(v)) for k, v in cfg.items() if k not in skip), ensure_ascii=False)


def build_session(cfg, seed):
    """按训练入口相同的顺序播种后构建会话 (分类头处于初始化状态, 尚未训练)。"""
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    c = dict(cfg)
    c.update(target_eval="none", save_model=False, save_head=False, grad_diag=False, wandb=False, overwrite_run=False, epochs=1)
    return _NoTargetSession(argparse.Namespace(**c))


def head_state(sess):
    """分类头的有效权重 A = W⊙γ 与偏置 c = W·β + b, 在加载之后立刻单独存一份 (clone)。
    同一结构的多个分类头共用一个会话对象, 会话里的分类头会被下一个头覆盖 —— 之后再用会话去取就拿到了别人的。
    注意: --ddo_dirs full_bf / split_bf 的"无偏置"只作用于 DDO 正则项的损失路径, 分类头的前向始终带偏置。"""
    A, c = effective_response(sess.model.classifier)
    return {"sess": sess, "A": A.detach().clone(), "c": c.detach().clone()}


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _mean(vals):
    v = [x for x in vals if x is not None and x == x]
    return float(np.mean(v)) if v else float("nan")


def _f(x, nd=3):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def prepare(device, check):
    """准备工作 (步骤 1–4): 初始化会话 → 概念矩阵 E / 域方向 / G → 五个椭球 → 量纲锚点 → 分类头与 heldout_val 上的 logits。
    正式诊断与事后补充检查共用这一段, 保证两者用的是同一批对象。"""
    # ===== 1. 初始化会话: 概念矩阵 E、域方向、固定批次 =====
    init_cfg, init_ref = {}, {}
    for s in SEEDS:
        with open(f"logs/results/CUB_p1_dose_a1_seed{s}.json", encoding="utf-8") as f:
            r = json.load(f)
        init_cfg[s], init_ref[s] = r["run_config"], r["grad_diag"][0]
    init_sess = {s: build_session(init_cfg[s], s) for s in SEEDS}
    ce_detail, init_ok = [], True
    for s in SEEDS:
        x, y = init_sess[s]._diag_batch()
        with torch.no_grad():
            ce0 = float(F.cross_entropy(init_sess[s].model(x)[1], y))
        init_ok &= init_ref[s]["tag"] == "before_epoch1" and abs(ce0 - init_ref[s]["ce"]) < 1e-5
        ce_detail.append(f"s{s}: {ce0:.6f} 对记录 {init_ref[s]['ce']:.6f}")
    check("三个初始化分类头与训练时的初始状态一致 (固定批次上的 CE = grad_diag 的 before_epoch1 记录)", init_ok, "; ".join(ce_detail))

    s0 = init_sess[1]
    E = s0.model.concept_embeddings.detach().double().cpu()          # (K, 768) 已归一化的概念矩阵
    U = domain_directions(s0.train_dataset.domain_diffs.double().cpu(), E)      # (N·C, K)
    G = second_moments(U)[0]
    K = E.size(0)

    # ===== 2. 椭球 (随机椭球的种子与分类头的种子配对) =====
    ELL, ell_meta, rand_seed_used = {}, {}, {}
    for s in SEEDS:
        ELL[s], ell_meta[s] = {}, {}
        for name, r, key, k_lang, k_rand in (("full", K - 1, "full", "lang_full", "rand_full"),
                                             ("32", 32, "r32", "lang32", "rand32")):
            used, built = s, None
            for cand in (s, s + 10, s + 20, s + 30):                # 构造时的自检不过 (首轴夹角) 就换种子, 并如实记录
                try:
                    built = build_ellipsoids(G, r=r, seed=cand, with_flat=False)
                except AssertionError:
                    continue
                used = cand
                break
            assert built is not None, f"r={r}: 找不到可用的随机椭球种子"
            rand_seed_used[(s, name)] = used
            ELL[s].update({k_lang: built["language"], k_rand: built["random"],
                           **({"identity": built["identity"]} if key == "full" else {})})
            ell_meta[s][key] = {k: built["spectrum"][k] for k in SPECTRUM_KEYS}
            ell_meta[s][key]["sha256"] = dict(built["sha256"])
    check("Language-Full / Language-32 / Identity 不随种子变化; Random-Full / Random-32 各个种子互不相同",
          all(torch.equal(ELL[1][e], ELL[s][e]) for s in SEEDS for e in ("lang_full", "lang32", "identity"))
          and all(sorted(ELL[s]) == sorted(ELL_ORDER) for s in SEEDS)
          and len({ell_meta[s]["full"]["sha256"]["random"] for s in SEEDS}) == len(SEEDS)
          and len({ell_meta[s]["r32"]["sha256"]["random"] for s in SEEDS}) == len(SEEDS))

    # ===== 3. 量纲锚点: 实际域方向与实际光度扰动在各椭球内的白化半径 =====
    dom = {e: {"eps_dom": _mean([float(torch.sqrt(whiten_radius_sq(ELL[s][e], U)[0].mean())) for s in SEEDS]),
               "resid_mean": _mean([float(whiten_radius_sq(ELL[s][e], U)[1].mean()) for s in SEEDS])} for e in ELL_ORDER}
    eps_dom = dom["lang_full"]["eps_dom"]
    aug = torch.load(AUG_CACHE, map_location="cpu", weights_only=False)
    val_idx = torch.as_tensor(aug["split"]["heldout_val"], dtype=torch.long)
    aug_names = list(aug["aug_feats"].keys())
    u_clean = ln0(aug["clean_feats"][val_idx].double() @ E.T)
    DU = {n: ln0(aug["aug_feats"][n][val_idx].double() @ E.T) - u_clean for n in aug_names}
    vis = {e: {"eps_vis_median": _mean([float(torch.sqrt(torch.cat([whiten_radius_sq(ELL[s][e], DU[n])[0] for n in aug_names]).median())) for s in SEEDS]),
               "eps_vis_q25": _mean([float(torch.sqrt(torch.cat([whiten_radius_sq(ELL[s][e], DU[n])[0] for n in aug_names]).quantile(0.25))) for s in SEEDS]),
               "eps_vis_q75": _mean([float(torch.sqrt(torch.cat([whiten_radius_sq(ELL[s][e], DU[n])[0] for n in aug_names]).quantile(0.75))) for s in SEEDS]),
               "resid_mean": _mean([float(torch.cat([whiten_radius_sq(ELL[s][e], DU[n])[1] for n in aug_names]).mean()) for s in SEEDS])}
          for e in ELL_ORDER}
    du_norm = {n: float(DU[n].norm(dim=-1).mean()) for n in aug_names}

    # ===== 4. 分类头: 初始化 3 个 + 收敛 27 个; 在 heldout_val 干净特征上的 logits =====
    clean_val = aug["clean_feats"][val_idx].float().to(device)
    y_val = aug["labels"][val_idx].to(device)

    def logits_of(sess):
        with torch.no_grad():
            sess.model.eval()
            return sess.model(clean_val)[1].detach()

    heads = {}
    for s in SEEDS:
        sess = init_sess[s]
        heads[("init", s)] = {**head_state(sess), "logits": logits_of(sess), "sha": None, "ddo_dirs": "full"}
    conv_sess, head_sha = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            path = f"logs/heads/CUB_s0_{arm}_seed{s}.pt"
            h = torch.load(path, map_location="cpu", weights_only=False)
            key = model_key(h["run_config"])
            if key not in conv_sess:                               # 只借模型结构 (概念矩阵、方向集), 分类头随后替换
                conv_sess[key] = build_session(h["run_config"], 1)
            sess = conv_sess[key]
            sess.model.classifier.load_state_dict(h["classifier"], strict=True)
            heads[(arm, s)] = {**head_state(sess), "logits": logits_of(sess), "ddo_dirs": h["run_config"].get("ddo_dirs")}
            head_sha[f"{arm}|{s}"] = _sha(path)
    check("全部分类头都已就位 (每个 seed: 1 个初始化 + 9 个收敛)", len(heads) == len(SEEDS) * (1 + len(ARMS)))
    # 一致性核对: 每个头存下的有效权重 (A, c) 必须能复现它自己的 logits —— 同一结构的多个头共用会话时,
    # 这能发现"拿错了头" (2026-11-26 的正式运行就出过这个错: 惩罚用了同一组里最后一个头的 A)
    u_val64 = ln0(clean_val.double().cpu() @ E.T)
    worst_head = 0.0
    for hkey, H in heads.items():
        rec = u_val64 @ H["A"].double().cpu().T + H["c"].double().cpu()
        worst_head = max(worst_head, float((rec - H["logits"].double().cpu()).abs().max()))
    check("每个分类头的有效权重 (A, c) 都能复现它自己的 logits (最大绝对差 < 1e-2)", worst_head < 1e-2, f"{worst_head:.2e}")

    return {"device": device, "init_sess": init_sess, "init_ref": init_ref, "E": E, "U": U, "G": G, "K": K,
            "ELL": ELL, "ell_meta": ell_meta, "rand_seed_used": rand_seed_used, "dom": dom, "eps_dom": eps_dom,
            "vis": vis, "du_norm": du_norm, "aug_names": aug_names, "heads": heads, "y_val": y_val, "head_sha": head_sha,
            "val_idx": val_idx}


def conv_extra(mults):
    """事后补充的 PGD 收敛检查 (不在预注册范围内; RESULTS §3y 里明确标注为事后补充)。
    预注册的收敛检查只覆盖 ε ≥ ε_dom/16, 而配平出来的半径约为 ε_dom/60; 这里对同样的 3 个分类头 × 5 个椭球,
    在给定的更小半径 (× ε_dom) 上重复同一个判据。与正式运行共用 prepare(), 结果写到单独的文件。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    ctx = prepare(device, check)
    heads, ELL, y_val, eps_dom = ctx["heads"], ctx["ELL"], ctx["y_val"], ctx["eps_dom"]

    def M_of(hkey, e):
        return heads[hkey]["A"] @ ELL[hkey[1]][e].float().to(device)

    conv = []
    for hkey in CONV_HEADS:
        for e in ELL_ORDER:
            M = M_of(hkey, e)
            lg, yy = heads[hkey]["logits"][:N_CONV], y_val[:N_CONV]
            for mult in mults:
                eps = eps_dom * mult
                b1 = pgd_worst_per_sample(lg, M, yy, eps, PGD_STEPS, PGD_RESTARTS, PGD_SEED)
                b2 = pgd_worst_per_sample(lg, M, yy, eps, CONV_STEPS, CONV_RESTARTS, PGD_SEED + 1)
                inc = b2 - b1
                conv.append({"head": f"{hkey[0]}|{hkey[1]}", "ell": e, "eps": eps, "mult": mult, "mean_inc": float(inc.mean()),
                             "max_inc": float(inc.max()), "n_gt_1e-3": int((inc > 1e-3).sum()),
                             "converged": bool(inc.mean() < CONV_MEAN_TOL and inc.max() < CONV_MAX_TOL)})
        print(f"  事后收敛检查完成 {hkey}", flush=True)
    n_ok = sum(c["converged"] for c in conv)
    check("事后补充检查: 全部检查点都满足收敛判据", n_ok == len(conv),
          f"{n_ok}/{len(conv)}; 最大平均增量 {_f(max(c['mean_inc'] for c in conv), 6)}, 最大单样本增量 {_f(max(c['max_inc'] for c in conv), 4)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"posthoc": True, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "note": "事后补充的 PGD 收敛检查, 不在预注册 (RESULTS §3x) 范围内; 预注册的检查只覆盖 ε ≥ ε_dom/16",
           "eps_dom": eps_dom, "mults": list(mults), "settings": {"steps": CONV_STEPS, "restarts": CONV_RESTARTS, "n": N_CONV,
                                                                    "eval_steps": PGD_STEPS, "eval_restarts": PGD_RESTARTS,
                                                                    "mean_tol": CONV_MEAN_TOL, "max_tol": CONV_MAX_TOL},
           "convergence": conv, "n_converged": n_ok, "all_converged": n_ok == len(conv)}
    with open(os.path.join(OUT_DIR, "stage1c_radius_cub_conv_extra.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {OUT_DIR}/stage1c_radius_cub_conv_extra.json  ({n_ok}/{len(conv)} 个检查点收敛)")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    if "--conv_extra" in sys.argv[1:]:
        toks = sys.argv[sys.argv.index("--conv_extra") + 1].split(",")
        return conv_extra([int(a) / int(b) for a, b in (t.split("/") for t in toks)])
    smoke = "--smoke" in sys.argv[1:]
    if smoke:
        global SEEDS, ARMS, MAIN_ARMS, GRID_MULT, CONV_HEADS, CONV_MULT
        SEEDS, ARMS, MAIN_ARMS = (1,), ("ce_a0", "full_a1"), ("ce_a0",)
        GRID_MULT, CONV_HEADS, CONV_MULT = (1 / 16, 1 / 4, 1.0), (), ()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    code = git_info()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    ctx = prepare(device, check)                                # 步骤 1–4
    ELL, eps_dom, heads, y_val = ctx["ELL"], ctx["eps_dom"], ctx["heads"], ctx["y_val"]
    dom, vis, du_norm, aug_names = ctx["dom"], ctx["vis"], ctx["du_norm"], ctx["aug_names"]
    init_sess, init_ref, head_sha, val_idx = ctx["init_sess"], ctx["init_ref"], ctx["head_sha"], ctx["val_idx"]
    ell_meta, rand_seed_used, E, U, G, K = ctx["ell_meta"], ctx["rand_seed_used"], ctx["E"], ctx["U"], ctx["G"], ctx["K"]

    # ===== 5. 逐 (分类头, 椭球): 单位半径、ε 网格上的闭式量、PGD 下界 =====
    grid = [eps_dom * m for m in GRID_MULT]
    table = {}

    def A_of(hkey):
        return heads[hkey]["A"]

    def M_of(hkey, e):
        return A_of(hkey) @ ELL[hkey[1]][e].float().to(device)

    for hkey in heads:
        for e in ELL_ORDER:
            M = M_of(hkey, e)
            rec = {"by_eps": [penalized_metrics(heads[hkey]["logits"], y_val, M, eps) for eps in grid]}
            is_pgd = hkey[0] == "init" or hkey[0] in MAIN_ARMS
            if is_pgd:
                lg, yy = heads[hkey]["logits"][:N_PGD], y_val[:N_PGD]
                rec["pgd"] = []
                for eps in grid:
                    b = pgd_worst_per_sample(lg, M, yy, eps, PGD_STEPS, PGD_RESTARTS, PGD_SEED)
                    ce, ld = float(F.cross_entropy(lg, yy)), float(ld_dro_loss(lg, M, yy, eps))
                    rec["pgd"].append({"eps": eps, "ce": ce, "pgd": float(b.mean()), "ld": ld, "gap": ld - float(b.mean()),
                                       "chain_ok": bool(ce <= float(b.mean()) + 1e-5 and float(b.mean()) <= ld + 1e-5)})
            table[f"{hkey[0]}|{hkey[1]}|{e}"] = rec
        print(f"  已完成 {hkey}", flush=True)
    chain_ok = all(p["chain_ok"] for v in table.values() for p in v.get("pgd", []))
    check("链 CE(0) ≤ PGD 下界 ≤ LD 上界 在全部 (分类头, 椭球, ε) 上成立", chain_ok)

    # ===== 6. 初始强度配平 (只用 3 个初始化头与固定批次; 不使用目标域) =====
    # 梯度比 = 正则部分 (LD / ISDA 损失减去 CE; 等权 DDO 本身) 的梯度范数 / CE 的梯度范数 ——
    # 与训练时记录的 grad_norm_ratio_ddo_over_ce 是同一个量
    def _init_batch(s):
        sess = init_sess[s]
        x, y = sess._diag_batch()
        sess.model.eval()
        params = [p for p in sess.model.classifier.parameters() if p.requires_grad]
        _, lg, _ = sess.model(x)
        return sess, lg, params, y

    def ratio_init(kind, e, strength, want_cos=False):
        vals, coss = [], []
        for s in SEEDS:
            sess, lg, params, y = _init_batch(s)
            A = effective_response(sess.model.classifier)[0]
            r_, c_ = reg_ratio(lg, params, A, ELL[s][e].float().to(device), y, strength, kind, ddo_fn=sess._ddo_loss_equal_weight)
            vals.append(r_)
            coss.append(c_)
        return (float(np.mean(vals)), float(np.mean(coss))) if want_cos else float(np.mean(vals))

    rho_ddo, cos_ddo = {}, {}
    for s in SEEDS:                                            # 原始等权 DDO (α = 1) 在初始化时的梯度比, 逐 seed
        sess, lg, params, y = _init_batch(s)
        rho_ddo[str(s)], cos_ddo[str(s)] = reg_ratio(lg, params, None, None, y, 0.0, "ddo", ddo_fn=sess._ddo_loss_equal_weight)
    rho_bar = float(np.mean(list(rho_ddo.values())))
    r_max_mean = {e: _mean([table[f"init|{s}|{e}"]["by_eps"][0]["r_max_mean"] for s in SEEDS]) for e in ELL_ORDER}
    r_max_sq_mean = {e: _mean([table[f"init|{s}|{e}"]["by_eps"][0]["r_max_sq_mean"] for s in SEEDS]) for e in ELL_ORDER}
    cal = {}
    for e in ELL_ORDER:
        curve = [ratio_init("ld", e, eps) for eps in grid]
        eps_grad, br1 = bisect(lambda x: ratio_init("ld", e, x), rho_bar)
        lam_grad, br2 = bisect(lambda x: ratio_init("isda", e, x), rho_bar)
        cal[e] = {"ld_ratio_on_grid": curve, "monotone_on_grid": bool(all(b >= a - 1e-6 for a, b in zip(curve, curve[1:]))),
                  "eps_grad": eps_grad, "eps_grad_bracketed": br1, "lam_grad": lam_grad, "lam_grad_bracketed": br2,
                  "cos_ld_at_eps_grad": ratio_init("ld", e, eps_grad, want_cos=True)[1],
                  "cos_isda_at_lam_grad": ratio_init("isda", e, lam_grad, want_cos=True)[1]}
    p_star = cal["lang_full"]["eps_grad"] * r_max_mean["lang_full"]
    for e in ELL_ORDER:
        cal[e]["eps_pen"] = p_star / r_max_mean[e]
        cal[e]["lam_pen"] = 2 * p_star / r_max_sq_mean[e]
    check("各椭球的初始梯度比在网格上单调、二分时两端都夹住了目标",
          all(c["monotone_on_grid"] and c["eps_grad_bracketed"] and c["lam_grad_bracketed"] for c in cal.values()),
          "; ".join(f"{e}: 单调 {c['monotone_on_grid']}, ε {c['eps_grad_bracketed']}, λ {c['lam_grad_bracketed']}" for e, c in cal.items()))
    # 配平后的半径 / 强度作用在每一个分类头上的闭式量 (同初始惩罚口径)
    for hkey in heads:
        for e in ELL_ORDER:
            M = M_of(hkey, e)
            lg = heads[hkey]["logits"]
            table[f"{hkey[0]}|{hkey[1]}|{e}"]["cal"] = {
                "ld_eps_grad": penalized_metrics(lg, y_val, M, cal[e]["eps_grad"]),
                "ld_eps_pen": penalized_metrics(lg, y_val, M, cal[e]["eps_pen"]),
                "isda_lam_grad": penalized_metrics(lg, y_val, M, cal[e]["lam_grad"], kind="isda"),
                "isda_lam_pen": penalized_metrics(lg, y_val, M, cal[e]["lam_pen"], kind="isda")}

    # ===== 7. PGD 收敛检查 =====
    conv = []
    for hkey in CONV_HEADS:
        for e in ELL_ORDER:
            M = M_of(hkey, e)
            lg, yy = heads[hkey]["logits"][:N_CONV], y_val[:N_CONV]
            for mult in CONV_MULT:
                eps = eps_dom * mult
                b1 = pgd_worst_per_sample(lg, M, yy, eps, PGD_STEPS, PGD_RESTARTS, PGD_SEED)
                b2 = pgd_worst_per_sample(lg, M, yy, eps, CONV_STEPS, CONV_RESTARTS, PGD_SEED + 1)
                inc = b2 - b1
                conv.append({"head": f"{hkey[0]}|{hkey[1]}", "ell": e, "eps": eps, "mean_inc": float(inc.mean()),
                             "max_inc": float(inc.max()), "n_gt_1e-3": int((inc > 1e-3).sum()),
                             "converged": bool(inc.mean() < CONV_MEAN_TOL and inc.max() < CONV_MAX_TOL)})
        print(f"  收敛检查完成 {hkey}", flush=True)
    if conv:
        check("PGD 收敛: 更强的 PGD 相对评估用 PGD 的增量在全部检查点上都满足判据", all(c["converged"] for c in conv),
              f"最大平均增量 {_f(max(c['mean_inc'] for c in conv), 5)}, 最大单样本增量 {_f(max(c['max_inc'] for c in conv), 5)}")

    # ===== 8. 公共 logit 平移的不变性 =====
    inv = []
    g = torch.Generator().manual_seed(11)
    for hkey in (("init", 1), ("full_a1", 1)):
        A = A_of(hkey).double().cpu()
        gap, dgap = invariance_check(heads[hkey]["logits"][:300].double().cpu(), U[:300], A, ELL[1]["lang_full"],
                                     y_val[:300].cpu(), eps_dom, torch.randn(K, generator=g, dtype=torch.float64), 0.37)
        inv.append({"head": f"{hkey[0]}|{hkey[1]}", "ld_gap": gap, "ddo_gap": dgap})
    check("LD 损失对公共 logit 平移严格不变 (差 < 1e-9); 原始 DDO 的损失发生变化", all(i["ld_gap"] < 1e-9 and i["ddo_gap"] > 1e-9 for i in inv))

    res = {"smoke": smoke, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code, "device": device, "K": int(K),
           "G_sha256": hashlib.sha256(G.numpy().tobytes()).hexdigest(), "E_sha256": hashlib.sha256(E.numpy().tobytes()).hexdigest(),
           "U_shape": list(U.shape), "tr_G": float(G.trace()), "eta_mean": second_moments(U)[3],
           "grid_mult": list(GRID_MULT), "eps_dom_lang_full": eps_dom, "dom": dom, "vis": vis, "du_norm_mean": du_norm,
           "ell_meta": {str(s): ell_meta[s] for s in SEEDS}, "rand_seed_used": {f"{k[0]}|{k[1]}": v for k, v in rand_seed_used.items()},
           "init_ref_ce": {str(s): init_ref[s]["ce"] for s in SEEDS}, "rho_ddo": rho_ddo, "cos_ddo": cos_ddo, "rho_bar": rho_bar,
           "r_max_mean_init": r_max_mean, "calibration": cal, "p_star": p_star, "grid": grid,
           "head_sha256": head_sha, "aug_cache_sha256": _sha(AUG_CACHE), "n_val": int(val_idx.numel()), "n_pgd": N_PGD,
           "pgd": {"steps": PGD_STEPS, "restarts": PGD_RESTARTS, "seed": PGD_SEED},
           "conv_settings": {"steps": CONV_STEPS, "restarts": CONV_RESTARTS, "n": N_CONV, "mult": list(CONV_MULT),
                             "mean_tol": CONV_MEAN_TOL, "max_tol": CONV_MAX_TOL},
           "convergence": conv, "invariance": inv, "all_checks_passed": ok, "table": table,
           "elapsed_seconds": round(time.time() - t0, 1)}
    if smoke:
        print("[smoke] 不写产物; 流程走通, 全部核对项:", ok)
        print(json.dumps({"eps_dom": round(eps_dom, 3), "rho_bar": round(rho_bar, 4), "p_star": round(p_star, 4),
                          "eps_grad": {e: round(c["eps_grad"], 3) for e, c in cal.items()},
                          "eps_vis": {e: round(v["eps_vis_median"], 2) for e, v in vis.items()}}, ensure_ascii=False))
        return 0 if ok else 1
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stage1c_radius_cub.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(OUT_DIR, "stage1c_radius_cub.md"), "w", encoding="utf-8") as f:
        f.write(write_report(res))
    print(f"[saved] {OUT_DIR}/stage1c_radius_cub.json、.md  (耗时 {res['elapsed_seconds']}s)")
    print(json.dumps({"eps_dom": round(eps_dom, 3), "rho_bar": round(rho_bar, 4), "p_star": round(p_star, 4),
                      "eps_grad": {e: round(c["eps_grad"], 3) for e, c in cal.items()},
                      "eps_pen": {e: round(c["eps_pen"], 3) for e, c in cal.items()},
                      "all_checks_passed": ok}, ensure_ascii=False, indent=1))
    return 0 if ok else 1


# ---------------- 报告 (由同一份结果生成) ----------------
def write_report(res):
    cal, dom, vis, table = res["calibration"], res["dom"], res["vis"], res["table"]
    L = [f"# Stage 1C 半径诊断（LD-DRO；CUB；{res['created'][:10]}）", "",
         "预注册见 `RESULTS.md` §3x。只用文本与源域数据；收敛分类头只用于事后量级诊断，初始强度配平只用 3 个初始化分类头。"
         f"评估样本：heldout_val 的 1200 个干净特征；PGD 只在前 {N_PGD} 个样本上做。本文件由脚本从同一份 JSON 生成。", "",
         "## 1. 量纲锚点", "",
         f"- K = {res['K']}；G 的迹 = {_f(res['tr_G'], 4)}；均值能量占比 = {_f(res['eta_mean'], 4)}",
         f"- **ε_dom（Language-Full）= {_f(res['eps_dom_lang_full'], 2)}**：实际域方向 u = LN0(E·d) 在语言椭球内的白化半径（均方根，3 个随机种子的平均对 Language / Identity 即其自身）",
         f"- 光度扰动 Δu 在概念空间里的范数（7 种扰动的均值）：" + "，".join(f"{n} {_f(v, 2)}" for n, v in res["du_norm_mean"].items()), "",
         "| 椭球 | 域方向的白化半径 ε_dom | 域方向在值域外的能量 | 扰动 Δu 的白化半径（中位） | [q25, q75] | Δu 在值域外的能量 |",
         "|---|---|---|---|---|---|"]
    for e in ELL_ORDER:
        L.append(f"| {ELL_LABEL[e]} | {_f(dom[e]['eps_dom'], 2)} | {_f(dom[e]['resid_mean'], 4)} | {_f(vis[e]['eps_vis_median'], 2)} | "
                 f"[{_f(vis[e]['eps_vis_q25'], 2)}, {_f(vis[e]['eps_vis_q75'], 2)}] | {_f(vis[e]['resid_mean'], 4)} |")
    L += ["", "## 2. 初始强度配平（3 个初始化分类头，train_core 的前 256 个样本）", "",
          f"- 原始等权 DDO（α = 1）在初始化时的梯度范数比 ρ̄_DDO = **{_f(res['rho_bar'], 4)}**（逐 seed：" + "，".join(f"s{k} {_f(v, 4)}" for k, v in res["rho_ddo"].items()) + "）",
          "- 梯度比 = **正则部分**的梯度范数 / CE 的梯度范数（LD / ISDA 的正则部分 = 损失 − CE；DDO 的正则部分 = 它自己），与训练时记录的 `grad_norm_ratio_ddo_over_ce` 是同一个量。",
          "- ε_grad：LD 的初始梯度比 = ρ̄_DDO 的 ε（二分求出）；P* = Language-Full 在 ε_grad 下的初始平均绑定惩罚；"
          "ε_pen：各椭球的初始平均绑定惩罚 = P* 的 ε；λ_grad / λ_pen：ISDA 的对应标定。", "",
          "| 椭球 | 单位半径 r_max 的均值 | ε_grad | 梯度夹角余弦 | ε_pen | λ_grad | λ_pen | ε_grad / ε_dom | 网格上单调 | 二分夹住 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for e in ELL_ORDER:
        c = cal[e]
        L.append(f"| {ELL_LABEL[e]} | {_f(res['r_max_mean_init'][e], 4)} | {_f(c['eps_grad'], 3)} | {_f(c['cos_ld_at_eps_grad'], 3)} | "
                 f"{_f(c['eps_pen'], 3)} | {_f(c['lam_grad'], 2)} | {_f(c['lam_pen'], 2)} | {_f(c['eps_grad'] / res['eps_dom_lang_full'], 4)} | "
                 f"{c['monotone_on_grid']} | {c['eps_grad_bracketed'] and c['lam_grad_bracketed']} |")
    L.append(f"\nP* = {_f(res['p_star'], 4)}。DDO 梯度与 CE 梯度的夹角余弦（逐 seed）：" + "，".join(f"s{k} {_f(v, 3)}" for k, v in res["cos_ddo"].items()))

    def avg(keys, path):
        vals = []
        for k in keys:
            d = table[k]
            for p in path:
                d = d[p]
            vals.append(d)
        return vals

    L += ["", "## 3. 同半径口径：ε 网格上的闭式量与 PGD", "",
          "CE / LD 按样本集分开报告（外审第十轮的口径更正）：CE（1200）与 LD（1200）用 heldout_val 全体；"
          "CE（300）/ LD（300）/ PGD 下界（300）/ gap（300）四列都来自**前 300 个样本的同一子集**，四列之间可以互相比较。"
          "最大类别对惩罚 = 每个样本在全部竞争类上的最大惩罚的均值（历史标签\"绑定惩罚\"名不副实，见 §3y 的更正）。", ""]
    for grp, label, keys_of in (("init", "初始化分类头（3 seeds 均值）", lambda e: [f"init|{s}|{e}" for s in SEEDS]),
                                *((arm, f"收敛分类头 {arm}（3 seeds 均值）", (lambda e, arm=arm: [f"{arm}|{s}|{e}" for s in SEEDS])) for arm in MAIN_ARMS)):
        L += [f"### {label}", "",
              "| ε (× ε_dom) | 椭球 | 最大类别对惩罚（均值） | 鲁棒最坏间隔 | 正确样本的翻转率 | CE（1200） | LD（1200） | CE（300） | LD（300） | PGD 下界（300） | gap（300） |",
              "|---|" + "---|" * 10]
        for i, mult in enumerate(res["grid_mult"]):
            for e in ELL_ORDER:
                m = avg(keys_of(e), ["by_eps", i])
                pg = [table[k]["pgd"][i] for k in keys_of(e) if "pgd" in table[k]]
                if pg:
                    same = [_mean([p["ce"] for p in pg]), _mean([p["ld"] for p in pg]),
                            _mean([p["pgd"] for p in pg]), _mean([p["gap"] for p in pg])]
                else:
                    same = [float("nan")] * 4
                L.append(f"| {_f(res['grid'][i], 2)} ({_f(mult, 4)}) | {ELL_LABEL[e]} | {_f(_mean([x['pen_bind_mean'] for x in m]))} | "
                         f"{_f(_mean([x['robust_worst_margin_mean'] for x in m]))} | {_f(100 * _mean([x['flip_rate_correct'] for x in m]), 1)}% | "
                         f"{_f(_mean([x['ce'] for x in m]), 3)} | {_f(_mean([x['loss'] for x in m]), 3)} | "
                         f"{_f(same[0], 3)} | {_f(same[1], 3)} | {_f(same[2], 3)} | {_f(same[3], 3)} |")
        L.append("")
    L += ["## 4. 同初始惩罚口径（ε_pen）与梯度配平口径（ε_grad）下的闭式量", "",
          "| 分类头 | 椭球 | ε_grad | 翻转率（正确样本）@ε_grad | ε_pen | 翻转率（正确样本）@ε_pen | λ_grad | 翻转率@λ_grad | λ_pen | 翻转率@λ_pen |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for hkey in [("init", s) for s in SEEDS] + [(arm, s) for arm in MAIN_ARMS for s in SEEDS]:
        for e in ELL_ORDER:
            c = table[f"{hkey[0]}|{hkey[1]}|{e}"]["cal"]
            L.append(f"| {hkey[0]} s{hkey[1]} | {ELL_LABEL[e]} | {_f(cal[e]['eps_grad'], 2)} | {_f(100 * c['ld_eps_grad']['flip_rate_correct'], 1)}% | "
                     f"{_f(cal[e]['eps_pen'], 2)} | {_f(100 * c['ld_eps_pen']['flip_rate_correct'], 1)}% | {_f(cal[e]['lam_grad'], 2)} | "
                     f"{_f(100 * c['isda_lam_grad']['flip_rate_correct'], 1)}% | {_f(cal[e]['lam_pen'], 2)} | {_f(100 * c['isda_lam_pen']['flip_rate_correct'], 1)}% |")
    L += ["", "## 5. PGD 收敛检查（评估用 PGD-50×10 对更强的 PGD-200×20；前 100 个样本）", "",
          f"判据：平均增量 < {res['conv_settings']['mean_tol']} 且最大增量 < {res['conv_settings']['max_tol']}。", "",
          "| 分类头 | 椭球 | ε (× ε_dom) | 平均增量 | 最大增量 | 增量 > 1e-3 的样本数 | 收敛 |", "|---|---|---|---|---|---|---|"]
    for c in res["convergence"]:
        L.append(f"| {c['head']} | {ELL_LABEL[c['ell']]} | {_f(c['eps'] / res['eps_dom_lang_full'], 4)} | {_f(c['mean_inc'], 5)} | "
                 f"{_f(c['max_inc'], 4)} | {c['n_gt_1e-3']}/{res['conv_settings']['n']} | {c['converged']} |")
    L += ["", "## 6. 公共 logit 平移的不变性", ""]
    for i in res["invariance"]:
        L.append(f"- {i['head']}：LD 损失之差 {_f(i['ld_gap'], 12)}；原始 DDO 损失之差 {_f(i['ddo_gap'], 6)}")
    L += ["", "## 读法与边界", "",
          "- ε 网格只是覆盖量级的采样；真正用于配平的是连续求出的 ε_grad / ε_pen / λ_grad / λ_pen。",
          "- 初始化分类头的间隔本身没有意义（随机），它们只用来度量惩罚与梯度的量级；间隔与翻转率看收敛分类头。",
          "- 收敛分类头是在完整训练集上训练的，heldout_val 的样本对它们是训练集内样本；这里只做量级诊断，不做任何判定。",
          "- 固定批次 (train_core 的前 256 个样本) 只用于 3 个初始化头的梯度比，不涉及收敛头。",
          "- 预注册的 PGD 收敛检查没有全部通过: 未收敛的检查点在报告里列出, 那里的 PGD 下界只能读作\"可能偏低\", gap 不能单独归因于上界松。",
          "- 事后补充的收敛检查只覆盖测过的离散半径、3 个分类头与前 100 个样本, 不是全范围收敛证明; 正式 Language-PGD 训练需要在训练过程中检查搜索质量 (预注册 §3z)。",
          "- \"绑定惩罚\"(历史键名 pen_bind_*) 的实际含义是平均最大类别对惩罚, 不是实际最坏竞争类(鲁棒间隔最小者)的惩罚; 后者由 pen_worstcomp_* 报告 (外审第十轮的命名更正)。",
          "- LD 交叉熵是闭式上界，PGD 是数值下界；gap 是上下界之间的距离，不是已知真值后的误差。大的 ε 下上界会很松。",
          "- 扰动发生在 LayerNorm 之后、affine 之前的概念空间：不满足真实图像的可达性约束，不能据此声称对真实域偏移有认证鲁棒性。",
          "- ε_dom / ε_vis 是文本域方向、源域光度扰动在椭球内的白化半径，只说明椭球的量纲，不代表目标域偏移的大小。", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
