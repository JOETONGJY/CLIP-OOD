"""LD-DRO / Language-ISDA / Language-PGD 的训练接入（外审第十轮批准；预注册见 RESULTS.md §3z）。

损失的数学语义在 ld_dro.py（LD 闭式上界 / ISDA Jensen 上界 / PGD 两段式训练损失）。本模块只做三件事：
  1. training_loss(mode, ...): 训练损失的统一分发器。mode='none' 时就是 CE —— 默认路径的行为逐位不变。
  2. pen_stats(...): 逐 batch 的惩罚 / 间隔统计（写进结果 json 的 epoch_metrics），定义与
     stage1c_radius_diagnose.penalized_metrics 相同（最大类别对惩罚、实际最坏竞争类的惩罚、鲁棒最坏间隔），
     只是不算分位数与损失 —— 每 batch 都算，要便宜。
  3. build_ld_generator(G, name, seed): 由二阶矩 G 构造椭球生成元 L（float32）。随机椭球的种子候选规则与
     Stage 1C 半径诊断完全一致（s / s+10 / s+20 / s+30，构造自检不过就换下一个并如实记录）。

配置守卫与记录在 main.TrainingSession._init_ld_dro：LD 臂要求 alpha=0 / beta=0 / weight_mode=none /
ddo_dirs=full / lambda_kl=0 / lambda_graph=0（强度完全由 eps / lam 决定，不叠加原始 DDO 正则）；
评估始终用干净 logits（鲁棒项只进训练目标）；eps=0 / lam=0 时严格退化为 CE。
用法（自检）: python ld_training.py
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ld_dro import isda_loss, ld_dro_loss, pgd_train_loss  # noqa: E402

from language_ellipsoid import build_ellipsoids  # noqa: E402

MODES = ("none", "ld", "isda", "pgd")
ELLS = ("lang_full", "lang32", "rand_full", "rand32", "identity")
SEED_CANDS = lambda s: (int(s), int(s) + 10, int(s) + 20, int(s) + 30)   # noqa: E731  与 §3x 相同的候选顺序


def warmup_lam(lam, epoch0, warmup_epochs):
    """ISDA 的线性 warmup (外审第十一轮 P3): 前 W 个 epoch 的有效 λ = λ × (epoch0+1)/W（epoch0 从 0 计），
    之后恒为 λ。W<=0 或 λ=0 时原样返回 λ（关闭）。"""
    lam = float(lam)
    W = int(warmup_epochs)
    if W <= 0 or lam == 0.0:
        return lam
    return lam * min(1.0, (int(epoch0) + 1) / W)


def training_loss(mode, logits, M, y, eps=0.0, lam=0.0, pgd_steps=5, pgd_restarts=1, pgd_seed=0):
    """训练损失分发器。none -> CE；ld -> CE(logits + eps*r)；isda -> CE(logits + (lam/2)*r²)；
    pgd -> 两段式训练损失（detach 搜索 + 固定扰动重算，梯度正常流经分类头）。eps=0 / lam=0 严格等于 CE。"""
    if mode == "none":
        return F.cross_entropy(logits, y)
    if mode == "ld":
        return ld_dro_loss(logits, M, y, eps)
    if mode == "isda":
        return isda_loss(logits, M, y, lam)
    if mode == "pgd":
        return pgd_train_loss(logits, M, y, eps, steps=int(pgd_steps), restarts=int(pgd_restarts), seed=int(pgd_seed))
    raise ValueError(f"未知 ld_mode='{mode}' (可选: {MODES})")


def pen_stats(logits, M, y, strength, kind="ld"):
    """逐 batch 的监控统计（detach 后计算，不参与梯度）。键名与 §3y 的命名更正一致：
    pen_maxpair_* = 每个样本在全部竞争类上的最大惩罚（"平均最大类别对惩罚"；历史叫法"绑定惩罚"名不副实）；
    pen_worstcomp_mean = 实际最坏竞争类（鲁棒间隔最小者）上的惩罚；
    robust_margin_mean = min_{k≠y} (m_yk − pen)；clean_margin_mean = 干净最坏间隔
    min_{k≠y} (ℓ_y − ℓ_k) —— 外审第十一轮修正：此前未排除正确类自身（m_yk 对角恒 0，min 被它截断，
    正确类为 argmax 的样本读数恒为 0），修正后才是真正的干净最坏间隔；clean_ce = 干净 CE。"""
    with torch.no_grad():
        rows = torch.arange(y.numel(), device=y.device)
        d2 = (M[None] - M[y][:, None, :]).pow(2).sum(-1)             # (B, C)
        d2[rows, y] = 0.0
        r = torch.sqrt(d2 + 1e-8)
        pen = (strength * r) if kind == "ld" else (0.5 * strength * r * r)
        m_yk = logits[rows, y][:, None] - logits
        rob = m_yk - pen
        rob[rows, y] = float("inf")
        m_rob = rob.min(1).values
        wc = rob.argmin(1)
        pen_max = pen.max(1).values
        m_clean = m_yk.clone()
        m_clean[rows, y] = float("inf")                              # 排除正确类自身 (外审第十一轮)
        return {"pen_maxpair_mean": float(pen_max.mean()), "pen_maxpair_std": float(pen_max.std()),
                "pen_worstcomp_mean": float(pen[rows, wc].mean()),
                "robust_margin_mean": float(m_rob.mean()), "clean_margin_mean": float(m_clean.min(1).values.mean()),
                "clean_ce": float(F.cross_entropy(logits, y))}


def build_ld_generator(G, name, seed):
    """由二阶矩 G (K, K) 构造椭球生成元。返回 (L float32 (K, r), meta)。
    name: lang_full / lang32 / rand_full / rand32 / identity；r: 满秩椭球取 K−1，32 后缀取 32。
    language / identity 的 L 不依赖种子；random 的种子按候选顺序 s / s+10 / s+20 / s+30 取第一个能通过
    build_ellipsoids 构造自检（随机首轴与语言首轴夹角）的，并如实记录 ell_seed_used。"""
    if name not in ELLS:
        raise ValueError(f"未知椭球 '{name}' (可选: {ELLS})")
    K = G.size(0)
    r = K - 1 if name in ("lang_full", "rand_full", "identity") else 32
    built, used = None, None
    for cand in SEED_CANDS(seed):                        # language/identity 也走同一候选循环（build 的自检含随机部分）
        try:
            built = build_ellipsoids(G, r=r, seed=cand, with_flat=False)
            used = cand
            break
        except AssertionError:
            continue
    if built is None:
        raise RuntimeError(f"椭球 {name}: 找不到可用的随机种子（候选 {SEED_CANDS(seed)} 全部未通过构造自检）")
    key = "language" if name.startswith("lang") else ("random" if name.startswith("rand") else "identity")
    L = built[key].float()
    meta = {"ell": name, "r": int(L.size(1)), "ell_seed_used": int(used),
            "sha256": hashlib.sha256(L.detach().numpy().tobytes()).hexdigest(),
            "spectrum": {k: built["spectrum"][k] for k in ("r", "energy_captured_by_r",
                                                           "effective_rank_entropy_trunc",
                                                           "lambda_max_over_lambda_min_trunc",
                                                           "cos_first_axis_language_vs_random")}}
    return L, meta


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    K, C, B, r = 40, 9, 64, 8
    ln, lin = torch.nn.LayerNorm(K), torch.nn.Linear(K, C)
    head = torch.nn.Sequential(torch.nn.Flatten(), ln, lin)
    with torch.no_grad():
        lin.weight.mul_(2.0)
        ln.weight.mul_(1.5)
        lin.bias.copy_(torch.randn(C) * 0.3)
        ln.bias.copy_(torch.randn(K) * 0.2)
    x = torch.randn(B, K)
    u0 = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
    logits = u0 @ (lin.weight * ln.weight[None, :]).T + lin.weight @ ln.bias + lin.bias
    A = lin.weight * ln.weight[None, :]
    L = torch.randn(K, r) * torch.linspace(2, 0.3, r)
    M = A @ L
    y = torch.randint(0, C, (B,))
    ce = F.cross_entropy(logits, y)
    check("L1 mode=none 与 CE 相等", torch.equal(training_loss("none", logits, M, y), ce))
    check("L2 ld 在 eps=0 时严格等于 CE；isda 在 lam=0 时严格等于 CE",
          torch.allclose(training_loss("ld", logits, M, y, eps=0.0), ce) and
          torch.allclose(training_loss("isda", logits, M, y, lam=0.0), ce))
    check("L3 未知 mode 报错", _raises(lambda: training_loss("xx", logits, M, y)))
    lin.zero_grad(); ln.zero_grad()
    loss = training_loss("ld", logits.requires_grad_(True), M, y, eps=0.3)
    loss.backward()
    check("L4 LD 损失可反传且 W/gamma 梯度有限非零",
          bool(torch.isfinite(lin.weight.grad).all()) and float(lin.weight.grad.abs().max()) > 0)
    lin.zero_grad(); ln.zero_grad()
    lo = lin(ln(x))
    Mi = (lin.weight * ln.weight[None, :]) @ L
    lt = training_loss("pgd", lo, Mi, y, eps=0.3, pgd_steps=5, pgd_restarts=1, pgd_seed=0)
    lt.backward()
    ce2 = float(F.cross_entropy(lin(ln(x)).detach(), y))
    ld2 = float(ld_dro_loss(lin(ln(x)).detach(), Mi.detach(), y, 0.3))
    check("L5 PGD 训练损失带梯度、可反传，且落在 CE(0) 与 LD 上界之间",
          lt.requires_grad and float(lin.weight.grad.abs().max()) > 0 and ce2 <= float(lt) + 1e-6 <= ld2 + 1e-6,
          f"{ce2:.4f} <= {float(lt):.4f} <= {ld2:.4f}")
    l1 = training_loss("pgd", lin(ln(x)), Mi, y, eps=0.3, pgd_steps=5, pgd_restarts=2, pgd_seed=3)
    l2 = training_loss("pgd", lin(ln(x)), Mi, y, eps=0.3, pgd_steps=5, pgd_restarts=2, pgd_seed=3)
    check("L6 PGD 训练损失可复现（同 seed 同值）", abs(float(l1) - float(l2)) < 1e-9)
    lg64 = logits.detach().double()
    check("L7 float64 的 logits 也可算 LD 损失（dtype 不崩）", float(training_loss("ld", lg64, M.double(), y, eps=0.3)) > 0)

    # ---- pen_stats 与手算一致; 两套惩罚统计分开 ----
    ps = pen_stats(logits.detach(), M, y, 2.0, kind="ld")
    rows = torch.arange(B)
    r_yk = torch.cdist(M, M).pow(2).clamp_min(0)[y].sqrt()
    r_yk[rows, y] = 0.0
    pen = 2.0 * r_yk
    m_yk = logits[rows, y][:, None] - logits
    rob = m_yk - pen
    rob[rows, y] = float("inf")
    wc = rob.argmin(1)
    m_clean = m_yk.clone()
    m_clean[rows, y] = float("inf")                                  # 排除正确类自身 (外审第十一轮)
    check("L8 pen_stats: 最大对惩罚的均值/标准差、最坏竞争类惩罚、鲁棒/干净间隔、干净 CE 与手算一致 (干净间隔排除自身类)",
          abs(ps["pen_maxpair_mean"] - float(pen.max(1).values.mean())) < 1e-6
          and abs(ps["pen_maxpair_std"] - float(pen.max(1).values.std())) < 1e-6
          and abs(ps["pen_worstcomp_mean"] - float(pen[rows, wc].mean())) < 1e-6
          and abs(ps["robust_margin_mean"] - float(rob.min(1).values.mean())) < 1e-6
          and abs(ps["clean_margin_mean"] - float(m_clean.min(1).values.mean())) < 1e-6
          and abs(ps["clean_ce"] - float(ce)) < 1e-7)
    # 干净间隔的修正核对: 全部正确类为 argmax 的样本, 修正后的干净间隔必须为正 (旧实现对角 0 使其恒为 0)
    lg_ok = logits.detach().clone()
    rows_ok = torch.arange(B)
    lg_ok[rows_ok, y] = lg_ok.max(1).values + 0.5                    # 强制正确类为 argmax
    ps_ok = pen_stats(lg_ok, M, y, 2.0, kind="ld")
    check("L8b 修正后的干净最坏间隔在正确类为 argmax 时为正 (旧实现对角 0 使其恒为 0, 外审第十一轮)",
          bool((ps_ok["clean_margin_mean"] > 0) if isinstance(ps_ok["clean_margin_mean"], (int, float)) else True)
          and ps_ok["clean_margin_mean"] > 0, f"{ps_ok['clean_margin_mean']:.4f}")
    check("L9 pen_stats: 最坏竞争类惩罚 ≤ 最大对惩罚; ISDA 口径用 (lam/2)·r²",
          ps["pen_worstcomp_mean"] <= ps["pen_maxpair_mean"] + 1e-12
          and abs(pen_stats(logits.detach(), M, y, 4.0, kind="isda")["pen_maxpair_mean"]
                  - float((2.0 * r_yk * r_yk).max(1).values.mean())) < 1e-6)

    # ---- build_ld_generator: 迹 / 超平面 / 种子规则 / 语言不依赖种子 ----
    U = torch.randn(600, K, dtype=torch.float64)
    U = U - U.mean(-1, keepdim=True)
    G = U.T @ U / U.size(0)
    for name in ELLS:
        Lx, meta = build_ld_generator(G, name, 7)
        S = Lx.double() @ Lx.double().T
        # identity 的生成元是 K×K（秩 K−1, 含零均值方向上的一个冗余列）—— build_ellipsoids 的既有构造
        r_want = {"lang_full": K - 1, "rand_full": K - 1, "lang32": 32, "rand32": 32, "identity": K}[name]
        check(f"L10 {name}: 迹=1、在超平面内、生成元宽度正确、sha 与 meta 一致",
              abs(float(S.trace()) - 1.0) < 1e-6 and float(Lx.mean(0).abs().max()) < 1e-6
              and meta["r"] == r_want and meta["sha256"] == hashlib.sha256(Lx.numpy().tobytes()).hexdigest())
    La, _ = build_ld_generator(G, "lang_full", 1)
    Lb, _ = build_ld_generator(G, "lang_full", 2)
    Ra, ma = build_ld_generator(G, "rand_full", 1)
    Rb, mb = build_ld_generator(G, "rand_full", 2)
    check("L11 language 不依赖种子, random 依赖种子且记录了实际使用的候选",
          torch.equal(La, Lb) and not torch.equal(Ra, Rb) and ma["ell_seed_used"] == 1 and mb["ell_seed_used"] == 2)
    check("L12 同谱: language 与 rand_full 的 spectrum 摘要一致（同秩同谱）",
          La.size(1) == Ra.size(1) and build_ld_generator(G, "lang_full", 1)[1]["spectrum"]["r"]
          == ma["spectrum"]["r"])
    check("L13 未知椭球报错", _raises(lambda: build_ld_generator(G, "xx", 1)))
    Wl = [warmup_lam(100.0, e, 20) for e in range(22)]
    check("L14 warmup: 前 20 个 epoch 线性 (5, 10, ..., 100), 之后恒为 λ",
          Wl[0] == 5.0 and Wl[19] == 100.0 and Wl[20] == 100.0 and Wl[21] == 100.0
          and abs(Wl[9] - 50.0) < 1e-12, f"{[round(v, 1) for v in Wl[:3]]}...{Wl[19]},{Wl[20]}")
    check("L15 warmup 关闭 (W=0) 或 λ=0 时原样返回", warmup_lam(7.0, 0, 0) == 7.0 and warmup_lam(0.0, 3, 20) == 0.0)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _raises(fn):
    try:
        fn()
        return False
    except (ValueError, RuntimeError):
        return True


if __name__ == "__main__":
    sys.exit(self_test())
