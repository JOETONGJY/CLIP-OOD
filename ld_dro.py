"""
Stage 1C: LD-DRO / Language-ISDA / Language-PGD 的损失 (规格见 doc/主线B实施规格.md §6; 记号见 §0)。

分类头 h(u) = A·u + c, A = W ⊙ gamma (C, K), c = W·beta + b。**u 必须是已过 LayerNorm 归一化的向量**
(LN0(x)); 对原始输入 x 不成立。扰动作用在 LayerNorm 后、affine 前的概念空间:
    u -> u + L q, ||q||_2 <= eps,  L (K, r) 是椭球生成元 (language_ellipsoid.build_ellipsoids)。于是
    h(u + L q) = h(u) + M q,   M = A L (C, r)          —— 对 q 仿射; beta、b 不进入 M (零偏置泄漏)
    ell_y - ell_k 在最坏扰动下减少 eps * r_yk,  r_yk = ||M_k - M_y||_2   —— 逐竞争类最坏间隔的精确闭式解

三个损失 (y 为真实类):
  LD-DRO  (闭式上界):  ell~_k = ell_k + eps * r_yk          (k != y)
           L_LD = CE(logits + eps * r)
           ** G3: 这是"共享扰动下最坏交叉熵"的闭式上界, 不是精确解 ** (各竞争类的最坏扰动不一定是同一个 q)
  ISDA    (Jensen 上界, 高斯扰动 delta ~ N(0, lam * Sigma)):
           ell~_k = ell_k + (lam / 2) * r_yk^2              (k != y)   —— 平方距离项, 不开根号 (G3)
  PGD     (共享扰动的数值最坏值): max_{||q||<=eps} CE(h(u) + M q)
           —— CE 对 q 是凸函数, 但凸函数的**最大化**不保证 PGD 找到全局最优 (G3);
              成立的是链  CE(0) <= L_PGD,best <= L_worst <= L_LD
           —— 必须把零扰动纳入候选, 并保留搜索过程中的历史最大损失

结论边界 (G3): 扰动位于 LayerNorm 后的概念空间, 不满足真实图像的范数/可达性约束;
               只能定位为"语言引导的概念空间扰动鲁棒性方法", 不能声称对真实图像偏移有可认证鲁棒性。
用法 (自检): python ld_dro.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SQRT_EPS = 1e-8


def effective_response(classifier):
    """由分类头 nn.Sequential(Flatten, LayerNorm(K), Linear(K, C)) 取 A = W ⊙ gamma (C, K) 与 c = W·beta + b (C,)。"""
    ln, lin = classifier[1], classifier[2]
    A = lin.weight * ln.weight[None, :]                      # W ⊙ gamma (按概念逐列)
    c = lin.weight @ ln.bias + lin.bias
    return A, c


def pair_distances(M, y):
    """r_yk = ||M_k - M_{y_b}|| (B, C) 与平方距离 (B, C)。M: (C, r) 或 (B, C, r); y: (B,)。
    第 b 个样本减去的是它自己真实类的行 M_b[y_b]; 位置 (b, y_b) 显式置 0 (C 细节 3.2: sqrt(0) 的反传会出 NaN)。"""
    B = y.size(0)
    Mb = M[None].expand(B, -1, -1) if M.dim() == 2 else M
    My = Mb[torch.arange(B, device=y.device), y]            # (B, r): 每个样本自己真实类的行
    d2 = (Mb - My[:, None, :]).pow(2).sum(-1)               # (B, C)
    r = torch.sqrt(d2 + SQRT_EPS)
    is_true = torch.zeros(B, Mb.size(1), dtype=torch.bool, device=Mb.device)
    is_true[torch.arange(B, device=y.device), y] = True     # (B, C): 每个样本自己的真实类位置
    return r.masked_fill(is_true, 0.0), d2.masked_fill(is_true, 0.0)


def ld_dro_loss(logits, M, y, eps):
    """LD-DRO: 闭式上界交叉熵。logits (B, C); M (C, r) 或 (B, C, r); y (B,)。eps=0 时严格等于 CE。"""
    if eps == 0:
        return F.cross_entropy(logits, y)
    r, _ = pair_distances(M, y)
    return F.cross_entropy(logits + eps * r, y)


def isda_loss(logits, M, y, lam):
    """Language-ISDA: Jensen 上界 (高斯扰动)。lam=0 时严格等于 CE。"""
    if lam == 0:
        return F.cross_entropy(logits, y)
    _, d2 = pair_distances(M, y)                             # 平方距离, 不开根号 (G3)
    return F.cross_entropy(logits + 0.5 * lam * d2, y)


def pgd_worst_loss(logits, M, y, eps, steps=5, restarts=1, seed=0):
    """共享扰动下的数值最坏交叉熵 (PGD, 投影梯度上升), 在 q 空间 (B, r) 中**逐样本**进行
    (与 LD-DRO 的逐样本形式一致: 每个样本有自己的最坏 q)。
    logits (B, C) 为干净 logits; M (C, r)。返回 (batch 平均的最坏损失, q_best (B, r))。
    **零扰动始终是候选之一, 并保留整个搜索过程中的历史最大损失** (G3)。
    注意: 凸函数的最大化不保证全局最优 —— 返回值是下界, LD-DRO 是上界。"""
    B, C = logits.shape
    r = M.size(-1)
    device = logits.device
    Mb = M[None].expand(B, -1, -1)

    def ce_of(q):                                           # 逐样本的 CE, (B,)
        return F.cross_entropy(logits + torch.einsum("bcr,br->bc", Mb, q), y, reduction="none")

    best = ce_of(torch.zeros(B, r, device=device)).detach()             # 候选: 零扰动
    best_q = torch.zeros(B, r, device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    for rs in range(int(restarts)):
        q = torch.zeros(B, r, device=device)
        if rs > 0:
            q = torch.randn(B, r, generator=g, device=device)
        q = (q / (q.norm(dim=-1, keepdim=True) + 1e-12) * eps).requires_grad_(True)
        for _ in range(int(steps)):
            loss = ce_of(q).sum()
            (gq,) = torch.autograd.grad(loss, q)
            with torch.no_grad():
                step = 2.0 * eps / max(steps, 1)
                q = q + step * gq / (gq.norm(dim=-1, keepdim=True) + 1e-12)
                n = q.norm(dim=-1, keepdim=True)
                q = torch.where(n > eps, q * (eps / n.clamp_min(1e-12)), q)
                cur = ce_of(q)                                   # 历史最大 (逐样本)
                better = cur > best
                best = torch.where(better, cur, best)
                best_q = torch.where(better[:, None], q, best_q)
            q = q.requires_grad_(True)
    return best.mean(), best_q


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD ") + name + (f"  [{detail}]" if detail else ""))

    torch.manual_seed(0)
    K, C, r, B = 24, 9, 5, 64
    ln, lin = torch.nn.LayerNorm(K), torch.nn.Linear(K, C)
    with torch.no_grad():
        lin.weight.mul_(3.0); ln.weight.mul_(2.0)
        lin.bias.copy_(torch.randn(C) * 0.5); ln.bias.copy_(torch.randn(K) * 0.3)
    head = torch.nn.Sequential(torch.nn.Flatten(), ln, lin)
    A, c = effective_response(head)
    x = torch.randn(B, K)
    u0 = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
    logits = u0 @ A.T + c
    diff = float((logits - lin(ln(x))).abs().max())
    check("E1 h(u) = A·u + c 与分类头前向一致 (u = LN0(x))", diff < 1e-5, f"max|diff|={diff:.2e}")
    check("E1b 对未归一化的原始 x 不成立 (提醒: u 必须先归一化)", not torch.allclose(x @ A.T + c, lin(ln(x)), atol=1e-3))
    L_ = torch.randn(K, r) * torch.linspace(2, 0.3, r)       # 椭球生成元 (K, r)
    M = A @ L_                                                # (C, r)
    y = torch.randint(0, C, (B,))
    ce = F.cross_entropy(logits, y)

    check("L1 LD-DRO 在 eps=0 时 == CE", torch.allclose(ld_dro_loss(logits, M, y, 0.0), ce))
    check("L2 ISDA 在 lam=0 时 == CE", torch.allclose(isda_loss(logits, M, y, 0.0), ce))
    rk, d2 = pair_distances(M, y)
    check("L3 r_yk 的对角线 (k=y) 恰为 0", float(rk[torch.arange(B), y].abs().max()) == 0.0)
    # 逐样本自己的真实类: 第 b 行是 ||M_k - M[y_b]||, k = 0..C-1
    explicit = torch.stack([torch.sqrt((M - M[yb]).pow(2).sum(-1) + SQRT_EPS) for yb in y])
    explicit[torch.arange(B), y] = 0.0                           # 真实类位置同样置 0
    check("L3b r_yk 与显式计算的 ||M_k - M_{y_b}|| 一致 (逐样本自己的真实类)", torch.allclose(rk, explicit, atol=1e-5))
    lin.zero_grad(); ln.zero_grad()
    lo = lin(ln(x))
    Mi = effective_response(head)[0] @ L_
    ld = ld_dro_loss(lo, Mi, y, 0.3)
    ld.backward()
    check("L4 LD-DRO 的损失与对 W、gamma 的梯度都有限 (sqrt(0) 没有产生 NaN)",
          bool(torch.isfinite(ld)) and bool(torch.isfinite(lin.weight.grad).all()) and bool(torch.isfinite(ln.weight.grad).all()),
          f"|dL/dW|max={float(lin.weight.grad.abs().max()):.3e}")
    check("L5 M = A·L 不含 beta 与 b (零偏置泄漏)", torch.allclose(Mi, (lin.weight * ln.weight[None, :]) @ L_) and Mi.shape == (C, r))
    lin.zero_grad(); ln.zero_grad()
    li = isda_loss(lin(ln(x)), effective_response(head)[0] @ L_, y, 1.0)
    li.backward()
    check("L6 ISDA 的损失与梯度有限", bool(torch.isfinite(li)) and bool(torch.isfinite(lin.weight.grad).all()))

    # ---- 逐竞争类最坏间隔的闭式解 == 数值最大化 ----
    k = int((y[0] + 1) % C)
    dk = (Mi[k] - Mi[y[0]]).detach()                         # 竞争类 k 相对真实类的方向响应差
    best_m = -1e9
    for _ in range(40):                                      # 最大化 (M_k - M_y)·q, ||q|| <= 0.3
        q = torch.randn(r); q = (q / q.norm() * 0.3).requires_grad_(True)
        for _ in range(200):
            (gq,) = torch.autograd.grad(dk @ q, q)
            q = (q.detach() + 0.02 * gq / (gq.norm() + 1e-12))
            n = q.norm()
            if n > 0.3: q = q * (0.3 / n)
            q = q.requires_grad_(True)
        best_m = max(best_m, float(dk @ q.detach()))
    check("L7 逐竞争类最坏间隔的闭式解 eps·||M_k - M_y|| == 数值最大化", abs(0.3 * float(dk.norm()) - best_m) < 1e-5,
          f"closed={0.3 * float(dk.norm()):.6f} numeric={best_m:.6f}")

    # ---- 链: CE(0) <= PGD-5 <= PGD-50(多重启) <= LD-DRO ----
    for eps in (0.05, 0.2, 0.5):
        p5, _ = pgd_worst_loss(logits, M, y, eps, steps=5)
        p50, _ = pgd_worst_loss(logits, M, y, eps, steps=50, restarts=4)
        lde = float(ld_dro_loss(logits, M, y, eps))
        a, b_, c_, d_ = float(ce), float(p5), float(p50), lde
        check(f"L8 eps={eps}: CE(0) <= PGD-5 <= PGD-50(4 restarts) <= LD-DRO", a <= b_ + 1e-6 <= c_ + 1e-6 <= d_ + 1e-6,
              f"{a:.4f} <= {b_:.4f} <= {c_:.4f} <= {d_:.4f}; gap(LD-PGD50)={d_ - c_:.5f}")
    p0, q0 = pgd_worst_loss(logits, M, y, 0.0, steps=3)
    check("L9 PGD 在 eps=0 时返回 CE(0) 且 q=0 (零扰动是候选)", abs(float(p0) - float(ce)) < 1e-6 and float(q0.abs().max()) == 0.0)
    pr, qr = pgd_worst_loss(logits, M, y, 0.3, steps=20, restarts=3, seed=0)
    pr2, _ = pgd_worst_loss(logits, M, y, 0.3, steps=20, restarts=3, seed=0)
    check("L10 PGD 可复现 (同 seed 同结果)", abs(float(pr) - float(pr2)) < 1e-7)
    check("L11 PGD 返回的 q 满足 ||q|| <= eps", float(qr.norm(dim=-1).max()) <= 0.3 + 1e-6)
    check("L12 ISDA >= CE (Jensen 上界)", float(isda_loss(logits, M, y, 0.5)) >= float(ce) - 1e-6)

    # ---- 不同椭球给出不同损失 (对照臂不是同一个东西) ----
    from language_ellipsoid import build_ellipsoids
    U = torch.randn(3000, K, dtype=torch.float64) * torch.linspace(3, 0.2, K)
    U = U - U.mean(-1, keepdim=True)
    e = build_ellipsoids(U.T @ U / U.size(0), r=r, seed=0)
    lds = {n: float(ld_dro_loss(logits, A @ e[n].float(), y, 0.3)) for n in ("language", "random", "identity", "flat")}
    print("  各椭球下的 LD-DRO:", {k: round(v, 4) for k, v in lds.items()})
    check("L13 language 与 random (同谱不同轴) 给出不同损失", abs(lds["language"] - lds["random"]) > 1e-4)
    check("L14 identity 与 language 不同", abs(lds["identity"] - lds["language"]) > 1e-4)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
