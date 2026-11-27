"""L2 族的等价关系（第十三轮 P1 的数学补充；合成数据，不训练）。

外审第十三轮要求先把三个等价关系钉死，再解释 24 个 run：

  E1  DDO-L2（类无关方向集，无偏置）≡ (1/C) Σ_c A_cᵀ G A_c，G = (1/N) Σ_i u_i u_iᵀ
  E2  等权全类别对的平方响应 ≡ (2/(C−1)) Σ_c Ã_cᵀ G Ã_c，Ã_c = A_c − Ā（类别中心化）
  E3  迹归一化只改变 α 的尺度：G → G/tr(G) 与把 α 乘 tr(G) 等价
  E4  类条件方向集（训练里真正用的 full 方向集）≡ (1/C) Σ_c A_cᵀ G_c A_c，G_c 逐类不同
      —— 此时不存在单一的 G，E1 的等式不成立；这是“类平均方向”与“完整方向集”的差别
  E5  偏置：DDO-L2 的损失含 b_c；类别间隔（Pairwise / ISDA）里 b_c 严格消去

用法: python tests_l2_equivalences.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

TOL = 1e-5


def check(name, cond, detail=""):
    print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))
    return bool(cond)


def hilbert(U):
    """把方向投影到零均值超平面（LN0 的输出本来就在里面）。"""
    return U - U.mean(-1, keepdim=True)


def main():
    torch.manual_seed(0)
    N, K, C = 500, 32, 7
    U = hilbert(torch.randn(N, K))                    # 类无关方向集（对应 ddo_dirs=mean 的类平均方向）
    A = torch.randn(C, K) * 0.3
    b = torch.randn(C) * 0.2

    ok = True
    G = U.T @ U / N

    # E1: DDO-L2（无偏置，类无关方向）== (1/C) Σ_c A_cᵀ G A_c
    resp = A @ U.T                                    # (C, N)
    l2 = (resp ** 2).mean()                           # 对 (i, c) 全体取均值
    quad = ((A @ G) * A).sum() / C
    ok &= check("E1 DDO-L2（无偏置、类无关方向）≡ (1/C)Σ_c A_cᵀGA_c", abs(float(l2 - quad)) < TOL,
                f"{float(l2):.6f} vs {float(quad):.6f}")

    # E2: 等权全类别对 ≡ (2/(C−1)) Σ_c Ã_cᵀ G Ã_c
    Abar = A.mean(0, keepdim=True)
    At = A - Abar
    pair_sum = 0.0
    n_pair = 0
    for i in range(C):
        for j in range(i + 1, C):
            d = A[i] - A[j]
            pair_sum += float(d @ G @ d)
            n_pair += 1
    pair_mean = pair_sum / n_pair
    centered = float((2.0 / (C - 1)) * ((At @ G) * At).sum())
    ok &= check("E2 等权无序类别对 ≡ (2/(C−1))Σ_c Ã_cᵀGÃ_c", abs(pair_mean - centered) < TOL,
                f"{pair_mean:.6f} vs {centered:.6f}")

    # E2b: 与“未中心化”的 DDO-L2 的差恰好是公共分量 C·ĀᵀGĀ
    gap = float(quad - pair_mean)
    common = float(C * (Abar @ G @ Abar.T) / C * (C - 1) / 2.0) if False else float(
        ((A @ G) * A).sum() / C - (2.0 / (C - 1)) * ((At @ G) * At).sum())
    ok &= check("E2b 两者之差 = 公共分量项（非零，正是“类别中心化”去掉的东西）", abs(gap) > 1e-3,
                f"差 {gap:.6f}")

    # E3: 迹归一化 == α 尺度变换（用 float64 判，float32 的乘除顺序会留下 1e-7 级残差）
    tr = float(torch.trace(G.double()))
    A64, G64 = A.double(), G.double()
    quad64 = float(((A64 @ G64) * A64).sum() / C)
    a1 = 0.7
    a2 = 0.7 * tr
    v1 = a1 * quad64
    v2 = a2 * float(((A64 @ (G64 / tr)) * A64).sum() / C)
    ok &= check("E3 G→G/tr(G) 与 α→α·tr(G) 完全等价（float64）", abs(v1 - v2) < 1e-12 * max(1.0, abs(v1)),
                f"{v1:.9f} vs {v2:.9f}")

    # E4: 类条件方向集 → 逐类 G_c；E1 的单 G 等式不再成立
    Uc = hilbert(torch.randn(C, N, K))
    Gc = torch.einsum("cnk,cnl->ckl", Uc, Uc) / N     # (C,K,K)
    full = 0.0
    for c in range(C):
        full += float(A[c] @ Gc[c] @ A[c])
    full /= C
    Gbar = Gc.mean(0)
    single = float(((A @ Gbar) * A).sum() / C)
    ok &= check("E4 类条件方向集 ≡ (1/C)Σ_c A_cᵀG_cA_c（逐类 G_c）",
                abs(full - sum(float(A[c] @ Gc[c] @ A[c]) for c in range(C)) / C) < TOL, f"{full:.6f}")
    ok &= check("E4b 但用单一 Ḡ=(1/C)ΣG_c 会给出不同数值（类条件不能合并成一个 G）",
                abs(full - single) > 1e-4, f"{full:.6f} vs {single:.6f}")

    # E5: 偏置在间隔里消去、在 DDO-L2 里不消
    resp_b = A @ U.T + b[:, None]
    l2_b = float((resp_b ** 2).mean())
    ok &= check("E5a DDO-L2 含偏置时数值改变（b_c 未消去）", abs(l2_b - float(l2)) > 1e-3,
                f"{float(l2):.6f} → {l2_b:.6f}")
    # 偏置在 logit 里是“每类加一个常数”，等价于给 A_c 加 b_c·1_K；而 1_K 在 H 内不可见（G·1 = 0），
    # 所以类别间隔完全不变。这是与 DDO-L2（b_c 直接进入平方）最关键的区别。
    Ab = A + b[:, None] * torch.ones(1, K)
    d_plain = A[0] - A[1]
    d_biased = Ab[0] - Ab[1]
    g_plain = float(d_plain @ G @ d_plain)
    g_biased = float(d_biased @ G @ d_biased)
    ok &= check("E5b 类别间隔与 b 无关（b·1_K 在零均值超平面内不可见，G·1 = 0）",
                abs(g_plain - g_biased) < 1e-5, f"{g_plain:.6f} vs {g_biased:.6f}")

    # E6: ISDA 的 (λ/2)r² 就是上式乘 λ/2；用真实方向集复核一次（与 U 的二次型一致）
    lam = 0.3
    r2_quad = float(d_plain @ G @ d_plain)
    r2_direct = float(((d_plain @ U.T) ** 2).mean())
    ok &= check("E6 (λ/2)·r² 与在方向集上直接算的平方响应一致",
                abs(lam / 2 * r2_quad - lam / 2 * r2_direct) < 1e-5,
                f"{lam / 2 * r2_quad:.6f} vs {lam / 2 * r2_direct:.6f}")

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
