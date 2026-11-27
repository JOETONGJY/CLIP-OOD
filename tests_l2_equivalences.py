"""L2 族的等价关系（第十三轮 P1 的数学补充；合成数据，不训练）。

外审第十三轮要求先把三个等价关系钉死，再解释 24 个 run：

  E1  DDO-L2（类无关方向集，无偏置）≡ (1/C) Σ_c A_cᵀ G A_c，G = (1/N) Σ_i u_i u_iᵀ
  E2  等权全类别对的平方响应 ≡ (2/(C−1)) Σ_c Ã_cᵀ G Ã_c，Ã_c = A_c − Ā（类别中心化）
  E3  迹归一化只改变 α 的尺度：G → G/tr(G) 与把 α 乘 tr(G) 等价
  E4  类条件方向集（训练里真正用的 full 方向集）≡ (1/C) Σ_c A_cᵀ G_c A_c，G_c 逐类不同
      —— 此时不存在单一的 G，E1 的等式不成立；这是“类平均方向”与“完整方向集”的差别
  E5  偏置：DDO-L2 的损失含 b_c；原始类别间隔含 (b_y − b_k)（不消去）；只有"域扰动造成的间隔变化量"
      （乘 (A_k − A_y) 的项）与 ISDA/LD 的惩罚 r_yk 中 b 严格消去 —— 外审第十五轮的表述更正

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

    # E2b (外审第十五轮: 补精确恒等式, 原来只查差非零): quad − pair_mean = ĀᵀGĀ − ((C+1)/(C(C−1)))·Σ_c Ã_cᵀGÃ_c
    # （由 Σ_c Ã_c = 0 与 Σ_c Ã_cᵀGÃ_c = Σ_c A_cᵀGA_c − C·ĀᵀGĀ 展开；两项都要数值吻合，不只是非零）
    gap = float(quad - pair_mean)
    m_common = float(Abar @ G @ Abar.T)
    centered_sum = float(((At @ G) * At).sum())
    gap_exact = m_common - (C + 1.0) / (C * (C - 1.0)) * centered_sum
    ok &= check("E2b 两者之差的精确恒等式 = ĀᵀGĀ − ((C+1)/(C(C−1)))·ΣÃ_cᵀGÃ_c（公式与数值都核对）",
                abs(gap - gap_exact) < TOL and abs(gap) > 1e-4, f"差 {gap:.6f} vs 恒等式 {gap_exact:.6f}")

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

    # E4 (外审第十五轮: 原来是同一表达式自比, 恒真): 二次型 vs 显式方向响应 —— 两个独立算法必须一致
    Uc = hilbert(torch.randn(C, N, K))
    Gc = torch.einsum("cnk,cnl->ckl", Uc, Uc) / N     # (C,K,K)
    resp_cc = torch.einsum("ck,cnk->cn", A, Uc)       # (C, N): 每类只对自己方向集的响应
    full_direct = float((resp_cc ** 2).mean())        # 显式: mean over (类, 方向)
    full_quad = float(sum(float(A[c] @ Gc[c] @ A[c]) for c in range(C)) / C)
    ok &= check("E4 类条件方向集: 显式方向响应 mean(h²) ≡ (1/C)Σ_c A_cᵀG_cA_c（两个独立算法）",
                abs(full_direct - full_quad) < TOL, f"{full_direct:.6f} vs {full_quad:.6f}")
    Gbar = Gc.mean(0)
    single = float(((A @ Gbar) * A).sum() / C)
    single_direct = float(((A @ Gc.mean(0)) * A).sum() / C)
    ok &= check("E4b 用单一 Ḡ=(1/C)ΣG_c 会给出不同数值（类条件不能合并成一个 G）",
                abs(full_direct - single) > 1e-4 and abs(single - single_direct) < TOL,
                f"{full_direct:.6f} vs {single:.6f}")

    # E5 (外审第十五轮表述更正): 偏置在原始间隔里【不】消去 (差值 b_y − b_k 保留); 只在"域扰动造成的
    # 间隔变化量"里消去 —— 该变化量乘的是 (A_k − A_y), 与 b 无关; b_c·1_K 的表示在 G 二次型里不可见 (G·1=0)
    resp_b = A @ U.T + b[:, None]
    l2_b = float((resp_b ** 2).mean())
    ok &= check("E5a DDO-L2 含偏置时数值改变（b_c 未消去）", abs(l2_b - float(l2)) > 1e-3,
                f"{float(l2):.6f} → {l2_b:.6f}")
    margin_plain = float((A[0] - A[1]) @ U[0]) + float(b[0] - b[1])      # 原始间隔 ℓ_0 − ℓ_1 (含 b_0 − b_1)
    margin_nobias = float((A[0] - A[1]) @ U[0])
    ok &= check("E5b 原始间隔【含】(b_y − b_k)，不随偏置消去（更正: 只在扰动变化量中消去）",
                abs(margin_plain - margin_nobias - float(b[0] - b[1])) < 1e-6
                and abs(float(b[0] - b[1])) > 1e-3, f"间隔差 {margin_plain - margin_nobias:.4f} = b_0 − b_1")
    delta = U[1] - U[0]                                                  # 域扰动造成的间隔变化量: (A_0−A_1)·δ
    g_plain = float((A[0] - A[1]) @ delta)
    Ab = A + b[:, None] * torch.ones(1, K)
    g_biased = float((Ab[0] - Ab[1]) @ delta)
    ok &= check("E5c 扰动造成的间隔变化量与 b 无关（这才是 b 严格消去的地方）",
                abs(g_plain - g_biased) < 1e-5, f"{g_plain:.6f}")

    # E6: ISDA 的 (λ/2)r² 就是上式乘 λ/2；用真实方向集复核一次（与 U 的二次型一致）
    d_plain = A[0] - A[1]
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
