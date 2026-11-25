"""projected / residual / split 臂的单元测试 (2026-11-25, 外审第三轮要求的 Recombined 数值验证)。

不加载 CLIP、不训练, 只做张量级检查:
  T1 投影 + 残余 = 原始方向 (Recombined 恢复输入)
  T2 同一损失路径下, recombined 的损失与原始方向的损失逐位一致
  T3 projected / residual 的行没有被重新归一化 (残余行范数明显小于 1, 保留投影幅值)
  T4 _affine_linear 只做 LayerNorm 的 affine + Linear (与对已归一化行施加完整 LayerNorm 一致, 到 eps 量级)
  T5 split 的缩放: cat 两分量后 abs().mean() x 2 == 两个分量各自 abs().mean() 之和
  T6 投影算子幂等、对称; 主方向正交且在零均值超平面内
用法: python tests_projected_dirs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from model.cbm_models import clip_cbm_orth  # noqa: E402

torch.manual_seed(0)
K_DIM, N_ROWS, KS = 312, 500, (8, 32)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'OK ' if cond else 'BAD'} {name}" + (f"  [{detail}]" if detail else ""))


def ln0(x, eps=1e-5):
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)


A = torch.randn(N_ROWS, K_DIM) * 2 + 0.5
U = ln0(A)                                                    # 分解空间: LN0 后、affine 前 (行范数约 sqrt(311))
ln = nn.LayerNorm(K_DIM)
lin = nn.Linear(K_DIM, 7)
with torch.no_grad():                                         # 固定随机参数, 只测路径
    for p in list(ln.parameters()) + list(lin.parameters()):
        p.copy_(torch.randn_like(p) * 0.3)

with torch.no_grad():
    loss = lambda x: clip_cbm_orth._affine_linear(x, ln, lin).abs().mean()

for k in KS:
    V, _ = clip_cbm_orth._principal_rows(U, k)
    P = V.T @ V
    Um, Ur = U @ P, U - U @ P
    pre = f"k={k}: "
    check(pre + "主方向正交且在零均值超平面内",
          float((V @ V.T - torch.eye(k)).abs().max()) < 1e-4 and float(V.mean(-1).abs().max()) < 1e-4)
    check(pre + "投影算子对称且幂等",
          float((P - P.T).abs().max()) < 1e-4 and float((P @ P - P).abs().max()) < 1e-4)
    check(pre + "T1 投影 + 残余恢复原始方向", torch.allclose(Um + Ur, U, atol=1e-4),
          f"max|diff|={float((Um + Ur - U).abs().max()):.2e}")
    check(pre + "T2 recombined 的损失与原始方向一致 (同一损失路径)", abs(loss(Um + Ur) - loss(U)) < 1e-6,
          f"{loss(Um + Ur):.6f} vs {loss(U):.6f}")
    nr_u, nr_r = U.norm(dim=-1), Ur.norm(dim=-1)
    check(pre + "T3 保留投影幅值: 残余行范数不等于原始行范数且整体更小 (未被重新归一化)",
          float((nr_u - nr_r).min()) > 0 and float(nr_r.max()) < float(nr_u.max()),
          f"原始 mean={float(nr_u.mean()):.2f}, 残余 mean={float(nr_r.mean()):.2f}, 残余/原始比 {float((nr_r / nr_u).std()):.4f} (非常数)")
    with torch.no_grad():
        x = U[:1]
        check(pre + "T3b _affine_linear 不做归一化: 输入缩放 2 倍时输出线性变化 (LayerNorm 路径则不变)",
              not torch.allclose(clip_cbm_orth._affine_linear(2 * x, ln, lin), clip_cbm_orth._affine_linear(x, ln, lin))
              and torch.allclose(ln(2 * x), ln(x), atol=1e-5))
    check(pre + "T4 _affine_linear(U) = Linear(LayerNorm(U)) (对已归一化的行, 两条路径一致, 到 eps 量级)",
          torch.allclose(clip_cbm_orth._affine_linear(U, ln, lin), lin(ln(U)), atol=1e-3),
          f"max|diff|={float((clip_cbm_orth._affine_linear(U, ln, lin) - lin(ln(U))).abs().max()):.2e}")
    both = torch.cat([Um, Ur], 0)
    check(pre + "T5 split 缩放: cat 后 abs().mean() x 2 = 两分量之和",
          abs(2 * loss(both) - (loss(Um) + loss(Ur))) < 1e-6,
          f"2x{loss(both):.6f} vs {loss(Um):.6f}+{loss(Ur):.6f}")
    check(pre + "T6(补充) |A u| 与 |A P u| + |A res u| 的三角不等式关系成立 (一般不相等, 这正是 split 要回答的)",
          float(loss(U)) <= float(loss(Um)) + float(loss(Ur)) + 1e-6,
          f"full={loss(U):.4f} <= proj+res={float(loss(Um)) + float(loss(Ur)):.4f}")

# ---- bias-free 路径 (外审第四轮) ----
bf = lambda x: clip_cbm_orth._affine_linear(x, ln, lin, with_bias=False)
manual = (U * ln.weight) @ lin.weight.T                               # 手算 W·diag(gamma)·u (不含 beta 与 b)
check("BF1 with_bias=False 的输出 == W·diag(gamma)·u (不含 W·beta + b)",
      torch.allclose(bf(U), manual, atol=1e-4), f"max|diff|={float((bf(U) - manual).abs().max()):.2e}")
c = lin.bias[None, :] + ln.bias[None, :] @ lin.weight.T             # 有效偏置 W·beta + b
check("BF2 with_bias=True 的输出 == with_bias=False 的输出 + (W·beta + b)",
      torch.allclose(clip_cbm_orth._affine_linear(U, ln, lin, with_bias=True), bf(U) + c, atol=1e-4),
      f"max|diff|={float((clip_cbm_orth._affine_linear(U, ln, lin, with_bias=True) - bf(U) - c).abs().max()):.2e}")
check("BF3 偏置非零 (否则 bias-free 对照没有意义)", float(c.abs().max()) > 1e-3, f"max|c|={float(c.abs().max()):.3f}")
split_bias = float(clip_cbm_orth._affine_linear(torch.cat([Um, Ur], 0), ln, lin, with_bias=True).abs().mean() * 2)
split_bf = float(bf(torch.cat([Um, Ur], 0)).abs().mean() * 2)
check("BF4 split(偏置各计一次) != split_bf(无偏置) —— 二者之差就是外审指出的混杂因素",
      abs(split_bias - split_bf) > 1e-3, f"split={split_bias:.4f}, split_bf={split_bf:.4f}")
# 梯度层面: bias-free 正则对 beta 与 b 没有直接梯度 —— 它们根本不在计算图里 (autograd.grad 返回 None)
reg = clip_cbm_orth._affine_linear(U[:20], ln, lin, with_bias=False).abs().mean()
g_ln_w, g_ln_b, g_w, g_b = torch.autograd.grad(
    reg, [ln.weight, ln.bias, lin.weight, lin.bias], allow_unused=True)
check("BF5 bias-free 正则对 LayerNorm.beta 与 Linear.bias 没有直接梯度 (不在计算图里)",
      g_ln_b is None and g_b is None)
check("BF6 bias-free 正则对 gamma 与 W 仍有非零梯度", float(g_ln_w.abs().max()) > 0 and float(g_w.abs().max()) > 0)

print("TESTS_PROJECTED_DIRS:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
