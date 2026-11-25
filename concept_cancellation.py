"""
Stage 0B: 概念脆弱抵消诊断的指标 (规格见 doc/主线B实施规格.md §2)。零训练, 只读收敛分类头与域方向。

问题: DDO 只要求概念贡献的**加权和**为零 sum_j A_{c,j} u_j ~ 0。模型可以不放弃域敏感概念, 只给两个域敏感概念
相反的大权重, 让它们在文本域方向上互相抵消; 而语言域子空间对真实视觉偏移的覆盖只有 10%~26%,
到了真实目标域, 这种抵消会失效 (相位错位)。

指标 (w 是一个"决策方向"向量, 见下):
  CR(w)  = E_u[ sum_j |w_j u_j| ] / E_u[ |sum_j w_j u_j| ]        原始定义 (R); 分母接近 0 时不稳定
  CI(w)  = 1 - E_u|w^T u| / (E_u sum_j |w_j u_j| + eps)            有界版 (G3); 随机基线 ~ 1 - O(1/sqrt(K)), 高维饱和
  PCR(w) = w^T G w / w^T D~ w,   G = E[u u^T] (非中心化),  D~ = K/(K-1) H diag(G) H
           tr(D~) = tr(G) 精确 (迹守恒); 分子惩罚"相关方向的联合响应", 分母惩罚"对各概念绝对依赖的加权和"。
           若靠脆弱抵消把分子压低而分母仍大, PCR 会明显 < 1。
  PCR-Pair   : w = gamma ⊙ (W_y - W_k)   (单个竞争类别对的有效权重差)
  PCR-Global : 所有预定义类别对的分子、分母分别求和再取比
  随机经验基线: 同一个 G 上的各向同性随机权重 (超平面内) 的 PCR 分布。**不能用 1.0 当阈值**:
           各向同性时基线恰为 1, 但谱陡峭时 E[PCR] 系统性 < 1 (首方向占 90% 能量时 0.993),
           且协方差非对角时离散度很大 (5%~95% 可达 [0.92, 1.08])。判定用分位数, 不用 1.0。

"w 是什么": 单类 logit 版 w = A_c (= W_c ⊙ gamma); 决策版 w = gamma ⊙ (W_y - W_k)。公共 logit 分量在决策版中消去。

用法 (自检): python concept_cancellation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402


def second_moment(U):
    """G = E[u u^T] (非中心化), float64"""
    Ud = U.double()
    return Ud.T @ Ud / Ud.size(0)


def diag_reference(G):
    """D~ = K/(K-1) H diag(G) H。tr(D~) = tr(G) 对零均值超平面内的 u 精确成立。"""
    K = G.size(0)
    H = torch.eye(K, dtype=G.dtype, device=G.device) - torch.ones(K, K, dtype=G.dtype, device=G.device) / K
    return K / (K - 1) * H @ torch.diag(torch.diagonal(G)) @ H


def quad(w, M):
    """w: (..., K); 返回 (...,) 的 w^T M w。内部统一到 float64。"""
    return torch.einsum("...i,ij,...j->...", w.double(), M.double(), w.double())


def cr_ci(w, U, eps=1e-12, chunk=8192):
    """CR 与 CI。w: (M, K) 的决策方向; U: (N, K) 域方向。返回 (CR, CI, E|w^T u|, E sum_j |w_j u_j|), 各 (M,)。
    分块计算, 避免构造 (M, N, K)。"""
    w = w.double()
    K = U.size(1)
    Hm = torch.eye(K, dtype=torch.float64, device=U.device) - torch.ones(K, K, dtype=torch.float64, device=U.device) / K
    w = w @ Hm                                             # 决策方向也投影到超平面 (u 在 H 内, 分量上的公共部分无意义)
    abs_sum = torch.zeros(w.size(0), dtype=torch.float64, device=U.device)     # E sum_j |w_j u_j|
    abs_net = torch.zeros(w.size(0), dtype=torch.float64, device=U.device)     # E |w^T u|
    n = 0
    for s in range(0, U.size(0), chunk):
        u = U[s:s + chunk].double()                        # (n, K)
        contrib = u.unsqueeze(1) * w.unsqueeze(0)          # (n, M, K): 第 m 个决策方向在方向 u 上的逐概念贡献
        abs_sum += contrib.abs().sum(-1).sum(0)
        abs_net += contrib.sum(-1).abs().sum(0)
        n += u.size(0)
    E_sum, E_net = abs_sum / n, abs_net / n
    cr = E_sum / (E_net + eps)
    ci = 1.0 - E_net / (E_sum + eps)
    return cr, ci, E_net, E_sum


QS = (0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995)


def random_baseline(G, n=20000, seed=0):
    """同一个 G 上各向同性随机权重的 PCR 分布 (经验基线)。返回 (info dict, 排序后的样本张量)。

    info 的分位数键为 "q" + 三位千分 (q005, q010, ..., q995)；排序样本供 pcr_quantile 做经验 CDF。
    **基线可能很宽**: 分母参考 D~ = K/(K-1) H diag(G) H 用的是 diag(G); 若少数概念高度域敏感
    (真实 CUB: diag(G) 最大值是中位数的 10 倍), 随机权重在这些概念上多一点, PCR 就低很多。
    实测真实 CUB 的 G: 中位数 0.80, 5%~95% 为 [0.36, 2.34]。所以
      (a) 不能拿 1.0 当阈值; (b) 单个类别对的 PCR 几乎没有判别力, 要看一批类别对的分布或 PCR-Global。
    """
    K = G.size(0)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ws = torch.randn(K, n, generator=g, dtype=torch.float64)
    ws = (ws - ws.mean(0, keepdim=True)).to(G.device)      # 超平面内
    Dt = diag_reference(G)
    num, den = quad(ws.T, G), quad(ws.T, Dt).clamp_min(1e-300)
    pcr = num / den
    q = torch.quantile(pcr, torch.tensor(QS, dtype=torch.float64, device=G.device))
    out = {f"q{int(round(p * 1000)):03d}": float(v) for p, v in zip(QS, q.tolist())}
    out.update({"mean": float(pcr.mean()), "sd": float(pcr.std()), "n": int(pcr.numel()),
                "E_num_over_E_den": float(num.mean() / den.mean()),
                "diag_max_over_median": float(torch.diagonal(G).max() / torch.diagonal(G).median())})
    return out, torch.sort(pcr).values                      # 升序样本: 经验 CDF 用


def pcr_quantile(pcr_values, baseline_samples):
    """观测 PCR 在随机基线中的**经验分位** (越小越异常)。

    外审第六轮指出: 早期版本只保留 13 个离散分位点再向下映射, 真实分位 0.10~0.49 的值都会被标成
    0.005~0.10 —— "分位 <= 0.10" 实际纳入了远多于最低 10% 的对。正确做法是保留基线的全部样本,
    用经验 CDF: F(t) = (#{PCR_j <= t}) / N。baseline_samples 是 random_baseline 返回的升序张量。"""
    s = baseline_samples.to(pcr_values.device)
    return torch.searchsorted(s, pcr_values.double().reshape(-1), right=True).double().div(s.numel()).reshape(pcr_values.shape)


def pcr_pair(A, gam, pairs, G, Dt):
    """PCR-Pair。A: (C, K) = W ⊙ gamma; gam: (K,); pairs: LongTensor (P, 2) 的 (y, k)。
    决策方向 w = gamma ⊙ (W_y - W_k) = A_y - A_k (A 已含 gamma)。返回 dict of (P,) 张量。"""
    w = A[pairs[:, 0]] - A[pairs[:, 1]]
    num, den = quad(w, G), quad(w, Dt).clamp_min(1e-300)
    return {"w": w, "num": num, "den": den, "pcr": num / den}


def pcr_global(pairs_out):
    """PCR-Global: 所有预定义类别对的分子、分母分别求和再取比"""
    return float(pairs_out["num"].sum() / pairs_out["den"].sum())


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD ") + name + (f"  [{detail}]" if detail else ""))

    torch.manual_seed(0)
    K, N, C = 40, 20000, 9
    sens = [3, 7, 11, 19, 25]
    # 域方向: 域敏感概念随同一个潜变量**同相**变化 (画风一变, 颜色/质感类概念一起变), 其余概念几乎不变。
    # "脆弱抵消"只在同相结构下才成立: 两个同相变化的域敏感概念配 +a / -a 的权重, 加权和为 0 但绝对依赖大。
    # (若各概念独立变化, +a/-a 根本不构成抵消 —— 自测初版用了独立变化的数据, 指标正确地没有报抵消。)
    t = torch.randn(N, 1)
    U = torch.randn(N, K) * 0.05
    U[:, sens] = t * 2.0 + torch.randn(N, len(sens)) * 0.4
    U = U - U.mean(-1, keepdim=True)
    G = second_moment(U)
    Dt = diag_reference(G)
    check("T1 tr(D~) == tr(G)", abs(float(torch.trace(Dt) - torch.trace(G))) < 1e-9)
    check("T2 D~ 对称、半正定", float((Dt - Dt.T).abs().max()) < 1e-12 and float(torch.linalg.eigvalsh(Dt).min()) > -1e-12)
    check("T3 1^T D~ 1 = 0 (D~ 不泄漏到全 1 方向)", float((Dt.sum(0) ** 2).sum()) < 1e-16)

    # ---- 四个决策方向, 带已知答案 ----
    # 抵消型: 两个同相的域敏感概念 +6 / -6 -> 加权和 ~ 0, 但绝对依赖大     => PCR 低, 依赖能量高
    # 同号型: 同两个概念 +6 / +6 -> 相加, 无法抵消                       => PCR 正常或偏高, 依赖能量高
    # 放弃型: 只用非敏感概念                                             => PCR 也低, 但依赖能量低
    # 依赖型: 单个域敏感概念                                             => PCR 正常, 依赖能量中
    w_cancel = torch.zeros(K); w_cancel[3] = 6.0; w_cancel[7] = -6.0
    w_same = torch.zeros(K); w_same[3] = 6.0; w_same[7] = 6.0
    w_drop = torch.zeros(K); w_drop[0] = 6.0; w_drop[1] = -6.0; w_drop[2] = 3.0
    w_dep = torch.zeros(K); w_dep[3] = 6.0
    ws = torch.stack([w_cancel, w_same, w_drop, w_dep])
    ws = ws - ws.mean(-1, keepdim=True)
    cr, ci, E_net, E_sum = cr_ci(ws, U)
    num, den = quad(ws, G), quad(ws, Dt)
    pcr = num / den
    base, base_samples = random_baseline(G, n=20000, seed=0)
    # 依赖能量基线: 与 w 同范数的随机权重 (超平面内) 的分母 w^T D~ w 的期望, 用于区分"抵消型"与"放弃型"
    # (两者 PCR 都低; 区别在绝对依赖 —— 这正是 C 文件三态图里 状态 B 与 状态 C 的区别)
    g = torch.Generator().manual_seed(1)
    rw = torch.randn(K, 40000, generator=g, dtype=torch.float64)
    rw = rw - rw.mean(0, keepdim=True)
    rw = rw / rw.norm(dim=0, keepdim=True)                 # 单位范数
    den_unit = float(quad(rw.T, Dt).mean())                # 单位随机权重的期望分母
    dep_energy = (den.double() / (ws.double().norm(dim=1) ** 2)) / den_unit      # 相对同范数随机权重的依赖倍数
    print(f"  随机基线: 中位数 {base['q500']:.4f}, 5%~95% [{base['q050']:.3f}, {base['q950']:.3f}], E[PCR] {base['mean']:.4f}, sd {base['sd']:.4f}")
    print(f"  {'方向':8s} {'CR':>7s} {'CI':>7s} {'PCR':>7s} {'E|w^T u|':>9s} {'E sum|.|':>9s} {'依赖能量/随机':>12s}")
    for nm, i in (("抵消型", 0), ("同号型", 1), ("放弃型", 2), ("依赖型", 3)):
        print(f"  {nm:8s} {cr[i]:7.3f} {ci[i]:7.3f} {float(pcr[i]):7.3f} {E_net[i]:9.3f} {E_sum[i]:9.3f} {float(dep_energy[i]):12.2f}")
    # 判定用"在随机基线中的分位", 不用固定阈值: 基线可能很宽 (合成数据 5%~95% 跨 [0.05, 3.2]),
    # 这时抵消型的 PCR (0.048) 只落在基线的 ~5% 分位附近 —— 单看 PCR 的确很难把它与随机权重分开,
    # 必须结合"依赖能量"与"CI"一起看 (这正是 C 文件三态图 / G3 "PCR 不是越低越好"的含义)。
    qtl = pcr_quantile(pcr, base_samples)
    print(f"  在随机基线中的分位: 抵消型 {float(qtl[0]):.3f} | 同号型 {float(qtl[1]):.3f} | 放弃型 {float(qtl[2]):.3f} | 依赖型 {float(qtl[3]):.3f}")
    check("T4 抵消型: PCR 落在基线的低尾 (分位 < 0.10)", float(qtl[0]) < 0.10, f"PCR={float(pcr[0]):.3f}, 分位={float(qtl[0]):.3f}")
    check("T5 抵消型: CI 高 (加权和相对绝对和很小)", float(ci[0]) > 0.8, f"CI={float(ci[0]):.3f}")
    check("T6 抵消型: 依赖能量远高于同范数随机权重 (仍在重度依赖域敏感概念)", float(dep_energy[0]) > 3.0, f"{float(dep_energy[0]):.1f}x")
    check("T7 放弃型: 依赖能量低于随机; PCR 分位低于中位但不必在极端低尾 (旧断言 <0.20 是按有偏离散分位校准的)",
          float(qtl[2]) < 0.30 and float(dep_energy[2]) < 1.0, f"PCR={float(pcr[2]):.3f}, 分位={float(qtl[2]):.3f}, 依赖={float(dep_energy[2]):.2f}x")
    check("T7b 抵消型的分位远低于放弃型 (极端低尾是抵消型的特征)", float(qtl[0]) < 0.25 * float(qtl[2]),
          f"抵消型 {float(qtl[0]):.3f} vs 放弃型 {float(qtl[2]):.3f}")
    check("T8 抵消型与放弃型靠'依赖能量'区分开 (单看 PCR 分不开)", float(pcr[0]) < 0.3 and float(pcr[2]) < 0.3
          and float(dep_energy[0]) > 5 * float(dep_energy[2]))
    check("T9 同号型: PCR 高于基线中位数 (同相相加, 无抵消)", float(pcr[1]) > base["q500"], f"{float(pcr[1]):.3f}")
    check("T10 依赖型: PCR 在基线 2.5%~97.5% 内", base["q025"] < float(pcr[3]) < base["q975"], f"{float(pcr[3]):.3f}")
    check("T11 CR 与 CI 满足 CR = 1/(1-CI)", torch.allclose(cr, 1.0 / (1.0 - ci + 1e-12), rtol=1e-3))
    check("T12 抵消型的 CI 是四者中最高的", int(ci.argmax()) == 0)
    # ---- PCR-Pair / PCR-Global ----
    Wt = torch.zeros(C, K)
    Wt[0, 3] = 5.0; Wt[0, 7] = -5.0                        # 类 0: 抵消型
    Wt[1, 3] = 5.0; Wt[1, 7] = 5.0                         # 类 1: 同号型
    Wt[2, 0] = 5.0                                        # 类 2: 放弃型
    gam = torch.ones(K)
    A = Wt * gam
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
    pp = pcr_pair(A, gam, pairs, G, Dt)
    check("T13 PCR-Pair: 决策方向 = A_y - A_k", torch.allclose(pp["w"], A[pairs[:, 0]] - A[pairs[:, 1]]))
    check("T14 PCR-Pair(0,2) = 抵消型 − 放弃型 => 抵消结构仍在 => 低 PCR", float(pp["pcr"][1]) < 0.3, f"{float(pp['pcr'][1]):.3f}")
    check("T15 PCR-Pair(1,2) = 同号型 − 放弃型 => 无抵消 => PCR 正常", float(pp["pcr"][2]) > base["q250"], f"{float(pp['pcr'][2]):.3f}")
    gl = pcr_global(pp)
    check("T16 PCR-Global = 分子和 / 分母和 (介于各对的 min 与 max 之间)", float(pp["pcr"].min()) <= gl <= float(pp["pcr"].max()), f"{gl:.3f}")
    check("T17 报告了分子与分母 (不只报比值)", "num" in pp and "den" in pp and float(pp["num"].min()) >= 0)
    b2info, b2s = random_baseline(G, n=20000, seed=0)
    check("T18 随机基线可复现 (info 与样本都相同)", b2info == base and torch.equal(b2s, base_samples))
    # 经验分位的精确性: 基线样本自身的分位应精确返回 (外审第六轮: 旧的离散映射把真实 0.10~0.49 都标成 <=0.10)
    probe = torch.tensor([0.05, 0.10, 0.20, 0.49, 0.75], dtype=torch.float64)
    got = pcr_quantile(torch.quantile(base_samples, probe), base_samples)
    check("T18b 经验分位精确 (探针分位与返回一致, 容差 0.01)", float((got - probe).abs().max()) < 0.01,
          f"probe={probe.tolist()}, got={[round(float(v), 3) for v in got]}")
    b_other, _ = random_baseline(G, n=20000, seed=7)
    check("T19 不同 seed 的基线不同 (是经验分布, 不是常数 1)", b_other["q050"] != base["q050"])
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
