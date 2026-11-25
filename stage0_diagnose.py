"""
Stage 0A / 0B 第一层的诊断 (预注册见 RESULTS.md §3r; 指标见 concept_cancellation.py; 规格见 doc/主线B实施规格.md)。
零训练: 只读 run_matrix.py stage0 保存的收敛分类头 (logs/heads/CUB_s0_*_seed*.pt)、域方向、源域训练集缓存特征。
**判定全部由本脚本按 §3r 的常量从数据算出**; 只用源域数据, 不读取任何目标域信息。

2026-11-25 外审第六轮后的更正 (判定与阈值本身不变, 指标实现与判定执行修正):
  1. 0A 的 rho/cos/幅值比此前实际是**全局标量** (对 (C,) 向量整体求范数), 却被报告成"逐类中位数"。
     现在两者都算并分开命名: *_global (整体) 与逐类分布 (带有效性掩码 |A·mu|_c > 1e-3·max, 避免除近零)。
     §3r 的判定用逐类中位数 (预注册原文"对全部 C 个类")。
  2. §3r 预注册的降级条件此前没有执行: "CE-Only 与 DDO 的幅值比相差 10 倍以上 -> 判定降级为不确定"。
     现在写入 verdicts()。实际数据 CE 0.040 对 Full 0.753/0.874 (约 19~22 倍), 条件成立。
  3. PCR 分位改用**经验 CDF** (外审第六轮: 旧的 13 点离散映射把真实分位 0.10~0.49 的值都标成 <=0.10)。
  4. CI 补充**精确值** (分块 |U @ W^T| 均值; 三角不等式保证它在 [0,1]); 旧的 RMS 版保留为 CI_RMS 并明确命名。
  5. 依赖能量补充绝对量: PCR 的分子 w^T G w、分母 w^T D~ w、||w||^2 与 PCR-Global, 不只报比值。
  6. 统计口径如实标注: 全部无序类别对 (分类头整体几何诊断), 不是 §3r 写的"样本x竞争类" —— 样本级
     决策诊断留待 0B 第二层 (需要源域增强特征)。

用法: python stage0_diagnose.py [--selftest]
产物: experiment_results/Stage0-Diagnostics/stage0_cub.json、stage0_cub.md
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from concept_cancellation import diag_reference, pcr_quantile, quad, random_baseline  # noqa: E402
from language_ellipsoid import domain_directions, second_moments  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage0-Diagnostics")
SEEDS = (1, 2, 3)
ARMS = [("ce_a0", "CE-Only（α=0）"), ("full_a1", "Full DDO（α=1）"), ("full_a2", "Full DDO（α=2）"),
        ("mean_a1", "Mean（α=1）"), ("mean_a2", "Mean（α=2）"), ("sym_a1", "Full-Symmetric（α=1）"),
        ("sym_a2", "Full-Symmetric（α=2）"), ("bf_a1", "Bias-Free（α=1）"), ("bf_a2", "Bias-Free（α=2）")]
FULL_ARMS, CE_ARM = ("full_a1", "full_a2"), "ce_a0"

# ---- 预注册的阈值 (RESULTS §3r); 结果出来之后不改 ----
A_EXIST_MIN, A_EXIST_GAP, A_NONE_MAX, A_NONE_GAP = 0.50, 0.25, 0.25, 0.10
A_DOWNGRADE_AMP_RATIO = 10.0        # §3r: CE 与 DDO 的幅值比相差超过此倍数 -> 判定降级为"不确定"
B_QTL_MAX, B_CI_MIN, B_DEP_MIN = 0.10, 0.80, 3.0
B_EXIST_RATIO, B_EXIST_MIN_ABS = 2.0, 0.05
B_NONE_MAX_DIFF = 0.02
CLASS_VALID_REL = 1e-3              # 逐类有效性掩码: |A·mu|_c > 1e-3 * max_c |A·mu|_c


def load_heads():
    out = {}
    for tag, _ in ARMS:
        for s in SEEDS:
            p = f"logs/heads/CUB_s0_{tag}_seed{s}.pt"
            if os.path.exists(p):
                out[(tag, s)] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def verify_heads(heads):
    """收敛分类头对应的 run 必须完整, 且 (核对项) 精度与已有的正式结果逐位相同"""
    checks = []
    # Stage 0 的 tag -> 对应的正式结果 tag (同配置; Stage 0 只是多加了 --save_head, 不改变训练的随机数消耗)
    pairs = dict([("full_a1", "dose_a1"), ("full_a2", "dose_a2"), ("ce_a0", "dose_a0"), ("mean_a1", "sub_mean_a1"),
                  ("mean_a2", "sub_mean_a2"), ("sym_a1", "sub_fullsym_a1"), ("sym_a2", "sub_fullsym_a2"),
                  ("bf_a1", "bf_full8_a1"), ("bf_a2", "bf_full8_a2")])
    n_ok = n_cmp = n_same = 0
    for (tag, s), h in sorted(heads.items()):
        rp = f"logs/results/CUB_s0_{tag}_seed{s}.json"
        r = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else None
        complete = bool(r) and r.get("status") == "completed" and r.get("completed_epochs") == 100
        n_ok += complete
        ref_tag = pairs.get(tag)
        if ref_tag:
            ref_p = f"logs/results/CUB_{ref_tag}_seed{s}.json"
            same = False
            if os.path.exists(ref_p) and r is not None:    # 参照或本 run 缺失都算未通过, 不静默放过
                ref = json.load(open(ref_p, encoding="utf-8"))
                same = (ref["final_diag"] == r["final_diag"])
            n_cmp += 1
            n_same += same
    checks.append(("27 个 Stage 0 run 都完整结束 (status / 100 epochs)", n_ok == 27 and len(heads) == 27,
                   f"{n_ok} / 27"))
    checks.append(("27 个重训 run 的 final_diag 与对应的正式结果逐位相同", n_cmp == 27 and n_same == 27,
                   f"{n_same} / {n_cmp}"))
    return checks


def exact_ci(U, Wk, abs_u, chunk=2048, device=None):
    """精确 CI = 1 - E|w^T u| / (E sum_j |w_j u_j| + eps), 对全部 P 个决策方向一次算出。
    E|w^T u| 用分块 |U @ W^T| 的均值 (外审第六轮建议); 三角不等式保证精确 CI 恒在 [0, 1]。
    返回 (CI (P,), 分子 E|w^T u| (P,), 分母 (P,))。"""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Uc = U.to(dev, torch.float32)
    Wc = Wk.to(dev, torch.float32)
    num = torch.zeros(Wk.size(0), dtype=torch.float64)
    for s in range(0, U.size(0), chunk):
        num += (U[s:s + chunk].to(dev, torch.float32) @ Wc.T).abs().sum(dim=0, dtype=torch.float64).cpu()
    num /= U.size(0)
    den = (Wk.abs().double() @ abs_u).clamp_min(1e-300)
    return 1.0 - num / den, num, den


def diagnose(heads, U, device=None):
    G, G_cov, mu, eta = second_moments(U)
    Dt = diag_reference(G)
    base, base_samples = random_baseline(G, n=40000, seed=0)
    K = U.size(1)
    abs_u = U.abs().mean(0)                                   # E|u_j| (逐概念)
    Hm = torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    g = torch.Generator().manual_seed(1)
    rw = torch.randn(K, 40000, generator=g, dtype=torch.float64)
    rw = rw - rw.mean(0, keepdim=True)
    rw = rw / rw.norm(dim=0, keepdim=True)
    den_unit = float(quad(rw.T, Dt).mean())                   # 单位范数随机权重的期望分母
    out = {"eta_mean": eta, "random_baseline": base, "den_unit": den_unit, "arms": {}}
    for tag, name in ARMS:
        per_seed = []
        for s in SEEDS:
            h = heads.get((tag, s))
            if h is None:
                continue
            hp = h["classifier"]
            W, gam, beta, b = hp["2.weight"].double(), hp["1.weight"].double(), hp["1.bias"].double(), hp["2.bias"].double()
            A = W * gam[None, :]
            c = W @ beta + b
            # ---------- 0A: 全局指标 (对整个 (C,) 向量) ----------
            Amu = A @ mu                                       # (C,)
            rho_g = float(1.0 - (Amu + c).norm() / Amu.norm())
            cos_g = float(torch.cosine_similarity(c, -Amu, dim=0))
            amp_g = float(c.norm() / Amu.norm())
            # ---------- 0A: 逐类分布 (§3r 判定用; 带有效性掩码) ----------
            valid = Amu.abs() > (Amu.abs().max() * CLASS_VALID_REL)
            rho_c = 1.0 - (Amu + c).abs()[valid] / Amu.abs()[valid]
            # ---------- 0B: 全部无序类别对 (分类头整体几何; 不是样本级决策) ----------
            y_idx, k_idx = torch.triu_indices(A.size(0), A.size(0), offset=1)
            Wk = (A[y_idx] - A[k_idx]) @ Hm                    # (P, K)
            num_q, den_q = quad(Wk, G), quad(Wk, Dt).clamp_min(1e-300)
            pcr = num_q / den_q
            qtl = pcr_quantile(pcr, base_samples)              # 经验分位 (外审第六轮修正)
            ci_exact, num_ci, den_ci = exact_ci(U, Wk, abs_u, device=device)
            ci_rms = 1.0 - torch.sqrt(num_q.clamp_min(0)) / ((Wk.abs() @ abs_u) + 1e-12)
            dep = (den_q / Wk.norm(dim=-1) ** 2) / den_unit    # 相对依赖 (单位权重能量)
            cancelling = (qtl <= B_QTL_MAX) & (ci_exact >= B_CI_MIN) & (dep >= B_DEP_MIN)
            per_seed.append({
                "seed": s,
                "rho_abs_global": rho_g, "cos_global": cos_g, "amp_ratio_global": amp_g,
                "rho_abs_class_median": float(rho_c.median()),
                "rho_abs_class_q10": float(rho_c.quantile(0.10)), "rho_abs_class_q90": float(rho_c.quantile(0.90)),
                "n_valid_classes": int(valid.sum()), "n_classes": int(A.size(0)),
                "class_amp_ratio_median": float(((c.abs() / Amu.abs())[valid]).median()),
                "n_pairs": int(Wk.size(0)),
                "cancel_frac": float(cancelling.double().mean()),
                "cancel_frac_pcr_only": float((qtl <= B_QTL_MAX).double().mean()),
                "cancel_frac_ci_only": float((ci_exact >= B_CI_MIN).double().mean()),
                "median_pcr_quantile": float(qtl.median()), "median_ci_exact": float(ci_exact.median()),
                "median_ci_rms": float(ci_rms.median()),
                "median_dep_energy": float(dep.median()),
                "median_num_Gw": float(num_q.median()), "median_den_Dw": float(den_q.median()),
                "median_wnorm2": float((Wk.norm(dim=-1) ** 2).median()),
                "pcr_global": float(num_q.sum() / den_q.sum()),
            })
        if not per_seed:                                      # 该臂没有任何收敛分类头: 记录为缺失, 不崩
            out["arms"][tag] = {"name": name, "per_seed": [], "agg": None}
            continue
        agg = {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
               for k in per_seed[0] if k != "seed"}
        out["arms"][tag] = {"name": name, "per_seed": per_seed, "agg": agg}
    return out


def verdicts(res):
    a = res["arms"]

    def med(tag, key):
        return a[tag]["agg"][key][0]
    # ---- 0A (判定用逐类中位数, §3r 原文"对全部 C 个类"; 幅值比降级用全局幅值比, §3r 公式) ----
    full_med = float(np.mean([med(t, "rho_abs_class_median") for t in FULL_ARMS]))
    ce_med = med(CE_ARM, "rho_abs_class_median")
    gap = full_med - ce_med
    # 双向 (外审第七轮: §3r 原文"相差 10 倍以上"是双向的; 场景1 的 CE 幅值比 67.6 >> Full 0.81 也应触发)
    amp_full = max(med(t, "amp_ratio_global") for t in FULL_ARMS)
    amp_ce = med(CE_ARM, "amp_ratio_global")
    if amp_ce > 0 and amp_full > 0:
        amp_ratio = max(amp_full / amp_ce, amp_ce / amp_full)     # 双向取大
    else:
        amp_ratio = float("inf")                                   # 零幅值比: 无法比较, 视为触发
    if amp_ratio > A_DOWNGRADE_AMP_RATIO:                     # §3r 预注册的降级条件 (外审第六轮前未执行)
        va = "不确定（按 §3r 降级：CE 与 DDO 的幅值比相差 >10×）"
    elif full_med >= A_EXIST_MIN and gap >= A_EXIST_GAP:
        va = "偏置吸收存在"
    elif full_med < A_NONE_MAX or gap < A_NONE_GAP:
        va = "偏置吸收不存在"
    else:
        va = "不确定"
    # ---- 0B 第一层 ----
    f_frac = [p["cancel_frac"] for t in FULL_ARMS for p in a[t]["per_seed"]]
    c_frac = [p["cancel_frac"] for p in a[CE_ARM]["per_seed"]]
    fm, cm = float(np.mean(f_frac)), float(np.mean(c_frac))
    # 外审第七轮: 0/0 不是 inf —— 分子分母都为 0 时比值记 None ("不适用"); 只有分母为 0 才是 inf
    if cm > 0:
        ratio = fm / cm
    elif fm > 0:
        ratio = float("inf")
    else:
        ratio = None
    dirs = []
    for t in FULL_ARMS:
        for p in a[t]["per_seed"]:
            cs = [q["cancel_frac"] for q in a[CE_ARM]["per_seed"] if q["seed"] == p["seed"]]
            if cs:
                dirs.append(p["cancel_frac"] > cs[0])
    consistent = all(dirs) and len(dirs) >= 4
    if ratio is not None and ratio >= B_EXIST_RATIO and consistent and fm >= B_EXIST_MIN_ABS:
        vb = "现象存在"
    elif abs(fm - cm) < B_NONE_MAX_DIFF or not consistent:
        vb = "现象不存在"
    else:
        vb = "不确定"
    return {"0A": {"full_class_median": full_med, "ce_class_median": ce_med, "gap": gap,
                   "amp_ratio_full": amp_full, "amp_ratio_ce": amp_ce, "amp_ratio": amp_ratio,
                   "downgraded": amp_ratio > A_DOWNGRADE_AMP_RATIO, "verdict": va},
            "0B": {"full_frac": fm, "ce_frac": cm, "ratio": ratio, "n_seed_pairs": len(dirs),
                   "all_directions_consistent": consistent, "verdict": vb}}


def self_test():
    """在真实数据之前, 用已知答案的合成分类头验证判定逻辑 (含外审第六轮要求补的降级路径)。"""
    torch.manual_seed(0)
    K, C, N = 60, 30, 8000
    sens = [3, 7, 11, 19, 25, 31]
    tt = torch.randn(N, 1)
    U = torch.randn(N, K) * 0.05
    U[:, sens] = tt * 2.0 + torch.randn(N, len(sens)) * 0.4
    U = (U - U.mean(-1, keepdim=True)).double()
    G, _, mu, _ = second_moments(U)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD ") + name + (f"  [{detail}]" if detail else ""))

    def make_heads(ce_mode):
        heads = {}
        for tag in ("ce_a0", "full_a1", "full_a2"):
            for s in SEEDS:
                W = torch.randn(C, K, dtype=torch.float64) * 0.5
                gam = torch.ones(K, dtype=torch.float64)
                if tag != "ce_a0":
                    W[: C // 2, 3] = 6.0; W[: C // 2, 7] = -6.0      # 一半类别: 抵消型权重
                A = W * gam[None, :]
                if tag == "ce_a0":
                    if ce_mode == "big_orthogonal":                   # 大偏置但与 A·mu 正交 (不触发降级)
                        c = torch.randn(C, dtype=torch.float64)
                        Amu = A @ mu
                        c = c - (c @ Amu) / (Amu @ Amu) * Amu
                    else:                                             # tiny_bias: 触发 §3r 降级 (相对该头自身的 A·mu 缩放)
                        c = 0.05 * (A @ mu) * torch.randn(C, dtype=torch.float64)
                else:
                    Amu_h = A @ mu
                    c = -0.8 * Amu_h + 0.1 * Amu_h * torch.randn(C, dtype=torch.float64)   # 吸收 80% + 10% 逐类扰动
                heads[(tag, s)] = {"classifier": {"2.weight": W, "1.weight": gam,
                                                  "1.bias": torch.zeros(K, dtype=torch.float64), "2.bias": c}}
        return heads

    def run(ce_mode):
        res = diagnose(make_heads(ce_mode), U, device="cpu")
        return res, verdicts(res)

    # ---- 场景 1: 大而正交的 CE 偏置 -> 0A "存在", 0B "现象存在" ----
    res, v = run("big_orthogonal")
    fa = res["arms"]["full_a1"]["agg"]
    ca = res["arms"]["ce_a0"]["agg"]
    print(f"  [场景1] Full: 逐类 rho 中位 {fa['rho_abs_class_median'][0]:.3f}, 全局 rho {fa['rho_abs_global'][0]:.3f}, "
          f"幅值比 {fa['amp_ratio_global'][0]:.2f} | CE: 逐类 rho {ca['rho_abs_class_median'][0]:.3f}, "
          f"幅值比 {ca['amp_ratio_global'][0]:.2f} | 抵消型比例 {fa['cancel_frac'][0] * 100:.1f}%")
    # 场景1 的 CE 是 big_orthogonal (幅值比 ~67), 双向判据下也触发降级 (外审第七轮的发现)
    for name, cond in (("S1 0A 判定为'不确定' (双向幅值比: CE 67.6 vs Full 0.81, 相差 83 倍)", v["0A"]["verdict"].startswith("不确定")),
                       ("S2 全局与逐类指标都已计算且数值不同 (分开命名有意义)",
                        fa["rho_abs_global"][0] != fa["rho_abs_class_median"][0]),
                       ("S3 0B 判定为'现象存在'", v["0B"]["verdict"] == "现象存在"),
                       ("S4 精确 CI 在 [0,1] 内", 0.0 <= fa["median_ci_exact"][0] <= 1.0),
                       ("S5 精确 CI 高于 CI_RMS (RMS >= E|.|)", fa["median_ci_exact"][0] >= fa["median_ci_rms"][0] - 1e-9),
                       ("S6 绝对依赖量已记录 (分子/分母/权重范数/PCR-Global)",
                        all(k in fa for k in ("median_num_Gw", "median_den_Dw", "median_wnorm2", "pcr_global")))):
        print(("  OK  " if cond else "  BAD ") + name)
        ok &= cond
    # ---- 场景 2: CE 偏置极小 -> 幅值比相差 >10x -> 0A 按 §3r 降级为'不确定' ----
    res2, v2 = run("tiny_bias")
    a2 = res2["arms"]["full_a1"]["agg"]["amp_ratio_global"][0] / res2["arms"]["ce_a0"]["agg"]["amp_ratio_global"][0]
    print(f"  [场景2] CE 偏置极小: 幅值比 Full/CE = {a2:.1f}x -> 判定 {v2['0A']['verdict']}")
    for name, cond in (("S7 幅值比 > 10x 被检出", a2 > 10.0),
                       ("S8 0A 判定按 §3r 降级为'不确定'", v2["0A"]["verdict"].startswith("不确定")),
                       ("S9 降级不依赖 rho 的取值 (即使 rho 很高也降级)", res2["arms"]["full_a1"]["agg"]["rho_abs_class_median"][0] > 0.5)):
        print(("  OK  " if cond else "  BAD ") + name)
        ok &= cond
    # ---- 场景 3: 干净头自比 -> 两项都'不存在' ----
    heads_clean = {(tag, s): make_heads("big_orthogonal")[("ce_a0", s)] for tag in ("ce_a0", "full_a1", "full_a2") for s in SEEDS}
    v3 = verdicts(diagnose(heads_clean, U, device="cpu"))
    print(f"  [场景3] 干净头自比: 0A = {v3['0A']['verdict']}, 0B = {v3['0B']['verdict']}")
    for name, cond in (("S10 [反向] 0A 判为'不存在' (无降级, 同臂幅值比 = 1)", v3["0A"]["verdict"] == "偏置吸收不存在"),
                       ("S11 [反向] 0B 判为'现象不存在'", v3["0B"]["verdict"] == "现象不存在")):
        print(("  OK  " if cond else "  BAD ") + name)
        ok &= cond
    # ---- verify_heads: 在临时目录里造假结果, 走通"通过"与"不通过"两条路 ----
    import tempfile
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        os.makedirs("logs/results", exist_ok=True)
        fake = lambda **kw: {"status": "completed", "completed_epochs": 100, "final_diag": {"target_acc": 1.0}, **kw}
        names = {"full_a1": "dose_a1", "full_a2": "dose_a2", "ce_a0": "dose_a0", "mean_a1": "sub_mean_a1", "mean_a2": "sub_mean_a2",
                 "sym_a1": "sub_fullsym_a1", "sym_a2": "sub_fullsym_a2", "bf_a1": "bf_full8_a1", "bf_a2": "bf_full8_a2"}
        hd = {}
        for tag, ref in names.items():
            for s in SEEDS:
                json.dump(fake(), open(f"logs/results/CUB_s0_{tag}_seed{s}.json", "w"))
                json.dump(fake(), open(f"logs/results/CUB_{ref}_seed{s}.json", "w"))
                hd[(tag, s)] = {}
        good = verify_heads(hd)
        c12 = len(good) == 2 and all(c for _, c, _ in good)
        print(("  OK  " if c12 else "  BAD ") + "S12 verify_heads: 27 个一致 -> 全部通过")
        ok &= c12
        json.dump(fake(final_diag={"target_acc": 2.0}), open("logs/results/CUB_s0_full_a1_seed1.json", "w"))
        c13 = any(not c for _, c, _ in verify_heads(hd))
        print(("  OK  " if c13 else "  BAD ") + "S13 verify_heads: 有一个不一致 -> 报未通过 (不静默放过)")
        ok &= c13
        os.remove("logs/results/CUB_dose_a2_seed3.json")
        c14 = any(not c for _, c, _ in verify_heads(hd))
        print(("  OK  " if c14 else "  BAD ") + "S14 verify_heads: 参照文件缺失 -> 报未通过")
        ok &= c14
        os.chdir(cwd)
    print("STAGE0_SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        return self_test()
    heads = load_heads()
    checks = verify_heads(heads)
    ok = all(c for _, c, _ in checks)
    for n, c, d in checks:
        print(("  OK  " if c else "  BAD ") + n + (f"  [{d}]" if d else ""))
    if not ok:
        print("STAGE0_DIAGNOSE: 核对未通过, 不进入诊断")
        return 1
    from args import get_args
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = __import__("data").get_dataset_classes(args)[0]
    diffs = ds.domain_diffs.double().cpu()
    import clip
    m, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=args.device)
    with torch.no_grad():
        E = m.encode_text(clip.tokenize(list(ds.concept2id.keys())).to(args.device)).double().cpu()
        E = E / E.norm(dim=-1, keepdim=True)
    del m
    U = domain_directions(diffs, E)
    device = torch.device(args.device)
    res = diagnose(heads, U, device=device)
    res["checks"] = [{"check": n, "ok": bool(c), "detail": d} for n, c, d in checks]
    res["thresholds"] = {"A_EXIST": [A_EXIST_MIN, A_EXIST_GAP], "A_NONE": [A_NONE_MAX, A_NONE_GAP],
                         "A_DOWNGRADE_AMP_RATIO": A_DOWNGRADE_AMP_RATIO,
                         "B_cancelling": [B_QTL_MAX, B_CI_MIN, B_DEP_MIN],
                         "B_EXIST": [B_EXIST_RATIO, B_EXIST_MIN_ABS], "B_NONE_MAX_DIFF": B_NONE_MAX_DIFF}
    res["corrections_6th_review"] = ("外审第六轮后的更正: 全局/逐类指标分开; §3r 幅值比降级条件已执行; "
                                     "PCR 分位改经验 CDF; CI 补精确值 (CI_RMS 另名); 依赖能量补绝对量; "
                                     "统计口径为全部无序类别对 (非样本级)。判定阈值本身未改。")
    res["verdicts"] = verdicts(res)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stage0_cub.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    def md():
        b = res["random_baseline"]
        L = ["# Stage 0A / 0B 第一层诊断（CUB，3 seeds；2026-11-25，含外审第六轮更正后的重算）", "",
             "预注册见 `RESULTS.md` §3r；判定全部由 `stage0_diagnose.py` 按预注册阈值算出（阈值未改）。"
             "更正内容：全局/逐类指标分开命名；**§3r 的幅值比降级条件本次已执行**；PCR 分位改经验 CDF；"
             "CI 补精确值（RMS 版另名 CI_RMS）；依赖能量补绝对量；统计口径为全部无序类别对（非样本级）。只用源域数据。", "",
             "## 核对", "", "| 核对项 | 结果 | 说明 |", "|---|---|---|"]
        for n, c, d in checks:
            L.append(f"| {n} | {'通过' if c else '**未通过**'} | {d} |")
        L += ["", f"域方向均值能量占比 η_mean = {res['eta_mean']:.4f}。PCR 随机基线（同一个 G，40000 个超平面内随机权重）："
              f"中位数 {b['q500']:.3f}，5%～95% [{b['q050']:.3f}, {b['q950']:.3f}]；diag(G) 最大值 / 中位数 = {b['diag_max_over_median']:.1f}。", "",
              "## 0A 偏置吸收", "",
              "| 臂 | ρ_abs 逐类中位数 | ρ_abs 全局 | cos(c, −A·μ) 全局 | ‖c‖/‖A·μ‖ 全局 | 有效类数 |", "|---|---|---|---|---|---|"]
        for tag, name in ARMS:
            if tag in res["arms"]:
                g = res["arms"][tag]["agg"]
                L.append(f"| {name} | {g['rho_abs_class_median'][0]:.3f} ± {g['rho_abs_class_median'][1]:.3f} | "
                         f"{g['rho_abs_global'][0]:.3f} | {g['cos_global'][0]:.3f} | {g['amp_ratio_global'][0]:.3f} | "
                         f"{g['n_valid_classes'][0]:.0f}/{g['n_classes'][0]:.0f} |")
        v = res["verdicts"]["0A"]
        L += ["", f"**判定：{v['verdict']}**（逐类中位数 Full {v['full_class_median']:.3f} 对 CE-Only {v['ce_class_median']:.3f}，"
              f"差 {v['gap']:+.3f}；幅值比 Full {v['amp_ratio_full']:.3f} 对 CE {v['amp_ratio_ce']:.3f}，"
              f"相差 {v['amp_ratio']:.1f}× —— §3r 预注册的降级条件（>10×）{'成立，按纪律降级' if v['downgraded'] else '不成立'}）。", "",
              "## 0B 第一层：抵消型类别对的比例（全部无序类别对，非样本级）", "",
              f"\"抵消型\" = PCR 经验分位 ≤ {B_QTL_MAX} 且 精确 CI ≥ {B_CI_MIN} 且 相对依赖能量 ≥ {B_DEP_MIN}× 随机。", "",
              "| 臂 | 抵消型比例 | 分位 ≤ 0.10 | 精确 CI ≥ 0.8 | 分位中位数 | 精确 CI 中位 | CI_RMS 中位 | 相对依赖中位 |",
              "|---|---|---|---|---|---|---|---|"]
        for tag, name in ARMS:
            if tag in res["arms"]:
                g = res["arms"][tag]["agg"]
                L.append(f"| {name} | **{g['cancel_frac'][0] * 100:.2f}%** | {g['cancel_frac_pcr_only'][0] * 100:.2f}% | "
                         f"{g['cancel_frac_ci_only'][0] * 100:.1f}% | {g['median_pcr_quantile'][0]:.3f} | "
                         f"{g['median_ci_exact'][0]:.3f} | {g['median_ci_rms'][0]:.3f} | {g['median_dep_energy'][0]:.2f} |")
        v = res["verdicts"]["0B"]
        ratio_str = "不适用（两者都为 0%）" if v["ratio"] is None else f"{v['ratio']:.2f}"
        L += ["", f"**判定：{v['verdict']}**（Full {v['full_frac'] * 100:.2f}%，CE-Only {v['ce_frac'] * 100:.2f}%，"
              f"比值 {ratio_str}；{v['n_seed_pairs']} 个 seed 配对方向一致 = {v['all_directions_consistent']}）。", "",
              "## 绝对量（外审第六轮要求补充；相对依赖 0.93 附近的解读需要这些）", "",
              "| 臂 | wᵀGw 中位 | wᵀD̃w 中位 | ‖w‖² 中位 | PCR-Global |", "|---|---|---|---|---|"]
        for tag, name in ARMS:
            if tag in res["arms"]:
                g = res["arms"][tag]["agg"]
                L.append(f"| {name} | {g['median_num_Gw'][0]:.4f} | {g['median_den_Dw'][0]:.4f} | "
                         f"{g['median_wnorm2'][0]:.2f} | {g['pcr_global'][0]:.4f} |")
        L += ["", "## 逐 seed", "",
              "| 臂 | seed | ρ_abs 逐类中位 | ρ_abs 全局 | 抵消型比例 | 分位中位 | 精确 CI 中位 |", "|---|---|---|---|---|---|---|"]
        for tag, name in ARMS:
            for p in res["arms"].get(tag, {"per_seed": []})["per_seed"]:
                L.append(f"| {name} | {p['seed']} | {p['rho_abs_class_median']:.3f} | {p['rho_abs_global']:.3f} | "
                         f"{p['cancel_frac'] * 100:.2f}% | {p['median_pcr_quantile']:.3f} | {p['median_ci_exact']:.3f} |")
        L += ["", "## 读法", "",
              "- 0A 判定用逐类中位数（§3r 原文），全局指标并列报告；两者的数值差异本身说明\"偏置吸收在类间不均匀\"。",
              "- 幅值比降级条件来自 §3r 的预先规定：CE 与 DDO 的偏置量级差一个数量级以上时，ρ_abs 之差不能单独作为吸收证据。",
              "- 0B 的比例是 C(C−1)/2 个类别对中\"抵消型\"的占比（分类头整体几何，非样本级决策）；单个类别对的 PCR 没有判别力。",
              "- 第一层只回答\"现象是否存在\"（且按预注册判据此处不说明问题——见 RESULTS §3s 更正）。"
              "是否在源域视觉扰动下失效（第二层）、能否改善（第三层）需要各自的实验。", ""]
        return "\n".join(L)

    with open(os.path.join(OUT_DIR, "stage0_cub.md"), "w", encoding="utf-8") as f:
        f.write(md())
    print(json.dumps(res["verdicts"], ensure_ascii=False, indent=1))
    print(f"[saved] {OUT_DIR}/stage0_cub.json, stage0_cub.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
