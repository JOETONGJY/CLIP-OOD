"""
Stage 0A / 0B 第一层的诊断 (预注册见 RESULTS.md §3r; 指标见 concept_cancellation.py; 规格见 doc/主线B实施规格.md)。
零训练: 只读 run_matrix.py stage0 保存的收敛分类头 (logs/heads/CUB_s0_*_seed*.pt)、域方向、源域训练集缓存特征。
**判定全部由本脚本按 §3r 的常量从数据算出**; 只用源域数据, 不读取任何目标域信息。

0A 偏置吸收: 对每个类 c, 有效偏置 c = W·beta + b, 主域偏移方向 A·mu (mu = E[u], 非中心化)。
   吸收比例 rho_abs = 1 - ||A·mu + c|| / ||A·mu||   —— 偏置把主方向响应抵消掉的比例 (逐类, 再取中位数)。
   判据比较 Full DDO 与 CE-Only: CE-Only 没有 DDO 压力, 它的偏置对齐来自分类本身, 不扣掉它就不是"吸收"。

0B 第一层: 决策方向 w = A_y - A_k 只依赖分类头 (与样本无关), 所以对全部 C*(C-1) 个有序类别对计算:
   PCR 分位 (在同一个 G 的随机经验基线中)、CI (RMS 形式)、依赖能量 (相对同范数随机权重)。
   "抵消型" = PCR 分位 <= 0.10 且 CI >= 0.8 且依赖能量 >= 3x。比例 = 抵消型对数 / 全部对数。
   判据比较 Full DDO 与 CE-Only 的比例 (按 seed 聚合)。

用法: python stage0_diagnose.py            (需要先跑完 run_matrix.py stage0)
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
B_QTL_MAX, B_CI_MIN, B_DEP_MIN = 0.10, 0.80, 3.0
B_EXIST_RATIO, B_EXIST_MIN_ABS = 2.0, 0.05
B_NONE_MAX_DIFF = 0.02


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
            if os.path.exists(ref_p) and r is not None:        # 参照或本 run 缺失都算未通过, 不静默放过
                ref = json.load(open(ref_p, encoding="utf-8"))
                same = (ref["final_diag"] == r["final_diag"])
            n_cmp += 1
            n_same += same
    checks.append(("27 个 Stage 0 run 都完整结束 (status / 100 epochs)", n_ok == 27 and len(heads) == 27,
                   f"{n_ok} / 27"))
    checks.append(("27 个重训 run 的 final_diag 与对应的正式结果逐位相同", n_cmp == 27 and n_same == 27,
                   f"{n_same} / {n_cmp}"))
    return checks


def diagnose(heads, U):
    G, G_cov, mu, eta = second_moments(U)
    Dt = diag_reference(G)
    base = random_baseline(G, n=40000, seed=0)
    K = U.size(1)
    abs_u = U.abs().mean(0)                                   # E|u_j| (逐概念)
    Hm = torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    g = torch.Generator().manual_seed(1)
    rw = torch.randn(K, 40000, generator=g, dtype=torch.float64)
    rw = rw - rw.mean(0, keepdim=True)
    rw = rw / rw.norm(dim=0, keepdim=True)
    den_unit = float(quad(rw.T, Dt).mean())                   # 单位范数随机权重的期望分母
    Amu = None
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
            # ---------- 0A ----------
            Amu = A @ mu
            rho = 1.0 - (Amu + c).norm(dim=-1) / Amu.norm(dim=-1)          # (C,)
            cosv = torch.cosine_similarity(c, -Amu, dim=-1)
            # ---------- 0B ----------
            y_idx, k_idx = torch.triu_indices(A.size(0), A.size(0), offset=1)
            Wk = (A[y_idx] - A[k_idx]) @ Hm                                 # 决策方向 (P, K), 投影到超平面
            num, den = quad(Wk, G), quad(Wk, Dt).clamp_min(1e-300)
            pcr = num / den
            qtl = pcr_quantile(pcr, base)
            ci = 1.0 - torch.sqrt(num.clamp_min(0)) / ((Wk.abs() @ abs_u) + 1e-12)   # RMS 形式
            dep = (den / Wk.norm(dim=-1) ** 2) / den_unit
            cancelling = (qtl <= B_QTL_MAX) & (ci >= B_CI_MIN) & (dep >= B_DEP_MIN)
            per_seed.append({
                "seed": s,
                "rho_abs_median": float(rho.median()), "rho_abs_mean": float(rho.mean()),
                "cos_c_vs_minusAmu_median": float(cosv.median()),
                "amp_ratio_median": float((c.norm(dim=-1) / Amu.norm(dim=-1)).median()),
                "n_pairs": int(Wk.size(0)),
                "cancel_frac": float(cancelling.double().mean()),
                "cancel_frac_pcr_only": float((qtl <= B_QTL_MAX).double().mean()),
                "cancel_frac_ci_only": float((ci >= B_CI_MIN).double().mean()),
                "median_pcr_quantile": float(qtl.median()), "median_ci": float(ci.median()),
                "median_dep_energy": float(dep.median()),
                "frac_dep_ge_3": float((dep >= B_DEP_MIN).double().mean()),
            })
        if not per_seed:                                     # 该臂没有任何收敛分类头: 记录为缺失, 不崩
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
    # ---- 0A ----
    full_med = float(np.mean([med(t, "rho_abs_median") for t in FULL_ARMS]))
    ce_med = med(CE_ARM, "rho_abs_median")
    gap = full_med - ce_med
    if full_med >= A_EXIST_MIN and gap >= A_EXIST_GAP:
        va = "偏置吸收存在"
    elif full_med < A_NONE_MAX or gap < A_NONE_GAP:
        va = "偏置吸收不存在"
    else:
        va = "不确定"
    # ---- 0B 第一层 ----
    f_frac = [p["cancel_frac"] for t in FULL_ARMS for p in a[t]["per_seed"]]
    c_frac = [p["cancel_frac"] for p in a[CE_ARM]["per_seed"]]
    fm, cm = float(np.mean(f_frac)), float(np.mean(c_frac))
    ratio = fm / cm if cm > 0 else float("inf")
    # 逐 seed 方向: 对每个 (Full 臂, seed), 抵消型比例是否高于 CE-Only 同 seed
    dirs = []
    for t in FULL_ARMS:
        for p in a[t]["per_seed"]:
            cs = [q["cancel_frac"] for q in a[CE_ARM]["per_seed"] if q["seed"] == p["seed"]]
            if cs:
                dirs.append(p["cancel_frac"] > cs[0])
    consistent = all(dirs) and len(dirs) >= 4
    if ratio >= B_EXIST_RATIO and consistent and fm >= B_EXIST_MIN_ABS:
        vb = "现象存在"
    elif abs(fm - cm) < B_NONE_MAX_DIFF or not consistent:
        vb = "现象不存在"
    else:
        vb = "不确定"
    return {"0A": {"full_median": full_med, "ce_median": ce_med, "gap": gap, "verdict": va},
            "0B": {"full_frac": fm, "ce_frac": cm, "ratio": ratio, "n_seed_pairs": len(dirs),
                   "all_directions_consistent": consistent, "verdict": vb}}


def self_test():
    """在真实数据之前, 用已知答案的合成分类头验证判定逻辑 (RESULTS §3r 的两类结论各造一个)。
    合成域方向: 域敏感概念随同一潜变量同相变化; 其余概念几乎不变。
      构造 A (吸收 + 抵消): c = -(0.8)·A·mu (偏置吸收 80%), 且一半类别的决策方向用"抵消型"权重;
      构造 B (干净):       c 与 A·mu 正交, 决策方向不用域敏感概念。
    期望: A -> 0A"存在"、0B 抵消型比例明显高; B -> 0A"不存在"、0B 抵消型比例 ~0。"""
    torch.manual_seed(0)
    K, C, N = 60, 30, 8000
    sens = [3, 7, 11, 19, 25, 31]
    tt = torch.randn(N, 1)
    U = torch.randn(N, K) * 0.05
    U[:, sens] = tt * 2.0 + torch.randn(N, len(sens)) * 0.4
    U = (U - U.mean(-1, keepdim=True)).double()
    G, _, mu, _ = second_moments(U)
    heads = {}
    for tag in ("ce_a0", "full_a1", "full_a2"):
        for s in SEEDS:
            W = torch.randn(C, K, dtype=torch.float64) * 0.5
            gam = torch.ones(K, dtype=torch.float64)
            A = W * gam[None, :]
            if tag == "ce_a0":                                      # 干净: 偏置与 A·mu 正交
                c = torch.randn(C, dtype=torch.float64)
                Amu = A @ mu
                c = c - (c @ Amu) / (Amu @ Amu) * Amu
                c = c[:, None] * torch.ones(1, K, dtype=torch.float64) if False else c
                beta = torch.zeros(K, dtype=torch.float64); b = c
            else:                                                   # 吸收 + 抵消
                W[: C // 2, 3] = 6.0; W[: C // 2, 7] = -6.0         # 一半类别: 抵消型权重
                A = W * gam[None, :]
                c = -0.8 * (A @ mu)                                 # 偏置吸收 80%
                beta = torch.zeros(K, dtype=torch.float64); b = c
            heads[(tag, s)] = {"classifier": {"2.weight": W, "1.weight": gam, "1.bias": beta, "2.bias": b}}
    res = diagnose(heads, U)
    v = verdicts(res)
    ok = True
    fa = res["arms"]["full_a1"]["agg"]["cancel_frac"][0]
    ca = res["arms"]["ce_a0"]["agg"]["cancel_frac"][0]
    fr = res["arms"]["full_a1"]["agg"]["rho_abs_median"][0]
    cr = res["arms"]["ce_a0"]["agg"]["rho_abs_median"][0]
    print(f"  合成 A (Full):  rho_abs 中位数 {fr:.3f}, 抵消型比例 {fa * 100:.1f}%")
    print(f"  合成 B (CE):    rho_abs 中位数 {cr:.3f}, 抵消型比例 {ca * 100:.1f}%")
    for name, cond in (("S1 偏置吸收被识别: Full 的 rho_abs >= 0.5", fr >= 0.5),
                       ("S2 干净头的 rho_abs 低 (< 0.25)", cr < 0.25),
                       ("S3 0A 判定为'存在'", v["0A"]["verdict"] == "偏置吸收存在"),
                       ("S4 抵消型比例: Full >= 2x CE 且 >= 5%", fa >= 2 * ca and fa >= 0.05),
                       ("S5 0B 判定为'现象存在'", v["0B"]["verdict"] == "现象存在"),
                       ("S6 干净头的抵消型比例很低 (< 2%)", ca < 0.02)):
        print(("  OK  " if cond else "  BAD ") + name)
        ok &= bool(cond)
    # 反向: 把干净头同时当作 Full 与 CE, 应判"不存在"
    heads_clean = {(tag, s): heads[("ce_a0", s)] for tag in ("ce_a0", "full_a1", "full_a2") for s in SEEDS}
    v2 = verdicts(diagnose(heads_clean, U))
    print(f"  [反向] 干净头自比: 0A = {v2['0A']['verdict']}, 0B = {v2['0B']['verdict']}")
    for name, cond in (("S7 [反向] 0A 判为'不存在'", v2["0A"]["verdict"] == "偏置吸收不存在"),
                       ("S8 [反向] 0B 判为'现象不存在'", v2["0B"]["verdict"] == "现象不存在")):
        print(("  OK  " if cond else "  BAD ") + name)
        ok &= bool(cond)
    # ---- verify_heads: 在临时目录里造假结果, 走通"通过"与"不通过"两条路 (这条路径 selftest 原先走不到) ----
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
        c9 = len(good) == 2 and all(c for _, c, _ in good)
        print(("  OK  " if c9 else "  BAD ") + "S9 verify_heads: 27 个一致 -> 全部通过")
        ok &= c9
        json.dump(fake(final_diag={"target_acc": 2.0}), open("logs/results/CUB_s0_full_a1_seed1.json", "w"))
        bad = verify_heads(hd)
        c10 = any(not c for _, c, _ in bad)
        print(("  OK  " if c10 else "  BAD ") + "S10 verify_heads: 有一个不一致 -> 报未通过 (不静默放过)")
        ok &= c10
        os.remove("logs/results/CUB_dose_a2_seed3.json")
        c11 = any(not c for _, c, _ in verify_heads(hd))
        print(("  OK  " if c11 else "  BAD ") + "S11 verify_heads: 参照文件缺失 -> 报未通过")
        ok &= c11
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
    cache = torch.load("cache/CUB_train.pt", map_location="cpu")
    res = diagnose(heads, U)
    res["checks"] = [{"check": n, "ok": bool(c), "detail": d} for n, c, d in checks]
    res["thresholds"] = {"A_EXIST": [A_EXIST_MIN, A_EXIST_GAP], "A_NONE": [A_NONE_MAX, A_NONE_GAP],
                         "B_cancelling": [B_QTL_MAX, B_CI_MIN, B_DEP_MIN],
                         "B_EXIST": [B_EXIST_RATIO, B_EXIST_MIN_ABS], "B_NONE_MAX_DIFF": B_NONE_MAX_DIFF}
    res["verdicts"] = verdicts(res)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stage0_cub.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    b = res["random_baseline"]
    L = ["# Stage 0A / 0B 第一层诊断（CUB，3 seeds，2026-11-25）", "",
         "预注册见 `RESULTS.md` §3r；判定全部由 `stage0_diagnose.py` 按预注册阈值算出。只用源域数据。", "",
         "## 核对", "", "| 核对项 | 结果 |", "|---|---|"]
    for n, c, d in checks:
        L.append(f"| {n} | {'通过' if c else '**未通过**'} | {d} |")
    L += ["", f"域方向均值能量占比 η_mean = {res['eta_mean']:.4f}。PCR 随机基线（同一个 G，40000 个超平面内随机权重）："
          f"中位数 {b['q500']:.3f}，5%～95% [{b['q050']:.3f}, {b['q950']:.3f}]；diag(G) 最大值 / 中位数 = {b['diag_max_over_median']:.1f}。", "",
          "## 0A 偏置吸收", "", "| 臂 | ρ_abs 中位数（均值 ± sd） | cos(c, −A·μ) 中位数 | ‖c‖/‖A·μ‖ 中位数 |", "|---|---|---|---|"]
    for tag, name in ARMS:
        if tag in res["arms"]:
            g = res["arms"][tag]["agg"]
            L.append(f"| {name} | {g['rho_abs_median'][0]:.3f} ± {g['rho_abs_median'][1]:.3f} | "
                     f"{g['cos_c_vs_minusAmu_median'][0]:.3f} | {g['amp_ratio_median'][0]:.3f} |")
    v = res["verdicts"]["0A"]
    L += ["", f"**判定：{v['verdict']}**（Full 中位数 {v['full_median']:.3f}，CE-Only {v['ce_median']:.3f}，差 {v['gap']:+.3f}；"
          f"存在需 ≥ {A_EXIST_MIN} 且差 ≥ {A_EXIST_GAP}；不存在需 < {A_NONE_MAX} 或差 < {A_NONE_GAP}）。", "",
          "## 0B 第一层：抵消型类别对的比例", "",
          f"\"抵消型\" = PCR 分位 ≤ {B_QTL_MAX} 且 CI ≥ {B_CI_MIN} 且依赖能量 ≥ {B_DEP_MIN}× 随机。", "",
          "| 臂 | 抵消型比例（均值 ± sd） | 仅看 PCR 分位 | 仅看 CI | PCR 分位中位数 | CI 中位数 | 依赖能量中位数 |",
          "|---|---|---|---|---|---|---|"]
    for tag, name in ARMS:
        if tag in res["arms"]:
            g = res["arms"][tag]["agg"]
            L.append(f"| {name} | **{g['cancel_frac'][0] * 100:.2f}% ± {g['cancel_frac'][1] * 100:.2f}** | "
                     f"{g['cancel_frac_pcr_only'][0] * 100:.1f}% | {g['cancel_frac_ci_only'][0] * 100:.1f}% | "
                     f"{g['median_pcr_quantile'][0]:.3f} | {g['median_ci'][0]:.3f} | {g['median_dep_energy'][0]:.2f} |")
    v = res["verdicts"]["0B"]
    L += ["", f"**判定：{v['verdict']}**（Full {v['full_frac'] * 100:.2f}%，CE-Only {v['ce_frac'] * 100:.2f}%，"
          f"比值 {v['ratio']:.2f}；{v['n_seed_pairs']} 个 seed 配对方向一致 = {v['all_directions_consistent']}；"
          f"存在需比值 ≥ {B_EXIST_RATIO}、方向一致且 ≥ {B_EXIST_MIN_ABS * 100:.0f}%；不存在需差 < {B_NONE_MAX_DIFF * 100:.0f} 个百分点或方向不一致）。", "",
          "## 逐 seed", "", "| 臂 | seed | ρ_abs 中位数 | 抵消型比例 | PCR 分位中位数 | CI 中位数 | 依赖能量中位数 |", "|---|---|---|---|---|---|---|"]
    for tag, name in ARMS:
        for p in res["arms"].get(tag, {"per_seed": []})["per_seed"]:
            L.append(f"| {name} | {p['seed']} | {p['rho_abs_median']:.3f} | {p['cancel_frac'] * 100:.2f}% | "
                     f"{p['median_pcr_quantile']:.3f} | {p['median_ci']:.3f} | {p['median_dep_energy']:.2f} |")
    L += ["", "## 读法", "",
          "- 0A 的 ρ_abs 是逐类计算后取的中位数；CE-Only 的偏置对齐来自分类本身，所以判据看的是相对 CE-Only 的增量。",
          "- 0B 的比例是 C(C−1)/2 个类别对中\"抵消型\"的占比；单个类别对的 PCR 没有判别力（随机基线很宽），所以只看比例。",
          "- 第一层只回答\"现象是否存在\"。存在不等于脆弱；是否在源域视觉增强下失效、收缩协方差能否改善，需要各自的实验。", ""]
    with open(os.path.join(OUT_DIR, "stage0_cub.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(json.dumps(res["verdicts"], ensure_ascii=False, indent=1))
    print(f"[saved] {OUT_DIR}/stage0_cub.json, stage0_cub.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
