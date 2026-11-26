"""P1 (Stage 0D 独立验证) 的分析与判定。预注册见 RESULTS.md §3v —— 规则与阈值都写在那里, 本脚本只负责执行。

故意分成两次运行, 中间隔一次 git 提交:
  --stage source : 只读 21 个结果 json 的源域验证字段, 按预注册规则算出选择结果并写盘
                   (experiment_results/Stage0D-Independent/p1_source_selection.json / .md)。
                   脚本逐个核对结果 json 的配置、训练子集与验证集的来源, 并断言其中没有任何目标域数字
                   (--target_eval none) —— 这一步在信息上接触不到目标域。
  --stage final  : 要求选择结果文件已被 git 跟踪且没有未提交的修改 (锁定), 重新计算源域部分并与锁定的文件
                   逐项比对, 然后读入事后目标域评估 (eval_heads_targets.py 的输出), 计算 Spearman 相关、
                   Selection Regret 与"能否识别过正则化", 写出 p1_final_report.json / .md。
  --selftest     : 合成数据上的规则自测 (并列取小、Spearman 的并列秩、精确置换 p 值、regret、四条判定分支、锁定检查)。

预注册的量 (与 RESULTS §3v 一致, 这里不重新定义):
  主指标   mean_aug_acc = heldout_val 上 7 种光度扰动 Acc 的平均; 按 3 个 seed 的均值选 α, 并列取较小的 α
  辅助指标 mean_aug_ce (取最小)、clean_acc (取最大)、clean_ce (取最小); 逐 seed 选择只报告不参与主判定
  事后量   A_T(α) = 源域选择锁定之后, 用保存的分类头在目标域 (CUB-200-Painting 测试集) 上算出的 Acc
           Spearman ρ (7 个剂量的 seed 均值曲线之间); R_sel = max_α A_T(α) − A_T(α̂)
  阈值     追踪: ρ ≥ 0.79 (n = 7 时双侧 p < 0.05 的临界值); 可用: R_sel ≤ 0.5 个百分点;
           过正则化: α = 4 在目标域上比最优剂量低 ≥ 0.5 个百分点, 且源域验证同样把 α = 4 压低 ≥ 0.5 个百分点
用法:
  python p1_independent_report.py --selftest
  python p1_independent_report.py --stage source
  python p1_independent_report.py --stage final
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage0D-Independent")
SEL_JSON, SEL_MD = os.path.join(OUT_DIR, "p1_source_selection.json"), os.path.join(OUT_DIR, "p1_source_selection.md")
FIN_JSON, FIN_MD = os.path.join(OUT_DIR, "p1_final_report.json"), os.path.join(OUT_DIR, "p1_final_report.md")
TGT_JSON = os.path.join(OUT_DIR, "p1_target_eval.json")

DOSES = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
TAGS = ["dose_a0", "dose_a0p125", "dose_a0p25", "dose_a0p5", "dose_a1", "dose_a2", "dose_a4"]
SEEDS = [1, 2, 3]
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]
N_TRAIN, N_VAL, N_FULL = 4790, 1200, 5990
COMMON = {"epochs": 100, "batch_size": 256, "beta": 0.0, "lr": 0.0001, "weight_decay": 0.0001, "model_selection": "final",
          "target_eval": "none", "weight_mode": "none", "ddo_dirs": "full", "save_head": True, "lambda_kl": 0.0, "lambda_graph": 0.0}
PRIMARY = "mean_aug_acc"
# ---- 预注册阈值 (RESULTS §3v); 不改 ----
PLATEAU_PP = 0.5          # 平台描述: 距最大值不超过 0.5 个百分点的剂量 (只报告, 不参与选择)
RHO_TRACK = 0.79          # Spearman 追踪判据 (n = 7, 双侧 p < 0.05)
REGRET_OK_PP = 0.5        # 选参可用判据: Selection Regret 不超过 0.5 个百分点
OVER_MARGIN_PP = 0.5      # 过正则化: 目标域与源域验证两侧的幅度门槛
OVER_DOSE = 4.0


# ---------------- 纯函数 (可在合成数据上自测) ----------------
def _rank(v):
    """1 起算的平均秩 (并列取平均)"""
    v = np.asarray(v, dtype=float)
    order, ranks = np.argsort(v, kind="mergesort"), np.empty(len(v))
    sv, r, i = v[order], np.arange(1, len(v) + 1, dtype=float), 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_exact_p(rho, n):
    """无并列时 Spearman 的精确置换双侧 p 值 (穷举 n! 个秩排列; n <= 8)。"""
    if not (n <= 8) or rho != rho:
        return None
    base = np.arange(1, n + 1, dtype=float)
    perms = np.array(list(itertools.permutations(base)), dtype=float)        # (n!, n)
    perms -= perms.mean(axis=1, keepdims=True)
    rhos = (perms @ (base - base.mean())) / np.sqrt((perms ** 2).sum(1) * ((base - base.mean()) ** 2).sum())
    return float((np.abs(rhos) >= abs(rho) - 1e-12).mean())


def argbest(curve, mode="max"):
    """取最优剂量; 并列 (均值完全相等) 取较小的 α。curve: {alpha: value}"""
    vals = dict(curve)
    best = max(vals.values()) if mode == "max" else min(vals.values())
    return min(a for a, v in vals.items() if v == best)


def over_reg_verdict(t_over, s_over):
    """α = 4 的过正则化识别: T_over = 目标域上 α=4 比最优剂量低 ≥ 门槛; S_over = 源域验证把它压低 ≥ 门槛且没有选它"""
    if t_over and s_over:
        return "识别成功"
    if t_over and not s_over:
        return "未识别 (目标域上 α=4 已过正则化, 但增强验证没有把它排除)"
    if s_over:
        return "无过正则化可识别 (目标域上 α=4 没有低于最优剂量; 源域验证却惩罚了它, 与目标域不一致)"
    return "无过正则化可识别 (目标域与源域验证都没有把 α=4 排除)"


def final_analysis(sel, T, src_curves):
    """sel: --stage source 写出的选择结果; T: {alpha: [逐 seed 的目标域 Acc]}; src_curves: {指标: {alpha: [逐 seed 值]}}。
    返回最终报告里的全部数字 (判定按预注册阈值由代码给出)。"""
    doses = sorted(T)
    mT = {a: float(np.mean(v)) for a, v in T.items()}
    alpha_hat = sel["selection"]["primary"]["alpha_hat"]
    alpha_star = argbest(mT, "max")
    src_mean = {k: {a: float(np.mean(v)) for a, v in c.items()} for k, c in src_curves.items()}
    tv = [mT[a] for a in doses]
    rho = {k: spearman([src_mean[k][a] for a in doses], tv) for k in ("mean_aug_acc", "clean_acc", "mean_aug_ce")}
    pooled_x = [x for a in doses for x in src_curves["mean_aug_acc"][a]]
    pooled_y = [y for a in doses for y in T[a]]
    per_seed = [spearman([src_curves["mean_aug_acc"][a][i] for a in doses], [T[a][i] for a in doses]) for i in range(len(SEEDS))]
    sel_hat = {"primary": alpha_hat, "aug_ce": sel["selection"]["secondary"]["mean_aug_ce"],
               "clean_acc": sel["selection"]["secondary"]["clean_acc"], "clean_ce": sel["selection"]["secondary"]["clean_ce"]}
    regret = {k: mT[alpha_star] - mT[a] for k, a in sel_hat.items()}
    ps = sel["selection"]["per_seed"][PRIMARY]
    regret["per_seed_mean"] = float(np.mean([max(T[a][i] for a in doses) - T[ps[i]][i] for i in range(len(SEEDS))]))
    regret["default_a1"] = mT[alpha_star] - mT[1.0]
    a4 = OVER_DOSE
    t_over = (alpha_star != a4) and (mT[a4] <= mT[alpha_star] - OVER_MARGIN_PP)
    s_over = (alpha_hat != a4) and (src_mean[PRIMARY][a4] <= max(src_mean[PRIMARY].values()) - OVER_MARGIN_PP)
    paired = {}
    for name, (x, y) in {"a4_vs_a2": (a4, 2.0), "a2_vs_a1": (2.0, 1.0), "selected_vs_a1": (alpha_hat, 1.0)}.items():
        if x == y:
            continue
        d_t = [T[x][i] - T[y][i] for i in range(len(SEEDS))]
        d_s = [src_curves[PRIMARY][x][i] - src_curves[PRIMARY][y][i] for i in range(len(SEEDS))]
        paired[name] = {"target_diff_per_seed": d_t, "target_n_positive": int(sum(d > 0 for d in d_t)),
                        "source_diff_per_seed": d_s, "source_n_positive": int(sum(d > 0 for d in d_s))}
    better_clean = (regret["primary"] < regret["clean_acc"]
                    and (rho["clean_acc"] != rho["clean_acc"] or rho["mean_aug_acc"] > rho["clean_acc"]))
    out = {
        "target_mean_acc": mT, "target_sd_acc": {a: float(np.std(v, ddof=1)) for a, v in T.items()},
        "alpha_star": alpha_star, "A_T_of_selected": mT[alpha_hat], "A_T_of_default_a1": mT[1.0], "A_T_of_best": mT[alpha_star],
        "spearman": rho, "spearman_exact_p": {k: spearman_exact_p(v, len(doses)) for k, v in rho.items()},
        "spearman_per_seed_mean_aug_acc": per_seed,
        "spearman_pooled_21_points": spearman(pooled_x, pooled_y),
        "regret_pp": regret,
        "over_reg": {"dose": a4, "A_T_a4": mT[a4], "A_T_best": mT[alpha_star], "target_drop_pp": mT[alpha_star] - mT[a4],
                     "source_mean_aug_acc_a4": src_mean[PRIMARY][a4], "source_max": max(src_mean[PRIMARY].values()),
                     "source_drop_pp": max(src_mean[PRIMARY].values()) - src_mean[PRIMARY][a4],
                     "T_over": bool(t_over), "S_over": bool(s_over), "verdict": over_reg_verdict(t_over, s_over)},
        "paired": paired,
        "verdict": {
            "tracks_target": bool(rho["mean_aug_acc"] >= RHO_TRACK),
            "selection_usable": bool(regret["primary"] <= REGRET_OK_PP),
            # 优于干净验证选参: R_sel 更小, 且 ρ 更高 (干净曲线无变化、ρ 无定义时只看前一条)
            "better_than_clean_selection": bool(better_clean),
            "beats_default_a1": bool(regret["default_a1"] > 0 and mT[alpha_hat] > mT[1.0]),
        },
    }
    if all("aug_acc:" + an in src_curves for an in AUGS):          # 各扰动单独的 ρ (检查是否由某一种扰动主导)
        out["spearman_per_aug"] = {an: spearman([src_mean["aug_acc:" + an][a] for a in doses], tv) for an in AUGS}
    return out


# ---------------- 读取与核对 ----------------
def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _num(x):
    return float(x)


def load_runs():
    """读取 21 个 run 的源域验证字段; 任何一项核对不过就抛错 (不产出部分结果)。"""
    runs, split_shas = {}, set()
    for a, tag in zip(DOSES, TAGS):
        for s in SEEDS:
            name = f"CUB_p1_{tag}_seed{s}"
            rp, hp = f"logs/results/{name}.json", f"logs/heads/{name}.pt"
            with open(rp, encoding="utf-8") as f:
                r = json.load(f)
            cfg = r["run_config"]
            assert r["status"] == "completed" and r["completed_epochs"] == cfg["epochs"], f"{name}: 未完整结束"
            assert _num(cfg["alpha"]) == a and cfg["seed"] == s and cfg["run_tag"] == f"p1_{tag}", f"{name}: 配置不符"
            for k, v in COMMON.items():
                assert _same(cfg.get(k), v), f"{name}: {k}={cfg.get(k)!r}, 应为 {v!r}"
            fd = r.get("final_diag") or {}
            assert all(r.get(k) is None for k in ("target_acc", "sel_target_acc")) and fd.get("target_acc") is None \
                and fd.get("target_ce") is None and all(e["target_acc"] is None for e in r["epoch_metrics"]), \
                f"{name}: 结果里出现了目标域数字 —— 违反预注册的先后顺序"
            ts, sv = r["train_subset"], r["source_val"]
            assert (ts["n_train"], ts["n_heldout"], ts["n_full"], ts["labels_match_train_cache"]) == (N_TRAIN, N_VAL, N_FULL, True) \
                and ts["split_name"] == "train_core", f"{name}: 训练子集不符"
            assert sv["n"] == N_VAL and sv["split_name"] == "heldout_val" and sorted(sv["aug"]) == sorted(AUGS), \
                f"{name}: 验证集不符"
            assert ts["split_cache_sha256"] == sv["cache_sha256"], f"{name}: 训练子集与验证集不是同一份划分"
            split_shas.add((ts["subset_sha256"], ts["split_cache_sha256"], ts["train_cache_sha256"], sv["cache_sha256"]))
            runs[name] = {"alpha": a, "seed": s, "head_sha256": _sha(hp), "result_sha256": _sha(rp), "git": r.get("git"),
                          "clean_acc": sv["clean"]["acc"], "clean_ce": sv["clean"]["ce"],
                          "mean_aug_acc": sv["mean_aug_acc"], "mean_aug_ce": sv["mean_aug_ce"],
                          "aug_acc": {an: sv["aug"][an]["acc"] for an in AUGS}, "aug_ce": {an: sv["aug"][an]["ce"] for an in AUGS},
                          "source_test_acc": fd.get("source_acc"), "source_test_ce": fd.get("source_ce"), "ddo_eq": fd.get("ddo_eq")}
    assert len(split_shas) == 1, "各 run 使用的划分 / 缓存不一致"
    runs["_provenance"] = {"subset_sha256": list(split_shas)[0][0], "split_cache_sha256": list(split_shas)[0][1],
                           "train_cache_sha256": list(split_shas)[0][2]}
    return runs


def _same(got, want):
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return got == want


def curves(runs, key):
    """{alpha: [逐 seed 的值]}; key 可以是 "mean_aug_acc" 这样的标量字段, 或 "aug_acc:solarize" 这样的嵌套字段"""
    if ":" in key:
        outer, inner = key.split(":")
        return {a: [runs[f"CUB_p1_{tag}_seed{s}"][outer][inner] for s in SEEDS] for a, tag in zip(DOSES, TAGS)}
    return {a: [runs[f"CUB_p1_{tag}_seed{s}"][key] for s in SEEDS] for a, tag in zip(DOSES, TAGS)}


def source_selection(runs):
    """预注册的选择规则。runs 来自 load_runs()。"""
    C = {k: curves(runs, k) for k in ("mean_aug_acc", "mean_aug_ce", "clean_acc", "clean_ce")}
    mean = {k: {a: float(np.mean(v)) for a, v in c.items()} for k, c in C.items()}
    a_hat = argbest(mean[PRIMARY], "max")
    per_seed = {k: [argbest({a: C[k][a][i] for a in DOSES}, "max" if k.endswith("acc") else "min") for i in range(len(SEEDS))]
                for k in C}
    table = []
    for a in DOSES:
        row = {"alpha": a}
        for k in ("clean_acc", "clean_ce", "mean_aug_acc", "mean_aug_ce", "source_test_acc", "source_test_ce"):
            v = curves(runs, k)[a]
            row[k] = float(np.mean(v))
            row[k + "_sd"] = float(np.std(v, ddof=1))
        row["aug_acc_mean"] = {an: float(np.mean(curves(runs, "aug_acc:" + an)[a])) for an in AUGS}
        row["aug_ce_mean"] = {an: float(np.mean(curves(runs, "aug_ce:" + an)[a])) for an in AUGS}
        table.append(row)
    top = max(mean[PRIMARY].values())
    return {"doses": DOSES, "seeds": SEEDS, "augmentations": AUGS, "table": table,
            "selection": {
                "primary": {"metric": PRIMARY, "rule": "3 个 seed 的均值最大; 并列取较小的 α", "alpha_hat": a_hat, "value": mean[PRIMARY][a_hat]},
                "secondary": {"mean_aug_ce": argbest(mean["mean_aug_ce"], "min"), "clean_acc": argbest(mean["clean_acc"], "max"),
                              "clean_ce": argbest(mean["clean_ce"], "min")},
                "per_seed": per_seed,
                "plateau_pp": PLATEAU_PP,
                "plateau_primary": [a for a in DOSES if mean[PRIMARY][a] >= top - PLATEAU_PP],
                "over_reg_source": {"dose": OVER_DOSE, "mean_aug_acc": mean[PRIMARY][OVER_DOSE], "max": top,
                                    "drop_pp": top - mean[PRIMARY][OVER_DOSE],
                                    "S_over": bool(a_hat != OVER_DOSE and mean[PRIMARY][OVER_DOSE] <= top - OVER_MARGIN_PP)},
            },
            "provenance": runs["_provenance"]}


# ---------------- 报告 ----------------
def _f(x, nd=2):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def source_md(sel, runs):
    L = ["# P1 · Stage 0D 独立验证：源域验证与选择结果（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3v。本文件只含源域数据：训练用 train_core（4790），验证用 heldout_val（1200）的干净特征与 "
         "7 种光度扰动特征；21 个 run 都以 `--target_eval none` 训练，结果 json 中没有任何目标域数字（脚本已逐个断言）。", "",
         f"- 剂量曲线（3 seeds 均值 ± 标准差；Acc 单位 %，CE 为交叉熵）", "",
         "| α | 干净 Acc | 干净 CE | 7 种扰动均值 Acc（主指标） | 7 种扰动均值 CE | 源测试集 Acc（仅监控） |",
         "|---|---|---|---|---|---|"]
    for r in sel["table"]:
        L.append(f"| {r['alpha']:g} | {_f(r['clean_acc'])} ± {_f(r['clean_acc_sd'])} | {_f(r['clean_ce'], 4)} | "
                 f"**{_f(r['mean_aug_acc'])}** ± {_f(r['mean_aug_acc_sd'])} | {_f(r['mean_aug_ce'], 4)} | {_f(r['source_test_acc'])} |")
    s = sel["selection"]
    L += ["", "## 选择结果（预注册规则）", "",
          f"- **主指标（7 种扰动均值 Acc，3 seeds 均值最大，并列取较小 α）：α̂ = {s['primary']['alpha_hat']:g}**（{_f(s['primary']['value'])}%）",
          f"- 平台（距最大值 ≤ {PLATEAU_PP} 个百分点，只报告）：{', '.join(f'{a:g}' for a in s['plateau_primary'])}",
          f"- 辅助指标的选择：扰动均值 CE → α = {s['secondary']['mean_aug_ce']:g}；干净 Acc → α = {s['secondary']['clean_acc']:g}；"
          f"干净 CE → α = {s['secondary']['clean_ce']:g}",
          f"- 逐 seed 的选择（主指标）：{dict(zip(SEEDS, s['per_seed'][PRIMARY]))}；"
          f"扰动均值 CE：{dict(zip(SEEDS, s['per_seed']['mean_aug_ce']))}",
          f"- 过正则化的源域一侧（α = 4）：扰动均值 Acc 比最大值低 {_f(s['over_reg_source']['drop_pp'])} 个百分点；"
          f"按门槛 {OVER_MARGIN_PP} 判为 S_over = {s['over_reg_source']['S_over']}", "",
          "各扰动单独的 Acc（3 seeds 均值，%）：", "", "| α | " + " | ".join(AUGS) + " |", "|---|" + "---|" * len(AUGS)]
    for r in sel["table"]:
        L.append(f"| {r['alpha']:g} | " + " | ".join(_f(r["aug_acc_mean"][an]) for an in AUGS) + " |")
    L += ["", "各扰动单独的 CE（3 seeds 均值）：", "", "| α | " + " | ".join(AUGS) + " |", "|---|" + "---|" * len(AUGS)]
    for r in sel["table"]:
        L.append(f"| {r['alpha']:g} | " + " | ".join(_f(r["aug_ce_mean"][an], 3) for an in AUGS) + " |")
    p = sel["provenance"]
    L += ["", "## 数据来源", "",
          f"- 划分（train_core 子集）sha256 `{p['subset_sha256'][:16]}`；划分缓存 `cache/CUB_train_aug.pt` sha256 `{p['split_cache_sha256'][:16]}`；"
          f"训练特征缓存 `cache/CUB_train.pt` sha256 `{p['train_cache_sha256'][:16]}`",
          "- 训练特征：编码器原始输出，L2 归一化在模型 forward 里按 float32 做；验证特征：预计算时 L2 归一化后的 fp16。两者走同一条 forward 路径。",
          "- 本文件由 `p1_independent_report.py --stage source` 生成，并在此之后提交；提交之后才运行事后目标域评估。", ""]
    return "\n".join(L)


def final_md(sel, fin, T):
    s = sel["selection"]
    mT, v, o = fin["target_mean_acc"], fin["verdict"], fin["over_reg"]
    L = ["# P1 · Stage 0D 独立验证：最终报告（源域选择锁定之后的目标域事后评估）", "",
         "预注册见 `RESULTS.md` §3v。目标域 = CUB-200-Painting 测试集（3047 个样本）；训练期间未读取目标域特征缓存，"
         "选择结果提交锁定后才用保存的分类头评估。", "",
         "| α | 干净 Acc（源） | 7 种扰动均值 Acc（源，主指标） | 扰动均值 CE（源） | 目标域 Acc（3 seeds 均值 ± 标准差） |",
         "|---|---|---|---|---|"]
    by = {r["alpha"]: r for r in sel["table"]}
    for a in DOSES:
        r = by[a]
        mark = " ← 选中" if a == s["primary"]["alpha_hat"] else (" ← 事后最优" if a == fin["alpha_star"] else "")
        L.append(f"| {a:g} | {_f(r['clean_acc'])} | {_f(r['mean_aug_acc'])} | {_f(r['mean_aug_ce'], 4)} | "
                 f"{_f(mT[a])} ± {_f(fin['target_sd_acc'][a])}{mark} |")
    rh, rg = fin["spearman"], fin["regret_pp"]
    L += ["", "## Spearman 秩相关（7 个剂量的 seed 均值曲线，源域指标 对 目标域 Acc）", "",
          "| 源域指标 | ρ | 精确置换双侧 p (n = 7) |", "|---|---|---|",
          f"| 7 种扰动均值 Acc（主指标） | {_f(rh['mean_aug_acc'], 3)} | {_f(fin['spearman_exact_p']['mean_aug_acc'], 4)} |",
          f"| 干净 Acc | {_f(rh['clean_acc'], 3)} | {_f(fin['spearman_exact_p']['clean_acc'], 4)} |",
          f"| 7 种扰动均值 CE（预期为负） | {_f(rh['mean_aug_ce'], 3)} | {_f(fin['spearman_exact_p']['mean_aug_ce'], 4)} |",
          "",
          f"- 逐 seed 的 ρ（主指标）：{', '.join(_f(x, 3) for x in fin['spearman_per_seed_mean_aug_acc'])}；"
          f"21 个 (剂量, seed) 点合并的 ρ：{_f(fin['spearman_pooled_21_points'], 3)}",
          "- 各扰动单独的 Acc 对目标域的 ρ（检查是否由某一种扰动主导）：" +
          ", ".join(f"{an} {_f(x, 2)}" for an, x in fin["spearman_per_aug"].items()), "",
          "## 选择与 Selection Regret（单位：百分点）", "",
          f"- 源域选择 α̂ = **{s['primary']['alpha_hat']:g}**，目标域 Acc = {_f(fin['A_T_of_selected'])}%；"
          f"默认剂量 α = 1：{_f(fin['A_T_of_default_a1'])}%；事后最优 α* = {fin['alpha_star']:g}：{_f(fin['A_T_of_best'])}%",
          f"- R_sel（主指标）= **{_f(rg['primary'])}**；对照：干净 Acc 选参 {_f(rg['clean_acc'])}、扰动均值 CE 选参 {_f(rg['aug_ce'])}、"
          f"干净 CE 选参 {_f(rg['clean_ce'])}、默认 α = 1 的 {_f(rg['default_a1'])}、逐 seed 选择的平均 {_f(rg['per_seed_mean'])}", "",
          "## α = 4 的过正则化识别", "",
          f"- 目标域：α = 4 比事后最优低 {_f(o['target_drop_pp'])} 个百分点（T_over = {o['T_over']}）",
          f"- 源域验证：α = 4 比最大值低 {_f(o['source_drop_pp'])} 个百分点（S_over = {o['S_over']}）",
          f"- **{o['verdict']}**", "",
          "逐 seed 配对差（正号个数 / 3）：", ""]
    for name, p in fin["paired"].items():
        L.append(f"- {name}：目标域 {', '.join(_f(x) for x in p['target_diff_per_seed'])}（{p['target_n_positive']}/3 为正）；"
                 f"源域主指标 {', '.join(_f(x) for x in p['source_diff_per_seed'])}（{p['source_n_positive']}/3 为正）")
    L += ["", "## 判定（预注册阈值由代码给出）", "",
          f"- 追踪目标域（ρ ≥ {RHO_TRACK}）：**{v['tracks_target']}**（ρ = {_f(rh['mean_aug_acc'], 3)}）",
          f"- 选参可用（R_sel ≤ {REGRET_OK_PP} 个百分点）：**{v['selection_usable']}**（R_sel = {_f(rg['primary'])}）",
          f"- 优于干净验证选参（R_sel 更小且 ρ 更高）：**{v['better_than_clean_selection']}**",
          f"- 优于默认 α = 1：**{v['beats_default_a1']}**", "",
          "## 边界", "",
          "- 选择规则是在看到 Stage 0D 的探索性结果之后才冻结的，不是事先盲定的假说；7 种扰动在更早的时候就已固定。",
          "- CUB 是开发基准：这里的目标域结果参与过研究讨论，跨域选参的证据需要在未参与规则设计的数据集上确认。",
          "- 训练集是完整训练集的 80%（4790），这里的数字不能与完整训练集的历史目标域精度当作同一协议比较。",
          "- 7 个剂量、3 个 seed：不把三种子均值差当作显著性证明。", ""]
    return "\n".join(L)


def _locked(path, root=None):
    """文件存在、被 git 跟踪、相对 HEAD 没有未提交修改。返回 (是否锁定, 说明; 锁定时为 HEAD 提交)。
    root 是仓库根目录, 默认为本脚本所在目录 (自测里用临时仓库)。"""
    root = root or os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    try:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
    except ValueError:                                          # Windows: 不在同一盘符
        return False, f"{path} 与仓库 {root} 不在同一盘符, 不在仓库内"
    if rel.startswith(".."):
        return False, f"{path} 不在仓库 {root} 内"
    if subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", rel], capture_output=True).returncode != 0:
        return False, f"{rel} 未被 git 跟踪"
    if subprocess.run(["git", "-C", root, "status", "--porcelain", "--", rel], capture_output=True).stdout.strip():
        return False, f"{rel} 有未提交的修改"
    return True, subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True).stdout.decode().strip()


def _json_safe(obj):
    """递归地把 nan / inf 换成 None (ρ 在曲线无变化时无定义), 保证写出的是标准 JSON"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def stage_source():
    from run_status import git_info
    runs = load_runs()
    sel = source_selection(runs)
    out = {**sel, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3v",
           "target_info_present": False,
           "runs": {k: {kk: v[kk] for kk in ("alpha", "seed", "head_sha256", "result_sha256")} for k, v in runs.items() if k != "_provenance"}}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_json(SEL_JSON, out)
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write(source_md(sel, runs))
    s = sel["selection"]
    print(f"[source] 21 个 run 核对通过 (无任何目标域数字); 主指标选择 α̂ = {s['primary']['alpha_hat']:g} "
          f"({s['primary']['value']:.2f}%); 平台 {s['plateau_primary']}; S_over = {s['over_reg_source']['S_over']}")
    print(f"[saved] {SEL_JSON}、{SEL_MD} —— 提交之后才能运行 --stage final")
    return 0


def stage_final():
    from run_status import git_info
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: 源域选择结果尚未锁定 —— " + why)
        return 2
    # 源域部分重算一遍, 必须与锁定的文件逐项相同 (原始结果文件按 sha256 核对)
    runs = load_runs()
    sel = source_selection(runs)
    with open(SEL_JSON, encoding="utf-8") as f:
        locked = json.load(f)
    for k, v in locked["runs"].items():
        assert v["head_sha256"] == runs[k]["head_sha256"] and v["result_sha256"] == runs[k]["result_sha256"], \
            f"{k}: 分类头或结果 json 在锁定之后被改动"
    assert json.dumps(locked["selection"], sort_keys=True) == json.dumps(sel["selection"], sort_keys=True) \
        and json.dumps(locked["table"], sort_keys=True) == json.dumps(sel["table"], sort_keys=True), "重算的源域结果与锁定的文件不同"
    with open(TGT_JSON, encoding="utf-8") as f:
        tgt = json.load(f)
    assert tgt["selection_sha256"] == _sha(SEL_JSON), "事后评估引用的选择文件与当前文件不同"
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", tgt["selection_lock_commit"], "HEAD"], capture_output=True)
    assert anc.returncode == 0, "事后评估记录的锁定提交不是当前 HEAD 的祖先"
    T = {a: [tgt["heads"][f"CUB_p1_{tag}_seed{s}"]["target_acc"] for s in SEEDS] for a, tag in zip(DOSES, TAGS)}
    for k, v in locked["runs"].items():
        assert tgt["heads"][k]["head_sha256"] == v["head_sha256"], f"{k}: 事后评估用的分类头不是锁定时的那一份"
    keys = ["mean_aug_acc", "clean_acc", "mean_aug_ce"] + ["aug_acc:" + an for an in AUGS]
    fin = final_analysis(locked, T, {k: curves(runs, k) for k in keys})
    fin["target_mean_ce"] = {a: float(np.mean([tgt["heads"][f"CUB_p1_{tag}_seed{s}"]["target_ce"] for s in SEEDS]))
                             for a, tag in zip(DOSES, TAGS)}
    out = {**fin, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": _sha(SEL_JSON), "target_eval_sha256": _sha(TGT_JSON), "target_cache": tgt["target_cache"]}
    _write_json(FIN_JSON, out)
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write(final_md(locked, fin, T))
    v = fin["verdict"]
    print(f"[final] 锁定于 {why[:7]}; α̂ = {locked['selection']['primary']['alpha_hat']:g} -> 目标域 {fin['A_T_of_selected']:.2f}%; "
          f"α* = {fin['alpha_star']:g} ({fin['A_T_of_best']:.2f}%); ρ = {fin['spearman']['mean_aug_acc']:.3f}; "
          f"R_sel = {fin['regret_pp']['primary']:.2f}; 追踪 {v['tracks_target']} / 可用 {v['selection_usable']}")
    print(f"[saved] {FIN_JSON}、{FIN_MD}")
    return 0


# ---------------- 自测 ----------------
def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    check("T1 argmax 并列取较小的 α", argbest({0.0: 50.0, 1.0: 55.0, 2.0: 55.0}) == 1.0 and argbest({0.0: 50.0}, "min") == 0.0)
    check("T2 argmin 并列取较小的 α", argbest({0.0: 2.2, 1.0: 2.1, 2.0: 2.1}, "min") == 1.0)
    check("T3 Spearman: 完全单调 = 1, 完全反号 = -1", abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
          and abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1) < 1e-12)
    # 手算的并列秩: x = [1, 2, 2, 4] -> 秩 [1, 2.5, 2.5, 4]; y = [1, 3, 2, 4] -> [1, 3, 2, 4]
    rx, ry = _rank([1, 2, 2, 4]), _rank([1, 3, 2, 4])
    check("T4 平均秩处理并列", np.allclose(rx, [1, 2.5, 2.5, 4]) and np.allclose(ry, [1, 3, 2, 4]))
    check("T5 Spearman 用平均秩计算 (手算值 0.948683)", abs(spearman([1, 2, 2, 4], [1, 3, 2, 4]) - 3 / np.sqrt(10)) < 1e-12)
    p = spearman_exact_p(1.0, 7)
    check("T6 精确置换 p: ρ = 1 时双侧 p = 2/5040", abs(p - 2 / 5040) < 1e-15, f"p = {p:.6f}")
    p_crit = spearman_exact_p(0.7857142857142857, 7)
    check("T7 n = 7 的临界值 0.7857 对应双侧 p < 0.05, 而下一档 0.75 不低于 0.05",
          p_crit < 0.05 and spearman_exact_p(0.75, 7) >= 0.05, f"p(0.786) = {p_crit:.4f}, p(0.75) = {spearman_exact_p(0.75, 7):.4f}")
    check("T8 Spearman 对并列退化返回 nan 而不崩溃", spearman([1, 1, 1], [1, 2, 3]) != spearman([1, 1, 1], [1, 2, 3]))
    check("T9 过正则化四条判定分支",
          over_reg_verdict(True, True).startswith("识别成功") and over_reg_verdict(True, False).startswith("未识别")
          and "不一致" in over_reg_verdict(False, True) and over_reg_verdict(False, False).startswith("无过正则化"))
    # ---- 合成的完整流程: 源域主指标与目标域秩次相同 (0 < 0.5 < 1 < 4 < 2) -> 选中 α=2, regret = 0; 干净验证是平的 ----
    doses = [0.0, 0.5, 1.0, 2.0, 4.0]
    src_acc = {0.0: [50.0, 50.5, 49.5], 0.5: [55.0, 55.5, 54.5], 1.0: [58.0, 58.5, 57.5], 2.0: [61.0, 61.5, 60.5], 4.0: [57.5, 58.0, 57.0]}
    tgt = {0.0: [50.0, 50.4, 49.6], 0.5: [54.0, 54.4, 53.6], 1.0: [56.0, 56.4, 55.6], 2.0: [57.0, 57.4, 56.6], 4.0: [55.0, 55.4, 54.6]}
    flat = {a: [84.0, 84.1, 83.9] for a in doses}
    inv = {a: [round(3.0 - v / 100, 4) for v in src_acc[a]] for a in doses}
    sel = {"selection": {"primary": {"alpha_hat": 2.0}, "secondary": {"mean_aug_ce": 2.0, "clean_acc": 0.0, "clean_ce": 0.0},
                         "per_seed": {"mean_aug_acc": [2.0, 2.0, 2.0]}}}
    fin = final_analysis(sel, tgt, {"mean_aug_acc": src_acc, "clean_acc": flat, "mean_aug_ce": inv})
    check("T10 合成: ρ = 1, 选中 α=2, R_sel = 0, 默认 α=1 的 regret = 1", abs(fin["spearman"]["mean_aug_acc"] - 1) < 1e-12
          and fin["alpha_star"] == 2.0 and abs(fin["regret_pp"]["primary"]) < 1e-9 and abs(fin["regret_pp"]["default_a1"] - 1.0) < 1e-9)
    check("T11 合成: 干净验证是平的 -> ρ 无定义, 其 regret = 7 (选中 α=0), 优于干净验证 = True",
          fin["spearman"]["clean_acc"] != fin["spearman"]["clean_acc"] and abs(fin["regret_pp"]["clean_acc"] - 7.0) < 1e-9
          and fin["verdict"]["better_than_clean_selection"] is True and fin["verdict"]["tracks_target"] is True
          and fin["verdict"]["selection_usable"] is True and fin["verdict"]["beats_default_a1"] is True)
    sel_x = json.loads(json.dumps(sel))
    sel_x["selection"]["secondary"]["clean_acc"] = 2.0                    # 干净验证也选中了最优剂量 -> 不算"优于"它
    fin_x = final_analysis(sel_x, tgt, {"mean_aug_acc": src_acc, "clean_acc": flat, "mean_aug_ce": inv})
    check("T11b 合成: 干净验证同样选中 α=2 时, '优于干净验证' = False",
          fin_x["verdict"]["better_than_clean_selection"] is False and abs(fin_x["regret_pp"]["clean_acc"]) < 1e-9)
    check("T11c 合成结果可以写成标准 JSON (ρ 无定义时写 null, 不报错)",
          json.dumps(_json_safe(fin), allow_nan=False).count("null") >= 2)
    check("T12 合成: α=4 在目标域低 2 个百分点且源域也压低了 3.5 个百分点 -> 识别成功",
          fin["over_reg"]["T_over"] and fin["over_reg"]["S_over"] and fin["over_reg"]["verdict"] == "识别成功"
          and abs(fin["over_reg"]["source_drop_pp"] - 3.5) < 1e-9 and abs(fin["over_reg"]["target_drop_pp"] - 2.0) < 1e-9)
    # 源域验证看不出 α=4 的损害 (与 α=2 持平) -> 未识别
    flat4 = {a: list(v) for a, v in src_acc.items()}
    flat4[4.0] = list(src_acc[2.0])
    fin2 = final_analysis(sel, tgt, {"mean_aug_acc": flat4, "clean_acc": flat, "mean_aug_ce": inv})
    check("T13 合成: 源域看不出 α=4 的损害 -> 未识别", fin2["over_reg"]["verdict"].startswith("未识别"))
    # 目标域上 α=4 最优 -> 无过正则化可识别; 此时不允许被选中的 α=4 触发 T_over
    tgt3 = {a: list(v) for a, v in tgt.items()}
    tgt3[4.0] = [60.0, 60.4, 59.6]
    sel3 = json.loads(json.dumps(sel))
    sel3["selection"]["primary"]["alpha_hat"] = 4.0
    fin3 = final_analysis(sel3, tgt3, {"mean_aug_acc": flat4, "clean_acc": flat, "mean_aug_ce": inv})
    check("T14 合成: 目标域上 α=4 最优 -> 无过正则化可识别, regret = 0",
          fin3["over_reg"]["verdict"].startswith("无过正则化") and abs(fin3["regret_pp"]["primary"]) < 1e-9)
    # ---- 锁定检查: 临时 git 仓库 ----
    with tempfile.TemporaryDirectory() as d:
        g = ["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", d], check=True)
        subprocess.run(g + ["commit", "-q", "--allow-empty", "-m", "init"], check=True)
        fp = os.path.join(d, "sel.json")
        r1 = _locked(fp, root=d)                                        # 不存在
        with open(fp, "w", encoding="utf-8") as f:
            f.write("{}")
        r2 = _locked(fp, root=d)                                        # 存在但未跟踪
        subprocess.run(g + ["add", "sel.json"], check=True)
        subprocess.run(g + ["commit", "-q", "-m", "lock"], check=True)
        r3 = _locked(fp, root=d)                                        # 已提交
        with open(fp, "a", encoding="utf-8") as f:
            f.write(" ")
        r4 = _locked(fp, root=d)                                        # 提交后又改动
        r5 = _locked(fp, root=os.path.dirname(os.path.abspath(__file__)))   # 文件在仓库之外
        head = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True).stdout.decode().strip()
    check("T15 锁定检查: 不存在 / 未跟踪 / 已提交 / 提交后又改动 / 在仓库之外",
          (r1[0], r2[0], r3[0], r4[0], r5[0]) == (False, False, True, False, False) and r3[1].strip() == head,
          f"{(r1[0], r2[0], r3[0], r4[0], r5[0])}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=["source", "final"])
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    if a.stage == "source":
        return stage_source()
    if a.stage == "final":
        return stage_final()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
