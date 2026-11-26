"""Stage 1C 八臂（LD-DRO 正式训练）的分析与判定。预注册见 RESULTS.md §3z —— 规则与阈值写在那里，本脚本只执行。

故意分成两次运行，中间隔一次 git 提交（与 P1 的 p1_independent_report.py 同一流程）：
  --stage source : 只读 57 个 LD run 与 P1 基线的**源域**验证字段；逐项核对配置与预注册一致（run_config 按
                   run_matrix 的 expected 比对）、断言结果里没有任何目标域数字、核对 57 个 run 之间的共享配置
                   一致、核对 P1 基线与 LD run 在共有配置键上逐项一致（复用基线的前提）、核对标定文件身份；
                   按预注册规则（P1 冻结的规则：heldout_val 7 种扰动均值 Acc，3 seeds 均值最大，并列取较小
                   强度）对 5 个扫描臂各选一个强度；写出 s1c_source_selection.{json,md} —— 提交锁定。
  --stage final  : 要求选择文件已被 git 跟踪且无未提交修改；重算源域并与锁定文件逐项比对；读入
                   eval_heads_targets.py --stage1c 的事后目标域评估与 P1 的事后评估（CE-Only / DDO α=1 / α=2
                   基线，复用 p1_target_eval.json）；按预注册门槛（目标域平均提升 ≥ 0.5 pp 且 ≥ 2/3 配对 seed
                   为正 —— 研发门槛，不是统计显著性）做配对判读；写 s1c_final_report.{json,md}。
  --selftest     : 合成数据上的规则自测（选择并列、门槛分支、配对差、核心判读）。

边界（预注册 §3z）：目标域评估只在源域选择锁定之后；CUB 是开发基准；P1 只验证过 DDO 剂量选择，LD 强度选择
是同一规则的第一次新验证；固定强度对照与选参后的性能对照分开报告；不把三 seeds 均值差当显著性证明。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from p1_independent_report import _locked, argbest  # noqa: E402  复用 P1 的锁定检查与并列规则
from run_matrix import COMMON, MATRICES, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1C-Training")
SEL_JSON = os.path.join(OUT_DIR, "s1c_source_selection.json")
SEL_MD = os.path.join(OUT_DIR, "s1c_source_selection.md")
FIN_JSON = os.path.join(OUT_DIR, "s1c_final_report.json")
FIN_MD = os.path.join(OUT_DIR, "s1c_final_report.md")
TGT_JSON = os.path.join(OUT_DIR, "s1c_target_eval.json")
CALIB_JSON = os.path.join("experiment_results", "Stage1C-Radius", "stage1c_calibration_traincore.json")
P1_TGT = os.path.join("experiment_results", "Stage0D-Independent", "p1_target_eval.json")
P1_BASE_TAGS = ("dose_a0", "dose_a1", "dose_a2")
SEEDS = [1, 2, 3]
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]
SCAN_ARMS = [("lang_full", "langfull", "Language-Full"), ("rand_full", "randfull", "Random-Full"),
             ("lang32", "lang32", "Language-32"), ("identity", "identity", "Identity-H"),
             ("isda", "isda", "Language-ISDA")]
SINGLE_ARMS = [("identity_grad", "Identity-H 梯度配平对照（ε_grad）"),
               ("isda_grad", "Language-ISDA 梯度配平对照（λ_grad）"),
               ("randfull_same", "Random-Full 同半径对照（ε 与 Language-Full ×1 逐位相同）"),
               ("pgd", "Language-PGD 锚点（ε 与 Language-Full ×1 逐位相同，PGD-5×1）")]
STRENGTHS = ["0.5", "1", "2"]
MTAG = {"0.5": "x0p5", "1": "x1", "2": "x2"}
PRIMARY = "mean_aug_acc"
# ---- 预注册阈值 (RESULTS §3z); 不改 ----
GATE_PP = 0.5             # 初步研发门槛: 目标域平均提升 ≥ 0.5 个百分点
GATE_POS = 2              # 且 ≥ 2/3 配对 seed 为正（3 seeds）
N_TRAIN, N_VAL = 4790, 1200
# 配置一致性比较时跳过的键: run 标识与各臂自己的强度参数
CFG_SKIP = {"run_tag", "seed", "alpha", "ld_mode", "ld_eps", "ld_lam", "ld_ell", "pgd_steps", "pgd_restarts",
            "pgd_seed", "ld_calib", "device", "wandb"}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _same(got, want):
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return got == want


def _tag_of(arm_tag, m, s):
    return f"CUB_s1c_{arm_tag}_{MTAG[m]}_seed{s}"


def load_runs():
    """读取 57 个 LD run 的源域验证字段 + P1 基线；任何一项核对不过就抛错（不产出部分结果）。"""
    cal_sha = sha256_file(CALIB_JSON)
    runs, shared, ref_cfg = {}, None, None
    for ds, tag, extra in MATRICES["stage1c"][1]:
        for s in SEEDS:
            name = f"{ds}_{tag}_seed{s}"
            rp, hp = f"logs/results/{name}.json", f"logs/heads/{name}.pt"
            with open(rp, encoding="utf-8") as f:
                r = json.load(f)
            cfg = r["run_config"]
            assert result_is_complete(rp, expected(COMMON + extra, ds, s, tag)), \
                f"{name}: 结果不完整或与矩阵命令行不一致"
            assert r.get("target_acc") is None and (r.get("final_diag") or {}).get("target_acc") is None \
                and all(e["target_acc"] is None for e in r["epoch_metrics"]), \
                f"{name}: 结果里出现了目标域数字 —— 违反预注册的先后顺序"
            ts, sv, ld = r["train_subset"], r["source_val"], r["ld_config"]
            assert (ts["n_train"], ts["n_heldout"]) == (N_TRAIN, N_VAL) and sv["n"] == N_VAL \
                and sorted(sv["aug"]) == sorted(AUGS) and ts["split_cache_sha256"] == sv["cache_sha256"], \
                f"{name}: 训练子集 / 验证集不符"
            assert ld["mode"] == cfg["ld_mode"] and abs(ld["eps"] - float(cfg["ld_eps"])) < 1e-12 \
                and abs(ld["lam"] - float(cfg["ld_lam"])) < 1e-12 and ld["generator"]["ell"] == cfg["ld_ell"], \
                f"{name}: ld_config 与 run_config 不一致"
            assert ld["calibration"]["sha256"] == cal_sha, f"{name}: 标定文件与当前的不一致"
            sk = json.dumps(sorted((k, str(v)) for k, v in cfg.items() if k not in CFG_SKIP), ensure_ascii=False)
            if shared is None:
                shared, ref_cfg = sk, dict(cfg)
            assert sk == shared, f"{name}: 与其他 run 的共享配置不一致"
            runs[name] = {"tag": tag, "seed": s, "head_sha256": sha256_file(hp), "result_sha256": sha256_file(rp),
                          "clean_acc": sv["clean"]["acc"], "clean_ce": sv["clean"]["ce"],
                          "mean_aug_acc": sv["mean_aug_acc"], "mean_aug_ce": sv["mean_aug_ce"],
                          "elapsed": r.get("elapsed_seconds")}
    # ---- P1 基线复用 (预注册 §3z 的前提): 与 LD run 在共有的配置键上逐项一致 ----
    p1_sha = {}
    for tag in P1_BASE_TAGS:
        for s in SEEDS:
            name = f"CUB_p1_{tag}_seed{s}"
            with open(f"logs/results/{name}.json", encoding="utf-8") as f:
                r = json.load(f)
            cfg = r["run_config"]
            diff = {k: (cfg.get(k), ref_cfg.get(k)) for k in cfg
                    if k not in CFG_SKIP | {"descriptor_bank"} and not _same(cfg.get(k), ref_cfg.get(k))}
            assert not diff, f"P1 基线 {name} 与 LD run 的共享配置不一致: {diff}"
            p1_sha[name] = {"head_sha256": sha256_file(f"logs/heads/{name}.pt"),
                            "result_sha256": sha256_file(f"logs/results/{name}.json"), "alpha": float(cfg["alpha"])}
    return runs, cal_sha, p1_sha


def selection(runs, cal_sha):
    """预注册的选择规则 = P1 冻结的规则: 主指标 mean_aug_acc, 3 seeds 均值最大, 并列取较小强度。"""
    with open(CALIB_JSON, encoding="utf-8") as f:
        cal = json.load(f)
    scan = {}
    for ell, tag, label in SCAN_ARMS:
        mean, per_seed_acc, clean, aug_ce = {}, {}, {}, {}
        for m in STRENGTHS:
            v = [runs[_tag_of(tag, m, s)]["mean_aug_acc"] for s in SEEDS]
            mean[m] = float(np.mean(v))
            per_seed_acc[m] = v
            clean[m] = float(np.mean([runs[_tag_of(tag, m, s)]["clean_acc"] for s in SEEDS]))
            aug_ce[m] = float(np.mean([runs[_tag_of(tag, m, s)]["mean_aug_ce"] for s in SEEDS]))
        scan[tag] = {"label": label, "ell": ell, "anchor": float(cal["arms"][ell]["anchor"]),
                     "equation": cal["arms"][ell]["equation"], "mean_aug_acc": mean, "per_seed_acc": per_seed_acc,
                     "clean_acc": clean, "mean_aug_ce": aug_ce,
                     "alpha_hat_strength": argbest(mean, "max"),
                     "per_seed_strength": [argbest({m: per_seed_acc[m][i] for m in STRENGTHS}, "max") for i in range(len(SEEDS))],
                     "secondary": {"clean_acc": argbest(clean, "max"), "mean_aug_ce": argbest(aug_ce, "min")}}
    singles = {t: {"label": lb, "mean_aug_acc": [runs[f"CUB_s1c_{t}_seed{s}"]["mean_aug_acc"] for s in SEEDS],
                   "clean_acc": [runs[f"CUB_s1c_{t}_seed{s}"]["clean_acc"] for s in SEEDS],
                   "elapsed": [runs[f"CUB_s1c_{t}_seed{s}"]["elapsed"] for s in SEEDS]} for t, lb in SINGLE_ARMS}
    return {"scan": scan, "singles": singles, "primary": PRIMARY,
            "rule": "3 seeds 均值最大; 并列取较小强度 (P1 冻结的规则)",
            "thresholds": {"GATE_PP": GATE_PP, "GATE_POS": GATE_POS},
            "scan_arms": {t: {"anchor": float(cal["arms"][e]["anchor"]),
                              "scan_values": {m: float(cal["arms"][e]["scan"][m]) for m in STRENGTHS}}
                          for e, t, _ in SCAN_ARMS}}


def _paired(x, y):
    """逐 seed 配对差 x − y。"""
    d = [xi - yi for xi, yi in zip(x, y)]
    return {"diff_per_seed": d, "n_positive": int(sum(v > 0 for v in d)), "mean": float(np.mean(d))}


def gate(p):
    """预注册门槛: 平均提升 ≥ GATE_PP 且 ≥ GATE_POS/3 个 seed 为正 —— 研发门槛, 不是统计显著性。"""
    return bool(p["mean"] >= GATE_PP and p["n_positive"] >= GATE_POS)


def selected_target(sel, tgt):
    """选中的强度对应的逐 seed 目标域精度。tgt: {run_name: target_acc}"""
    out = {}
    for tag, s in sel["scan"].items():
        m = s["alpha_hat_strength"]
        out[tag] = {"strength": m, "eps_or_lam": s["anchor"] * float(m),
                    "target_acc": [tgt[_tag_of(tag, m, se)] for se in SEEDS]}
    if any(k.startswith("CUB_s1c_pgd_seed") for k in tgt):        # PGD 是固定锚点, 不参与选择; 缺席时跳过 (自测用)
        out["pgd"] = {"strength": "1 (锚点)", "target_acc": [tgt[f"CUB_s1c_pgd_seed{se}"] for se in SEEDS]}
    return out


def final_analysis(sel_t, B):
    """sel_t: selected_target 的输出; B: 基线 {ce/a1/a2: 逐 seed}。判读按预注册 §3z, 门槛由代码给出。"""
    out = {"selected_target": {}, "paired": {}, "gates": {}}
    for tag, v in sel_t.items():
        out["selected_target"][tag] = {"strength": v["strength"], "target_acc": v["target_acc"],
                                       "mean": float(np.mean(v["target_acc"]))}
    for tag in sel_t:
        for bname, bkey in (("vs_DDO_a2", "a2"), ("vs_DDO_a1", "a1"), ("vs_CE", "ce")):
            p = _paired(sel_t[tag]["target_acc"], B[bkey])
            out["paired"][f"{tag}_{bname}"] = p
            out["gates"][f"{tag}_{bname}"] = gate(p)
    for a, b, name in (("langfull", "randfull", "langfull_vs_randfull_same_budget"),
                       ("langfull", "identity", "langfull_vs_identity_same_budget"),
                       ("langfull", "lang32", "langfull_vs_lang32_same_budget"),
                       ("langfull", "isda", "langfull_vs_isda_same_budget")):
        p = _paired(sel_t[a]["target_acc"], sel_t[b]["target_acc"])
        out["paired"][name] = p
        out["gates"][name] = gate(p)
    out["core_questions"] = {
        "q1_langfull_beats_selected_ddo_a2": out["gates"]["langfull_vs_DDO_a2"],
        "q2_langfull_beats_spectral_random_same_budget": out["gates"]["langfull_vs_randfull_same_budget"],
    }
    return out


# ---------------- 报告 ----------------
def _f(x, nd=2):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def source_md(sel, cal_sha, base_ok):
    L = ["# Stage 1C 八臂训练：源域验证与强度选择（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3z。57 个 LD run 与 P1 的 21 个 run 一样以 `--target_eval none` 训练"
         "（train_core 4790 / heldout_val 1200；训练期间不读目标域特征缓存），结果 json 中没有任何目标域数字"
         "（脚本已逐个断言）。", "",
         f"- 选择规则 = P1 冻结的规则：**heldout_val 7 种光度扰动的均值 Acc，3 seeds 均值最大，并列取较小强度**。"
         "P1 只验证过 DDO 剂量选择，LD 强度选择是同一规则的第一次新验证。",
         f"- 强度锚点：统一批次标定 `stage1c_calibration_traincore.json`（sha256 `{cal_sha[:16]}`）。",
         f"- P1 基线复用的配置一致性核对：{'通过' if base_ok else '失败'}（CE-Only / DDO α=1 / α=2 与 LD run 在"
         "两个配置共有的全部键上逐项一致）。", "",
         "## 扫描臂（锚点 × {0.5, 1, 2}）", "",
         "| 臂 | 椭球 | 锚点 | 强度 | 源域增强 Acc（主指标，3 seeds 均值） | 逐 seed | 干净 Acc | 扰动均值 CE |",
         "|---|---|---|---|---|---|---|---|"]
    for tag, s in sel["scan"].items():
        for m in STRENGTHS:
            mark = " **← 选中**" if m == s["alpha_hat_strength"] else ""
            L.append(f"| {s['label']} | {s['ell']} | {_f(s['anchor'], 4)}（{s['equation']}） | {m}× | "
                     f"**{_f(s['mean_aug_acc'][m])}**{mark} | {' / '.join(_f(x) for x in s['per_seed_acc'][m])} | "
                     f"{_f(s['clean_acc'][m])} | {_f(s['mean_aug_ce'][m], 4)} |")
    L += ["", "## 固定强度对照臂（不扫描）", "",
          "| 臂 | 源域增强 Acc（3 seeds 均值） | 逐 seed |", "|---|---|---|"]
    for t, lb in SINGLE_ARMS:
        v = sel["singles"][t]["mean_aug_acc"]
        L.append(f"| {lb} | {_f(float(np.mean(v)))} | {' / '.join(_f(x) for x in v)} |")
    L += ["", "## 各扫描臂的选择结果", ""]
    for tag, s in sel["scan"].items():
        L.append(f"- **{s['label']}：强度 = {s['alpha_hat_strength']}× 锚点**（ε/λ = "
                 f"{_f(s['anchor'] * float(s['alpha_hat_strength']), 4)}）；逐 seed 的选择：{s['per_seed_strength']}；"
                 f"辅助指标：干净 Acc → {s['secondary']['clean_acc']}×，扰动均值 CE → {s['secondary']['mean_aug_ce']}×")
    L += ["", "## 平均训练成本（秒 / run，含最终评估；PGD 的搜索开销单独可见）", ""]
    for t, lb in SINGLE_ARMS:
        e = sel["singles"][t]["elapsed"]
        L.append(f"- {lb}：{' / '.join(_f(x, 0) for x in e)}")
    L.append("- 扫描臂的逐 run 耗时见结果 json 的 elapsed_seconds")
    return "\n".join(L)


def final_md(fin, sel_t, B):
    L = ["# Stage 1C 八臂训练：最终报告（源域强度选择锁定之后的目标域事后评估）", "",
         "预注册见 `RESULTS.md` §3z。目标域 = CUB-200-Painting 测试集；57 个 LD run 的目标域精度在源域选择提交锁定"
         "之后才用保存的分类头评估（`eval_heads_targets.py --stage1c`）；CE-Only / DDO α=1 / α=2 复用 P1 的事后评估。", "",
         "## 主表：选参后的目标域 Acc（3 seeds）", "",
         "| 方法 | 强度 | 目标域 Acc（均值 ± sd） | 逐 seed |", "|---|---|---|---|"]
    rows = [("CE-Only（P1 基线）", "—", B["ce"]), ("Full DDO α = 1（P1 基线）", "—", B["a1"]),
            ("Full DDO α = 2（P1 源域选参基线）", "—", B["a2"])]
    for tag, v in fin["selected_target"].items():
        lb = {"langfull": "LD-Language-Full", "randfull": "LD-Random-Full", "lang32": "LD-Language-32",
              "identity": "LD-Identity-H", "isda": "Language-ISDA", "pgd": "Language-PGD（锚点）"}.get(tag, tag)
        rows.append((lb, str(v["strength"]), v["target_acc"]))
    for lb, st, v in rows:
        L.append(f"| {lb} | {st} | **{_f(float(np.mean(v)))}** ± {_f(float(np.std(v, ddof=1)))} | "
                 f"{' / '.join(_f(x) for x in v)} |")
    L += ["", "## 预注册的两条核心判读（门槛：平均提升 ≥ 0.5 pp 且 ≥ 2/3 配对 seed 为正；研发门槛，不是显著性）", ""]
    q = fin["core_questions"]
    p1 = fin["paired"]["langfull_vs_DDO_a2"]
    p2 = fin["paired"]["langfull_vs_randfull_same_budget"]
    L.append(f"1. **Language-Full（源域选参）能否超过 DDO α=2（源域选参）**：Δ = {_f(p1['mean'])} pp"
             f"（逐 seed {' / '.join(_f(x) for x in p1['diff_per_seed'])}，{p1['n_positive']}/3 为正）→ "
             f"**{'过门槛' if q['q1_langfull_beats_selected_ddo_a2'] else '未过门槛'}**")
    L.append(f"2. **能否超过同谱 Random-Full（同预算选参）**：Δ = {_f(p2['mean'])} pp"
             f"（逐 seed {' / '.join(_f(x) for x in p2['diff_per_seed'])}，{p2['n_positive']}/3 为正）→ "
             f"**{'过门槛' if q['q2_langfull_beats_spectral_random_same_budget'] else '未过门槛'}**"
             "（这是语言域几何能否作为创新依据的关键对照）")
    L += ["", "## 全部配对比较（选中配置，逐 seed 配对）", "",
          "| 比较 | Δ（均值，pp） | 逐 seed 差 | 正向 seeds | 过门槛 |", "|---|---|---|---|---|"]
    for name, p in fin["paired"].items():
        L.append(f"| {name} | {_f(p['mean'])} | {' / '.join(_f(x) for x in p['diff_per_seed'])} | "
                 f"{p['n_positive']}/3 | {fin['gates'][name]} |")
    L += ["", "## 固定强度对照（锚点 ×1，机制对照；与选参后的性能对照分开读）", "",
          "| 比较（都在锚点 ×1） | Δ（均值，pp） | 逐 seed 差 | 正向 seeds |", "|---|---|---|---|"]
    for name, p in fin["fixed_anchor"].items():
        L.append(f"| {name} | {_f(p['mean'])} | {' / '.join(_f(x) for x in p['diff_per_seed'])} | {p['n_positive']}/3 |")
    pq = fin["pgd_search_quality"]
    L += ["", f"## Language-PGD 的搜索质量抽查（预注册 §3z；{pq['n_records']} 条 grad_diag 记录）", "",
          f"- 链 CE ≤ PGD-训练(PGD-5×1) ≤ PGD-强(50×4+闭式起点) ≤ LD 在全部记录上成立：{pq['all_chain_ok']}；"
          f"强搜索相对训练搜索的最大增量 {pq['worst_strong_minus_train']:.5f}"
          "（增量小说明训练用的 PGD-5 在诊断批次上已接近强搜索的下界；增量大说明训练搜索没追上最坏情况）。"]
    L += ["", "## 边界", "",
          "- 目标域结果只在源域强度选择提交锁定之后评估；顺序由 git 提交记录保证。",
          "- CUB 是开发基准；P1 的选择规则只对 DDO 剂量验证过，LD 强度选择是它的第一次新验证。",
          "- 训练集是完整训练集的 80%，不能与完整训练集的历史目标域精度当作同一协议比较。",
          "- 门槛（≥ 0.5 pp 且 ≥ 2/3 seeds）是**继续推进的研发门槛**，不是统计显著性证明。",
          "- PGD 只跑了标定锚点（PGD-5×1）：不能把\"未调参的 PGD 与已调参的 LD 之差\"解释为算法全面优势。", ""]
    return "\n".join(L)


def _write_json(path, obj):
    def safe(o):
        if isinstance(o, dict):
            return {k: safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def stage_source():
    runs, cal_sha, p1_sha = load_runs()
    sel = selection(runs, cal_sha)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3z",
           "target_info_present": False, "calibration_sha256": cal_sha, "selection": sel,
           "runs": {k: {kk: v[kk] for kk in ("tag", "seed", "head_sha256", "result_sha256")} for k, v in runs.items()},
           "p1_baseline_sha256": p1_sha}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_json(SEL_JSON, out)
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write(source_md(sel, cal_sha, base_ok=True))
    s = sel["scan"]
    print("[source] 57 个 run 核对通过 (无任何目标域数字; P1 基线配置一致); 选择: " +
          ", ".join(f"{v['label']} {v['alpha_hat_strength']}×" for v in s.values()))
    print(f"[saved] {SEL_JSON}、{SEL_MD} —— 提交之后才能运行 --stage final / eval_heads_targets.py --stage1c")
    return 0


def stage_final():
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: 源域选择结果尚未锁定 —— " + why)
        return 2
    runs, cal_sha, p1_sha = load_runs()
    sel_now = selection(runs, cal_sha)
    with open(SEL_JSON, encoding="utf-8") as f:
        locked = json.load(f)
    for k, v in locked["runs"].items():
        assert v["head_sha256"] == runs[k]["head_sha256"] and v["result_sha256"] == runs[k]["result_sha256"], \
            f"{k}: 分类头或结果 json 在锁定之后被改动"
    assert json.dumps(locked["selection"], sort_keys=True) == json.dumps(sel_now, sort_keys=True), \
        "重算的源域选择与锁定的文件不同"
    with open(TGT_JSON, encoding="utf-8") as f:
        tgt = json.load(f)
    assert tgt["selection_sha256"] == sha256_file(SEL_JSON), "事后评估引用的选择文件与当前文件不同"
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", tgt["selection_lock_commit"], "HEAD"], capture_output=True)
    assert anc.returncode == 0, "事后评估记录的锁定提交不是当前 HEAD 的祖先"
    T_all = {r: v["target_acc"] for r, v in tgt["heads"].items()}
    with open(P1_TGT, encoding="utf-8") as f:
        p1t = json.load(f)
    B = {"ce": [p1t["heads"][f"CUB_p1_dose_a0_seed{s}"]["target_acc"] for s in SEEDS],
         "a1": [p1t["heads"][f"CUB_p1_dose_a1_seed{s}"]["target_acc"] for s in SEEDS],
         "a2": [p1t["heads"][f"CUB_p1_dose_a2_seed{s}"]["target_acc"] for s in SEEDS]}
    sel_t = selected_target(sel_now, T_all)
    fin = final_analysis(sel_t, B)
    # 固定强度对照 (锚点 ×1): 机制比较, 与选参后的对照分开 (预注册 §3z)
    x1 = [T_all[f"CUB_s1c_langfull_x1_seed{s}"] for s in SEEDS]
    fin["fixed_anchor"] = {
        "langfull_x1_vs_randfull_same": _paired(x1, [T_all[f"CUB_s1c_randfull_same_seed{s}"] for s in SEEDS]),
        "langfull_x1_vs_pgd_x1": _paired(x1, [T_all[f"CUB_s1c_pgd_seed{s}"] for s in SEEDS]),
        "langfull_x1_vs_identity_x1": _paired(x1, [T_all[f"CUB_s1c_identity_x1_seed{s}"] for s in SEEDS]),
        "langfull_x1_vs_isda_x1": _paired(x1, [T_all[f"CUB_s1c_isda_x1_seed{s}"] for s in SEEDS]),
        "langfull_x1_vs_DDO_a2": _paired(x1, B["a2"]),
    }
    fin["baselines"] = B
    fin["pgd_search_quality"] = _pgd_quality()
    out = {**fin, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "selection_lock_commit": why, "selection_sha256": sha256_file(SEL_JSON),
           "target_eval_sha256": sha256_file(TGT_JSON), "target_cache": tgt["target_cache"]}
    _write_json(FIN_JSON, out)
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write(final_md(fin, sel_t, B))
    q = fin["core_questions"]
    print(f"[final] 锁定于 {why[:7]}; Language-Full 选 {sel_t['langfull']['strength']}× "
          f"-> 目标域 {float(np.mean(sel_t['langfull']['target_acc'])):.2f}% (DDO α=2: {float(np.mean(B['a2'])):.2f}%); "
          f"过门槛: q1(超 DDO α=2)={q['q1_langfull_beats_selected_ddo_a2']}, "
          f"q2(超同谱 Random)={q['q2_langfull_beats_spectral_random_same_budget']}")
    print(f"[saved] {FIN_JSON}、{FIN_MD}")
    return 0


def _pgd_quality():
    """PGD 臂的搜索质量抽查汇总 (预注册 §3z): grad_diag 的 pgd_search_quality 记录。"""
    recs = []
    for s in SEEDS:
        with open(f"logs/results/CUB_s1c_pgd_seed{s}.json", encoding="utf-8") as f:
            r = json.load(f)
        for g in r["grad_diag"]:
            if "pgd_search_quality" in g:
                recs.append({"seed": s, "tag": g["tag"], **g["pgd_search_quality"]})
    return {"n_records": len(recs), "all_chain_ok": all(x["chain_ok"] for x in recs),
            "worst_strong_minus_train": max((x["strong_minus_train"] for x in recs), default=0.0),
            "final_records": [x for x in recs if x["tag"] == "final"]}


# ---------------- 自测 ----------------
def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    check("S1 并列取较小强度 (P1 的规则)", argbest({"0.5": 55.0, "1": 55.0, "2": 54.0}, "max") == "0.5"
          and argbest({"0.5": 54.0, "1": 55.0, "2": 55.0}, "max") == "1"
          and argbest({"0.5": 54.0, "1": 54.0, "2": 55.0}, "max") == "2")
    check("S2 配对差与正向计数", _paired([57.0, 56.0, 58.0], [56.0, 56.5, 57.0]) ==
          {"diff_per_seed": [1.0, -0.5, 1.0], "n_positive": 2, "mean": 0.5})
    check("S3 门槛: ≥0.5pp 且 ≥2/3 正向 (边界值含在内)", gate({"mean": 0.5, "n_positive": 2})
          and not gate({"mean": 0.49, "n_positive": 3}) and not gate({"mean": 0.6, "n_positive": 1}))
    check("S4 强度标签与 run tag 的映射", _tag_of("langfull", "0.5", 1) == "CUB_s1c_langfull_x0p5_seed1"
          and _tag_of("langfull", "1", 3) == "CUB_s1c_langfull_x1_seed3")
    # 合成的最终判读: Language-Full 比 DDO α=2 高 (3/3) → q1 过; 比 Random-Full 平均高 0.47 → q2 不过
    sel_t = {"langfull": {"strength": "1", "target_acc": [57.0, 57.4, 56.8]},
             "randfull": {"strength": "2", "target_acc": [56.6, 56.8, 56.4]},
             "identity": {"strength": "1", "target_acc": [55.0, 55.2, 54.8]},
             "lang32": {"strength": "1", "target_acc": [55.5, 55.3, 55.4]},
             "isda": {"strength": "0.5", "target_acc": [56.0, 56.2, 55.8]},
             "pgd": {"strength": "1 (锚点)", "target_acc": [56.4, 56.6, 56.2]}}
    B = {"ce": [48.0, 48.2, 48.1], "a1": [54.5, 54.7, 54.6], "a2": [56.2, 56.0, 56.1]}
    fin = final_analysis(sel_t, B)
    check("S5 合成: q1 过门槛 (Δ=+0.63, 3/3), q2 不过 (Δ≈0.47 < 0.5)",
          fin["core_questions"]["q1_langfull_beats_selected_ddo_a2"] is True
          and fin["core_questions"]["q2_langfull_beats_spectral_random_same_budget"] is False
          and abs(fin["paired"]["langfull_vs_randfull_same_budget"]["mean"] - 0.4666666667) < 1e-6)
    check("S6 合成: vs CE / vs DDO α=1 的门槛通过, vs identity 的门槛通过 (Δ=+2.07)",
          fin["gates"]["langfull_vs_CE"] and fin["gates"]["langfull_vs_DDO_a1"]
          and fin["gates"]["langfull_vs_identity_same_budget"])
    sel_u = {**sel_t, "randfull": {**sel_t["randfull"], "target_acc": [57.1, 56.9, 57.3]}}
    fin2 = final_analysis(sel_u, B)
    check("S7 合成: Random-Full 反超时 q2 = False 且 Δ 为负",
          fin2["gates"]["langfull_vs_randfull_same_budget"] is False
          and fin2["paired"]["langfull_vs_randfull_same_budget"]["mean"] < 0)
    sel_d = {"scan": {"langfull": {"alpha_hat_strength": "2", "anchor": 5.345863580703735}}}
    st = selected_target(sel_d, {"CUB_s1c_langfull_x2_seed1": 60.0, "CUB_s1c_langfull_x2_seed2": 61.0,
                                 "CUB_s1c_langfull_x2_seed3": 62.0})
    check("S9 selected_target 取选中强度的逐 seed 值", st["langfull"]["target_acc"] == [60.0, 61.0, 62.0]
          and abs(st["langfull"]["eps_or_lam"] - 2 * 5.345863580703735) < 1e-9)
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
