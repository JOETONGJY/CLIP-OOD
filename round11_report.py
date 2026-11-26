"""第十一轮的分析与判定（预注册 RESULTS §3ad）：扩展强度扫描 + SDB-204 的 2×2 因子 + ISDA warmup。

--stage source : 读取 stage1c (57) + s1c_ext (12) + sdb22 (36) 的源域验证字段，逐项核对配置与边界
                 （无目标域数字 / 词库与椭球身份 / 标定引用 / SDB 与 P1 的共享配置一致）；
                 对三个扫描臂（Original-Language-Full、Original-ISDA、SDB-ISDA，各 5 个倍数）按 P1 冻结规则
                 选择强度；SDB+DDO 的七剂量是描述性曲线，不做选参。写 s11_source_selection.{json,md} → 提交锁定。
--stage final  : 锁定后读 eval_heads_targets.py --s11 的事后目标域评估，输出扩展曲线、两条核心判读的更新、
                 2×2 的描述符/损失/交互效应、P3 warmup 配对。写 s11_final_report.{json,md}。
--selftest     : 合成数据上的规则自测。
预注册阈值沿用 §3z：门槛 = 平均 ≥ 0.5 pp 且 ≥ 2/3 配对 seed 正向（研发门槛，不是显著性）。
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

from p1_independent_report import _locked, argbest  # noqa: E402
from run_matrix import COMMON, MATRICES, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402
from stage1c_training_report import (CALIB_JSON, GATE_POS, GATE_PP, P1_TGT, SEEDS, _paired, gate,  # noqa: E402
                                     sha256_file)
from ld_training import warmup_lam  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1C-Training")
SEL_JSON = os.path.join(OUT_DIR, "s11_source_selection.json")
SEL_MD = os.path.join(OUT_DIR, "s11_source_selection.md")
FIN_JSON = os.path.join(OUT_DIR, "s11_final_report.json")
FIN_MD = os.path.join(OUT_DIR, "s11_final_report.md")
TGT_JSON = os.path.join(OUT_DIR, "s11_target_eval.json")
SDB_CALIB = os.path.join("experiment_results", "Stage1C-Radius", "stage1c_calibration_sdb.json")
SDB_BANK = "prompts/banks/sdb-204-v1.json"
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]
MULTS = ["0.5", "1", "2", "4", "8"]
MTAG = {"0.5": "x0p5", "1": "x1", "2": "x2", "4": "x4", "8": "x8"}
DOSES = ["dose_a0", "dose_a0p125", "dose_a0p25", "dose_a0p5", "dose_a1", "dose_a2", "dose_a4"]
N_TRAIN, N_VAL = 4790, 1200
CFG_SKIP = {"run_tag", "seed", "alpha", "ld_mode", "ld_eps", "ld_lam", "ld_ell", "pgd_steps", "pgd_restarts",
            "pgd_seed", "ld_calib", "device", "wandb", "descriptor_bank", "isda_warmup_epochs"}


def _tag_of(arm, m, s):
    return f"CUB_{arm}_{MTAG[m]}_seed{s}"


def load_runs():
    """读 105 个 run（stage1c 57 + s1c_ext 12 + sdb22 36）；任一核对不过即抛错。"""
    with open(SDB_CALIB, encoding="utf-8") as f:
        scal = json.load(f)
    cal_sha, sdb_cal_sha = sha256_file(CALIB_JSON), sha256_file(SDB_CALIB)
    runs, shared = {}, {}
    for mname in ("stage1c", "s1c_ext", "sdb22"):
        for ds, tag, extra in MATRICES[mname][1]:
            for s in SEEDS:
                name = f"{ds}_{tag}_seed{s}"
                rp, hp = f"logs/results/{name}.json", f"logs/heads/{name}.pt"
                with open(rp, encoding="utf-8") as f:
                    r = json.load(f)
                cfg = r["run_config"]
                assert result_is_complete(rp, expected(COMMON + extra, ds, s, tag)), f"{name}: 不完整或不匹配"
                assert r.get("target_acc") is None and (r.get("final_diag") or {}).get("target_acc") is None \
                    and all(e["target_acc"] is None for e in r["epoch_metrics"]), f"{name}: 有目标域数字"
                ts, sv = r["train_subset"], r["source_val"]
                assert (ts["n_train"], ts["n_heldout"]) == (N_TRAIN, N_VAL) and sv["n"] == N_VAL \
                    and sorted(sv["aug"]) == sorted(AUGS) and ts["split_cache_sha256"] == sv["cache_sha256"], \
                    f"{name}: 划分/验证集不符"
                sk = json.dumps(sorted((k, str(v)) for k, v in cfg.items() if k not in CFG_SKIP),
                                ensure_ascii=False)
                grp = "sdb" if cfg.get("descriptor_bank", "original") != "original" else "orig"
                if grp in shared:
                    assert sk == shared[grp], f"{name}: 与同组 run 的共享配置不一致"
                else:
                    shared[grp] = sk
                if cfg.get("descriptor_bank", "original") != "original":
                    assert cfg["descriptor_bank"].replace("\\", "/").endswith(SDB_BANK), f"{name}: 词库不对"
                    ld = r["ld_config"]
                    if cfg["ld_mode"] == "isda":
                        assert ld["calibration"]["sha256"] == sdb_cal_sha, f"{name}: SDB 标定引用不符"
                        assert ld["generator"]["sha256"][:16] == scal["generator"]["sha256"][:16], \
                            f"{name}: SDB 椭球生成元与标定不符"
                runs[name] = {"tag": tag, "seed": s, "matrix": mname, "grp": grp,
                              "head_sha256": sha256_file(hp), "result_sha256": sha256_file(rp),
                              "clean_acc": sv["clean"]["acc"], "mean_aug_acc": sv["mean_aug_acc"],
                              "mean_aug_ce": sv["mean_aug_ce"], "elapsed": r.get("elapsed_seconds")}
    # SDB+DDO 的 α=0 与 P1 的 α=0: 词库无关 (CE 训练), 逐 seed 精度应逐位一致 —— 词库无关性核对
    for s in SEEDS:
        with open(f"logs/results/CUB_p1_dose_a0_seed{s}.json", encoding="utf-8") as f:
            p1 = json.load(f)
        a_sdb, a_p1 = runs[f"CUB_sdb_dose_a0_seed{s}"], p1
        assert abs(a_sdb["mean_aug_acc"] - a_p1["source_val"]["mean_aug_acc"]) < 1e-12, \
            f"sdb_dose_a0_seed{s}: 与 P1 α=0 不一致 —— α=0 应与词库无关"
    return runs, scal, cal_sha, sdb_cal_sha


def selection(runs, cal, sdb_cal):
    """P1 冻结规则: mean_aug_acc 3-seed 均值最大, 并列取较小倍数。三个扫描臂, 各 5 点。"""
    arms = [("langfull", "LD-Language-Full (Original)"), ("isda", "Language-ISDA (Original)"),
            ("sdb_isda", "Language-ISDA (SDB-204)")]
    scan = {}
    for tag, lb in arms:
        mean, per_seed, clean = {}, {}, {}
        for m in MULTS:
            v = [runs[_tag_of(tag, m, s)]["mean_aug_acc"] for s in SEEDS]
            mean[m] = float(np.mean(v))
            per_seed[m] = v
            clean[m] = float(np.mean([runs[_tag_of(tag, m, s)]["clean_acc"] for s in SEEDS]))
        scan[tag] = {"label": lb, "mean_aug_acc": mean, "per_seed_acc": per_seed, "clean_acc": clean,
                     "alpha_hat_strength": argbest(mean, "max"),
                     "per_seed_strength": [argbest({m: per_seed[m][i] for m in MULTS}, "max") for i in range(len(SEEDS))]}
    dose = {}
    for bank, pfx, ref in (("original", "p1", None), ("sdb", "sdb_dose", "sdb")):
        curve = {}
        for i, dt in enumerate(DOSES):
            nm = (f"CUB_p1_{dt}_seed" if bank == "original" else f"CUB_sdb_{dt}_seed")
            src = "source_val" if bank == "original" else None
            vals = []
            for s in SEEDS:
                if bank == "original":
                    with open(f"logs/results/CUB_p1_{dt}_seed{s}.json", encoding="utf-8") as f:
                        vals.append(json.load(f)["source_val"]["mean_aug_acc"])
                else:
                    vals.append(runs[f"CUB_sdb_{dt}_seed{s}"]["mean_aug_acc"])
            curve[dt] = float(np.mean(vals))
        dose[bank] = curve
    return {"scan": scan, "sdb_dose_curve": dose["sdb"], "primary": "mean_aug_acc",
            "rule": "3 seeds 均值最大; 并列取较小倍数 (P1 冻结规则)", "thresholds": {"GATE_PP": 0.5, "GATE_POS": 2}}


def selected_target(sel, tgt):
    out = {}
    for tag in sel["scan"]:
        m = sel["scan"][tag]["alpha_hat_strength"]
        out[tag] = {"strength": m, "target_acc": [tgt[_tag_of(tag, m, s)] for s in SEEDS]}
    return out


def final_analysis(sel_t, B, runs, sel):
    out = {"selected_target": {}, "paired": {}, "gates": {}}
    for tag, v in sel_t.items():
        out["selected_target"][tag] = {"strength": v["strength"], "target_acc": v["target_acc"],
                                       "mean": float(np.mean(v["target_acc"]))}
    for tag in sel_t:
        for bname, bkey in (("vs_DDO_a2", "a2"), ("vs_DDO_a1", "a1"), ("vs_CE", "ce")):
            p = _paired(sel_t[tag]["target_acc"], B[bkey])
            out["paired"][f"{tag}_{bname}"] = p
            out["gates"][f"{tag}_{bname}"] = gate(p)
    for a, b, name in (("langfull", "sdb_isda", "orig_ld_vs_sdb_isda"),
                       ("isda", "sdb_isda", "orig_isda_vs_sdb_isda")):
        p = _paired(sel_t[a]["target_acc"], sel_t[b]["target_acc"])
        out["paired"][name] = p
        out["gates"][name] = gate(p)
    out["core_questions"] = {
        "q1_langfull_beats_selected_ddo_a2": out["gates"]["langfull_vs_DDO_a2"],
        "q2_langfull_beats_spectral_random_same_budget": None,      # 随机对照未扩展; 不重判, 保留 §3aa 结论
    }
    return out


def _f(x, nd=2):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


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
    runs, scal, cal_sha, sdb_cal_sha = load_runs()
    sel = selection(runs, cal_sha, sdb_cal_sha)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3ad",
           "target_info_present": False, "calibration_sha256": cal_sha, "sdb_calibration_sha256": sdb_cal_sha,
           "selection": sel,
           "runs": {k: {kk: v[kk] for kk in ("tag", "seed", "matrix", "grp", "head_sha256", "result_sha256")}
                    for k, v in runs.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_json(SEL_JSON, out)
    L = ["# 第十一轮：扩展强度扫描与 SDB-204 2×2 的源域验证与选择（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3ad。105 个 run（stage1c 57 + s1c_ext 12 + sdb22 36）全部 `--target_eval none`，"
         "无任何目标域数字（逐个断言）。选择规则 = P1 冻结规则（heldout_val 7 种光度扰动均值 Acc，3 seeds 均值最大，"
         "并列取较小倍数）。SDB+DDO 的七剂量是描述性曲线，不做选参。", "", "## 扫描臂（5 个倍数）", "",
         "| 臂 | 倍数 | 源域增强 Acc（3 seeds 均值） | 逐 seed | 干净 Acc |", "|---|---|---|---|---|"]
    for tag, lb in (("langfull", "LD-Language-Full (Original)"), ("isda", "Language-ISDA (Original)"),
                    ("sdb_isda", "Language-ISDA (SDB-204)")):
        sc = sel["scan"][tag]
        for m in MULTS:
            mark = " **← 选中**" if m == sc["alpha_hat_strength"] else ""
            L.append(f"| {lb}{mark} | {m}× | **{_f(sc['mean_aug_acc'][m])}** | "
                     f"{' / '.join(_f(x) for x in sc['per_seed_acc'][m])} | {_f(sc['clean_acc'][m])} |")
    L += ["", "## SDB+DDO 剂量曲线（描述性；源域增强 Acc）", "",
          "| α | SDB | Original (P1) |", "|---|---|---|"]
    with open(os.path.join("experiment_results", "Stage0D-Independent", "p1_source_selection.json"),
              encoding="utf-8") as f:
        p1 = json.load(f)
    p1_curve = {str(r["alpha"]): r["mean_aug_acc"] for r in p1["table"]}
    for dt in DOSES:
        a = {"dose_a0": "0", "dose_a0p125": "0.125", "dose_a0p25": "0.25", "dose_a0p5": "0.5",
             "dose_a1": "1", "dose_a2": "2", "dose_a4": "4"}[dt]
        L.append(f"| {a} | {_f(sel['sdb_dose_curve'][dt])} | {_f(p1_curve[a])} |")
    L += ["", "## 选择结果", ""]
    for tag in sel["scan"]:
        sc = sel["scan"][tag]
        L.append(f"- **{sc['label']}：{sc['alpha_hat_strength']}×**；逐 seed：{sc['per_seed_strength']}")
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    s = sel["scan"]
    print("[source] 105 个 run 核对通过; 选择: " + ", ".join(f"{v['label']} {v['alpha_hat_strength']}×"
                                                          for v in s.values()))
    print(f"[saved] {SEL_JSON}、{SEL_MD} —— 提交之后才能运行 --stage final / eval_heads_targets.py --s11")
    return 0


def stage_final():
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: 源域选择结果尚未锁定 —— " + why)
        return 2
    runs, scal, cal_sha, sdb_cal_sha = load_runs()
    sel_now = selection(runs, cal_sha, sdb_cal_sha)
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
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", tgt["selection_lock_commit"], "HEAD"],
                         capture_output=True)
    assert anc.returncode == 0, "事后评估记录的锁定提交不是当前 HEAD 的祖先"
    T_all = {r: v["target_acc"] for r, v in tgt["heads"].items()}
    with open(P1_TGT, encoding="utf-8") as f:
        p1t = json.load(f)
    B = {"ce": [p1t["heads"][f"CUB_p1_dose_a0_seed{s}"]["target_acc"] for s in SEEDS],
         "a1": [p1t["heads"][f"CUB_p1_dose_a1_seed{s}"]["target_acc"] for s in SEEDS],
         "a2": [p1t["heads"][f"CUB_p1_dose_a2_seed{s}"]["target_acc"] for s in SEEDS]}
    sel_t = selected_target(sel_now, T_all)
    fin = final_analysis(sel_t, B, runs, sel_now)
    # 扩展曲线的目标域均值 (语言 Full 与 Original-ISDA 的 5 点; SDB-ISDA 5 点)
    curves = {}
    for arm in ("langfull", "isda", "sdb_isda"):
        curves[arm] = {m: float(np.mean([T_all[_tag_of(arm, m, s)] for s in SEEDS])) for m in MULTS}
    fin["target_curves"] = curves
    # 2×2 (描述性; 预注册 §3ad): 描述符效应 = SDB − Original (同一损失同一倍数/剂量), 损失效应 = ISDA − DDO,
    # 交互 = (描述符效应 under ISDA) − (描述符效应 under DDO), 逐 seed 配对后取均值差
    isda_eff, desc_isda, desc_ddo, inter = {}, {}, {}, {}
    for m in ("0.5", "1", "2", "4", "8"):
        o = [T_all[_tag_of("isda", m, s)] for s in SEEDS]
        sv = [T_all[_tag_of("sdb_isda", m, s)] for s in SEEDS]
        isda_eff[m] = _paired(o, B["a2"])
        desc_isda[m] = _paired(sv, o)
    for dt in DOSES:
        o = [p1t["heads"][f"CUB_p1_{dt}_seed{s}"]["target_acc"] for s in SEEDS] if os.path.exists(
            f"logs/heads/CUB_p1_{dt}_seed1.pt") else None
        sv = [T_all[f"CUB_sdb_{dt}_seed{s}"] for s in SEEDS]
        desc_ddo[dt] = _paired(sv, o) if o else None
    for m in ("0.5", "1", "2"):
        dt = {"0.5": "dose_a0p5", "1": "dose_a1", "2": "dose_a2"}[m]
        if desc_ddo.get(dt):
            inter[m] = {"desc_under_isda": desc_isda[m]["mean"], "desc_under_ddo": desc_ddo[dt]["mean"],
                        "interaction": desc_isda[m]["mean"] - desc_ddo[dt]["mean"]}
    # 匹配初始惩罚下的损失效应: ISDA@1× 的锚点按"与 DDO α=1 同等初始压力"标定 (两库各自的 ρ̄)
    match_pen = {}
    for arm, isda_tag, ddo_tag in (("original", "isda", "p1_dose_a1"), ("sdb", "sdb_isda", "sdb_dose_a1")):
        o = [p1t["heads"][f"CUB_{ddo_tag}_seed{s}"]["target_acc"] for s in SEEDS] if arm == "original"             else [T_all[f"CUB_sdb_dose_a1_seed{s}"] for s in SEEDS]
        i = [T_all[_tag_of(isda_tag, "1", s)] for s in SEEDS]
        match_pen[arm] = _paired(i, o)
    fin["factor2x2"] = {"loss_effect_isda_minus_ddoa2": isda_eff, "descriptor_effect_under_isda": desc_isda,
                        "descriptor_effect_under_ddo": desc_ddo, "interaction": inter,
                        "loss_effect_at_matched_penalty_isda1x_vs_ddo_a1": match_pen}
    # P3: warmup 配对 (runs 存在才配对; 选择锁定后的 λ)
    m_sel = sel_now["scan"]["isda"]["alpha_hat_strength"]
    warm = {}
    wkeys = [f"CUB_isda_warm_{MTAG[m_sel]}_seed{s}" for s in SEEDS]
    if all(k in tgt["heads"] for k in wkeys):                     # warmup 头由 eval_heads_targets --s11 事后评估
        w = [tgt["heads"][k]["target_acc"] for k in wkeys]
        p = _paired(w, sel_t["isda"]["target_acc"])
        warm = {"strength": m_sel, "warmup_epochs": 20, "target_acc": w, "paired_vs_fixed": p, "gate": gate(p)}
    fin["p3_warmup"] = warm
    fin["baselines"] = B
    out = {**fin, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "selection_lock_commit": why, "selection_sha256": sha256_file(SEL_JSON),
           "target_eval_sha256": sha256_file(TGT_JSON), "target_cache": tgt["target_cache"]}
    _write_json(FIN_JSON, out)
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write(final_md(fin, sel_t, B, curves))
    q = fin["core_questions"]
    print(f"[final] 锁定于 {why[:7]}; Language-Full 选 {sel_t['langfull']['strength']}× -> "
          f"{np.mean(sel_t['langfull']['target_acc']):.2f}% (DDO α=2: {np.mean(B['a2']):.2f}%); "
          f"ISDA(Orig) 选 {sel_t['isda']['strength']}× -> {np.mean(sel_t['isda']['target_acc']):.2f}%; "
          f"SDB-ISDA 选 {sel_t['sdb_isda']['strength']}× -> {np.mean(sel_t['sdb_isda']['target_acc']):.2f}%; "
          f"Q1 过门槛={q['q1_langfull_beats_selected_ddo_a2']}")
    print(f"[saved] {FIN_JSON}、{FIN_MD}")
    return 0


def final_md(fin, sel_t, B, curves):
    L = ["# 第十一轮：扩展扫描与 SDB 2×2 的最终报告（源域选择锁定之后的目标域事后评估）", "",
         "预注册见 `RESULTS.md` §3ad。门槛沿用 §3z（平均 ≥ 0.5 pp 且 ≥ 2/3 配对 seed 正向；研发门槛）。", "",
         "## 选中的目标域 Acc", "",
         "| 方法 | 倍数 | 目标域（均值） | 逐 seed |", "|---|---|---|---|"]
    for tag, v in fin["selected_target"].items():
        L.append(f"| {tag} | {v['strength']}× | **{_f(v['mean'])}** | {' / '.join(_f(x) for x in v['target_acc'])} |")
    L += ["", "## 目标域强度曲线（5 点）", "",
          "| 倍数 | LD-Language-Full (Orig) | Language-ISDA (Orig) | Language-ISDA (SDB) |", "|---|---|---|---|"]
    for m in MULTS:
        L.append(f"| {m}× | {_f(curves['langfull'][m])} | {_f(curves['isda'][m])} | {_f(curves['sdb_isda'][m])} |")
    L += ["", "## 配对判读（门槛同上）", "",
          "| 比较 | Δ（均值 pp） | 逐 seed | 正向 | 过门槛 |", "|---|---|---|---|---|"]
    for name, p in fin["paired"].items():
        L.append(f"| {name} | {_f(p['mean'])} | {' / '.join(_f(x) for x in p['diff_per_seed'])} | "
                 f"{p['n_positive']}/3 | {fin['gates'][name]} |")
    L += ["", "## 2×2 因子（描述性）", "",
          "| ISDA@倍数 − DDO α=2 | Δ | 正向 |", "|---|---|---|"]
    for m, p in fin["factor2x2"]["loss_effect_isda_minus_ddoa2"].items():
        L.append(f"| {m}× | {_f(p['mean'])} | {p['n_positive']}/3 |")
    L += ["", "| 匹配初始惩罚下的损失效应 (ISDA@1× − DDO α=1) | Δ | 正向 |", "|---|---|---|"]
    for arm, p in fin["factor2x2"]["loss_effect_at_matched_penalty_isda1x_vs_ddo_a1"].items():
        L.append(f"| {arm} | {_f(p['mean'])} | {p['n_positive']}/3 |")
    L += ["", "## 边界", "",
          "- 目标域只在源域选择锁定后评估；CUB 开发基准；3 seeds 均值差不是显著性证明。",
          "- 随机/各向同性对照未扩展到 4×/8×：若语言臂在高强度突破 DDO α=2，按 §3ad 需补同预算随机对照后才能"
          "更新 Q2 的表述。", ""]
    return "\n".join(L)


# ---------------- 自测 ----------------
def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    check("U1 并列取较小倍数", argbest({"0.5": 55.0, "1": 55.0, "2": 54.0, "4": 53.0, "8": 52.0}, "max") == "0.5"
          and argbest({"0.5": 54.0, "1": 55.0, "2": 55.0, "4": 55.0, "8": 54.0}, "max") == "1")
    check("U2 配对与门槛", gate(_paired([57.0, 56.0, 58.0], [56.0, 56.5, 57.0]))
          and not gate(_paired([56.5, 56.0, 56.4], [56.0, 56.5, 57.0])))
    check("U3 warmup 曲线", warmup_lam(100.0, 19, 20) == 100.0 and warmup_lam(100.0, 4, 20) == 25.0
          and warmup_lam(100.0, 25, 20) == 100.0)
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
