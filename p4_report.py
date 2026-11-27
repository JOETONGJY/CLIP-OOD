"""第十二轮 P4：跨数据集验证的分析与判定（预注册 RESULTS §3ai）。

每个数据集单独跑一遍（AwA2 / LADV / PACS 的缓存、划分与 λ 都是各自独立的）：
  --stage source : 读该数据集的 72 个 run（DDO 七剂量 + Language/Random/Identity 各五倍数 + lang warmup 4×/8×），
                   逐 run 核对：配置与 run_matrix.expected 一致、结果完整、无目标域数字、划分与 7 种扰动齐全、
                   λ 与重新标定的值逐位相同、生成元 sha256 与标定文件记录一致。
                   按两条规则（R_aug / R_joint）对四个可扫臂选强度；写 {ds}_source_selection.{json,md} → 提交锁定。
  --stage final  : 要求选择文件已锁定；重算源域并逐项比对；读 {ds}_target_eval.json，输出曲线、G1（语言 vs 同谱随机）、
                   Q1（语言 vs DDO 强基线，两条规则）与 regret / Spearman。写 {ds}_final_report.{json,md}。
  --selftest     : 复用 round12 的规则自测（同一套门槛/并列/regret/Spearman）。

判据沿用 §3z / §3ag：平均 ≥ 0.5 pp 且 ≥ 2/3 配对 seed 正向 —— 研发门槛，不是统计显著性。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from p1_independent_report import _locked, argbest  # noqa: E402
from round12_report import RULE_TEXT, _f, _write_json, self_test as _round12_selftest, spearman  # noqa: E402
from run_matrix import COMMON, MATRICES, P4_DS, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402
from stage1c_training_report import GATE_POS, SEEDS, _paired, gate, sha256_file  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "P4-CrossDataset")
MULTS = ["0.5", "1", "2", "4", "8"]
MTAG = {"0.5": "x0p5", "1": "x1", "2": "x2", "4": "x4", "8": "x8"}
DOSES = ["0", "0.125", "0.25", "0.5", "1", "2", "4"]
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]


def _paths(ds):
    return (os.path.join(OUT_DIR, f"{ds}_source_selection.json"),
            os.path.join(OUT_DIR, f"{ds}_source_selection.md"),
            os.path.join(OUT_DIR, f"{ds}_final_report.json"),
            os.path.join(OUT_DIR, f"{ds}_final_report.md"),
            os.path.join(OUT_DIR, f"{ds}_target_eval.json"))


def _calib(ds):
    p = os.path.join("experiment_results", "Stage1C-Radius", f"stage1c_calibration_{ds}.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f), p


def _name(ds, tag, s):
    return f"{ds}_{tag}_seed{s}"


def _arm_of(tag):
    if tag.startswith("p4_dose"):
        return "ddo"
    for a in ("lang", "rand", "ident"):
        if f"_isda_{a}_" in tag:
            return a
    raise ValueError(tag)


def _key_of(tag):
    if tag.startswith("p4_dose"):
        return tag.replace("p4_dose_", "")
    if "warm20" in tag:
        return None
    return [m for m, t in MTAG.items() if tag.endswith("_" + t)][0]


def load_runs(ds):
    cal, cal_path = _calib(ds)
    cal_sha = sha256_file(cal_path)
    entries = MATRICES[f"p4_{ds.lower()}"][1]
    assert len(entries) == 24, len(entries)
    runs = {}
    for _, tag, ex in entries:
        for s in SEEDS:
            rp = f"logs/results/{_name(ds, tag, s)}.json"
            assert os.path.exists(rp), f"缺少 {rp}"
            with open(rp, encoding="utf-8") as f:
                r = json.load(f)
            assert result_is_complete(rp, expected(COMMON + ex, ds, s, tag)), f"{rp}: 不完整或配置不匹配"
            assert r.get("target_acc") is None and (r.get("final_diag") or {}).get("target_acc") is None \
                and all(e["target_acc"] is None for e in r["epoch_metrics"]), f"{rp}: 有目标域数字"
            ts, sv = r["train_subset"], r["source_val"]
            assert sorted(sv["aug"]) == sorted(AUGS) and ts["split_cache_sha256"] == sv["cache_sha256"], \
                f"{rp}: 增强集不全或训练/验证不是同一份划分"
            cfg = r["run_config"]
            if cfg["ld_mode"] == "isda":
                assert ((r.get("ld_config") or {}).get("calibration") or {}).get("sha256") == cal_sha, \
                    f"{rp}: 标定引用不是本数据集的标定文件"
                arm = _arm_of(tag)
                key = {"lang": "isda", "rand": "isda_rand", "ident": "isda_identity"}[arm]
                m = "1" if "warm20" in tag else _key_of(tag)
                want = float(cal["arms"][key]["scan"][m])
                assert float(cfg["ld_lam"]) == want, f"{rp}: λ 与重标定值不符 ({cfg['ld_lam']} vs {want})"
            runs[_name(ds, tag, s)] = {"clean_acc": sv["clean"]["acc"], "mean_aug_acc": sv["mean_aug_acc"],
                                       "mean_aug_ce": sv["mean_aug_ce"], "elapsed": r.get("elapsed_seconds"),
                                       "head_sha256": sha256_file(f"logs/heads/{_name(ds, tag, s)}.pt"),
                                       "result_sha256": sha256_file(rp),
                                       "arm": _arm_of(tag), "key": _key_of(tag), "tag": tag}
    return runs, cal


def selection(runs, ds):
    out = {}
    for arm in ("ddo", "lang", "rand", "ident"):
        keys = DOSES if arm == "ddo" else MULTS

        def rec(a, k, s):
            if arm == "ddo":
                tag = "p4_dose_a" + a.replace(".", "p")
            else:
                tag = f"p4_isda_{arm}_{MTAG[a]}"
            return runs[_name(ds, tag, s)][k]

        aug = {a: float(np.mean([rec(a, "mean_aug_acc", s) for s in SEEDS])) for a in keys}
        clean = {a: float(np.mean([rec(a, "clean_acc", s) for s in SEEDS])) for a in keys}
        joint = {a: 0.5 * clean[a] + 0.5 * aug[a] for a in keys}
        per = {"R_aug": [], "R_joint": []}
        for s in SEEDS:
            per["R_aug"].append(argbest({a: rec(a, "mean_aug_acc", s) for a in keys}, "max"))
            per["R_joint"].append(argbest({a: 0.5 * rec(a, "clean_acc", s) + 0.5 * rec(a, "mean_aug_acc", s)
                                           for a in keys}, "max"))
        out[arm] = {"keys": keys, "mean_aug_acc": aug, "clean_acc": clean, "score_joint": joint,
                    "R_aug": argbest(aug, "max"), "R_joint": argbest(joint, "max"), "per_seed": per}
    return out


def _vec(t, ds, arm, key):
    if arm == "ddo":
        tag = "p4_dose_a" + key.replace(".", "p")
    else:
        tag = f"p4_isda_{arm}_{MTAG[key]}"
    return [t[_name(ds, tag, s)] for s in SEEDS]


def final_analysis(ds, sel, tgt):
    out = {"target_curves": {}, "paired": {}, "gates": {}, "q1": {}, "selection": {}}
    T = out["target_curves"]
    for arm in ("ddo", "lang", "rand", "ident"):
        T[arm] = {k: float(np.mean(_vec(tgt, ds, arm, k))) for k in sel[arm]["keys"]}
    p = _paired(_vec(tgt, ds, "lang", "4"), _vec(tgt, ds, "rand", "4"))
    out["paired"]["G1_lang_x4_vs_rand_x4"] = p
    out["gates"]["G1_lang_x4_vs_rand_x4"] = gate(p)
    p = _paired(_vec(tgt, ds, "lang", "4"), _vec(tgt, ds, "ident", "4"))
    out["paired"]["G1b_lang_x4_vs_ident_x4"] = p
    out["gates"]["G1b_lang_x4_vs_ident_x4"] = gate(p)
    for rule in ("R_aug", "R_joint"):
        sl, sd = sel["lang"][rule], sel["ddo"][rule]
        pa = _paired(_vec(tgt, ds, "lang", sl), _vec(tgt, ds, "ddo", sd))
        out["paired"][f"Q1_{rule}_lang_vs_ddo"] = pa
        out["gates"][f"Q1_{rule}_lang_vs_ddo"] = gate(pa)
        out["q1"][rule] = {"lang_strength": sl, "ddo_strength": sd,
                           "lang_acc": float(np.mean(_vec(tgt, ds, "lang", sl))),
                           "ddo_acc": float(np.mean(_vec(tgt, ds, "ddo", sd))),
                           "mean": pa["mean"], "n_positive": pa["n_positive"],
                           "diff_per_seed": pa["diff_per_seed"], "gate": gate(pa)}
    # warmup（两条曲线点，不参与选参）
    for m in ("4", "8"):
        a = [tgt[_name(ds, f"p4_isda_lang_{MTAG[m]}_warm20", s)] for s in SEEDS]
        b = _vec(tgt, ds, "lang", m)
        out["paired"][f"G2_lang_warm{m}_vs_fixed"] = _paired(a, b)
        out["gates"][f"G2_lang_warm{m}_vs_fixed"] = gate(_paired(a, b))
    # 选参后悔值
    for arm in ("ddo", "lang", "rand", "ident"):
        tc = T[arm]
        best_key = max(tc, key=lambda k: tc[k])
        sm = {}
        for rule, rs in (("R_aug", sel[arm]["R_aug"]), ("R_joint", sel[arm]["R_joint"])):
            sm[rule] = {"strength": rs, "target_acc": tc[rs], "regret": float(tc[best_key] - tc[rs]),
                        "rho_vs_target": spearman([sel[arm]["mean_aug_acc" if rule == "R_aug" else "score_joint"][k]
                                                   for k in sel[arm]["keys"]], [tc[k] for k in sel[arm]["keys"]]),
                        "per_seed_strength": sel[arm]["per_seed"][rule]}
        sm["post_hoc_best"] = {"strength": best_key, "target_acc": tc[best_key]}
        out["selection"][arm] = sm
    out["verdict"] = {
        "G1": ("几何成立" if out["gates"]["G1_lang_x4_vs_rand_x4"] else
               ("几何不成立" if (out["paired"]["G1_lang_x4_vs_rand_x4"]["mean"] <= 0
                                 or out["paired"]["G1_lang_x4_vs_rand_x4"]["n_positive"] < GATE_POS) else "不确定")),
        "Q1_R_joint": bool(out["gates"]["Q1_R_joint_lang_vs_ddo"]),
        "Q1_R_aug": bool(out["gates"]["Q1_R_aug_lang_vs_ddo"]),
    }
    return out


def stage_source(ds):
    runs, cal = load_runs(ds)
    sel = selection(runs, ds)
    sel_json, sel_md, _, _, _ = _paths(ds)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3ai",
           "dataset": ds, "target_info_present": False, "calibration_sha256": sha256_file(
               os.path.join("experiment_results", "Stage1C-Radius", f"stage1c_calibration_{ds}.json")),
           "selection": sel,
           "runs": {k: {kk: v.get(kk) for kk in ("arm", "key", "clean_acc", "mean_aug_acc", "head_sha256",
                                                 "result_sha256")} for k, v in runs.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_json(sel_json, out)
    L = [f"# P4 / {ds}：源域验证与选参（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3ai。72 个 run 全部 `--target_eval none`（逐个断言），λ 与重新标定的值逐位核对。", "",
         "## 源域剂量曲线（3 seeds 均值，7 扰动 Acc）", "",
         "| 臂 | " + " | ".join(f"{m}×" for m in MULTS) + " |", "|---|---|---|---|---|---|"]
    for arm in ("lang", "rand", "ident"):
        c = sel[arm]["mean_aug_acc"]
        L.append(f"| {arm} | " + " | ".join(_f(c[m]) for m in MULTS) + " |")
    L += ["", "## 选参", "", "| 臂 | R_aug | R_joint | 干净 Acc @R_joint |", "|---|---|---|---|"]
    for arm in ("ddo", "lang", "rand", "ident"):
        s = sel[arm]
        L.append(f"| {arm} | {s['R_aug']} | {s['R_joint']} | {_f(s['clean_acc'][s['R_joint']])} |")
    L += ["", f"规则：R_aug = {RULE_TEXT['R_aug']}；R_joint = {RULE_TEXT['R_joint']}。"]
    with open(sel_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[{ds}/source] 核对通过; 选择: " + ", ".join(f"{a} {sel[a]['R_aug']}/{sel[a]['R_joint']}"
                                                    for a in ("ddo", "lang", "rand", "ident")))
    print(f"[saved] {sel_json} —— 提交后才能 --stage final / eval_heads_targets.py --p4 {ds}")
    return 0


def stage_final(ds):
    sel_json, _, fin_json, fin_md, tgt_json = _paths(ds)
    ok, why = _locked(sel_json)
    if not ok:
        print("拒绝运行: 源域选择尚未锁定 —— " + why)
        return 2
    runs, cal = load_runs(ds)
    sel_now = selection(runs, ds)
    with open(sel_json, encoding="utf-8") as f:
        locked = json.load(f)["selection"]
    for arm in sel_now:
        for k in ("R_aug", "R_joint", "mean_aug_acc", "score_joint"):
            assert json.dumps(sel_now[arm][k], sort_keys=True) == json.dumps(locked[arm][k], sort_keys=True), \
                f"{ds}: 源域重算与锁定不一致 {arm}.{k}"
    assert os.path.exists(tgt_json), f"缺少 {tgt_json}"
    with open(tgt_json, encoding="utf-8") as f:
        tgt = {k: v["target_acc"] for k, v in json.load(f)["heads"].items()}
    fin = final_analysis(ds, sel_now, tgt)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "dataset": ds,
           "selection_lock_commit": why, "selection_sha256": sha256_file(sel_json),
           "target_eval_sha256": sha256_file(tgt_json), **fin}
    _write_json(fin_json, out)
    L = [f"# P4 / {ds}：跨数据集结果（源域选择锁定之后）", "",
         f"选择锁定于 `{why[:7]}`。", "", "## 目标域曲线（3 seeds 均值）", "",
         "| 臂 | " + " | ".join(f"{m}×" for m in MULTS) + " |", "|---|---|---|---|---|---|"]
    for arm in ("lang", "rand", "ident"):
        c = fin["target_curves"][arm]
        L.append(f"| {arm} | " + " | ".join(_f(c[m]) for m in MULTS) + " |")
    d = fin["target_curves"]["ddo"]
    L += ["", "DDO（α = 0 / 0.125 / 0.25 / 0.5 / 1 / 2 / 4）：" + " / ".join(_f(d[k]) for k in DOSES), "", "## 判读", ""]
    g1 = fin["paired"]["G1_lang_x4_vs_rand_x4"]
    L.append(f"- **G1**（Language@4× − Random@4×）：Δ = {_f(g1['mean'])} pp，逐 seed "
             f"{[round(x, 2) for x in g1['diff_per_seed']]}，{g1['n_positive']}/3 → **{fin['verdict']['G1']}**")
    for rule in ("R_aug", "R_joint"):
        q = fin["q1"][rule]
        L.append(f"- **Q1（{rule}）**：Language@{q['lang_strength']}× = {_f(q['lang_acc'])} 对 DDO@{q['ddo_strength']}× "
                 f"= {_f(q['ddo_acc'])}，Δ = {_f(q['mean'])} pp，{q['n_positive']}/3 → "
                 f"**{'过门槛' if q['gate'] else '未过门槛'}**")
    for m in ("4", "8"):
        w = fin["paired"][f"G2_lang_warm{m}_vs_fixed"]
        L.append(f"- warmup@{m}× − 固定：{_f(w['mean'])} pp（{w['n_positive']}/3）")
    L += ["", "## 选参（目标域事后）", "", "| 臂 | 规则 | 选中 | 目标域 Acc | Regret | ρ |", "|---|---|---|---|---|---|"]
    for arm in ("ddo", "lang", "rand", "ident"):
        for rule in ("R_aug", "R_joint"):
            s = fin["selection"][arm][rule]
            L.append(f"| {arm} | {rule} | {s['strength']} | {_f(s['target_acc'])} | {_f(s['regret'])} | "
                     f"{_f(s['rho_vs_target'], 3)} |")
        b = fin["selection"][arm]["post_hoc_best"]
        L.append(f"| {arm} | — | 最优 {b['strength']} | {_f(b['target_acc'])} | 0.00 | — |")
    with open(fin_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[{ds}/final] G1 {fin['verdict']['G1']} ({g1['mean']:+.2f}); Q1 R_joint "
          f"{fin['q1']['R_joint']['mean']:+.2f} ({fin['q1']['R_joint']['n_positive']}/3)")
    print(f"[saved] {fin_json}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", choices=list(P4_DS))
    ap.add_argument("--stage", choices=["source", "final"])
    a = ap.parse_args()
    if a.selftest:
        return _round12_selftest()
    assert a.dataset and a.stage, "--dataset 与 --stage 都必填"
    return stage_source(a.dataset) if a.stage == "source" else stage_final(a.dataset)


if __name__ == "__main__":
    sys.exit(main())
