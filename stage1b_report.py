"""Stage 1B（L1/L2 × 偏置）的分析（第十三轮，预注册 §3ao）。

--stage source : 读 24 个 s1b run 与复用的基线（Original-L1 = p1 七剂量、Language-ISDA 五倍数、
                 Identity-ISDA 五倍数），逐 run 核对配置/无目标域/划分与扰动，写源域曲线 + 锁定文件。
--stage final  : 锁定之后读目标域评估，输出 C1（L1→L2，单因素）、C3（DDO-L2 vs ISDA，捆绑）、
                 曲线与源域干净/增强 Acc。

**协议一致性检查（重要）**：Bias-Free × L1 的旧 run（`CUB_bf_full8_a*`）用的是**完整训练集**（§3q 时期，
没有 train_core），与本轮的 train_core（4790）不可比；C2（Original vs Bias-Free 的 2×2）因此需要另补
12 个 train_core 协议的 BF-L1 run。本脚本把这件事写成显式告警，不静默混用。
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
from round12_report import _f, _write_json, spearman  # noqa: E402
from run_matrix import COMMON, MATRICES, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402
from stage1c_training_report import GATE_POS, SEEDS, _paired, gate, sha256_file  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")
SEL_JSON = os.path.join(OUT_DIR, "stage1b_source.json")
SEL_MD = os.path.join(OUT_DIR, "stage1b_source.md")
FIN_JSON = os.path.join(OUT_DIR, "stage1b_final.json")
FIN_MD = os.path.join(OUT_DIR, "stage1b_final.md")
TGT_JSON = os.path.join(OUT_DIR, "stage1b_target_eval.json")
MULTS = ["0.5", "1", "2", "4"]
MTAG = {"0.5": "0p5", "1": "1", "2": "2", "4": "4"}
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]


def _name(prefix, m, s):
    return f"{prefix}_a{MTAG[m]}_seed{s}"


def load_runs():
    out = {}
    entries = MATRICES["stage1b"][1]
    assert len(entries) == 8, len(entries)
    for _, tag, ex in entries:
        for s in SEEDS:
            rp = f"logs/results/CUB_{tag}_seed{s}.json"
            with open(rp, encoding="utf-8") as f:
                r = json.load(f)
            assert result_is_complete(rp, expected(COMMON + ex, "CUB", s, tag)), f"{rp}: 不完整或配置不匹配"
            assert r.get("target_acc") is None and all(e["target_acc"] is None for e in r["epoch_metrics"]), \
                f"{rp}: 有目标域数字"
            sv = r["source_val"]
            assert sorted(sv["aug"]) == sorted(AUGS), f"{rp}: 扰动不全"
            out[f"CUB_{tag}_seed{s}"] = {"clean": sv["clean"]["acc"], "aug": sv["mean_aug_acc"],
                                         "ce": sv["clean"]["ce"], "elapsed": r.get("elapsed_seconds"),
                                         "head_sha256": sha256_file(f"logs/heads/CUB_{tag}_seed{s}.pt"),
                                         "result_sha256": sha256_file(rp)}
    # 复用的基线：Original-L1 七剂量（train_core 协议，与 s1b 相同）
    for a, tag in (("0.5", "dose_a0p5"), ("1", "dose_a1"), ("2", "dose_a2"), ("4", "dose_a4")):
        for s in SEEDS:
            with open(f"logs/results/CUB_p1_{tag}_seed{s}.json", encoding="utf-8") as f:
                r = json.load(f)
            sv = r["source_val"]
            out[f"CUB_p1_{tag}_seed{s}"] = {"clean": sv["clean"]["acc"], "aug": sv["mean_aug_acc"],
                                            "ce": sv["clean"]["ce"], "elapsed": r.get("elapsed_seconds"),
                                            "head_sha256": sha256_file(f"logs/heads/CUB_p1_{tag}_seed{s}.pt"),
                                            "result_sha256": sha256_file(f"logs/results/CUB_p1_{tag}_seed{s}.json")}
    return out


def arm_curve(runs, prefix, key):
    return {m: float(np.mean([runs[_name(prefix, m, s)][key] for s in SEEDS])) for m in MULTS}


def stage_source():
    runs = load_runs()
    curves = {}
    for arm, prefix in (("orig_l1", "CUB_p1_dose_"), ("orig_l2", "CUB_s1b_orig_l2"),
                        ("bf_l2", "CUB_s1b_bf_l2")):
        if arm == "orig_l1":
            curves[arm] = {m: float(np.mean([runs[f"CUB_p1_dose_a{MTAG[m]}_seed{s}"]["aug"] for s in SEEDS]))
                           for m in MULTS}
            curves[arm + "_clean"] = {m: float(np.mean(
                [runs[f"CUB_p1_dose_a{MTAG[m]}_seed{s}"]["clean"] for s in SEEDS])) for m in MULTS}
        else:
            curves[arm] = arm_curve(runs, prefix, "aug")
            curves[arm + "_clean"] = arm_curve(runs, prefix, "clean")
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "target_info_present": False,
           "curves": curves,
           "protocol_warning": "Bias-Free × L1 的旧 run（CUB_bf_full8_a*）是完整训练集协议（§3q），"
                               "与本轮 train_core 不可比；C2 的 2×2 需另补 12 个 train_core 协议的 BF-L1 run。",
           "runs": runs}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_json(SEL_JSON, out)
    L = ["# Stage 1B：源域曲线（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3ao。24 个 s1b run 全部 `--target_eval none`（逐个断言）。", "",
         "| 臂 | " + " | ".join(f"{m}×" for m in MULTS) + " |", "|---|---|---|---|---|"]
    for arm in ("orig_l1", "orig_l2", "bf_l2"):
        L.append(f"| {arm}（增强 Acc） | " + " | ".join(_f(curves[arm][m]) for m in MULTS) + " |")
    for arm in ("orig_l1", "orig_l2", "bf_l2"):
        L.append(f"| {arm}（干净 Acc） | " + " | ".join(_f(curves[arm + '_clean'][m]) for m in MULTS) + " |")
    L += ["", "> 协议警告：" + out["protocol_warning"]]
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[source] 核对通过; 24 个 s1b run + 12 个 L1 基线")
    for arm in ("orig_l1", "orig_l2", "bf_l2"):
        print(f"  {arm:8s} 增强 Acc: " + " / ".join(_f(curves[arm][m]) for m in MULTS))
    print(f"[saved] {SEL_JSON}")
    return 0


def target_of(prefix, m, s):
    return f"CUB_{prefix}_a{MTAG[m]}_seed{s}"


def stage_final():
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: 源域文件尚未锁定 —— " + why)
        return 2
    assert os.path.exists(TGT_JSON), f"缺少 {TGT_JSON}"
    t = {}
    for p in (TGT_JSON,
              os.path.join("experiment_results", "Stage0D-Independent", "p1_target_eval.json"),
              os.path.join("experiment_results", "Stage1C-Training", "s1c_target_eval.json"),
              os.path.join("experiment_results", "Stage1C-Training", "s11_target_eval.json")):
        with open(p, encoding="utf-8") as f:
            for k, v in json.load(f)["heads"].items():
                if v.get("target_acc") is not None:
                    t[k] = v["target_acc"]

    def vec(prefix, m, s_is=None):
        return [t[target_of(prefix, m, s)] for s in SEEDS]

    def vec_l1(m):
        return [t[f"CUB_p1_dose_a{MTAG[m]}_seed{s}"] for s in SEEDS]

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "source_lock_commit": why, "target_curves": {}, "paired": {}, "gates": {}}
    for arm in ("orig_l1", "orig_l2", "bf_l2"):
        out["target_curves"][arm] = {m: float(np.mean(vec_l1(m) if arm == "orig_l1" else vec(
            "s1b_orig_l2" if arm == "orig_l2" else "s1b_bf_l2", m))) for m in MULTS}
    for m in MULTS:
        p = _paired(vec("s1b_orig_l2", m), vec_l1(m))          # C1：L2 − L1（同 α 倍数）
        out["paired"][f"C1_l2_minus_l1_at_{m}x"] = p
        out["gates"][f"C1_l2_minus_l1_at_{m}x"] = gate(p)
    # C3：DDO-L2 vs Language-ISDA（同语言几何；捆绑了约束对象与损失位置）
    for m in ("1", "2", "4"):
        p = _paired(vec("s1b_orig_l2", m), [t[f"CUB_s1c_isda_x{m.replace('.', 'p')}_seed{s}"] for s in SEEDS])
        out["paired"][f"C3_l2_minus_isda_at_{m}x"] = p
        out["gates"][f"C3_l2_minus_isda_at_{m}x"] = gate(p)
    _write_json(FIN_JSON, out)
    L = ["# Stage 1B：结果（源域锁定之后）", "", f"锁定于 `{why[:7]}`。", "",
         "| 臂 | 0.5× | 1× | 2× | 4× |", "|---|---|---|---|---|"]
    for arm in ("orig_l1", "orig_l2", "bf_l2"):
        L.append(f"| {arm} | " + " | ".join(_f(out['target_curves'][arm][m]) for m in MULTS) + " |")
    L += ["", "## C1（L1 → L2，单因素）", ""]
    for m in MULTS:
        p = out["paired"][f"C1_l2_minus_l1_at_{m}x"]
        L.append(f"- {m}×：Δ = {p['mean']:+.2f}（{[round(x, 2) for x in p['diff_per_seed']]}，"
                 f"{p['n_positive']}/3）")
    L += ["", "## C3（DDO-L2 vs Language-ISDA，捆绑）", ""]
    for m in ("1", "2", "4"):
        p = out["paired"][f"C3_l2_minus_isda_at_{m}x"]
        L.append(f"- {m}×：Δ = {p['mean']:+.2f}（{[round(x, 2) for x in p['diff_per_seed']]}，"
                 f"{p['n_positive']}/3）")
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[final] " + "; ".join(f"C1@{m}× {out['paired'][f'C1_l2_minus_l1_at_{m}x']['mean']:+.2f}" for m in MULTS))
    print(f"[saved] {FIN_JSON}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["source", "final"])
    a = ap.parse_args()
    if a.stage == "source":
        return stage_source()
    if a.stage == "final":
        return stage_final()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
