"""第十五轮 P2：Bias-Free × L1 补齐（§3ao 2×2 的最后一格）的选择与判定。
预注册见 RESULTS §3as（写于 12 个 run 的目标域评估之前；run 矩阵注册与标定先于训练提交）。

  --stage source : 读取 12 个 bf_l1 run 的源域验证字段（无任何目标域数字），逐项核对配置
                   （full_bf / ddo_norm l1 / 字面 α {0.5,1,2,4} / 独立划分）；描述性曲线，不做选参
                   （2×2 的四格都在同一 α 网格上直接可比，选参由 §3ap 的口径承担）；写
                   stage1b_c2_source.json —— 提交锁定。
  --stage final  : 锁定后读 eval_heads_targets.py --stage1b_c2 的事后目标域评估，给出 C1/C2 对照
                   （BF-L1 对 DDO-L1 = 偏置效应；BF-L1 对 BF-L2 = 范数效应，同一偏置状态）与
                   修正后的等压 C3（L2@1× 对 ISDA@λ_grad）。写 stage1b_c2_final.{json,md}。
  --selftest     : 合成数据自测。
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from p1_independent_report import _locked  # noqa: E402
from run_matrix import COMMON, MATRICES, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402
from stage1c_training_report import P1_TGT, SEEDS, _paired, gate, sha256_file  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")
SEL_JSON = os.path.join(OUT_DIR, "stage1b_c2_source.json")
SEL_MD = os.path.join(OUT_DIR, "stage1b_c2_source.md")
FIN_JSON = os.path.join(OUT_DIR, "stage1b_c2_final.json")
FIN_MD = os.path.join(OUT_DIR, "stage1b_c2_final.md")
TGT_JSON = os.path.join(OUT_DIR, "stage1b_c2_target_eval.json")
STRENGTHS = ["0.5", "1", "2", "4"]
MTAG = {"0.5": "0p5", "1": "1", "2": "2", "4": "4"}
N_TRAIN, N_VAL = 4790, 1200


def _f(x, nd=2):
    return f"{x:.{nd}f}"


def load_runs():
    runs = {}
    for ds, tag, extra in MATRICES["stage1b_c2"][1]:
        for s in SEEDS:
            name = f"{ds}_{tag}_seed{s}"
            rp, hp = f"logs/results/{name}.json", f"logs/heads/{name}.pt"
            with open(rp, encoding="utf-8") as f:
                r = json.load(f)
            assert result_is_complete(rp, expected(COMMON + extra, ds, s, tag)), f"{name}: 不完整或不匹配"
            assert r.get("target_acc") is None and (r.get("final_diag") or {}).get("target_acc") is None \
                and all(e["target_acc"] is None for e in r["epoch_metrics"]), f"{name}: 有目标域数字"
            ts, sv = r["train_subset"], r["source_val"]
            assert (ts["n_train"], ts["n_heldout"]) == (N_TRAIN, N_VAL) and sv["n"] == N_VAL \
                and ts["split_cache_sha256"] == sv["cache_sha256"], f"{name}: 划分不符"
            cfg = r["run_config"]
            assert cfg["ddo_dirs"] == "full_bf" and cfg["ddo_norm"] == "l1", f"{name}: 臂配置不符"
            runs[name] = {"tag": tag, "seed": s, "alpha": float(cfg["alpha"]),
                          "head_sha256": sha256_file(hp), "result_sha256": sha256_file(rp),
                          "clean_acc": sv["clean"]["acc"], "mean_aug_acc": sv["mean_aug_acc"],
                          "mean_aug_ce": sv["mean_aug_ce"]}
    return runs


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    check("Z1 配对差与门槛", gate(_paired([57.0, 56.0, 58.0], [56.0, 56.5, 57.0]))
          and not gate(_paired([56.5, 56.0, 56.4], [56.0, 56.5, 57.0])))
    check("Z2 网格与 tag 映射", all(MTAG[m] == t for m, t in
                                   (("0.5", "0p5"), ("1", "1"), ("2", "2"), ("4", "4"))))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def stage_source():
    runs = load_runs()
    curve = {}
    for m in STRENGTHS:
        v = [runs[f"CUB_s1b_bf_l1_a{MTAG[m]}_seed{s}"]["mean_aug_acc"] for s in SEEDS]
        curve[m] = {"mean": float(np.mean(v)), "per_seed": v,
                    "clean": float(np.mean([runs[f"CUB_s1b_bf_l1_a{MTAG[m]}_seed{s}"]["clean_acc"] for s in SEEDS]))}
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3as",
           "target_info_present": False, "curve": curve,
           "runs": {k: {kk: v[kk] for kk in ("tag", "seed", "alpha", "head_sha256", "result_sha256")}
                    for k, v in runs.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SEL_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    L = ["# Stage 1B C2：Bias-Free × L1（§3ao 2×2 的最后一格）——源域曲线（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3as。12 个 run 全部 `--target_eval none`（无目标域数字，逐个断言）。"
         "α = 字面 {0.5, 1, 2, 4}（等压标定：full_bf 等权 L1 在 α=1 的初始梯度比恰为 ρ̄=0.104922，"
         "见 `stage1b_calibration_bfl1.json`），与 DDO-L1 的四剂量逐项同 α。", "",
         "| α | 源域增强 Acc（3 seeds 均值） | 逐 seed | 干净 Acc |", "|---|---|---|---|"]
    for m in STRENGTHS:
        c = curve[m]
        L.append(f"| {m} | **{_f(c['mean'])}** | {' / '.join(_f(x) for x in c['per_seed'])} | {_f(c['clean'])} |")
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[source] 12 个 run 核对通过 (无任何目标域数字); 曲线: " +
          ", ".join(f"α={m}: {_f(curve[m]['mean'])}" for m in STRENGTHS))
    print(f"[saved] {SEL_JSON}、{SEL_MD} —— 提交之后才能运行 --stage final / eval_heads_targets.py --stage1b_c2")
    return 0


def stage_final():
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: c2 源域曲线尚未锁定 —— " + why)
        return 2
    runs = load_runs()
    with open(SEL_JSON, encoding="utf-8") as f:
        locked = json.load(f)
    for k, v in locked["runs"].items():
        assert v["head_sha256"] == runs[k]["head_sha256"] and v["result_sha256"] == runs[k]["result_sha256"], \
            f"{k}: 分类头或结果 json 在锁定之后被改动"
    with open(TGT_JSON, encoding="utf-8") as f:
        tgt = json.load(f)
    assert tgt["selection_sha256"] == sha256_file(SEL_JSON), "事后评估引用的选择文件与当前文件不同"
    anc = __import__("subprocess").run(["git", "merge-base", "--is-ancestor", tgt["selection_lock_commit"], "HEAD"],
                                       capture_output=True)
    assert anc.returncode == 0, "事后评估记录的锁定提交不是当前 HEAD 的祖先"
    T = {tag: [tgt["heads"][f"CUB_s1b_bf_l1_a{MTAG[m]}_seed{s}"]["target_acc"] for s in SEEDS]
         for m, tag in MTAG.items()}
    with open(os.path.join("experiment_results", "Stage1B-L2", "stage1b_target_eval.json"), encoding="utf-8") as f:
        t1b = json.load(f)
    with open("experiment_results/Stage1C-Training/s1c_target_eval.json", encoding="utf-8") as f:
        t1c = json.load(f)
    with open("experiment_results/Stage0D-Independent/p1_target_eval.json", encoding="utf-8") as f:
        p1t = json.load(f)
    B = {"l1_0p5": [p1t["heads"][f"CUB_p1_dose_a0p5_seed{s}"]["target_acc"] for s in SEEDS],
         "l1_1": [p1t["heads"][f"CUB_p1_dose_a1_seed{s}"]["target_acc"] for s in SEEDS],
         "l1_2": [p1t["heads"][f"CUB_p1_dose_a2_seed{s}"]["target_acc"] for s in SEEDS],
         "l1_4": [p1t["heads"][f"CUB_p1_dose_a4_seed{s}"]["target_acc"] for s in SEEDS]}
    pairs = {}
    for m in STRENGTHS:
        pairs[f"bfl1_vs_l1@{m}"] = _paired(T[m], B[f"l1_{m}"])          # 偏置效应 (同 α 同范数)
        bf_l2 = [t1b["heads"][f"CUB_s1b_bf_l2_a{MTAG[m]}_seed{s}"]["target_acc"] for s in SEEDS]
        pairs[f"bfl1_vs_bfl2@{m}"] = _paired(T[m], bf_l2)               # 范数效应 (同偏置状态同 α)
    # 修正后的等压 C3 (外审第十五轮): L2@1× (梯度配平 0.1050) 对 ISDA@λ_grad=97.38 (0.1049)
    l2_1 = [t1b["heads"][f"CUB_s1b_bf_l2_a1_seed{s}"]["target_acc"] for s in SEEDS]
    isda_grad = [t1c["heads"][f"CUB_s1c_isda_grad_seed{s}"]["target_acc"] for s in SEEDS]
    c3 = _paired(l2_1, isda_grad)
    fin = {"curve": {m: T[m] for m in STRENGTHS}, "paired": pairs, "gates": {k: gate(p) for k, p in pairs.items()},
           "c3_corrected": {"l2_1x": l2_1, "isda_grad_lam": 97.37995529174803, "isda_grad": isda_grad,
                            "paired": c3, "gate": gate(c3),
                            "note": "严格等初始梯度压力 (L2 0.1050 vs ISDA@λ_grad 0.1049)"}}
    out = {**fin, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "selection_lock_commit": why, "selection_sha256": sha256_file(SEL_JSON),
           "target_eval_sha256": sha256_file(TGT_JSON), "target_cache": tgt["target_cache"]}
    with open(FIN_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    M = ["# Stage 1B C2：Bias-Free × L1 的最终报告（源域曲线锁定之后的目标域事后评估）", "",
         "预注册见 `RESULTS.md` §3as；门槛沿用 §3z。", "",
         "| α | BF-L1 目标域（3 seeds） | 对 DDO-L1 同 α（偏置效应） | 对 BF-L2 同 α（范数效应） |",
         "|---|---|---|---|"]
    for m in STRENGTHS:
        p1, p2 = pairs[f"bfl1_vs_l1@{m}"], pairs[f"bfl1_vs_bfl2@{m}"]
        L.append(f"| {m} | **{_f(float(np.mean(T[m])))}**（{' / '.join(_f(x) for x in T[m])}） | "
                 f"{_f(p1['mean'])}（{p1['n_positive']}/3） | {_f(p2['mean'])}（{p2['n_positive']}/3） |")
    L += ["", f"修正后的等压 C3：L2@1×（{_f(float(np.mean(l2_1)))}）对 ISDA@λ_grad（{_f(float(np.mean(isda_grad)))}）"
          f"= **{_f(c3['mean'])} pp**（逐 seed {' / '.join(_f(x) for x in c3['diff_per_seed'])}，"
          f"{c3['n_positive']}/3 正向，过门槛={gate(c3)}）——严格等初始梯度压力（0.1050 vs 0.1049）。", ""]
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(M))
    print(f"[final] 锁定于 {why[:7]}; C3 修正 (L2@1× vs ISDA@λ_grad): {c3['mean']:.2f} pp ({c3['n_positive']}/3)")
    print(f"[saved] {FIN_JSON}、{FIN_MD}")
    return 0


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
