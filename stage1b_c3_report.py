"""第十六轮 P2：Centered-L2 的选择与判定。预注册见 RESULTS §3aw；门槛沿用 §3z。

  --stage source : 读取 12 个 l2c run 的源域验证字段（无任何目标域数字），核对配置
                   （full / ddo_norm l2c / 闭式 / α 从标定现算）；描述性曲线；写
                   stage1b_c3_source.json —— 提交锁定。
  --stage final  : 锁定后读 eval_heads_targets.py --stage1b_c3 的事后目标域评估：
                   ①对 BF-L2 同 α（去除公共分量的效应；两者 α 网格几乎相同）；
                   ②等压下对 ISDA（BF-L2 1×=0.7218 / Centered 1×=0.7222 都 ≈ 等压点）；
                   ③公共分量能量占比 ĀᵀGĀ / R_L2（每个头事后计算）；
                   ④F_rel 与 rel top-10（由 eval_decision_fidelity 扩展后另行入库）。
                   写 stage1b_c3_final.{json,md}。
  --selftest
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

from p1_independent_report import _locked  # noqa: E402
from run_matrix import COMMON, MATRICES, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402
from stage1b_c2_report import MTAG, STRENGTHS  # noqa: E402
from stage1c_training_report import P1_TGT, SEEDS, _paired, gate, sha256_file  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")
SEL_JSON = os.path.join(OUT_DIR, "stage1b_c3_source.json")
SEL_MD = os.path.join(OUT_DIR, "stage1b_c3_source.md")
FIN_JSON = os.path.join(OUT_DIR, "stage1b_c3_final.json")
FIN_MD = os.path.join(OUT_DIR, "stage1b_c3_final.md")
TGT_JSON = os.path.join(OUT_DIR, "stage1b_c3_target_eval.json")
N_TRAIN, N_VAL = 4790, 1200


def _f(x, nd=2):
    return f"{x:.2f}"


def _f4(x, nd=4):
    return f"{x:.{nd}f}"


def load_runs():
    runs = {}
    for ds, tag, extra in MATRICES["stage1b_c3"][1]:
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
            assert cfg["ddo_dirs"] == "full" and cfg["ddo_norm"] == "l2c", f"{name}: 臂配置不符"
            ld = r["ld_config"]
            assert ld is None or ld.get("mode") in (None, "none"), f"{name}: ld_config 应为空"
            runs[name] = {"tag": tag, "seed": s, "alpha": float(cfg["alpha"]),
                          "head_sha256": sha256_file(hp), "result_sha256": sha256_file(rp),
                          "clean_acc": sv["clean"]["acc"], "mean_aug_acc": sv["mean_aug_acc"],
                          "mean_aug_ce": sv["mean_aug_ce"], "elapsed": r.get("elapsed_seconds")}
    return runs


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    check("Z1 网格与 tag", all(MTAG[m] == t for m, t in (("0.5", "0p5"), ("1", "1"), ("2", "2"), ("4", "4"))))
    check("Z2 配对与门槛", gate(_paired([57.0, 56.0, 58.0], [56.0, 56.5, 57.0]))
          and not gate(_paired([56.5, 56.0, 56.4], [56.0, 56.5, 57.0])))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def stage_source():
    runs = load_runs()
    curve = {}
    for m in STRENGTHS:
        v = [runs[f"CUB_s1b_cen_a{MTAG[m]}_seed{s}"]["mean_aug_acc"] for s in SEEDS]
        curve[m] = {"mean": float(np.mean(v)), "per_seed": v,
                    "clean": float(np.mean([runs[f"CUB_s1b_cen_a{MTAG[m]}_seed{s}"]["clean_acc"] for s in SEEDS]))}
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3aw",
           "target_info_present": False, "curve": curve,
           "runs": {k: {kk: v[kk] for kk in ("tag", "seed", "alpha", "head_sha256", "result_sha256")}
                    for k, v in runs.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SEL_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    L = ["# Stage 1B C3：Centered-L2——源域曲线（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3aw。12 个 run 全部 `--target_eval none`（无目标域数字，逐个断言）；"
         "闭式路径（forward 跳过显式响应）。α = 等压标定（1× = 0.722237），与 BF-L2 的 α 网格几乎相同"
         "（0.5× = 0.3611 对 0.3609 等）——同 α 与等压两种口径在本轮几乎重合，都报告。", "",
         "| 强度 | α | 源域增强 Acc（3 seeds 均值） | 逐 seed | 干净 Acc |", "|---|---|---|---|---|"]
    alpha_ref = {"0.5": 0.3611184308082789, "1": 0.7222368616165578, "2": 1.4444737232331156,
                 "4": 2.888947446466231}
    for m in STRENGTHS:
        c = curve[m]
        L.append(f"| {m}× | {alpha_ref[m]:.4f} | **{_f(c['mean'])}** | "
                 f"{' / '.join(_f(x) for x in c['per_seed'])} | {_f(c['clean'])} |")
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[source] 12 个 run 核对通过 (无任何目标域数字); 曲线: " +
          ", ".join(f"{m}×: {_f(curve[m]['mean'])}" for m in STRENGTHS))
    print(f"[saved] {SEL_JSON}、{SEL_MD} —— 提交之后才能运行 --stage final / eval_heads_targets.py --stage1b_c3")
    return 0


def stage_final():
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: c3 源域曲线尚未锁定 —— " + why)
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
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", tgt["selection_lock_commit"], "HEAD"],
                         capture_output=True)
    assert anc.returncode == 0, "事后评估记录的锁定提交不是当前 HEAD 的祖先"
    T = {m: [tgt["heads"][f"CUB_s1b_cen_a{MTAG[m]}_seed{s}"]["target_acc"] for s in SEEDS] for m in STRENGTHS}
    with open(os.path.join(OUT_DIR, "stage1b_target_eval.json"), encoding="utf-8") as f:
        t1b = json.load(f)
    with open(os.path.join(OUT_DIR, "stage1b_c2_target_eval.json"), encoding="utf-8") as f:
        t1c2 = json.load(f)
    with open("experiment_results/Stage1C-Training/s11_target_eval.json", encoding="utf-8") as f:
        t11 = json.load(f)
    with open(P1_TGT, encoding="utf-8") as f:
        p1t = json.load(f)
    pairs = {}
    for m in STRENGTHS:
        bf_l2 = [t1b["heads"][f"CUB_s1b_bf_l2_a{MTAG[m]}_seed{s}"]["target_acc"] for s in SEEDS]
        pairs[f"cen_vs_bfl2@{m}"] = _paired(T[m], bf_l2)                 # 去公共分量效应 (同 α ≈ 同等压)
    l2_1 = [t1b["heads"][f"CUB_s1b_bf_l2_a1_seed{s}"]["target_acc"] for s in SEEDS]
    l2_2 = [t1b["heads"][f"CUB_s1b_bf_l2_a2_seed{s}"]["target_acc"] for s in SEEDS]
    l2_4 = [t1b["heads"][f"CUB_s1b_bf_l2_a4_seed{s}"]["target_acc"] for s in SEEDS]
    isda_4 = [t11["heads"][f"CUB_s1c_isda_x4_seed{s}"]["target_acc"] for s in SEEDS]
    l1_2 = [p1t["heads"][f"CUB_p1_dose_a2_seed{s}"]["target_acc"] for s in SEEDS]
    # Centered 选中强度的 F_rel 由 eval_decision_fidelity 扩展后另行入库 (§3ax 引用)
    fin = {"curve": T, "paired": pairs, "gates": {k: gate(p) for k, p in pairs.items()},
           "context": {"bf_l2_1x": l2_1, "bf_l2_2x": l2_2, "bf_l2_4x": l2_4, "isda_4x": isda_4,
                       "ddo_l1_a2": l1_2}}
    out = {**fin, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "selection_lock_commit": why, "selection_sha256": sha256_file(SEL_JSON),
           "target_eval_sha256": sha256_file(TGT_JSON), "target_cache": tgt["target_cache"]}
    with open(FIN_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    M = ["# Stage 1B C3：Centered-L2 的最终报告（源域曲线锁定之后的目标域事后评估）", "",
         "预注册见 `RESULTS.md` §3aw；门槛沿用 §3z。", "",
         "| 强度 | Centered-L2 目标域（3 seeds） | 对 BF-L2 同倍数（Δ，去公共分量效应） |",
         "|---|---|---|"]
    for m in STRENGTHS:
        p = pairs[f"cen_vs_bfl2@{m}"]
        M.append(f"| {m}× | **{_f(float(np.mean(T[m])))}**（{' / '.join(_f(x) for x in T[m])}） | "
                 f"{_f(p['mean'])}（{p['n_positive']}/3，过门槛={fin['gates'][f'cen_vs_bfl2@{m}']}） |")
    M += ["", "参照（同表上下文）：BF-L2 1× " + _f(float(np.mean(l2_1))) + " / 2× " + _f(float(np.mean(l2_2)))
          + " / 4× " + _f(float(np.mean(l2_4))) + "；ISDA@4× " + _f(float(np.mean(isda_4)))
          + "；DDO α=2 " + _f(float(np.mean(l1_2))) + "。", ""]
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(M))
    print(f"[final] 锁定于 {why[:7]}; Centered-L2 目标域: " +
          ", ".join(f"{m}× {np.mean(T[m]):.2f}" for m in STRENGTHS))
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
