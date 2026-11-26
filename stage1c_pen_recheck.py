"""Stage 1C 半径诊断的口径更正重算（外审第十轮 P2 收尾第 1、2 项；更正说明见 RESULTS §3y 末尾的更正小节）。

只重算**闭式量**（不重跑任何 PGD）。对 §3y 用过的同一批 30 个分类头 × 5 个椭球 × 10 个 ε 网格点：
  1. 补"实际最坏竞争类的惩罚" pen_worstcomp_*。此前的结果 JSON 只有 pen_bind_* —— 它是每个样本在全部
     竞争类上的**最大**惩罚的均值（平均最大类别对惩罚），不一定是决定鲁棒间隔的那个竞争类（"绑定"的叫法
     名不副实，外审第十轮指出）。
  2. 核对全部闭式量与已提交的 stage1c_radius_cub.json 逐项一致 —— 证明这次重算管线没有漂移，
     更正只是**补充**统计，不改动任何已有数字。
  3. PGD 汇总的同样本口径**不需要重算**：已提交 JSON 的 pgd 条目里 ce / ld / pgd / gap 本来就是
     前 300 个样本的同一子集（§3y 的表格错在渲染时混用了 1200 与 300 两个样本集，改渲染即可）。

边界（与 §3x 相同）：只用源域数据与文本；会话以 --target_eval none 构建，目标域特征缓存全程不读取；
收敛分类头只做事后量级诊断，不做任何判定；本脚本不选择 ε。
用法: python stage1c_pen_recheck.py [--selftest]
产物: experiment_results/Stage1C-Radius/stage1c_pen_recheck.json
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

import torch  # noqa: E402

from stage1c_radius_diagnose import (ELL_ORDER, GRID_MULT, OUT_DIR, SEEDS, ARMS,  # noqa: E402
                                     penalized_metrics, prepare)
from run_status import git_info  # noqa: E402

OLD_JSON = os.path.join(OUT_DIR, "stage1c_radius_cub.json")
NEW_JSON = os.path.join(OUT_DIR, "stage1c_pen_recheck.json")
# 逐项核对的闭式量键（pen_worstcomp_* 是本次新增, 不参与与旧 JSON 的比对）
COMPARE = ("acc", "ce", "loss", "pen_bind_mean", "pen_bind_q50", "pen_bind_q90", "pen_all_q50", "pen_all_q90",
           "clean_worst_margin_mean", "clean_worst_margin_q10", "robust_worst_margin_mean", "flip_rate_all",
           "flip_rate_correct", "r_max_mean", "r_max_sq_mean", "r_all_q50", "r_all_q90")
# 写进产物文件的键（保持文件在 1 MB 内; q50/q90 一类的分位数只保留惩罚相关的几列）
KEEP = ("strength", "kind", "acc", "ce", "loss", "pen_bind_mean", "pen_bind_q50", "pen_bind_q90",
        "pen_worstcomp_mean", "pen_worstcomp_q50", "pen_worstcomp_q90", "pen_all_q50", "pen_all_q90",
        "clean_worst_margin_mean", "robust_worst_margin_mean", "flip_rate_all", "flip_rate_correct",
        "r_max_mean", "r_max_sq_mean")
CAL_NAMES = (("ld_eps_grad", "eps_grad", "ld"), ("ld_eps_pen", "eps_pen", "ld"),
             ("isda_lam_grad", "lam_grad", "isda"), ("isda_lam_pen", "lam_pen", "isda"))


def _diff(a, b, keys):
    """两组同构记录在给定键上的最大绝对差（核对脚本自身也要能被自测）。"""
    return max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in keys)


def _keep(rec):
    return {k: rec[k] for k in KEEP}


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    a = [{"x": 1.0, "y": 2.5}, {"x": 0.0, "y": -1.0}]
    b = [{"x": 1.0 + 1e-9, "y": 2.5}, {"x": 1e-9, "y": -1.0}]
    check("R1 _diff 取给定键上的最大绝对差", abs(_diff(a, b, ("x", "y")) - 1e-9) < 1e-15)
    check("R2 _keep 只保留列出的键", set(_keep({**{k: 0.0 for k in KEEP}, "zz": 2}).keys()) == set(KEEP))
    check("R3 CAL_NAMES 覆盖两套口径 × 两种损失", {n for n, _, _ in CAL_NAMES} == {"ld_eps_grad", "ld_eps_pen", "isda_lam_grad", "isda_lam_pen"})
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    with open(OLD_JSON, encoding="utf-8") as f:
        old = json.load(f)
    ctx = prepare(device, check)                     # 步骤 1–4 与正式诊断完全相同（含 5 项核对）
    heads, ELL, y_val = ctx["heads"], ctx["ELL"], ctx["y_val"]
    grid = [ctx["eps_dom"] * m for m in GRID_MULT]

    n_pgd_expected = sum(1 for v in old["table"].values() for p in v.get("pgd", []))
    pgd_same = all(p["ce"] <= p["pgd"] + 1e-5 <= p["ld"] + 1e-5 for v in old["table"].values() for p in v.get("pgd", []))
    check("已提交 JSON 的 pgd 条目本身是同样本口径（链 CE ≤ PGD ≤ LD 在每个条目上成立，无需重算）",
          n_pgd_expected > 0 and pgd_same, f"{n_pgd_expected} 个条目")

    new_table, worst_by_eps, worst_cal = {}, 0.0, 0.0
    for hkey in heads:
        for e in ELL_ORDER:
            M = heads[hkey]["A"] @ ELL[hkey[1]][e].float().to(device)
            lg = heads[hkey]["logits"]
            key = f"{hkey[0]}|{hkey[1]}|{e}"
            recs = [penalized_metrics(lg, y_val, M, eps) for eps in grid]
            worst_by_eps = max(worst_by_eps, _diff(recs, old["table"][key]["by_eps"], COMPARE))
            cal_new = {}
            for name, skey, kind in CAL_NAMES:
                m = penalized_metrics(lg, y_val, M, old["calibration"][e][skey], kind=kind)
                cal_new[name] = m
                worst_cal = max(worst_cal, _diff([m], [old["table"][key]["cal"][name]], COMPARE))
            new_table[key] = {"by_eps": [_keep(r) for r in recs],
                              "cal": {name: _keep(m) for name, m in cal_new.items()}}
        print(f"  已重算 {hkey}", flush=True)

    check("全部闭式量与已提交的 §3y JSON 逐项一致（最大绝对差 < 1e-5）",
          worst_by_eps < 1e-5 and worst_cal < 1e-5, f"by_eps 最大差 {worst_by_eps:.2e}, cal 最大差 {worst_cal:.2e}")
    wc_lt_bind = all(t["by_eps"][i]["pen_worstcomp_mean"] <= t["by_eps"][i]["pen_bind_mean"] + 1e-12
                     for t in new_table.values() for i in range(len(grid)))
    check("最坏竞争类惩罚 ≤ 最大类别对惩罚（全部 1500 个 (头, 椭球, ε) 上成立）", wc_lt_bind)

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": device,
           "note": "外审第十轮的口径更正重算: 只重算闭式量, 补 pen_worstcomp_*; PGD 的同样本口径复用已提交 JSON "
                   "(stage1c_radius_cub.json 的 pgd 条目)。与已提交 JSON 的逐项一致性见 worst_*_diff。",
           "old_json_sha256": hashlib.sha256(open(OLD_JSON, "rb").read()).hexdigest(),
           "eps_dom": ctx["eps_dom"], "grid_mult": list(GRID_MULT), "grid": grid, "n_heads": len(heads),
           "compare_keys": list(COMPARE), "worst_by_eps_diff": worst_by_eps, "worst_cal_diff": worst_cal,
           "all_checks_passed": ok, "table": new_table, "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {NEW_JSON}  (耗时 {out['elapsed_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
