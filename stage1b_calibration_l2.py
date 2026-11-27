"""Stage 1B 的 L2 强度标定（第十三轮，预注册 §3ao）：找出使 L2 正则的初始梯度比等于 ρ̄_DDO 的 α。

方程与 §3z 完全相同：ρ̄_DDO = 0.1049224（DDO-L1 α=1 在 3 个初始化头上的实测均值）。
L2 的正则梯度对 α 严格线性（固定参数点），所以用 log-α 二分即可。

两个格子：
  Original × L2  : --ddo_dirs full     + --ddo_norm l2
  Bias-Free × L2 : --ddo_dirs full_bf  + --ddo_norm l2   （正则路径不含 W·β + b）

梯度统计用与训练完全相同的对象：P1 的初始化会话、train_core 前 256 的固定批次、
分类头的全部可学习参数（None 梯度按 0 处理，bias-free 下 LN/Lin 的 bias 不在计算图里）。

用法: python stage1b_calibration_l2.py [--selftest]
产物: experiment_results/Stage1B-L2/stage1b_calibration_l2.json
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from run_status import git_info  # noqa: E402
from stage1c_radius_diagnose import build_session  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")
CALIB = os.path.join("experiment_results", "Stage1C-Radius", "stage1c_calibration_traincore.json")
BASE_RUN = "logs/results/CUB_p1_dose_a1_seed1.json"
SEEDS = (1, 2, 3)
CELLS = {"original_l2": "full", "biasfree_l2": "full_bf"}
LO, HI, IT = 1e-5, 1e3, 60


def reg_and_ce(sess, x, y, alpha):
    """返回 (正则梯度范数, CE 梯度范数)。正则 = alpha * mean(h^2)；CE 用干净 logits。"""
    m = sess.model
    m.eval()
    params = [p for p in m.classifier.parameters() if p.requires_grad]
    m.zero_grad(set_to_none=True)
    _, cls, reg = m(x)
    assert reg is not None, "该 arm 不产出正则张量"
    R = alpha * (reg ** 2).mean()
    gR = torch.autograd.grad(R, params, retain_graph=True, allow_unused=True)
    ce = F.cross_entropy(cls, y)
    gC = torch.autograd.grad(ce, params, allow_unused=True)

    def norm(gs):
        return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in gs if g is not None)))
    return norm(gR), norm(gC)


def bisect(fn, target, lo=LO, hi=HI, it=IT):
    """fn(alpha) 单调增；返回 (alpha*, 是否夹住)。"""
    flo, fhi = fn(lo), fn(hi)
    if not (flo <= target <= fhi):
        return float("nan"), False
    for _ in range(it):
        mid = float(np.sqrt(lo * hi))
        fm = fn(mid)
        if fm < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-10:
            break
    return float(np.sqrt(lo * hi)), True


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    # 线性性：ratio(α) = c·α ⇒ 二分应回到解析解
    c = 0.37
    est, br = bisect(lambda a: c * a, 0.1049224)
    check("T1 线性函数的二分回归解析解", br and abs(est - 0.1049224 / c) / (0.1049224 / c) < 1e-6,
          f"{est:.8f} vs {0.1049224 / c:.8f}")
    est2, br2 = bisect(lambda a: c * a, 1e3 * c * 10)
    check("T2 目标超出区间时返回未夹住", (not br2) and est2 != est2)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    with open(CALIB, encoding="utf-8") as f:
        rho = float(json.load(f)["rho_bar"])
    with open(BASE_RUN, encoding="utf-8") as f:
        cfg0 = json.load(f)["run_config"]
    t0 = time.time()
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "note": "Stage 1B 的 L2 强度标定；目标 = ρ̄_DDO（DDO-L1 α=1 的初始梯度比）",
           "rho_bar_target": rho, "calibration_source": CALIB, "cells": {}}
    grids = {}
    for cell, dirs in CELLS.items():
        per_seed, anchors = {}, []
        for s in SEEDS:
            c = dict(cfg0)
            c.update(alpha=0.0, ddo_dirs=dirs, ddo_norm="l2", weight_mode="none",
                     run_tag=f"calib_{cell}_seed{s}", seed=s)
            sess = build_session(c, s)
            x, y = sess._diag_batch()
            base = reg_and_ce(sess, x, y, 1.0)

            def ratio(al):
                gr, gc = reg_and_ce(sess, x, y, al)
                return gr / gc

            alpha, br = bisect(ratio, rho)
            per_seed[str(s)] = {"alpha_star": alpha, "bracketed": br,
                                "ratio_at_1": base[0] / base[1], "grad_reg": base[0], "grad_ce": base[1]}
            anchors.append(alpha)
            print(f"  [{cell}] seed {s}: α*={alpha:.6f} (bracketed={br}); ratio@α=1 {base[0] / base[1]:.4f}")
        anchor = float(np.mean(anchors))
        ok = all(v["bracketed"] for v in per_seed.values()) and abs(anchor - anchor) == 0
        grids[cell] = {f"{m:g}": anchor * m for m in (0.5, 1.0, 2.0, 4.0)}
        out["cells"][cell] = {"ddo_dirs": dirs, "anchor": anchor, "per_seed": per_seed,
                              "scan": grids[cell], "all_bracketed": ok}
        print(f"[{cell}] anchor = {anchor!r}; grid = { {k: repr(v) for k, v in grids[cell].items()} }")
    out["all_checks_passed"] = all(v["all_bracketed"] for v in out["cells"].values())
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "stage1b_calibration_l2.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {p}  ({round(time.time() - t0, 1)}s)")
    return 0 if out["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
