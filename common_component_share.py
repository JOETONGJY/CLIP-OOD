"""公共分量能量占比：ĀᵀGĀ / R_L2（外审第十六轮 P2 的记录清单项；§3ax 引用）。
对每个头（BF-L2 / Centered-L2 / DDO-L1 四强度 × 3 seeds）用与训练闭式同一构造的 G（float64）计算。
只用已保存的分类头，零训练。产物: experiment_results/Stage1B-L2/common_component_share.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from stage1c_radius_diagnose import build_session  # noqa: E402

OUT = os.path.join("experiment_results", "Stage1B-L2", "common_component_share.json")
C_CLS = 200
ARMS = [("BF-L2", "CUB_s1b_bf_l2_a{mtag}_seed{s}"),
        ("Centered-L2", "CUB_s1b_cen_a{mtag}_seed{s}"),
        ("DDO-L1", "CUB_p1_dose_a{mtag}_seed{s}")]
STRENGTHS = [("0.5", "0p5"), ("1", "1"), ("2", "2"), ("4", "4")]


def main():
    with open("logs/results/CUB_p1_dose_a1_seed1.json", encoding="utf-8") as f:
        cfg = json.load(f)["run_config"]
    cfg.update(ddo_dirs="full", ddo_norm="l2c", run_tag="dbg_G", seed=1)
    sess = build_session(cfg, 1)
    G = sess.model.l2_G.double().cpu()
    out = {"note": "公共分量能量占比 = ĀᵀGĀ / R_L2（BF 口径，float64；G 与训练闭式同一构造）。"
                   "Centered-L2 的训练目标不含此项——它的占比是『训练是否重新引入公共分量』的观测量。",
           "per_arm": {}}
    for label, pat in ARMS:
        per = {}
        for m, mt in STRENGTHS:
            shares = []
            for s in (1, 2, 3):
                p = pat.format(mtag=mt, s=s) + ".pt"
                h = torch.load(f"logs/heads/{p}", map_location="cpu", weights_only=False)
                A = (h["classifier"]["2.weight"] * h["classifier"]["1.weight"][None, :]).double()
                r_l2 = float(((A @ G) * A).sum() / C_CLS)
                Abar = A.mean(0, keepdim=True)
                shares.append(float(Abar @ G @ Abar.T) / r_l2)
            per[m] = {"mean": round(sum(shares) / 3, 4), "per_seed": [round(x, 4) for x in shares]}
        out["per_arm"][label] = per
        print(label, {k: v["mean"] for k, v in per.items()})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
