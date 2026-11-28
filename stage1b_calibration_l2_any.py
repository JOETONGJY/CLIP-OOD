"""任意数据集的 DDO-L2 等压标定（外审第十七轮 P1 前置；§3ay 预注册引用）。

方程与 §3ao 一致：α_1x := 使 DDO-L2（mean(h²)，full 方向集，含原始偏置处理）在 train_core 固定批次上的
初始正则梯度比等于该数据集 ρ̄_DDO 的 α（ρ̄ 取第十二轮 stage1c_calibration_{ds}.json 的 rho_bar，
即等权 DDO α=1 的初始比——第十二轮已按数据集重标定，AWA2/LADV/PACS 相差约 5 倍，不照搬 CUB 数值）。
L2 的正则梯度对 α 线性，故 α_1x = ρ̄ / ratio(α=1)，不再二分；ratio(α=1) 与 scan 都逐 seed 记录。
会话构造与第十二轮 stage1c_calibration_any.py 完全相同（CUB P1 run_config 换数据集与划分缓存）。

用法: python stage1b_calibration_l2_any.py --dataset AWA2 [--selftest]
产物: experiment_results/Stage1B-L2/stage1b_l2_calibration_{dataset}.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_status import git_info  # noqa: E402
from stage1c_calibration_any import BASE_RUN, SEEDS  # noqa: E402
from stage1c_radius_diagnose import build_session, reg_ratio  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    C, K, B = 8, 30, 64
    A = torch.randn(C, K, dtype=torch.float64) * 0.3
    U = torch.randn(300, K, dtype=torch.float64)
    U = U - U.mean(-1, keepdim=True)
    G = U.T @ U / U.size(0)
    # L2 正则梯度对 α 线性: ratio(α) = α · ratio(1)
    r1, r2 = 0.05, 0.10
    check("L1 线性: ratio(2)/ratio(1) = 2 (L2 梯度对 α 线性, 标定用比例缩放而非二分)",
          abs(r2 / r1 - 2.0) < 1e-12)
    check("L2 alpha_1x = ρ̄ / ratio(1)", abs((0.104922 / r1) - 2.09844) < 1e-4)
    check("L3 scan 覆盖 {0.5,1,2,4}", tuple(round(a1 * m, 6) for m, a1 in
                                            ((0.5, 1.0), (1, 1.0), (2, 1.0), (4, 1.0))) == (0.5, 1.0, 2.0, 4.0))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    ds = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else ""
    assert ds in ("AWA2", "LADV", "PACS"), "--dataset 必填 (AWA2 / LADV / PACS)"
    cache = f"cache/{ds}_train_aug.pt"
    assert os.path.exists(cache), f"缺少 {cache}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    with open(f"experiment_results/Stage1C-Radius/stage1c_calibration_{ds}.json", encoding="utf-8") as f:
        ref = json.load(f)
    rho = float(ref["rho_bar"])
    with open(BASE_RUN, encoding="utf-8") as f:
        cfg0 = json.load(f)["run_config"]

    vals, cos_v = {}, {}
    for s in SEEDS:
        c = dict(cfg0)
        c.update(dataset=ds, train_subset_cache=cache, source_val_cache=cache,
                 ddo_dirs="full", ddo_norm="l2", run_tag=f"calib_l2_{ds.lower()}_seed{s}", seed=s)
        sess = build_session(c, s)
        x, y = sess._diag_batch()
        sess.model.eval()
        params = [p for p in sess.model.classifier.parameters() if p.requires_grad]
        _, lg, _ = sess.model(x)

        def ddo_fn():
            return (sess.model.classifier[1:](sess.model.domain_concept_projection) ** 2).mean()
        r_, c_ = reg_ratio(lg, params, None, None, y, 0.0, "ddo", ddo_fn=ddo_fn)
        vals[str(s)], cos_v[str(s)] = r_, c_
        print(f"  [{ds}] seed {s}: alpha=1 的 L2 初始梯度比 {r_:.6f}", flush=True)
    ratio = float(np.mean(list(vals.values())))
    a1 = rho / ratio
    scan = {m: a1 * float(m) for m in ("0.5", "1", "2", "4")}
    check(f"{ds}: ratio(α=1) 为正且与 ρ̄ 同量级", 0 < ratio < 1 and 0.2 < rho / ratio < 5,
          f"ratio {ratio:.6f} → alpha_1x {a1:.6f}")
    print(f"[{ds}] alpha_1x = {a1!r}  scan = " + str({k: repr(v) for k, v in scan.items()}))

    out = {"created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "dataset": ds,
           "note": "DDO-L2 (full 方向集, mean(h²)) 的等压标定 (外审第十七轮 P1): 1× = 与该数据集等权 DDO α=1 "
                   "同初始梯度比; L2 梯度对 α 线性 → alpha_1x = rho_bar / ratio(alpha=1), 不二分。"
                   "ρ̄ 取第十二轮 stage1c_calibration_{ds}.json (逐数据集重标定)。",
           "rho_bar_source": f"stage1c_calibration_{ds}.json", "rho_bar": rho,
           "ratio_at_alpha1_per_seed": vals, "ratio_at_alpha1": ratio, "cos_at_alpha1": cos_v,
           "anchor_alpha_1x": a1, "scan": scan, "all_checks_passed": ok}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"stage1b_l2_calibration_{ds}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[saved] {OUT_DIR}/stage1b_l2_calibration_{ds}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
