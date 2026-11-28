"""跨数据集的 DDO-L2 闭式等价核对（外审第十七轮 P1："正式训练前对各数据集的闭式损失及梯度做轻量等价核对"）。

对 AWA2 / LADV / PACS 各建一个初始化会话（full 方向集、含偏置路径、闭式 buffer），
显式损失 mean(h²)（forward 响应）vs 闭式 (1/C)[tr(AGAᵀ)+2cᵀAμ+‖c‖²] 的值与梯度逐项对比。
零训练。用法: python stage1b_l2_closed_crosscheck.py [--dataset DS]
产物: experiment_results/Stage1B-L2/stage1b_l2_closed_crosscheck.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from run_status import git_info  # noqa: E402
from stage1c_calibration_any import BASE_RUN, SEEDS  # noqa: E402
from stage1c_radius_diagnose import build_session  # noqa: E402

OUT = os.path.join("experiment_results", "Stage1B-L2", "stage1b_l2_closed_crosscheck.json")


def check_ds(ds):
    cache = f"cache/{ds}_train_aug.pt"
    ok = True
    with open(BASE_RUN, encoding="utf-8") as f:
        cfg0 = json.load(f)["run_config"]
    c = dict(cfg0)
    c.update(dataset=ds, train_subset_cache=cache, source_val_cache=cache, ddo_dirs="full",
             ddo_norm="l2", ddo_l2_closed=True, run_tag=f"chk_l2c_{ds.lower()}", seed=1)
    sess = build_session(c, 1)
    x, y = sess._diag_batch()
    sess.model.eval()
    params = [p for p in sess.model.classifier.parameters() if p.requires_grad]
    # 显式对照: 与训练同一 LN-affine 路径 (classifier[1:](P)) 的 mean(h²); 闭式会话的 forward 跳过响应,
    # 所以显式值单独手算 (P 不参与梯度, no_grad 足够)
    ln, lin = sess.model.classifier[1], sess.model.classifier[2]
    P = sess.model.domain_concept_projection
    with torch.no_grad():
        explicit = float((sess.model.classifier[1:](P) ** 2).mean())
    closed = float(sess.model.ddo_l2_closed_loss())
    rel = abs(closed - explicit) / max(abs(explicit), 1e-12)
    # 梯度
    g_exp = torch.autograd.grad(explicit_t(sess.model), params, retain_graph=True, allow_unused=True)
    g_exp = torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                       for t, p in zip(g_exp, params)])
    closed_t = sess.model.ddo_l2_closed_loss()
    g_clo = torch.autograd.grad(closed_t, params, retain_graph=True, allow_unused=True)
    g_clo = torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                       for t, p in zip(g_clo, params)])
    grel = float((g_exp - g_clo).norm() / g_exp.norm().clamp_min(1e-12))
    print(f"  [{ds}] 显式 {explicit:.6f} 闭式 {closed:.6f} | 损失相对差 {rel:.2e} | 梯度相对差 {grel:.2e}", flush=True)
    passed = rel < 1e-4 and grel < 1e-3
    if not passed:
        ok = False
    return {"dataset": ds, "explicit": explicit, "closed": closed, "rel_loss_diff": rel,
            "grad_rel_diff": grel, "passed": passed}


def explicit_t(model):
    """带图的显式损失 (full 路径): classifier[1:](P) 的 mean(h²)。P 无梯度, 反传只到分类头参数。"""
    P = model.domain_concept_projection
    return (model.classifier[1:](P) ** 2).mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = [check_ds(ds) for ds in ("PACS", "LADV", "AWA2")]
    ok = all(r["passed"] for r in results)
    out = {"created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": device,
           "note": "外审第十七轮 P1: 三数据集的闭式等价核对 (零训练, 初始化会话)。容差: 损失 1e-4 / 梯度 1e-3 (float32)。",
           "results": results, "all_passed": ok}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[saved]", OUT, "| ALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
