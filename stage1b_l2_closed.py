"""DDO-L2 二阶矩闭式的验证与基准（外审第十五轮 P1；预注册见 RESULTS §3ar，先于运行提交）。

数学（§3ar）：当前 cached forward 对**全部** 40800 个方向输出**全部** C_cls=200 类的响应
（classifier[1:] 的 Linear 是 K→C_cls；reg_loss 形状 (40800, 200)），因此
  Bias-Free（full_bf，ln-free 无偏置路径）: R = mean(h²) = (1/(M·C))Σ‖A u_n‖² = tr(A G Aᵀ)/C   —— 精确等价
  Original（full，LN-affine+偏置路径）:  r_{n,c} = A_c·ū_n + c_c（ū = LN0(方向)），
          R = (1/C)·[ tr(A G Aᵀ) + 2 cᵀAμ + ‖c‖² ]，G = (1/M)Σ ūūᵀ，μ = (1/M)Σ ū  —— 完整公式（含偏置与均值项）
实现：`--ddo_l2_closed`（cbm_models._build_l2_closed_buffers / ddo_l2_closed_loss），默认关闭=逐方向显式路径。

本脚本做四件事（不训练任何正式 run）：
  V1 前向路径确认：reg_loss 的形状必须是 (40800, 200) —— "对所有方向算所有类别输出"的运行时证据；
  V2 真实数据等价：3 个 P1 初始化头（Original 与 full_bf 各一套配置），显式损失/梯度 vs 闭式损失/梯度
     （float64 参考值 + float32 实现的容差核对）；
  V3 修正表述的自测：tests_l2_equivalences.py 的 E2b/E4/E5 更正已单独运行；
  V4 短程基准：同 seed 的 2-epoch 冒烟（显式 vs 闭式），对比初始损失、epoch 指标、每 epoch 时间与显存。

用法: python stage1b_l2_closed.py [--selftest] [--skip_bench]
产物: experiment_results/Stage1B-L2/stage1b_l2_closed.json
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

from ld_dro import effective_response  # noqa: E402
from run_status import git_info  # noqa: E402
from stage1c_radius_diagnose import build_session  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")
NEW_JSON = os.path.join(OUT_DIR, "stage1b_l2_closed.json")
SEEDS = (1, 2, 3)


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    M, K, C = 300, 24, 7
    U = torch.randn(M, K, dtype=torch.float64)
    U = U - U.mean(-1, keepdim=True)
    U = U / torch.sqrt(U.var(-1, unbiased=False, keepdim=True) + 1e-5)   # LN0
    A = (torch.randn(C, K, dtype=torch.float64) * 0.3).requires_grad_(True)
    c = (torch.randn(C, dtype=torch.float64) * 0.2)

    # BF: closed == explicit
    G = U.T @ U / M
    closed_bf = ((A @ G) * A).sum() / C
    explicit_bf = ((A @ U.T) ** 2).mean()
    check("B1 BF: tr(AGAᵀ)/C == mean(‖Au‖²)（float64 精确）", abs(float(closed_bf - explicit_bf)) < 1e-10,
          f"{float(closed_bf):.9f} vs {float(explicit_bf):.9f}")
    g1 = torch.autograd.grad(closed_bf, A, retain_graph=True)[0]
    g2 = torch.autograd.grad(explicit_bf, A)[0]
    check("B2 BF: 梯度也精确相等", float((g1 - g2).abs().max()) < 1e-9, f"max|Δ|={float((g1 - g2).abs().max()):.2e}")

    # Orig: closed (含 μ 与偏置) == explicit (A·ū + c)
    mu = U.mean(0)
    closed_o = ((A @ G) * A).sum() / C + 2 * (c @ (A @ mu)) / C + (c * c).sum() / C
    explicit_o = (((A @ U.T) + c[:, None]) ** 2).mean()
    check("B3 Orig: (1/C)[tr(AGAᵀ)+2cᵀAμ+‖c‖²] == mean((A·ū+c)²)（float64 精确）",
          abs(float(closed_o - explicit_o)) < 1e-10, f"{float(closed_o):.9f}")
    g1 = torch.autograd.grad(closed_o, A, retain_graph=True)[0]
    g2 = torch.autograd.grad(explicit_o, A)[0]
    check("B4 Orig: 梯度精确相等", float((g1 - g2).abs().max()) < 1e-9)

    # 偏置为零时 Orig 公式退化到 BF 公式
    z = torch.zeros_like(c)
    closed_z = ((A @ G) * A).sum() / C + 2 * (z @ (A @ mu)) / C + (z * z).sum() / C
    check("B5 c=0 时 Orig 公式 == BF 公式", float(closed_z - closed_bf) == 0.0)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip_bench", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    # ===== 1. P1 初始化配置 (把 ddo_dirs/ddo_norm 换成 L2 两格的取值) =====
    base_cfg = {}
    for s in SEEDS:
        with open(f"logs/results/CUB_p1_dose_a1_seed{s}.json", encoding="utf-8") as f:
            base_cfg[s] = json.load(f)["run_config"]

    def make_cfg(s, **kw):
        c = dict(base_cfg[s])
        c.update(ddo_norm="l2", alpha=1.0, seed=s, run_tag="r15_l2closed")
        c.update(kw)                                     # 显式传入的键覆盖默认值
        return c

    # ===== 2. V1: 前向路径 = 全方向 × 全类别 =====
    sess_bf = build_session(make_cfg(SEEDS[0], ddo_dirs="full_bf"), SEEDS[0])
    x, y = sess_bf._diag_batch()
    with torch.no_grad():
        _, _, reg = sess_bf.model(x)
    check("V1 前向对全部方向输出全部类别: reg_loss 形状 = (40800, 200)",
          tuple(reg.shape) == (40800, 200), str(tuple(reg.shape)))

    # ===== 3. V2: 真实数据等价 (BF 与 Orig 各 3 个初始化头; 损失与梯度) =====
    def compare(sess, tag):
        """显式 vs 闭式: 损失值与对分类头的梯度。返回 dict。"""
        x, y = sess._diag_batch()
        sess.model.eval()
        params = [p for p in sess.model.classifier.parameters() if p.requires_grad]
        _, _, reg = sess.model(x)
        explicit = (reg ** 2).mean()
        closed = sess.model.ddo_l2_closed_loss()
        g_exp = torch.autograd.grad(explicit, params, retain_graph=True, allow_unused=True)
        g_exp = torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(g_exp, params)])
        g_clo = torch.autograd.grad(closed, params, retain_graph=True, allow_unused=True)
        g_clo = torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(g_clo, params)])
        rel = float(abs(closed - explicit) / max(abs(float(explicit)), 1e-12))
        gdiff = float((g_exp - g_clo).norm() / g_exp.norm().clamp_min(1e-12))
        # float64 参考 (显式): 用 double 参数重算
        with torch.no_grad():
            ln, lin = sess.model.classifier[1], sess.model.classifier[2]
            P = sess.model.domain_concept_projection
            if getattr(sess.model, "_ddo_dirs_ln_free", False):
                A64 = (lin.weight * ln.weight[None, :]).double()
                r64 = A64 @ P.reshape(-1, P.size(-1)).double().T
                ref = float((r64 ** 2).mean())
            else:
                P2 = P.reshape(-1, P.size(-1)).double()
                ub = (P2 - P2.mean(-1, keepdim=True)) / torch.sqrt(P2.var(-1, unbiased=False, keepdim=True) + 1e-5)
                A64 = (lin.weight * ln.weight[None, :]).double()
                c64 = (lin.weight @ ln.bias + lin.bias).double()
                r64 = A64 @ ub.T + c64[:, None]
                ref = float((r64 ** 2).mean())
        return {"explicit": float(explicit), "closed": float(closed), "float64_ref": ref,
                "rel_loss_diff": rel, "grad_rel_diff": gdiff}

    res = {}
    for name, dirs in (("bf", "full_bf"), ("orig", "full")):
        recs = []
        for s in SEEDS:
            cfg = make_cfg(s, ddo_dirs=dirs, ddo_l2_closed=True)
            sess = build_session(cfg, s)
            recs.append(compare(sess, name))
        res[name] = {"per_seed": recs,
                     "max_rel_loss_diff": max(r["rel_loss_diff"] for r in recs),
                     "max_grad_rel_diff": max(r["grad_rel_diff"] for r in recs)}
        check(f"V2 {name}: 闭式与显式的损失相对差 < 1e-5（float32 实现）",
              res[name]["max_rel_loss_diff"] < 1e-5, f"max {res[name]['max_rel_loss_diff']:.2e}")
        check(f"V2 {name}: 梯度相对差 < 1e-3（float32 实现；float64 下为精确等价）",
              res[name]["max_grad_rel_diff"] < 1e-3, f"max {res[name]['max_grad_rel_diff']:.2e}")
        check(f"V2 {name}: 显式损失与 float64 参考一致（< 1e-4 相对）",
              max(abs(r["explicit"] - r["float64_ref"]) / abs(r["float64_ref"]) for r in recs) < 1e-4)

    # ===== 4. V4: 短程基准 (2-epoch 冒烟, 同 seed, full_bf; 显式 vs 闭式) =====
    bench = {}
    if not a.skip_bench:
        import random

        import numpy as np_mem

        import random as _r
        import shutil
        import traceback

        def run_two_epoch(tag, closed):
            for stale in (f"logs/results/CUB_{tag}_seed1.json", f"logs/weights/CUB_{tag}_seed1",
                      f"logs/heads/CUB_{tag}_seed1.pt"):                     # save_head 也来自 P1 配置, 一并清理
                if os.path.isdir(stale):
                    shutil.rmtree(stale)
                elif os.path.exists(stale):
                    os.remove(stale)
            cfg = make_cfg(1, ddo_dirs="full_bf", ddo_l2_closed=closed, run_tag=tag)
            torch.manual_seed(99)
            _r.seed(99)
            np_mem.random.seed(99)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            args = argparse.Namespace(**dict(cfg))
            args.device = str(device)
            args.epochs = 2
            args.wandb = False
            from main_cached import CachedTrainingSession
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t = time.time()
            sess = CachedTrainingSession(args)
            try:
                sess.run()
            except Exception:
                traceback.print_exc()
                raise
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
            jp = f"logs/results/CUB_{tag}_seed1.json"            # make_cfg 的 seed=1 (manual_seed(99) 只是 RNG)
            if not os.path.exists(jp):
                raise RuntimeError(f"{tag}: 2-epoch run 没有写出结果 json (训练未完成, 见上方 traceback)")
            with open(jp, encoding="utf-8") as f:
                r = json.load(f)
            return {"epochs": r["epoch_metrics"][:2], "elapsed": r["elapsed_seconds"], "mem_MiB": mem,
                    "loss_step0": r["epoch_metrics"][0]["ddo_own_unweighted"]}
        r_exp = run_two_epoch("r15_bench_exp", False)
        r_clo = run_two_epoch("r15_bench_clo", True)
        for p in ("logs/results/CUB_r15_bench_exp_seed99.json", "logs/results/CUB_r15_bench_clo_seed99.json"):
            if os.path.exists(p):
                os.remove(p)
        e1, c1 = r_exp["epochs"][0], r_clo["epochs"][0]
        check("V4 两种实现的 epoch-1 指标接近（训练损失相对差 < 1e-3; 源域 Acc 差 < 0.5 pp）",
              abs(e1["cls"] - c1["cls"]) / max(e1["cls"], 1e-9) < 1e-3
              and abs(e1["source_acc"] - c1["source_acc"]) < 0.5,
              f"损失 {e1['cls']:.4f} vs {c1['cls']:.4f}; 源 Acc {e1['source_acc']:.2f} vs {c1['source_acc']:.2f}")
        bench = {"explicit": r_exp, "closed": r_clo, "note": "epoch_metrics 无 train_acc; 判据用训练损失与源域 Acc",
                 "speedup_total": r_exp["elapsed"] / max(r_clo["elapsed"], 1e-9),
                 "mem_MiB_explicit": r_exp["mem_MiB"], "mem_MiB_closed": r_clo["mem_MiB"],
                 "sec_per_epoch_explicit": r_exp["elapsed"] / 2, "sec_per_epoch_closed": r_clo["elapsed"] / 2}
        print(f"  基准: 显式 {r_exp['elapsed']:.1f}s / 闭式 {r_clo['elapsed']:.1f}s "
              f"(含加载; epoch1 损失 {r_exp['epochs'][0]['cls']:.4f} vs {r_clo['epochs'][0]['cls']:.4f})")

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": str(device),
           "note": "DDO-L2 二阶矩闭式的验证与基准 (外审第十五轮 P1; §3ar)。V1 前向全类输出; "
                   "V2 真实数据等价 (损失/梯度, float64 参考); V4 短程基准 (2-epoch, full_bf)。",
           "verification": res, "benchmark": bench, "all_checks_passed": ok,
           "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {NEW_JSON}  (耗时 {out['elapsed_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
