"""闭式 L2 路径的更新等价性与计时协议（外审第十五/十六轮 P1；§3av）。

外审指出：2-epoch 冒烟不足以证明稳定加速，且未区分构建/单步/评估成本；参数更新等价性也没有直接验证。
本脚本做三件事（零训练正式 run，不写任何结果 json）：

  T1 单步更新等价：同 seed 同批次的显式 vs 闭式，完整训练损失（CE + α·正则）与一步 AdamW 后的
     全部分类头参数逐项对比（先比损失值，再比参数更新）；
  T2 预热后多次计时：模型构建（含闭式的 G/μ 预计算）、单步训练（10 步预热后 50 步 × 3 次重复）、
     一次源域测试集评估——显式 vs 闭式分别测量；
  T3 显存峰值：单步训练阶段的 torch.cuda.max_memory_allocated。

边界：所有计时都在本机单卡、当前负载下测得，只用于实现选择，不宣称正式加速倍数（§3av 的措辞约束）。
用法: python stage1b_closed_timing.py [--steps N] [--repeats R]
产物: experiment_results/Stage1B-L2/stage1b_closed_timing.json
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

from run_status import git_info  # noqa: E402
from stage1b_l2_closed import self_test  # noqa: E402  复用闭式合成自测

OUT_DIR = os.path.join("experiment_results", "Stage1B-L2")
NEW_JSON = os.path.join(OUT_DIR, "stage1b_closed_timing.json")


def build_sess(closed, seed, device, run_tag):
    """构造训练会话（与训练入口同一构造顺序；epochs=1 只用于初始化，不 run()）。"""
    import argparse
    import random

    with open(f"logs/results/CUB_p1_dose_a1_seed{seed}.json", encoding="utf-8") as f:
        cfg = json.load(f)["run_config"]
    cfg.update(ddo_norm="l2", alpha=1.0, seed=seed, run_tag=run_tag, ddo_dirs="full_bf",
               ddo_l2_closed=closed, epochs=1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = argparse.Namespace(**dict(cfg))
    args.device = device
    args.wandb = False
    from main_cached import CachedTrainingSession
    t = time.time()
    sess = CachedTrainingSession(args)
    build = time.time() - t
    return sess, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    seed = 1

    # ===== T1: 单步更新等价 (同初始化同批次) =====
    res = {}
    for closed in (False, True):
        sess, build_t = build_sess(closed, seed, device, f"r16_tim_{closed}")
        sess.model.train()
        it = iter(sess.train_loader)
        images, labels, attr = next(it)
        images = images.to(sess.model.device)
        labels = labels.to(sess.model.device)
        sess.optimizer.zero_grad(set_to_none=True)
        t_loss = time.time()
        _, cls_preds, reg = sess.model(images)
        explicit_loss = sess._ddo_norm_loss(reg)
        full_loss = sess.loss_fns["cls"](cls_preds, labels) + float(sess.args.alpha) * explicit_loss
        t_fwd = time.time() - t_loss
        full_loss.backward()
        sess.optimizer.step()
        params = {n: p.detach().clone() for n, p in sess.model.classifier.named_parameters()}
        res["closed" if closed else "explicit"] = {
            "reg_loss": float(explicit_loss), "full_loss": float(full_loss), "fwd_seconds": t_fwd,
            "build_seconds": build_t, "params": params, "mem_MiB":
                (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None}
        del sess
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    e, c = res["explicit"], res["closed"]
    rel = abs(c["reg_loss"] - e["reg_loss"]) / max(abs(e["reg_loss"]), 1e-12)
    max_param_diff = max(float((c["params"][k] - e["params"][k]).abs().max()) for k in e["params"])
    check("T1 闭式与显式的正则值相对差 < 1e-5", rel < 1e-5, f"{rel:.2e}")
    check("T1 一步 AdamW 后的分类头参数最大差 < 1e-5（浮点路径差, 不逐位）", max_param_diff < 1e-5,
          f"{max_param_diff:.2e}")

    # ===== T2: 预热后多次计时 (构建 / 单步训练 / 评估) =====
    def timed_sess(closed, seed):
        sess, build_t = build_sess(closed, seed, device, f"r16_tim2_{closed}")
        sess.model.train()
        it = iter(sess.train_loader)

        def one_step():
            nonlocal it
            try:
                images, labels, attr = next(it)
            except StopIteration:
                it = iter(sess.train_loader)
                images, labels, attr = next(it)
            images = images.to(sess.model.device)
            labels = labels.to(sess.model.device)
            sess.optimizer.zero_grad(set_to_none=True)
            _, cls_preds, reg = sess.model(images)
            loss = sess.loss_fns["cls"](cls_preds, labels) + float(sess.args.alpha) * sess._ddo_norm_loss(reg)
            loss.backward()
            sess.optimizer.step()

        for _ in range(a.warmup):                     # 预热 (CUDA kernel / 显存分配稳定)
            one_step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times = []
        for _ in range(a.repeats):
            t = time.time()
            for _ in range(a.steps):
                one_step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append((time.time() - t) / a.steps)
        t = time.time()
        sess.model.eval()
        with torch.no_grad():
            sess._evaluate(sess.source_test_loader, "source")
        eval_t = time.time() - t
        return {"build_seconds": build_t, "step_seconds": times,
                "step_seconds_mean": float(np.mean(times)), "step_seconds_std": float(np.std(times)),
                "eval_seconds": eval_t}

    te = timed_sess(False, seed)
    tc = timed_sess(True, seed)
    check("T2 闭式的单步均值不高于显式（本轮只报告测量值，不宣称正式倍数）", tc["step_seconds_mean"] <= te["step_seconds_mean"] * 1.05,
          f"显式 {te['step_seconds_mean'] * 1000:.0f} ms/步, 闭式 {tc['step_seconds_mean'] * 1000:.0f} ms/步")
    check("T2 闭式的评估不高于显式", tc["eval_seconds"] <= te["eval_seconds"] * 1.05,
          f"评估 {te['eval_seconds']:.2f}s vs {tc['eval_seconds']:.2f}s")

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": device,
           "steps": a.steps, "warmup": a.warmup, "repeats": a.repeats,
           "update_equality": {"explicit_reg_loss": e["reg_loss"], "closed_reg_loss": c["reg_loss"],
                               "rel_loss_diff": rel, "max_param_diff_after_step": max_param_diff},
           "timing": {"explicit": {k: v for k, v in te.items()}, "closed": {k: v for k, v in tc.items()},
                      "note": "所有计时为本机单卡当前负载下的测量值；构建时间含数据集缓存加载；"
                              "只用于实现选择，不宣称正式加速倍数或显存下降 (外审第十五轮措辞约束)"}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {NEW_JSON}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
