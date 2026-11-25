"""实验矩阵驱动 (2026-11-25): 顺序跑一组配置。

  - 以结果 json 的完成状态判定跳过 (status / 完整 epoch / 关键配置), 不只看文件是否存在
  - 检查子进程退出码 (训练入口在失败时返回非零)
  - 每个训练的 stdout 写到 experiment_results/<目录>/driver_logs/ (已加入 .gitignore, 不入库)
  - 公共配置与 2026-11-25 之后的全部新实验保持一致: 严格 final 协议, 训练期间不访问目标域,
    固定 epoch 梯度诊断, 不保存模型

用法:
  python run_matrix.py dose      [--seeds 1 2 3] [--dry_run] [--overwrite_run]
  python run_matrix.py weights   ...
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, ROOT)
from run_status import result_is_complete  # noqa: E402

CLIP = os.path.join(ROOT, "clip_weights", "ViT-L-14.pt")
CA_PRIOR = os.path.join(ROOT, "experiment_results", "CA-Diagnostic", "ca_prior_cub.pt")

# 公共配置: CUB 缓存特征管线, 100 epochs, batch 256, AdamW lr/wd 与历史一致
COMMON = ["--epochs", "100", "--batch_size", "256", "--beta", "0",
          "--lr", "0.0001", "--weight_decay", "0.0001", "--model_selection", "final",
          "--target_eval", "final_only", "--grad_diag", "--no_save_model", "--CLIP_type", CLIP]

DOSE = [(0.0, "dose_a0"), (0.125, "dose_a0p125"), (0.25, "dose_a0p25"), (0.5, "dose_a0p5"),
        (1.0, "dose_a1"), (2.0, "dose_a2"), (4.0, "dose_a4")]
EQUAL = ["--CBM_type", "clip_cbm", "--weight_mode", "none", "--lambda_kl", "0", "--lambda_graph", "0"]
# 第三章入口的公共段 (不含 alpha, 由各配置给出, 避免同一参数出现两次)
CH3 = ["--CBM_type", "clip_cbm", "--lambda_kl", "0.01", "--lambda_graph", "0"]
A1 = ["--alpha", "1.0"]

# 目录 -> [(dataset, run_tag, 额外参数)]
MATRICES = {
    # DDO-Dose: 第三章入口 + 等权 DDO, CUB x 3 seeds x 7 个 alpha
    "dose": ("DDO-Dose", [("CUB", tag, EQUAL + ["--alpha", str(a)]) for a, tag in DOSE]),
    # 与剂量曲线对照的加权变体 (统一评价口径由 final_diag / grad_diag 提供)
    "weights": ("DDO-Dose", [
        ("CUB", "w_rwddo_a1", CH3 + A1 + ["--weight_mode", "prior_residual"]),
        ("CUB", "w_rwfixed_a1", CH3 + A1 + ["--weight_mode", "fixed"]),
        ("CUB", "w_grouped_a1", CH3 + A1 + ["--weight_mode", "prior_residual", "--grouped_weights"]),
        ("CUB", "w_cafixed_a1", CH3 + A1 + ["--weight_mode", "ca_fixed", "--ca_prior_file", CA_PRIOR,
                                             "--ca_beta", "10"]),
    ]),
    # 剂量 x 加权交互: RW-DDO (prior_residual) 在其余三个剂量下 (alpha=1 已在 weights 矩阵里),
    # 与等权 DDO 的同剂量 run 逐 seed 配对
    "weights_dose": ("DDO-Dose", [
        ("CUB", f"w_rwddo_a{t}", CH3 + ["--weight_mode", "prior_residual", "--alpha", str(a)])
        for a, t in [(0.5, "0p5"), (2.0, "2"), (4.0, "4")]]),
}


def expected(extra):
    """完成判定用的关键配置 (同一参数出现多次时取最后一个, 与 argparse 一致)"""
    out = {"epochs": 100, "batch_size": 256}
    for k, name in (("--alpha", "alpha"), ("--weight_mode", "weight_mode"), ("--batch_size", "batch_size"),
                    ("--epochs", "epochs")):
        if k in extra:
            v = extra[len(extra) - extra[::-1].index(k)]        # 最后一次出现后的那个值
            out[name] = float(v) if k == "--alpha" else (int(v) if k in ("--batch_size", "--epochs") else v)
    return out


def _self_test_expected():
    """导入时的自检: 2026-11-25 weights_dose 矩阵曾因 alpha 出现两次而被误判为失败"""
    e = expected(CH3 + A1 + ["--weight_mode", "prior_residual", "--alpha", "2.0"] + COMMON)
    assert e == {"epochs": 100, "batch_size": 256, "weight_mode": "prior_residual", "alpha": 2.0}, e
    e = expected(EQUAL + ["--alpha", "0.125"] + COMMON)
    assert e["alpha"] == 0.125 and e["weight_mode"] == "none", e
    for name, (out_dir, entries) in MATRICES.items():
        for _ds, tag, extra in entries:
            assert "alpha" in expected(extra + COMMON), f"矩阵 {name} 的 {tag} 没有 alpha"


_self_test_expected()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix", choices=sorted(MATRICES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite_run", action="store_true")
    opt = ap.parse_args()

    out_dir, entries = MATRICES[opt.matrix]
    log_dir = os.path.join(ROOT, "experiment_results", out_dir, "driver_logs")
    os.makedirs(log_dir, exist_ok=True)
    jobs = [(ds, tag, extra, seed) for ds, tag, extra in entries for seed in opt.seeds]
    print(f"[matrix {opt.matrix}] {len(entries)} 个配置 x {len(opt.seeds)} seeds = {len(jobs)} runs", flush=True)
    if opt.dry_run:
        for ds, tag, extra, seed in jobs:
            print(f"  {ds}_{tag}_seed{seed}: {extra}")
        return

    t0 = time.time()
    done = skipped = failed = 0
    failures = []
    for i, (ds, tag, extra, seed) in enumerate(jobs):
        run_name = f"{ds}_{tag}_seed{seed}"
        path = os.path.join(ROOT, "logs", "results", run_name + ".json")
        exp = expected(extra + COMMON)
        if result_is_complete(path, exp) and not opt.overwrite_run:
            skipped += 1
            print(f"[skip {i + 1}/{len(jobs)}] {run_name}", flush=True)
            continue
        cmd = [PY, "-u", "main_cached.py", "--dataset", ds, "--seed", str(seed), "--run_tag", tag,
               "--overwrite_run"] + COMMON + extra
        log = os.path.join(log_dir, run_name + ".log")
        print(f"[run  {i + 1}/{len(jobs)}] {run_name}  {time.strftime('%H:%M:%S')}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        ok = result_is_complete(path, exp)
        if ok:
            done += 1
        else:
            failed += 1
            failures.append(run_name)
        print(f"       -> {'OK' if ok else 'FAIL(exit %s)' % r.returncode}", flush=True)

    print(f"\n新完成 {done} | 跳过 {skipped} | 失败 {failed} | 耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    if failures:
        print("失败: " + ", ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
