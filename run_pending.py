"""
待补实验管线 (断点续跑, 独立进程运行):

  阶段 1  tau 维数扫描: tau ∈ {0.6, 0.7, 0.9, 0.95} × {CUB, AWA2} × 3 seeds = 24 runs
          (tau=0.8 已有, 见 CUB_full_seed*.json / AWA2_full_seed*.json)
  阶段 2  双头消融跨数据集: {hardproj, softproj, headonly} × {AWA2, LADA, LADV} × 3 seeds = 27 runs

  - 每个 run 以 logs/results/{run_name}.json 是否存在判断完成, 重复启动自动跳过
    (2026-11-25 起: 还会校验结果内容 —— status / 完整 epoch / 关键配置)
  - 特征缓存 cache/*.pt 已就绪, 无需重新编码
  - 日志写 downloads/pipe_{run_name}.log (collect_metrics.py 按此命名查找)

启动方式:
  双击 run_pending.bat  或  D:\\anaconda3\\envs\\trl4060\\python.exe run_pending.py
  只跑其中一段: python run_pending.py --only tau     /  --only dual
中断恢复: 直接再次启动即可, 已完成的自动跳过。
"""
import argparse
import os
import subprocess
import sys
import time

from run_status import result_is_complete

ROOT = r"F:\CLIP-OOD"
PY = r"D:\anaconda3\envs\trl4060\python.exe"
CLIP = r"F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
os.chdir(ROOT)

TAUS = (0.6, 0.7, 0.9, 0.95)
DUAL_MODES = ("hardproj", "softproj", "headonly")


def result_name(dataset, tag, extra, seed):
    """结果文件名, 与 main.py 的规则一致: 严格 final 协议 (默认) 且未指定 --run_tag 时,
    方法 tag 会加 _sel 后缀 (2026-11-25 修正: 旧版此处漏了后缀, dual 阶段会永远判定未完成)。"""
    if "--run_tag" not in extra and not tag.endswith("_sel"):
        tag += "_sel"
    return f"{dataset}_{tag}_seed{seed}.json"


def build_runs(stage):
    runs = []  # (run_name_json, [args...])

    def add(dataset, tag, extra, seed):
        # 2026-11-25: --no_save_model —— 管线不需要检查点, 且训练入口现在默认拒绝覆盖已存在的检查点
        args = ["--dataset", dataset, "--epochs", "100", "--batch_size", "256",
                "--alpha", "1.0", "--beta", "0", "--seed", str(seed),
                "--CLIP_type", CLIP, "--no_save_model"] + extra
        runs.append((result_name(dataset, tag, extra, seed), args))

    if stage in ("tau", "all"):
        for tau in TAUS:
            tag = f"full_tau{int(round(tau * 100)):02d}"
            for ds in ("CUB", "AWA2"):
                for seed in (1, 2, 3):
                    add(ds, tag, ["--CBM_type", "clip_cbm_subspace", "--ch4_mode", "full",
                                  "--tau", str(tau), "--lambda_inv", "1.0",
                                  "--run_tag", tag], seed)

    if stage in ("dual", "all"):
        for mode in DUAL_MODES:
            for ds in ("AWA2", "LADA", "LADV"):
                for seed in (1, 2, 3):
                    add(ds, mode, ["--CBM_type", "clip_cbm_subspace", "--ch4_mode", mode,
                                   "--tau", "0.8", "--lambda_inv", "1.0"], seed)

    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="all", choices=["all", "tau", "dual"])
    opt = parser.parse_args()

    runs = build_runs(opt.only)
    print(f"待补实验: {len(runs)} runs (--only {opt.only})", flush=True)

    t0 = time.time()
    done = skipped = failed = 0
    for i, (json_name, args) in enumerate(runs):
        path = os.path.join("logs", "results", json_name)
        # 2026-11-25: 完成判定改为校验结果内容 (status / 完整 epoch / 关键配置), 不再只看文件是否存在
        expect = {"epochs": 100, "batch_size": 256, "alpha": 1.0}
        if result_is_complete(path, expect):
            skipped += 1
            print(f"[skip {i+1}/{len(runs)}] {json_name}", flush=True)
            continue
        log = os.path.join("downloads", f"pipe_{json_name.replace('.json', '')}.log")
        print(f"[run  {i+1}/{len(runs)}] {json_name}  {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run([PY, "main_cached.py"] + args,
                           stdout=open(log, "w"), stderr=subprocess.STDOUT)
        ok = result_is_complete(path, expect)
        if ok:
            done += 1
        else:
            failed += 1
        print(f"       -> {'OK' if ok else 'FAIL(exit %s)' % r.returncode}", flush=True)
        # 每个 run 结束即保存指标 (中断也不丢)
        subprocess.run([PY, "collect_metrics.py"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"\n新完成 {done} | 跳过 {skipped} | 失败 {failed}", flush=True)

    print("\n===== 汇总 (tau 扫描) =====", flush=True)
    subprocess.run([PY, "aggregate_results.py"])
    subprocess.run([PY, "collect_metrics.py"])
    print(f"\n管线结束, 总耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
