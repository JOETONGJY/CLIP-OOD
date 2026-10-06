"""
可断点续跑的实验总管线 (独立进程运行, 不依赖 ZCode 会话):

  - 每个 run 以 logs/results/{run_name}.json 是否存在判断完成, 重复启动自动跳过
  - 特征编码以 cache/*.pt 是否存在判断完成
  - 顺序: 编码(缺的) -> 闸门实验 -> 第四章消融 -> 汇总

启动方式:
  双击 run_pipeline.bat (或)  D:/anaconda3/envs/trl4060/python.exe run_pipeline.py
中断恢复: 直接再次启动即可, 已完成的自动跳过。
"""
import os
import subprocess
import sys
import time

ROOT = r"F:\CLIP-OOD"
PY = r"D:\anaconda3\envs\trl4060\python.exe"
CLIP = r"F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
os.chdir(ROOT)

RUNS = []  # (run_name_json, [args...])


def add(dataset, tag, extra):
    for seed in (1, 2, 3):
        args = ["--dataset", dataset, "--epochs", "100", "--batch_size", "256",
                "--alpha", "1.0", "--beta", "0", "--seed", str(seed),
                "--CLIP_type", CLIP] + extra
        RUNS.append((f"{dataset}_{tag}_seed{seed}.json", args))


# ===== 闸门实验: 4 组 × {CUB, AWA2} × 3 seeds =====
for ds in ("CUB", "AWA2"):
    add(ds, "none", ["--CBM_type", "clip_cbm", "--weight_mode", "none"])
    add(ds, "prior_residual", ["--CBM_type", "clip_cbm",
                               "--weight_mode", "prior_residual", "--lambda_kl", "0.01"])
    add(ds, "full", ["--CBM_type", "clip_cbm_subspace", "--ch4_mode", "full",
                     "--tau", "0.8", "--lambda_inv", "1.0"])
    add(ds, "c3+full", ["--CBM_type", "clip_cbm_subspace", "--ch4_mode", "full",
                        "--tau", "0.8", "--lambda_inv", "1.0",
                        "--use_c3_weights", "--weight_mode", "prior_residual",
                        "--lambda_kl", "0.01"])

# ===== 第四章消融 (CUB) =====
for mode in ("headonly", "softproj", "hardproj"):
    add("CUB", mode, ["--CBM_type", "clip_cbm_subspace", "--ch4_mode", mode,
                      "--tau", "0.8", "--lambda_inv", "1.0"])


def encode_missing(dataset):
    need = [f"cache/{dataset}_{s}.pt" for s in ("train", "source_test", "target_test")]
    if all(os.path.exists(p) for p in need):
        print(f"[skip] {dataset} 特征已编码")
        return
    print(f"[encode] {dataset} ...", flush=True)
    subprocess.run([PY, "encode_features.py", "--dataset", dataset,
                    "--batch_size", "64", "--CLIP_type", CLIP],
                   stdout=open(f"downloads/encode_{dataset}.log", "w"),
                   stderr=subprocess.STDOUT)


def main():
    t0 = time.time()
    for ds in ("CUB", "AWA2"):
        encode_missing(ds)

    for i, (json_name, args) in enumerate(RUNS):
        path = os.path.join("logs", "results", json_name)
        if os.path.exists(path):
            print(f"[skip {i+1}/{len(RUNS)}] {json_name}")
            continue
        log = os.path.join("downloads", f"pipe_{json_name.replace('.json', '')}.log")
        print(f"[run  {i+1}/{len(RUNS)}] {json_name}  {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run([PY, "main_cached.py"] + args,
                           stdout=open(log, "w"), stderr=subprocess.STDOUT)
        ok = os.path.exists(path)
        print(f"       -> {'OK' if ok else 'FAIL(exit %s)' % r.returncode}", flush=True)
        # 每个 run 结束即保存指标 (断点/中断也不丢)
        subprocess.run([PY, "collect_metrics.py"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("\n===== 汇总 =====", flush=True)
    subprocess.run([PY, "aggregate_results.py"])
    subprocess.run([PY, "collect_metrics.py"])
    subprocess.run([PY, os.path.join("experiment_results", "export_results.py")])
    print(f"\n管线结束, 总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
