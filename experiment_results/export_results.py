"""
实验结果归档器: 把散落在 logs/ 的产物复制/生成为 experiment_results/ 的分类结构。
幂等可重复运行(数据量小, 全量覆盖复制), 建议每批实验后运行一次。

用法: D:/anaconda3/envs/trl4060/python.exe experiment_results/export_results.py
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # F:\CLIP-OOD
os.chdir(ROOT)
PY = sys.executable
OUT = "experiment_results"


def cp(src, dstdir):
    os.makedirs(dstdir, exist_ok=True)
    shutil.copy2(src, dstdir)


def main():
    # 02_单run指标: 结果 json + 逐 epoch CSV
    n = 0
    for f in glob.glob("logs/results/*.json"):
        cp(f, os.path.join(OUT, "02_单run指标")); n += 1
    for f in glob.glob("logs/metrics/*.csv"):
        cp(f, os.path.join(OUT, "02_单run指标")); n += 1
    print(f"02_单run指标: {n} 个文件")

    # 01_汇总表: aggregate_results 输出 + summary.csv
    r = subprocess.run([PY, "aggregate_results.py"], capture_output=True, text=True)
    with open(os.path.join(OUT, "01_汇总表", "aggregate_all.txt"), "w", encoding="utf-8") as f:
        f.write(r.stdout)
        if r.stderr:
            f.write("\n[stderr]\n" + r.stderr)
    for ds in ("CUB", "AWA2"):
        r = subprocess.run([PY, "aggregate_results.py", "--dataset", ds],
                           capture_output=True, text=True)
        if r.stdout.strip():
            with open(os.path.join(OUT, "01_汇总表", f"aggregate_{ds}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(r.stdout)
    print("01_汇总表: 已刷新 (aggregate_all/CUB/AWA2)")

    # 03_机制分析: 每个 run 的最终权重诊断向量(只拷最后一个 epoch)
    n = 0
    for d in glob.glob("logs/weights/*"):
        eps = sorted(glob.glob(os.path.join(d, "epoch_*.pt")))
        if eps:
            cp(eps[-1], os.path.join(OUT, "03_机制分析", "final_weights")); n += 1
    print(f"03_机制分析: 最终权重向量 {n} 个")

    print("归档完成 -> experiment_results/")


if __name__ == "__main__":
    main()
