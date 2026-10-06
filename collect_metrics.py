"""
指标收集器: 把每个 run 的日志解析成结构化 CSV, 供画图/答辩材料直接使用。

产物 (logs/metrics/):
  {run_name}.csv      逐 epoch: epoch,time_s,train_acc,source_acc,target_acc,
                                 best_target,gamma,H_q,w_max,w_min,w_std
  summary.csv         全部 run: run_name,dataset,tag,seed,best_target,best_source,
                                 gamma_final,H_q_final,w_std_final

用法: python collect_metrics.py   (幂等, 重复运行覆盖更新)
"""
import csv
import glob
import os
import re

RESULTS_DIR = "logs/results"
OUT_DIR = "logs/metrics"
DOWNLOADS = "downloads"

DATASETS = ("CUB", "AWA2", "LADA", "LADV", "RIVAL10", "PACS", "OfficeHome", "DomainNet")


def split_run(stem):
    """'CUB_prior_residual_seed1' -> (CUB, prior_residual, 1); 解析失败返回 None"""
    for ds in DATASETS:
        if stem.startswith(ds + "_"):
            rest = stem[len(ds) + 1:]
            if "_seed" in rest:
                tag, seed = rest.rsplit("_seed", 1)
                if seed.isdigit():
                    return ds, tag, int(seed)
    return None


def candidate_logs(run_name):
    """返回候选日志路径, 按优先级: fix(修复后) > pipe > gate(修复前) > baseline;
    每个 run 先试标准命名再试别名 (none→ddo 等)"""
    stem = run_name.replace(".json", "")
    parts = split_run(stem)
    cands = [os.path.join(DOWNLOADS, f"pipe_{stem}.log")]
    if parts:
        ds, tag, seed = parts
        alias = {"none": "ddo", "prior_residual": "c3", "full": "c4", "c3+full": "c3+c4"}
        for prefix in ("fix", "gate"):
            for t in (tag, alias.get(tag, tag)):
                cands.append(os.path.join(DOWNLOADS, f"{prefix}_{ds}_{t}_s{seed}.log"))
        cands.append(os.path.join(DOWNLOADS, f"baseline_{stem}.log"))
    return cands


def parse_log(path):
    rows = []
    cur = {}
    pending = {}  # 诊断行([Weights]/[Ch4])先于其 Epoch Summary 出现, 先攒后并入
    text = open(path, encoding="utf-8", errors="ignore").read()
    for line in text.splitlines():
        m = re.search(r"\[Ch4\] gamma=([0-9.]+)", line)
        if m:
            pending["gamma"] = float(m.group(1))
        m = re.search(r"\[Weights\] H\(q\)=([0-9.]+) max=([0-9.]+) min=([0-9.]+) std=([0-9.]+)", line)
        if m:
            pending["H_q"], pending["w_max"], pending["w_min"], pending["w_std"] = map(float, m.groups())
        m = re.search(r"Epoch (\d+) Summary:", line)
        if m:
            cur = {"epoch": int(m.group(1), ), **pending}
            pending = {}
            continue
        for pat, key in [
            (r"Time: ([0-9.]+)s", "time_s"),
            (r"Train Acc: ([0-9.]+)%", "train_acc"),
            (r"Source Acc: ([0-9.]+)%", "source_acc"),
            (r"Target Acc: ([0-9.]+)%", "target_acc"),
        ]:
            m = re.search(pat, line)
            if m and cur:
                cur[key] = float(m.group(1))
        if "train_acc" in cur and "source_acc" in cur and "target_acc" in cur and "epoch" in cur:
            if not rows or rows[-1]["epoch"] != cur["epoch"]:
                rows.append(cur)
            cur = {}
    best = re.findall(r"Best Target Accuracy: ([0-9.]+)%", text)
    best_src = re.findall(r"Best Source Accuracy: ([0-9.]+)%", text)
    return rows, (float(best[-1]) if best else None), (float(best_src[-1]) if best_src else None)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    fields = ["epoch", "time_s", "train_acc", "source_acc", "target_acc",
              "gamma", "H_q", "w_max", "w_min", "w_std"]
    for jf in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        run_name = os.path.basename(jf)
        log = next((c for c in candidate_logs(run_name) if os.path.exists(c)), None)
        if log is None:
            continue
        rows, best_t, best_s = parse_log(log)
        if not rows:
            continue
        with open(os.path.join(OUT_DIR, run_name.replace(".json", ".csv")), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        last = rows[-1]
        parts = split_run(run_name.replace(".json", ""))
        ds, tag, seed = parts if parts else ("?", "?", "?")
        summary.append({
            "run_name": run_name.replace(".json", ""), "dataset": ds, "tag": tag,
            "seed": seed, "epochs_done": len(rows),
            "best_target": best_t, "best_source": best_s,
            "gamma_final": last.get("gamma", ""),
            "H_q_final": last.get("H_q", ""), "w_std_final": last.get("w_std", ""),
        })
        print(f"[saved] {run_name}: {len(rows)} epochs, best_target={best_t}")
    if summary:
        with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"summary.csv 已更新 ({len(summary)} runs)")


if __name__ == "__main__":
    main()
