"""
E1 可信-困难象限干预实验 (2026-11-21, 预注册)

研究问题 (承 07_权重逃逸诊断): RW-DDO 的 weight avoidance 是否伤害泛化?
干预: 可靠性 r 前 1/3 ∩ 收敛 DDO 损失 ℓ 前 1/3 的描述符, 其 residual δ 冻结为 0
      (权重退化为先验), 其余照常训练。
比较: 同 seed、同协议 (严格 DG final) 的基线 vs 干预, 唯一差异 = 冻结掩码。
对照: CUB 额外做"随机冻结 68 个" (seed 97 生成), 排除"缩减 δ 容量本身有效"的混淆。

预注册判据 (跑之前写死):
  主判定 = CUB 配对 uplift (interv - base, sel_target_acc) 为正
           且 3 seeds 中至少 2 个方向一致;
  次判定 = 4 数据集 uplift 方向一致性 (binomial 符号检验视角, 仅描述)。

用法:
  python e1_trusted_hard_intervention.py all            # 全流程 (断点续跑, json 存在即跳过)
  python e1_trusted_hard_intervention.py baseline --dataset CUB --seed 2
  python e1_trusted_hard_intervention.py mask --dataset CUB
  python e1_trusted_hard_intervention.py intervene --dataset CUB --seed 1 \
      --mask freeze_masks/CUB_trusted_hard.pt --tag freeze_th
  python e1_trusted_hard_intervention.py verify         # 检查冻结是否生效 (δ[mask]==0)
  python e1_trusted_hard_intervention.py report
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import argparse
import glob
import json
import math
import subprocess

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
E1_DIR = os.path.join("experiment_results", "E1_可信困难干预")
MASK_DIR = os.path.join(E1_DIR, "freeze_masks")
RUNS_DIR = os.path.join(E1_DIR, "runs")
RESULTS_DIR = os.path.join("logs", "results")
WEIGHTS_DIR = os.path.join("logs", "weights")

DATASET_BATCH = {"CUB": 256, "AWA2": 256, "LADV": 256, "PACS": 128}
SEEDS = [1, 2, 3]
FRAC = 3   # 象限取前 1/3
CLIP_PATH = os.path.join(HERE, "clip_weights", "ViT-L-14.pt")
RAND_CTRL_SEED = 97


# ---------------------------------------------------------------- 工具
def base_argv(dataset, seed):
    return ["--dataset", dataset, "--seed", str(seed),
            "--CLIP_type", CLIP_PATH,
            "--epochs", "100", "--batch_size", str(DATASET_BATCH[dataset])]


def baseline_json(dataset, seed):
    return os.path.join(RESULTS_DIR, f"{dataset}_prior_residual_sel_seed{seed}.json")


def run_tag(dataset, seed, kind):
    return f"{dataset}_{kind}_seed{seed}"


def sh(args_list, log_path):
    """子进程跑一次训练, 输出追加到 driver_log (不用 -I: 它会忽略 PYTHONIOENCODING)"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(args_list) + "\n")
        f.flush()
        ret = subprocess.call(args_list, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    if ret != 0:
        raise RuntimeError(f"子进程失败 (exit={ret}), 见 {log_path}")


# ---------------------------------------------------------------- 子命令
def cmd_baseline(dataset, seed):
    """基线: 与主表同管线、严格 DG final 协议, 快照存独立目录"""
    tag = run_tag(dataset, seed, "prior_residual_sel")
    snap_dir = os.path.join(RUNS_DIR, tag, "snapshots")
    env = dict(os.environ)
    env["DIAG_SNAP_DIR"] = snap_dir
    os.environ["DIAG_SNAP_DIR"] = snap_dir   # 子进程继承
    sh([PY, "-I", "weight_loss_coupling_diagnosis.py"] + base_argv(dataset, seed),
       os.path.join(E1_DIR, "driver_log.txt"))


def cmd_mask(dataset):
    """从 seed1 基线构建冻结掩码: r 前1/3 ∩ ℓ(最终) 前1/3"""
    prior = torch.load(os.path.join(WEIGHTS_DIR, run_tag(dataset, 1, "prior_residual_sel"),
                                    "epoch_0000.pt"), map_location="cpu",
                       weights_only=False)["reliability_prior"].float().numpy()
    snap_path = os.path.join(RUNS_DIR, run_tag(dataset, 1, "prior_residual_sel"),
                             "snapshots", "epoch_0099.pt")
    ell = None
    if os.path.exists(snap_path):
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        if "ell" in snap:
            ell = snap["ell"].numpy()
    if ell is None:
        # 回退: 07 诊断目录的 diag json 里有逐项收敛 ℓ (CUB seed1 为早期快照格式)
        diag = os.path.join("experiment_results", "07_权重逃逸诊断",
                            f"diag_{run_tag(dataset, 1, 'prior_residual_sel')}.json")
        pp = json.load(open(diag, encoding="utf-8"))["per_prompt"]
        ell = np.zeros(len(pp))
        for p in pp:
            ell[p["idx"]] = p["ell"]
        print(f"[mask] {dataset}: 快照无 ell, 使用 07 诊断 json 回退")
    n = len(prior)
    k = math.ceil(n / FRAC)
    rk_r = np.argsort(np.argsort(-prior))
    rk_e = np.argsort(np.argsort(-ell))
    both = (rk_r < k) & (rk_e < k)
    os.makedirs(MASK_DIR, exist_ok=True)
    torch.save({"indices": torch.where(torch.from_numpy(both))[0].long(),
                "r": prior, "ell": ell, "k": k, "n": n,
                "note": f"r top-{k} ∩ ell top-{k} (seed1 基线收敛 ℓ)"},
               os.path.join(MASK_DIR, f"{dataset}_trusted_hard.pt"))
    print(f"[mask] {dataset}: n={n}, top-{k}∩top-{k} -> 冻结 {int(both.sum())} 个: "
          f"{np.where(both)[0].tolist()[:20]}{'...' if both.sum() > 20 else ''}")


def cmd_intervene(dataset, seed, mask_file, tag_kind):
    """干预 run: 掩码内 δ 恒 0; 结果/权重日志按 {ds}_{tag}_seed{k} 命名, 与正式产物零冲突"""
    tag = run_tag(dataset, seed, tag_kind)
    snap_dir = os.path.join(RUNS_DIR, tag, "snapshots")
    os.environ["DIAG_SNAP_DIR"] = snap_dir
    sh([PY, "-I", "e1_trusted_hard_intervention.py", "_train",
        ] + base_argv(dataset, seed)
       + ["--run_tag", tag_kind, "--freeze_delta_file", mask_file],
       os.path.join(E1_DIR, "driver_log.txt"))


def cmd_train(argv_rest):
    """(内部) 干预训练: 与诊断脚本同一封装, 但走 --run_tag/--freeze_delta_file"""
    from args import get_args
    sys.argv = [sys.argv[0]] + argv_rest
    args = get_args()
    args.save_model = False
    args.wandb = False
    args.model_selection = "final"
    args.CBM_type = "clip_cbm"
    args.weight_mode = "prior_residual"
    if not getattr(args, "freeze_delta_file", ""):
        raise ValueError("intervene 必须提供 --freeze_delta_file")

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from weight_loss_coupling_diagnosis import DiagSession
    session = DiagSession(args).run()
    print(f"[intervene done] run_name={session.run_name}")


def cmd_verify():
    """核验干预是否真的生效: 干预 run 的最终 δ 在掩码内必须为 0"""
    ok = True
    for jf in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_freeze*_seed*.json"))
                     + glob.glob(os.path.join(RESULTS_DIR, "*_rand68_seed*.json"))):
        base = os.path.splitext(os.path.basename(jf))[0]      # e.g. CUB_freeze_th_seed1
        ds = base.split("_")[0]
        mask_file = os.path.join(MASK_DIR, f"{ds}_trusted_hard.pt") \
            if "freeze_th" in base else os.path.join(MASK_DIR, f"{ds}_rand68.pt")
        if not os.path.exists(mask_file):
            continue
        idx = torch.load(mask_file, map_location="cpu", weights_only=False)["indices"]
        wl = os.path.join(WEIGHTS_DIR, base, "epoch_0099.pt")
        if not os.path.exists(wl):
            print(f"[verify] {base}: 缺权重日志 (跳过)")
            continue
        res = torch.load(wl, map_location="cpu", weights_only=False)["residual"]
        frozen_max = float(res[idx].abs().max()) if len(idx) else 0.0
        free_mean = float(res[torch.ones(len(res), dtype=torch.bool)].mean()) if len(idx) < len(res) else 0.0
        good = frozen_max == 0.0
        ok = ok and good
        print(f"[verify] {base}: 冻结 {len(idx)} 个, max|δ_frozen|={frozen_max:.2e} "
              f"{'OK' if good else '!! 未生效 !!'}  未冻结δ均值={free_mean:+.4f}")
    print("[verify]", "ALL OK" if ok else "存在未生效的冻结!")


def cmd_report():
    """配对比较 + 报告"""
    rows = []
    for ds, bs in DATASET_BATCH.items():
        for kind in ["freeze_th", "rand68"]:
            trip = []
            for k in SEEDS:
                bj, ij = baseline_json(ds, k), os.path.join(
                    RESULTS_DIR, run_tag(ds, k, kind) + ".json")
                if not (os.path.exists(bj) and os.path.exists(ij)):
                    trip = None
                    break
                b = json.load(open(bj, encoding="utf-8"))["sel_target_acc"]
                i = json.load(open(ij, encoding="utf-8"))["sel_target_acc"]
                trip.append((b, i))
            if trip:
                diffs = [i - b for b, i in trip]
                rows.append((ds, kind, trip, diffs))
    lines = ["# E1 可信-困难象限干预 — 结果汇总 (严格 DG final, sel_target_acc)\n",
             "| 数据集 | 干预 | seed1 基/干 | seed2 基/干 | seed3 基/干 | uplift 均值 | 方向 |",
             "|---|---|---|---|---|---|---|"]
    for ds, kind, trip, diffs in rows:
        cells = " | ".join(f"{b:.2f}/{i:.2f}" for b, i in trip)
        pos = sum(d > 0 for d in diffs)
        direction = f"{pos}/3 正" + ("  ✅" if pos >= 2 else "")
        lines.append(f"| {ds} | {kind} | {cells} | {np.mean(diffs):+.3f} | {direction} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(E1_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # 预注册判据核对 (仅对 freeze_th)
    cub = [r for r in rows if r[0] == "CUB" and r[1] == "freeze_th"]
    if cub:
        diffs = cub[0][3]
        pos = sum(d > 0 for d in diffs)
        verdict = "✅ 判据满足: 逃逸伤害泛化成立" if (np.mean(diffs) > 0 and pos >= 2) \
            else "❌ 判据不满足: 冻结可信-困难象限未带来提升 (逃逸在该协议下未见泛化伤害)"
        print("\n[预注册判据 CUB]", verdict)


def cmd_all():
    """全流程 (断点续跑)"""
    # 1) 基线
    for ds in DATASET_BATCH:
        for k in SEEDS:
            if os.path.exists(baseline_json(ds, k)):
                print(f"[skip] baseline {ds} s{k} (json 已存在)")
                continue
            print(f"[run ] baseline {ds} s{k}")
            cmd_baseline(ds, k)
    # 2) 掩码
    for ds in DATASET_BATCH:
        mf = os.path.join(MASK_DIR, f"{ds}_trusted_hard.pt")
        if not os.path.exists(mf):
            cmd_mask(ds)
    # CUB 随机对照掩码
    rand_f = os.path.join(MASK_DIR, "CUB_rand68.pt")
    if not os.path.exists(rand_f):
        g = torch.Generator().manual_seed(RAND_CTRL_SEED)
        idx = torch.randperm(204, generator=g)[:math.ceil(204 / FRAC)].long()
        torch.save({"indices": idx, "note": "随机 68 个 (seed97), 容量对照"},
                   rand_f)
    # 3) 干预
    for ds in DATASET_BATCH:
        for k in SEEDS:
            jf = os.path.join(RESULTS_DIR, run_tag(ds, k, "freeze_th") + ".json")
            if os.path.exists(jf):
                print(f"[skip] intervene {ds} s{k}")
                continue
            print(f"[run ] intervene {ds} s{k}")
            cmd_intervene(ds, k, os.path.join(MASK_DIR, f"{ds}_trusted_hard.pt"), "freeze_th")
    # 4) CUB 随机对照
    for k in SEEDS:
        jf = os.path.join(RESULTS_DIR, run_tag("CUB", k, "rand68") + ".json")
        if os.path.exists(jf):
            print(f"[skip] rand68 s{k}")
            continue
        print(f"[run ] rand68 CUB s{k}")
        cmd_intervene("CUB", k, rand_f, "rand68")
    # 5) 核验 + 报告
    cmd_verify()
    cmd_report()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    os.makedirs(E1_DIR, exist_ok=True)
    if cmd == "all":
        cmd_all()
    elif cmd == "baseline":
        a = argparse.ArgumentParser(); a.add_argument("--dataset"); a.add_argument("--seed", type=int)
        v = a.parse_args(sys.argv[2:]); cmd_baseline(v.dataset, v.seed)
    elif cmd == "mask":
        a = argparse.ArgumentParser(); a.add_argument("--dataset")
        v = a.parse_args(sys.argv[2:]); cmd_mask(v.dataset)
    elif cmd == "intervene":
        a = argparse.ArgumentParser()
        a.add_argument("--dataset"); a.add_argument("--seed", type=int)
        a.add_argument("--mask"); a.add_argument("--tag", default="freeze_th")
        v = a.parse_args(sys.argv[2:]); cmd_intervene(v.dataset, v.seed, v.mask, v.tag)
    elif cmd == "_train":
        cmd_train(sys.argv[2:])
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "report":
        cmd_report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
