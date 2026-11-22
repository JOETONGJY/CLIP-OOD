"""
E3 方案 B: 簇级分组去冗余 (grouped RW-DDO) + 注入扰动实验 (2026-11-21, 预注册)

研究问题 (承 E1/E2 结论): RW-DDO 的 weight avoidance 无泛化伤害 (E1), 但 C-1 去冗余
的"相对冗余度"修正是 softmax 下的数学空操作 (07 诊断), 描述符集合冗余问题仍未解决。
方案 B 用近重复聚簇实现严格复制不变: 簇总话语权与成员数无关。

预注册判据 (跑之前写死):
  ① 干净场景: grouped vs 原 RW-DDO, 4 数据集 uplift 均值 |Δ| ≤ 0.3 (不伤;
     机制上全单元素簇时严格退化, 预期≈0);
  ② 注入场景 (CUB, 注入 25 个 cos>0.95 近似重复): grouped − 原 ≥ +0.3 且 ≥2/3
     seeds 正 → 分组机制在污染下有效;
  ③ 离线份额: 注入方向权重份额膨胀, 原 ≈×2.0 (C-2 已测), grouped 预期 ≈×1.0~1.2。

用法:
  python e3_grouped_injection_experiment.py all          # 全流程 (断点续跑)
  python e3_grouped_injection_experiment.py gen_inject   # 生成注入列表 + CLIP cos 验证
  python e3_grouped_injection_experiment.py stats        # 各数据集干净聚簇统计
  python e3_grouped_injection_experiment.py offline_share
  python e3_grouped_injection_experiment.py train --dataset CUB --seed 1 --kind grp
  python e3_grouped_injection_experiment.py report
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import subprocess

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
E3_DIR = os.path.join("experiment_results", "E3_分组去冗余")
INJECT_JSON = os.path.join(E3_DIR, "inject_229.json")
SHARE_JSON = os.path.join(E3_DIR, "offline_share.json")
RESULTS_DIR = os.path.join("logs", "results")

DATASET_BATCH = {"CUB": 256, "AWA2": 256, "LADV": 256, "PACS": 128}
SEEDS = [1, 2, 3]
CLIP_PATH = os.path.join(HERE, "clip_weights", "ViT-L-14.pt")
BASE_START, BASE_STEP, N_INJECT = 4, 8, 25     # 预注册: 均匀取 25 个基描述符
COS_MIN = 0.95                                  # 预注册: 变体与原型 cos 下限


# ---------------------------------------------------------------- 工具
def sh(args_list, log_path):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(args_list) + "\n")
        f.flush()
        ret = subprocess.call(args_list, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    if ret != 0:
        raise RuntimeError(f"子进程失败 (exit={ret}), 见 {log_path}")


def json_name(dataset, seed, kind):
    return os.path.join(RESULTS_DIR, f"{dataset}_{kind}_seed{seed}.json")


def e1_baseline(dataset, seed):
    return os.path.join(RESULTS_DIR, f"{dataset}_prior_residual_sel_seed{seed}.json")


def make_variant(p):
    """预注册规则: 首字母大写。CLIP BPE 分词器小写化文本 => 变体与原型 token 级完全相同,
    即精确重复注入 (无变体语义漂移混淆; cos 实测 = 1.0000, 必然被 theta=0.9 聚簇合并)。
    ('print' 插入规则实测仅 3/25 过 cos>0.95, 已废弃)"""
    return p[0].upper() + p[1:]


# ---------------------------------------------------------------- 子命令
def cmd_gen_inject():
    from prompts.prompt200new import target_text_prompts
    base_idx = list(range(BASE_START, len(target_text_prompts), BASE_STEP))[:N_INJECT]
    variants = [make_variant(target_text_prompts[i]) for i in base_idx]
    prompts = list(target_text_prompts) + variants
    # CLIP cos 验证
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(CLIP_PATH, device=device)
    texts = [target_text_prompts[i] for i in base_idx] + variants
    toks = clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = model.encode_text(toks).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    cos = (emb[:N_INJECT] * emb[N_INJECT:]).sum(-1).cpu().numpy()
    print(f"[gen_inject] cos(orig, variant): min={cos.min():.4f} mean={cos.mean():.4f}")
    assert cos.min() > COS_MIN, f"存在变体 cos<{COS_MIN}: {cos.min():.4f}"
    os.makedirs(E3_DIR, exist_ok=True)
    with open(INJECT_JSON, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts, "base_indices": base_idx,
                   "cosines": [round(float(c), 4) for c in cos],
                   "rule": "插入 'print' (与 CUB 已知近重复对同款手法)", "cos_min": float(cos.min())},
                  f, ensure_ascii=False, indent=1)
    print(f"[gen_inject] {len(prompts)} prompts -> {INJECT_JSON}")


def cmd_stats():
    """各数据集干净 204 描述符在 theta=0.9 下的聚簇统计 (离线, 零训练)"""
    from args import get_args
    from reliability import compute_descriptor_clusters
    from prompts.prompt200new import target_text_prompts
    out = {}
    for ds, bs in DATASET_BATCH.items():
        sys.argv = [sys.argv[0], "--dataset", ds, "--seed", "1", "--batch_size", str(bs),
                    "--CLIP_type", CLIP_PATH, "--epochs", "1"]
        args = get_args()
        args.save_model = False
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_dataset = __import__("data").get_dataset_classes(args)[0]
        a, g = compute_descriptor_clusters(train_dataset.domain_diffs, threshold=0.9)
        counts = torch.bincount(a)
        multi = counts[counts > 1]
        out[ds] = {"n_groups": int(g), "multi_clusters": int(len(multi)),
                   "largest": int(counts.max()), "in_multi": int(multi.sum().item())}
        print(f"[stats] {ds}: {len(target_text_prompts)} -> {g} 簇 (多成员簇 {len(multi)} 个, "
              f"最大 {counts.max().item()}, 涉及 {multi.sum().item()} 个描述符)")
    with open(os.path.join(E3_DIR, "cluster_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def _build_dataset(args):
    return __import__("data").get_dataset_classes(args)[0]


def cmd_offline_share():
    """注入方向权重份额膨胀: 原公式 vs C-1(topk5) vs grouped (零训练, CUB)"""
    from args import get_args
    from reliability import (compute_domain_diffs_and_scores,
                             compute_descriptor_redundancy, compute_descriptor_clusters)
    from prompts.prompt200new import source_text_prompts
    import clip as clip_mod

    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", CLIP_PATH, "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = _build_dataset(args)
    class_names = list(ds.classname2id.keys())
    clean_diffs = ds.domain_diffs            # (204, C, D) 缓存命中
    clean_scores = ds.reliability_scores
    inj = json.load(open(INJECT_JSON, encoding="utf-8"))
    inj_prompts, base_idx = inj["prompts"], inj["base_indices"]

    device = args.device
    model, _ = clip_mod.load(CLIP_PATH, device=device)
    print("[offline] 计算注入后 229 prompts 的 diffs (首次 ~1-2 分钟)...")
    inj_diffs, inj_scores = compute_domain_diffs_and_scores(
        model, list(source_text_prompts), inj_prompts, class_names, device, verbose=False)

    n_clean, n_inj = len(clean_scores), len(inj_scores)
    base_t = torch.tensor(base_idx, device=device)
    var_t = torch.arange(n_clean, n_inj, device=device)

    def w_orig(scores):
        return torch.softmax(scores, dim=0) * len(scores)

    def w_c1(scores, diffs, topk=5):
        c = compute_descriptor_redundancy(diffs, topk=topk, relative=True)
        return torch.softmax(torch.log(scores.clamp_min(1e-8)) - torch.log(c.clamp_min(1e-8)), 0) * len(scores)

    def w_grouped(scores, diffs, theta=0.9):
        a, g = compute_descriptor_clusters(diffs, threshold=theta)
        a = a.to(device)
        counts = torch.bincount(a, minlength=g)
        cp = torch.zeros(g, device=device)
        cp.scatter_add_(0, a, scores)
        cp = cp / counts.clamp_min(1).to(device).float()
        Q = torch.softmax(cp, 0) * g
        e = torch.exp(scores - scores.max())      # 全局 shift, 数值等价
        den = torch.zeros(g, device=device)
        den.scatter_add_(0, a, e)
        q = e / den[a]
        w = Q[a] * q
        return w * (len(scores) / w.sum())

    out = {}
    for name, wf in [("orig", lambda s, d: w_orig(s)),
                     ("c1_topk5", lambda s, d: w_c1(s, d)),
                     ("grouped", lambda s, d: w_grouped(s, d))]:
        w_clean = wf(clean_scores, clean_diffs)
        w_inj = wf(inj_scores, inj_diffs)
        s_clean = float(w_clean[base_t].sum())
        s_inj = float(w_inj[torch.cat([base_t, var_t])].sum())
        out[name] = {"clean_base25_share": round(s_clean, 4),
                     "inj_50_share": round(s_inj, 4),
                     "inflation": round(s_inj / s_clean, 3)}
        print(f"[offline] {name:10s} 干净基25份额={s_clean:.4f}  注入后50份额={s_inj:.4f}  "
              f"膨胀 x{s_inj / s_clean:.3f}")
    with open(SHARE_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def cmd_train(argv_rest):
    from args import get_args
    sys.argv = [sys.argv[0]] + argv_rest
    args = get_args()
    args.save_model = False
    args.wandb = False
    args.model_selection = "final"
    args.CBM_type = "clip_cbm"
    args.weight_mode = "prior_residual"
    if getattr(args, "inject_file", ""):
        inj = json.load(open(args.inject_file, encoding="utf-8"))
        prompts = list(inj["prompts"])
        import data as data_mod
        import prompts.prompt200new as pmod
        data_mod.target_text_prompts = prompts
        pmod.target_text_prompts = prompts
        print(f"[inject] 描述符集 204 -> {len(prompts)} (patch data/prompt 模块, reliability 缓存自动重算)")

    import random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from weight_loss_coupling_diagnosis import DiagSession
    session = DiagSession(args).run()
    print(f"[train done] run_name={session.run_name}")


def cmd_all():
    import argparse as _ap
    # 1) 生成注入列表 + 统计 + 离线份额
    if not os.path.exists(INJECT_JSON):
        cmd_gen_inject()
    cmd_stats()
    if not os.path.exists(SHARE_JSON):
        cmd_offline_share()
    # 2) 干净 grouped: 4 数据集 x 3 seeds
    for ds, bs in DATASET_BATCH.items():
        for k in SEEDS:
            if os.path.exists(json_name(ds, k, "grp")):
                print(f"[skip] grp {ds} s{k}")
                continue
            print(f"[run ] grp {ds} s{k}")
            sh([PY, "-I", "e3_grouped_injection_experiment.py", "train",
                "--dataset", ds, "--seed", str(k), "--batch_size", str(bs),
                "--CLIP_type", CLIP_PATH, "--epochs", "100",
                "--run_tag", "grp", "--grouped_weights"],
               os.path.join(E3_DIR, "driver_log.txt"))
    # 3) 注入 CUB: 原方法 vs grouped
    for k in SEEDS:
        if not os.path.exists(json_name("CUB", k, "rwddo_inj")):
            print(f"[run ] rwddo_inj CUB s{k}")
            sh([PY, "-I", "e3_grouped_injection_experiment.py", "train",
                "--dataset", "CUB", "--seed", str(k), "--batch_size", "256",
                "--CLIP_type", CLIP_PATH, "--epochs", "100",
                "--run_tag", "rwddo_inj", "--inject_file", INJECT_JSON],
               os.path.join(E3_DIR, "driver_log.txt"))
    for k in SEEDS:
        if not os.path.exists(json_name("CUB", k, "grp_inj")):
            print(f"[run ] grp_inj CUB s{k}")
            sh([PY, "-I", "e3_grouped_injection_experiment.py", "train",
                "--dataset", "CUB", "--seed", str(k), "--batch_size", "256",
                "--CLIP_type", CLIP_PATH, "--epochs", "100",
                "--run_tag", "grp_inj", "--grouped_weights",
                "--inject_file", INJECT_JSON],
               os.path.join(E3_DIR, "driver_log.txt"))
    cmd_report()


def cmd_report():
    rows = []
    # 干净场景: grouped vs E1 基线 (同 seed 同协议)
    for ds in DATASET_BATCH:
        trip = []
        for k in SEEDS:
            bj, gj = e1_baseline(ds, k), json_name(ds, k, "grp")
            if not (os.path.exists(bj) and os.path.exists(gj)):
                trip = None
                break
            b = json.load(open(bj, encoding="utf-8"))["sel_target_acc"]
            g = json.load(open(gj, encoding="utf-8"))["sel_target_acc"]
            trip.append((b, g))
        if trip:
            d = [g - b for b, g in trip]
            rows.append(("干净", ds, trip, d))
    # 注入场景
    trip = []
    for k in SEEDS:
        bj, gj = json_name("CUB", k, "rwddo_inj"), json_name("CUB", k, "grp_inj")
        if not (os.path.exists(bj) and os.path.exists(gj)):
            trip = None
            break
        b = json.load(open(bj, encoding="utf-8"))["sel_target_acc"]
        g = json.load(open(gj, encoding="utf-8"))["sel_target_acc"]
        trip.append((b, g))
    if trip:
        d = [g - b for b, g in trip]
        rows.append(("注入25重复", "CUB", trip, d))

    lines = ["# E3 grouped RW-DDO — 结果汇总 (严格 DG final, sel_target_acc)\n",
             "| 场景 | 数据集 | seed1 基/组 | seed2 基/组 | seed3 基/组 | uplift 均值 | 方向 |",
             "|---|---|---|---|---|---|---|"]
    for scene, ds, trip, d in rows:
        cells = " | ".join(f"{b:.2f}/{g:.2f}" for b, g in trip)
        pos = sum(x > 0 for x in d)
        lines.append(f"| {scene} | {ds} | {cells} | {np.mean(d):+.3f} | {pos}/3 |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(E3_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    os.makedirs(E3_DIR, exist_ok=True)
    if cmd == "all":
        cmd_all()
    elif cmd == "gen_inject":
        cmd_gen_inject()
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "offline_share":
        cmd_offline_share()
    elif cmd == "train":
        a = argparse.ArgumentParser()
        a.add_argument("--dataset"); a.add_argument("--seed", type=int)
        a.add_argument("--kind", default="grp")
        cmd_train(sys.argv[2:])
    elif cmd == "report":
        cmd_report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
