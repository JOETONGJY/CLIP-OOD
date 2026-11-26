"""实验矩阵驱动 (2026-11-25): 顺序跑一组配置。

  - 以结果 json 的完成状态判定跳过 (status / 完整 epoch / 命令行里出现的每一个配置项), 不只看文件是否存在
  - 成功 = 子进程退出码为 0 且结果完整且配置匹配, 三者缺一即失败
  - 只在显式 --overwrite_run 时才把覆盖开关传给训练入口; 否则同名的不完整 / 不匹配结果会让训练入口
    拒绝启动, 驱动把它记为失败 (不静默覆盖)
  - 矩阵级运行锁 logs/locks/matrix_<name>.lock: 同一个矩阵不允许两个驱动同时跑 (训练入口另有逐 run 的锁)
  - 每个训练的 stdout 写到 experiment_results/<目录>/driver_logs/ (已加入 .gitignore, 不入库)
  - 公共配置与 2026-11-25 之后的全部新实验保持一致: 严格 final 协议 (目标域不参与训练与选模, 仅最终评估一次;
    main_cached 入口在该模式下把目标域特征缓存延迟到最终评估时读入), 固定 epoch 梯度诊断, 不保存模型

用法 (只想看进度时必须加 --dry_run; 2026-11-25 漏加过一次, 同一组 run 被两个进程同时跑):
  python run_matrix.py dose      [--seeds 1 2 3] [--dry_run] [--overwrite_run]
  python run_matrix.py subspace  ...
  python run_matrix.py p1        ...          (P1 以 --target_eval none 训练, 见 RESULTS §3v)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
sys.path.insert(0, ROOT)
from run_status import RunLock, result_is_complete  # noqa: E402

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

# ---- DDO-Subspace 正式矩阵 (2026-11-25, 外审确认后的版本) ----
# 各臂的剂量: k=32 的臂初始化正则强度约为完整方向集的 0.44 倍, 需要更高的 alpha 才能覆盖同样的强度范围
SUB_DOSES = {"mean": (0.5, 1.0, 2.0, 4.0), "pca8": (0.5, 1.0, 2.0, 4.0), "pca32": (1.0, 2.0, 4.0, 8.0),
             "rand8": (0.5, 1.0, 2.0, 4.0), "rand32": (1.0, 2.0, 4.0, 8.0), "fullsym": (1.0, 2.0)}
# 外审第三轮 (2026-11-25) 后追加的低剂量 (描述性, 不参与 §3n 的判定配对, 见 RESULTS §3p):
# pca 两臂都在各自最低非零剂量处最好, 不能在剂量覆盖不足时断言任何非零剂量都有害
SUB_POST_HOC = {"pca8": (0.125, 0.25), "pca32": (0.125, 0.25)}
SUB_DIRS = {"mean": ["--ddo_dirs", "mean"],
            "pca8": ["--ddo_dirs", "pca", "--ddo_pca_src", "ln0", "--ddo_k", "8"],
            "pca32": ["--ddo_dirs", "pca", "--ddo_pca_src", "ln0", "--ddo_k", "32"],
            # full_sym: 完整方向集 + 相反数。pca / random 臂相对完整方向集同时改了方向空间、方向分布、
            # 符号分布三个因素, 这一臂只改符号分布
            "fullsym": ["--ddo_dirs", "full_sym"]}
ATAG = {0.125: "0p125", 0.25: "0p25", 0.5: "0p5", 1.0: "1", 2.0: "2", 4.0: "4", 8.0: "8"}

# P1: 划分 (train_core / heldout_val, seed 42) 与验证用的增强特征都在这个缓存里 (cache/ 被 gitignore, 不入库)
AUG_CACHE = os.path.join("cache", "CUB_train_aug.pt")


def _random_dirs(k):
    # 随机标架的种子与训练 seed 配对 (seed 1/2/3 各用一个标架), 同一标架在不同剂量之间保持不变:
    # 这样每个训练 seed 都有配对的随机对照, 同时覆盖三个不同的随机子空间, 而不是只验证一个
    return lambda seed: ["--ddo_dirs", "random", "--ddo_k", str(k), "--ddo_dirs_seed", str(seed)]


def _subspace_entries():
    out = []
    for arm in ("mean", "pca8", "pca32", "rand8", "rand32", "fullsym"):
        dirs = _random_dirs(8 if arm == "rand8" else 32) if arm.startswith("rand") else SUB_DIRS[arm]
        for a in SUB_DOSES[arm]:
            base = EQUAL + ["--alpha", str(a)]
            out.append(("CUB", f"sub_{arm}_a{ATAG[a]}", (lambda s, b=base, d=dirs: b + d(s)) if callable(dirs) else base + dirs))
    for arm, doses in SUB_POST_HOC.items():
        for a in doses:
            out.append(("CUB", f"sub_{arm}_a{ATAG[a]}", EQUAL + ["--alpha", str(a)] + SUB_DIRS[arm]))
    return out


# ---- 机制对照矩阵 (2026-11-25, 外审第三轮; 设计与判定规则预注册于 RESULTS §3p) ----
# projected / residual / split: 分离子空间截断与方向分布重构 (GPT 的分析: PCA 臂的失败可能不是保留了
# 错误的子空间, 而是用等权正交标架替换了原始经验方向分布)。分解在 LN0 后、affine 前, 不重新逐行归一化
MECH_DIRS = {"proj8": ["--ddo_dirs", "projected", "--ddo_k", "8"],
             "proj32": ["--ddo_dirs", "projected", "--ddo_k", "32"],
             "res8": ["--ddo_dirs", "residual", "--ddo_k", "8"],
             "res32": ["--ddo_dirs", "residual", "--ddo_k", "32"],
             "split8": ["--ddo_dirs", "split", "--ddo_k", "8"],
             "split32": ["--ddo_dirs", "split", "--ddo_k", "32"]}
MECH_DOSES = {arm: ((0.5, 1.0, 2.0, 4.0) if arm.endswith("8") else (1.0, 2.0, 4.0, 8.0)) for arm in MECH_DIRS}


# 目录 -> [(dataset, run_tag, 额外参数 或 由 seed 生成额外参数的函数)]
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
    # DDO-Subspace: 5 个子空间臂 x 4 个剂量 + 完整方向集对称化对照 x 2 个剂量 + 外审后追加的低剂量
    # (CUB x 3 seeds = 78 个 run; 其中 66 个为 §3n 预注册集合, 12 个低剂量为 §3p 的描述性补充)。
    # full 的剂量曲线已在 dose 矩阵里。判定口径见 RESULTS.md §3n / §3p
    "subspace": ("DDO-Subspace", _subspace_entries()),
    # 机制对照: projected / residual / split x k in {8,32} x 4 个剂量, CUB x 3 seeds = 72 个 run
    "mechanism": ("DDO-Subspace", [
        ("CUB", f"mech_{arm}_a{ATAG[a]}", EQUAL + ["--alpha", str(a)] + MECH_DIRS[arm])
        for arm in ("proj8", "res8", "split8", "proj32", "res32", "split32") for a in MECH_DOSES[arm]]),
    # Stage 0 机制诊断用的收敛分类头 (主线 B): 只产出模型, 不做任何判定。CE-Only、Full DDO、Mean、Full-Sym、
    # Bias-Free 各在预先确定的剂量下重训并 --save_head (正式矩阵都是 --no_save_model, 没有保存分类头)。
    # 配置与正式矩阵一致 (同 batch / epochs / 协议), 所以精度应与已有的正式结果逐位相同 —— 这本身就是一项核对。
    "stage0": ("DDO-Subspace", [
        ("CUB", "s0_ce_a0", EQUAL + ["--alpha", "0.0", "--save_head"]),
        ("CUB", "s0_full_a1", EQUAL + ["--alpha", "1.0", "--save_head"]),
        ("CUB", "s0_full_a2", EQUAL + ["--alpha", "2.0", "--save_head"]),
        ("CUB", "s0_mean_a1", EQUAL + ["--alpha", "1.0", "--ddo_dirs", "mean", "--save_head"]),
        ("CUB", "s0_mean_a2", EQUAL + ["--alpha", "2.0", "--ddo_dirs", "mean", "--save_head"]),
        ("CUB", "s0_sym_a1", EQUAL + ["--alpha", "1.0", "--ddo_dirs", "full_sym", "--save_head"]),
        ("CUB", "s0_sym_a2", EQUAL + ["--alpha", "2.0", "--ddo_dirs", "full_sym", "--save_head"]),
        ("CUB", "s0_bf_a1", EQUAL + ["--alpha", "1.0", "--ddo_dirs", "full_bf", "--save_head"]),
        ("CUB", "s0_bf_a2", EQUAL + ["--alpha", "2.0", "--ddo_dirs", "full_bf", "--save_head"]),
    ]),
    # Split 的偏置对照 (§3q): bias-free 的 split 对 bias-free 的 full, k in {8,32} x 4 个剂量 x 3 seeds = 48 个 run
    "biasfree": ("DDO-Subspace", [
        ("CUB", f"bf_{arm}_a{ATAG[a]}", EQUAL + ["--alpha", str(a), "--ddo_dirs", dirs, "--ddo_k", str(k)])
        for arm, dirs, k in (("full8", "full_bf", 8), ("split8", "split_bf", 8),
                             ("full32", "full_bf", 32), ("split32", "split_bf", 32))
        for a in ((0.5, 1.0, 2.0, 4.0) if k == 8 else (1.0, 2.0, 4.0, 8.0))]),
    # ---- P1: Stage 0D 独立验证 (2026-11-26, 外审第九轮; 预注册 RESULTS §3v) ----
    # 与 dose 矩阵同样的 7 个剂量 x 3 seeds, 配置相同, 另加: 只用 train_core (4790) 训练; 训练结束后在
    # heldout_val (1200) 上评估干净 + 7 种扰动; --target_eval none (训练期间完全不评估目标域 —— 源域选择
    # 结果提交锁定之后, 才用 --save_head 保存的分类头做事后评估, 见 eval_heads_targets.py)
    "p1": ("Stage0D-Independent", [
        ("CUB", f"p1_{tag}", EQUAL + ["--alpha", str(a), "--target_eval", "none", "--save_head",
                                     "--train_subset_cache", AUG_CACHE, "--source_val_cache", AUG_CACHE])
        for a, tag in DOSE]),
}

# 不带取值的开关: store_true -> True, store_false -> 其 dest 为 False
# 新增任何 store_true/store_false 参数都必须在这里登记, 否则 expected() 会把它当成"缺值的参数"而报错 (导入时自检)
STORE_TRUE = {"--grad_diag": "grad_diag", "--overwrite_run": "overwrite_run", "--grouped_weights": "grouped_weights",
              "--save_head": "save_head"}
STORE_FALSE = {"--no_save_model": "save_model"}


def expected(extra, ds=None, seed=None, tag=None):
    """完成判定用的配置: 命令行里出现的每一个参数 (同一参数出现多次时取最后一个, 与 argparse 一致),
    外加 dataset / seed / run_tag。由 run_status.result_is_complete 与结果 json 的 run_config 逐项比对
    (数值按数值比较)。--overwrite_run 由驱动控制、与训练结果无关, 不参与比对。"""
    out, toks = {}, list(extra)
    for i, t in enumerate(toks):
        if t in STORE_TRUE:
            out[STORE_TRUE[t]] = True
        elif t in STORE_FALSE:
            out[STORE_FALSE[t]] = False
        elif t.startswith("--"):
            assert i + 1 < len(toks) and not toks[i + 1].startswith("--"), f"参数 {t} 缺少取值"
            out[t[2:]] = toks[i + 1]
    out.pop("overwrite_run", None)
    if ds is not None:
        out.update(dataset=ds, seed=seed, run_tag=tag)
    return out


def _self_test_expected():
    """导入时的自检。2026-11-25 weights_dose 矩阵曾因 alpha 出现两次而被误判为失败;
    同日外审指出 expected() 只核验少数字段、成功判定不看退出码、epoch 判据对 0/None 不设防。"""
    e = expected(CH3 + A1 + ["--weight_mode", "prior_residual", "--alpha", "2.0"] + COMMON, "CUB", 2, "w_rwddo_a2")
    assert e["alpha"] == "2.0" and e["weight_mode"] == "prior_residual" and e["epochs"] == "100", e
    assert (e["grad_diag"] is True and e["save_model"] is False and e["target_eval"] == "final_only"
            and e["model_selection"] == "final" and e["lr"] == "0.0001" and "overwrite_run" not in e), e
    assert (e["dataset"], e["seed"], e["run_tag"]) == ("CUB", 2, "w_rwddo_a2"), e
    e = expected(EQUAL + ["--alpha", "0.125"] + COMMON)
    assert e["alpha"] == "0.125" and e["weight_mode"] == "none" and e["lambda_kl"] == "0", e
    for name, (out_dir, entries) in MATRICES.items():
        for ds, tag, extra in entries:
            ex = extra(1) if callable(extra) else extra
            assert "alpha" in expected(ex + COMMON), f"矩阵 {name} 的 {tag} 没有 alpha"
    assert len(_subspace_entries()) == 26, len(_subspace_entries())        # 22 预注册 + 4 低剂量配置
    assert len(MATRICES["mechanism"][1]) == 24, len(MATRICES["mechanism"][1])
    assert len(MATRICES["biasfree"][1]) == 16, len(MATRICES["biasfree"][1])       # 4 臂 x 4 剂量
    assert len(MATRICES["stage0"][1]) == 9, len(MATRICES["stage0"][1])
    # P1: 完成判定与实际命令行是同一份 (COMMON + extra)。P1 在 COMMON 之后又给了 --target_eval none, argparse 与
    # expected() 都取最后一个 -> none。(2026-11-26 自检抓到: 此前判定用的是 extra + COMMON, 会拿 final_only 去比对
    # 实际按 none 训练的结果, 21 个 run 全被误判为失败。)
    assert len(MATRICES["p1"][1]) == 7, len(MATRICES["p1"][1])
    e = expected(COMMON + MATRICES["p1"][1][0][2], "CUB", 1, "p1_dose_a0")
    assert (e["target_eval"] == "none" and e["save_head"] is True and e["alpha"] == "0.0" and e["weight_mode"] == "none"
            and e["train_subset_cache"] == AUG_CACHE and e["source_val_cache"] == AUG_CACHE and e["epochs"] == "100"), e
    assert expected(COMMON + MATRICES["p1"][1][4][2])["alpha"] == "1.0"
    # 完成判定: 配置按数值比对; epochs 必须是正整数; 任一项不匹配都不算完成
    base = {"status": "completed", "completed_epochs": 2, "run_config": {
        "epochs": 2, "lambda_kl": 0.0, "alpha": 1.0, "grad_diag": True, "save_model": False, "ddo_k": 8}}
    want = expected(["--epochs", "2", "--lambda_kl", "0", "--alpha", "1.0", "--grad_diag", "--no_save_model",
                     "--ddo_k", "8"])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.json")

        def verdict(res):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(res, f)
            return result_is_complete(p, want)

        assert verdict(base) is True                                  # 命令行的 "0" 与 json 的 0.0 视为相同
        assert verdict({**base, "completed_epochs": 1}) is False
        assert verdict({**base, "status": "incomplete"}) is False
        assert verdict({**base, "run_config": {**base["run_config"], "ddo_k": 32}}) is False
        assert verdict({"status": "completed", "completed_epochs": 0, "run_config": {"epochs": 0}}) is False
        assert verdict({"status": "completed", "run_config": {}}) is False      # None == None 不算完成


_self_test_expected()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix", choices=sorted(MATRICES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dry_run", action="store_true", help="只打印每个 run 的配置与完成状态, 不训练、不加锁")
    ap.add_argument("--overwrite_run", action="store_true",
                    help="全部重跑: 不跳过已完成的 run, 并把覆盖开关传给训练入口")
    opt = ap.parse_args()

    out_dir, entries = MATRICES[opt.matrix]
    jobs = []
    for ds, tag, extra in entries:
        for seed in opt.seeds:
            jobs.append((ds, tag, extra(seed) if callable(extra) else extra, seed))

    def path_of(ds, tag, seed):
        return os.path.join(ROOT, "logs", "results", f"{ds}_{tag}_seed{seed}.json")

    print(f"[matrix {opt.matrix}] {len(entries)} 个配置 x {len(opt.seeds)} seeds = {len(jobs)} runs", flush=True)
    if opt.dry_run:
        n_done = 0
        for ds, tag, extra, seed in jobs:
            done = result_is_complete(path_of(ds, tag, seed), expected(COMMON + extra, ds, seed, tag))
            n_done += done
            print(f"  {'[完成] ' if done else '[待跑] '}{ds}_{tag}_seed{seed}: {' '.join(extra)}")
        print(f"已完成 {n_done} / {len(jobs)}", flush=True)
        return

    log_dir = os.path.join(ROOT, "experiment_results", out_dir, "driver_logs")
    os.makedirs(log_dir, exist_ok=True)
    lock = RunLock(os.path.join(ROOT, "logs", "locks", f"matrix_{opt.matrix}.lock"), what=f"实验矩阵 {opt.matrix}")
    try:
        with lock:
            _drive(jobs, opt, log_dir, path_of, lock)
    except RuntimeError as e:
        print(f"{e}\n(只看进度请用 --dry_run)", flush=True)
        sys.exit(2)


def _drive(jobs, opt, log_dir, path_of, lock):
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    done = skipped = failed = 0
    failures = []
    for i, (ds, tag, extra, seed) in enumerate(jobs):
        lock.heartbeat()
        run_name = f"{ds}_{tag}_seed{seed}"
        path = path_of(ds, tag, seed)
        argv = COMMON + extra                        # 实际执行的命令行; 完成判定必须用同一份 (见下方 expected)
        exp = expected(argv, ds, seed, tag)
        if result_is_complete(path, exp) and not opt.overwrite_run:
            skipped += 1
            print(f"[skip {i + 1}/{len(jobs)}] {run_name}", flush=True)
            continue
        cmd = [PY, "-u", "main_cached.py", "--dataset", ds, "--seed", str(seed), "--run_tag", tag]
        cmd += argv + (["--overwrite_run"] if opt.overwrite_run else [])
        log = os.path.join(log_dir, run_name + ".log")
        print(f"[run  {i + 1}/{len(jobs)}] {run_name}  {time.strftime('%H:%M:%S')}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        # 成功必须同时满足: 退出码为 0, 且结果完整并与命令行配置逐项匹配
        ok = r.returncode == 0 and result_is_complete(path, exp)
        if ok:
            done += 1
        else:
            failed += 1
            failures.append(run_name)
        why = "" if ok else (f"exit={r.returncode}" if r.returncode != 0
                             else "exit=0, 但结果不完整或与命令行配置不匹配")
        print(f"       -> {'OK' if ok else 'FAIL ' + why}  (日志 {os.path.relpath(log, ROOT)})", flush=True)

    print(f"\n新完成 {done} | 跳过 {skipped} | 失败 {failed} | 耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    if failures:
        print("失败: " + ", ".join(failures), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
