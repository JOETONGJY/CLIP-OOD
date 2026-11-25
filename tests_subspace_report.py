"""ddo_subspace_report.py 的自测 (预注册判定规则的单元级检查): 只用 21 个已知的 full 剂量结果 + 合成的子空间结果 (数值是人为编的),
检查预注册的四类判定是否按 RESULTS.md §3n 的规则给出标签、报告能否完整生成。
不读取任何真实的子空间结果。用法: python tests_subspace_report.py"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)                      # git 命令按仓库根目录解析
sys.path.insert(0, REPO)
import run_matrix as rm  # noqa: E402

SEEDS = (1, 2, 3)
ROWS = {"mean": 204, "pca8": 16, "pca32": 64, "rand8": 16, "rand32": 64, "fullsym": 81600,
        "proj8": 40800, "proj32": 40800, "res8": 40800, "res32": 40800, "split8": 81600, "split32": 81600}
DIRS = {"mean": "mean", "pca8": "pca", "pca32": "pca", "rand8": "random", "rand32": "random", "fullsym": "full_sym",
        "proj8": "projected", "proj32": "projected", "res8": "residual", "res32": "residual",
        "split8": "split", "split32": "split"}
K = {"pca8": 8, "pca32": 32, "rand8": 8, "rand32": 32, "proj8": 8, "proj32": 32, "res8": 8, "res32": 32,
     "split8": 8, "split32": 32}
MECH = ("proj8", "proj32", "res8", "res32", "split8", "split32")

# 合成数值 = full 在同一 alpha (没有则取最近的) 的目标域精度 + 人为偏移
OFFSET = {"pca8": (1.0, 1.0, 1.0), "rand8": (0.0, 0.0, 0.0),
          "pca32": (0.1, 0.8, 0.1), "rand32": (-0.2, -0.2, -0.2),
          "mean": (0.3, 0.3, -0.9), "fullsym": (0.1, 0.1, 0.1),       # 按 seed 1/2/3
          "proj8": (0.8, 0.8, 0.8), "proj32": (0.0, 0.0, 0.0),
          "res8": (0.0, 0.0, 0.0), "res32": (0.0, 0.0, 0.0),
          "split8": (0.0, 0.0, 0.0), "split32": (0.0, 0.0, 0.0)}
EXPECT = {  # (问题, 臂) -> 期望标签
    ("q1", "k=8"): "支持", ("q1", "k=32"): "支持",
    ("q2", "pca8"): "支持", ("q2", "pca32"): "不确定", ("q2", "mean"): "不支持",
    ("q3", ""): "未见影响",
    ("q4", "pca8"): "改善权衡", ("q4", "rand8"): "只改变工作点", ("q4", "mean"): "只改变工作点",
    ("q_mech", "proj8"): "支持",                                   # +0.8, 三个 seed 均为正
    ("q5", "k=8"): "不支持"}                                       # proj8 - pca8 = 0.8 - 1.0 = -0.2

real = os.path.join(REPO, "logs", "results")
full = {}
for a, tag in rm.DOSE:
    for s in SEEDS:
        full[(a, s)] = json.load(open(os.path.join(real, f"CUB_{tag}_seed{s}.json"), encoding="utf-8"))
full_a = [a for a, _ in rm.DOSE]
near = lambda a: min(full_a, key=lambda x: abs(x - a))
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, stdout=subprocess.PIPE).stdout.decode().strip()

tmp = tempfile.mkdtemp(prefix="subreport_selftest_")
res, outd = os.path.join(tmp, "results"), os.path.join(tmp, "out")
os.makedirs(res)
for a, tag in rm.DOSE:
    for s in SEEDS:
        shutil.copy(os.path.join(real, f"CUB_{tag}_seed{s}.json"), res)
sha = lambda key: hashlib.sha256(key.encode()).hexdigest()
for matrix in ("subspace", "mechanism"):
    for ds, tag, extra in rm.MATRICES[matrix][1]:
        arm = tag.split("_")[1]
        for s in SEEDS:
            ex = extra(s) if callable(extra) else extra
            alpha = float([ex[i + 1] for i, t in enumerate(ex) if t == "--alpha"][-1])
            j = copy.deepcopy(full[(near(alpha), s)])
            off = OFFSET[arm][s - 1]
            j.update(seed=s, alpha=alpha)
            j["git"] = {"commit": HEAD, "dirty": False}     # 真实的子空间 run 会记录当时的提交
            j["run_config"].update({"ddo_dirs": DIRS[arm], "ddo_k": K.get(arm, 32), "ddo_pca_src": "ln0",
                                    "ddo_dirs_seed": s if arm.startswith("rand") else 0, "alpha": alpha,
                                    "seed": s, "run_tag": tag})
            key = (arm, s) if arm.startswith("rand") else arm
            info = {"ddo_dirs": DIRS[arm], "pairing": "plus_minus" if arm in ("pca8", "pca32", "rand8", "rand32",
                                                                             "fullsym") else "none",
                    "rows": ROWS[arm], "sha256": sha(str(key))}
            if arm in MECH:
                info["loss_path"] = "affine_linear (LN0 已施加, 不再逐行归一化)"   # 报告的完整性核对要求机制臂带此字段
            if arm.startswith("rand"):
                info["seed"] = s
            j["ddo_dirs_info"] = info
            fd = j["final_diag"]
            fd["target_acc"] = fd["target_acc"] + off
            if arm == "rand8" and alpha == 4.0:
                fd["ddo_eq"] = 3.0                     # 落在 full 曲线范围之外 (3 个 run)
            if alpha == 8.0:
                fd["ddo_eq"] = 0.2                     # 低于 full 的最小值 (3 个 run)
            for g in j["grad_diag"]:
                g["alpha"] = alpha
                g["grad_norm_ratio_train_alpha_over_ce"] = alpha * 0.05
            with open(os.path.join(res, f"CUB_{tag}_seed{s}.json"), "w", encoding="utf-8") as f:
                json.dump(j, f)

r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", os.path.join(REPO, "ddo_subspace_report.py"),
                   "--results_dir", res, "--out_dir", outd], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
txt = r.stdout.decode("utf-8", "replace")
print(txt)
ok = r.returncode == 0
rep = json.load(open(os.path.join(outd, "subspace_cub.json"), encoding="utf-8"))
ok &= all(c["ok"] for c in rep["integrity"])


def verdict(q, k):
    if q == "q1":
        return rep["q1_pca_vs_random"][k]["verdict"]
    if q == "q2":
        return rep["q2_lowdim_vs_full"][k]["verdict"]
    if q == "q3":
        return rep["q3_symmetric"]["verdict"]
    if q == "q4":
        return rep["q4_tradeoff"][k]["verdict"]
    if q == "q_mech":
        return rep["q_mech_vs_full"][k]["verdict"]
    return rep["q_mech_proj_vs_pca"][k]["verdict"]


for (q, k), want in EXPECT.items():
    got = verdict(q, k)
    good = got == want
    ok &= good
    print(("  OK  " if good else "  BAD ") + f"{q} {k}: 期望 {want}, 实际 {got}")
n_out = rep["q4_tradeoff"]["rand8"]["n_out_of_range"] + rep["q4_tradeoff"]["pca32"]["n_out_of_range"]
ok &= n_out == 6
print(("  OK  " if n_out == 6 else "  BAD ") + f"范围外的 run 数: 期望 6, 实际 {n_out}")
# 低剂量补充不进入 §3n 的判定: q2 pca8 的配对数应保持 12 (而不是 18)
n_pca8 = rep["q2_lowdim_vs_full"]["pca8"]["n_pairs"]
ok &= n_pca8 == 12
print(("  OK  " if n_pca8 == 12 else "  BAD ") + f"§3n 判定的配对集合未被低剂量扩大: pca8 - full 配对数 {n_pca8} (期望 12)")
md = open(os.path.join(outd, "subspace_cub.md"), encoding="utf-8").read()
ok &= all(h in md for h in ("## 2. 判定汇总", "## 6. ④", "## 8. 机制对照", "## 9. 低剂量补充",
                            "## 10. 读法与限制")) and "| **支持** |" in md
ok &= all(os.path.exists(os.path.join(outd, n)) for n in ("subspace_cub.json", "subspace_cub.md",
                                                          "fig_subspace_curve.png", "fig_subspace_tradeoff.png"))
print("  OK  报告、数据与两张图都已生成" if ok else "  BAD  产物不全")
print('KEEP:', tmp) if os.environ.get('KEEP') else shutil.rmtree(tmp, ignore_errors=True)
print("REPORT_SELFTEST:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
