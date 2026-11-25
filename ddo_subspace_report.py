"""
DDO-Subspace 正式矩阵的报告 (2026-11-25)。

判定规则预注册于 experiment_results/RESULTS.md §3n (写于任何子空间 run 之前), 本脚本只是把它实现成代码。
规则里的阈值是下面的常量, 结果出来之后不改:
  MIN_EFFECT = 0.5      最小关心效应 (目标域精度, 百分点)
  OPERATING_BAND = 0.3  "只改变工作点" 的残差带宽 (目标域精度与源域验证 CE 各自的均值)
矩阵定义直接取自 run_matrix (subspace / dose), 不在这里重复写剂量与臂。
本脚本在矩阵运行期间写成, 只用 21 个已知的 full 剂量结果加合成数据测试过 (tests_subspace_report.py),
在它提交之前没有读过任何真实的子空间结果。

  ① pca − rand (同 k、同 α)                 语言域方向是否有特异性价值
  ② pca / mean − full (同 α)                 低维结构是否优于完整方向集
  ③ full_sym − full (α=1、2)                 正负成对本身有没有影响
  ④ 各子空间 run 相对 full 的 CE–DDO 曲线残差   改善的是权衡关系还是只换了工作点
"配对" 指同一训练 seed 相减; "seed 均值" 指某个 seed 在该比较涉及的全部剂量上的平均差。

产物: experiment_results/DDO-Subspace/subspace_cub.json、subspace_cub.md、fig_subspace_curve.png、fig_subspace_tradeoff.png
用法: python ddo_subspace_report.py [--results_dir DIR] [--out_dir DIR]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from run_matrix import COMMON, MATRICES, expected  # noqa: E402  (导入时执行驱动的自检)
from run_status import result_is_complete  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
MIN_EFFECT = 0.5
OPERATING_BAND = 0.3
SEEDS = (1, 2, 3)
ARMS = ["mean", "pca8", "pca32", "rand8", "rand32", "fullsym"]          # §3n 预注册的 6 个臂
MECH_ARMS = ["proj8", "res8", "split8", "proj32", "res32", "split32"]   # §3p 机制对照的 6 个臂
NAMES = {"full": "full（原始方向）", "mean": "mean（类平均方向）", "pca8": "pca8（前 8 个主方向）",
         "pca32": "pca32（前 32 个主方向）", "rand8": "rand8（8 维随机标架）", "rand32": "rand32（32 维随机标架）",
         "fullsym": "full_sym（原始方向 + 相反数）",
         "proj8": "proj8（原始方向在 8 维主子空间的投影）", "res8": "res8（8 维主子空间外的残余）",
         "split8": "split8（投影 + 残余，损失求和）",
         "proj32": "proj32（原始方向在 32 维主子空间的投影）", "res32": "res32（32 维主子空间外的残余）",
         "split32": "split32（投影 + 残余，损失求和）"}
EXPECT_DIRS = {"full": ("full", None, 32, 0), "mean": ("mean", None, 32, 0), "fullsym": ("full_sym", None, 32, 0),
               "pca8": ("pca", "ln0", 8, 0), "pca32": ("pca", "ln0", 32, 0), "rand8": ("random", "ln0", 8, None),
               "rand32": ("random", "ln0", 32, None),
               "proj8": ("projected", "ln0", 8, 0), "proj32": ("projected", "ln0", 32, 0),
               "res8": ("residual", "ln0", 8, 0), "res32": ("residual", "ln0", 32, 0),
               "split8": ("split", "ln0", 8, 0), "split32": ("split", "ln0", 32, 0)}
EXPECT_ROWS = {"full": 40800, "mean": 204, "fullsym": 81600, "pca8": 16, "pca32": 64, "rand8": 16, "rand32": 64,
               "proj8": 40800, "proj32": 40800, "res8": 40800, "res32": 40800, "split8": 81600, "split32": 81600}
# 外审第三轮后追加的低剂量 run (RESULTS §3p): 描述性, 不进入 §3n 的判定配对 —— §3n 的判定集合保持 66 run 不变
POST_HOC = {("pca8", 0.125), ("pca8", 0.25), ("pca32", 0.125), ("pca32", 0.25)}
GD_TAGS = ["before_epoch1", "before_epoch10", "before_epoch30", "final"]


def _git(*args):
    r = subprocess.run(["git"] + list(args), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.returncode, r.stdout.decode("utf-8", "replace").strip()


def jobs_of(matrix):
    """[(arm, alpha, dataset, tag, seed, 命令行额外参数)]"""
    out = []
    for ds, tag, extra in MATRICES[matrix][1]:
        for seed in SEEDS:
            ex = extra(seed) if callable(extra) else extra
            alpha = float([ex[i + 1] for i, t in enumerate(ex) if t == "--alpha"][-1])
            arm = tag.split("_")[1] if tag.startswith(("sub_", "mech_")) else "full"
            out.append((arm, alpha, ds, tag, seed, ex))
    return out


def gd(j, tag, key):
    for g in j.get("grad_diag") or []:
        if g["tag"] == tag:
            return g.get(key)
    return None


def alpha_ratio(j, tag):
    """乘以 alpha 之后的实际梯度范数比 (自身正则项 / 分类损失)。旧结果没有新字段时用 alpha x 未乘的比值;
    full 的剂量曲线是等权 DDO, 自身正则项就是等权 DDO。"""
    v = gd(j, tag, "grad_norm_ratio_train_alpha_over_ce")
    if v is None:
        v = gd(j, tag, "grad_norm_ratio_train_over_ce")
        if v is None:
            v = gd(j, tag, "grad_norm_ratio_ddo_over_ce")
        v = None if v is None else float(j["alpha"]) * v
    return v


class Data:
    def __init__(self, results_dir):
        self.r = {}                       # (arm, alpha, seed) -> json
        self.checks = []                  # (名称, 是否通过, 说明)
        by_name = {}
        jobs = jobs_of("subspace") + jobs_of("mechanism") + jobs_of("dose")
        want_sub = sum(1 for j in jobs_of("subspace"))
        want_mech = sum(1 for j in jobs_of("mechanism"))
        for arm, alpha, ds, tag, seed, ex in jobs:
            path = os.path.join(results_dir, f"{ds}_{tag}_seed{seed}.json")
            if not os.path.exists(path):
                continue
            j = json.load(open(path, encoding="utf-8"))
            self.r[(arm, alpha, seed)] = j
            by_name[(arm, alpha, seed)] = (path, ex, ds, tag, seed)
        n_sub = sum(1 for k in self.r if k[0] in {a for a, *_ in jobs_of("subspace")})
        self.checks.append((f"子空间矩阵的 {want_sub} 个结果文件都存在（§3n 预注册 66 + §3p 追加低剂量 12）",
                            n_sub == want_sub, f"{n_sub} / {want_sub}"))
        n_mech = sum(1 for k in self.r if k[0] in {a for a, *_ in jobs_of("mechanism")})
        self.checks.append((f"机制对照矩阵（§3p）的 {want_mech} 个结果文件都存在", n_mech == want_mech,
                            f"{n_mech} / {want_mech}"))
        bad = [k for k, (path, ex, ds, tag, seed) in by_name.items()
               if not result_is_complete(path, expected(ex + COMMON, ds, seed, tag))]
        self.checks.append(("全部 run 在驱动的完成判据下为完成 (status / 100 epochs / 命令行全部配置逐项匹配)",
                            not bad, f"不满足 {len(bad)} 个" if bad else f"{len(by_name)} 个"))
        wrong = []
        for (arm, alpha, seed), j in self.r.items():
            if arm == "full":      # full 的 21 个剂量 run 早于 --ddo_dirs 参数与 ddo_dirs_info 字段; 其配置已由上一项核对
                continue
            rc, info = j["run_config"], j.get("ddo_dirs_info") or {}
            d, src, k, dseed = EXPECT_DIRS[arm]
            cond = ((src is None or rc["ddo_pca_src"] == src) and rc["ddo_dirs"] == d and rc["ddo_k"] == k
                    and (dseed is None or rc["ddo_dirs_seed"] == dseed)
                    and info.get("rows") == EXPECT_ROWS[arm] and len(info.get("sha256", "")) == 64)
            if arm in ("proj8", "proj32", "res8", "res32", "split8", "split32"):
                cond = cond and "loss_path" in info          # 这些臂必须走"不再逐行归一化"的损失路径
            if arm.startswith("rand"):
                cond = cond and rc["ddo_dirs_seed"] == seed and (info.get("seed") == seed)
            if not cond:
                wrong.append(f"{arm} a{alpha} s{seed}")
        self.checks.append(("每个子空间/机制 run 的方向集与设计一致 (类型 / PCA 来源 / k / 方向种子 / 方向数 / 摘要 / 损失路径)",
                            not wrong, "; ".join(wrong[:5]) if wrong else "通过"))
        sha = {}
        for (arm, alpha, seed), j in self.r.items():
            if arm == "full":
                continue
            key = (arm, seed) if arm.startswith("rand") else arm
            sha.setdefault(key, set()).add(j["ddo_dirs_info"]["sha256"])
        one = all(len(v) == 1 for v in sha.values())
        frames = {s: next(iter(sha[("rand8", s)])) for s in SEEDS if ("rand8", s) in sha}
        diff_frames = len(set(frames.values())) == len(frames)
        self.checks.append(("方向集摘要一致: 同一臂 (随机臂: 同一方向种子) 的所有 run 摘要相同", one,
                            "通过" if one else "不一致"))
        self.checks.append(("三个随机标架 (方向种子 1/2/3) 互不相同", diff_frames, f"{len(set(frames.values()))} / {len(frames)}"))
        # 代码版本: 每个 run 的结果里记录了 git.commit / git.dirty。矩阵运行期间新增的只能是分析脚本与报告,
        # 训练代码必须与当前 HEAD 逐字节相同。(参照用的 full 剂量曲线是更早的提交上跑的, 单独列出)
        sub = [j for (arm, _, _), j in self.r.items() if arm != "full" and isinstance(j.get("git"), dict)
               and j["git"].get("commit")]
        commits = sorted({j["git"]["commit"] for j in sub})
        n_dirty = sum(bool(j["git"].get("dirty")) for j in sub)
        head = _git("rev-parse", "HEAD")[1]
        ancestors = bool(commits) and all(_git("merge-base", "--is-ancestor", c, "HEAD")[0] == 0 for c in commits)
        self.checks.append(("每个子空间 run 记录的代码提交都是当前 HEAD 的祖先", ancestors,
                            " / ".join(c[:7] for c in commits) + f" (HEAD {head[:7]})"))
        changed = set()
        for c in commits:
            changed |= set(x for x in _git("diff", "--name-only", c, "HEAD")[1].splitlines() if x)
        non_code = all(x.startswith(("experiment_results/", "doc/")) or x in
                       ("ddo_subspace_report.py", "tests_subspace_report.py", "tests_projected_dirs.py",
                        "ddo_dose_report.py", "ddo_principal_class_relevance.py") for x in changed)
        self.checks.append(("子空间 run 的提交与当前 HEAD 之间只有分析脚本与报告不同, 训练代码逐字节相同", non_code,
                            "; ".join(sorted(changed))[:300] if changed else "没有差异"))
        ref = sorted({j["git"]["commit"][:7] for (arm, _, _), j in self.r.items()
                      if arm == "full" and isinstance(j.get("git"), dict) and j["git"].get("commit")})
        self.checks.append((f"子空间 run 中记录为 dirty 的: {n_dirty} / {len(sub)}；参照用的 full 曲线的提交: "
                            + (" / ".join(ref) if ref else "未记录"), True,
                            "dirty 为 true 是因为工作区里有未跟踪的分析脚本 (它不参与训练, 见上一项)"))
        self.integrity_ok = all(c[1] for c in self.checks)

    def has(self, arm, alpha, seed):
        return (arm, alpha, seed) in self.r

    def get(self, arm, alpha, seed, key):
        return self.r[(arm, alpha, seed)]["final_diag"][key]

    def alphas(self, arm):
        return sorted({a for (m, a, _) in self.r if m == arm})


def ms(x):
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)


def paired(d, arm_a, arm_b, key="target_acc", skip=frozenset()):
    """arm_a − arm_b, 同 α、同训练 seed。返回 {alpha: {seed: diff}} (只含两边都有的组合)。
    skip: {(arm, alpha)} 集合 —— arm_a 在这些 (臂, α) 上的 run 不参与配对 (§3p 追加的低剂量不进 §3n 的判定)。"""
    out = {}
    for a in sorted(set(d.alphas(arm_a)) & set(d.alphas(arm_b))):
        if (arm_a, a) in skip:
            continue
        for s in SEEDS:
            if d.has(arm_a, a, s) and d.has(arm_b, a, s):
                out.setdefault(a, {})[s] = d.get(arm_a, a, s, key) - d.get(arm_b, a, s, key)
    return out


def flat(p):
    return [v for per in p.values() for v in per.values()]


def seed_means(p):
    """{seed: 该 seed 在全部 α 上的平均差}"""
    out = {}
    for a, per in p.items():
        for s, v in per.items():
            out.setdefault(s, []).append(v)
    return {s: float(np.mean(v)) for s, v in sorted(out.items())}


def label_effect(p):
    """§3n 的 ① / ② 规则"""
    f, sm = flat(p), seed_means(p)
    if not f:
        return "无法判定（没有配对）"
    mean = float(np.mean(f))
    n_pos = sum(v > 0 for v in sm.values())
    if mean >= MIN_EFFECT and n_pos == len(sm):
        return "支持"
    if mean <= 0 or n_pos < 2:
        return "不支持"
    return "不确定"


def label_sym(p):
    """§3n 的 ③ 规则"""
    f = flat(p)
    if not f:
        return "无法判定（没有配对）"
    mean = float(np.mean(f))
    if abs(mean) >= MIN_EFFECT and (all(v > 0 for v in f) or all(v < 0 for v in f)):
        return "有影响（正向）" if mean > 0 else "有影响（负向）"
    return "未见影响"


def label_trade(r_acc_by_seed, r_ce):
    """§3n 的 ④ 规则 (r_acc_by_seed: {seed: [每个落在曲线范围内的 run 的目标域精度残差]})"""
    f = [v for vs in r_acc_by_seed.values() for v in vs]
    if not f:
        return "无法判定（没有落在曲线范围内的 run）"
    sm = {s: float(np.mean(vs)) for s, vs in r_acc_by_seed.items()}
    mean_acc = float(np.mean(f))
    if mean_acc >= MIN_EFFECT and all(v > 0 for v in sm.values()):
        return "改善权衡"
    if abs(mean_acc) < OPERATING_BAND and abs(r_ce) < OPERATING_BAND:
        return "只改变工作点"
    return "不确定"


def fmt_seeds(per):
    return " / ".join(f"{per[s]:+.2f}" for s in sorted(per) if s in per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join("logs", "results"))
    ap.add_argument("--out_dir", default=os.path.join("experiment_results", "DDO-Subspace"))
    opt = ap.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)
    d = Data(opt.results_dir)
    get = lambda arm, a, key: np.array([d.get(arm, a, s, key) for s in SEEDS if d.has(arm, a, s)])
    has = lambda arm, a: all(d.has(arm, a, s) for s in SEEDS)
    out = {"integrity": [{"check": n, "ok": bool(ok), "detail": det} for n, ok, det in d.checks],
           "thresholds": {"min_effect": MIN_EFFECT, "operating_band": OPERATING_BAND}}

    # ---------- 1. 剂量曲线 (目标域精度; 统一口径的源域验证 CE 与最终等权 DDO 损失) ----------
    all_a = sorted({a for arm in ARMS + MECH_ARMS + ["full"] for a in d.alphas(arm)})
    curve_arms = ARMS + MECH_ARMS + ["full"]
    curves = {arm: {str(a): {"target_acc": {"seeds": [round(float(v), 3) for v in get(arm, a, "target_acc")],
                                         "mean": round(ms(get(arm, a, "target_acc"))[0], 3),
                                         "sd": round(ms(get(arm, a, "target_acc"))[1], 3)},
                             "source_ce": round(float(np.mean(get(arm, a, "source_ce"))), 4),
                             "source_acc": round(float(np.mean(get(arm, a, "source_acc"))), 3),
                             "ddo_eq": round(float(np.mean(get(arm, a, "ddo_eq"))), 4)}
                   for a in d.alphas(arm) if has(arm, a)} for arm in curve_arms}
    out["curves"] = curves

    # ---------- ① / ② ----------
    def summary(p):
        return {"pairs": {str(a): {str(s): round(v, 3) for s, v in per.items()} for a, per in p.items()},
                "mean": round(float(np.mean(flat(p))), 3) if flat(p) else None,
                "seed_means": {str(s): round(v, 3) for s, v in seed_means(p).items()},
                "n_pairs": len(flat(p))}
    q1, q2 = {}, {}
    for k in (8, 32):
        p = paired(d, f"pca{k}", f"rand{k}")
        q1[f"k={k}"] = {**summary(p), "verdict": label_effect(p)}
    for arm in ("pca8", "pca32", "mean"):
        p = paired(d, arm, "full", skip=POST_HOC)
        q2[arm] = {**summary(p), "verdict": label_effect(p)}
    out["q1_pca_vs_random"], out["q2_lowdim_vs_full"] = q1, q2

    # ---------- §3p: 机制对照 (projected / residual / split) ----------
    q_mech = {}
    for arm in MECH_ARMS:
        p = paired(d, arm, "full")
        q_mech[arm] = {**summary(p), "verdict": label_effect(p)}
    q_proj_vs_pca = {}
    for k in (8, 32):
        p = paired(d, f"proj{k}", f"pca{k}")
        q_proj_vs_pca[f"k={k}"] = {**summary(p), "verdict": label_effect(p)}
    out["q_mech_vs_full"], out["q_mech_proj_vs_pca"] = q_mech, q_proj_vs_pca

    # ---------- ③ ----------
    p3 = paired(d, "fullsym", "full")
    p3b = {f"k={k}": {str(a): {str(s): round(d.get(f"pca{k}", a, s, "target_acc") - d.get("fullsym", a, s, "target_acc"), 3)
                              for s in SEEDS if d.has(f"pca{k}", a, s) and d.has("fullsym", a, s)}
                      for a in (1.0, 2.0) if any(d.has("fullsym", a, s) for s in SEEDS)} for k in (8, 32)}
    out["q3_symmetric"] = {**summary(p3), "verdict": label_sym(p3), "pca_minus_fullsym": p3b}

    # ---------- ④: 相对 full 的 CE–DDO 曲线残差 ----------
    full_a = [a for a in d.alphas("full") if has("full", a)]
    fx = [d.get("full", a, s, "ddo_eq") for a in d.alphas("full") for s in SEEDS if d.has("full", a, s)]
    x_lo, x_hi = (min(fx), max(fx)) if fx else (float("nan"), float("nan"))
    cx = np.array([np.mean(get("full", a, "ddo_eq")) for a in full_a])
    cy_acc = np.array([np.mean(get("full", a, "target_acc")) for a in full_a])
    cy_ce = np.array([np.mean(get("full", a, "source_ce")) for a in full_a])
    order = np.argsort(cx)
    cx, cy_acc, cy_ce = cx[order], cy_acc[order], cy_ce[order]
    out["full_curve"] = {"ddo_eq": cx.tolist(), "target_acc": cy_acc.tolist(), "source_ce": cy_ce.tolist(),
                         "range_all_runs": [x_lo, x_hi]}
    q4 = {}
    for arm in ARMS:
        res_acc, res_ce, n_out, rows = {}, [], 0, []
        for a in d.alphas(arm):
            if (arm, a) in POST_HOC:                # §3p 追加的低剂量不进入 §3n 的判定
                continue
            for s in SEEDS:
                if not d.has(arm, a, s):
                    continue
                x = d.get(arm, a, s, "ddo_eq")
                if not (x_lo <= x <= x_hi):
                    n_out += 1
                    rows.append({"alpha": a, "seed": s, "ddo_eq": round(x, 4), "in_range": False})
                    continue
                ra = d.get(arm, a, s, "target_acc") - float(np.interp(x, cx, cy_acc))
                rc = d.get(arm, a, s, "source_ce") - float(np.interp(x, cx, cy_ce))
                res_acc.setdefault(s, []).append(ra)
                res_ce.append(rc)
                rows.append({"alpha": a, "seed": s, "ddo_eq": round(x, 4), "in_range": True,
                             "res_target_acc": round(ra, 3), "res_source_ce": round(rc, 4)})
        f = [v for vs in res_acc.values() for v in vs]
        q4[arm] = {"n_in_range": len(f), "n_out_of_range": n_out,
                   "mean_res_target_acc": round(float(np.mean(f)), 3) if f else None,
                   "mean_res_source_ce": round(float(np.mean(res_ce)), 4) if res_ce else None,
                   "seed_means_res_target_acc": {str(s): round(float(np.mean(v)), 3) for s, v in sorted(res_acc.items())},
                   "verdict": label_trade(res_acc, float(np.mean(res_ce)) if res_ce else float("nan")), "runs": rows}
    out["q4_tradeoff"] = q4

    # ---------- 描述性: 剂量敏感性 / 实际梯度强度 ----------
    sens = {}
    for arm in ARMS + ["full"]:
        al = [a for a in d.alphas(arm) if has(arm, a)]
        m = [float(np.mean(get(arm, a, "target_acc"))) for a in al]
        sens[arm] = {"alphas": al, "mean_acc": [round(v, 3) for v in m],
                     "range": round(max(m) - min(m), 3) if m else None,
                     "drop_to_highest_alpha": round(max(m) - m[-1], 3) if m else None}
    out["dose_sensitivity"] = sens
    grad = {}
    for arm in ARMS + MECH_ARMS + ["full"]:
        grad[arm] = {}
        for a in d.alphas(arm):
            rs = [d.r[(arm, a, s)] for s in SEEDS if d.has(arm, a, s)]
            grad[arm][str(a)] = {}
            for t in GD_TAGS:
                vs = [alpha_ratio(j, t) for j in rs]
                vs = [v for v in vs if v is not None]
                grad[arm][str(a)][t] = round(float(np.mean(vs)), 4) if vs else None
    out["grad_ratio_alpha_scaled"] = grad

    # ---------- Markdown ----------
    L = ["# DDO-Subspace 矩阵报告（CUB，3 seeds；§3n 预注册 66 run + §3p 追加 84 run，2026-11-25）", "",
         "**设计**：第三章入口 + 等权 DDO，batch 256、100 epochs，严格 final 协议（目标域不参与训练与选模，特征缓存在最终评估时才读入）。",
         "**判定规则预注册于 `RESULTS.md` §3n（66 run）与 §3p（机制对照 72 run + 低剂量补充 12 run）**"
         "（最小关心效应 +0.5；\"只改变工作点\"的残差带宽 ±0.3），由 `ddo_subspace_report.py` 计算，"
         "该脚本及规则在读取任何子空间结果之前提交。随机标架的种子与训练 seed 配对。"
         "数据 `subspace_cub.json`，图 `fig_subspace_curve.png`、`fig_subspace_tradeoff.png`。", "",
         "## 0. 完整性核对", "", "| 核对项 | 结果 | 说明 |", "|---|---|---|"]
    for n, ok, det in d.checks:
        L.append(f"| {n} | {'通过' if ok else '**未通过**'} | {det} |")
    L += ["", "## 1. 剂量曲线（目标域精度，3 seeds）", "",
          "| α | " + " | ".join(NAMES[a] for a in ["full"] + ARMS) + " |",
          "|---|" + "---|" * (1 + len(ARMS))]
    for a in all_a:
        cells = []
        for arm in ["full"] + ARMS:
            c = curves[arm].get(str(a))
            cells.append(f"{c['target_acc']['mean']:.2f} ± {c['target_acc']['sd']:.2f}" if c else "—")
        L.append(f"| {a:g} | " + " | ".join(cells) + " |")
    L += ["", "统一口径（各臂 3 seeds 均值）：", "",
          "| 臂 | α | 源域验证 CE | 最终等权 DDO 损失（原始完整方向集） | 源域精度 |", "|---|---|---|---|---|"]
    for arm in ["full"] + ARMS:
        for a in d.alphas(arm):
            if str(a) in curves[arm]:
                c = curves[arm][str(a)]
                L.append(f"| {NAMES[arm]} | {a:g} | {c['source_ce']:.4f} | {c['ddo_eq']:.4f} | {c['source_acc']:.2f} |")

    def block(title, q, rule):
        L.extend([f"### {title}", ""])
        if not q["n_pairs"]:
            L.extend(["没有可用的配对（结果缺失）。", ""])
            return
        L.extend(["| α | 逐 seed 配对差值 | 均值 |", "|---|---|---|"])
        for a, per in q["pairs"].items():            # pairs 里的 seed 键是字符串, 按 seed 排序后列出
            v = [per[s] for s in sorted(per, key=int)]
            L.append(f"| {a} | {' / '.join(f'{x:+.2f}' for x in v)} | {np.mean(v):+.2f} |")
        L.extend(["", f"全部 {q['n_pairs']} 对：均值 **{q['mean']:+.2f}**；seed 均值 "
                      + " / ".join(f"s{s}: {v:+.2f}" for s, v in q["seed_means"].items())
                  + f"；判定：**{q['verdict']}**。{rule}", ""])

    L += ["", "## 2. 判定汇总", "", "| 维度 | 比较 | 配对数 | 均值 | seed 均值 | 判定 |", "|---|---|---|---|---|---|"]
    for k, q in q1.items():
        if q["n_pairs"]:
            L.append(f"| ① 语言域方向的特异性价值 | pca − rand（{k}） | {q['n_pairs']} | {q['mean']:+.2f} | "
                     + " / ".join(f"{v:+.2f}" for v in q["seed_means"].values()) + f" | **{q['verdict']}** |")
    for arm, q in q2.items():
        if q["n_pairs"]:
            L.append(f"| ② 低维结构是否优于完整方向集 | {NAMES[arm]} − full | {q['n_pairs']} | {q['mean']:+.2f} | "
                     + " / ".join(f"{v:+.2f}" for v in q["seed_means"].values()) + f" | **{q['verdict']}** |")
    if p3:
        L.append(f"| ③ 正负成对本身的影响 | full_sym − full（α=1、2） | {len(flat(p3))} | {out['q3_symmetric']['mean']:+.2f} | "
                 + " / ".join(f"{v:+.2f}" for v in seed_means(p3).values()) + f" | **{out['q3_symmetric']['verdict']}** |")
    for arm in ARMS:
        q = q4[arm]
        if q["mean_res_target_acc"] is None:
            L.append(f"| ④ 改善权衡关系，还是只换工作点 | {NAMES[arm]} 相对 full 曲线的残差 | 0（范围外 {q['n_out_of_range']}） | — | — | **{q['verdict']}** |")
            continue
        L.append(f"| ④ 改善权衡关系，还是只换工作点 | {NAMES[arm]} 相对 full 曲线的残差 | {q['n_in_range']}（范围外 {q['n_out_of_range']}） | "
                 f"精度 {q['mean_res_target_acc']:+.2f} / CE {q['mean_res_source_ce']:+.4f} | "
                 + " / ".join(f"{v:+.2f}" for v in q["seed_means_res_target_acc"].values()) + f" | **{q['verdict']}** |")
    L += ["", "## 3. ① pca − rand（同 k、同 α，逐 seed 配对）"]
    for k, q in q1.items():
        kk = k.split("=")[1]
        block(f"{k}：{NAMES['pca' + kk]} − {NAMES['rand' + kk]}", q,
              "规则：均值 ≥ +0.5 且三个 seed 均值都 > 0 为支持；均值 ≤ 0 或 seed 均值为正的少于 2 个为不支持；其余为不确定。")
    L += ["", "## 4. ② 低维结构 − full（同 α，逐 seed 配对）"]
    for arm, q in q2.items():
        block(f"{NAMES[arm]} − {NAMES['full']}", q, "规则同 ①。pca32 的 α=8 没有 full 对照，不参与配对。")
    L += ["", "## 5. ③ full_sym − full（α=1、2，逐 seed 配对）", ""]
    if p3:
        L += ["| α | 逐 seed 配对差值 | 均值 |", "|---|---|---|"]
        for a, per in p3.items():
            L.append(f"| {a} | {fmt_seeds(per)} | {np.mean(list(per.values())):+.2f} |")
        L += ["", f"全部 {len(flat(p3))} 对：均值 **{out['q3_symmetric']['mean']:+.2f}**；判定：**{out['q3_symmetric']['verdict']}**"
                  "（规则：|均值| ≥ 0.5 且三个 seed 同号才算有影响）。"]
    L += ["", "pca − full_sym（同 α；只报告，不判定）：", "", "| α | pca8 − full_sym | pca32 − full_sym |", "|---|---|---|"]
    for a in ("1.0", "2.0"):
        cells = ["/".join(f"{v:+.2f}" for _, v in sorted(p3b[f"k={k}"].get(a, {}).items(), key=lambda kv: int(kv[0]))) or "—"
                 for k in (8, 32)]
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", "## 6. ④ 相对 full 的 CE–DDO 曲线残差", "",
          "横轴为最终模型在原始完整方向集上的等权 DDO 损失，纵轴为目标域精度与源域验证 CE。曲线取 full 各剂量 3 seeds 均值的线性插值"
          f"（横坐标范围 {x_lo:.3f}～{x_hi:.3f}，即 full 全部 21 个 run 的范围）。残差 = 实际值 − 曲线在同一 DDO 损失处的值。", "",
          "| 臂 | 范围内 run 数 | 范围外 run 数 | 目标域精度残差（均值；逐 seed 均值） | 源域 CE 残差均值 | 判定 |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        q = q4[arm]
        if q["mean_res_target_acc"] is None:
            L.append(f"| {NAMES[arm]} | 0 | {q['n_out_of_range']} | — | — | **{q['verdict']}** |")
            continue
        L.append(f"| {NAMES[arm]} | {q['n_in_range']} | {q['n_out_of_range']} | "
                 f"{q['mean_res_target_acc']:+.2f}；" + " / ".join(f"{v:+.2f}" for v in q["seed_means_res_target_acc"].values())
                 + f" | {q['mean_res_source_ce']:+.4f} | **{q['verdict']}** |")
    L += ["", "规则：目标域精度残差均值 ≥ +0.5 且三个 seed 均值都 > 0 为改善权衡；精度残差与 CE 残差的均值都在 ±0.3 以内为只改变工作点；其余为不确定。", "",
          "## 7. 描述性结果（不下判定）", "",
          "剂量敏感性（各臂目标域精度均值）：", "",
          "| 臂 | 各 α 的精度 | 极差 | 最好剂量到最高剂量的回落 |", "|---|---|---|---|"]
    for arm in ["full"] + ARMS:
        s_ = sens[arm]
        if not s_["alphas"]:
            continue
        L.append(f"| {NAMES[arm]} | " + " / ".join(f"{a:g}: {v:.2f}" for a, v in zip(s_["alphas"], s_["mean_acc"]))
                 + f" | {s_['range']:.2f} | {s_['drop_to_highest_alpha']:.2f} |")
    L += ["", "乘以 α 之后的实际梯度范数比（α·‖∇正则‖/‖∇CE‖，固定批次、分类头参数，3 seeds 均值）：", "",
          "| 臂 | α | " + " | ".join(GD_TAGS) + " |", "|---|---|" + "---|" * len(GD_TAGS)]
    for arm in ["full"] + ARMS + MECH_ARMS:
        for a in d.alphas(arm):
            g = grad[arm][str(a)]
            L.append(f"| {NAMES[arm]} | {a:g} | " + " | ".join("—" if g[t] is None else f"{g[t]:.3f}" for t in GD_TAGS) + " |")
    L += ["", "## 8. 机制对照（§3p，预注册于机制矩阵运行之前）", "",
          "外审第三轮的分析：PCA 臂的失败可能不是因为保留了错误的子空间，而是用等权正交标架替换了原始经验方向分布、"
          "改变了 L1 正则化的几何。下面的臂把每个原始方向（经 LN0）投影到同一个主子空间内（projected）、"
          "或只约束子空间外的余量（residual）、或对两个分量分别求损失后相加（split，因此其有效强度约为同剂量 projected 与 residual 之和）。"
          "方向不重新逐行归一化（保留投影幅值），损失路径绕过分类头的 LayerNorm。判定规则与 §3n 相同（+0.5 / 配对同 α 同训练 seed）。",
          "", "剂量曲线（目标域精度，3 seeds 均值 ± sd）：", "",
          "| α | " + " | ".join(NAMES[a] for a in ["full"] + MECH_ARMS) + " |",
          "|---|" + "---|" * (1 + len(MECH_ARMS))]
    for a in all_a:
        cells = []
        for arm in ["full"] + MECH_ARMS:
            c = curves[arm].get(str(a))
            cells.append(f"{c['target_acc']['mean']:.2f} ± {c['target_acc']['sd']:.2f}" if c else "—")
        L.append(f"| {a:g} | " + " | ".join(cells) + " |")
    L += ["", "| 比较 | 配对数 | 均值 | seed 均值 | 判定 |", "|---|---|---|---|---|"]
    for arm in MECH_ARMS:
        q = q_mech[arm]
        if q["n_pairs"]:
            L.append(f"| {NAMES[arm]} − full | {q['n_pairs']} | {q['mean']:+.2f} | "
                     + " / ".join(f"{v:+.2f}" for v in q["seed_means"].values()) + f" | **{q['verdict']}** |")
    for k, q in q_proj_vs_pca.items():
        if q["n_pairs"]:
            L.append(f"| proj − pca（{k}，同 α） | {q['n_pairs']} | {q['mean']:+.2f} | "
                     + " / ".join(f"{v:+.2f}" for v in q["seed_means"].values()) + f" | **{q['verdict']}** |")
    L += ["", "逐 α 的配对差值：", ""]
    for arm in MECH_ARMS:
        block(f"{NAMES[arm]} − {NAMES['full']}", q_mech[arm], "规则同 §3n ①。")
    for k, q in q_proj_vs_pca.items():
        kk = k.split("=")[1]
        block(f"proj − pca（{k}）", q, "规则同 §3n ①。回答：同一子空间上，原始方向分布与等权正交标架哪个更重要。")
    L += ["**事后解释框架（不是判定）**：Projected 接近 Full 而 PCA-Basis 差 → 方向分布比标架重要（→ L1 感知压缩）；"
          "Projected 与 PCA-Basis 都差而 Residual 较好 → 主方向过度携带类别语义（→ 残差约束与判别保护）；"
          "Projected 与 Residual 都差 → 主方向与长尾互补（→ 多尺度联合约束）；Split 优于/差于 Full → 分量间抵消有害/有益。",
          "", "## 9. 低剂量补充（外审后追加，描述性，不进入 §3n 的判定配对）", "",
          "外审指出 pca8 / pca32 都在各自最低的非零剂量处最好，不能在剂量覆盖不足时断言任何非零剂量下都有害。"
          "以下 12 个 run（α=0.125 / 0.25）按外审建议追加，只用于补全剂量曲线的下端：", "",
          "| 臂 | α | 目标域精度（3 seeds） | 对 full 同 α 的逐 seed 差值 | 最终等权 DDO 损失 |", "|---|---|---|---|---|"]
    for arm in ("pca8", "pca32"):
        for a in (0.125, 0.25):
            if not has(arm, a):
                continue
            t = get(arm, a, "target_acc")
            diff = " / ".join(f"{d.get(arm, a, s, 'target_acc') - d.get('full', a, s, 'target_acc'):+.2f}" for s in SEEDS)
            L.append(f"| {NAMES[arm]} | {a:g} | {np.mean(t):.2f} ± {ms(t)[1]:.2f} | {diff} | "
                     f"{np.mean(get(arm, a, 'ddo_eq')):.4f} |")
    L += ["", "## 10. 读法与限制", "",
          "- 3 个 seed、仅 CUB。CUB 是方法探索基准：这里的效应量要在未参与方法设计的数据集上独立确认之后才能当作结论。",
          "- 本矩阵不用于挑选 α，也不直接决定方法。所有比较都是同 α、同训练 seed 的配对，不拿各臂最好的 α 互比。",
          "- 同一问题在 k=8 与 k=32 上分别判定，只在一个 k 上得到\"支持\"时照实写，不合并。",
          "- 谱能量占比（k=8 为 80.0%，k=32 为 96.7%）是奇异值平方的累计，不等于 L1-DDO 约束的覆盖率。",
          "- 梯度比只在固定批次、所列时刻、分类头参数上测量，不代表两个目标之间没有冲突。",
          "- 判定阈值是预先写定的描述性规则，不是统计检验；§3n 与 §3p 两轮合计 17 次判定，没有做多重比较校正。",
          "- §3p 的机制对照与低剂量补充都写于运行之前（RESULTS §3p）；低剂量 run 不进入 §3n 的判定配对，§3n 的判定集合保持 66 run 不变。",
          "- 事后解释框架（第 8 节末）与 2.1/2.2 节的机制解释都是看到结果之后的分析，不能当作独立验证。",
          ""]
    with open(os.path.join(opt.out_dir, "subspace_cub.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(opt.out_dir, "subspace_cub.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---------- 图 (英文标注, 避免字体缺字) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

        def draw(ax, arms, title):
            for arm in arms:
                al = [a for a in d.alphas(arm) if a > 0 and has(arm, a)]     # 对数横轴画不了 alpha=0
                if not al:
                    continue
                m = np.array([np.mean(get(arm, a, "target_acc")) for a in al])
                sd = np.array([ms(get(arm, a, "target_acc"))[1] for a in al])
                ax.errorbar(al, m, yerr=sd, marker="o", capsize=3, label=NAMES[arm].split("（")[0])
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
            ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2", "4", "8"])
            ax.set_xlabel("alpha (dose)")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        draw(axes[0], ["full", "mean", "pca8", "rand8"], "k = 8 (and class-mean directions)")
        draw(axes[1], ["full", "pca32", "rand32", "fullsym"], "k = 32 (and full + negatives)")
        axes[0].set_ylabel("Target accuracy (%, mean +/- sd, 3 seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(opt.out_dir, "fig_subspace_curve.png"), dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(cx, cy_acc, "k.-", label="full (dose curve, mean)")
        for arm in ARMS:
            al = [a for a in d.alphas(arm) if has(arm, a)]
            if not al:
                continue
            xs = [np.mean(get(arm, a, "ddo_eq")) for a in al]
            ys = [np.mean(get(arm, a, "target_acc")) for a in al]
            ax.scatter(xs, ys, label=NAMES[arm].split("（")[0], s=28)
        ax.set_xlabel("final equal-weight DDO loss on the full direction set (lower = stronger constraint)")
        ax.set_ylabel("Target accuracy (%, mean of 3 seeds)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(opt.out_dir, "fig_subspace_tradeoff.png"), dpi=130)
        plt.close(fig)
    except Exception as e:  # 图不是必需产物
        print("[fig] skipped:", repr(e))

    print(f"完整性: {'全部通过' if d.integrity_ok else '有未通过项'}")
    for n, ok, det in d.checks:
        print(f"  {'OK ' if ok else 'BAD'} {n}: {det}")
    print("\n判定:")
    print("  ① pca − rand: " + "; ".join(f"{k}: {q['verdict']} (均值 {q['mean']:+.2f})" for k, q in q1.items() if q["n_pairs"]))
    print("  ② − full: " + "; ".join(f"{arm}: {q['verdict']} (均值 {q['mean']:+.2f})" for arm, q in q2.items() if q["n_pairs"]))
    if p3:
        print(f"  ③ full_sym − full: {out['q3_symmetric']['verdict']} (均值 {out['q3_symmetric']['mean']:+.2f})")
    print("  ④ 权衡: " + "; ".join(
        f"{arm}: {q['verdict']} (精度残差 {q['mean_res_target_acc']:+.2f}, 范围外 {q['n_out_of_range']})"
        if q["mean_res_target_acc"] is not None else f"{arm}: {q['verdict']} (全部 {q['n_out_of_range']} 个 run 在范围外)"
        for arm, q in q4.items()))
    if any(q["n_pairs"] for q in q_mech.values()):
        print("  §3p 机制对照 − full: " + "; ".join(f"{arm}: {q['verdict']} (均值 {q['mean']:+.2f})"
                                                    for arm, q in q_mech.items() if q["n_pairs"]))
        print("  §3p proj − pca: " + "; ".join(f"{k}: {q['verdict']} (均值 {q['mean']:+.2f})"
                                               for k, q in q_proj_vs_pca.items() if q["n_pairs"]))
    print(f"\n[saved] {os.path.join(opt.out_dir, 'subspace_cub.json')} / subspace_cub.md")


if __name__ == "__main__":
    main()
