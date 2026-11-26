"""Stage 0B 第二层 + 0D 诊断 (预注册见 RESULTS.md §3t; 外审第八轮 P0 修正版 + 同日收尾补正)。
零训练: 只读增强缓存 (cache/CUB_train_aug.pt)、训练缓存 (cache/CUB_train.pt, 仅做一致性核对)、
27 个 Stage 0 分类头 (logs/heads/)、概念矩阵 E 与域方向 (从数据集对象获取)。只用源域数据。

外审第八轮修正的六项实现问题 (预注册设计不变, 实现对齐预注册):
  1. 扰动强度 ‖Δu‖ 加入回归 (此前 X1/X2 均未包含, 违背 §3t 的三控制变量设计)
  2. 逐类别对的 avg_pert_strength (此前回退为全局常数)
  3. 补逐样本干净 top-5 竞争类口径 (此前仅类别平均 logits top-5)
  4. 补 S_vis = |wᵀΔu|/(‖Δu‖+ε) (预注册要求但未实现)
  5. 逐 seed 判定方向一致性 (此前用跨 seed 均值)
  6. 补投影范数 ‖Hw‖ (PCR 用投影后权重, w_norm 此前用原始权重)

P0 提交 (5e9f1ff) 之后自审发现的遗留, 本次补正 (判定阈值 0.05 / 0.01 未动):
  7. 第 6 项只算了没用: 回归的控制变量仍是未投影的 ‖w‖。PCR 与第一层报告的 ‖w‖² 用的都是投影后的决策方向
     (u 在超平面 H 内, 公共分量对决策无意义), 所以回归控制改用 ‖Hw‖; 未投影口径作为敏感性分析一并报告
     (它也是核对项: 本文件把逐样本循环改成了向量化累加, 未投影口径必须复现 5e9f1ff 的数字)。
  8. 预注册要求报告 PCR 系数的"显著性", 此前没有算: 补 OLS t 统计量与按锚定类别整群自助法的 95% 百分位区间。
     类别对共享分类头参数、不是独立样本 —— 这两个量只作指示。
  9. 如实记录评估集合为 heldout_val 1200 样本 (非 §3t 写的"完整训练集")。
  10. 恢复 md 报告 (5e9f1ff 只写了 JSON); 三套竞争类口径都保存 R² / 配对数 / S_vis / 扰动强度。
  11. 判定标签里"等待修正重算确认"是初版标签的残留 —— 本脚本本身就是修正重算, 改为"探索性"。
  12. 增强缓存的干净特征与训练缓存的一致性核对 (torch.equal / 最大绝对误差 / 相对误差) 写入结果。

用法: python stage0_layer2_diagnose.py [--selftest] [--bootstrap 1000]
产物: experiment_results/Stage0-Diagnostics/stage0_layer2_cub.json 与 stage0_layer2_cub.md (由同一份结果生成)
"""
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from concept_cancellation import diag_reference, pcr_quantile, quad, random_baseline
from language_ellipsoid import domain_directions, second_moments
from run_status import git_info

OUT_DIR = os.path.join("experiment_results", "Stage0-Diagnostics")
OUT_JSON, OUT_MD = "stage0_layer2_cub.json", "stage0_layer2_cub.md"
SEEDS = (1, 2, 3)
ARMS = [("ce_a0", "CE-Only（α=0）"), ("full_a1", "Full DDO（α=1）"), ("full_a2", "Full DDO（α=2）"),
        ("mean_a1", "Mean（α=1）"), ("mean_a2", "Mean（α=2）"), ("sym_a1", "Full-Symmetric（α=1）"),
        ("sym_a2", "Full-Symmetric（α=2）"), ("bf_a1", "Bias-Free（α=1）"), ("bf_a2", "Bias-Free（α=2）")]
FULL_ARMS, CE_ARM = ("full_a1", "full_a2"), "ce_a0"
OD_ARMS = (CE_ARM, "full_a1", "full_a2")
TOP_K = 5
EPS_LN = 1e-5
SCHEMES = {"fixed": "固定竞争类（CE-Only 头的类别平均 top-5）", "own_avg": "自身类别平均 top-5",
           "own_sample": "逐样本自身 top-5"}
NORMS = {"proj": "投影范数 ‖Hw‖", "raw": "未投影范数 ‖w‖"}     # 回归控制变量: proj 为主, raw 为敏感性
AUG_CACHE, AUG_META, TRAIN_CACHE = "cache/CUB_train_aug.pt", "cache/CUB_train_aug_meta.json", "cache/CUB_train.pt"
BOOT_SEED = 20261126

# ---- 预注册阈值 (§3t); 不改 ----
B2_DELTAR2_SUPPORT = 0.05
B2_DELTAR2_NONE = 0.01


def load_heads():
    out = {}
    for tag, _ in ARMS:
        for s in SEEDS:
            p = f"logs/heads/CUB_s0_{tag}_seed{s}.pt"
            if os.path.exists(p):
                out[(tag, s)] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def layer_norm(u_raw):
    """LN0: 零均值单位方差 (分类头 LayerNorm 的归一化部分, eps=EPS_LN)"""
    return (u_raw - u_raw.mean(-1, keepdim=True)) / torch.sqrt(u_raw.var(-1, unbiased=False, keepdim=True) + EPS_LN)


def head_logits(u_norm, hp):
    """hp: 分类头的参数字典 (键为 '1.weight' 等)。h(u) = (W⊙γ)·u + W·β + b"""
    W, gam, beta, b = hp["2.weight"].double(), hp["1.weight"].double(), hp["1.bias"].double(), hp["2.bias"].double()
    return u_norm @ (W * gam[None, :]).T + W @ beta + b


def compute_logits(z, head_params, E):
    """z: (N, 768) 已归一化的 CLIP 视觉嵌入 → logits (N, C)"""
    return head_logits(layer_norm(z.double() @ E.T), head_params["classifier"])


def top_comps_class_avg(logits, labels, n_classes, k=TOP_K):
    """类别平均 logits 的 top-k 竞争类 (k 不超过 n_classes − 1)"""
    k = min(k, n_classes - 1)
    comps = {}
    for y in range(n_classes):
        mask = labels == y
        if mask.sum() == 0:
            continue
        avg_logit = logits[mask].mean(0)
        avg_logit[y] = float("-inf")
        comps[y] = torch.topk(avg_logit, k).indices.tolist()
    return comps


def top_comps_per_sample(logits, labels, k=TOP_K):
    """逐样本干净 logits 的 top-k 竞争类: list of list (k 不超过类别数 − 1)"""
    k = min(k, logits.size(1) - 1)
    out = []
    for si in range(logits.size(0)):
        logit = logits[si].clone()
        logit[int(labels[si])] = float("-inf")
        out.append(torch.topk(logit, k).indices.tolist())
    return out


def competitor_index(labels, scheme, fixed_comps, own_comps_avg, own_comps_sample):
    """逐样本的竞争类下标 (N, TOP_K) 与有效掩码。某类不足 TOP_K 个竞争类时, 空位补真实类并由掩码排除。"""
    N = labels.size(0)
    kidx = labels[:, None].expand(N, TOP_K).clone()
    kmask = torch.zeros(N, TOP_K, dtype=torch.bool)
    for si in range(N):
        y = int(labels[si])
        if scheme == "own_sample":
            ks = own_comps_sample[si]
        else:
            ks = (fixed_comps if scheme == "fixed" else own_comps_avg).get(y, [])
        ks = [int(k) for k in ks if k != y][:TOP_K]
        if ks:
            kidx[si, :len(ks)] = torch.tensor(ks, dtype=torch.long)
            kmask[si, :len(ks)] = True
    return kidx, kmask


def new_acc(C):
    """以 y*C+k 为键的累加器: 各量在 (样本, 竞争类, 扰动) 上求和, n 为次数"""
    return {name: torch.zeros(C * C, dtype=torch.float64) for name in ("damage", "margin", "pert", "svis", "n")}


def accumulate(acc, clean_logits, pert_logits, du_vec, du_str, labels, kidx, kmask, A):
    """把一个 (头, 扰动) 的逐样本竞争类统计累加进 acc。
    damage = m(干净) − m(扰动) (正值 = 扰动使正确类间隔缩小); margin = m(干净);
    pert = ‖Δu‖; S_vis = |(A_y − A_k)ᵀ Δu| / (‖Δu‖ + 1e-12)。"""
    N, C = clean_logits.shape
    clean_logits, pert_logits, du_vec, du_str, A = (t.to(torch.float64) for t in (clean_logits, pert_logits, du_vec, du_str, A))
    rows = torch.arange(N)
    mc = clean_logits[rows, labels][:, None] - clean_logits.gather(1, kidx)
    mp = pert_logits[rows, labels][:, None] - pert_logits.gather(1, kidx)
    svis = ((A[labels][:, None, :] - A[kidx]) * du_vec[:, None, :]).sum(-1).abs() / (du_str[:, None] + 1e-12)
    vals = {"damage": mc - mp, "margin": mc, "pert": du_str[:, None].expand(N, TOP_K), "svis": svis}
    key = (labels[:, None] * C + kidx)[kmask]
    for name, v in vals.items():
        acc[name].index_add_(0, key, v[kmask])
    acc["n"].index_add_(0, key, torch.ones(key.numel(), dtype=torch.float64))


def pair_table(acc, C, A, Hm, pair_qtl):
    """类别对层面 (§3t 的"按 seed 聚合"): 每个出现过的 (y, k) 一行"""
    n = acc["n"]
    idx = torch.nonzero(n > 0).flatten()
    ys, ks = (idx // C).tolist(), (idx % C).tolist()
    Wd = A[idx // C] - A[idx % C]
    return {"y": np.array(ys), "k": np.array(ks),
            "avg_damage": (acc["damage"][idx] / n[idx]).numpy(),
            "avg_clean_margin": (acc["margin"][idx] / n[idx]).numpy(),
            "avg_pert_strength": (acc["pert"][idx] / n[idx]).numpy(),
            "avg_svis": (acc["svis"][idx] / n[idx]).numpy(),
            "pcr_quantile": np.array([pair_qtl[(a, b)] for a, b in zip(ys, ks)]),
            "w_norm_proj": (Wd @ Hm).norm(dim=-1).numpy(),
            "w_norm_raw": Wd.norm(dim=-1).numpy(),
            "n_obs": n[idx].numpy()}


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def ols(X, yv):
    """返回 (R², 系数, 系数的 OLS t 统计量)。R² 分母的 1e-12 保护与此前实现一致。"""
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    r2 = 1 - np.sum(resid ** 2) / (np.sum((yv - yv.mean()) ** 2) + 1e-12)
    sigma2 = float(resid @ resid) / max(X.shape[0] - X.shape[1], 1)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    return float(r2), beta, beta / np.sqrt(np.diag(cov) + 1e-300)


def _fit_fast(Z, yv):
    """同一对回归的正规方程版 (自助法内部用; 与 lstsq 版的一致性见自测 S9)。
    Z 的最后一列是 PCR, 其余列是基线模型。返回 (ΔR², PCR 系数)。"""
    def one(X):
        try:
            b = np.linalg.solve(X.T @ X, X.T @ yv)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(X, yv, rcond=None)[0]
        rss = float(((yv - X @ b) ** 2).sum())
        return 1 - rss / (float(((yv - yv.mean()) ** 2).sum()) + 1e-12), b
    r2_1, _ = one(Z[:, :-1])
    r2_2, b = one(Z)
    return r2_2 - r2_1, float(b[-1])


def cluster_bootstrap(Z, yv, clusters, B, seed):
    """按锚定类别 y 整群重抽样 (有放回抽类别, 取其全部类别对), 重复 B 次。
    返回 ΔR² 与 PCR 系数的 2.5% / 97.5% 百分位区间。类别对共享分类头参数, 区间只作指示。"""
    clu = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in clu]
    rng = np.random.default_rng(int(seed))
    d_ls, c_ls = [], []
    for _ in range(int(B)):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        d_, c_ = _fit_fast(Z[rows], yv[rows])
        d_ls.append(d_)
        c_ls.append(c_)
    d_arr, c_arr = np.array(d_ls), np.array(c_ls)
    return {"B": int(B), "seed": int(seed),
            "dr2_ci": [float(np.percentile(d_arr, 2.5)), float(np.percentile(d_arr, 97.5))],
            "coef_ci": [float(np.percentile(c_arr, 2.5)), float(np.percentile(c_arr, 97.5))],
            "boot_frac_coef_le_0": float((c_arr <= 0).mean())}


def regress(tab, norm_key, B, seed):
    """§3t 的分层回归。模型 1: ⟨D⟩ ~ ‖w‖ + ⟨m⟩ + ‖Δu‖; 模型 2: 再加 PCR (经验分位)。各列标准化。"""
    yv = zscore(tab["avg_damage"])
    Z = np.column_stack([np.ones(len(yv)), zscore(tab[norm_key]), zscore(tab["avg_clean_margin"]),
                         zscore(tab["avg_pert_strength"]), zscore(tab["pcr_quantile"])])
    r2_1, _, _ = ols(Z[:, :4], yv)
    r2_2, beta, t = ols(Z, yv)
    out = {"delta_r2": float(r2_2 - r2_1), "pcr_coef": float(beta[-1]), "coef_t": float(t[-1]),
           "r2_base": r2_1, "r2_full": r2_2, "n_pairs": int(len(yv)),
           "avg_svis": float(tab["avg_svis"].mean()), "avg_pert_strength": float(tab["avg_pert_strength"].mean())}
    if B > 0:
        out["boot"] = cluster_bootstrap(Z, yv, tab["y"], B, seed)
    return out


def cache_consistency(new, old):
    """增强缓存的干净特征 (预计算时已 L2 归一化) 对训练缓存的逐样本一致性, 全部样本。
    训练缓存存的是编码器的原始输出 —— **未归一化**, 归一化在模型 forward 里按 float32 做
    (model/cbm_models.py: encode_image(...).float() / norm)。所以先把训练缓存按同样的方式归一化再比;
    直接比原始张量只会得到尺度差 (行范数 16~21 对 1)。相对误差 = ‖new − old‖₂ / ‖old‖₂ (逐样本, 归一化后)。
    外审第八轮: 不用"逐位一致"这类笼统说法, 直接报 torch.equal / 最大绝对误差 / 最大相对误差。"""
    nf, of = new.float(), old.float()
    old_norm = of.norm(dim=-1, keepdim=True)
    out = {"n_new": int(new.size(0)), "n_old": int(old.size(0)),
           "shape_equal": tuple(new.shape) == tuple(old.shape),
           "dtype_new": str(new.dtype), "dtype_old": str(old.dtype),
           "old_is_l2_normalized": bool((old_norm - 1).abs().max() < 1e-2),
           "old_row_norm_min": float(old_norm.min()), "old_row_norm_max": float(old_norm.max()),
           "new_row_norm_min": float(nf.norm(dim=-1).min()), "new_row_norm_max": float(nf.norm(dim=-1).max())}
    if not out["shape_equal"]:
        return out
    ref = of if out["old_is_l2_normalized"] else of / old_norm   # 与模型 forward 相同: float32 下 L2 归一化 (已归一化则不重复做)
    torch_equal_raw, torch_equal_norm = bool(torch.equal(new, old)), bool(torch.equal(nf.half(), ref.half()))
    diff = (nf - ref).abs()
    row_err, row_norm = diff.norm(dim=-1), ref.norm(dim=-1)
    rel = row_err / row_norm.clamp_min(1e-12)
    cos = (nf * ref).sum(-1) / (nf.norm(dim=-1) * row_norm).clamp_min(1e-12)
    diff16 = (nf.half() - ref.half()).abs()                        # 两边都舍入到存储精度 (fp16) 再比
    out.update({"torch_equal": torch_equal_norm, "torch_equal_raw_tensors": torch_equal_raw,
                "n_equal_rows": int((row_err == 0).sum()),
                "frac_equal_elements": float((diff == 0).double().mean()),
                "frac_equal_elements_fp16": float((diff16 == 0).double().mean()),
                "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()),
                "max_rel_error_l2": float(rel.max()), "mean_rel_error_l2": float(rel.mean()),
                "median_rel_error_l2": float(rel.median()),
                "n_rows_rel_gt_1e-2": int((rel > 1e-2).sum()), "n_rows_rel_gt_1e-3": int((rel > 1e-3).sum()),
                "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                "max_norm_dev_new": float((nf.norm(dim=-1) - 1).abs().max())})
    return out


def feature_source_impact(heads, z_new, z_old, E, labels):
    """同一批分类头分别用两份干净特征评估 (其余计算完全相同), 差异有多大 —— 缓存不一致对结论的可见影响的上界。"""
    u = {"new": layer_norm(z_new.double() @ E.T), "old": layer_norm(z_old.double() @ E.T)}
    out = {"max_abs_acc_diff": 0.0, "max_abs_ce_diff": 0.0, "max_abs_logit_diff": 0.0, "n_heads": 0, "n_heads_with_acc_diff": 0}
    for h in heads.values():
        lg = {k: head_logits(v, h["classifier"]).detach() for k, v in u.items()}
        acc = {k: float((v.argmax(1) == labels).double().mean()) for k, v in lg.items()}
        ce = {k: float(torch.nn.functional.cross_entropy(v, labels)) for k, v in lg.items()}
        out["max_abs_acc_diff"] = max(out["max_abs_acc_diff"], abs(acc["new"] - acc["old"]))
        out["max_abs_ce_diff"] = max(out["max_abs_ce_diff"], abs(ce["new"] - ce["old"]))
        out["max_abs_logit_diff"] = max(out["max_abs_logit_diff"], float((lg["new"] - lg["old"]).abs().max()))
        out["n_heads"] += 1
        out["n_heads_with_acc_diff"] += int(acc["new"] != acc["old"])
    return out


def layer2_verdict(res, scheme):
    """§3t 的判定, 逐 seed 判断后合并 (外审第八轮第 5 项)。
    支持脆弱抵消: Full 的每个 seed 都有 ΔR² > 0.05 且 PCR 系数 < 0, 且 CE-Only 不满足;
    不支持: Full 的 ΔR² 全部 < 0.01, 或系数并非全部为负, 或在 CE-Only 上同样成立; 其余为不确定。"""
    def per_seed(tag, col):
        return [p.get(col, 0.0) for p in res["arms"].get(tag, {}).get("per_seed", [])]
    d_col, c_col = f"delta_r2_{scheme}", f"pcr_coef_{scheme}"
    full_dr2 = {t: per_seed(t, d_col) for t in FULL_ARMS}
    full_coef = {t: per_seed(t, c_col) for t in FULL_ARMS}
    ce_dr2, ce_coef = per_seed(CE_ARM, d_col), per_seed(CE_ARM, c_col)
    full_ok = {t: [d > B2_DELTAR2_SUPPORT and c < 0 for d, c in zip(full_dr2[t], full_coef[t])] for t in FULL_ARMS}
    ce_ok = [d > B2_DELTAR2_SUPPORT and c < 0 for d, c in zip(ce_dr2, ce_coef)]
    full_holds = bool(full_ok) and all(len(v) > 0 and all(v) for v in full_ok.values())
    ce_holds = bool(ce_ok) and all(ce_ok)
    if full_holds and not ce_holds:
        label = "支持脆弱抵消"
    elif (all(d < B2_DELTAR2_NONE for v in full_dr2.values() for d in v)
          or not all(c < 0 for v in full_coef.values() for c in v) or (full_holds and ce_holds)):
        label = "初步证据不支持（探索性）"
    else:
        label = "不确定（ΔR² 介于 0.01~0.05 之间或各 seed 方向不一致；探索性）"
    return {"scheme": scheme, "norm": "proj", "label": label, "full_holds": full_holds, "ce_holds": ce_holds,
            "full_dr2_per_seed": full_dr2, "full_coef_per_seed": full_coef,
            "ce_dr2_per_seed": ce_dr2, "ce_coef_per_seed": ce_coef,
            "full_support_per_seed": full_ok, "ce_support_per_seed": ce_ok}


# ---------------- 报告 ----------------
def _ms(agg, key, nd=4):
    if key not in agg:
        return "—"
    m, s = agg[key]
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def _range(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if hi - lo < 10 ** (-nd) / 2 else f"{lo:.{nd}f} ~ {hi:.{nd}f}"


def _ci_union(ps, key, ci_key, nd=4):
    """各 seed 的 95% 区间 (ps[i][key][ci_key] = [下界, 上界]) 的并集 [最小下界, 最大上界]"""
    cs = [p[key][ci_key] for p in ps if key in p]
    if not cs:
        return "—"
    return f"[{min(c[0] for c in cs):.{nd}f}, {max(c[1] for c in cs):.{nd}f}]"


def write_report(res):
    """由 results (即 stage0_layer2_cub.json 的内容) 生成 md 报告。"""
    arm = lambda tag: res["arms"].get(tag, {})                        # noqa: E731
    seeds_of = lambda tag: arm(tag).get("per_seed", [])               # noqa: E731
    agg_of = lambda tag: arm(tag).get("agg", {})                     # noqa: E731
    code = res.get("code") or {}
    L = []
    add = L.append
    add(f"# Stage 0B 第二层 + 0D（CUB，heldout_val {res['n_val']} 样本，{res['created'][:10]}）")
    add("")
    add("预注册见 `RESULTS.md` §3t；判定阈值 ΔR² > 0.05 / < 0.01 未改动。**探索性诊断**：这些分类头是用完整训练集"
        "（含 heldout_val 的样本）训练的，下面的评估是训练集内的，不作为正式选参依据（正式结论需要 train_core 重训的头）。")
    add("")
    add(f"- 代码版本：`{str(code.get('commit', '?'))[:7]}`（dirty={code.get('dirty', '?')}）；"
        f"运行命令：`python stage0_layer2_diagnose.py --bootstrap {res['bootstrap_B']}`")
    add(f"- 数据：增强缓存 `{AUG_CACHE}`（sha256 前 16 位 `{res['aug_cache_sha256_16']}`，"
        f"与元数据记录一致 = {res['aug_cache_sha256_matches_meta']}）；分类头 `logs/heads/CUB_s0_*_seed1..3.pt` 共 {res['n_heads']} 个")
    add(f"- 回归的控制变量：{NORMS['proj']}（与 PCR 所用的决策方向一致）；{NORMS['raw']} 作为敏感性分析")
    add("- 本文件由脚本从同一份 `stage0_layer2_cub.json` 生成，数字不经手工录入")
    add("")

    # ---- 0D ----
    add("## 0D：源域增强验证（7 种扰动，3 seeds 均值）")
    add("")
    add("| 臂 | 干净 heldout_val | 7 种扰动均值 Acc | 相对 CE-Only（干净） | 相对 CE-Only（增强） | 7 种扰动均值 CE |")
    add("|---|---|---|---|---|---|")
    c_clean, c_aug = agg_of(CE_ARM)["clean_acc"][0], agg_of(CE_ARM)["avg_aug_acc"][0]
    for tag in OD_ARMS:
        cl, au = agg_of(tag)["clean_acc"][0], agg_of(tag)["avg_aug_acc"][0]
        rel = "— | —" if tag == CE_ARM else f"{100 * (cl - c_clean):+.2f} | {100 * (au - c_aug):+.2f}"
        add(f"| {arm(tag)['name']} | {100 * cl:.2f}% | {100 * au:.2f}% | {rel} | {agg_of(tag)['avg_aug_ce'][0]:.3f} |")
    add("")
    acc_opt, ce_opt = res["verdict_0d"]["acc_optimal"], res["verdict_0d"]["ce_optimal"]
    show = lambda x_: {"a1": "1", "a2": "2"}.get(x_, "无法判定")      # noqa: E731
    add(f"- 按增强 Acc 选出的剂量：α = {show(acc_opt)}；按增强 CE 选出的剂量：α = {show(ce_opt)}"
        + ("。**两者不一致，必须同时报告。**" if acc_opt != ce_opt else "。"))
    add("")
    add("各扰动的逐项 Acc（%，3 seeds 均值）：")
    add("")
    names = res["aug_names"]
    add("| 臂 | " + " | ".join(names) + " |")
    add("|---|" + "---|" * len(names))
    for tag, _ in ARMS:
        rows = seeds_of(tag)
        if rows:
            accs = [100 * float(np.mean([r["aug_accs"][an] for r in rows])) for an in names]
            add(f"| {arm(tag)['name']} | " + " | ".join(f"{a_:.1f}" for a_ in accs) + " |")
    add("")

    # ---- 0B 第二层 ----
    add("## 0B 第二层：PCR 对间隔损害的增量解释力（ΔR²，3 seeds 的均值 ± 标准差）")
    add("")
    add("| 臂 | ΔR²（固定竞争类） | PCR 系数 | ΔR²（自身类别平均） | ΔR²（逐样本自身） | ΔR²（未投影 ‖w‖，敏感性） | 配对数 |")
    add("|---|---|---|---|---|---|---|")
    for tag, _ in ARMS:
        rows = seeds_of(tag)
        if rows:
            g = agg_of(tag)
            add(f"| {arm(tag)['name']} | {_ms(g, 'delta_r2_fixed')} | {_ms(g, 'pcr_coef_fixed', 3)} | "
                f"{_ms(g, 'delta_r2_own_avg')} | {_ms(g, 'delta_r2_own_sample')} | "
                f"{_ms(g, 'delta_r2_rawnorm_fixed')} | {_range([r.get('n_pairs_fixed') for r in rows], 0)} |")
    add("")
    add("显著性（指示性：类别对共享分类头参数，不是独立样本；区间为按锚定类别整群自助法的 95% 百分位区间，"
        f"B = {res['bootstrap_B']}，表中取各 seed 区间的并集）：")
    add("")
    add("| 臂 | ΔR² 的 95% 区间 | PCR 系数的 95% 区间 | PCR 系数的 t（各 seed 范围） | 重抽样中系数 ≤ 0 的比例（各 seed 最大） |")
    add("|---|---|---|---|---|")
    for tag, _ in ARMS:
        rows = seeds_of(tag)
        if rows and "boot_fixed" in rows[0]:
            add(f"| {arm(tag)['name']} | {_ci_union(rows, 'boot_fixed', 'dr2_ci')} | {_ci_union(rows, 'boot_fixed', 'coef_ci', 3)} | "
                f"{_range([r.get('coef_t_fixed') for r in rows])} | "
                f"{max(r['boot_fixed']['boot_frac_coef_le_0'] for r in rows):.3f} |")
    add("")

    # ---- 判定 ----
    v = res["verdict_0b_layer2"]
    add(f"## 判定（{SCHEMES[v['scheme']]}；{NORMS['proj']}）：**{v['label']}**")
    add("")
    add("§3t 的规则：Full DDO 的每个 seed 都有 ΔR² > 0.05 且 PCR 系数 < 0（系数为负 = 低 PCR 的边界损害更大）才算\"支持\"，"
        "且 CE-Only 不能同样成立；ΔR² 全部 < 0.01、或系数并非全部为负、或 CE-Only 同样成立，为\"不支持\"；其余为\"不确定\"。")
    add("")
    d_col, c_col = "delta_r2_" + v["scheme"], "pcr_coef_" + v["scheme"]
    add("| 臂 | seed | ΔR² | PCR 系数 | ΔR²（逐样本自身） | 满足单 seed 的\"支持\"条件 |")
    add("|---|---|---|---|---|---|")
    for tag in FULL_ARMS + (CE_ARM,):
        flags = v["full_support_per_seed"].get(tag, v["ce_support_per_seed"])
        for r, ok_ in zip(seeds_of(tag), flags):
            add(f"| {arm(tag)['name']} | {r['seed']} | {r[d_col]:.4f} | {r[c_col]:+.3f} | "
                f"{r.get('delta_r2_own_sample', float('nan')):.4f} | {'是' if ok_ else '否'} |")
    add("")
    add(f"- Full DDO 两个剂量的全部 seed 都满足：{v['full_holds']}；CE-Only 的全部 seed 都满足：{v['ce_holds']}")
    for other, label in res.get("verdict_0b_layer2_sensitivity", {}).items():
        add(f"- 换竞争类口径后按同一规则的判定（{SCHEMES[other]}）：{label}")
    add("")

    # ---- 缓存一致性 ----
    cc = res.get("cache_consistency", {})
    nan = float("nan")
    add("## 增强缓存的干净特征与训练缓存的一致性（全部样本）")
    add("")
    add(f"`{AUG_CACHE}` 的 clean_feats 对 `{TRAIN_CACHE}` 的 feats（{cc.get('n_new')} 对 {cc.get('n_old')} 个样本，"
        f"{cc.get('dtype_new')} 对 {cc.get('dtype_old')}）。训练缓存存的是编码器的原始输出——未归一化，行范数 "
        f"{cc.get('old_row_norm_min', nan):.1f}~{cc.get('old_row_norm_max', nan):.1f}，归一化在模型 forward 里按 float32 做；"
        "所以下面是把它按同样方式归一化之后再比的：")
    add("")
    add(f"- `torch.equal` = **{cc.get('torch_equal')}**；完全相同的行 {cc.get('n_equal_rows')} / {cc.get('n_new')}；"
        f"逐元素比较：与 float32 参考完全相同的占 {100 * cc.get('frac_equal_elements', 0):.2f}%，"
        f"参考也舍入到 fp16 后相同的占 {100 * cc.get('frac_equal_elements_fp16', 0):.2f}%")
    add(f"- 最大绝对误差 {cc.get('max_abs_error', nan):.3e}（均值 {cc.get('mean_abs_error', nan):.3e}）")
    add(f"- 最大相对误差 ‖new − old‖₂ / ‖old‖₂ = {cc.get('max_rel_error_l2', nan):.3e}"
        f"（中位数 {cc.get('median_rel_error_l2', nan):.3e}；超过 1e-2 的样本 {cc.get('n_rows_rel_gt_1e-2')} 个，超过 1e-3 的 "
        f"{cc.get('n_rows_rel_gt_1e-3')} 个）；逐样本余弦最小 {cc.get('min_cosine', nan):.7f}")
    hi = cc.get("head_impact", {})
    add(f"- 标签逐项一致：{cc.get('labels_equal')}；新特征的范数偏离 1 的最大值 {cc.get('max_norm_dev_new', nan):.2e}")
    add(f"- **对结论的可见影响**：{hi.get('n_heads')} 个分类头分别用两份干净特征在 heldout_val 上评估（其余计算完全相同），"
        f"Acc 最大相差 {100 * hi.get('max_abs_acc_diff', nan):.2f} 个百分点（{hi.get('n_heads_with_acc_diff')} 个头有差异），"
        f"CE 最大相差 {hi.get('max_abs_ce_diff', nan):.4f}，logit 最大绝对差 {hi.get('max_abs_logit_diff', nan):.3f}")
    add("")
    add("## 读法与边界")
    add("")
    add("- ΔR² 是在控制权重范数、干净间隔、‖Δu‖ 之后，PCR（经验分位）对间隔损害的增量解释力；PCR 系数为负才符合\"脆弱抵消\""
        "（低 PCR → 更大间隔损害）。")
    add("- ΔR² 在零假设下的期望约为 1/n（多一个参数带来的向上偏差，n ≈ 1000 时约 0.001），区间下界略高于 0 不构成证据；"
        "看系数的符号与区间。")
    add("- 判定按 seed 逐个判断后合并；三个 seed 的系数同号只说明对训练随机性稳定，不代表对数据扰动的稳健性。")
    add("- 扰动是训练集图像上的光度变换（在 LN 之后、affine 之前的概念空间上比较），未参考任何目标域结果。")
    add("- 无论结果如何都不下 \"Shrinkage 有效\" 的结论 —— 那需要干预实验（第三层）。")
    add("")
    return "\n".join(L)


def self_test():
    """合成数据: (a) 已知 PCR-损害关系时回归与判定逻辑正确; (b) 向量化累加与逐样本循环一致;
    (c) t 统计量、自助法、缓存一致性、判定分支、报告生成各自可用。不需要任何数据文件。"""
    torch.manual_seed(0)
    np.random.seed(0)
    K, C, N, n_pert = 40, 20, 300, 3
    sens = [3, 7, 11]
    tt = torch.randn(N, 1)
    U = torch.randn(N, K) * 0.05
    U[:, sens] = tt * 2.0 + torch.randn(N, len(sens)) * 0.4
    U = (U - U.mean(-1, keepdim=True)).double()
    G, _, _, _ = second_moments(U)
    Dt = diag_reference(G)
    _, samples = random_baseline(G, n=10000, seed=0)
    Hm = torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + (f"  [{detail}]" if detail else ""))

    # 构造一个头: 一半类别用抵消型权重 (PCR 低), 一半不用
    W = torch.randn(C, K, dtype=torch.float64) * 0.5
    W[: C // 2, 3] = 6.0; W[: C // 2, 7] = -6.0
    gam = torch.ones(K, dtype=torch.float64)
    A = W * gam[None, :]
    y_i, k_i = torch.triu_indices(C, C, offset=1)
    Wk = (A[y_i] - A[k_i]) @ Hm
    pcr = quad(Wk, G) / quad(Wk, Dt).clamp_min(1e-300)
    qtl = pcr_quantile(pcr, samples)
    pair_qtl = {}
    for pi in range(len(y_i)):
        pair_qtl[(int(y_i[pi]), int(k_i[pi]))] = float(qtl[pi])
        pair_qtl[(int(k_i[pi]), int(y_i[pi]))] = float(qtl[pi])

    # 合成 logits 与概念扰动: 低 PCR 的对间隔损害更大 (模拟脆弱抵消)
    labels = torch.randint(0, C, (N,))
    clean_lg = torch.randn(N, C, dtype=torch.float64) * 2
    clean_lg[torch.arange(N), labels] += 1.5
    comps = top_comps_class_avg(clean_lg, labels, C)
    pert = []
    for _ in range(n_pert):
        du = torch.randn(N, K, dtype=torch.float64) * 0.1
        pl = clean_lg + torch.randn(N, C, dtype=torch.float64) * 0.3
        for si in range(N):
            y = int(labels[si])
            for k in comps[y]:
                pl[si, k] += (1 - pair_qtl[(y, k)]) * 0.5          # PCR 分位越低, 竞争类 logit 抬得越高 → 该类别对的间隔损害越大
        pert.append((pl, du, du.norm(dim=-1)))

    # ---- (b) 向量化累加 == 逐样本显式循环 ----
    kidx, kmask = competitor_index(labels, "fixed", comps, {}, [])
    acc = new_acc(C)
    ref = {}
    for pl, du, dn in pert:
        accumulate(acc, clean_lg, pl, du, dn, labels, kidx, kmask, A)
        for si in range(N):
            y = int(labels[si])
            for k in comps[y]:
                key = (y, k)
                r_ = ref.setdefault(key, {"damage": 0.0, "margin": 0.0, "pert": 0.0, "svis": 0.0, "n": 0})
                r_["damage"] += float(clean_lg[si, y] - clean_lg[si, k]) - float(pl[si, y] - pl[si, k])
                r_["margin"] += float(clean_lg[si, y] - clean_lg[si, k])
                r_["pert"] += float(dn[si])
                r_["svis"] += float(((A[y] - A[k]) @ du[si]).abs() / (dn[si] + 1e-12))
                r_["n"] += 1
    worst = max(abs(float(acc[nm][y * C + k]) - r_[nm]) for (y, k), r_ in ref.items() for nm in r_)
    check("S7 向量化累加与逐样本显式循环一致 (全部键的全部量)", worst < 1e-9, f"最大差 {worst:.2e}")
    lab4 = torch.randint(0, 4, (40,))                              # 只有 4 个类: 每类最多 3 个竞争类, 不足 TOP_K
    lg4 = torch.randn(40, 4, dtype=torch.float64)
    comp4 = top_comps_class_avg(lg4, lab4, 4)
    ki4, km4 = competitor_index(lab4, "fixed", comp4, {}, [])
    ac4 = new_acc(4)
    accumulate(ac4, lg4, lg4 + 1, torch.zeros(40, K, dtype=torch.float64), torch.zeros(40), lab4, ki4, km4, A[:4])
    check("S7b 不足 TOP_K 的空位由掩码排除 (次数 = 样本 × 实际竞争类数, 真实类位置计 0)",
          float(ac4["n"].sum()) == 40 * 3 and int(km4.sum()) == 40 * 3
          and float(ac4["n"][torch.arange(4) * 5].sum()) == 0.0)
    tab = pair_table(acc, C, A, Hm, pair_qtl)
    check("S7c 类别对表的行数 = 出现过的 (y, k) 数", len(tab["y"]) == len(ref) and int(tab["n_obs"].sum()) == N * n_pert * TOP_K)

    # ---- (a) 回归 ----
    fit_p = regress(tab, "w_norm_proj", B=300, seed=0)
    fit_r = regress(tab, "w_norm_raw", B=0, seed=0)
    check("S1 回归: PCR 系数为负 (合成数据中低 PCR → 高损害)", fit_p["pcr_coef"] < 0, f"coef = {fit_p['pcr_coef']:.3f}")
    check("S2 回归: ΔR² > 0.2 (合成信号强: 控制权重范数等之后 PCR 仍有增量解释力)", fit_p["delta_r2"] > 0.2, f"ΔR² = {fit_p['delta_r2']:.3f}")
    # 合成数据里低 PCR 的配对恰好是权重大的抵消类, 权重范数本身与损害强相关 —— 基线模型先解释一部分,
    # 所以 ΔR² 低于 PCR 的单变量解释力。两层含义分开验证。
    r_q = float(np.corrcoef(tab["pcr_quantile"], tab["avg_damage"])[0, 1])
    check("S2b PCR 单变量的解释力高 (R² > 0.7), 且基线模型确实先解释了一部分 (0 < R² 基线 < 0.8)",
          r_q ** 2 > 0.7 and 0 < fit_p["r2_base"] < 0.8, f"R² = {r_q ** 2:.3f}, R² 基线 = {fit_p['r2_base']:.3f}")
    check("S3 S_vis 已计算且非退化", 0 < fit_p["avg_svis"] < 10 and np.isfinite(tab["avg_svis"]).all())
    check("S4 投影范数 ≤ 未投影范数", bool((tab["w_norm_proj"] <= tab["w_norm_raw"] + 1e-12).all()))
    check("S5 逐对扰动强度有变异 (非全局常数)", tab["avg_pert_strength"].std() > 0)
    check("S9 正规方程版与 lstsq 版一致 (ΔR² 与系数, 容差 1e-9)",
          abs(_fit_fast(np.column_stack([np.ones(len(tab["y"])), zscore(tab["w_norm_proj"]), zscore(tab["avg_clean_margin"]),
                                         zscore(tab["avg_pert_strength"]), zscore(tab["pcr_quantile"])]),
                        zscore(tab["avg_damage"]))[0] - fit_p["delta_r2"]) < 1e-9)
    x = np.random.default_rng(1).normal(size=500)
    yv_ = 0.3 * x + np.random.default_rng(2).normal(size=500)
    r1 = np.corrcoef(x, yv_)[0, 1]
    t_closed = r1 * np.sqrt((len(x) - 2) / (1 - r1 ** 2))
    _, _, tGot = ols(np.column_stack([np.ones(len(x)), x]), yv_)
    check("S9b OLS t 统计量与一元回归的闭式解一致", abs(tGot[-1] - t_closed) < 1e-8, f"{tGot[-1]:.6f} vs {t_closed:.6f}")
    b1 = cluster_bootstrap(np.column_stack([np.ones(len(x)), x]), yv_, np.arange(len(x)) // 5, B=400, seed=3)
    b2 = cluster_bootstrap(np.column_stack([np.ones(len(x)), x]), yv_, np.arange(len(x)) // 5, B=400, seed=3)
    check("S10 整群自助法可复现 (同 seed 同区间)", b1 == b2, f"ΔR² 区间 [{b1['dr2_ci'][0]:.4f}, {b1['dr2_ci'][1]:.4f}]")
    d_point = _fit_fast(np.column_stack([np.ones(len(x)), x]), yv_)[0]
    check("S10a 自助法区间包住点估计, 且不含 0 (x 与 y 确实相关)", b1["dr2_ci"][0] <= d_point <= b1["dr2_ci"][1] and b1["dr2_ci"][0] > 0,
          f"点估计 {d_point:.4f}")
    check("S10b 合成强信号下 ΔR² 的 95% 区间下界 > 0", fit_p["boot"]["dr2_ci"][0] > 0,
          f"[{fit_p['boot']['dr2_ci'][0]:.3f}, {fit_p['boot']['dr2_ci'][1]:.3f}], 系数区间 "
          f"[{fit_p['boot']['coef_ci'][0]:.3f}, {fit_p['boot']['coef_ci'][1]:.3f}]")

    # ---- (c) 判定分支 ----
    def fake(full, ce):
        return {"arms": {t: {"per_seed": [{"delta_r2_fixed": d, "pcr_coef_fixed": c} for d, c in v]}
                         for t, v in list(zip(FULL_ARMS, full)) + [(CE_ARM, ce)]}}
    big_neg = [(0.08, -0.2)] * 3
    check("S11a Full 全部 seed 满足且 CE 不满足 → 支持", layer2_verdict(fake([big_neg, big_neg], [(0.001, 0.01)] * 3), "fixed")["label"] == "支持脆弱抵消")
    check("S11b ΔR² 全部 < 0.01 → 不支持", layer2_verdict(fake([[(0.004, -0.2)] * 3, [(0.003, -0.1)] * 3], [(0.001, 0.0)] * 3), "fixed")["label"].startswith("初步证据不支持"))
    check("S11c 系数全部为正 (方向与假说相反) → 不支持", layer2_verdict(fake([[(0.02, 0.12)] * 3, [(0.02, 0.1)] * 3], [(0.001, 0.03)] * 3), "fixed")["label"].startswith("初步证据不支持"))
    check("S11d ΔR² 介于 0.01~0.05 且系数为负 → 不确定", layer2_verdict(fake([[(0.03, -0.1)] * 3, [(0.03, -0.1)] * 3], [(0.001, 0.0)] * 3), "fixed")["label"].startswith("不确定"))
    check("S11e CE-Only 上同样成立 → 不支持 (不是 DDO 特有)", layer2_verdict(fake([big_neg, big_neg], big_neg), "fixed")["label"].startswith("初步证据不支持"))
    check("S11f 一个 seed 方向相反 → 不满足逐 seed 一致 → 不支持", layer2_verdict(fake([big_neg, big_neg[:2] + [(0.08, 0.2)]], [(0.001, 0.0)] * 3), "fixed")["label"].startswith("初步证据不支持"))

    # ---- (c) 缓存一致性 ----
    raw = (torch.randn(50, 8) * 20).half()                             # 训练缓存的存法: 未归一化的 fp16
    a_ = (raw.float() / raw.float().norm(dim=-1, keepdim=True)).half()   # 增强缓存的存法: float32 归一化后存 fp16
    same = cache_consistency(a_, raw)
    check("S13a 同一批特征、两种存法: fp16 下 torch.equal = True, 差别只有 fp16 舍入 (尺度差不计入)",
          same["torch_equal"] and same["max_abs_error"] < 1e-2 and same["old_is_l2_normalized"] is False
          and same["torch_equal_raw_tensors"] is False)
    sc = cache_consistency(a_, a_.clone())
    check("S13d 完全相同的已归一化张量: 误差为 0", sc["old_is_l2_normalized"] and sc["torch_equal"]
          and sc["max_abs_error"] == 0.0 and sc["max_rel_error_l2"] == 0.0)
    b_ = a_.clone(); b_[3, 2] += 0.5
    diff_ = cache_consistency(b_, a_)
    check("S13b 新特征改动一个元素: torch.equal = False, 最大绝对误差 = 0.5, 相同行 49/50",
          (not diff_["torch_equal"]) and abs(diff_["max_abs_error"] - 0.5) < 1e-6 and diff_["n_equal_rows"] == 49)
    check("S13c 形状不同时不比较数值", cache_consistency(a_, torch.zeros(10, 8))["shape_equal"] is False
          and "max_abs_error" not in cache_consistency(a_, torch.zeros(10, 8)))
    heads_ = {"h": {"classifier": {"1.weight": torch.ones(K, dtype=torch.float64), "1.bias": torch.zeros(K, dtype=torch.float64),
                                   "2.weight": torch.randn(C, K, dtype=torch.float64), "2.bias": torch.zeros(C, dtype=torch.float64)}}}
    z_ = torch.randn(40, K, dtype=torch.float64)
    lab_ = torch.randint(0, C, (40,))
    imp0 = feature_source_impact(heads_, z_, z_, torch.eye(K, dtype=torch.float64), lab_)
    imp1 = feature_source_impact(heads_, z_, z_ + 0.05 * torch.randn(40, K, dtype=torch.float64), torch.eye(K, dtype=torch.float64), lab_)
    check("S14 两份特征相同时影响量为 0; 特征被扰动时影响量非零",
          imp0["max_abs_logit_diff"] == 0.0 and imp0["n_heads"] == 1 and imp1["max_abs_logit_diff"] > 0 and imp1["max_abs_ce_diff"] > 0)

    # ---- (c) 报告 ----
    res = {"arms": {}, "n_val": N, "created": "2026-11-26 00:00:00", "bootstrap_B": 300, "aug_names": ["p1", "p2", "p3"],
           "code": {"commit": "abcdef1234567890", "dirty": False}, "n_heads": 27, "aug_cache_sha256_16": "0123456789abcdef",
           "aug_cache_sha256_matches_meta": True, "verdict_0b_layer2_sensitivity": {}}
    for tag, name in ARMS:
        per = []
        for s in (1, 2, 3):
            rec = {"seed": s, "clean_acc": 0.84, "clean_ce": 0.6, "avg_aug_acc": 0.6, "avg_aug_ce": 2.1,
                   "aug_accs": {an: 0.5 + 0.03 * s for an in res["aug_names"]}}
            for sch in SCHEMES:
                f_ = fit_p if sch == "fixed" else {**fit_r, "boot": fit_p["boot"]}
                rec.update({f"delta_r2_{sch}": f_["delta_r2"] + 0.001 * s, f"pcr_coef_{sch}": f_["pcr_coef"], f"coef_t_{sch}": f_["coef_t"],
                            f"r2_base_{sch}": f_["r2_base"], f"r2_full_{sch}": f_["r2_full"], f"n_pairs_{sch}": f_["n_pairs"],
                            f"avg_svis_{sch}": f_["avg_svis"], f"avg_pert_strength_{sch}": f_["avg_pert_strength"],
                            f"boot_{sch}": f_["boot"], f"delta_r2_rawnorm_{sch}": fit_r["delta_r2"], f"pcr_coef_rawnorm_{sch}": f_["pcr_coef"]})
            per.append(rec)
        ag = {}
        for k_ in per[0]:
            if k_ != "seed" and all(isinstance(p_.get(k_), float) for p_ in per):
                ag[k_] = (float(np.mean([p_[k_] for p_ in per])), float(np.std([p_[k_] for p_ in per], ddof=1)))
        res["arms"][tag] = {"name": name, "per_seed": per, "agg": ag}
    res["verdict_0b_layer2"] = layer2_verdict(res, "fixed")
    res["verdict_0d"] = {"acc_optimal": "a2", "ce_optimal": "a1"}
    res["cache_consistency"] = cache_consistency(a_, a_)
    res["cache_consistency"]["head_impact"] = imp0
    md = write_report(res)
    check("S12 报告包含 0D / 0B / 判定 / 缓存一致性四节", all(h in md for h in ("## 0D", "## 0B 第二层", "## 判定", "一致性")))
    check("S12b 报告里的数字来自结果 (判定标签出现, 且没有 nan 数值)",
          res["verdict_0b_layer2"]["label"] in md and re.search(r"\bnan\b", md) is None)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return self_test()
    B = int(argv[argv.index("--bootstrap") + 1]) if "--bootstrap" in argv else 1000
    t0 = time.time()
    code = git_info()                                   # 在写任何产物之前取: 工作区是否干净以运行开始时为准
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + (f"  [{detail}]" if detail else ""))

    # ===== 1. 缓存与一致性核对 =====
    print("加载增强缓存与训练缓存...")
    aug = torch.load(AUG_CACHE, map_location="cpu", weights_only=False)
    old = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    with open(AUG_CACHE, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(AUG_META, encoding="utf-8") as f:
        sha_meta = json.load(f).get("cache_sha256")
    cc = cache_consistency(aug["clean_feats"], old["feats"])
    cc["labels_equal"] = bool(aug["labels"].shape == old["labels"].shape and torch.equal(aug["labels"], old["labels"]))
    print(f"  干净特征对训练缓存: torch.equal={cc['torch_equal']}, 最大绝对误差 {cc.get('max_abs_error', float('nan')):.3e}, "
          f"最大相对误差 {cc.get('max_rel_error_l2', float('nan')):.3e}, 标签一致={cc['labels_equal']}")

    val_idx = aug["split"]["heldout_val"]
    n_val = len(val_idx)
    labels_val = aug["labels"][val_idx]
    clean_val = aug["clean_feats"][val_idx].float()
    aug_names = list(aug["aug_feats"].keys())
    aug_val = {name: aug["aug_feats"][name][val_idx].float() for name in aug_names}
    print(f"  评估集: heldout_val {n_val} 个样本 (§3t 写'完整训练集', 实际为 heldout_val)")

    # ===== 2. 概念矩阵与域方向 =====
    print("加载概念矩阵与域方向...")
    from args import get_args
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = __import__("data").get_dataset_classes(args)[0]
    diffs = ds.domain_diffs.double().cpu()
    concept_names = list(ds.concept2id.keys())
    n_classes = len(ds.classname2id)
    import clip
    m, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=args.device)
    with torch.no_grad():
        E = m.encode_text(clip.tokenize(concept_names).to(args.device)).double().cpu()
        E = E / E.norm(dim=-1, keepdim=True)
    del m, ds
    torch.cuda.empty_cache()
    U = domain_directions(diffs, E)
    G, _, _, _ = second_moments(U)
    Dt = diag_reference(G)
    _, base_samples = random_baseline(G, n=40000, seed=0)
    K_dim = E.size(0)
    Hm = torch.eye(K_dim, dtype=torch.float64) - torch.ones(K_dim, K_dim, dtype=torch.float64) / K_dim

    # ===== 3. 分类头 =====
    heads = load_heads()
    check(f"27 个分类头都已加载", len(heads) == 27, f"实际 {len(heads)}")
    old_val = old["feats"][val_idx].float()
    cc["head_impact"] = feature_source_impact(heads, clean_val, old_val / old_val.norm(dim=-1, keepdim=True), E, labels_val)
    print(f"  两份干净特征对 27 个头评估的影响: Acc 最大差 {100 * cc['head_impact']['max_abs_acc_diff']:.2f} 个百分点, "
          f"CE 最大差 {cc['head_impact']['max_abs_ce_diff']:.4f}, logit 最大绝对差 {cc['head_impact']['max_abs_logit_diff']:.3f}")
    ce_head = heads[(CE_ARM, 1)]
    fixed_comps = top_comps_class_avg(compute_logits(clean_val, ce_head, E).detach(), labels_val, n_classes)

    # ===== 4. 逐头 =====
    results = {"arms": {}, "aug_names": aug_names, "n_val": n_val, "n_heads": len(heads),
               "evaluation_set": "heldout_val (1200 samples; §3t 原文写'完整训练集', 实际为 heldout_val)",
               "note": "外审第八轮 P0 修正版 + 同日收尾补正: 控制变量用投影范数 (未投影为敏感性), 补显著性与 md 报告",
               "norm_primary": "proj", "bootstrap_B": B, "code": code, "cache_consistency": cc,
               "aug_cache_sha256_16": sha[:16], "aug_cache_sha256_matches_meta": sha == sha_meta,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    for tag, name in ARMS:
        per_seed = []
        for s in SEEDS:
            h = heads.get((tag, s))
            if h is None:
                continue
            hp = h["classifier"]
            A = hp["2.weight"].double() * hp["1.weight"].double()[None, :]
            y_i, k_i = torch.triu_indices(A.size(0), A.size(0), offset=1)
            Wk = (A[y_i] - A[k_i]) @ Hm
            qtl = pcr_quantile(quad(Wk, G) / quad(Wk, Dt).clamp_min(1e-300), base_samples)
            pair_qtl = {}
            for pi in range(len(y_i)):
                pair_qtl[(int(y_i[pi]), int(k_i[pi]))] = float(qtl[pi])
                pair_qtl[(int(k_i[pi]), int(y_i[pi]))] = float(qtl[pi])

            u_c_n = layer_norm(clean_val.double() @ E.T)
            with torch.no_grad():
                clean_logits = head_logits(u_c_n, hp).detach()
                own_avg = top_comps_class_avg(clean_logits, labels_val, n_classes)
                own_sample = top_comps_per_sample(clean_logits, labels_val)
                rec = {"seed": s,
                       "clean_acc": float((clean_logits.argmax(1) == labels_val).double().mean()),
                       "clean_ce": float(torch.nn.functional.cross_entropy(clean_logits, labels_val)),
                       "aug_accs": {}, "aug_ces": {}}
                per_aug = {}
                for an in aug_names:                           # 每个 (头, 扰动) 只算一次, 三套口径共用
                    u_p_n = layer_norm(aug_val[an].double() @ E.T)
                    pl = head_logits(u_p_n, hp).detach()
                    du_vec = u_p_n - u_c_n
                    per_aug[an] = (pl, du_vec, du_vec.norm(dim=-1))
                    rec["aug_accs"][an] = float((pl.argmax(1) == labels_val).double().mean())
                    rec["aug_ces"][an] = float(torch.nn.functional.cross_entropy(pl, labels_val))
                rec["avg_aug_acc"] = float(np.mean(list(rec["aug_accs"].values())))
                rec["avg_aug_ce"] = float(np.mean(list(rec["aug_ces"].values())))

                for scheme in SCHEMES:
                    kidx, kmask = competitor_index(labels_val, scheme, fixed_comps, own_avg, own_sample)
                    acc = new_acc(n_classes)
                    for an in aug_names:
                        pl, du_vec, du_str = per_aug[an]
                        accumulate(acc, clean_logits, pl, du_vec, du_str, labels_val, kidx, kmask, A)
                    tab = pair_table(acc, n_classes, A, Hm, pair_qtl)
                    boot_seed = BOOT_SEED + 10 * s + list(SCHEMES).index(scheme)
                    for norm in NORMS:
                        f_ = regress(tab, "w_norm_proj" if norm == "proj" else "w_norm_raw",
                                     B if norm == "proj" else 0, boot_seed)      # 自助法只对主口径做
                        sfx = f"_{scheme}" if norm == "proj" else f"_rawnorm_{scheme}"
                        rec[f"delta_r2{sfx}"] = f_["delta_r2"]
                        rec[f"pcr_coef{sfx}"] = f_["pcr_coef"]
                        if norm == "proj":
                            rec.update({f"coef_t_{scheme}": f_["coef_t"], f"r2_base_{scheme}": f_["r2_base"],
                                        f"r2_full_{scheme}": f_["r2_full"], f"n_pairs_{scheme}": f_["n_pairs"],
                                        f"avg_svis_{scheme}": f_["avg_svis"], f"avg_pert_strength_{scheme}": f_["avg_pert_strength"]})
                            if B > 0:
                                rec[f"boot_{scheme}"] = f_["boot"]
            per_seed.append(rec)
            print(f"  {name} s{s}: clean {rec['clean_acc'] * 100:.1f}%, aug {rec['avg_aug_acc'] * 100:.1f}%, "
                  f"ΔR²(固定) {rec['delta_r2_fixed']:.4f} [t {rec['coef_t_fixed']:+.1f}], coef {rec['pcr_coef_fixed']:+.3f}, "
                  f"ΔR²(逐样本) {rec['delta_r2_own_sample']:.4f}", flush=True)

        agg = {}
        if per_seed:
            for k_ in per_seed[0]:
                if k_ == "seed":
                    continue
                v = [p[k_] for p in per_seed if k_ in p]
                if all(isinstance(x, float) for x in v):
                    agg[k_] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        results["arms"][tag] = {"name": name, "per_seed": per_seed, "agg": agg}

    # ===== 5. 判定 =====
    results["verdict_0b_layer2"] = layer2_verdict(results, "fixed")
    results["verdict_0b_layer2_sensitivity"] = {sch: layer2_verdict(results, sch)["label"] for sch in ("own_avg", "own_sample")}
    def arm_mean(tag, key):
        return float(np.mean([p.get(key, 0.0) for p in results["arms"].get(tag, {}).get("per_seed", [])]))
    od = {(t_, key): arm_mean(t_, key) for t_ in OD_ARMS for key in ("avg_aug_acc", "avg_aug_ce")}
    acc_opt = "a2" if od[("full_a2", "avg_aug_acc")] > od[("full_a1", "avg_aug_acc")] > od[(CE_ARM, "avg_aug_acc")] else \
        ("a1" if od[("full_a1", "avg_aug_acc")] > max(od[("full_a2", "avg_aug_acc")], od[(CE_ARM, "avg_aug_acc")]) else "ambiguous")
    ce_opt = "a1" if od[("full_a1", "avg_aug_ce")] < min(od[(CE_ARM, "avg_aug_ce")], od[("full_a2", "avg_aug_ce")]) else \
        ("a2" if od[("full_a2", "avg_aug_ce")] < min(od[(CE_ARM, "avg_aug_ce")], od[("full_a1", "avg_aug_ce")]) else "ambiguous")
    results["verdict_0d"] = {"aug_acc": {"ce": od[(CE_ARM, "avg_aug_acc")], "a1": od[("full_a1", "avg_aug_acc")], "a2": od[("full_a2", "avg_aug_acc")]},
                              "aug_ce": {"ce": od[(CE_ARM, "avg_aug_ce")], "a1": od[("full_a1", "avg_aug_ce")], "a2": od[("full_a2", "avg_aug_ce")]},
                              "acc_optimal": acc_opt, "ce_optimal": ce_opt,
                              "note": "Acc 和 CE 给出不同的最优剂量时必须同时报告两者"}

    # ===== 6. 保存 (JSON 与 md 来自同一份 results) =====
    results["elapsed_seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, OUT_JSON), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, allow_nan=False)      # 出现 NaN/Inf 就报错, 不写非标准 JSON
    with open(os.path.join(OUT_DIR, OUT_MD), "w", encoding="utf-8") as f:
        f.write(write_report(results))
    print(json.dumps({"verdict_0b": results["verdict_0b_layer2"]["label"], "verdict_0d_acc": acc_opt,
                      "verdict_0d_ce": ce_opt, "sensitivity": results["verdict_0b_layer2_sensitivity"]}, ensure_ascii=False, indent=1))
    print(f"[saved] {OUT_DIR}/{OUT_JSON}、{OUT_MD}  (耗时 {results['elapsed_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
