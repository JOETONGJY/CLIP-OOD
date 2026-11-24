"""
DS-Local 诊断套件 v2 (2026-11-24, 二次外审后完全重写)

v1 两处确定错误 (外审发现):
  1. s_sem 筛选失败后 default=0.05 回退, 被误报为"95% 语义保持尺度"
  2. DS-Tail 误用 strengths 循环最后一个尺度 (0.10) 的数据

v2 修正:
  - 条件语义保持率 K(s) = P(margin(s)>0 | margin(0)>0), K(0)=1
  - 细粒度网格; s_sem = max{s: K(s)>=0.95}; 无合格尺度 → s_sem=None,
    明确判定"尺度筛选失败" (不回退; 此时在最大网格尺度做探索性测量并显式标注)
  - Margin/Flip/Tail 全部在同一尺度计算; 204 维数组全落盘
  - 实际扰动距离闭式公式; 随机方向对照逐样本匹配实际距离 (闭式二分) × 3 seeds
  - 三度量一致性网格: Raw-L1 / Centered-L1 / Centered-L2
  - scipy.stats.spearmanr; 配对差值 + 描述符维 bootstrap 95% CI (无比值门控)

产物: experiment_results/DS-Diagnostic/ds_local_cub_v2.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import get_args
from prompts.prompt200new import target_text_prompts

OUT_DIR = os.path.join("experiment_results", "DS-Diagnostic")
SNAP = os.path.join("experiment_results", "07_权重逃逸诊断", "cls_snapshots", "epoch_0099.pt")
N_SAMPLES = 2000
KEEP_TARGET = 0.95
S_GRID = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1]
N_RAND_SEEDS = 3
N_BOOT = 2000

METRICS = {
    "raw_l1": lambda z: z.abs().mean(dim=-1),
    "centered_l1": lambda z: (z - z.mean(dim=-1, keepdim=True)).abs().mean(dim=-1),
    "centered_l2": lambda z: (z - z.mean(dim=-1, keepdim=True)).norm(dim=-1),
}


def head_from_sd(sd, k):
    head = nn.Sequential(nn.LayerNorm(312), nn.Linear(312, k))
    head[0].load_state_dict({"weight": sd["1.weight"], "bias": sd["1.bias"]})
    head[1].load_state_dict({"weight": sd["2.weight"], "bias": sd["2.bias"]})
    return head


def _rankdata(a):
    """平均秩 rankdata (正确处理并列)"""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = np.cumsum(obs)[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(a, b):
    import numpy as _np
    return float(_np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def displaced_norm_ratio(c, cos_theta):
    """||norm(z + c·d)||² = 1 + c² + 2c·cosθ (||z||=||d||=1)"""
    return 1 + c * c + 2 * c * cos_theta


def displaced_distance(c, cos_theta):
    """||norm(z + c·d) - z|| 闭式公式 (||z||=||d||=1, cos_theta = z·d)"""
    return torch.sqrt(torch.clamp(
        2 - 2 * (1 + c * cos_theta) / torch.sqrt(displaced_norm_ratio(c, cos_theta)),
        min=1e-12))


def solve_c_for_distance(cos_theta, target_dist, iters=40):
    """给定逐样本目标距离, 二分解出扰动系数 c (距离对 c 单调递增)"""
    c_lo = torch.zeros_like(cos_theta)
    c_hi = torch.full_like(cos_theta, 2.0)
    for _ in range(8):
        too_small = displaced_distance(c_hi, cos_theta) < target_dist
        if not too_small.any():
            break
        c_hi[too_small] *= 2
    for _ in range(iters):
        c_mid = (c_lo + c_hi) / 2
        too_small = displaced_distance(c_mid, cos_theta) < target_dist
        c_lo = torch.where(too_small, c_mid, c_lo)
        c_hi = torch.where(~too_small, c_mid, c_hi)
    return (c_lo + c_hi) / 2


def main():
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    train_dataset = __import__("data").get_dataset_classes(args)[0]
    diffs = train_dataset.domain_diffs.float().to(device)
    n_t, C = diffs.size(0), diffs.size(1)
    concept_names = list(train_dataset.concept2id.keys())

    import clip
    clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)
    with torch.no_grad():
        E = clip_model.encode_text(clip.tokenize(concept_names).to(device)).float()
        E = E / E.norm(dim=-1, keepdim=True)
        texts_s = [tp.format(cn) for tp, cn in
                   zip(["a photo of a {}."] * C, list(train_dataset.classname2id.keys()))]
        S_proto = clip_model.encode_text(clip.tokenize(texts_s).to(device)).float()
        S_proto = S_proto / S_proto.norm(dim=-1, keepdim=True)
    del clip_model
    torch.cuda.empty_cache()

    tr = torch.load("cache/CUB_train.pt", map_location="cpu", weights_only=False)
    z_all = torch.nn.functional.normalize(tr["feats"].float(), dim=-1).to(device)
    y_all = tr["labels"].long().to(device)

    snap = torch.load(SNAP, map_location="cpu", weights_only=False)
    K_cls = snap["cls"]["2.bias"].numel()
    head = head_from_sd(snap["cls"], K_cls).to(device).eval()

    # ===== 采样正确分类的源域样本 =====
    with torch.no_grad():
        logits_all = head(z_all @ E.T)
    correct_mask = logits_all.argmax(-1) == y_all
    idx = torch.nonzero(correct_mask, as_tuple=True)[0]
    idx = idx[torch.randperm(idx.size(0), generator=torch.Generator().manual_seed(0))[:N_SAMPLES]]
    z_pts, y_pts = z_all[idx], y_all[idx]
    n = len(idx)
    print(f"[DS] 正确分类源样本采样: {n}")

    def semantic_margin(z, y):
        sims = z @ S_proto.T
        gt = sims.gather(1, y.unsqueeze(1)).squeeze(1)
        other = sims.scatter(1, y.unsqueeze(1), -1e9).max(dim=-1).values
        return gt - other

    # ===== DS-SourceScale: 条件语义保持率 K(s) =====
    desc_of_pt = torch.randint(0, n_t, (n,), generator=torch.Generator().manual_seed(1))
    d_pts = diffs[desc_of_pt, y_pts]
    m_base_sem = semantic_margin(z_pts, y_pts)
    valid = m_base_sem > 0
    n_valid = int(valid.sum())
    keep_curve = {"0": 1.0}
    for s in S_GRID[1:]:
        z_pert = torch.nn.functional.normalize(z_pts + s * d_pts, dim=-1)
        mp = semantic_margin(z_pert, y_pts)
        keep_curve[str(s)] = round(float((mp[valid] > 0).float().mean()), 4)
    passed = [s for s in S_GRID[1:] if keep_curve[str(s)] >= KEEP_TARGET]
    s_sem = max(passed) if passed else None

    # 逐描述符保持率 (参照尺度 = s_sem 或最大网格尺度)
    s_ref = s_sem if s_sem else S_GRID[-1]
    z_ref = torch.nn.functional.normalize(z_pts + s_ref * d_pts, dim=-1)
    mp_ref = semantic_margin(z_ref, y_pts)
    keep_per_desc = np.full(n_t, np.nan)
    valid_cpu = valid.cpu()
    desc_cpu = desc_of_pt.cpu()
    for i in range(n_t):
        msk = (desc_cpu == i) & valid_cpu
        if msk.sum() >= 5:
            keep_per_desc[i] = float((mp_ref[msk] > 0).float().mean())
    finite = np.isfinite(keep_per_desc)
    worst_order = np.argsort(np.where(finite, keep_per_desc, np.inf))[:5]
    worst_finite = [int(worst_order[j]) for j in range(len(worst_order)) if finite[worst_order[j]]]
    print(f"[DS-SourceScale] K(s) 曲线: {keep_curve}")
    print(f"[DS-SourceScale] n_valid={n_valid}/{n}  s_sem={s_sem} "
          f"{'✓ 筛选通过' if s_sem else '⚠ 无满足 K>=0.95 的尺度 → 明确判定筛选失败, 不回退'}")
    print(f"[DS-SourceScale] 最差保持描述符: "
          f"{[(int(i), round(float(keep_per_desc[i]), 3)) for i in worst_finite]}")

    # ===== 测量强度 =====
    if s_sem:
        strengths = sorted({round(s_ref / 2, 4), round(s_ref, 4),
                            round(min(2 * s_ref, S_GRID[-1]), 4)})
    else:
        strengths = [0.025, 0.05, 0.1]   # 探索性测量 (显式标注)

    # ===== 公共量 =====
    with torch.no_grad():
        a_base = z_pts @ E.T
        h_base = head(a_base)
        P = diffs @ E.T
        reg_iso = head(P)                                   # (204, C, K)
    iso_by_metric = {m: fn(reg_iso).mean(dim=1).cpu().numpy()
                     for m, fn in METRICS.items()}   # fn 已降最后维 → (204, C) → mean over C

    def margins(h):
        return h.gather(1, y_pts.unsqueeze(1)).squeeze(1) - \
            h.scatter(1, y_pts.unsqueeze(1), -1e9).max(dim=-1).values

    m_base = margins(h_base).cpu().numpy()
    correct_base = h_base.argmax(-1) == y_pts

    per_scale, tail_by_scale = {}, {}
    for s in strengths:
        exploratory = (s_sem is None and s == max(strengths))
        # ---- 文本方向: 全部 204 描述符, 逐样本 ----
        margin_drop = np.zeros(n_t)          # (204,) 每描述符平均有符号间隔下降
        flip = np.zeros(n_t)
        sens = {m: np.zeros(n_t) for m in METRICS}
        dist_mat = np.zeros((n_t, n))        # (204, n) 逐样本实际扰动距离
        with torch.no_grad():
            for i in range(n_t):
                d_iy = diffs[i][y_pts]
                cos_theta = (z_pts * d_iy).sum(-1)
                z_pert = torch.nn.functional.normalize(z_pts + s * d_iy, dim=-1)
                dist_i = displaced_distance(torch.full_like(cos_theta, s), cos_theta)
                dist_mat[i] = dist_i.cpu().numpy()
                h_pert = head(z_pert @ E.T)
                m_pert = margins(h_pert).cpu().numpy()
                margin_drop[i] = float((m_base - m_pert).mean())
                flip[i] = float(((h_pert.argmax(-1) != y_pts) & correct_base).float().mean()
                                / correct_base.float().mean())
                dh = h_pert - h_base
                for mname, fn in METRICS.items():
                    sens[mname][i] = float(fn(dh).mean())
        dist_mean = dist_mat.mean(axis=1)    # (204,) 每描述符平均距离
        # ---- 随机对照: 逐样本距离匹配 (匹配文本方向逐样本平均距离) × 3 seeds ----
        target_dist = torch.from_numpy(dist_mat.mean(axis=0)).to(device)   # 每样本目标距离
        r_margin, r_flip, r_sens = [], [], {m: [] for m in METRICS}
        with torch.no_grad():
            for rs in range(N_RAND_SEEDS):
                r = torch.nn.functional.normalize(
                    torch.randn(n, z_pts.size(1), generator=torch.Generator().manual_seed(100 + rs)),
                    dim=-1).to(device)
                cos_psi = (z_pts * r).sum(-1)
                c_match = solve_c_for_distance(cos_psi, target_dist)
                z_rand = torch.nn.functional.normalize(z_pts + c_match.unsqueeze(1) * r, dim=-1)
                h_rand = head(z_rand @ E.T)
                m_rand_p = margins(h_rand).cpu().numpy()
                r_margin.append(float((m_base - m_rand_p).mean()))
                r_flip.append(float(((h_rand.argmax(-1) != y_pts) & correct_base).float().mean()
                                    / correct_base.float().mean()))
                dh_r = h_rand - h_base
                for mname, fn in METRICS.items():
                    r_sens[mname].append(float(fn(dh_r).mean()))

        entry = {
            "s": s, "exploratory_only": bool(s_sem is None and s == max(strengths)),
            "text_margin_drop_mean": round(float(margin_drop.mean()), 5),
            "rand_margin_drop_seeds": [round(v, 5) for v in r_margin],
            "paired_margin_diff_mean": round(float(margin_drop.mean() - np.mean(r_margin)), 5),
            "paired_margin_diff_per_seed": [round(float(margin_drop.mean() - v), 5) for v in r_margin],
            "text_flip_rate_mean": round(float(flip.mean()), 5),
            "rand_flip_seeds": [round(v, 5) for v in r_flip],
            "paired_flip_diff_mean": round(float(flip.mean() - np.mean(r_flip)), 5),
            "paired_flip_diff_per_seed": [round(float(flip.mean() - v), 5) for v in r_flip],
            "text_dist_mean": round(float(dist_mean.mean()), 5),
        }
        # bootstrap CI (描述符维重采样)
        rng_b = np.random.default_rng(7)
        bm, bf = [], []
        for _ in range(N_BOOT):
            bi = rng_b.integers(0, n_t, n_t)
            bm.append(float(margin_drop[bi].mean() - np.mean(r_margin)))
            bf.append(float(flip[bi].mean() - np.mean(r_flip)))
        entry["margin_diff_ci95"] = [round(float(v), 5) for v in np.quantile(bm, [0.025, 0.975])]
        entry["flip_diff_ci95"] = [round(float(v), 5) for v in np.quantile(bf, [0.025, 0.975])]
        for mname in METRICS:
            entry[f"text_sens_{mname}"] = round(float(sens[mname].mean()), 5)
            entry[f"rand_sens_{mname}_seeds"] = [round(v, 5) for v in r_sens[mname]]
        per_scale[str(s)] = entry

        # ---- DS-Tail (每尺度独立计算, 三度量一致网格) ----
        tail_s = {"s": s, "exploratory_only": entry["exploratory_only"]}
        for mname, fn in METRICS.items():
            tail_s[f"spearman_{mname}"] = round(spearman(iso_by_metric[mname], sens[mname]), 4)
        iso_v = iso_by_metric["centered_l2"]
        loc_v = sens["centered_l2"]
        q_i, q_l = np.quantile(iso_v, 0.25), np.quantile(loc_v, 0.75)
        blind = (iso_v <= q_i) & (loc_v >= q_l)
        tail_s["blind_count"] = int(blind.sum())
        tail_s["blind_expect_if_independent"] = round(n_t * 0.25 * 0.25, 1)
        tail_s["blind_descriptors"] = [
            {"i": int(i), "sens_centered_l2": round(float(sens["centered_l2"][i]), 4),
             "text": target_text_prompts[i]} for i in np.nonzero(blind)[0]]
        tail_by_scale[str(s)] = tail_s
        print(f"[DS s={s}] margin_drop 文本={entry['text_margin_drop_mean']:.4f} "
              f"随机均值={np.mean(r_margin):.4f} 配对差={entry['paired_margin_diff_mean']:+.4f} "
              f"CI={entry['margin_diff_ci95']} | flip 文本={entry['text_flip_rate_mean']:.4f} "
              f"随机={np.mean(r_flip):.4f} | Tail Spearman(centered_l2)={tail_s['spearman_centered_l2']} "
              f"盲区={tail_s['blind_count']}")

    gate = {
        "scale_selection_passed": s_sem is not None,
        "verdict": ("尺度筛选通过 → 进入失配判定" if s_sem else
                    "尺度筛选失败: 无满足 95% 条件语义保持的尺度; 最大网格尺度上的探索性测量见 per_scale"),
    }

    results = {
        "dataset": "CUB", "version": "v2",
        "keep_curve_conditional": keep_curve,
        "n_valid_baseline_positive": n_valid,
        "s_sem": s_sem,
        "scale_selection_failed": s_sem is None,
        "keep_rate_per_descriptor_at_sref": {
            str(i): (round(float(v), 4) if np.isfinite(v) else None)
            for i, v in enumerate(keep_per_desc)},
        "worst_keep_descriptors": [{"i": int(i), "keep": round(float(keep_per_desc[i]), 3),
                                    "text": target_text_prompts[i]} for i in worst_finite],
        "strengths_measured": strengths,
        "per_scale": per_scale,
        "tail_by_scale": tail_by_scale,
        "gate": gate,
        "arrays": {
            "iso_raw_l1": iso_by_metric["raw_l1"].tolist(),
            "iso_centered_l1": iso_by_metric["centered_l1"].tolist(),
            "iso_centered_l2": iso_by_metric["centered_l2"].tolist(),
            "local_centered_l2_at_max_grid_scale": sens["centered_l2"].tolist(),
        },
        "note_v1_errors": ("v1 两处确定错误: ① s_sem default=0.05 回退被误报为 95% 保持尺度; "
                          "② DS-Tail 误用循环最后尺度(0.10)数据。v2 均已修复。"),
    }
    def _np_default(o):
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(os.path.join(OUT_DIR, "ds_local_cub_v2.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=_np_default)
    print(f"[GATE] {gate['verdict']}")
    print(f"[saved] {os.path.join(OUT_DIR, 'ds_local_cub_v2.json')}")


if __name__ == "__main__":
    main()
