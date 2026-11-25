"""
DS-Local 诊断套件 v2.1 (2026-11-25; v2 经自查后重写)

v1 的两处确定错误 (外审发现):
  1. s_sem 筛选失败后 default=0.05 回退, 被误报为"95% 语义保持尺度"
  2. DS-Tail 误用 strengths 循环最后一个尺度 (0.10) 的数据

v2 自查发现的逻辑问题 (v2.1 修正):
  a. gate 只判定尺度筛选, 未计算失配判据 —— "通过"是写在文字里的, 不是代码算出来的
  b. bootstrap CI 只对描述符重采样, 把 3 个随机方向种子当作固定值 → CI 过窄
     (翻转率一项的随机种子散布已超过所报 CI 的半宽) → 随机种子增至 20, 合并两项方差
  c. 逐描述符保持率每个描述符仅 ~7 个有效样本 (0.714 = 5/7), 却被当作"最差描述符"报告
     → 改为 全部204描述符 x 全部有效样本 的精确计算; K(s) 曲线同样改为精确值
  d. 孤立响应对全部文本类别等权, 局部敏感性按采样到的视觉类别分布加权 → 增加
     "类别分布匹配"的孤立响应, 与局部敏感性做同口径比较
  e. 落盘的局部敏感度数组只有最后一个尺度, 且键名误标为 max_grid_scale → 每尺度全数组落盘
  f. s_sem=None 时仅最大强度被标 exploratory → 该分支下全部强度标注

判据说明 (诚实声明): 失配判据的阈值是在看过 v2 输出之后才写进代码的 (post-hoc),
     不能当作预注册检验; 输出中显式标注 criteria_set="post-hoc"。

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
N_RAND_SEEDS = 20
N_BOOT = 2000
TAIL_GATE = 0.6            # post-hoc
MARGINAL_BAND = 0.03       # post-hoc: |rho - TAIL_GATE| 小于此值视为边缘

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
    """平均秩 (正确处理并列)"""
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
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def displaced_norm_ratio(c, cos_theta):
    """||z + c·d||² = 1 + c² + 2c·cosθ  (||z||=||d||=1)"""
    return 1 + c * c + 2 * c * cos_theta


def displaced_distance(c, cos_theta):
    """||norm(z + c·d) - z||² = 2 - 2(1 + c·cosθ)/||z + c·d||"""
    return torch.sqrt(torch.clamp(
        2 - 2 * (1 + c * cos_theta) / torch.sqrt(displaced_norm_ratio(c, cos_theta)), min=1e-12))


def solve_c_for_distance(cos_theta, target_dist, iters=40):
    """距离对 c 单调递增 → 二分解出与目标距离对应的扰动系数 (逐样本)"""
    c_lo = torch.zeros_like(cos_theta)
    c_hi = torch.full_like(cos_theta, 2.0)
    for _ in range(8):
        too_small = displaced_distance(c_hi, cos_theta) < target_dist
        if not too_small.any():
            break
        c_hi = torch.where(too_small, c_hi * 2, c_hi)
    for _ in range(iters):
        c_mid = (c_lo + c_hi) / 2
        too_small = displaced_distance(c_mid, cos_theta) < target_dist
        c_lo = torch.where(too_small, c_mid, c_lo)
        c_hi = torch.where(~too_small, c_mid, c_hi)
    return (c_lo + c_hi) / 2


def combined_ci(diff, boot_samples, rand_seeds):
    """合并方差: 描述符维 bootstrap 的 SE 与 随机方向种子间 的 SE 按平方和相加"""
    se_boot = float(np.std(boot_samples, ddof=1))
    se_rand = float(np.std(rand_seeds, ddof=1)) / np.sqrt(len(rand_seeds))
    se = float(np.sqrt(se_boot ** 2 + se_rand ** 2))
    return {"se_descriptor_bootstrap": round(se_boot, 6),
            "se_random_seed": round(se_rand, 6),
            "ci95_combined": [round(float(diff - 1.96 * se), 6), round(float(diff + 1.96 * se), 6)]}


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
    diffs = train_dataset.domain_diffs.float().to(device)      # (204, C, 768) 逐类单位方向
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
    z_all = F.normalize(tr["feats"].float(), dim=-1).to(device)
    y_all = tr["labels"].long().to(device)

    snap = torch.load(SNAP, map_location="cpu", weights_only=False)
    K_cls = snap["cls"]["2.bias"].numel()
    head = head_from_sd(snap["cls"], K_cls).to(device).eval()

    # ===== 采样: 分类头在基线处分类正确的源域样本 =====
    with torch.no_grad():
        correct_mask = head(z_all @ E.T).argmax(-1) == y_all
    idx = torch.nonzero(correct_mask, as_tuple=True)[0]
    perm = torch.randperm(idx.size(0), generator=torch.Generator().manual_seed(0))[:N_SAMPLES].to(device)
    z_pts, y_pts = z_all[idx[perm]], y_all[idx[perm]]
    n = int(z_pts.size(0))
    print(f"[DS] head-correct source samples: {int(correct_mask.sum())}/{int(z_all.size(0))}, sampled {n}")

    def semantic_margin(z, y):
        sims = z @ S_proto.T
        gt = sims.gather(1, y.unsqueeze(1)).squeeze(1)
        other = sims.scatter(1, y.unsqueeze(1), -1e9).max(dim=-1).values
        return gt - other

    # ===== DS-SourceScale: 条件语义保持率 K(s), 精确 (全部描述符 x 全部有效样本) =====
    m_base_sem = semantic_margin(z_pts, y_pts)
    valid = m_base_sem > 0
    zv, yv = z_pts[valid], y_pts[valid]
    n_valid = int(valid.sum())

    def keep_all_desc(s):
        """(n_t,) 每个描述符在全部有效样本上的语义保持率"""
        out = np.zeros(n_t)
        with torch.no_grad():
            for i in range(n_t):
                zp = F.normalize(zv + s * diffs[i][yv], dim=-1)
                out[i] = float((semantic_margin(zp, yv) > 0).float().mean())
        return out

    keep_curve, keep_per_desc = {"0": 1.0}, {}
    for s in S_GRID[1:]:
        k = keep_all_desc(s)
        keep_per_desc[str(s)] = k
        keep_curve[str(s)] = round(float(k.mean()), 4)
    passed = [s for s in S_GRID[1:] if keep_curve[str(s)] >= KEEP_TARGET]
    s_sem = max(passed) if passed else None
    s_ref = s_sem if s_sem else S_GRID[-1]
    k_ref = keep_per_desc[str(s_ref)]
    worst_idx = np.argsort(k_ref)[:8]
    print(f"[DS-SourceScale] exact K(s): {keep_curve}")
    print(f"[DS-SourceScale] n_valid={n_valid}/{n}; s_sem={s_sem} "
          f"{'(passed)' if s_sem else '(FAILED: no scale with K>=0.95; no fallback)'}")
    print(f"[DS-SourceScale] per-descriptor keep at s_ref={s_ref}: min={k_ref.min():.3f} "
          f"p5={np.quantile(k_ref, 0.05):.3f} median={np.median(k_ref):.3f}; "
          f"n_desc<0.80={int((k_ref < 0.80).sum())}")

    strengths = (sorted({round(s_ref / 2, 5), round(s_ref, 5), round(min(2 * s_ref, S_GRID[-1]), 5)})
                 if s_sem else [0.025, 0.05, 0.1])

    # ===== 公共量 =====
    with torch.no_grad():
        h_base = head(z_pts @ E.T)
        reg_iso = head(diffs @ E.T)                            # (204, C, K)
    class_w = torch.bincount(y_pts, minlength=C).float() / n   # 采样到的视觉类别分布
    class_w = class_w.to(device)
    iso_uniform, iso_classmatched = {}, {}
    for mname, fn in METRICS.items():
        per_ic = fn(reg_iso)                                   # (204, C)
        iso_uniform[mname] = per_ic.mean(dim=1).cpu().numpy()
        iso_classmatched[mname] = (per_ic @ class_w).cpu().numpy()

    def margins(h):
        return h.gather(1, y_pts.unsqueeze(1)).squeeze(1) - \
            h.scatter(1, y_pts.unsqueeze(1), -1e9).max(dim=-1).values

    m_base = margins(h_base).cpu().numpy()
    print(f"[DS] baseline head margin: mean={m_base.mean():.4f} median={np.median(m_base):.4f}")

    per_scale, tail_by_scale, arrays = {}, {}, {}
    for s in strengths:
        exploratory = s_sem is None
        margin_drop, flip = np.zeros(n_t), np.zeros(n_t)
        sens = {m: np.zeros(n_t) for m in METRICS}
        dist_mean = np.zeros(n_t)
        dist_per_sample = torch.zeros(n, device=device)
        with torch.no_grad():
            for i in range(n_t):
                d_iy = diffs[i][y_pts]
                cos_theta = (z_pts * d_iy).sum(-1)
                dist = displaced_distance(torch.full_like(cos_theta, s), cos_theta)
                dist_mean[i] = float(dist.mean())
                dist_per_sample += dist / n_t
                h_pert = head(F.normalize(z_pts + s * d_iy, dim=-1) @ E.T)
                margin_drop[i] = float((m_base - margins(h_pert).cpu().numpy()).mean())
                flip[i] = float((h_pert.argmax(-1) != y_pts).float().mean())   # 采样点基线全部正确
                dh = h_pert - h_base
                for mname, fn in METRICS.items():
                    sens[mname][i] = float(fn(dh).mean())

        # ---- 随机对照: 逐样本匹配实际扰动距离, N_RAND_SEEDS 个随机方向 ----
        r_margin, r_flip = [], []
        r_sens = {m: [] for m in METRICS}
        with torch.no_grad():
            for rs in range(N_RAND_SEEDS):
                r = F.normalize(torch.randn(n, z_pts.size(1),
                                            generator=torch.Generator().manual_seed(100 + rs)), dim=-1).to(device)
                c_match = solve_c_for_distance((z_pts * r).sum(-1), dist_per_sample)
                h_rand = head(F.normalize(z_pts + c_match.unsqueeze(1) * r, dim=-1) @ E.T)
                r_margin.append(float((m_base - margins(h_rand).cpu().numpy()).mean()))
                r_flip.append(float((h_rand.argmax(-1) != y_pts).float().mean()))
                dh_r = h_rand - h_base
                for mname, fn in METRICS.items():
                    r_sens[mname].append(float(fn(dh_r).mean()))

        # ---- 配对差值 + 合并 CI ----
        rng_b = np.random.default_rng(7)
        bm, bf = [], []
        for _ in range(N_BOOT):
            bi = rng_b.integers(0, n_t, n_t)
            bm.append(float(margin_drop[bi].mean() - np.mean(r_margin)))
            bf.append(float(flip[bi].mean() - np.mean(r_flip)))
        dm, df = float(margin_drop.mean() - np.mean(r_margin)), float(flip.mean() - np.mean(r_flip))
        entry = {
            "s": s, "exploratory_only": exploratory,
            "text_dist_mean": round(float(dist_mean.mean()), 5),
            "text_margin_drop_mean": round(float(margin_drop.mean()), 6),
            "rand_margin_drop_mean": round(float(np.mean(r_margin)), 6),
            "rand_margin_drop_sd": round(float(np.std(r_margin, ddof=1)), 6),
            "paired_margin_diff": round(dm, 6), "margin_ci": combined_ci(dm, bm, r_margin),
            "text_flip_rate": round(float(flip.mean()), 6),
            "rand_flip_rate_mean": round(float(np.mean(r_flip)), 6),
            "rand_flip_rate_sd": round(float(np.std(r_flip, ddof=1)), 6),
            "paired_flip_diff": round(df, 6), "flip_ci": combined_ci(df, bf, r_flip),
        }
        for mname in METRICS:
            t, r = float(sens[mname].mean()), float(np.mean(r_sens[mname]))
            entry[f"sens_{mname}"] = {"text": round(t, 5), "rand_mean": round(r, 5),
                                      "text_over_rand": round(t / r, 4)}

        # ---- DS-Tail: 同口径 (度量一致 + 类别分布匹配) ----
        tail_s = {"s": s, "exploratory_only": exploratory}
        for mname in METRICS:
            tail_s[f"spearman_uniform_{mname}"] = round(spearman(iso_uniform[mname], sens[mname]), 4)
            tail_s[f"spearman_classmatched_{mname}"] = round(
                spearman(iso_classmatched[mname], sens[mname]), 4)
        iso_v, loc_v = iso_classmatched["centered_l2"], sens["centered_l2"]
        blind = (iso_v <= np.quantile(iso_v, 0.25)) & (loc_v >= np.quantile(loc_v, 0.75))
        tail_s["blind_count"] = int(blind.sum())
        tail_s["blind_expect_if_independent"] = round(n_t * 0.0625, 1)
        tail_s["blind_margin_drop_mean"] = (round(float(margin_drop[blind].mean()), 6) if blind.any() else None)
        tail_s["blind_flip_mean"] = round(float(flip[blind].mean()), 6) if blind.any() else None
        tail_s["blind_descriptors"] = [{"i": int(i), "text": target_text_prompts[i]}
                                       for i in np.nonzero(blind)[0]]
        per_scale[str(s)], tail_by_scale[str(s)] = entry, tail_s
        arrays[str(s)] = {"sens_" + m: sens[m].tolist() for m in METRICS}
        arrays[str(s)].update({"margin_drop": margin_drop.tolist(), "flip": flip.tolist(),
                               "dist_mean": dist_mean.tolist()})

        c3 = tail_s["spearman_classmatched_centered_l2"]
        print(f"[DS s={s}{' exploratory' if exploratory else ''}] "
              f"margin drop text={entry['text_margin_drop_mean']:+.5f} rand={entry['rand_margin_drop_mean']:+.5f} "
              f"diff={dm:+.5f} CI={entry['margin_ci']['ci95_combined']} | "
              f"flip text={entry['text_flip_rate']:.5f} rand={entry['rand_flip_rate_mean']:.5f} "
              f"diff={df:+.5f} CI={entry['flip_ci']['ci95_combined']} | "
              f"sens ratio(cL2)={entry['sens_centered_l2']['text_over_rand']} | "
              f"tail rho cm/unif={c3}/{tail_s['spearman_uniform_centered_l2']} blind={int(blind.sum())}")

    # ===== gate: 由代码计算 (阈值 post-hoc) =====
    gate = {"scale_selection_passed": s_sem is not None,
            "criteria_set": "post-hoc (thresholds written into code after v2 output was inspected)"}
    if s_sem is None:
        gate["verdict"] = "FAIL: no scale with conditional semantic keep >= 0.95; mismatch not assessed"
    else:
        e, t = per_scale[str(round(s_sem, 5))], tail_by_scale[str(round(s_sem, 5))]
        c1 = e["margin_ci"]["ci95_combined"][0] > 0
        c2 = e["flip_ci"]["ci95_combined"][0] > 0
        rho = t["spearman_classmatched_centered_l2"]
        c3 = rho < TAIL_GATE
        gate.update({"c1_margin_diff_ci_excludes_0": bool(c1), "c2_flip_diff_ci_excludes_0": bool(c2),
                     "c3_classmatched_spearman_lt_0.6": bool(c3), "tail_rho_at_s_sem": rho,
                     "tail_marginal": bool(abs(rho - TAIL_GATE) < MARGINAL_BAND),
                     "pass": bool(c1 and c2 and c3)})
        gate["verdict"] = ("PASS (post-hoc criteria)" if gate["pass"] else "FAIL (post-hoc criteria)") + \
                          ("; tail criterion marginal" if gate["tail_marginal"] else "")

    def _np_default(o):
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    results = {
        "dataset": "CUB", "version": "v2.1",
        "n_rand_seeds": N_RAND_SEEDS,
        "keep_curve_conditional_exact": keep_curve,
        "n_valid_baseline_positive": n_valid,
        "s_sem": s_sem, "scale_selection_failed": s_sem is None,
        "per_descriptor_keep_at_sref": {"s_ref": s_ref, "min": round(float(k_ref.min()), 4),
                                        "p5": round(float(np.quantile(k_ref, 0.05)), 4),
                                        "median": round(float(np.median(k_ref)), 4),
                                        "n_below_0.80": int((k_ref < 0.80).sum())},
        "worst_keep_descriptors": [{"i": int(i), "keep": round(float(k_ref[i]), 4),
                                    "text": target_text_prompts[i]} for i in worst_idx],
        "strengths_measured": strengths,
        "per_scale": per_scale, "tail_by_scale": tail_by_scale,
        "gate": gate,
        "arrays": {"iso_uniform": {m: iso_uniform[m].tolist() for m in METRICS},
                   "iso_classmatched": {m: iso_classmatched[m].tolist() for m in METRICS},
                   "per_scale": arrays},
        "notes": [
            "v1 errors fixed: s_sem default fallback misreported as 95% scale; DS-Tail used last-loop scale (0.10).",
            "v2 logic fixes: gate now computed; CI merges descriptor-bootstrap and random-seed variance; "
            "exact all-descriptor keep rates; class-distribution-matched isolated response; per-scale arrays saved.",
            "Mismatch thresholds are post-hoc and must not be read as a pre-registered test.",
        ],
    }
    out = os.path.join(OUT_DIR, "ds_local_cub_v2.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=_np_default)
    print(f"[GATE] {json.dumps(gate, ensure_ascii=False)}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
