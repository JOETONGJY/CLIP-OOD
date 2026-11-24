"""
DS-Local 诊断套件 (2026-11-24, Stage 0 优先级 1; 语义化命名规范)

研究问题: 原始 DDO 约束的是"孤立文本域方向的响应", 而 DS-Local 关心的是
"实际源域样本沿域方向被扰动时, 分类决策(类别间隔/翻转)变化多少"。
两者因 LayerNorm 非线性而不等。若存在稳定的局部约束失配
(低孤立响应但高局部敏感性的方向/样本), 则 DS-Local 约束形式值得立项。

诊断项 (外审 2026-11-24 指定, 全部 source-only):
  DS-SourceScale   源域扰动强度确定: 语义保持率(类原型 margin>0)曲线,
                   s_sem = 保持率>=95% 的最大强度; 参照 = 类原型离散度
  DS-Margin        有符号的正确类别间隔下降 (文本方向 vs 匹配尺度随机方向)
  DS-Flip          原本分类正确样本的预测翻转率 (文本 vs 随机)
  DS-Tail          低孤立 DDO 响应但高局部敏感性的方向:
                   Spearman(孤立响应, 局部敏感度) + 盲区占比
  DS-RandomControl 同尺度随机单位方向对照 (贯穿全部诊断)

预注册立项判据: 在 s_sem 强度下, 文本方向的间隔下降与翻转率均超过随机方向 30% 以上,
且 DS-Tail 显示孤立响应与局部敏感性脱钩 (Spearman < 0.6)。
产物: experiment_results/DS-Diagnostic/ds_local_cub.json
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
S_GRID = [round(0.05 * k, 2) for k in range(1, 21)]      # 0.05 ~ 1.00
MARGIN_RATIO_GATE = 1.30
FLIP_RATIO_GATE = 1.30


def head_from_sd(sd, k):
    head = nn.Sequential(nn.LayerNorm(312), nn.Linear(312, k))
    head[0].load_state_dict({"weight": sd["1.weight"], "bias": sd["1.bias"]})
    head[1].load_state_dict({"weight": sd["2.weight"], "bias": sd["2.bias"]})
    return head


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
    class_names = list(train_dataset.classname2id.keys())
    C = len(class_names)
    diffs = train_dataset.domain_diffs.float().to(device)       # (204, C, 768) 单位方向
    concept_names = list(train_dataset.concept2id.keys())

    import clip
    clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)
    with torch.no_grad():
        E = clip_model.encode_text(clip.tokenize(concept_names).to(device)).float()
        E = E / E.norm(dim=-1, keepdim=True)
        # 源类文本原型 (photo 模板) —— 语义保持的参照系, 纯源域
        texts_s = [tp.format(cn) for tp, cn in
                   zip(["a photo of a {}."] * C, class_names)]
        S_proto = clip_model.encode_text(clip.tokenize(texts_s).to(device)).float()
        S_proto = S_proto / S_proto.norm(dim=-1, keepdim=True)
    del clip_model
    torch.cuda.empty_cache()

    # ===== 缓存特征 (源域训练集) =====
    tr = torch.load("cache/CUB_train.pt", map_location="cpu", weights_only=False)
    z_all = torch.nn.functional.normalize(tr["feats"].float(), dim=-1).to(device)
    y_all = tr["labels"].long().to(device)

    # ===== 收敛分类器 (E1 诊断协议快照) =====
    snap = torch.load(SNAP, map_location="cpu", weights_only=False)
    K = snap["cls"]["2.bias"].numel()
    head = head_from_sd(snap["cls"], K).to(device).eval()

    # ===== 采样: 正确分类的源域样本 =====
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        logits_all = []
        for i in range(0, z_all.size(0), 4096):
            logits_all.append(head(z_all[i:i + 4096] @ E.T))
        logits_all = torch.cat(logits_all)
    correct_mask = logits_all.argmax(-1) == y_all
    idx = torch.nonzero(correct_mask, as_tuple=True)[0]
    idx = idx[torch.randperm(idx.size(0), generator=rng)[:N_SAMPLES]]
    z_pts, y_pts = z_all[idx], y_all[idx]
    print(f"[DS] 正确分类源样本: {int(correct_mask.sum())}/{z_all.size(0)}, 采样 {len(idx)}")

    # ===== DS-SourceScale: 语义保持率曲线 (source-only) =====
    # 每点随机选一个描述符方向扰动, 语义保持 = CLIP 原型 margin > 0
    desc_of_pt = torch.randint(0, 204, (len(idx),), generator=rng).to(device)
    d_pts = diffs[desc_of_pt, y_pts]                            # (M, 768) 单位方向
    keep_curve = {}
    for s in S_GRID:
        z_pert = torch.nn.functional.normalize(z_pts + s * d_pts, dim=-1)
        with torch.no_grad():
            sims = z_pert @ S_proto.T
            gt = sims.gather(1, y_pts.unsqueeze(1)).squeeze(1)
            others = sims.scatter(1, y_pts.unsqueeze(1), -1e9).max(dim=-1).values
        keep_curve[s] = float(((gt - others) > 0).float().mean())
    s_sem = max([s for s in S_GRID if keep_curve[s] >= KEEP_TARGET], default=0.05)
    proto = []
    for c in range(C):
        m = y_all == c
        if m.any():
            proto.append(torch.nn.functional.normalize(z_all[m].mean(0), dim=-1))
    proto = torch.stack(proto)
    d_proto = float(torch.cdist(proto, proto)[~torch.eye(C, dtype=torch.bool, device=device)].median())
    print(f"[DS-SourceScale] keep-rate 曲线: "
          f"{ {s: round(v, 3) for s, v in list(keep_curve.items())[::4]} }")
    print(f"[DS-SourceScale] s_sem(95%保持) = {s_sem}   类原型离散度参照 = {d_proto:.4f}")

    # ===== 诊断主循环 (s = s_sem/2, s_sem, 2*s_sem) =====
    strengths = [s_sem / 2, s_sem, min(2 * s_sem, S_GRID[-1])]
    rand_dirs = torch.nn.functional.normalize(
        torch.randn(N_SAMPLES, z_pts.size(1), generator=rng), dim=-1).to(device)

    with torch.no_grad():
        a_base = z_pts @ E.T
        h_base = head(a_base)
    y_dev = y_pts
    iso_mag = reg_mag = None
    with torch.no_grad():
        P = diffs @ E.T
        reg_iso = head(P)                                        # (204, C, K) 孤立响应
        iso_mag = reg_iso.abs().mean(dim=(1, 2))                 # (204,) 每描述符孤立响应

    def margins(h):
        return h.gather(1, y_dev.unsqueeze(1)).squeeze(1) - \
            h.scatter(1, y_dev.unsqueeze(1), -1e9).max(dim=-1).values

    m_base = margins(h_base)
    correct_base = h_base.argmax(-1) == y_dev                    # 采样点中基线正确的

    ds_local = {}
    for s in strengths:
        # ---- 文本方向: 全部 204 个描述符 ----
        drop_list, flip_list, csens_list = [], [], []
        for i in range(204):
            d_iy = diffs[i][y_dev]
            z_pert = torch.nn.functional.normalize(z_pts + s * d_iy, dim=-1)
            with torch.no_grad():
                h_pert = head(z_pert @ E.T)
            m_pert = margins(h_pert)
            drop_list.append(float((m_base - m_pert).mean()))    # 有符号间隔下降 (正=下降)
            flip_list.append(float(((h_pert.argmax(-1) != y_dev) & correct_base).float().mean()
                                   / correct_base.float().mean()))
            dh = h_pert - h_base
            dh_c = dh - dh.mean(dim=-1, keepdim=True)
            csens_list.append(float(dh_c.norm(dim=-1).mean()))   # 局部居中敏感度
        # ---- 随机方向对照 (同尺度) ----
        z_rand = torch.nn.functional.normalize(z_pts + s * rand_dirs, dim=-1)
        with torch.no_grad():
            h_rand = head(z_rand @ E.T)
        m_rand = margins(h_rand)
        drop_rand = float((m_base - m_rand).mean())
        flip_rand = float(((h_rand.argmax(-1) != y_dev) & correct_base).float().mean()
                          / correct_base.float().mean())
        dh_r_c = (h_rand - h_base)
        dh_r_c = dh_r_c - dh_r_c.mean(dim=-1, keepdim=True)
        csens_rand = float(dh_r_c.norm(dim=-1).mean())

        ds_local[f"s={round(s, 3)}"] = {
            "margin_drop_text_mean": round(float(np.mean(drop_list)), 4),
            "margin_drop_random": round(drop_rand, 4),
            "margin_ratio_text_over_random": round(float(np.mean(drop_list)) / max(drop_rand, 1e-9), 3),
            "flip_rate_text_mean": round(float(np.mean(flip_list)), 4),
            "flip_rate_random": round(flip_rand, 4),
            "flip_ratio_text_over_random": round(float(np.mean(flip_list)) / max(flip_rand, 1e-9), 3),
            "local_sens_text_mean": round(float(np.mean(csens_list)), 4),
            "local_sens_random": round(csens_rand, 4),
            "per_descriptor_margin_drop_top5": sorted(
                [{"i": i, "drop": round(d, 4), "text": target_text_prompts[i]}
                 for i, d in enumerate(drop_list)], key=lambda x: -x["drop"])[:5],
        }
        print(f"[DS-Local] s={round(s, 3)}: 间隔下降 文本={np.mean(drop_list):.4f} 随机={drop_rand:.4f} "
              f"(比 {np.mean(drop_list)/max(drop_rand,1e-9):.2f}x) | "
              f"翻转率 文本={np.mean(flip_list):.4f} 随机={flip_rand:.4f}")

    # ===== DS-Tail: 孤立响应 vs 局部敏感度 (方向级) =====
    iso_np = iso_mag.cpu().numpy()
    csens_np = np.array(csens_list)
    tail_spearman = spearman(iso_np, csens_np)
    # 盲区方向: 孤立响应后 25% 但局部敏感度前 25%
    q_iso = np.quantile(iso_np, 0.25)
    q_sens = np.quantile(csens_np, 0.75)
    blind = (iso_np <= q_iso) & (csens_np >= q_sens)
    blind_texts = [target_text_prompts[i] for i in np.nonzero(blind)[0]]
    results = {
        "dataset": "CUB",
        "head": "CUB_prior_residual_sel_seed1 epoch99 (E1 协议快照)",
        "ds_sourcescale": {
            "keep_curve": keep_curve, "s_sem_95pct": s_sem,
            "proto_dispersion_ref": round(d_proto, 4),
        },
        "ds_local": ds_local,
        "ds_tail": {
            "spearman_isolated_vs_local": round(tail_spearman, 4),
            "blind_spot_count": int(blind.sum()),
            "blind_spot_texts": blind_texts,
        },
        "verdict": {
            "margin_ratio_at_ssem": ds_local[f"s={round(s_sem, 3)}"]["margin_ratio_text_over_random"],
            "flip_ratio_at_ssem": ds_local[f"s={round(s_sem, 3)}"]["flip_ratio_text_over_random"],
            "tail_decoupled": tail_spearman < 0.6,
            "pass": (ds_local[f"s={round(s_sem, 3)}"]["margin_ratio_text_over_random"] >= MARGIN_RATIO_GATE
                     and ds_local[f"s={round(s_sem, 3)}"]["flip_ratio_text_over_random"] >= FLIP_RATIO_GATE
                     and tail_spearman < 0.6),
        },
    }
    print(f"[DS-Tail] Spearman(孤立响应, 局部敏感度) = {tail_spearman:.4f} "
          f"盲区方向 {int(blind.sum())} 个")
    v = results["verdict"]
    print(f"[GATE] margin比={v['margin_ratio_at_ssem']} flip比={v['flip_ratio_at_ssem']} "
          f"尾部脱钩={v['tail_decoupled']} → DS-Local 立项判据: {'通过' if v['pass'] else '不通过'}")
    with open(os.path.join(OUT_DIR, "ds_local_cub.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, (np.ndarray, torch.Tensor)) else
                  (float(o) if isinstance(o, np.floating) else str(o)))
    print(f"[saved] {os.path.join(OUT_DIR, 'ds_local_cub.json')}")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
