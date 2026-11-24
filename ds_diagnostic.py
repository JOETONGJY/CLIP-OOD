"""
DS-Diagnostic 决策敏感性诊断 (2026-11-24, 原 G-D; 命名规范见 CA-Diagnostic 报告)

三个诊断 (外审 2026-11-24 版):
  DS-CommonMode: 分类器对"孤立文本域方向"响应的公共/居中分量能量占比
      —— 若公共分量占比很小, Logit 居中 (DS-CenteredL1/L2) 无空间, DS-DDO 降级
  DS-Reparam: W 中心化 (W' = W - 1·w̄ᵀ) 前后的预测等价性与正则损失变化
      —— 验证"公共分量约束可被重参数化免费消除"的理论预测 (外审 D1 修正)
  DS-Local: 孤立方向响应 h(E·d_{i,y}) vs 实际局部预测变化
      Δh = h(a(norm(z + s·d_{i,y}))) - h(a(norm(z))) 在源域样本上的差异
      —— LayerNorm 非线性使两者不等; 若相关低, 说明原 DDO 约束的对象
         (孤立方向响应) 与真实决策敏感性脱节 → DS-Local 路线立项依据

全部基于缓存特征 + 收敛分类器快照 (E1 诊断协议产物), 零训练。
产物: experiment_results/DS-Diagnostic/ds_diagnostic_cub.json
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

from args import get_args

OUT_DIR = os.path.join("experiment_results", "DS-Diagnostic")
SNAP = os.path.join("experiment_results", "07_权重逃逸诊断", "cls_snapshots", "epoch_0099.pt")
N_PERT_SAMPLES = 2000
PERT_SUBSET_STRIDE = 4          # 描述符子集步长 (204 -> 51 个)
SEED = 0


def head_from_sd(sd, k):
    head = nn.Sequential(nn.LayerNorm(312), nn.Linear(312, k))
    head[0].load_state_dict({"weight": sd["1.weight"], "bias": sd["1.bias"]})
    head[1].load_state_dict({"weight": sd["2.weight"], "bias": sd["2.bias"]})
    return head


def l1_share(z):
    """公共分量在 L1 语义下的占比 (损失实际用的是 |·|)"""
    common = z.mean(dim=-1, keepdim=True).expand_as(z)
    c = common.abs().sum()
    t = z.abs().sum()
    return float(c / t.clamp_min(1e-12)), float(c.sum() / z.size(0)), float(t.sum() / z.size(0))


def main():
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    train_dataset = __import__("data").get_dataset_classes(args)[0]
    diffs = train_dataset.domain_diffs.float().to(device)          # (204, C, 768) 已归一化
    concept_names = list(train_dataset.concept2id.keys())

    # ===== 收敛分类器 (E1 诊断协议的最终快照) =====
    snap = torch.load(SNAP, map_location="cpu", weights_only=False)
    K = snap["cls"]["2.bias"].numel()
    head = head_from_sd(snap["cls"], K).to(device).eval()

    # ===== E: 概念嵌入 (与模型 init 同法) =====
    import clip
    clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)
    toks = clip.tokenize(concept_names).to(device)
    with torch.no_grad():
        E = clip_model.encode_text(toks).float()
        E = E / E.norm(dim=-1, keepdim=True)
    del clip_model
    torch.cuda.empty_cache()

    # ===== 缓存特征 =====
    tr = torch.load("cache/CUB_train.pt", map_location="cpu", weights_only=False)
    z_all = tr["feats"].float()
    y_all = tr["labels"].long()
    z_all = torch.nn.functional.normalize(z_all, dim=-1).to(device)

    # ===== DS-Reparam: W 中心化的预测等价性 =====
    W = head[1].weight.detach().clone()          # (K, 312)
    b = head[1].bias.detach().clone()
    w_bar = W.mean(dim=0)                        # (312,) 行均值
    W_c = W - torch.ones_like(W) * w_bar.unsqueeze(0)
    head_c = nn.Sequential(head[0], nn.Linear(312, K)).to(device).eval()
    head_c[1].load_state_dict({"weight": W_c, "bias": b})

    te = torch.load("cache/CUB_source_test.pt", map_location="cpu", weights_only=False)
    z_te = torch.nn.functional.normalize(te["feats"].float(), dim=-1).to(device)
    y_te = te["labels"].long().to(device)
    with torch.no_grad():
        logits_a = head(z_te @ E.T)
        logits_b = head_c(z_te @ E.T)
    prob_diff = float((torch.softmax(logits_a, -1) - torch.softmax(logits_b, -1)).abs().max())
    acc_a = float((logits_a.argmax(-1) == y_te).float().mean() * 100)
    acc_b = float((logits_b.argmax(-1) == y_te).float().mean() * 100)

    # 孤立方向响应的 L1 损失在中心化后下降多少 ("免费"消除的部分)
    P = diffs @ E.T                                             # (204, C, 312)
    with torch.no_grad():
        reg_a = head(P)                                          # (204, C, K)
        reg_b = head_c(P)
    l1_drop = float(reg_a.abs().mean() - reg_b.abs().mean())

    # ===== DS-CommonMode: 孤立响应的公共/居中能量占比 =====
    with torch.no_grad():
        common = reg_a.mean(dim=-1, keepdim=True)
        centered = reg_a - common
        l1_share_iso, _, _ = None, None, None
        common_l1 = common.abs().mean().item()
        centered_l1 = centered.abs().mean().item()
        l1_share_iso = common_l1 / (common_l1 + centered_l1)
        common_e = float((common ** 2).mean())
        centered_e = float((centered ** 2).mean())
        l2_share_iso = common_e / (common_e + centered_e)

    # ===== 真实域偏移尺度 s (源/目标类原型距离中位数) =====
    tgt = torch.load("cache/CUB_target_test.pt", map_location="cpu", weights_only=False)
    z_t = torch.nn.functional.normalize(tgt["feats"].float(), dim=-1)
    y_t = tgt["labels"].long()
    src_proto, tgt_proto = [], []
    for c in range(K):
        sm, tm = y_all == c, y_t == c
        if sm.any() and tm.any():
            src_proto.append(torch.nn.functional.normalize(z_all.cpu()[sm].mean(0), dim=-1))
            tgt_proto.append(torch.nn.functional.normalize(z_t[tm].mean(0), dim=-1))
    d_real = torch.stack([torch.norm(tp - sp) for sp, tp in zip(src_proto, tgt_proto)])
    s_scale = float(d_real.median())
    print(f"[DS] 真实域偏移尺度 s (类原型距离中位数) = {s_scale:.4f}")

    # ===== DS-Local: 孤立响应 vs 实际局部预测变化 (多强度 + 随机方向对照, 2026-11-24 升级) =====
    rng = torch.Generator().manual_seed(SEED)
    pts = torch.randperm(z_all.size(0), generator=rng)[:N_PERT_SAMPLES]
    z_pts = z_all[pts]                       # (M, 768) 已归一化
    y_pts = y_all[pts].to(device)
    subset = list(range(0, 204, PERT_SUBSET_STRIDE))
    rand_dirs = torch.nn.functional.normalize(
        torch.randn(N_PERT_SAMPLES, z_pts.size(1), generator=rng), dim=-1).to(device)

    def margins(h):
        """类别间隔: logit_y - max_{k≠y} logit_k"""
        return h.gather(1, y_pts.unsqueeze(1)).squeeze(1) - \
            h.scatter(1, y_pts.unsqueeze(1), -1e9).max(dim=-1).values

    with torch.no_grad():
        a_base = z_pts @ E.T                                  # (M, 312)
        h_base = head(a_base)
    m_base = margins(h_base)

    strengths = [0.5, 1.0, 2.0]                  # 真实域偏移尺度的倍数 (用户要求多强度)
    ds_local = {"strengths": strengths}
    with torch.no_grad():
        for s_frac in strengths:
            s = s_frac * s_scale
            cos_list, mag_list, cshare_list = [], [], []
            txt_centered_mag, txt_margin_change = [], []
            rand_centered_mag, rand_margin_change = [], []
            for i in subset:
                d_iy = diffs[i][y_pts]                            # (M, 768) 文本域方向
                z_pert = torch.nn.functional.normalize(z_pts + s * d_iy, dim=-1)
                h_pert = head(z_pert @ E.T)
                dh = h_pert - h_base
                iso = reg_a[i][y_pts]
                dh_c = dh - dh.mean(dim=-1, keepdim=True)
                iso_c = iso - iso.mean(dim=-1, keepdim=True)
                cos_list.append(float(torch.nn.functional.cosine_similarity(dh_c, iso_c, -1).mean()))
                mag_list.append(float(dh.norm(dim=-1).mean() / iso.norm(dim=-1).mean()))
                cshare_list.append(float(dh.mean(-1).abs().mean() / (dh.abs().mean() + 1e-12)))
                txt_centered_mag.append(float(dh_c.norm(dim=-1).mean()))
                txt_margin_change.append(float((margins(h_pert) - m_base).abs().mean()))
                # 随机方向对照 (同尺度同归一化)
                z_rand = torch.nn.functional.normalize(z_pts + s * rand_dirs, dim=-1)
                dh_r = head(z_rand @ E.T) - h_base
                rand_centered_mag.append(float((dh_r - dh_r.mean(-1, keepdim=True)).norm(dim=-1).mean()))
                rand_margin_change.append(float((margins(h_base + dh_r) - m_base).abs().mean()))
            ds_local[f"s={s_frac}x"] = {
                "mean_cos_centered_actual_vs_isolated": round(float(np.mean(cos_list)), 4),
                "mean_magnitude_ratio_actual_vs_isolated": round(float(np.mean(mag_list)), 4),
                "mean_common_l1_share_actual": round(float(np.mean(cshare_list)), 4),
                "centered_mag_text_mean": round(float(np.mean(txt_centered_mag)), 4),
                "centered_mag_random_mean": round(float(np.mean(rand_centered_mag)), 4),
                "text_over_random_centered_ratio": round(
                    float(np.mean(txt_centered_mag) / np.mean(rand_centered_mag)), 4),
                "mean_abs_margin_change_text": round(float(np.mean(txt_margin_change)), 4),
                "mean_abs_margin_change_random": round(float(np.mean(rand_margin_change)), 4),
                "per_descriptor_cos": [round(c, 4) for c in cos_list],
            }
            print(f"[DS-Local] s={s_frac}x: cos={ds_local[f's={s_frac}x']['mean_cos_centered_actual_vs_isolated']} "
                  f"文本/随机居中幅度比={ds_local[f's={s_frac}x']['text_over_random_centered_ratio']} "
                  f"间隔变化 文本={ds_local[f's={s_frac}x']['mean_abs_margin_change_text']:.4f} "
                  f"随机={ds_local[f's={s_frac}x']['mean_abs_margin_change_random']:.4f}")

    result = {
        "dataset": "CUB", "run": "CUB_prior_residual_sel_seed1 (E1 诊断协议收敛快照)",
        "s_scale_real_domain_shift": round(s_scale, 4),
        "ds_reparam": {
            "max_prob_difference": prob_diff,
            "source_acc_before": round(acc_a, 4),
            "source_acc_after_centering": round(acc_b, 4),
            "isolated_L1_loss_drop": round(l1_drop, 5),
            "l1_before": round(float(reg_a.abs().mean()), 5),
            "l1_after_centered_W": round(float(reg_b.abs().mean()), 5),
        },
        "ds_commonmode_isolated": {
            "l1_common_share": round(l1_share_iso, 4),
            "l2_common_energy_share": round(l2_share_iso, 4),
        },
        "ds_local": ds_local,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "ds_local"}, ensure_ascii=False, indent=1))
    print(f"[DS-Local] 详见 JSON (多强度+随机对照)")

    with open(os.path.join(OUT_DIR, "ds_diagnostic_cub.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[saved] {os.path.join(OUT_DIR, 'ds_diagnostic_cub.json')}")


if __name__ == "__main__":
    main()
