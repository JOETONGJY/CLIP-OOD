"""
TU-Diagnostic + TU-VirtualStep (2026-11-24, 原 G-B)

研究问题: 当前 RW-DDO 的权重学习 (r 先验 + δ 残差) 是否与描述符的"任务效用"
一致? 任务效用 = 沿该描述符的正交约束梯度更新分类器, 对代理域分类损失的
一阶改善 (u_i = <∇θℓ_i, ∇θL_proxy>)。

设计 (外审 2026-11-24 要求):
  Proxy-1 类别学习效用: held-out 源域验证样本的 CE (source_test 缓存)
  Proxy-2 语言域扰动效用: 特征扰动 z̃ = norm(z + s·d_{j,y}) 上的 CE,
      描述符留出 (u_i 聚合时排除 j=i, 防循环论证), 语义保持筛选
      (CLIP 原型 margin>0 的扰动样本才计入)
  TU-VirtualStep: θ_i' = θ - η·∇θℓ_i, 实测 Δ_i = L_proxy(θ) - L_proxy(θ_i'),
      检验一阶效用排序能否预测真实单步更新效果 (固定 η 与归一化 η 双口径)
  稳定性: 跨 Proxy / 跨 epoch (25/50/99 快照) 的效用排序一致性
  相关分析: corr(u, r), corr(u, δ_final), corr(u, ℓ)

全部基于缓存特征 + 收敛快照, 零训练。产物:
  experiment_results/TU-Diagnostic/tu_diagnostic_cub.json
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

OUT_DIR = os.path.join("experiment_results", "TU-Diagnostic")
SNAP_DIR = os.path.join("experiment_results", "07_权重逃逸诊断", "cls_snapshots")
EPOCHS = [25, 50, 99]
N_PROXY_SAMPLES = 2000
N_PERM_SEED = 0


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def head_from_sd(sd, k):
    head = nn.Sequential(nn.LayerNorm(312), nn.Linear(312, k))
    head[0].load_state_dict({"weight": sd["1.weight"], "bias": sd["1.bias"]})
    head[1].load_state_dict({"weight": sd["2.weight"], "bias": sd["2.bias"]})
    return head


def flat_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def main():
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(N_PERM_SEED)

    train_dataset = __import__("data").get_dataset_classes(args)[0]
    concept_names = list(train_dataset.concept2id.keys())
    diffs = train_dataset.domain_diffs.float().to(device)          # (204, C, 768)
    C = diffs.size(1)
    n_t = diffs.size(0)

    import clip
    clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)
    with torch.no_grad():
        toks = clip.tokenize(concept_names).to(device)
        E = clip_model.encode_text(toks).float()
        E = E / E.norm(dim=-1, keepdim=True)
        # 源类文本原型 (photo 模板) 用于语义保持筛选
        texts_s = [tp.format(cn) for tp, cn in
                   zip(["a photo of a {}."] * C, list(train_dataset.classname2id.keys()))]
        S_proto = clip_model.encode_text(clip.tokenize(texts_s).to(device)).float()
        S_proto = S_proto / S_proto.norm(dim=-1, keepdim=True)
    del clip_model
    torch.cuda.empty_cache()

    # ===== 收敛分类器快照 =====
    snaps = {}
    for e in EPOCHS:
        p = os.path.join(SNAP_DIR, f"epoch_{e:04d}.pt")
        snaps[e] = torch.load(p, map_location="cpu", weights_only=False)["cls"]

    # ===== 缓存特征 =====
    tr = torch.load("cache/CUB_train.pt", map_location="cpu", weights_only=False)
    z_tr = torch.nn.functional.normalize(tr["feats"].float(), dim=-1).to(device)
    y_tr = tr["labels"].long().to(device)
    te = torch.load("cache/CUB_source_test.pt", map_location="cpu", weights_only=False)
    z_val = torch.nn.functional.normalize(te["feats"].float(), dim=-1).to(device)
    y_val = te["labels"].long().to(device)
    rng = torch.Generator().manual_seed(N_PERM_SEED)
    val_idx = torch.randperm(z_val.size(0), generator=rng)[:N_PROXY_SAMPLES]
    z_P1, y_P1 = z_val[val_idx], y_val[val_idx]
    tr_idx = torch.randperm(z_tr.size(0), generator=rng)[:N_PROXY_SAMPLES]
    z_P2, y_P2 = z_tr[tr_idx], y_tr[tr_idx]

    a_P1 = z_P1 @ E.T
    a_P2_base = z_P2 @ E.T

    results = {"dataset": "CUB", "epochs": EPOCHS, "s_source": None}
    epoch_u = {}

    for e in EPOCHS:
        head = head_from_sd(snaps[e], K := snap_k(snaps[e])).to(device)
        params = [p for p in head.parameters() if p.requires_grad]

        # ===== 逐描述符正交损失梯度 g_i =====
        P = diffs @ E.T                                          # (204, C, 312)
        reg = head(P)                                            # (204, C, K)
        g_list = []
        for i in range(n_t):
            g = torch.autograd.grad(reg[i].abs().mean(), params, retain_graph=(i < n_t - 1))
            g_list.append(flat_grads(g).detach())
        G = torch.stack(g_list)                                  # (204, D_theta)
        G_n = torch.nn.functional.normalize(G, dim=-1)

        # ===== Proxy-1: held-out 源域验证 CE =====
        ce_P1 = F.cross_entropy(head(a_P1), y_P1)
        g_P1 = flat_grads(torch.autograd.grad(ce_P1, params)).detach()
        g_P1_n = torch.nn.functional.normalize(g_P1, dim=0)

        # ===== Proxy-2: 语言域扰动 CE (语义筛选, 逐 j 梯度) =====
        s_scale = torch.tensor(0.5456)                            # 真实域偏移尺度 (DS-Diagnostic 实测)
        ce_P2_list, keep_rates = [], []
        for j in range(n_t):
            d_jy = diffs[j][y_P2]                                # (M, 768)
            z_pert = torch.nn.functional.normalize(z_P2 + 0.5456 * d_jy, dim=-1)
            # 语义保持筛选: CLIP 原型 margin > 0
            with torch.no_grad():
                sims_pert = z_pert @ S_proto.T                   # (M, C)
                gt = sims_pert.gather(1, y_P2.unsqueeze(1)).squeeze(1)
                others = sims_pert.scatter(1, y_P2.unsqueeze(1), -1e9).max(dim=-1).values
                keep = (gt - others) > 0
            keep_rates.append(float(keep.float().mean()))
            if keep.sum() < 10:
                ce_P2_list.append(None)
                continue
            ce_j = F.cross_entropy(head(z_pert @ E.T)[keep], y_P2[keep])
            g_j = flat_grads(torch.autograd.grad(ce_j, params)).detach()
            g_jn = torch.nn.functional.normalize(g_j, dim=0)
            ce_P2_list.append(g_jn)

        valid_j = [j for j, g in enumerate(ce_P2_list) if g is not None]
        # (2026-11-24 修复: valid 条目压缩后索引错位 → 按 j 建字典)
        g_P2_by_j = {j: ce_P2_list[j].cpu() for j in valid_j}
        dropped_j = [j for j in range(n_t) if j not in valid_j]
        # ===== 效用 u_i =====
        dot_P1 = G @ g_P1                                        # (204,)
        cos_P1 = G_n @ g_P1_n
        # Proxy-2 留出聚合: u_i = mean_{j≠i} cos(g_i, g_P2(j))  (描述符留出防循环论证)
        cos_P2 = np.zeros(n_t)
        Gn_cpu = torch.nn.functional.normalize(G, dim=-1).cpu()
        for i in range(n_t):
            js = [j for j in valid_j if j != i]
            cos_P2[i] = float(np.mean([(Gn_cpu[i] @ g_P2_by_j[j]).item() for j in js]))
        dot_P2 = np.zeros(n_t)
        G_cpu = G.detach().cpu()
        for i in range(n_t):
            js = [j for j in valid_j if j != i]
            dot_P2[i] = float(np.mean([(G_cpu[i] @ g_P2_by_j[j]).item() for j in js]))

        # ===== 相关分析 =====
        # δ_final 与 ℓ_i (来自最终快照)
        wl = os.path.join("logs", "weights", "CUB_prior_residual_sel_seed1", "epoch_0099.pt")
        wl_d = torch.load(wl, map_location="cpu", weights_only=False)
        delta_final = wl_d["residual"].numpy()
        ell_i = reg.abs().mean(dim=(1, 2)).detach().cpu().numpy()

        epoch_u[e] = {
            "u_P1_cos": cos_P1.cpu().numpy(),
            "u_P2_cos_meanLeaveOut": cos_P2,
            "u_P1_dot": dot_P1.cpu().numpy(),
            "corr_uP1_r": round(spearman(cos_P1.cpu().numpy(), wl_d["reliability_prior"].numpy()), 4),
            "corr_uP2_r": round(spearman(cos_P2, wl_d["reliability_prior"].numpy()), 4),
            "corr_uP1_delta": round(spearman(cos_P1.cpu().numpy(), delta_final), 4),
            "corr_uP2_delta": round(spearman(cos_P2, delta_final), 4),
            "corr_uP1_ell": round(spearman(cos_P1.cpu().numpy(), ell_i), 4),
            "corr_uP2_ell": round(spearman(cos_P2, ell_i), 4),
            "corr_uP1_uP2": round(spearman(cos_P1.cpu().numpy(), cos_P2), 4),
            "keep_rate_P2_mean": round(float(np.mean(keep_rates)), 4),
            "keep_rate_P2_min": round(float(np.min(keep_rates)), 4),
            "dropped_descriptors": dropped_j,
            "dropped_texts": [target_text_prompts[j] for j in dropped_j],
            "_g_P1": g_P1.cpu(), "_g_P1n": g_P1_n.cpu(),
            "_head_sd": {
                "1.weight": head[0].weight.detach().cpu().clone(),
                "1.bias": head[0].bias.detach().cpu().clone(),
                "2.weight": head[1].weight.detach().cpu().clone(),
                "2.bias": head[1].bias.detach().cpu().clone(),
            },
            "_a_P1": a_P1.detach().cpu(), "_y_P1": y_P1.cpu(),
            "_reg_ell": ell_i,
        }

    # ===== 跨 epoch 效用排序稳定性 =====
    es = {}
    for e_a in EPOCHS:
        for e_b in EPOCHS:
            if e_a < e_b:
                key = f"{e_a}|{e_b}"
                es[key] = round(spearman(epoch_u[e_a]["u_P1_cos"], epoch_u[e_b]["u_P1_cos"]), 4)
    results["epoch_rank_stability_uP1"] = es

    # ===== TU-VirtualStep (最终快照, Proxy-1) =====
    fin = epoch_u[99]
    head99 = head_from_sd(fin["_head_sd"], snap_k(snaps[99])).to(device)
    theta0 = [p.detach().clone() for p in head99.parameters()]
    a_P1_dev = fin["_a_P1"].to(device)
    y_P1_dev = fin["_y_P1"].to(device)

    def ce_at(params_list):
        with torch.no_grad():
            ln = nn.LayerNorm(312).to(device)
            ln.load_state_dict({"weight": params_list[0], "bias": params_list[1]})
            ln_w, ln_b = params_list[0].to(device), params_list[1].to(device)
            W, b = params_list[2].to(device), params_list[3].to(device)
            x = torch.nn.functional.layer_norm(a_P1_dev, (312,), ln_w, ln_b)
            logits = x @ W.T + b
            return float(F.cross_entropy(logits, y_P1_dev))

    base_ce = ce_at(theta0)
    # 重新计算最终快照的 g_i (设备上)
    head = head99
    params = [p for p in head.parameters() if p.requires_grad]
    P = diffs @ E.T
    reg = head(P)
    g_vecs = []
    for i in range(n_t):
        g = torch.autograd.grad(reg[i].abs().mean(), params, retain_graph=(i < n_t - 1))
        g_vecs.append(flat_grads(g).detach())
    G_final = torch.stack(g_vecs)

    vs = {"eta_fixed": 1e-4, "eta_normalized_frac": 0.01}
    deltas_fixed, deltas_norm = [], []
    theta_flat0 = torch.cat([t.reshape(-1) for t in theta0]).to(device)
    theta_norm0 = float(torch.norm(theta_flat0))
    shapes = [t.shape for t in theta0]
    for i in range(n_t):
        g = G_final[i]
        for eta, store in [(1e-4, deltas_fixed), (0.01 * theta_norm0, deltas_norm)]:
            gn = g / (g.norm() + 1e-12)
            step = eta * gn if store is deltas_norm else eta * g
            theta_new = theta_flat0 - step
            parts = []
            off = 0
            for sh in shapes:
                n = int(np.prod(sh))
                parts.append(theta_new[off:off + n].reshape(sh))
                off += n
            deltas_i = base_ce - ce_at(parts)
            store.append(deltas_i)
    vs["spearman_u_vs_delta_fixedEta"] = round(spearman(fin["u_P1_cos"], deltas_fixed), 4)
    vs["spearman_u_vs_delta_normEta"] = round(spearman(fin["u_P1_cos"], deltas_norm), 4)
    vs["spearman_delta_fixedEta_vs_normEta"] = round(spearman(deltas_fixed, deltas_norm), 4)
    vs["frac_positive_delta"] = round(float(np.mean(np.array(deltas_fixed) > 0)), 4)
    results["tu_virtualstep"] = vs
    results["epoch_u_summary"] = {
        str(e): {k: (v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v)
                 for k, v in d.items() if not k.startswith("_")}
        for e, d in epoch_u.items()}

    # 清理大对象后落盘
    for e in EPOCHS:
        epoch_u[e].pop("_a_P1", None)
        epoch_u[e].pop("_y_P1", None)
        epoch_u[e].pop("_head_sd", None)
    results["u_vectors"] = {str(e): {k: v.tolist() if isinstance(v, torch.Tensor) else v
                                     for k, v in epoch_u[e].items()
                                     if k in ("u_P1_cos", "u_P2_cos_meanLeaveOut", "u_P1_dot")}
                            for e in EPOCHS}
    def _np_default(o):
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    with open(os.path.join(OUT_DIR, "tu_diagnostic_cub.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=_np_default)
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("epoch_rank_stability_uP1", "tu_virtualstep",
                               "epoch_u_summary")}, ensure_ascii=False, indent=1))
    print(f"[saved] {os.path.join(OUT_DIR, 'tu_diagnostic_cub.json')}")


def snap_k(sd):
    return sd["2.bias"].numel()


if __name__ == "__main__":
    main()
