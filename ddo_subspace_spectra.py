"""
DDO-Subspace 准备分析 (2026-11-25): 各层域方向矩阵的奇异值谱 / 有效秩, 以及 LayerNorm 之后
文本主方向与随机方向的结构和初始正则梯度强度对照。全部只用到文本侧数据, 不涉及任何图像。

四层矩阵 (CUB; 204 个描述符 x 200 个文本类别 x 768 维):
  M1 描述符类平均方向        204   x 768   (逐行归一化; 第四章构造子空间用的就是它)
  M2 全部类别条件方向        40800 x 768   (逐行归一化; 原始 DDO 实际约束的方向)
  M3 概念投影 A = D E^T     40800 x 312   (CBM 瓶颈保留的域信息; E 为冻结概念矩阵)
  M4 无 affine 的 LayerNorm   40800 x 312   (LN0(A); 分类头实际看到的约束输入, 去掉可学习 affine)

LayerNorm 对照 (外审提醒: LN 只能统一逐行尺度, 不能统一方向相关性 / 有效秩 / 梯度强度):
  对 k ∈ {8, 32, 64}, 比较 文本主方向 (M3 的前 k 个右奇异向量) 与 同维随机正交标架,
  均取 ± 成对、过 LN0, 测: 两两 |cos|、有效秩、对随机初始化分类头的正则梯度范数 (20 个初始化)。

产物: experiment_results/DDO-Subspace/spectra_cub.json
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

from args import get_args

OUT_DIR = os.path.join("experiment_results", "DDO-Subspace")
K_LIST = [1, 2, 4, 5, 8, 16, 32, 48, 64, 96, 128, 192, 256]
KS = (8, 32, 64)
N_INITS = 20


def svd_rows(X):
    """X (n, d) float64 -> 奇异值 s (降序), 右奇异向量 V (d 列向量)。用 Gram 矩阵特征分解。"""
    n, d = X.shape
    if n >= d:
        evals, evecs = torch.linalg.eigh(X.T @ X)
        s = evals.clamp_min(0).sqrt().flip(0)
        V = evecs.flip(1)
    else:
        evals, U = torch.linalg.eigh(X @ X.T)
        s = evals.clamp_min(0).sqrt().flip(0)
        U = U.flip(1)
        V = (X.T @ U) / s.clamp_min(1e-30)
    return s, V


def spec_stats(s):
    p = s ** 2 / (s ** 2).sum()
    cum = torch.cumsum(p, 0)
    k_for = lambda f: int(torch.searchsorted(cum, torch.tensor(f)).item()) + 1
    return {
        "n_singular_values": int(s.numel()),
        "sigma_max": round(float(s[0]), 6),
        "cum_energy": {str(k): round(float(cum[min(k, s.numel()) - 1]), 4) for k in K_LIST if k <= s.numel()},
        "k_for_80pct": k_for(0.80), "k_for_90pct": k_for(0.90), "k_for_95pct": k_for(0.95),
        "numerical_rank_rel_1e-6": int((s > s[0] * 1e-6).sum()),
        "numerical_rank_rel_1e-3": int((s > s[0] * 1e-3).sum()),
        "numerical_rank_rel_1e-2": int((s > s[0] * 1e-2).sum()),
        "effective_rank_entropy": round(float(torch.exp(-(p * (p + 1e-300).log()).sum())), 3),
        "stable_rank": round(float((s ** 2).sum() / s[0] ** 2), 3),
        "participation_ratio": round(float(1.0 / (p ** 2).sum()), 3),
    }


def ln0(x, eps=1e-5):
    """无 affine 的 LayerNorm (与 nn.LayerNorm 的归一化部分一致: 有偏方差)"""
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)


def frame_stats(rows2k):
    """rows2k: (2k, d) 已含 ± 成对。返回 +k 行之间的两两 |cos| 与有效秩。
    (2026-11-25 修正: LN0 输出的行范数约为 sqrt(d), 必须先归一化才是余弦 ——
     初版未归一化, 得到的是点积, 出现了 >1 的"余弦"。)"""
    k = rows2k.size(0) // 2
    pos = rows2k[:k] if torch.allclose(rows2k[k:].abs(), rows2k[:k].abs()) else rows2k[::2]
    pos = pos / pos.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    assert float(pos.norm(dim=-1).max()) < 1.0 + 1e-4, "归一化失败"
    g = (pos @ pos.T).abs()
    off = g[~torch.eye(k, dtype=torch.bool)]
    assert float(off.max()) <= 1.0 + 1e-4, f"余弦超过 1: {float(off.max())}"
    s, _ = svd_rows(pos.double())
    p = s ** 2 / (s ** 2).sum()
    return {"pairwise_abs_cos_mean": round(float(off.mean()), 4),
            "pairwise_abs_cos_max": round(float(off.max()), 4),
            "effective_rank_entropy": round(float(torch.exp(-(p * (p + 1e-300).log()).sum())), 3)}


def grad_norm_at_init(Y, seed, n_cls):
    """随机初始化的 LayerNorm+Linear 分类头, 对等权 L1 响应损失 |h(Y)|.mean() 的梯度范数 (对分类头全部参数)"""
    torch.manual_seed(seed)
    dev = Y.device
    head = nn.Sequential(nn.LayerNorm(Y.size(1)), nn.Linear(Y.size(1), n_cls)).to(dev)   # 初始化在 CPU 上生成后搬移
    loss = head(Y).abs().mean()
    grads = torch.autograd.grad(loss, list(head.parameters()))
    g = torch.cat([t.reshape(-1) for t in grads])
    return float(g.norm()), float(loss)


def main():
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device                       # 数据集构造需要 (TrainingSession.__init__ 里也是这样设置的)
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = __import__("data").get_dataset_classes(args)[0]
    diffs = ds.domain_diffs.float().to(device)                     # (204, C, 768) 逐类单位方向
    n_t, C, D = diffs.shape
    concept_names = list(ds.concept2id.keys())

    import clip
    clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)
    with torch.no_grad():
        E = clip_model.encode_text(clip.tokenize(concept_names).to(device)).float()
        E = E / E.norm(dim=-1, keepdim=True)                       # (312, 768)
    del clip_model
    torch.cuda.empty_cache()

    mean_dirs = diffs.mean(dim=1)                                   # (204, 768)
    M1 = (mean_dirs / mean_dirs.norm(dim=-1, keepdim=True)).double().cpu()
    M2 = diffs.reshape(n_t * C, D).double().cpu()
    A = (diffs @ E.T).reshape(n_t * C, -1)                         # (40800, 312)
    M3 = A.double().cpu()
    M4 = ln0(A).double().cpu()

    mats = {"M1_mean_dirs_204x768": M1, "M2_class_conditional_40800x768": M2,
            "M3_concept_proj_40800x312": M3, "M4_ln0_concept_40800x312": M4}
    out, svd = {}, {}
    for name, X in mats.items():
        s, V = svd_rows(X)
        svd[name] = (s, V)
        out[name] = spec_stats(s)
        print(f"[spectra] {name}: {out[name]}")

    # ---- 覆盖关系 ----
    s1, V1 = svd["M1_mean_dirs_204x768"]
    s2, V2 = svd["M2_class_conditional_40800x768"]
    s3, V3 = svd["M3_concept_proj_40800x312"]
    s4, V4 = svd["M4_ln0_concept_40800x312"]
    cover = {}
    for k in KS:
        B = V1[:, :k]                                               # 第四章子空间的构造方式 (类平均方向的前 k 个主方向)
        e2 = float((M2 @ B).norm() ** 2 / M2.norm() ** 2)           # 类别条件方向的能量落在 span(B) 内的比例
        eb, _ = torch.linalg.qr(E.double().cpu() @ B)               # E B 正交化 -> 概念空间
        e3 = float((M3 @ eb).norm() ** 2 / M3.norm() ** 2)
        # M3 / M4 的前 k 个主方向 与 E·span(B) 的主角余弦
        ang3 = torch.linalg.svdvals(V3[:, :k].T @ eb)
        ang4 = torch.linalg.svdvals(V4[:, :k].T @ eb)
        # M1 前 k 与 M2 前 k 的主角余弦 (类平均子空间 vs 类别条件子空间)
        ang12 = torch.linalg.svdvals(V1[:, :k].T @ V2[:, :k])
        cover[str(k)] = {
            "energy_of_M2_in_topk_M1": round(e2, 4),
            "energy_of_M3_in_E_span_topk_M1": round(e3, 4),
            "principal_cos_M3_topk_vs_EB": [round(float(ang3.min()), 4), round(float(ang3.mean()), 4)],
            "principal_cos_M4_topk_vs_EB": [round(float(ang4.min()), 4), round(float(ang4.mean()), 4)],
            "principal_cos_M1_topk_vs_M2_topk": [round(float(ang12.min()), 4), round(float(ang12.mean()), 4)],
        }
        print(f"[cover k={k}] {cover[str(k)]}")

    # ---- LayerNorm 之后的结构与初始正则梯度强度: 文本主方向 vs 随机正交标架 ----
    n_cls = len(ds.classname2id)
    E_cpu = E.double().cpu()
    ln_check = {}
    for k in KS:
        res = {}
        for name, frame in (("text_pca_M3", V3[:, :k].T),                 # 概念空间的主方向 (k, 312)
                            ("text_pcaM1_thru_E", None)):
            if frame is None:
                q, _ = torch.linalg.qr(E_cpu @ V1[:, :k])                  # 类平均方向的主方向经 E 映射并正交化
                frame = q.T
            rows = torch.cat([frame, -frame], 0)                           # ± 成对
            Y = ln0(rows)
            fs = frame_stats(Y)
            gn = [grad_norm_at_init(Y.float(), sd, n_cls) for sd in range(N_INITS)]
            res[name] = {"frame": fs,
                         "grad_norm_mean": round(float(np.mean([g for g, _ in gn])), 6),
                         "grad_norm_sd": round(float(np.std([g for g, _ in gn])), 6),
                         "loss_at_init_mean": round(float(np.mean([l for _, l in gn])), 6)}
        rnd_frames, rnd_grad = [], []
        for sd in range(N_INITS):
            g = torch.Generator().manual_seed(1000 + sd)
            R = torch.linalg.qr(torch.randn(312, k, generator=g, dtype=torch.float64))[0].T   # 随机正交标架 (k, 312)
            Y = ln0(torch.cat([R, -R], 0))
            gn, l = grad_norm_at_init(Y.float(), sd, n_cls)
            rnd_grad.append(gn)
            rnd_frames.append(frame_stats(Y))
        res["random_orthonormal"] = {
            "frame": {key: round(float(np.mean([f[key] for f in rnd_frames])), 4) for key in rnd_frames[0]},
            "grad_norm_mean": round(float(np.mean(rnd_grad)), 6),
            "grad_norm_sd": round(float(np.std(rnd_grad)), 6)}
        # 原始完整集合 (40800 行过 LN0) 的初始梯度范数, 作为尺度参照
        full_g = []
        Yfull = ln0(A).float()
        for sd in range(N_INITS):
            g, _ = grad_norm_at_init(Yfull.to(device), sd, n_cls)
            full_g.append(g)
        res["full_40800_rows"] = {"grad_norm_mean": round(float(np.mean(full_g)), 6),
                                  "grad_norm_sd": round(float(np.std(full_g)), 6)}
        for name in ("text_pca_M3", "text_pcaM1_thru_E", "random_orthonormal"):
            res[name]["grad_norm_ratio_to_full"] = round(res[name]["grad_norm_mean"] / res["full_40800_rows"]["grad_norm_mean"], 4)
        ln_check[str(k)] = res
        print(f"[LN k={k}] grad-norm  text(M3)={res['text_pca_M3']['grad_norm_mean']:.4f}  "
              f"text(M1->E)={res['text_pcaM1_thru_E']['grad_norm_mean']:.4f}  "
              f"random={res['random_orthonormal']['grad_norm_mean']:.4f}  "
              f"full40800={res['full_40800_rows']['grad_norm_mean']:.4f} | "
              f"eff.rank text={res['text_pca_M3']['frame']['effective_rank_entropy']} "
              f"random={res['random_orthonormal']['frame']['effective_rank_entropy']}")

    # ---- 谱形 (供画图), 归一化奇异值的前 64 个 ----
    spec = {name: [round(float(v), 8) for v in s[:64] / s[0]] for name, (s, _) in svd.items()}
    result = {"dataset": "CUB", "n_descriptors": n_t, "n_text_classes": C, "spectra": out,
              "coverage": cover, "ln_check": ln_check, "normalized_sigma_first64": spec}
    with open(os.path.join(OUT_DIR, "spectra_cub.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[saved] {os.path.join(OUT_DIR, 'spectra_cub.json')}")


if __name__ == "__main__":
    main()
