"""
DDO-Subspace 事后诊断 (2026-11-25): 分类头输入空间里的各个方向携带多少类别信息。

起因: 正式矩阵 (RESULTS.md §3n 的结果) 里主方向臂 pca8 / pca32 远差于完整方向集与同维随机标架, 且源域精度随剂量
明显下降 (pca32 在 alpha=8 时源域精度 49%)。待检验的假设: LN0(D E^T) 的主方向与类别判别方向重合,
把分类头在这些方向上的响应压到零会损失源域判别力。
**这是看到结果之后才做的事后诊断, 只用来解释, 不作为任何判定依据; 没有预注册。**

只用源域训练集的缓存特征 (cache/<dataset>_train.pt), 不涉及任何目标域数据。对分类头的输入
(概念激活经无 affine 的 LayerNorm 之后的向量 z) 计算:
  类间散度 B = sum_c n_c (mu_c - mu)(mu_c - mu)^T,  类内散度 W = sum_i (z_i - mu_{c(i)})(z_i - mu_{c(i)})^T
  子空间 V (行正交) 捕获的类间散度占比  tr(V B V^T) / tr(B)   —— 随机 k 维子空间的本底约为 k / 311
  (逐方向 eta^2 = Var_c(mean_c p) / Var(p) 也一并输出, 但它的本底不是 1/C, 而是类间散度占总散度的比例,
   本数据为 0.66 —— 区分力弱, 结论以子空间捕获的类间散度占比为准)
对象: LN0(D E^T) 的主方向 (与训练实现同一算法)、零均值超平面内的随机正交标架 (20 个种子)、204 个类平均方向。

产物: experiment_results/DDO-Subspace/class_relevance_cub.json
用法: python ddo_principal_class_relevance.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from args import get_args  # noqa: E402
from model.cbm_models import clip_cbm_orth  # noqa: E402  (只用它的 _principal_rows, 与训练同一算法)

OUT_DIR = os.path.join("experiment_results", "DDO-Subspace")
KS = (1, 2, 4, 8, 16, 32)
N_RANDOM = 20


def ln0(x, eps=1e-5):
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)


def main():
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    ds = __import__("data").get_dataset_classes(args)[0]
    diffs = ds.domain_diffs.float().to(device)                     # (204, C, 768)
    n_t, C, D = diffs.shape
    import clip
    clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)
    with torch.no_grad():
        E = clip_model.encode_text(clip.tokenize(list(ds.concept2id.keys())).to(device)).float()
        E = E / E.norm(dim=-1, keepdim=True)                       # (312, 768)
    del clip_model
    torch.cuda.empty_cache()

    # ---- 源域训练集特征 -> 分类头输入 z (与模型 forward 相同: 归一化, 投影, 无 affine 的 LayerNorm) ----
    cache = torch.load(os.path.join("cache", f"{args.dataset}_train.pt"), map_location="cpu")
    feats = cache["feats"].float().to(device)
    labels = cache["labels"].to(device)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    Z = ln0(feats @ E.T).double()                                  # (N, 312)
    N, d = Z.shape
    mu = Z.mean(0)
    counts = torch.bincount(labels, minlength=C).double()
    sums = torch.zeros(C, d, dtype=torch.float64, device=device).index_add_(0, labels, Z)
    mus = sums / counts[:, None]
    dc = mus - mu
    B = (dc.T * counts) @ dc                                       # 类间散度 (312, 312)
    W = Z.T @ Z - (mus.T * counts) @ mus                           # 类内散度
    trB, trW = torch.trace(B), torch.trace(W)

    def captured(V):
        """V (k, d) 行正交: 捕获的类间 / 类内散度占比"""
        return float(torch.trace(V @ B @ V.T) / trB), float(torch.trace(V @ W @ V.T) / trW)

    def eta2(V):
        """逐方向 eta^2 (k,)"""
        P = Z @ V.T                                                # (N, k)
        pm = (torch.zeros(C, V.size(0), dtype=torch.float64, device=device).index_add_(0, labels, P)
              / counts[:, None])
        ssb = ((pm - P.mean(0)) ** 2).T @ counts
        return ssb / ((P - P.mean(0)) ** 2).sum(0)

    # ---- 方向集 ----
    A = (diffs @ E.T).reshape(-1, E.size(0))
    X = ln0(A)
    Vp, ev = clip_cbm_orth._principal_rows(X, max(KS))             # 主方向 (k, 312), 与训练实现同一算法
    Vp = Vp.double()
    cum = torch.cumsum(ev, 0) / ev.sum()
    mean_dirs = diffs.mean(dim=1)
    mean_dirs = mean_dirs / (mean_dirs.norm(dim=-1, keepdim=True) + 1e-8)
    M = (mean_dirs @ E.T)                                          # (204, 312): mean 臂约束的方向 (不成对)
    # mean 臂方向 (经 LN0) 的张成空间: QR 给出 204 个正交基向量, 但数值秩未必是 204 ——
    # 外审第四轮: 需按奇异值容差确认数值秩, 再谈它覆盖了多少类间散度
    Mln = ln0(M).double()
    sM = torch.linalg.svdvals(Mln)                                  # (204,) 降序
    rank_tol = dict(rel_1e3=int((sM > sM[0] * 1e-3).sum()), rel_1e2=int((sM > sM[0] * 1e-2).sum()),
                    rel_1e1=int((sM > sM[0] * 1e-1).sum()),
                    eff_rank_entropy=None)
    p_ = (sM ** 2) / (sM ** 2).sum()
    rank_tol["eff_rank_entropy"] = round(float(torch.exp(-(p_ * (p_ + 1e-300).log()).sum())), 2)
    rank_tol["s_over_max"] = [round(float(v), 4) for v in sM[[0, 7, 15, 31, 63, 127, 203]]]
    Qm = torch.linalg.qr(Mln.T)[0].T                                # 其张成空间的正交基 (204 行)
    Qm_r = Qm[:rank_tol["rel_1e3"]]                                 # 按数值秩截断 (rel 1e-3)
    g_cache = torch.Generator().manual_seed(0)
    rnd = {}
    for k in KS:
        caps, etas = [], []
        for s in range(N_RANDOM):
            g = torch.Generator().manual_seed(1000 + s)
            R = torch.randn(d, k, generator=g)
            R = R - R.mean(dim=0, keepdim=True)
            R = torch.linalg.qr(R)[0].T.double().to(device)        # 与训练实现相同: 中心化后正交化
            caps.append(captured(R))
            etas.append(eta2(R))
        rnd[k] = {"capB_mean": float(np.mean([c[0] for c in caps])), "capB_sd": float(np.std([c[0] for c in caps])),
                  "capW_mean": float(np.mean([c[1] for c in caps])),
                  "eta2_mean": float(np.mean([float(e.mean()) for e in etas]))}

    out = {"dataset": args.dataset, "n_train": int(N), "n_classes": int(C),
           "trace_B_over_total": float(trB / (trB + trW)),
           "note": "事后诊断 (看到正式矩阵结果之后), 仅用源域训练集特征; 不作为判定依据",
           "principal": {}, "random_centered": {str(k): v for k, v in rnd.items()},
           "mean_dirs_span": {"rank_qr": int(Qm.size(0)), "rank_by_tolerance": rank_tol,
                              "capB_full_basis": captured(Qm)[0], "capW_full_basis": captured(Qm)[1],
                              "capB_rank_truncated": captured(Qm_r)[0], "capW_rank_truncated": captured(Qm_r)[1],
                              "rank_truncated_dim": int(Qm_r.size(0))},
           "principal_eta2_top8": [round(float(v), 5) for v in eta2(Vp[:8])],
           "mean_dirs_eta2": {"mean": round(float(eta2(ln0(M).double()).mean()), 5),
                              "max": round(float(eta2(ln0(M).double()).max()), 5)}}
    for k in KS:
        cb, cw = captured(Vp[:k])
        out["principal"][str(k)] = {"capB": cb, "capW": cw, "spectral_energy_cum": float(cum[k - 1]),
                                    "eta2_mean": float(eta2(Vp[:k]).mean())}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "class_relevance_cub.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"源域训练样本 {N}, 类别 {C}; 类间散度占总散度 {out['trace_B_over_total']:.4f}")
    print(f"{'子空间':10s} {'谱能量累计':>10s} {'类间占比':>10s} {'类内占比':>10s} {'随机(类间,均值±sd)':>22s} {'随机(类内)':>12s} {'本底 k/311':>10s}")
    for k in KS:
        p, r = out["principal"][str(k)], out["random_centered"][str(k)]
        print(f"{'主方向':8s} k={k:<3d} {p['spectral_energy_cum']:10.3f} {p['capB']:10.3f} {p['capW']:10.3f} "
              f"{r['capB_mean']:12.3f} ± {r['capB_sd']:.3f} {r['capW_mean']:12.3f} {k / (d - 1):10.3f}")
    m = out["mean_dirs_span"]
    print(f"{'mean 臂':8s} QR 基={m['rank_qr']:<3d} 数值秩(σ/σmax>1e-3)={m['rank_by_tolerance']['rel_1e3']:<3d} "
          f"类间 {m['capB_full_basis']:.3f} (截断后 {m['capB_rank_truncated']:.3f}) "
          f"类内 {m['capW_full_basis']:.3f} (截断后 {m['capW_rank_truncated']:.3f})")
    rt = m["rank_by_tolerance"]
    print(f"  mean 臂数值秩: σ/σmax > 1e-3: {rt['rel_1e3']} | > 1e-2: {rt['rel_1e2']} | > 1e-1: {rt['rel_1e1']} | "
          f"有效秩(熵): {rt['eff_rank_entropy']} | σ/σmax@[1,8,16,32,64,128,204]={rt['s_over_max']}")
    print("\n读法 (外审第四轮修正): 主方向对类间与类内散度的捕获比例相近, 所以准确的说法是"
          "\"语言域主方向与源域概念空间的高方差结构高度重合\", 不是\"选择性地与类别判别信息重合\"。"
          "做判别保护时须同时看 B 与 W, 不能简单保护所有高方差方向。")
    print("\n逐方向 eta^2 (类间方差 / 总方差):")
    print("  前 8 个主方向:", out["principal_eta2_top8"])
    print("  204 个类平均方向 (经 LN0): 均值 %.5f, 最大 %.5f" % (out["mean_dirs_eta2"]["mean"], out["mean_dirs_eta2"]["max"]))
    print("  随机方向 (k=8, 20 个标架平均): %.5f  —— 与全局类间占比 %.4f 相当, 逐方向 eta^2 没有区分力"
          % (out["random_centered"]["8"]["eta2_mean"], out["trace_B_over_total"]))
    print(f"\n[saved] {os.path.join(OUT_DIR, 'class_relevance_cub.json')}")


if __name__ == "__main__":
    main()
