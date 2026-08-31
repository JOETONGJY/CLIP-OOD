"""
专家第五轮要求的两个免图像结构分析:
  A. 跨数据集 subspace 相似性 (principal angles / subspace overlap)
     — 回答: 不同类别集合诱导出的 domain subspace 是否在几何上高度一致
  B. Prompt reconstruction analysis (explained ratio r_i, 不同 τ)
     — 回答: 全局能量保留 80% 落实到 prompt 级是什么样, 哪些语义 prompt 被表达最好/最差

无需 GPU, 直接读 logs/domain_structure/{CUB,AWA2}/domain_artifacts.pt。

用法: python3 subspace_geometry_analysis.py
"""
import os
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_dirs(dataset):
    art = torch.load(f"logs/domain_structure/{dataset}/domain_artifacts.pt")
    dirs = art['normed_diffs'].mean(dim=1).float()          # (N, D) 类平均
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    return dirs, art['prompts'], art['reliability']


def build_basis(dirs, tau):
    """SVD + energy-threshold 自适应选维, 返回 (B, m, S)"""
    D = dirs.numpy().astype(np.float64)
    U, S, Vh = np.linalg.svd(D, full_matrices=False)
    cum_ev = np.cumsum(S ** 2) / (S ** 2).sum()
    m = int(np.searchsorted(cum_ev, tau) + 1)
    return Vh[:m].T, m, S                                     # (D, m)


def principal_angles(B1, B2):
    """两个子空间的 principal angles 的 cos 值 (SVD of B1^T B2 的奇异值)"""
    M = B1.T @ B2                                              # (m1, m2)
    U, sv, Vt = np.linalg.svd(M)
    return sv                                                  # cos(theta_k), k=1..min(m1,m2)


def subspace_overlap(B1, B2):
    """‖B1^T B2‖_F^2 / min(m1,m2)  ∈ [0,1]"""
    m = min(B1.shape[1], B2.shape[1])
    return (np.linalg.norm(B1.T @ B2, 'fro') ** 2) / m


def main():
    out_dir = "logs/domain_structure"
    report = []
    report.append("=" * 70)
    report.append("A. 跨数据集 Domain Subspace 几何一致性 (CUB vs AWA2, τ=0.8)")
    report.append("=" * 70)

    dirs_cub, prompts_cub, rel_cub = load_dirs("CUB")
    dirs_awa, prompts_awa, rel_awa = load_dirs("AWA2")

    B_cub, m_cub, _ = build_basis(dirs_cub, 0.8)
    B_awa, m_awa, _ = build_basis(dirs_awa, 0.8)
    report.append(f"CUB subspace: m={m_cub} (768→{m_cub})")
    report.append(f"AWA2 subspace: m={m_awa} (768→{m_awa})")

    pa = principal_angles(B_cub, B_awa)
    ov = subspace_overlap(B_cub, B_awa)
    report.append(f"\nPrincipal angles (cos, 前 10 个): {np.round(pa[:10], 4)}")
    report.append(f"mean cos(principal angle) = {pa.mean():.4f}")
    report.append(f"min  cos(principal angle) = {pa.min():.4f}")
    report.append(f"subspace overlap ‖B1ᵀB2‖²_F/m = {ov:.4f}")
    report.append("\n解读: mean cos ≈ 1 表示两个数据集的 domain subspace 几何高度一致")

    # 多个 τ 下的稳定性
    report.append("\n不同 τ 下的跨数据集一致性:")
    report.append(f"{'τ':<6}{'m(CUB)':<8}{'m(AWA2)':<9}{'mean cos':<10}{'overlap':<9}")
    for tau in [0.6, 0.7, 0.8, 0.9, 0.95]:
        B1, m1, _ = build_basis(dirs_cub, tau)
        B2, m2, _ = build_basis(dirs_awa, tau)
        pa = principal_angles(B1, B2)
        ov = subspace_overlap(B1, B2)
        report.append(f"{tau:<6}{m1:<8}{m2:<9}{pa.mean():<10.4f}{ov:<9.4f}")

    # ===== B. Prompt Reconstruction Analysis =====
    report.append("\n" + "=" * 70)
    report.append("B. Prompt Reconstruction Analysis (explained ratio r_i = ‖Bᵀd_i‖²/‖d_i‖²)")
    report.append("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, dirs, prompts, rel) in [
        (axes[0], ("CUB", dirs_cub, prompts_cub, rel_cub)),
        (axes[1], ("AWA2", dirs_awa, prompts_awa, rel_awa)),
    ]:
        D = dirs.numpy().astype(np.float64)
        U, S, Vh = np.linalg.svd(D, full_matrices=False)
        cum_ev = np.cumsum(S ** 2) / (S ** 2).sum()

        # 不同 τ 下的平均 explained ratio
        taus = [0.6, 0.7, 0.8, 0.9, 0.95]
        mean_r = []
        for tau in taus:
            m = int(np.searchsorted(cum_ev, tau) + 1)
            B = Vh[:m].T
            r = np.sum((D @ B) ** 2, axis=1)                   # ‖Bᵀd_i‖² (d_i 已归一化)
            mean_r.append(r.mean())
        ax.plot(taus, mean_r, 'o-', label='mean r_i')
        ax.plot(taus, taus, 's--', color='gray', label='y=τ (global CE)')
        ax.set_xlabel('τ'); ax.set_ylabel('mean explained ratio')
        ax.set_title(f"{name}: prompt-level reconstruction")
        ax.legend()
        report.append(f"\n[{name}] 不同 τ 下的平均 prompt-level explained ratio:")
        for tau, r in zip(taus, mean_r):
            report.append(f"  τ={tau}: mean r_i = {r:.4f}")

        # τ=0.8 下的 Top/Bottom prompts
        m = int(np.searchsorted(cum_ev, 0.8) + 1)
        B = Vh[:m].T
        r = np.sum((D @ B) ** 2, axis=1)
        order = np.argsort(r)
        report.append(f"\n[{name}] Top-10 (被 subspace 表达最好的 prompt, τ=0.8):")
        for i in order[::-1][:10]:
            report.append(f"  r={r[i]:.4f}  rel={rel[i]:.4f}  {prompts[i]}")
        report.append(f"[{name}] Bottom-10 (表达最差的 prompt, τ=0.8):")
        for i in order[:10]:
            report.append(f"  r={r[i]:.4f}  rel={rel[i]:.4f}  {prompts[i]}")

        # r 与 reliability 的相关 (顺带: 两个度量是否捕捉不同东西)
        corr = np.corrcoef(r, rel.numpy())[0, 1]
        report.append(f"[{name}] corr(r_i, reliability_i) = {corr:.4f} "
                      f"(低相关 => reconstruction 与 reliability 是互补的两种 prompt 视角)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruction_analysis.png"), dpi=150)
    plt.close(fig)

    text = "\n".join(report)
    print(text)
    with open(os.path.join(out_dir, "subspace_geometry_report.txt"), 'w') as f:
        f.write(text)
    print(f"\nSaved: {out_dir}/subspace_geometry_report.txt, reconstruction_analysis.png")


if __name__ == '__main__':
    main()
