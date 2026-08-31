"""
PC top-loading 语义分析 (专家第 14 节建议):
对前 5 个 principal directions, 找 |<d_i, v_k>| 最大的 prompts,
回答 "SVD 压出来的到底是什么" — 第四章解释性证据。

用法: python3 pc_loading_analysis.py --dataset CUB   (无需 GPU, 读已保存的 artifacts)
"""
import os
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'AWA2'])
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()

    art_path = f"logs/domain_structure/{args.dataset}/domain_artifacts.pt"
    art = torch.load(art_path)
    normed_diffs, prompts = art['normed_diffs'], art['prompts']

    # prompt 级方向 (类平均, 归一化) — 与诊断脚本一致
    dirs = normed_diffs.mean(dim=1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    D = dirs.numpy().astype(np.float64)                    # (204, 768)

    # SVD
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    V = Vt.T                                               # (768, 204) 右奇异向量

    out_dir = f"logs/domain_structure/{args.dataset}"
    report = []
    report.append(f"===== PC Top-{args.topk} Loading Prompts ({args.dataset}) =====\n")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for k in range(5):
        v = V[:, k]                                         # (768,)
        loadings = np.abs(D @ v)                            # (204,) 每个 prompt 在 PC_k 上的载荷
        order = np.argsort(loadings)[::-1]
        ev = (S[k] ** 2) / (S ** 2).sum()

        report.append(f"--- PC{k+1}  (EV = {ev:.2%}) ---")
        for rank, idx in enumerate(order[:args.topk]):
            sign = '+' if (D[idx] @ v) > 0 else '-'
            report.append(f"  [{sign}] {loadings[idx]:.4f}  {prompts[idx]}")
        report.append("")

        # 画 loading 分布
        axes[k].hist(loadings, bins=40, color=f'C{k}')
        axes[k].set_title(f"PC{k+1} (EV={ev:.1%})")
        axes[k].set_xlabel("|loading|")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pc_loading_hist.png"), dpi=150)
    plt.close(fig)

    text = "\n".join(report)
    print(text)
    with open(os.path.join(out_dir, "pc_top_loading.txt"), 'w') as f:
        f.write(text)
    print(f"\nSaved to {out_dir}/pc_top_loading.txt, pc_loading_hist.png")


if __name__ == '__main__':
    main()
