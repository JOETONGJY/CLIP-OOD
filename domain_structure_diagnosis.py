"""
第四章动机诊断 (专家要求的两个免训练诊断):
  A. 204×204 prompt 方向 cosine 相似度矩阵 —— redundancy / cluster 证据
  B. SVD spectrum + EV(k) —— 低秩性 -> domain subspace 的动机
  顺带: 保存 reliability scores (第三章 Top/Bottom 语义分析用)

不依赖图片数据集, 只需: 类名 (classes.txt) + 204 prompts + CLIP 文本编码器。

用法:
    CUDA_VISIBLE_DEVICES=2 python3 domain_structure_diagnosis.py --dataset CUB
    CUDA_VISIBLE_DEVICES=2 python3 domain_structure_diagnosis.py --dataset AWA2
"""
import os
import argparse
import numpy as np
import torch
import clip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prompts.prompt200new import source_text_prompts, target_text_prompts


def load_class_names(dataset):
    """读真实类名 (与数据集代码同一解析规则)
    CUB: classes.txt 在未下载的数据集包内, 改从 cub_train.txt 的路径提取
         (路径形如 "001.Black_footed_Albatross/xxx.jpg", 与 cub_data.py 解析规则一致)
    AWA2: data/awa2/classes.txt 仓库自带
    """
    if dataset == "CUB":
        seen = {}
        with open("data/CUB/original_annos/cub_train.txt") as f:
            for line in f:
                path = line.split(",")[0]
                dirname = path.split("/")[0]              # "001.Black_footed_Albatross"
                if dirname not in seen:
                    seen[dirname] = int(line.split(",")[1])
        # 按 label 排序, 解析成与 cub_data.py 相同的类名
        items = sorted(seen.items(), key=lambda x: x[1])
        names = [d[4:].replace("_", " ").lower().rstrip() for d, _ in items]
    elif dataset == "AWA2":
        with open("data/awa2/classes.txt") as f:
            names = [x.split(" ")[1].rstrip() for x in f.readlines()]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    names = [n for n in names if n]
    return names


@torch.no_grad()
def encode_all(clip_model, texts, device, batch_size=2048):
    """批量编码文本, 返回归一化嵌入 (n, D)"""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        e = clip_model.encode_text(tokens).float()
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def compute_domain_diffs(clip_model, src_prompts, tgt_prompts, class_names, device):
    """
    返回:
      normed_diffs: (N, K, D)  每个 prompt 每个类的归一化迁移方向
      reliability:  (N,)        类间方向一致性
    """
    src_prompt = src_prompts[0]
    N, K = len(tgt_prompts), len(class_names)

    # source 嵌入 (K, D)
    src_texts = [src_prompt.format(c) for c in class_names]
    src_emb = encode_all(clip_model, src_texts, device)          # (K, D)

    # target 嵌入批量编码 (N*K, D)
    tgt_texts = [p.format(c) for p in tgt_prompts for c in class_names]
    tgt_emb = encode_all(clip_model, tgt_texts, device)          # (N*K, D)
    tgt_emb = tgt_emb.view(N, K, -1)

    # diffs + 归一化
    diffs = tgt_emb - src_emb.unsqueeze(0)                        # (N, K, D)
    normed = diffs / (diffs.norm(dim=-1, keepdim=True) + 1e-8)

    # reliability: 类间平均 cosine (去对角线)
    reliability = torch.zeros(N)
    for i in range(N):
        cs = normed[i] @ normed[i].T                              # (K, K)
        mask = ~torch.eye(K, dtype=torch.bool)
        reliability[i] = cs[mask].mean()
    return normed, reliability


def prompt_direction(normed_diffs):
    """prompt 级方向: 类平均后归一化 (N, D)"""
    d = normed_diffs.mean(dim=1)                                  # (N, D)
    return d / d.norm(dim=-1, keepdim=True)


def diagnosis_cosine(dirs, out_dir, prompts):
    """诊断 A: 204×204 cosine matrix + 统计 + 聚类"""
    N = dirs.shape[0]
    cos_mat = dirs @ dirs.T                                       # (N, N)
    off = cos_mat[~torch.eye(N, dtype=torch.bool)]

    stats = {
        'N': N,
        'cos_mean': off.mean().item(),
        'cos_median': off.median().item(),
        'cos_p90': torch.quantile(off, 0.90).item(),
        'cos_p99': torch.quantile(off, 0.99).item(),
        'cos_max': off.max().item(),
        'frac_gt_07': (off > 0.7).float().mean().item(),
        'frac_gt_09': (off > 0.9).float().mean().item(),
    }

    # 热图
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_mat.numpy(), cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"Prompt Direction Cosine Similarity ({N}x{N})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cosine_matrix.png"), dpi=150)
    plt.close(fig)

    np.save(os.path.join(out_dir, "cosine_matrix.npy"), cos_mat.numpy())
    with open(os.path.join(out_dir, "cosine_stats.txt"), 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    # 找最相似的 prompt 对 (redundancy 证据)
    cm = cos_mat.clone()
    cm.fill_diagonal_(-2)
    flat_idx = cm.argmax()
    i, j = flat_idx // N, flat_idx % N
    stats['most_similar_pair'] = (prompts[i], prompts[j], cm[i, j].item())

    return cos_mat, stats


def diagnosis_svd(dirs, out_dir, ks=(5, 10, 20, 40, 60, 100)):
    """诊断 B: SVD spectrum + EV(k) 低秩性"""
    D = dirs.numpy().astype(np.float64)                            # (N, D)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)

    ev = (S ** 2) / (S ** 2).sum()                                # 每个方向的能量占比
    cum_ev = np.cumsum(ev)

    ev_at = {k: float(cum_ev[k - 1]) for k in ks if k <= len(S)}

    # 谱图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(S[:100], marker='.', linestyle='-', markersize=3)
    axes[0].set_title("Singular Value Spectrum")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("σ")
    axes[1].plot(cum_ev[:100], marker='.', linestyle='-', markersize=3)
    axes[1].set_title("Cumulative Explained Variance EV(k)")
    axes[1].set_xlabel("k (top singular directions)")
    axes[1].set_ylabel("EV(k)")
    axes[1].axhline(0.8, color='r', ls='--', lw=0.8, label='80%')
    axes[1].axhline(0.92, color='g', ls='--', lw=0.8, label='92%')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "svd_spectrum.png"), dpi=150)
    plt.close(fig)

    np.save(os.path.join(out_dir, "singular_values.npy"), S)
    np.save(os.path.join(out_dir, "cum_ev.npy"), cum_ev)
    with open(os.path.join(out_dir, "svd_stats.txt"), 'w') as f:
        for k, v in ev_at.items():
            f.write(f"EV({k}) = {v:.4f}\n")
        # 最小 m 使 EV>=80% / 92%
        m80 = int(np.searchsorted(cum_ev, 0.80) + 1)
        m92 = int(np.searchsorted(cum_ev, 0.92) + 1)
        f.write(f"min m for EV>=80%: {m80}\n")
        f.write(f"min m for EV>=92%: {m92}\n")

    return ev_at, S, cum_ev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'AWA2'])
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[{args.dataset}] Loading CLIP ViT-L/14 on {device}...")
    clip_model, _ = clip.load('ViT-L/14', device=device)
    clip_model.eval()

    class_names = load_class_names(args.dataset)
    print(f"[{args.dataset}] {len(class_names)} classes, {len(target_text_prompts)} target prompts")

    out_dir = f"logs/domain_structure/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    # ===== 核心计算: domain diffs + reliability =====
    print("Encoding prompts (text only)...")
    normed_diffs, reliability = compute_domain_diffs(
        clip_model, source_text_prompts, target_text_prompts, class_names, device)
    print(f"normed_diffs: {tuple(normed_diffs.shape)}")

    torch.save({
        'normed_diffs': normed_diffs,          # (N, K, D)
        'reliability': reliability,            # (N,)
        'prompts': target_text_prompts,
        'class_names': class_names,
    }, os.path.join(out_dir, "domain_artifacts.pt"))
    print(f"Saved domain_artifacts.pt")

    # ===== 诊断 A: cosine matrix =====
    print("\n===== Diagnosis A: 204x204 cosine similarity =====")
    dirs = prompt_direction(normed_diffs)
    cos_mat, cstats = diagnosis_cosine(dirs, out_dir, target_text_prompts)
    for k, v in cstats.items():
        print(f"  {k}: {v}")

    # ===== 诊断 B: SVD spectrum =====
    print("\n===== Diagnosis B: SVD spectrum / low-rank =====")
    ev_at, S, cum_ev = diagnosis_svd(dirs, out_dir)
    for k, v in ev_at.items():
        print(f"  EV({k}) = {v:.4f}")
    m80 = int(np.searchsorted(cum_ev, 0.80) + 1)
    m92 = int(np.searchsorted(cum_ev, 0.92) + 1)
    print(f"  min m for EV>=80%: {m80}")
    print(f"  min m for EV>=92%: {m92}")

    # ===== 顺带: reliability Top/Bottom (第三章用) =====
    print("\n===== Reliability Top-10 / Bottom-10 =====")
    order = torch.argsort(reliability, descending=True)
    with open(os.path.join(out_dir, "reliability_top_bottom.txt"), 'w') as f:
        f.write("Top-10 (most reliable):\n")
        for i in order[:10]:
            line = f"  {reliability[i]:.4f}  {target_text_prompts[i]}"
            print(line); f.write(line + "\n")
        f.write("\nBottom-10 (least reliable):\n")
        bottom = torch.flip(order[-10:], dims=[0])
        for i in bottom:
            line = f"  {reliability[i]:.4f}  {target_text_prompts[i]}"
            print(line); f.write(line + "\n")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == '__main__':
    main()
