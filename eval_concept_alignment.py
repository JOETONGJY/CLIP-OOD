"""E2: P4 机制证据——校对前后概念得分与人工标注的相关性对比

对比同一模型内部两个投影矩阵读出的概念得分与人工属性标注的一致性:
  E_frozen : 文本构建的冻结词典 (训练前)
  E_calib  : E + cal_U @ cal_V  (训练后, 来自 checkpoint)

指标: 逐概念 Pearson 相关 (概念得分 vs 二值标注), 汇总均值/中位数/正比例;
      另报 AUROC (标注为正的概念上, 得分区分正负样本的能力)。

用法: python eval_concept_alignment.py --dataset CUB --weights best_CUB.pth
输出: experiment_results/e2_concept_alignment_{DS}.json
"""
import os
import argparse
import pickle
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CUB')
    ap.add_argument('--weights', default='best_CUB.pth')
    ap.add_argument('--out_dir', default='experiment_results')
    args = ap.parse_args()

    # ---- 1. 人工标注 (pkl 顺序 = 缓存特征顺序, pkl 多出的 4 张是读取失败图) ----
    pkl_path = f'data/{args.dataset}/{args.dataset}train.pkl'
    meta = pickle.load(open(pkl_path, 'rb'))
    attrs = np.array([x['attribute_label'] for x in meta], dtype=np.float32)

    cache = torch.load(f'cache/{args.dataset}_train.pt', map_location='cpu',
                       weights_only=False)
    feats = cache['feats'].float()                       # (N, 768)
    n = min(len(attrs), feats.size(0))                   # 对齐 (5990 vs 5994)
    attrs, feats = attrs[:n], feats[:n]
    labels = cache['labels'].long()[:n]
    # 丢弃 pkl 尾部多出的样本对不齐时的行 (保险: 以 cache 长度为准)
    print(f'[{args.dataset}] 样本 {n}, 概念 {attrs.shape[1]}, '
          f'标注正例比 {(attrs > 0).mean():.4f}')

    # ---- 2. 两个投影矩阵 ----
    sd = torch.load(args.weights, map_location='cpu', weights_only=False)
    if 'cal_U' not in sd or 'cal_V' not in sd:
        raise RuntimeError(f'{args.weights} 中没有 cal_U/cal_V (不是 cal 模式的权重)')
    dE = sd['cal_U'].float() @ sd['cal_V'].float()
    print(f'校对量 ‖ΔE‖_F = {torch.norm(dE):.4f}')

    # 冻结 E 不在 state_dict (它是 buffer 性质的常量) —— 现场用 CLIP 文本编码器重建,
    # 与模型构建路径一致 (概念名列表来自数据管线的概念表文本文件, 顺序即标注列顺序)
    import sys
    sys.path.insert(0, os.getcwd())
    import clip as clip_pkg

    concept_files = {
        'CUB': 'data/CUB/cub_concepts.txt',
        'AWA2': 'data/awa2/awa2_conceptNet_concepts.txt',
    }
    with open(concept_files[args.dataset], 'r', encoding='utf-8') as f:
        concept_names = [x.rstrip('\n') for x in f.readlines() if x.strip()]
    assert len(concept_names) == attrs.shape[1], \
        f'概念数 {len(concept_names)} != 标注列数 {attrs.shape[1]}'

    clip_model, _ = clip_pkg.load('F:/CLIP-OOD/clip_weights/ViT-L-14.pt', device='cpu')
    with torch.no_grad():
        tokens = clip_pkg.tokenize(concept_names)
        E_frozen = clip_model.encode_text(tokens).float()
        E_frozen = E_frozen / E_frozen.norm(dim=-1, keepdim=True)
    print(f'重建冻结词典: {tuple(E_frozen.shape)}')

    # ---- 3. 概念得分 ----
    with torch.no_grad():
        s_frozen = feats @ E_frozen.T                    # (N, K)
        s_calib = feats @ (E_frozen + dE).T

    # ---- 3. 逐概念 Pearson + AUROC ----
    a_t = torch.from_numpy(attrs)
    a_c = a_t - a_t.mean(0, keepdim=True)
    std = a_c.std(0)
    valid = std > 1e-6                                   # 常数列(全0概念)无相关可言

    def pearson(s):
        sc = s - s.mean(0, keepdim=True)
        num = (a_c * sc).sum(0)
        den = a_c.norm(dim=0) * sc.norm(dim=0) + 1e-8
        return (num / den)[valid]

    def auroc(s):
        """逐概念 AUROC (正例得分应高于负例), rank 实现"""
        out = []
        for j in torch.nonzero(valid).flatten().tolist():
            y = attrs[:, j] > 0
            if y.sum() < 5 or (~y).sum() < 5:
                continue
            x = s[:, j].numpy()
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(x) + 1)
            # 并列平均秩 (粗略: 得分连续, 影响极小)
            r_pos = ranks[y].mean()
            n_pos, n_neg = y.sum(), (~y).sum()
            out.append((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        return torch.tensor(out)

    r_frozen, r_calib = pearson(s_frozen), pearson(s_calib)
    auc_frozen, auc_calib = auroc(s_frozen), auroc(s_calib)

    report = {
        'dataset': args.dataset,
        'weights': args.weights,
        'delta_E_fro': float(torch.norm(dE)),
        'n_concepts_eval': int(valid.sum()),
        'pearson_frozen': {'mean': float(r_frozen.mean()), 'median': float(r_frozen.median())},
        'pearson_calib': {'mean': float(r_calib.mean()), 'median': float(r_calib.median())},
        'auroc_frozen': {'mean': float(auc_frozen.mean()), 'median': float(auc_frozen.median())},
        'auroc_calib': {'mean': float(auc_calib.mean()), 'median': float(auc_calib.median())},
        'improved_ratio_pearson': float((r_calib > r_frozen).float().mean()),
        'improved_ratio_auroc': float((auc_calib > auc_frozen).float().mean()),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'e2_concept_alignment_{args.dataset}.json')
    import json
    json.dump(report, open(out, 'w'), indent=2)
    print(f'Pearson  冻结 {report["pearson_frozen"]["mean"]:.4f} → 校准 {report["pearson_calib"]["mean"]:.4f}'
          f'  (提升概念比例 {report["improved_ratio_pearson"]:.2%})')
    print(f'AUROC    冻结 {report["auroc_frozen"]["mean"]:.4f} → 校准 {report["auroc_calib"]["mean"]:.4f}'
          f'  (提升概念比例 {report["improved_ratio_auroc"]:.2%})')
    print(f'→ {out}')


if __name__ == '__main__':
    main()
