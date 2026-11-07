"""E2-R: 概念保真度评估 (2026-11-07 修正版, 回应专家意见)

对比冻结词典 E 与校准词典 E' = E + cal_U@cal_V 读出的概念得分与人工属性标注的一致性。

相对旧版 (作废) 的修正:
  1. AUROC 改用 sklearn.metrics.roc_auc_score (旧手写秩公式有误, 输出负值作废);
  2. 特征 L2 归一化 (与模型前向一致: model.forward 中 z = z/||z||);
  3. 标注来源改为 cub_train.txt 自带的逐样本属性列 (与 cache 特征顺序严格一致;
     旧版用 CUBtrain.pkl 截断对齐——已证实 720 行起错位, 旧结论作废);
  4. checkpoint 参数化 (--weights), 不再依赖固定名 best_{DS}.pth;
  5. 同时报告"raw 特征×E"(P4 单独影响) 与可选"z_inv×E'"(完整模型路径)——
     当前版本先做 raw 特征对照, 完整模型路径在 E4 干预实验中覆盖。

用法:
  python eval_concept_alignment.py --dataset CUB --weights best_CUB_sel.pth \
      --rank 16 --seed 1
输出: experiment_results/e2r_align_{DS}_r{rank}_s{seed}.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONCEPT_FILES = {
    'CUB': 'data/CUB/cub_concepts.txt',
    'AWA2': 'data/awa2/awa2_conceptNet_concepts.txt',
}


def load_alignment(dataset):
    """返回 (feats_norm, labels, attrs) —— 标注与 cache 特征逐行对齐"""
    d = torch.load(f'cache/{dataset}_train.pt', map_location='cpu', weights_only=False)
    feats = F.normalize(d['feats'].float(), dim=-1)          # 修正2: 与前向一致
    labels = d['labels'].long()
    if dataset == 'CUB':
        rows = [x.strip().split(',') for x in open('data/CUB/cub_train.txt').readlines()]
        # txt: path, cls(1-based), bbox×4, attrs×312
        assert len(rows) == feats.size(0), f'txt {len(rows)} != cache {feats.size(0)}'
        lab_txt = torch.tensor([int(r[1]) for r in rows]) - 1
        assert (lab_txt == labels).all(), 'txt 与 cache 顺序不一致!'
        attrs = torch.tensor([list(map(float, r[6:6 + 312])) for r in rows])
    else:
        raise NotImplementedError(f'{dataset} 的逐样本属性对齐待补 (AWA2 用 predicate-matrix 为类级, 不可逐样本)')
    return feats, labels, attrs


def build_E(dataset):
    """现场重建冻结词典 E (概念短语 → CLIP text encoder)"""
    import clip as clip_pkg
    with open(CONCEPT_FILES[dataset], 'r', encoding='utf-8') as f:
        names = [x.rstrip('\n') for x in f.readlines() if x.strip()]
    clip_model, _ = clip_pkg.load('F:/CLIP-OOD/clip_weights/ViT-L-14.pt', device='cpu')
    with torch.no_grad():
        E = clip_model.encode_text(clip_pkg.tokenize(names)).float()
        E = E / E.norm(dim=-1, keepdim=True)
    return E


def eval_E(scores, attrs, min_pos=5):
    """逐概念 Pearson + AUROC (sklearn); 返回 (per_pearson, per_auroc, valid_mask)"""
    a = attrs.numpy()
    n_concept = attrs.shape[1]
    per_p, per_a, mask = [], [], []
    for j in range(n_concept):
        y = (a[:, j] > 0).astype(int)
        if y.sum() < min_pos or (1 - y).sum() < min_pos or a[:, j].std() == 0:
            mask.append(False)
            continue
        mask.append(True)
        per_p.append(float(np.corrcoef(scores[:, j], a[:, j])[0, 1]))
        per_a.append(float(roc_auc_score(y, scores[:, j])))   # 修正1: sklearn
    return np.array(per_p), np.array(per_a), np.array(mask)


def summarize(per_p, per_a, mask):
    return {
        'n_valid_concepts': int(mask.sum()),
        'pearson_mean': float(np.mean(per_p)),
        'pearson_median': float(np.median(per_p)),
        'pearson_frac_positive': float((per_p > 0).mean()),
        'auroc_mean': float(np.mean(per_a)),
        'auroc_median': float(np.median(per_a)),
        'auroc_frac_above_0.5': float((per_a > 0.5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CUB')
    ap.add_argument('--weights', required=True, help='含 cal_U/cal_V 的 checkpoint')
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--model_tag', default='c3+spda_cal', help='记录训练模式')
    ap.add_argument('--out_dir', default='experiment_results')
    args = ap.parse_args()

    feats, labels, attrs = load_alignment(args.dataset)
    E_frozen = build_E(args.dataset)

    sd = torch.load(args.weights, map_location='cpu', weights_only=False)
    assert 'cal_U' in sd and 'cal_V' in sd, f'{args.weights} 缺 cal_U/cal_V'
    dE = sd['cal_U'].float() @ sd['cal_V'].float()
    E_calib = E_frozen + dE

    with torch.no_grad():
        s_frozen = (feats @ E_frozen.T).numpy()
        s_calib = (feats @ E_calib.T).numpy()

    p_f, a_f, mask = eval_E(s_frozen, attrs)
    p_c, a_c, _ = eval_E(s_calib, attrs)

    rep = {
        'dataset': args.dataset,
        'model_tag': args.model_tag,
        'rank': args.rank,
        'seed': args.seed,
        'weights': args.weights,
        'protocol': 'final-selection checkpoint' if 'sel' in args.weights else 'legacy checkpoint',
        'delta_E_fro': float(torch.norm(dE)),
        'E_fro': float(torch.norm(E_frozen)),
        'rel_shift': float(torch.norm(dE) / torch.norm(E_frozen)),
        'frozen': summarize(p_f, a_f, mask),
        'calib': summarize(p_c, a_c, mask),
        'improved_frac_pearson': float((p_c > p_f).mean()),
        'improved_frac_auroc': float((a_c > a_f).mean()),
        'per_concept_pearson_frozen': p_f.tolist(),
        'per_concept_pearson_calib': p_c.tolist(),
        'per_concept_auroc_frozen': a_f.tolist(),
        'per_concept_auroc_calib': a_c.tolist(),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"e2r_align_{args.dataset}_r{args.rank}_s{args.seed}.json")
    with open(out, 'w') as f:
        json.dump(rep, f, indent=2)

    print(f"[{args.dataset}] r={args.rank} seed={args.seed} ({args.weights})")
    print(f"  ‖ΔE‖/‖E‖ = {rep['rel_shift']:.3f}")
    print(f"  Pearson  冻结 {rep['frozen']['pearson_mean']:.4f} → 校准 {rep['calib']['pearson_mean']:.4f}"
          f"  (改善概念 {rep['improved_frac_pearson']:.1%})")
    print(f"  AUROC    冻结 {rep['frozen']['auroc_mean']:.4f} → 校准 {rep['calib']['auroc_mean']:.4f}"
          f"  (改善概念 {rep['improved_frac_auroc']:.1%})")
    print(f"  有效概念 {rep['frozen']['n_valid_concepts']}/312 → {out}")


if __name__ == '__main__':
    main()
