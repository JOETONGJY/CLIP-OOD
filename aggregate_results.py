"""
多 seed 结果汇总: 读取 logs/results/*.json, 按 (dataset, weight_mode) 分组,
报告 mean±std 的 target/source accuracy。

用法:
    python3 aggregate_results.py                        # 汇总所有
    python3 aggregate_results.py --dataset CUB          # 只看 CUB
    python3 aggregate_results.py --seeds 1 2 3          # 只统计指定 seed
"""
import os
import json
import glob
import argparse
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='logs/results')
    parser.add_argument('--dataset', type=str, default=None, help='只看指定数据集')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='只统计指定 seeds')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, '*.json')))
    if not files:
        print(f"No result files found in {args.results_dir}")
        return

    groups = defaultdict(list)
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        if args.dataset and r['dataset'] != args.dataset:
            continue
        if args.seeds and r['seed'] not in args.seeds:
            continue
        groups[(r['dataset'], r['weight_mode'])].append(r)

    # 输出表格
    print(f"\n{'Dataset':<10} {'Weight Mode':<18} {'#Seeds':<8} "
          f"{'Target Acc (mean±std)':<24} {'Source Acc (mean±std)':<24}")
    print("-" * 90)

    order = ['none', 'fixed', 'residual', 'prior_residual', 'ddo', 'hardproj', 'softproj', 'headonly', 'full']
    for (dataset, mode), runs in sorted(groups.items(), key=lambda x: (x[0][0], order.index(x[0][1]) if x[0][1] in order else 99)):
        target_accs = [r['target_acc'] for r in runs]
        source_accs = [r['source_acc'] for r in runs]
        t_mean, t_std = np.mean(target_accs), np.std(target_accs)
        s_mean, s_std = np.mean(source_accs), np.std(source_accs)
        print(f"{dataset:<10} {mode:<18} {len(runs):<8} "
              f"{t_mean:.2f} ± {t_std:.2f}          {s_mean:.2f} ± {s_std:.2f}")

    # 理想因果链检查 (专家期望: none < fixed < prior_residual, residual < prior_residual)
    print("\n因果链检查 (若各模式均有结果):")
    for dataset in set(k[0] for k in groups):
        means = {}
        for mode in order:
            runs = groups.get((dataset, mode), [])
            if runs:
                means[mode] = np.mean([r['target_acc'] for r in runs])
        checks = []
        if 'none' in means and 'fixed' in means:
            checks.append(f"DDO<Soft: {'✓' if means['none'] < means['fixed'] else '✗'}")
        if 'fixed' in means and 'prior_residual' in means:
            checks.append(f"Soft<Prior+Res: {'✓' if means['fixed'] < means['prior_residual'] else '✗'}")
        if 'residual' in means and 'prior_residual' in means:
            checks.append(f"ResOnly<Prior+Res: {'✓' if means['residual'] < means['prior_residual'] else '✗'}")
        if checks:
            print(f"  {dataset}: " + "  ".join(checks))


if __name__ == '__main__':
    main()
