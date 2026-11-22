"""
簇统计补全 (2026-11-22, 外审复查问题 4): 在 cluster_stats.json (G/多成员簇数/最大簇/
涉及描述符数) 基础上补两项 —— 簇内最小两两相似度 (检验 single-linkage 链式合并程度)
与多成员簇的语义示例。产物: experiment_results/E3_分组去冗余/cluster_stats_detail.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import json

import torch

from args import get_args
from reliability import compute_descriptor_clusters
from prompts.prompt200new import target_text_prompts

OUT = os.path.join("experiment_results", "E3_分组去冗余", "cluster_stats_detail.json")

DATASET_BATCH = {"CUB": 256, "AWA2": 256, "LADV": 256, "PACS": 128}

if __name__ == "__main__":
    out = {}
    for ds, bs in DATASET_BATCH.items():
        sys.argv = [sys.argv[0], "--dataset", ds, "--seed", "1", "--batch_size", str(bs),
                    "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
        args = get_args()
        args.save_model = False
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_dataset = __import__("data").get_dataset_classes(args)[0]

        dirs = train_dataset.domain_diffs.mean(dim=1).float()
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        assignment, g = compute_descriptor_clusters(train_dataset.domain_diffs, threshold=0.9)
        S = (dirs @ dirs.T).abs()

        counts = torch.bincount(assignment, minlength=g)
        multi = [c for c in range(g) if counts[c] > 1]
        # 全体多成员簇内的最小/平均两两相似度 (链式合并强度)
        min_pairs = []
        all_pairs = []
        clusters_detail = []
        for c in multi:
            members = torch.nonzero(assignment == c, as_tuple=True)[0].tolist()
            sub = S[members][:, members]
            n = len(members)
            pairs = sub[~torch.eye(n, dtype=torch.bool)]
            min_pairs.append(float(pairs.min()))
            all_pairs.extend(pairs.tolist())
            clusters_detail.append({
                "members": [int(m) for m in members],
                "texts": [target_text_prompts[m].replace(" {}.", " [类]") for m in members],
                "min_pairwise_cos": round(float(pairs.min()), 4),
            })
        out[ds] = {
            "n_groups": int(g),
            "multi_clusters": len(multi),
            "largest_cluster": int(counts.max()),
            "min_pairwise_cos_all_multi": round(min(min_pairs), 4) if min_pairs else None,
            "mean_pairwise_cos_multi": round(sum(all_pairs) / len(all_pairs), 4) if all_pairs else None,
            "clusters": clusters_detail,
        }
        print(f"[detail] {ds}: G={g}, 多成员簇 {len(multi)}, 最大 {counts.max().item()}, "
              f"簇内最小 cos={out[ds]['min_pairwise_cos_all_multi']}")
        for cd in clusters_detail:
            print(f"    簇{cd['members']} min_cos={cd['min_pairwise_cos']}: {cd['texts'][:3]}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[saved] {OUT}")
