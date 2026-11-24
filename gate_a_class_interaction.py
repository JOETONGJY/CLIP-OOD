"""
G-A 门控 (2026-11-24): 描述符×类别交互可靠性诊断 — Stage 0 优先级 1

研究问题 (外审 2026-11-24 版): 类别感知 DDO 的前提是 R 矩阵 (204×C) 中存在
显著的"描述符×类别交互效应"——即某描述符对某类别的可靠性偏离
(全局均值 + 描述符效应 + 类别效应) 的可预测范围, 且该交互跨 Prompt 模板稳定。

方法:
  R_{i,c} = β·z(align) + (1-β)·z(sem), 其中
    align_{i,c} = cos(d_{i,c}, mean_c d_{i,c})   (类方向与描述符公共方向的对齐度)
    sem_{i,c}   = cos(t_{i,c}, t_c^0) - max_{k≠c} cos(t_{i,c}, t_k^0)
                 (语义保持: 风格化类文本仍最接近自己的类原型)
  双重中心化分解: I_{i,c} = R_{i,c} - R̄_{i,·} - R̄_{·,c} + R̄
  零假设检验: 全矩阵置换 2000 次构造 var(I) 的零分布
  模板稳定性: 3 个源模板 ("a photo of / a photograph of / an image of")
              的 I 矩阵两两 Spearman 相关 + Top-K 重合率
  与现有 r_i 的关系: 行效应 a_i 与当前可靠性先验 r 的相关 (交互项是否是 r 之外的新信息)

通过条件 (预注册): 交互效应显著 (置换 p<0.01) 且跨模板 Spearman>0.6, 才进 Stage 1。
产物: experiment_results/Gate_A_类别交互/gate_a_cub.json
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

from args import get_args
from prompts.prompt200new import source_text_prompts, target_text_prompts

OUT_DIR = os.path.join("experiment_results", "Gate_A_类别交互")
TEMPLATES = ["a photo of a {}.", "a photograph of a {}.", "an image of a {}."]
BETA = 0.5
N_PERM = 2000
TOP_K = 15


def encode_texts(model, texts, device, bs=1024):
    import clip
    embs = []
    for i in range(0, len(texts), bs):
        toks = clip.tokenize(texts[i:i + bs]).to(device)
        with torch.no_grad():
            e = model.encode_text(toks).float()
        embs.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(embs, 0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def interaction_stats(R):
    """双重中心化交互项 + Tukey 1-DF 非加性检验 (无重复双因素表的规范检验;
    2026-11-24 修正: 全矩阵置换会连主效应一起销毁, 零分布失真 → 弃用)"""
    row = R.mean(axis=1, keepdims=True)
    col = R.mean(axis=0, keepdims=True)
    grand = R.mean()
    I = R - row - col + grand
    var_I = float(I.var())
    # Tukey 1-DF 非加性检验: 残差 I 对 (行均值×列均值) 乘积项回归, 检验乘性交互
    rm, cm = row.flatten() - grand, col.flatten() - grand
    x = (rm[:, None] * cm[None, :]).flatten()
    y = I.flatten()
    slope = float((x @ y) / (x @ x))
    ss = float(((y - slope * x) ** 2).sum())
    t_stat = slope * np.sqrt(max(len(x) - 2, 1) * float((x * x).sum()) / max(ss, 1e-12))
    # 正态近似双尾 p (n 大)
    from math import erf, sqrt
    p_two = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    return I, {"var_R": float(R.var()), "var_I": var_I,
               "interaction_ratio": round(var_I / max(float(R.var()), 1e-12), 4),
               "iid_noise_ratio_reference": round(1 - 1 / R.shape[0] - 1 / R.shape[1], 4),
               "tukey_1df_slope": round(slope, 6),
               "tukey_1df_t": round(float(t_stat), 2),
               "tukey_1df_p": float(p_two)}


def main():
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = __import__("data").get_dataset_classes(args)[0]
    class_names = list(train_dataset.classname2id.keys())
    C = len(class_names)
    device = args.device

    import clip
    model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=device)

    n_t, n_c = len(target_text_prompts), C
    t_cache = os.path.join("cache", "gate_a_T_cub.pt")
    if os.path.exists(t_cache):
        T = torch.load(t_cache, map_location=device, weights_only=False)
        print(f"[G-A] 复用缓存 T 矩阵: {t_cache}")
    else:
        texts_t = [tp.format(cn) for tp in target_text_prompts for cn in class_names]
        print(f"[G-A] 编码 {n_t}×{C}={len(texts_t)} 条目标文本 (~2-4 分钟)...")
        T = encode_texts(model, texts_t, device).reshape(n_t, C, -1)   # (204, C, D) 已归一化
        os.makedirs("cache", exist_ok=True)
        torch.save(T, t_cache)

    # 现有可靠性 r_i (与 reliability.py 同公式, 用 photo 模板)
    S_photo = encode_texts(model, [t.format(cn) for t, cn in
                                   zip([TEMPLATES[0]] * C, class_names)], device)
    diffs_photo = torch.nn.functional.normalize(T - S_photo.unsqueeze(0), dim=-1)
    cos_mat = torch.matmul(diffs_photo, diffs_photo.transpose(1, 2))
    eye = torch.eye(C, dtype=torch.bool, device=device)
    r_prior = cos_mat[:, ~eye].mean(dim=1).cpu().numpy()           # (204,)

    align_all, sem_all, Is = {}, {}, {}
    templates_stats = {}
    for ti, tpl in enumerate(TEMPLATES):
        S_c = encode_texts(model, [t.format(cn) for t, cn in
                                   zip([tpl] * C, class_names)], device)   # (C, D) t^0_c
        d = torch.nn.functional.normalize(T - S_c.unsqueeze(0), dim=-1)    # diffs (204,C,D)
        d_bar = d.mean(dim=1, keepdim=True)
        d_bar = d_bar / d_bar.norm(dim=-1, keepdim=True)
        align = (d * d_bar).sum(-1).cpu().numpy()                          # (204, C)
        # 语义保持 m_{i,c}: 风格化类文本与"最接近的非自身类原型"的 margin
        # (2026-11-24 修复: 旧实现行级 -inf 掩码把整行置 -inf → NaN 污染)
        sims = torch.einsum("icd,cd->ic", T, S_c).cpu().numpy()            # (204, C)
        top2 = np.partition(sims, -2, axis=1)[:, -2:]                      # 每行前两大
        max1, max2 = top2[:, 1], top2[:, 0]
        arg1 = sims.argmax(axis=1)                                         # 每行最近原型
        runner_up = np.where(np.arange(C)[None, :] == arg1[:, None],
                             max2[:, None], max1[:, None])                 # max_{k≠c}
        m = sims - runner_up                                               # (204, C) m_{i,c}
        sem = m
        zA, zS = zscore(align), zscore(sem)
        assert np.isfinite(zA).all() and np.isfinite(zS).all(), "R 分量含 NaN/Inf"
        R = BETA * zA + (1 - BETA) * zS
        if ti == 0:
            R_main = R.copy()   # photo 模板, 供行效应/Top-K 分析

        I, stats = interaction_stats(R)
        Is[tpl] = I
        templates_stats[tpl] = {"stats": stats,
                                "align_mean": float(align.mean()), "sem_mean": float(m.mean())}
        print(f"[G-A] 模板 '{tpl}': var(R)={stats['var_R']:.4f} var(I)={stats['var_I']:.4f} "
              f"交互占比={stats['interaction_ratio']} (iid噪声参照 {stats['iid_noise_ratio_reference']}) "
              f"Tukey1df p={stats['tukey_1df_p']:.2e}")

    # 模板稳定性: I 矩阵两两 Spearman + Top-K 重合
    flat = {t: I.flatten() for t, I in Is.items()}
    stab = {}
    for a in range(len(TEMPLATES)):
        for b in range(a + 1, len(TEMPLATES)):
            ta, tb = TEMPLATES[a], TEMPLATES[b]
            rho = spearman(flat[ta], flat[tb])
            ka = set(np.argsort(-Is[ta].flatten())[:TOP_K].tolist())
            kb = set(np.argsort(-Is[tb].flatten())[:TOP_K].tolist())
            stab[f"{ta}|{tb}"] = {"spearman": round(rho, 4),
                                  "topk_overlap": round(len(ka & kb) / TOP_K, 3)}
            print(f"[G-A] 模板稳定性 {ta} vs {tb}: Spearman={rho:.4f} Top{TOP_K}重合={len(ka & kb)}/{TOP_K}")

    # 交互项 Top-K 语义示例 (photo 模板)
    I_main = Is[TEMPLATES[0]]
    order = np.argsort(-np.abs(I_main.flatten()))
    examples = []
    for f in order[:TOP_K]:
        i, c = divmod(int(f), C)
        examples.append({"i": i, "c": c, "I": round(float(I_main.flatten()[f]), 3),
                         "desc": target_text_prompts[i], "class": class_names[c]})
    # 行(描述符)主效应与现有 r_i 的相关: r 是否已捕获描述符主效应
    row_eff = R_main.mean(axis=1)
    corr_r_row = spearman(r_prior, row_eff)

    result = {
        "dataset": "CUB", "beta": BETA,
        "templates_stats": templates_stats,
        "template_stability": stab,
        "corr_r_row_effect": round(corr_r_row, 4),
        "top_interactions": examples,
        "verdict": {
            "interaction_significant": all(s["stats"]["tukey_1df_p"] < 0.01
                                           for s in templates_stats.values()),
            "template_stable": all(v["spearman"] > 0.6 for v in stab.values()),
        },
    }
    result["verdict"]["pass"] = result["verdict"]["interaction_significant"] and \
        result["verdict"]["template_stable"]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gate_a_cub.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[G-A] 判定: 交互显著={result['verdict']['interaction_significant']} "
          f"模板稳定={result['verdict']['template_stable']} → "
          f"{'通过' if result['verdict']['pass'] else '不通过'}")
    print(f"[G-A] corr(r_i, 行效应)={corr_r_row:.4f} (低=交互项是 r 之外的新信息)")
    print(f"[saved] {os.path.join(OUT_DIR, 'gate_a_cub.json')}")


if __name__ == "__main__":
    main()
