"""P0 验证2: DDO + 源域风格增强 + 预测一致性 (无新结构的对照实验)

专家指定的关键对照: 在加入任何新结构之前, 先验证
  "风格增强 + 一致性损失" 本身能否提升目标域性能
若能 → 新结构引入增强有坚实依据; 若不能 → 增强路线需谨慎。

设计 (与主管线严格同协议, 缓存特征):
  A. DDO 基线            — 纯源域训练 (对照, 已有 56.14)
  B. DDO + 增强数据       — 概念激活级: 用 (a_orig, a_aug) 对扩充训练集, 都算 CE
  C. DDO + 增强 + 一致性  — B 基础上加 L_cons = D_JS(p(x), p(T(x)))

协议修正 (验证3同步落实): 源域验证集选模型, 目标域仅最终测试。

用法: PROBE_DS=CUB python p1_aug_consistency.py --CLIP_type ...
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip as clip_lib
from args import get_args

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

DS = os.environ.get("PROBE_DS", "CUB")
CONCEPTS = {"CUB": "data/CUB/cub_concepts.txt", "AWA2": "data/awa2/awa_concept.txt"}
SEED = int(os.environ.get("AUG_SEED", "1"))
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)


def js_divergence(p, q):
    """D_JS(p‖q) = 0.5·KL(p‖m) + 0.5·KL(q‖m), m=(p+q)/2"""
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + 1e-8) - torch.log(m + 1e-8))).sum(1)
    kl_qm = (q * (torch.log(q + 1e-8) - torch.log(m + 1e-8))).sum(1)
    return 0.5 * (kl_pm + kl_qm)


def train_variant(a_tr, y_tr, pairs, a_val, y_val, a_tg, y_tg, C,
                  use_cons=False, lambda_cons=1.0, epochs=100, lr=1e-3, tag=""):
    """CBM 线性头训练 (LayerNorm+Linear, 与主管线同构).
    pairs: dict {orig_idx: [(aug_act, aug_label), ...]} 或 None"""
    K = a_tr.shape[1]
    head = nn.Sequential(nn.Flatten(), nn.LayerNorm(K), nn.Linear(K, C)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(a_tr)
    best_val, tg_at_best, ep_at_best = 0.0, 0.0, -1

    pair_keys = list(pairs.keys()) if pairs else []

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            loss = F.cross_entropy(head(a_tr[idx]), y_tr[idx])

            # 增强样本: 概率性混入 (每 batch 50% 概率抽 128 对)
            if pair_keys and random.random() < 0.5:
                pk = random.sample(pair_keys, min(128, len(pair_keys)))
                aug_acts = torch.stack([pairs[k][0] for k in pk]).to(device)
                aug_lbls = torch.tensor([pairs[k][1] for k in pk]).to(device)
                loss_aug = F.cross_entropy(head(aug_acts), aug_lbls)
                loss = loss + 0.5 * loss_aug

                # 一致性: 原始 vs 增强的预测分布
                if use_cons:
                    with torch.no_grad():
                        pass
                    orig_idx_t = torch.tensor(pk, device=device)
                    p_orig = F.softmax(head(a_tr[orig_idx_t]), dim=1)
                    p_aug = F.softmax(head(aug_acts), dim=1)
                    l_cons = js_divergence(p_orig, p_aug).mean()
                    loss = loss + lambda_cons * l_cons

            opt.zero_grad(); loss.backward(); opt.step()

        # 源域验证选模型
        head.eval()
        with torch.no_grad():
            val_acc = (head(a_val).argmax(1) == y_val).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                with torch.no_grad():
                    tg_at_best = (head(a_tg).argmax(1) == y_tg).float().mean().item() * 100
                ep_at_best = ep

    return best_val * 100, tg_at_best, ep_at_best


def main():
    print(f"===== P0 验证2: DDO+增强+一致性 ({DS}, seed {SEED}) =====")
    train = torch.load(f"cache/{DS}_train.pt", map_location="cpu")
    src = torch.load(f"cache/{DS}_source_test.pt", map_location="cpu")
    tgt = torch.load(f"cache/{DS}_target_test.pt", map_location="cpu")

    model, _ = clip_lib.load(args.CLIP_type, device=device)
    model.eval()
    concepts = [l.strip() for l in open(CONCEPTS[DS], encoding="utf-8") if l.strip()]
    with torch.no_grad():
        E = model.encode_text(clip_lib.tokenize(concepts, truncate=True).to(device)).float()
        E = E / E.norm(dim=-1, keepdim=True)
    del model; torch.cuda.empty_cache()
    K = len(concepts)

    def prep(d):
        z = d["feats"].float().to(device)
        z = z / z.norm(dim=-1, keepdim=True)
        return (z @ E.T).cpu(), d["labels"]  # 留 CPU, 训练时再搬

    a_tr, y_tr = prep(train); a_sv, y_sv = prep(src); a_tg, y_tg = prep(tgt)
    a_tr, y_tr = a_tr.to(device), y_tr.to(device)
    a_sv, y_sv = a_sv.to(device), y_sv.to(device)
    a_tg, y_tg = a_tg.to(device), y_tg.to(device)
    C = int(y_tr.max()) + 1

    # 加载增强概念激活 (style_aug 缓存: 2000 样本 × 8 视图)
    aug_cache = f"cache/style_aug_{DS}.pt"
    pairs = None
    if os.path.exists(aug_cache):
        aug = torch.load(aug_cache, map_location="cpu")
        # aug["feats"] 已经是概念激活 (N, M+1, K); 标签 aug["labels"]
        # 构造 pairs: 对每个增强样本, 原始激活→(每个增强视图的激活, 标签)
        af = aug["feats"].float()  # (N, 8, K)
        al = aug["labels"]
        # 关键: 样本顺序与 train 缓存不同 (采样了 10/类), 用自身索引即可
        # 训练时按 batch 内原始样本混入——这里直接把增强视图当作额外样本
        # 对齐方式: pairs[i] = (aug_act_view_m, label_i) for m in 1..7
        pairs = {}
        for i in range(len(af)):
            for m in range(1, af.shape[1]):
                pairs[(i, m)] = (af[i, m].to(device), int(al[i]))
        # 上面 dict 结构不利于随机抽——改为列表索引
        pair_list = list(pairs.values())
        print(f"增强样本: {len(pair_list)} (来自 {len(af)} 张 × 7 视图)")
        # 重构 train_variant 需要的 pairs 接口: 随机抽样列表
        pairs = pair_list
    else:
        print("无增强缓存, 只跑基线")

    # === A. 纯基线 (对照) ===
    v, t, ep = train_variant(a_tr, y_tr, None, a_sv, y_sv, a_tg, y_tg, C, tag="A")
    print(f"A. 纯源域训练:      src_val {v:.2f} | tgt {t:.2f} @ep{ep+1}")

    results = {"A_baseline_tgt": t, "A_baseline_src": v}

    if pairs:
        # 为了让 train_variant 的 pairs 接口工作: 传入 (aug_acts列表, labels列表)
        aug_acts = torch.stack([p[0] for p in pairs])
        aug_lbls = torch.tensor([p[1] for p in pairs]).to(device)

        # === B. + 增强数据 ===
        # 简化: 直接把增强激活拼进训练集 (等价于数据扩充)
        a_tr_aug = torch.cat([a_tr, aug_acts])
        y_tr_aug = torch.cat([y_tr, aug_lbls])
        v, t, ep = train_variant(a_tr_aug, y_tr_aug, None, a_sv, y_sv, a_tg, y_tg, C, tag="B")
        print(f"B. +增强数据:       src_val {v:.2f} | tgt {t:.2f} @ep{ep+1}")
        results["B_aug_tgt"] = t

        # === C. + 增强 + 一致性 ===
        # 需要配对信息: 用 (aug_act, 对应原始act, label)
        # 从 style_aug 缓存: feats[i, 0] 是原始, feats[i, m] 是增强
        orig_acts = af[:, 0].to(device)  # (N, K)
        aug_acts2 = af[:, 1:].reshape(-1, K).to(device)  # (N*7, K)
        orig_expanded = orig_acts.repeat_interleave(7, dim=0)  # 对齐
        lbl_expanded = al.repeat_interleave(7).to(device)

        head = nn.Sequential(nn.Flatten(), nn.LayerNorm(K), nn.Linear(K, C)).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        best_val, tg_at_best = 0.0, 0.0
        for ep in range(100):
            head.train()
            perm = torch.randperm(len(a_tr), device=device)
            for i in range(0, len(a_tr), 256):
                idx = perm[i:i+256]
                loss = F.cross_entropy(head(a_tr[idx]), y_tr[idx])
                pidx = torch.randint(0, len(aug_acts2), (128,), device=device)
                loss_aug = F.cross_entropy(head(aug_acts2[pidx]), lbl_expanded[pidx])
                p_o = F.softmax(head(orig_expanded[pidx]), dim=1)
                p_a = F.softmax(head(aug_acts2[pidx]), dim=1)
                l_cons = js_divergence(p_o, p_a).mean()
                loss = loss + 0.5 * loss_aug + 1.0 * l_cons
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                va = (head(a_sv).argmax(1) == y_sv).float().mean().item()
                if va > best_val:
                    best_val = va
                    tg_at_best = (head(a_tg).argmax(1) == y_tg).float().mean().item() * 100
        print(f"C. +增强+一致性:    src_val {best_val*100:.2f} | tgt {tg_at_best:.2f}")
        results["C_aug_cons_tgt"] = tg_at_best

    out = f"downloads/p1_augcons_{DS}_seed{SEED}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"已保存 → {out}")


if __name__ == "__main__":
    main()
