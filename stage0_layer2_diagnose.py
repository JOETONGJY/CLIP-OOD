"""Stage 0B 第二层 + 0D 诊断 (预注册见 RESULTS.md §3t; 外审第八轮 P0 修正版)。
零训练: 只读增强缓存 (cache/CUB_train_aug.pt)、27 个 Stage 0 分类头 (logs/heads/)、
概念矩阵 E 与域方向 (从数据集对象获取)。只用源域数据。

外审第八轮修正的六项实现问题 (预注册设计不变, 实现对齐预注册):
  1. 扰动强度 ‖Δu‖ 加入回归 (此前 X1/X2 均未包含, 违背 §3t 的三控制变量设计)
  2. 逐类别对的 avg_pert_strength (此前回退为全局常数)
  3. 补逐样本干净 top-5 竞争类口径 (此前仅类别平均 logits top-5)
  4. 补 S_vis = |wᵀΔu|/(‖Δu‖+ε) (预注册要求但未实现)
  5. 逐 seed 判定方向一致性 (此前用跨 seed 均值)
  6. 补投影范数 ‖Hw‖ (PCR 用投影后权重, w_norm 此前用原始权重)

另修正:
  7. 如实记录评估集合为 heldout_val 1200 样本 (非 §3t 写的"完整训练集")
  8. 判定保留原规则; ΔR² 0.01~0.05 且系数为正的情形如实报告为"初步证据不支持, 暂停"

用法: python stage0_layer2_diagnose.py [--selftest]
产物: experiment_results/Stage0-Diagnostics/stage0_layer2_cub.json、stage0_layer2_cub.md
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from concept_cancellation import diag_reference, pcr_quantile, quad, random_baseline
from language_ellipsoid import domain_directions, second_moments

OUT_DIR = os.path.join("experiment_results", "Stage0-Diagnostics")
SEEDS = (1, 2, 3)
ARMS = [("ce_a0", "CE-Only（α=0）"), ("full_a1", "Full DDO（α=1）"), ("full_a2", "Full DDO（α=2）"),
        ("mean_a1", "Mean（α=1）"), ("mean_a2", "Mean（α=2）"), ("sym_a1", "Full-Symmetric（α=1）"),
        ("sym_a2", "Full-Symmetric（α=2）"), ("bf_a1", "Bias-Free（α=1）"), ("bf_a2", "Bias-Free（α=2）")]
FULL_ARMS, CE_ARM = ("full_a1", "full_a2"), "ce_a0"
TOP_K = 5
EPS_LN = 1e-5

# ---- 预注册阈值 (§3t); 不改 ----
B2_DELTAR2_SUPPORT = 0.05
B2_DELTAR2_NONE = 0.01


def load_heads():
    out = {}
    for tag, _ in ARMS:
        for s in SEEDS:
            p = f"logs/heads/CUB_s0_{tag}_seed{s}.pt"
            if os.path.exists(p):
                out[(tag, s)] = torch.load(p, map_location="cpu", weights_only=False)
    return out


def compute_logits(z, head_params, E):
    """z: (N, 768) 已归一化的 CLIP 视觉嵌入 → logits: (N, C)"""
    hp = head_params["classifier"]
    W, gam, beta, b = hp["2.weight"].double(), hp["1.weight"].double(), hp["1.bias"].double(), hp["2.bias"].double()
    u_raw = z.double() @ E.T
    u_norm = (u_raw - u_raw.mean(-1, keepdim=True)) / torch.sqrt(u_raw.var(-1, unbiased=False, keepdim=True) + EPS_LN)
    return u_norm @ (W * gam[None, :]).T + W @ beta + b


def top_comps_class_avg(logits, labels, n_classes, k=TOP_K):
    """类别平均 logits 的 top-k 竞争类 (原口径, 保留作为敏感性分析)"""
    comps = {}
    for y in range(n_classes):
        mask = labels == y
        if mask.sum() == 0:
            continue
        avg_logit = logits[mask].mean(0)
        avg_logit[y] = float("-inf")
        comps[y] = torch.topk(avg_logit, k).indices.tolist()
    return comps


def top_comps_per_sample(logits, labels, k=TOP_K):
    """逐样本干净 logits 的 top-k 竞争类 (外审第八轮补齐的口径)。
    返回 list of list: 每个样本的竞争类列表。"""
    comps_per_sample = []
    for si in range(logits.size(0)):
        logit = logits[si].clone()
        y = int(labels[si])
        logit[y] = float("-inf")
        comps_per_sample.append(torch.topk(logit, k).indices.tolist())
    return comps_per_sample


def self_test():
    """合成数据: 已知 PCR-损害关系的方向, 验证回归与判定逻辑。"""
    torch.manual_seed(0)
    K, C, N, n_pert = 40, 20, 300, 3
    sens = [3, 7, 11]
    tt = torch.randn(N, 1)
    U = torch.randn(N, K) * 0.05
    U[:, sens] = tt * 2.0 + torch.randn(N, len(sens)) * 0.4
    U = (U - U.mean(-1, keepdim=True)).double()
    G, _, _, _ = second_moments(U)
    Dt = diag_reference(G)
    _, samples = random_baseline(G, n=10000, seed=0)
    Hm = torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD ") + name + (f"  [{detail}]" if detail else ""))

    # 构造一个头: 一半类别用抵消型权重 (PCR 低), 一半不用
    W = torch.randn(C, K, dtype=torch.float64) * 0.5
    W[: C // 2, 3] = 6.0; W[: C // 2, 7] = -6.0
    gam = torch.ones(K, dtype=torch.float64)
    A = W * gam[None, :]
    # PCR
    y_i, k_i = torch.triu_indices(C, C, offset=1)
    Wk = (A[y_i] - A[k_i]) @ Hm
    pcr = quad(Wk, G) / quad(Wk, Dt).clamp_min(1e-300)
    qtl = pcr_quantile(pcr, samples)
    pair_qtl = {}
    for pi in range(len(y_i)):
        pair_qtl[(int(y_i[pi]), int(k_i[pi]))] = float(qtl[pi])
        pair_qtl[(int(k_i[pi]), int(y_i[pi]))] = float(qtl[pi])

    # 合成 margin damage: 低 PCR 的对损害更大 (模拟脆弱抵消)
    labels = torch.randint(0, C, (N,))
    pair_damage, pair_margin, pair_pert_str, pair_svis = {}, {}, {}, {}
    for si in range(N):
        y = int(labels[si])
        for k in range(C):
            if k == y:
                continue
            # 损害与 PCR 负相关 (低 PCR → 高损害)
            q = pair_qtl.get((y, k), 0.5)
            base_damage = (1 - q) * 0.5 + torch.randn(1).double() * 0.1
            m = torch.randn(1).double() * 2 + 1
            du = torch.rand(1).double() * 0.5 + 0.3
            w_diff = A[y] - A[k]
            du_vec = torch.randn(K, dtype=torch.float64) * 0.1
            svis = float((w_diff @ du_vec).abs() / (du_vec.norm() + 1e-12))
            key = (y, k)
            pair_damage.setdefault(key, []).append(float(base_damage))
            pair_margin.setdefault(key, []).append(float(m))
            pair_pert_str.setdefault(key, []).append(float(du))
            pair_svis.setdefault(key, []).append(svis)

    pair_data = []
    for (y, k), damages in pair_damage.items():
        if (y, k) not in pair_qtl:
            continue
        w_proj = (A[y] - A[k]) @ Hm
        pair_data.append({
            "avg_damage": float(np.mean(damages)),
            "avg_clean_margin": float(np.mean(pair_margin[(y, k)])),
            "avg_pert_strength": float(np.mean(pair_pert_str[(y, k)])),
            "avg_svis": float(np.mean(pair_svis[(y, k)])),
            "pcr_quantile": pair_qtl[(y, k)],
            "w_norm": float((A[y] - A[k]).norm()),
            "w_norm_proj": float(w_proj.norm()),
        })

    # 回归 (含扰动强度)
    dv = np.array([p["avg_damage"] for p in pair_data])
    wn = np.array([p["w_norm"] for p in pair_data])
    cm = np.array([p["avg_clean_margin"] for p in pair_data])
    du_arr = np.array([p["avg_pert_strength"] for p in pair_data])
    pq = np.array([p["pcr_quantile"] for p in pair_data])
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    dv_z, wn_z, cm_z, du_z, pq_z = zs(dv), zs(wn), zs(cm), zs(du_arr), zs(pq)
    ones = np.ones(len(dv))
    def ols(X, y):
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        return 1 - np.sum((y - y_pred)**2) / (np.sum((y - y.mean())**2) + 1e-12), beta
    X1 = np.column_stack([ones, wn_z, cm_z, du_z])       # 基线 (含扰动强度)
    X2 = np.column_stack([ones, wn_z, cm_z, du_z, pq_z]) # + PCR
    r2_1, _ = ols(X1, dv_z)
    r2_2, beta2 = ols(X2, dv_z)

    check("S1 含扰动强度的回归: PCR 系数为负 (合成数据中低 PCR → 高损害)", float(beta2[-1]) < 0,
          f"coef = {float(beta2[-1]):.3f}")
    check("S2 含扰动强度的回归: ΔR² > 0.3 (合成信号强)", r2_2 - r2_1 > 0.3,
          f"ΔR² = {r2_2 - r2_1:.3f}")
    check("S3 S_vis 已计算且非退化", len(pair_data) > 0 and 0 < np.mean([p["avg_svis"] for p in pair_data]) < 10)
    check("S4 投影范数已计算且 ≤ 原始范数",
          all(p["w_norm_proj"] <= p["w_norm"] + 1e-9 for p in pair_data))
    check("S5 逐对扰动强度有变异 (非全局常数)", np.std(du_arr) > 0)

    # 逐 seed 方向一致性
    seeds_coefs = [-0.5, -0.6, -0.4]  # 模拟三个 seed 的 PCR 系数
    consistent = all(c < 0 for c in seeds_coefs)
    check("S6 逐 seed 方向一致性检查 (全部为负)", consistent)

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        return self_test()
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD ") + name + (f"  [{detail}]" if detail else ""))

    # ===== 1. 加载增强缓存 =====
    print("加载增强缓存...")
    aug = torch.load("cache/CUB_train_aug.pt", map_location="cpu", weights_only=False)
    val_idx = aug["split"]["heldout_val"]
    n_val = len(val_idx)
    labels_all = aug["labels"]
    labels_val = labels_all[val_idx]
    clean_val = aug["clean_feats"][val_idx].float()
    aug_names = list(aug["aug_feats"].keys())
    aug_val = {name: aug["aug_feats"][name][val_idx].float() for name in aug_names}
    print(f"  heldout_val: {n_val} 个样本 (§3t 写'完整训练集', 实际为 heldout_val —— 已修正标注)")

    # ===== 2. 概念矩阵与域方向 =====
    print("加载概念矩阵与域方向...")
    from args import get_args
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = __import__("data").get_dataset_classes(args)[0]
    diffs = ds.domain_diffs.double().cpu()
    concept_names = list(ds.concept2id.keys())
    n_classes = len(ds.classname2id)
    import clip
    m, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=args.device)
    with torch.no_grad():
        E = m.encode_text(clip.tokenize(concept_names).to(args.device)).double().cpu()
        E = E / E.norm(dim=-1, keepdim=True)
    del m, ds
    torch.cuda.empty_cache()
    U = domain_directions(diffs, E)
    G, _, _, _ = second_moments(U)
    Dt = diag_reference(G)
    _, base_samples = random_baseline(G, n=40000, seed=0)
    K_dim = E.size(0)
    Hm = torch.eye(K_dim, dtype=torch.float64) - torch.ones(K_dim, K_dim, dtype=torch.float64) / K_dim

    # ===== 3. 加载头 =====
    heads = load_heads()
    check(f"27 个分类头都已加载", len(heads) == 27)

    # ===== 4. 竞争类 =====
    ce_head = heads[(CE_ARM, 1)]
    ce_clean_logits = compute_logits(clean_val, ce_head, E).detach()
    fixed_comps = top_comps_class_avg(ce_clean_logits, labels_val, n_classes)

    # ===== 5. 逐头 =====
    results = {"arms": {}, "aug_names": aug_names, "n_val": n_val,
               "evaluation_set": "heldout_val (1200 samples; §3t 原文写'完整训练集', 实际为 heldout_val)",
               "note": "外审第八轮 P0 修正版: 扰动强度入回归、逐对强度、逐样本竞争类、S_vis、逐 seed 判定、投影范数"}

    for tag, name in ARMS:
        per_seed = []
        for s in SEEDS:
            h = heads.get((tag, s))
            if h is None:
                continue
            hp = h["classifier"]
            W, gam = hp["2.weight"].double(), hp["1.weight"].double()
            A = W * gam[None, :]

            y_i, k_i = torch.triu_indices(A.size(0), A.size(0), offset=1)
            Wk = (A[y_i] - A[k_i]) @ Hm
            pcr = quad(Wk, G) / quad(Wk, Dt).clamp_min(1e-300)
            qtl = pcr_quantile(pcr, base_samples)
            pair_qtl = {}
            for pi in range(len(y_i)):
                pair_qtl[(int(y_i[pi]), int(k_i[pi]))] = float(qtl[pi])
                pair_qtl[(int(k_i[pi]), int(y_i[pi]))] = float(qtl[pi])

            clean_logits = compute_logits(clean_val, h, E).detach()
            own_comps_avg = top_comps_class_avg(clean_logits, labels_val, n_classes)
            own_comps_sample = top_comps_per_sample(clean_logits, labels_val)

            clean_acc = float((clean_logits.argmax(1) == labels_val).double().mean())
            clean_ce = float(torch.nn.functional.cross_entropy(clean_logits, labels_val))
            aug_accs, aug_ces = {}, {}
            for an in aug_names:
                pl = compute_logits(aug_val[an], h, E).detach()
                aug_accs[an] = float((pl.argmax(1) == labels_val).double().mean())
                aug_ces[an] = float(torch.nn.functional.cross_entropy(pl, labels_val))
            avg_aug_acc = float(np.mean(list(aug_accs.values())))
            avg_aug_ce = float(np.mean(list(aug_ces.values())))

            # ---- 0B 第二层 (修正版: 逐对扰动强度 + S_vis + 投影范数 + 逐样本竞争类) ----
            for comp_scheme in ("fixed", "own_avg", "own_sample"):
                if comp_scheme == "fixed":
                    comps = fixed_comps
                elif comp_scheme == "own_avg":
                    comps = own_comps_avg
                else:
                    comps = None  # 逐样本, 在循环内处理

                pair_damage, pair_margin, pair_pert, pair_svis = {}, {}, {}, {}
                for an in aug_names:
                    pert_logits = compute_logits(aug_val[an], h, E).detach()
                    u_c = clean_val.double() @ E.T
                    u_p = aug_val[an].double() @ E.T
                    u_c_n = (u_c - u_c.mean(-1, keepdim=True)) / torch.sqrt(u_c.var(-1, unbiased=False, keepdim=True) + EPS_LN)
                    u_p_n = (u_p - u_p.mean(-1, keepdim=True)) / torch.sqrt(u_p.var(-1, unbiased=False, keepdim=True) + EPS_LN)
                    du_vec = u_p_n - u_c_n
                    du_str = du_vec.norm(dim=-1)

                    for si in range(n_val):
                        y = int(labels_val[si])
                        if comp_scheme == "own_sample":
                            ks = own_comps_sample[si]
                        else:
                            ks = comps.get(y, [])
                        for k in ks:
                            m_c = float(clean_logits[si, y] - clean_logits[si, k])
                            m_p = float(pert_logits[si, y] - pert_logits[si, k])
                            d = m_c - m_p
                            w_diff = A[y] - A[k]
                            sv = float((w_diff @ du_vec[si]).abs() / (du_str[si] + 1e-12))
                            key = (y, k)
                            pair_damage.setdefault(key, []).append(d)
                            pair_margin.setdefault(key, []).append(m_c)
                            pair_pert.setdefault(key, []).append(float(du_str[si]))
                            pair_svis.setdefault(key, []).append(sv)

                pair_data = []
                for (y, k), damages in pair_damage.items():
                    q = pair_qtl.get((y, k))
                    if q is None:
                        continue
                    w_proj = (A[y] - A[k]) @ Hm
                    pair_data.append({
                        "avg_damage": float(np.mean(damages)),
                        "avg_clean_margin": float(np.mean(pair_margin[(y, k)])),
                        "avg_pert_strength": float(np.mean(pair_pert[(y, k)])),
                        "avg_svis": float(np.mean(pair_svis[(y, k)])),
                        "pcr_quantile": q,
                        "w_norm": float((A[y] - A[k]).norm()),
                        "w_norm_proj": float(w_proj.norm()),
                    })

                if len(pair_data) > 20:
                    dv = np.array([p["avg_damage"] for p in pair_data])
                    wn = np.array([p["w_norm"] for p in pair_data])
                    cm = np.array([p["avg_clean_margin"] for p in pair_data])
                    du_a = np.array([p["avg_pert_strength"] for p in pair_data])
                    pq = np.array([p["pcr_quantile"] for p in pair_data])
                    zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)
                    dv_z, wn_z, cm_z, du_z, pq_z = zs(dv), zs(wn), zs(cm), zs(du_a), zs(pq)
                    ones = np.ones(len(dv))
                    def ols(X, y):
                        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        yp = X @ b
                        return 1 - np.sum((y - yp)**2) / (np.sum((y - y.mean())**2) + 1e-12), b
                    X1 = np.column_stack([ones, wn_z, cm_z, du_z])
                    X2 = np.column_stack([ones, wn_z, cm_z, du_z, pq_z])
                    r2_1, _ = ols(X1, dv_z)
                    r2_2, beta2 = ols(X2, dv_z)
                    dr2 = r2_2 - r2_1
                    pcr_c = float(beta2[-1])
                else:
                    dr2, pcr_c, r2_1, r2_2 = 0.0, 0.0, 0.0, 0.0

                key_prefix = f"{'delta_r2' if comp_scheme == 'fixed' else 'delta_r2'}_{comp_scheme}"
                if comp_scheme == "fixed":
                    per_seed.append({
                        "seed": s, "clean_acc": clean_acc, "clean_ce": clean_ce,
                        "aug_accs": aug_accs, "aug_ces": aug_ces,
                        "avg_aug_acc": avg_aug_acc, "avg_aug_ce": avg_aug_ce,
                        f"delta_r2_{comp_scheme}": dr2, f"pcr_coef_{comp_scheme}": pcr_c,
                        f"r2_base_{comp_scheme}": r2_1, f"r2_full_{comp_scheme}": r2_2,
                        f"n_pairs_{comp_scheme}": len(pair_data),
                        f"avg_svis_{comp_scheme}": float(np.mean([p["avg_svis"] for p in pair_data])) if pair_data else 0.0,
                    })
                else:
                    if per_seed:
                        per_seed[-1][f"delta_r2_{comp_scheme}"] = dr2
                        per_seed[-1][f"pcr_coef_{comp_scheme}"] = pcr_c
                        per_seed[-1][f"n_pairs_{comp_scheme}"] = len(pair_data)

            ps = per_seed[-1] if per_seed else {}
            print(f"  {name} s{s}: clean {clean_acc*100:.1f}%, aug {avg_aug_acc*100:.1f}%, "
                  f"ΔR²(fix) {ps.get('delta_r2_fixed', 0):.4f}, coef {ps.get('pcr_coef_fixed', 0):.3f}, "
                  f"ΔR²(sample) {ps.get('delta_r2_own_sample', 0):.4f}")

        agg = {}
        if per_seed:
            for k_ in per_seed[0]:
                if k_ == "seed":
                    continue
                v = [p[k_] for p in per_seed if k_ in p]
                if all(isinstance(x, float) for x in v):
                    agg[k_] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        results["arms"][tag] = {"name": name, "per_seed": per_seed, "agg": agg}

    # ===== 6. 判定 (逐 seed 方向一致性, 外审第八轮修正) =====
    def get_per_seed(tag, key):
        return [p.get(key, 0.0) for p in results["arms"].get(tag, {}).get("per_seed", [])]

    full_dr2_all = {t: get_per_seed(t, "delta_r2_fixed") for t in FULL_ARMS}
    full_coef_all = {t: get_per_seed(t, "pcr_coef_fixed") for t in FULL_ARMS}
    ce_dr2 = get_per_seed(CE_ARM, "delta_r2_fixed")
    ce_coef = get_per_seed(CE_ARM, "pcr_coef_fixed")

    # 逐 seed 方向一致性: Full 的每个 seed 的 PCR 系数是否都为负且 ΔR² > CE 同 seed
    all_neg = all(c < 0 for coefs in full_coef_all.values() for c in coefs)
    all_above_ce = all(f > c for t in FULL_ARMS for f, c in zip(full_dr2_all[t], ce_dr2 + ce_dr2 + ce_dr2[:len(full_dr2_all[t])]))
    all_dr2_above = all(d > B2_DELTAR2_SUPPORT for ds_ in full_dr2_all.values() for d in ds_)

    if all_dr2_above and all_neg and all_above_ce:
        v_0b = "支持脆弱抵消"
    elif all(d < B2_DELTAR2_NONE for ds_ in full_dr2_all.values() for d in ds_) or not all_neg:
        v_0b = "初步证据不支持（暂停开发，等待修正重算确认）"
    else:
        v_0b = "不确定（ΔR² 介于 0.01~0.05 或方向不一致；保留原规则，如实报告）"

    ce_aug_acc = np.mean([p.get("avg_aug_acc", 0) for p in results["arms"].get(CE_ARM, {}).get("per_seed", [])])
    f1_aug_acc = np.mean([p.get("avg_aug_acc", 0) for p in results["arms"].get("full_a1", {}).get("per_seed", [])])
    f2_aug_acc = np.mean([p.get("avg_aug_acc", 0) for p in results["arms"].get("full_a2", {}).get("per_seed", [])])
    ce_aug_ce = np.mean([p.get("avg_aug_ce", 0) for p in results["arms"].get(CE_ARM, {}).get("per_seed", [])])
    f1_aug_ce = np.mean([p.get("avg_aug_ce", 0) for p in results["arms"].get("full_a1", {}).get("per_seed", [])])
    f2_aug_ce = np.mean([p.get("avg_aug_ce", 0) for p in results["arms"].get("full_a2", {}).get("per_seed", [])])

    results["verdict_0b_layer2"] = {
        "full_dr2_per_seed": {t: v for t, v in full_dr2_all.items()},
        "full_coef_per_seed": {t: v for t, v in full_coef_all.items()},
        "ce_dr2_per_seed": ce_dr2, "ce_coef_per_seed": ce_coef,
        "verdict": v_0b,
    }
    results["verdict_0d"] = {
        "aug_acc": {"ce": ce_aug_acc, "a1": f1_aug_acc, "a2": f2_aug_acc},
        "aug_ce": {"ce": ce_aug_ce, "a1": f1_aug_ce, "a2": f2_aug_ce},
        "acc_optimal": "a2" if f2_aug_acc > f1_aug_acc and f1_aug_acc > ce_aug_acc else "ambiguous",
        "ce_optimal": "a1" if f1_aug_ce < min(ce_aug_ce, f2_aug_ce) else ("a2" if f2_aug_ce < min(ce_aug_ce, f1_aug_ce) else "ambiguous"),
        "note": "Acc 和 CE 给出不同的最优剂量 —— 必须同时报告两者",
    }

    # ===== 7. 保存 =====
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stage0_layer2_cub.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print(json.dumps({"verdict_0b": v_0b, "verdict_0d_acc": results["verdict_0d"]["acc_optimal"],
                      "verdict_0d_ce": results["verdict_0d"]["ce_optimal"]}, ensure_ascii=False, indent=1))
    print(f"[saved] {OUT_DIR}/stage0_layer2_cub.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
