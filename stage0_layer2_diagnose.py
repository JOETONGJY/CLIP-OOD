"""Stage 0B 第二层 + 0D 诊断 (预注册见 RESULTS.md §3t)。
零训练: 只读增强缓存 (cache/CUB_train_aug.pt)、27 个 Stage 0 分类头 (logs/heads/)、
概念矩阵 E 与域方向 (从数据集对象获取)。只用源域数据。

0B 第二层: 低 PCR 的类别边界在视觉扰动下是否更脆弱 (连续变量, 控制权重规模与干净间隔后的增量解释力)
0D: 源域增强验证能否 source-only 地选参 (α=0/1/2 头在 7 种扰动下的 heldout_val 精度)

用法: python stage0_layer2_diagnose.py
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

# ---- 预注册阈值 (§3t) ----
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
    u_raw = z.double() @ E.T                                            # (N, 312)
    u_norm = (u_raw - u_raw.mean(-1, keepdim=True)) / torch.sqrt(u_raw.var(-1, unbiased=False, keepdim=True) + EPS_LN)
    return u_norm @ (W * gam[None, :]).T + W @ beta + b


def top_competitors(logits, labels, n_classes, k=TOP_K):
    """每个类别的 top-k 竞争类: 在该类样本的平均干净 logit 中排前 k (排除自身)。返回 {y: [k1,...,kk]}"""
    comps = {}
    for y in range(n_classes):
        mask = labels == y
        if mask.sum() == 0:
            continue
        avg_logit = logits[mask].mean(0)
        avg_logit[y] = float("-inf")
        comps[y] = torch.topk(avg_logit, k).indices.tolist()
    return comps


def main():
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
    clean_val = aug["clean_feats"][val_idx].float()          # (n_val, 768)
    aug_names = list(aug["aug_feats"].keys())
    aug_val = {name: aug["aug_feats"][name][val_idx].float() for name in aug_names}
    print(f"  heldout_val: {n_val} 个样本, {aug_names}")

    # ===== 2. 加载概念矩阵与域方向 =====
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

    # 域方向二阶矩 G 与 PCR 基线
    U = domain_directions(diffs, E)
    G, _, mu, _ = second_moments(U)
    Dt = diag_reference(G)
    base_info, base_samples = random_baseline(G, n=40000, seed=0)
    print(f"  概念矩阵: {E.shape}, 类别数: {n_classes}, G 迹: {torch.trace(G):.2f}")

    # ===== 3. 加载 27 个分类头 =====
    heads = load_heads()
    check(f"27 个分类头都已加载", len(heads) == 27, f"{len(heads)}")

    # ===== 4. 竞争类 (两套口径) =====
    print("确定竞争类...")
    ce_head = heads[(CE_ARM, 1)]
    ce_clean_logits = compute_logits(clean_val, ce_head, E).detach()
    fixed_comps = top_competitors(ce_clean_logits, labels_val, n_classes)

    # ===== 5. 逐头计算 =====
    print("逐头计算 margin damage 与 PCR...")
    results = {"arms": {}, "aug_names": aug_names, "n_val": n_val, "n_classes": n_classes}

    for tag, name in ARMS:
        per_seed = []
        for s in SEEDS:
            h = heads.get((tag, s))
            if h is None:
                continue
            hp = h["classifier"]
            W, gam = hp["2.weight"].double(), hp["1.weight"].double()
            A = W * gam[None, :]

            # ---- Layer 1 指标 (PCR) ----
            y_i, k_i = torch.triu_indices(A.size(0), A.size(0), offset=1)
            K_dim = A.size(1)
            Hm = torch.eye(K_dim, dtype=torch.float64) - torch.ones(K_dim, K_dim, dtype=torch.float64) / K_dim
            Wk = (A[y_i] - A[k_i]) @ Hm
            num_q, den_q = quad(Wk, G), quad(Wk, Dt).clamp_min(1e-300)
            pcr_vals = num_q / den_q
            qtl_vals = pcr_quantile(pcr_vals, base_samples)
            # 建立 (y,k) → 索引 的映射
            pair_qtl = {}
            for pi in range(len(y_i)):
                pair_qtl[(int(y_i[pi]), int(k_i[pi]))] = float(qtl_vals[pi])
                pair_qtl[(int(k_i[pi]), int(y_i[pi]))] = float(qtl_vals[pi])

            # ---- 干净与增强 logits ----
            clean_logits = compute_logits(clean_val, h, E).detach()
            own_comps = top_competitors(clean_logits, labels_val, n_classes)

            # 干净准确率 (0D)
            clean_acc = float((clean_logits.argmax(1) == labels_val).double().mean())
            clean_ce = float(torch.nn.functional.cross_entropy(clean_logits, labels_val))

            # 0D: 各扰动下的准确率与 CE
            aug_accs, aug_ces = {}, {}
            for an in aug_names:
                pert_logits = compute_logits(aug_val[an], h, E).detach()
                aug_accs[an] = float((pert_logits.argmax(1) == labels_val).double().mean())
                aug_ces[an] = float(torch.nn.functional.cross_entropy(pert_logits, labels_val))
            avg_aug_acc = float(np.mean(list(aug_accs.values())))
            avg_aug_ce = float(np.mean(list(aug_ces.values())))

            # ---- 0B 第二层: margin damage ----
            # 对两种竞争类口径分别计算
            for comp_scheme, comps in [("fixed", fixed_comps), ("own", own_comps)]:
                # 逐类别对的 margin damage
                pair_damage = {}       # (y, k) → list of D_yk across samples × perturbations
                pair_clean_margin = {} # (y, k) → list of m_yk across samples
                pert_strength = []     # ‖Δu‖ across samples × perturbations

                for an in aug_names:
                    pert_logits = compute_logits(aug_val[an], h, E).detach()
                    # 概念空间扰动
                    u_clean = clean_val.double() @ E.T
                    u_pert = aug_val[an].double() @ E.T
                    du = (u_pert - u_clean)
                    du_norm = du / torch.sqrt(u_clean.var(-1, unbiased=False, keepdim=True) + EPS_LN)  # 近似归一化后的 Δu
                    # 更准确: 分别归一化再相减
                    u_c_n = (u_clean - u_clean.mean(-1, keepdim=True)) / torch.sqrt(u_clean.var(-1, unbiased=False, keepdim=True) + EPS_LN)
                    u_p_n = (u_pert - u_pert.mean(-1, keepdim=True)) / torch.sqrt(u_pert.var(-1, unbiased=False, keepdim=True) + EPS_LN)
                    du_norm = u_p_n - u_c_n
                    du_strength = du_norm.norm(dim=-1)  # (n_val,)

                    for si in range(n_val):
                        y = int(labels_val[si])
                        for k in comps.get(y, []):
                            m_clean = float(clean_logits[si, y] - clean_logits[si, k])
                            m_pert = float(pert_logits[si, y] - pert_logits[si, k])
                            d = m_clean - m_pert
                            key = (y, k)
                            pair_damage.setdefault(key, []).append(d)
                            pair_clean_margin.setdefault(key, []).append(m_clean)
                        pert_strength.append(float(du_strength[si]))

                # 汇总为类别对层面的统计
                pair_data = []
                for (y, k), damages in pair_damage.items():
                    qtl = pair_qtl.get((y, k))
                    if qtl is None:
                        continue
                    w_norm = float((A[y] - A[k]).norm())
                    pair_data.append({
                        "y": y, "k": k,
                        "avg_damage": float(np.mean(damages)),
                        "avg_clean_margin": float(np.mean(pair_clean_margin[(y, k)])),
                        "pcr_quantile": qtl,
                        "w_norm": w_norm,
                        "w_Gw": float(quad((A[y] - A[k]) @ Hm, G)),
                        "w_Dw": float(quad((A[y] - A[k]) @ Hm, Dt)),
                    })

                # ---- 回归分析 ----
                if len(pair_data) > 20:
                    dv = np.array([p["avg_damage"] for p in pair_data])
                    wn = np.array([p["w_norm"] for p in pair_data])
                    cm = np.array([p["avg_clean_margin"] for p in pair_data])
                    pq = np.array([p["pcr_quantile"] for p in pair_data])
                    du = np.array([p.get("avg_pert_strength", np.mean(pert_strength)) for p in pair_data])

                    # 标准化
                    zscore = lambda v: (v - v.mean()) / (v.std() + 1e-12)
                    dv_z, wn_z, cm_z, pq_z = zscore(dv), zscore(wn), zscore(cm), zscore(pq)
                    n_obs = len(dv)
                    ones = np.ones(n_obs)

                    def ols_r2(X, y):
                        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        y_pred = X @ beta
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2)
                        return 1 - ss_res / (ss_tot + 1e-12), beta

                    X1 = np.column_stack([ones, wn_z, cm_z])         # 基线
                    X2 = np.column_stack([ones, wn_z, cm_z, pq_z])   # + PCR
                    r2_1, _ = ols_r2(X1, dv_z)
                    r2_2, beta2 = ols_r2(X2, dv_z)
                    delta_r2 = r2_2 - r2_1
                    pcr_coef = float(beta2[-1])  # PCR 的标准化系数
                else:
                    delta_r2, pcr_coef, r2_1, r2_2 = 0.0, 0.0, 0.0, 0.0
                    n_obs = len(pair_data)

                if comp_scheme == "fixed":
                    per_seed.append({
                        "seed": s, "clean_acc": clean_acc, "clean_ce": clean_ce,
                        "aug_accs": aug_accs, "aug_ces": aug_ces,
                        "avg_aug_acc": avg_aug_acc, "avg_aug_ce": avg_aug_ce,
                        "delta_r2_fixed": delta_r2, "pcr_coef_fixed": pcr_coef,
                        "r2_base_fixed": r2_1, "r2_full_fixed": r2_2,
                        "n_pairs_fixed": n_obs,
                        "avg_pert_strength": float(np.mean(pert_strength)),
                    })
                else:
                    per_seed[-1]["delta_r2_own"] = delta_r2
                    per_seed[-1]["pcr_coef_own"] = pcr_coef
                    per_seed[-1]["n_pairs_own"] = n_obs

            print(f"  {name} s{s}: clean {clean_acc*100:.1f}%, avg_aug {avg_aug_acc*100:.1f}%, "
                  f"ΔR²(fixed) {per_seed[-1]['delta_r2_fixed']:.4f}, PCR coef {per_seed[-1]['pcr_coef_fixed']:.3f}")

        agg = {}
        if per_seed:
            for k_ in per_seed[0]:
                if k_ == "seed":
                    continue
                v = [p[k_] for p in per_seed if k_ in p]
                if all(isinstance(x, float) for x in v):
                    agg[k_] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        results["arms"][tag] = {"name": name, "per_seed": per_seed, "agg": agg}

    # ===== 6. 判定 =====
    def get_agg(tag, key):
        a = results["arms"].get(tag, {}).get("agg", {})
        return a.get(key, (0, 0))[0]

    # 0B 第二层判定
    full_dr2 = [get_agg(t, "delta_r2_fixed") for t in FULL_ARMS]
    ce_dr2 = get_agg(CE_ARM, "delta_r2_fixed")
    full_coef = [get_agg(t, "pcr_coef_fixed") for t in FULL_ARMS]
    # 检查方向一致性: Full 的 ΔR² > CE 的 ΔR², 且 PCR 系数为负
    if all(d > B2_DELTAR2_SUPPORT for d in full_dr2) and all(c < 0 for c in full_coef) and all(d > ce_dr2 for d in full_dr2):
        v_0b = "支持脆弱抵消"
    elif all(d < B2_DELTAR2_NONE for d in full_dr2) or any(c >= 0 for c in full_coef):
        v_0b = "不支持"
    else:
        v_0b = "不确定"
    results["verdict_0b_layer2"] = {
        "full_delta_r2": full_dr2, "ce_delta_r2": ce_dr2, "full_pcr_coef": full_coef,
        "verdict": v_0b,
    }

    # 0D 判定 (描述性)
    ce_aug = get_agg(CE_ARM, "avg_aug_acc")
    f1_aug = get_agg("full_a1", "avg_aug_acc")
    f2_aug = get_agg("full_a2", "avg_aug_acc")
    ce_cl = get_agg(CE_ARM, "clean_acc")
    f1_cl = get_agg("full_a1", "clean_acc")
    f2_cl = get_agg("full_a2", "clean_acc")
    results["verdict_0d"] = {
        "clean": {"ce": ce_cl, "full_a1": f1_cl, "full_a2": f2_cl},
        "avg_aug": {"ce": ce_aug, "full_a1": f1_aug, "full_a2": f2_aug},
        "ddo_advantage_clean": {"a1": f1_cl - ce_cl, "a2": f2_cl - ce_cl},
        "ddo_advantage_aug": {"a1": f1_aug - ce_aug, "a2": f2_aug - ce_aug},
    }

    # ===== 7. 保存 =====
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stage0_layer2_cub.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ===== 8. Markdown 报告 =====
    L = ["# Stage 0B 第二层 + 0D（CUB，heldout_val 1200 样本，2026-11-26）", "",
         "预注册见 `RESULTS.md` §3t。探索性诊断（现有头见过完整训练集）。", "",
         "## 0D：源域增强验证（α=0/1/2 在 7 种扰动下的 heldout_val 精度）", "",
         "| 臂 | 干净 | 7 种扰动均值 | DDO 优势（干净） | DDO 优势（增强） |", "|---|---|---|---|---|"]
    v0d = results["verdict_0d"]
    L.append(f"| CE-Only | {v0d['clean']['ce']*100:.2f}% | {v0d['avg_aug']['ce']*100:.2f}% | — | — |")
    L.append(f"| Full α=1 | {v0d['clean']['full_a1']*100:.2f}% | {v0d['avg_aug']['full_a1']*100:.2f}% | "
             f"+{v0d['ddo_advantage_clean']['a1']*100:.2f} | +{v0d['ddo_advantage_aug']['a1']*100:.2f} |")
    L.append(f"| Full α=2 | {v0d['clean']['full_a2']*100:.2f}% | {v0d['avg_aug']['full_a2']*100:.2f}% | "
             f"+{v0d['ddo_advantage_clean']['a2']*100:.2f} | +{v0d['ddo_advantage_aug']['a2']*100:.2f} |")
    L += ["", "各扰动的逐项精度：", "", "| 臂 | " + " | ".join(aug_names) + " |", "|---|" + "---|" * len(aug_names)]
    for tag in (CE_ARM, "full_a1", "full_a2"):
        a = results["arms"].get(tag, {}).get("agg", {})
        accs = a.get("aug_acc_mean_by_type", {})
        if not accs:
            per_s = results["arms"].get(tag, {}).get("per_seed", [])
            if per_s:
                accs = {an: float(np.mean([p["aug_accs"][an] for p in per_s if "aug_accs" in p])) for an in aug_names}
        row = f"| {results['arms'][tag]['name']} | " + " | ".join(f"{accs.get(an, 0)*100:.1f}%" for an in aug_names) + " |"
        L.append(row)

    L += ["", "## 0B 第二层：PCR 的增量解释力（ΔR²）", "",
          "| 臂 | ΔR²（固定竞争类） | PCR 系数 | ΔR²（自身竞争类） | 配对数 |", "|---|---|---|---|---|"]
    for tag, name in ARMS:
        a = results["arms"].get(tag, {}).get("agg", {})
        if not a:
            continue
        L.append(f"| {name} | {a.get('delta_r2_fixed', (0,0))[0]:.4f} ± {a.get('delta_r2_fixed', (0,0))[1]:.4f} | "
                 f"{a.get('pcr_coef_fixed', (0,0))[0]:+.3f} | {a.get('delta_r2_own', (0,0))[0]:.4f} | "
                 f"{a.get('n_pairs_fixed', (0,0))[0]:.0f} |")
    v0b = results["verdict_0b_layer2"]
    L += ["", f"**判定：{v0b['verdict']}**", "",
          f"Full ΔR²: {v0b['full_delta_r2']}, CE ΔR²: {v0b['ce_delta_r2']:.4f}, PCR 系数: {v0b['full_pcr_coef']}", "",
          "## 读法", "",
          "- ΔR² 是在控制 ‖w‖ 与干净间隔后，PCR 对间隔损害的增量解释力（§3t 预注册）。",
          "- 探索性诊断：现有头见过完整训练集（含 heldout_val），结果不作为正式选参依据。",
          "- 无论结果如何都不下 \"Shrinkage 有效\" 的结论 —— 那需要干预实验（第三层）。", ""]
    with open(os.path.join(OUT_DIR, "stage0_layer2_cub.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(json.dumps({"verdict_0b_layer2": v0b["verdict"], "verdict_0d": v0d}, ensure_ascii=False, indent=1))
    print(f"[saved] {OUT_DIR}/stage0_layer2_cub.json, stage0_layer2_cub.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
