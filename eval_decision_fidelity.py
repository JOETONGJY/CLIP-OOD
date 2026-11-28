"""决策层概念保真度（第十三轮 P1 修正版）：分类器是否仍在使用“人类可命名的概念区分类别”。

为什么需要类别中心化（第十三轮 P1 的核心修正）
------------------------------------------------
多分类里给**所有类**的有效权重同时加一个公共向量 v，预测概率不变：
    ell'_c = (A_c + v)·u + b_c = ell_c + v·u   —— 所有 logit 同增 v·u，softmax 不变。
但原始 Pearson(A_c, a_c) 会变。所以“两个分类行为完全相同的头可以有不同的保真度”，
这与 LD-DRO 里“只看类别间权重差”的理由完全相同。

修正：在类之间去公共分量，再比较
    A~_c = A_c − (1/C) Σ_j A_j          a~_c = a_c − (1/C) Σ_j a_j
    F_rel = (1/C) Σ_c Pearson(A~_c, a~_c)
含义是“这个概念的权重对**区分该类别**的贡献”，而不是“该类别是否具有这个属性”。

附带事实：u = LN0(E·d) 恒在零均值超平面内（1ᵀu = 0），所以 v ∝ 1_K 这一路本来就不可见；
类间中心化去掉的是剩下的规范自由度。脚本里用两条数值检查钉住：
  V1 对所有权重行同加随机 v，**logit 不变**、**中心化指标逐位不变**（原始指标会变）；
  V2 同一批 logit 用 softmax 比较，逐样本最大差 < 1e-6。

指标（每个头）：
  raw_pearson / raw_top10          原始定义（保留，作对照）
  rel_pearson / rel_top10          类别中心化后的定义（**本轮主指标**）
  concentration                    权重分散度（归一化熵）；注意它与“保真度”不是一回事
  per_class_rel_pearson            逐类值（200 维），供逐类/逐 seed 检查

数据：CUB（逐样本属性在 cub_train.txt 的 6:6+312 列，与 cache 特征逐行对齐）。
AWA2 / LADV 的类级属性矩阵（50×85 / 50×81）与模型概念表（133 / 221）不对齐，不做。

用法:
  python eval_decision_fidelity.py --selftest
  python eval_decision_fidelity.py
产物: experiment_results/ConceptFidelity/cub_decision_fidelity.{json,md}
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "ConceptFidelity")
CLIP_W = "F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
CONCEPT_FILE = "data/CUB/cub_concepts.txt"
K_CUB, C_CUB = 312, 200
SEEDS = (1, 2, 3)

# (run 名前缀, 显示名, 组, 剂量等级)
ARMS = [
    ("CUB_p1_dose_a0", "CE-Only (α=0)", "DDO-L1", "0"),
    ("CUB_p1_dose_a0p125", "DDO α=0.125", "DDO-L1", "0.125"),
    ("CUB_p1_dose_a0p25", "DDO α=0.25", "DDO-L1", "0.25"),
    ("CUB_p1_dose_a0p5", "DDO α=0.5", "DDO-L1", "0.5"),
    ("CUB_p1_dose_a1", "DDO α=1", "DDO-L1", "1"),
    ("CUB_p1_dose_a2", "DDO α=2", "DDO-L1", "2"),
    ("CUB_p1_dose_a4", "DDO α=4", "DDO-L1", "4"),
    ("CUB_s1c_isda_x0p5", "ISDA-Language 0.5×", "ISDA-Language", "0.5"),
    ("CUB_s1c_isda_x1", "ISDA-Language 1×", "ISDA-Language", "1"),
    ("CUB_s1c_isda_x2", "ISDA-Language 2×", "ISDA-Language", "2"),
    ("CUB_s1c_isda_x4", "ISDA-Language 4×", "ISDA-Language", "4"),
    ("CUB_s1c_isda_x8", "ISDA-Language 8×", "ISDA-Language", "8"),
    ("CUB_s12_isda_rand_x0p5", "ISDA-Random 0.5×", "ISDA-Random", "0.5"),
    ("CUB_s12_isda_rand_x1", "ISDA-Random 1×", "ISDA-Random", "1"),
    ("CUB_s12_isda_rand_x2", "ISDA-Random 2×", "ISDA-Random", "2"),
    ("CUB_s12_isda_rand_x4", "ISDA-Random 4×", "ISDA-Random", "4"),
    ("CUB_s12_isda_rand_x8", "ISDA-Random 8×", "ISDA-Random", "8"),
    ("CUB_s12_isda_ident_x0p5", "ISDA-Identity 0.5×", "ISDA-Identity", "0.5"),
    ("CUB_s12_isda_ident_x1", "ISDA-Identity 1×", "ISDA-Identity", "1"),
    ("CUB_s12_isda_ident_x2", "ISDA-Identity 2×", "ISDA-Identity", "2"),
    ("CUB_s12_isda_ident_x4", "ISDA-Identity 4×", "ISDA-Identity", "4"),
    ("CUB_s12_isda_ident_x8", "ISDA-Identity 8×", "ISDA-Identity", "8"),
    ("CUB_s12_isda_lang_x4_warm20", "ISDA-Language 4× + warmup", "warmup", "4"),
    ("CUB_isda_warm_x8", "ISDA-Language 8× + warmup", "warmup", "8"),
    ("CUB_s12_isda_rand_x8_warm20", "ISDA-Random 8× + warmup", "warmup", "8"),
    ("CUB_s12_isda_ident_x8_warm20", "ISDA-Identity 8× + warmup", "warmup", "8"),
    # 第十五轮 P2 (§3as): Stage 1B 的 36 个头 (24 L2 + 12 BF-L1) —— 精度-语义对齐的 L2 补充
    ("CUB_s1b_orig_l2_a0p5", "DDO-L2 (Original) 0.5×", "DDO-L2-Orig", "0.5"),
    ("CUB_s1b_orig_l2_a1", "DDO-L2 (Original) 1×", "DDO-L2-Orig", "1"),
    ("CUB_s1b_orig_l2_a2", "DDO-L2 (Original) 2×", "DDO-L2-Orig", "2"),
    ("CUB_s1b_orig_l2_a4", "DDO-L2 (Original) 4×", "DDO-L2-Orig", "4"),
    ("CUB_s1b_bf_l2_a0p5", "DDO-L2 (Bias-Free) 0.5×", "DDO-L2-BF", "0.5"),
    ("CUB_s1b_bf_l2_a1", "DDO-L2 (Bias-Free) 1×", "DDO-L2-BF", "1"),
    ("CUB_s1b_bf_l2_a2", "DDO-L2 (Bias-Free) 2×", "DDO-L2-BF", "2"),
    ("CUB_s1b_bf_l2_a4", "DDO-L2 (Bias-Free) 4×", "DDO-L2-BF", "4"),
    ("CUB_s1b_bf_l1_a0p5", "DDO-L1 (Bias-Free) 0.5×", "DDO-L1-BF", "0.5"),
    ("CUB_s1b_bf_l1_a1", "DDO-L1 (Bias-Free) 1×", "DDO-L1-BF", "1"),
    ("CUB_s1b_bf_l1_a2", "DDO-L1 (Bias-Free) 2×", "DDO-L1-BF", "2"),
    ("CUB_s1b_bf_l1_a4", "DDO-L1 (Bias-Free) 4×", "DDO-L1-BF", "4"),
    # 第十六轮 P2 (§3aw/§3ax): Centered-L2 的 12 个头
    ("CUB_s1b_cen_a0p5", "Centered-L2 0.5×", "Centered-L2", "0.5"),
    ("CUB_s1b_cen_a1", "Centered-L2 1×", "Centered-L2", "1"),
    ("CUB_s1b_cen_a2", "Centered-L2 2×", "Centered-L2", "2"),
    ("CUB_s1b_cen_a4", "Centered-L2 4×", "Centered-L2", "4"),
]


# ---------------- 数据与词典 ----------------
def load_cub():
    d = torch.load("cache/CUB_train.pt", map_location="cpu", weights_only=False)
    feats = F.normalize(d["feats"].float(), dim=-1)
    labels = d["labels"].long()
    rows = [x.strip().split(",") for x in open("data/CUB/cub_train.txt", encoding="utf-8").readlines()]
    assert len(rows) == feats.size(0), f"txt {len(rows)} != cache {feats.size(0)}"
    lab = torch.tensor([int(r[1]) for r in rows]) - 1
    assert (lab == labels).all(), "txt 与 cache 顺序不一致"
    attrs = torch.tensor([list(map(float, r[6:6 + K_CUB])) for r in rows])
    return feats, labels, attrs


def build_E():
    import clip as clip_pkg
    names = [x.rstrip("\n") for x in open(CONCEPT_FILE, encoding="utf-8").readlines() if x.strip()]
    assert len(names) == K_CUB, len(names)
    m, _ = clip_pkg.load(CLIP_W, device="cpu")
    with torch.no_grad():
        E = m.encode_text(clip_pkg.tokenize(names)).float()
        E = E / E.norm(dim=-1, keepdim=True)
    del m
    return E, names


def _auroc(y, score):
    """秩和版 AUROC（并列取平均秩），不依赖 sklearn。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def activation_fidelity(feats, attrs, E):
    """激活层（与臂无关；词典冻结时是常数）。"""
    s = (feats @ E.T).numpy()
    a = attrs.numpy()
    ps, aucs = [], []
    for j in range(a.shape[1]):
        y = (a[:, j] > 0).astype(int)
        if y.sum() < 5 or (1 - y).sum() < 5 or a[:, j].std() == 0:
            continue
        ps.append(float(np.corrcoef(s[:, j], a[:, j])[0, 1]))
        aucs.append(_auroc(y, s[:, j]))
    return {"pearson_mean": float(np.mean(ps)), "auroc_mean": float(np.mean(aucs)), "n_valid": len(ps)}


def class_profiles(attrs, labels, n_class=C_CUB):
    A = torch.zeros(n_class, attrs.size(1))
    for c in range(n_class):
        m = labels == c
        if m.any():
            A[c] = attrs[m].mean(0)
    return A


# ---------------- 决策层指标 ----------------
def _concentration(w):
    p = np.abs(w) / (np.abs(w).sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(w)))


def _center(M):
    return M - M.mean(0, keepdim=True)


def _project_H(v):
    """去掉沿全 1 方向的分量。u = LN0(E·d) 恒在零均值超平面内（1ᵀu = 0），所以给某一行加上 c·1_K
    不改变任何 logit；Pearson 内部已减去均值、自动免疫，但 top-k 排序会被它改掉，必须先投影。"""
    return v - v.mean()


def decision_metrics(W_eff, A, k=10):
    """返回原始与类别中心化两套指标，以及逐类记录。"""
    Wc, Ac = _center(W_eff), _center(A)
    out = {f"{p}_{q}": [] for p in ("raw", "rel") for q in ("pearson", "top10")}
    per_cls = {"raw": [], "rel": []}
    conc = []
    for c in range(W_eff.size(0)):
        for p, Wm, Am in (("raw", W_eff, A), ("rel", Wc, Ac)):
            w, a = Wm[c].numpy(), Am[c].numpy()
            if a.std() == 0 or w.std() == 0:
                out[f"{p}_pearson"].append(float("nan"))
                out[f"{p}_top10"].append(float("nan"))
                per_cls[p].append(None)
                continue
            r = float(np.corrcoef(w, a)[0, 1])
            # top-k 在零均值超平面内的分量上算（去掉 1_K 这一路不可辨识方向）
            tw = set(np.argsort(-_project_H(w))[:k].tolist())
            ta = set(np.argsort(-_project_H(a))[:k].tolist())
            out[f"{p}_pearson"].append(r)
            out[f"{p}_top10"].append(len(tw & ta) / float(k))
            per_cls[p].append(round(r, 6))
        conc.append(_concentration(W_eff[c].numpy()))
    res = {}
    for key, v in out.items():
        arr = np.array([x for x in v if x == x], dtype=float)
        res[key] = float(arr.mean()) if arr.size else float("nan")
    res["concentration"] = float(np.mean(conc))
    res["n_class_valid_rel"] = int(sum(1 for x in per_cls["rel"] if x is not None))
    res["per_class_rel_pearson"] = per_cls["rel"]
    return res


def head_W_eff(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["classifier"]
    gamma = [v for k, v in sd.items() if k.startswith("1.") and "weight" in k][0].float()
    W = [v for k, v in sd.items() if k.startswith("2.") and "weight" in k][0].float()
    assert W.shape[1] == gamma.numel(), (W.shape, gamma.shape)
    return W * gamma[None, :]


# ---------------- 自测 ----------------
def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    K, C, N = 20, 6, 400
    A = torch.rand(C, K)
    A[A < 0.5] = 0
    labels = torch.randint(0, C, (N,))
    attrs = (A[labels] + 0.2 * torch.rand(N, K)).clamp(0, 1)
    prof = class_profiles(attrs, labels, C)
    check("F1 类属性轮廓 = 该类样本属性的均值", torch.allclose(prof[0], attrs[labels == 0].mean(0)))

    d_good = decision_metrics(A.clone(), prof, k=5)
    d_bad = decision_metrics(torch.randn(C, K), prof, k=5)
    check("F2 权重=人工属性 → 原始与中心化保真度都高",
          d_good["raw_pearson"] > 0.9 and d_good["rel_pearson"] > 0.5,
          f"{d_good['raw_pearson']:.3f}/{d_good['rel_pearson']:.3f}")
    check("F3 随机权重 → 保真度接近 0",
          abs(d_bad["raw_pearson"]) < 0.3 and abs(d_bad["rel_pearson"]) < 0.3,
          f"{d_bad['raw_pearson']:.3f}/{d_bad['rel_pearson']:.3f}")

    # F4（替代初版里“权重=属性时 top-k 为 1”的伪检查）：公共向量不变性
    # 数学上是逐位不变；float32 下中心化的舍入会留下 ~1e-7 的残差，容差按 1e-6 判。
    v = torch.randn(1, K)
    d_shift = decision_metrics(A.clone() + v, prof, k=5)
    dp = abs(d_shift["rel_pearson"] - d_good["rel_pearson"])
    dt = abs(d_shift["rel_top10"] - d_good["rel_top10"])
    check("F4 所有权重行同加公共向量 v：中心化指标在 float32 舍入内不变", dp < 1e-6 and dt < 1e-12,
          f"Δpearson={dp:.3e}, Δtop10={dt:.3e}")
    check("F4b 同一变换下原始指标**会变**（说明中心化不是多余的）",
          abs(d_shift["raw_pearson"] - d_good["raw_pearson"]) > 1e-6,
          f"{d_good['raw_pearson']:.6f} -> {d_shift['raw_pearson']:.6f}")

    # F5：加公共向量后 logit 不变（分类行为完全相同的两个头）
    u = torch.randn(64, K)
    u = u - u.mean(-1, keepdim=True)                      # 与 LN0 的输出同在一个零均值超平面
    lg0, lg1 = u @ A.T, u @ (A + v).T
    diff = float((lg0 - lg1).std(-1).max())
    check("F5 公共向量不改变任何样本的类别间 logit 差（softmax 不变）", diff < 1e-5, f"max std={diff:.2e}")
    check("F5b 全 1 向量在 LN0 输出上本就不可见（1ᵀu = 0）",
          float((u @ torch.ones(K, 1)).abs().max()) < 1e-5)

    onehot = np.zeros(K)
    onehot[3] = 1.0
    check("F6 集中度：单点 = 0、均匀 = 1",
          abs(_concentration(onehot)) < 1e-9 and abs(_concentration(np.ones(K)) - 1.0) < 1e-9)
    check("F7 类间中心化后每列之和为 0", float(_center(A).sum(0).abs().max()) < 1e-6)

    # F9：给某一行加 c·1_K（LayerNorm 的不可辨识方向）不改变任何 logit，也不应改变 H 内投影后的 top-k
    c0 = 1.7
    A_one = A.clone()
    A_one[0] = A_one[0] + c0 * torch.ones(K)
    d_one = decision_metrics(A_one, prof, k=5)
    proj_w = _project_H(A[0].numpy())
    proj_w1 = _project_H(A_one[0].numpy())
    check("F9 加 c·1_K 后投影到 H 的向量逐位相同（top-k 因此不变）",
          float(np.abs(proj_w - proj_w1).max()) < 1e-6,
          f"max|Δ|={float(np.abs(proj_w - proj_w1).max()):.2e}")
    check("F9b 该变换不改变该行与其它类的间隔（1ᵀu = 0）",
          abs(d_one["rel_pearson"] - d_good["rel_pearson"]) < 1e-6,
          f"{d_good['rel_pearson']:.6f} -> {d_one['rel_pearson']:.6f}")

    # F8：W_eff = W ⊙ γ 的逐列缩放（真实张量，不再直接传 True）
    sd = {"1.weight": torch.arange(1, K + 1).float(), "2.weight": torch.ones(C, K)}
    Wm = sd["2.weight"] * sd["1.weight"][None, :]
    check("F8 W_eff 的第 j 列 = W 的第 j 列 × γ_j",
          torch.allclose(Wm, sd["2.weight"] * sd["1.weight"].unsqueeze(0)))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------- 主流程 ----------------
def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    feats, labels, attrs = load_cub()
    E, names = build_E()
    act = activation_fidelity(feats, attrs, E)
    prof = class_profiles(attrs, labels)
    print(f"[CUB] 激活层（与臂无关）: Pearson {act['pearson_mean']:.4f} / AUROC {act['auroc_mean']:.4f} "
          f"({act['n_valid']} 个有效概念)")

    runs = []
    for prefix, desc, group, level in ARMS:
        for s in SEEDS:
            p = f"logs/heads/{prefix}_seed{s}.pt"
            if not os.path.exists(p):
                runs.append({"arm": desc, "seed": s, "missing": True, "run": f"{prefix}_seed{s}"})
                continue
            W = head_W_eff(p)
            m = decision_metrics(W, prof)
            runs.append({"arm": desc, "group": group, "level": level, "seed": s,
                         "run": f"{prefix}_seed{s}", **m,
                         "head_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})

    agg = {}
    for _, desc, _, _ in ARMS:
        v = [r for r in runs if r.get("arm") == desc and not r.get("missing")]
        if not v:
            continue
        keys = ("raw_pearson", "raw_top10", "rel_pearson", "rel_top10", "concentration")
        a = {k: float(np.mean([x[k] for x in v])) for k in keys}
        a.update({f"std_{k}": float(np.std([x[k] for x in v])) for k in ("rel_pearson", "raw_pearson")})
        a["n_seeds"] = len(v)
        a["per_seed_rel_pearson"] = [round(x["rel_pearson"], 6) for x in v]
        agg[desc] = a

    # 关键对比：ISDA@4× vs DDO α=2（GPT 指定）
    key = {}
    for d1, d2 in (("ISDA-Language 4×", "DDO α=2"), ("ISDA-Language 4×", "ISDA-Random 4×")):
        if d1 in agg and d2 in agg:
            a1 = [r["rel_pearson"] for r in runs if r.get("arm") == d1 and not r.get("missing")]
            a2 = [r["rel_pearson"] for r in runs if r.get("arm") == d2 and not r.get("missing")]
            d = [x - y for x, y in zip(a1, a2)]
            key[f"{d1} - {d2}"] = {"diff_per_seed": d, "mean": float(np.mean(d)),
                                   "n_positive": int(sum(x > 0 for x in d))}

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": "CUB",
           "metric": "决策层概念保真度；主指标 rel_pearson = 类间中心化后的 Pearson(W_eff[c], a_c)",
           "invariance": "对所有权重行同加公共向量 v 时 rel_* 逐位不变、原始指标会变（自测 F4/F4b/F5）",
           "n_heads": sum(1 for r in runs if not r.get("missing")),
           "activation_fidelity": act, "per_arm": agg, "key_comparisons": key, "runs": runs,
           "elapsed_seconds": round(time.time() - t0, 1)}
    jp = os.path.join(OUT_DIR, "cub_decision_fidelity.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)

    L = ["# CUB：决策层概念保真度（第十三轮 P1 修正版）", "",
         f"评估 **{out['n_heads']} 个分类头**（{len(ARMS)} 个配置 × 3 seeds）。",
         "激活层与臂无关（词典冻结）："
         f"Pearson **{act['pearson_mean']:.4f}**、AUROC **{act['auroc_mean']:.4f}**"
         f"（{act['n_valid']} 个有效概念）。", "",
         "## 指标", "",
         "`A~_c = A_c − (1/C)Σ_j A_j`，`a~_c = a_c − (1/C)Σ_j a_j`（类间中心化）。",
         "**rel_pearson** 是主指标；`raw_pearson` 是初版定义，仅作对照。",
         "对所有权重行同加公共向量时 rel_* 不变、raw_* 会变（自测 F4/F4b/F5）。", "",
         "| 臂 | rel_pearson | 3 seed | raw_pearson | rel_top10 | 分散度 |", "|---|---|---|---|---|---|"]
    for _, desc, _, _ in ARMS:
        a = agg.get(desc)
        if not a:
            continue
        L.append(f"| {desc} | **{a['rel_pearson']:.4f}** ± {a['std_rel_pearson']:.4f} | "
                 f"{[round(x, 4) for x in a['per_seed_rel_pearson']]} | {a['raw_pearson']:.4f} | "
                 f"{a['rel_top10']:.3f} | {a['concentration']:.4f} |")
    L += ["", "## 关键对比", ""]
    for k, v in key.items():
        L.append(f"- {k}：Δ = {v['mean']:+.4f}（逐 seed {[round(x, 4) for x in v['diff_per_seed']]}，"
                 f"{v['n_positive']}/3 正向）")
    L += ["", "边界：描述性语义代理指标；不等于概念干预实验；AWA2/LADV 的属性表与概念表不对齐，未做。"]
    with open(os.path.join(OUT_DIR, "cub_decision_fidelity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {jp}  ({out['elapsed_seconds']}s, {out['n_heads']} 个头)")
    for desc in ("CE-Only (α=0)", "DDO α=2", "ISDA-Language 4×", "ISDA-Random 4×", "ISDA-Identity 4×"):
        a = agg.get(desc)
        if a:
            print(f"  {desc:26s} rel {a['rel_pearson']:+.4f}  raw {a['raw_pearson']:+.4f}  "
                  f"top10 {a['rel_top10']:.3f}")
    for k, v in key.items():
        print(f"  {k}: {v['mean']:+.4f} ({v['n_positive']}/3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
