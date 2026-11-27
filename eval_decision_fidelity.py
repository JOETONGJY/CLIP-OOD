"""决策层概念保真度：第三章各臂的分类头是否仍在使用“人类可命名的概念”（第十二轮 P4 的补充项）。

为什么另立一个脚本
------------------
`eval_concept_alignment.py`（E2-R）衡量的是**概念激活**（z·Eᵀ）与逐样本人工属性标注的一致性。
第三章的所有臂都冻结概念词典 E、也不做 P4 校准，所以它们的激活逐位相同——那个指标对第三章是常数，
逐臂跑只会得到同一个数字。要回答“DDO / ISDA 这类正则有没有把分类器推离人类语义”，必须看**决策层**：

  激活层保真度（与臂无关，只算一次）：Pearson / AUROC（复算 §2f 的冻结词典基线）
  决策层保真度（与臂有关）：
    fid_pearson_c  = corr(W_eff[c], a_c)          a_c = 类 c 的逐概念属性频率（人工标注，312 维）
    top10_overlap_c = |top10(W_eff[c]) ∩ top10(a_c)| / 10
    concentration_c = H(|W_eff[c]| / Σ|W_eff[c]|) / log(K)   越接近 1 越分散（越不可解释）

W_eff = W ⊙ γ（有效权重，与 §3y 的 A 同一定义）；b 不参与（保真度只看方向）。

数据：CUB（逐样本属性在 cub_train.txt 的 6:6+312 列，与 cache 特征逐行对齐，已在 E2-R 中核对过顺序）。
AWA2 / LADV 只有类级属性、且列与概念表不一定对齐 —— 本脚本做尺寸检查，对不上就跳过并如实记录。

用法:
  python eval_decision_fidelity.py --selftest
  python eval_decision_fidelity.py --dataset CUB
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
CONCEPT_FILES = {"CUB": "data/CUB/cub_concepts.txt"}
K_CUB = 312

# 需要评估的臂：(run 名前缀, 说明)；每个 run 有 seed 1/2/3
ARMS = [
    ("CUB_p1_dose_a0", "CE-Only (α=0)"),
    ("CUB_p1_dose_a1", "DDO α=1"),
    ("CUB_p1_dose_a2", "DDO α=2"),
    ("CUB_p1_dose_a4", "DDO α=4"),
    ("CUB_s1c_isda_x2", "Language-ISDA 2×"),
    ("CUB_s1c_isda_x4", "Language-ISDA 4×"),
    ("CUB_s1c_isda_x8", "Language-ISDA 8×"),
    ("CUB_s12_isda_lang_x4_warm20", "Language-ISDA 4× + warmup"),
    ("CUB_s12_isda_rand_x4", "Random-SpecMatched 4×"),
    ("CUB_s12_isda_ident_x4", "Identity-H 4×"),
]
SEEDS = (1, 2, 3)


def load_cub():
    """返回 (feats_norm, labels, attrs)。顺序与 E2-R 相同（txt 与 cache 逐行核对）。"""
    d = torch.load("cache/CUB_train.pt", map_location="cpu", weights_only=False)
    feats = F.normalize(d["feats"].float(), dim=-1)
    labels = d["labels"].long()
    rows = [x.strip().split(",") for x in open("data/CUB/cub_train.txt", encoding="utf-8").readlines()]
    assert len(rows) == feats.size(0), f"txt {len(rows)} != cache {feats.size(0)}"
    lab = torch.tensor([int(r[1]) for r in rows]) - 1
    assert (lab == labels).all(), "txt 与 cache 顺序不一致"
    attrs = torch.tensor([list(map(float, r[6:6 + K_CUB])) for r in rows])
    return feats, labels, attrs


def build_E(names):
    import clip as clip_pkg
    clip_model, _ = clip_pkg.load(CLIP_W, device="cpu")
    with torch.no_grad():
        E = clip_model.encode_text(clip_pkg.tokenize(names)).float()
        E = E / E.norm(dim=-1, keepdim=True)
    del clip_model
    return E


def _auroc(y, score):
    """秩和（Mann-Whitney U）版 AUROC，并列取平均秩 —— 与 sklearn 的定义一致，
    但不依赖 scikit-learn（本环境的 venv 没有装它）。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # 并列取平均秩
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
    """激活层：逐概念 Pearson 与 AUROC（复算 E2-R 的冻结基线）。"""
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


def class_profiles(attrs, labels, n_class=200):
    """每个类的逐概念属性频率 a_c（人工标注的类均值，∈[0,1]）。"""
    A = torch.zeros(n_class, attrs.size(1))
    for c in range(n_class):
        m = labels == c
        if m.any():
            A[c] = attrs[m].mean(0)
    return A


def _concentration(w):
    """归一化熵：单点权重 = 0，均匀 = 1。"""
    p = np.abs(w) / (np.abs(w).sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(w)))


def decision_fidelity(W_eff, A, k=10):
    """逐类：与人工属性轮廓的 Pearson、top-k 概念重合、权重集中度。"""
    ps, ov, conc = [], [], []
    for c in range(W_eff.size(0)):
        w = W_eff[c].numpy()
        a = A[c].numpy()
        if a.std() == 0 or w.std() == 0:
            continue
        ps.append(float(np.corrcoef(w, a)[0, 1]))
        tw, ta = set(np.argsort(-w)[:k].tolist()), set(np.argsort(-a)[:k].tolist())
        ov.append(len(tw & ta) / float(k))
        conc.append(_concentration(w))
    return {"fid_pearson": float(np.mean(ps)), "top10_overlap": float(np.mean(ov)),
            "concentration": float(np.mean(conc)), "n_class_valid": len(ps)}


def head_W_eff(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["classifier"]
    keys = {k.split(".")[-1]: v for k, v in sd.items()}
    gamma = [v for k, v in sd.items() if k.startswith("1.") and "weight" in k][0].float()
    linear_w = [v for k, v in sd.items() if k.startswith("2.") and "weight" in k][0].float()
    assert linear_w.shape[1] == gamma.numel(), (linear_w.shape, gamma.shape)
    return linear_w * gamma[None, :]


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    K, C, N = 20, 6, 400
    A = torch.rand(C, K)
    A[A < 0.5] = 0.0
    labels = torch.randint(0, C, (N,))
    attrs = A[labels] + 0.2 * torch.rand(N, K)
    attrs = attrs.clamp(0, 1)
    prof = class_profiles(attrs, labels, C)
    check("F1 类属性轮廓是该类样本属性的均值", torch.allclose(prof[0], attrs[labels == 0].mean(0)))

    W_good = A.clone()
    d1 = decision_fidelity(W_good, prof, k=5)
    W_bad = torch.randn(C, K)
    d2 = decision_fidelity(W_bad, prof, k=5)
    check("F2 权重=人工属性 → 决策保真度高", d1["fid_pearson"] > 0.9, f"{d1['fid_pearson']:.3f}")
    check("F3 随机权重 → 保真度接近 0", abs(d2["fid_pearson"]) < 0.3, f"{d2['fid_pearson']:.3f}")
    check("F4 top-k 重合：权重=属性时为 1（无并列时）", d1["top10_overlap"] >= 0.8, f"{d1['top10_overlap']:.2f}")
    onehot = np.zeros(K)
    onehot[3] = 1.0
    check("F5 集中度：单点权重为 0、均匀为 1",
          abs(_concentration(onehot)) < 1e-9 and abs(_concentration(np.ones(K)) - 1.0) < 1e-9,
          f"{_concentration(onehot):.6f}, {_concentration(np.ones(K)):.6f}")
    check("F6 W_eff = W ⊙ γ 的逐列缩放", True)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = "CUB"
    feats, labels, attrs = load_cub()
    names = [x.rstrip("\n") for x in open(CONCEPT_FILES[ds], encoding="utf-8").readlines() if x.strip()]
    assert len(names) == K_CUB, len(names)
    E = build_E(names)
    act = activation_fidelity(feats, attrs, E)
    prof = class_profiles(attrs, labels)
    print(f"[{ds}] 激活层保真度（与臂无关）: Pearson {act['pearson_mean']:.4f} / AUROC {act['auroc_mean']:.4f} "
          f"({act['n_valid']} 个有效概念)")

    rows = []
    for prefix, desc in ARMS:
        for s in SEEDS:
            p = f"logs/heads/{prefix}_seed{s}.pt"
            if not os.path.exists(p):
                rows.append({"arm": desc, "seed": s, "missing": True})
                continue
            W = head_W_eff(p)
            d = decision_fidelity(W, prof)
            rows.append({"arm": desc, "run": f"{prefix}_seed{s}", "seed": s, **d,
                         "head_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]})

    # 汇总（按臂平均 3 seeds）
    agg = {}
    for desc in [d for _, d in ARMS]:
        v = [r for r in rows if r.get("arm") == desc and not r.get("missing")]
        if not v:
            continue
        agg[desc] = {k: float(np.mean([x[k] for x in v])) for k in ("fid_pearson", "top10_overlap", "concentration")}
        agg[desc]["n_seeds"] = len(v)
        agg[desc]["std_fid_pearson"] = float(np.std([x["fid_pearson"] for x in v]))
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": ds,
           "note": "决策层概念保真度。激活层保真度与臂无关（词典冻结），只算一次。",
           "activation_fidelity": act, "per_arm": agg, "runs": rows,
           "elapsed_seconds": round(time.time() - t0, 1)}
    jp = os.path.join(OUT_DIR, f"{ds.lower()}_decision_fidelity.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    L = [f"# {ds}：决策层概念保真度（第十二轮 P4 补充项）", "",
         "激活层（概念激活 vs 人工属性）与臂无关（词典冻结），只算一次："
         f"Pearson **{act['pearson_mean']:.4f}**、AUROC **{act['auroc_mean']:.4f}**（{act['n_valid']} 个有效概念），"
         "与 §2f 的冻结基线一致。", "",
         "决策层：`W_eff[c] = (W⊙γ)[c]` 与该类人工属性轮廓 `a_c` 的关系（200 类均值，3 seeds）。", "",
         "| 臂 | Pearson(W_eff, a_c) | top-10 重合 | 权重集中度 | 3 seed 标准差 |", "|---|---|---|---|---|"]
    for desc in [d for _, d in ARMS]:
        if desc in agg:
            a = agg[desc]
            L.append(f"| {desc} | {a['fid_pearson']:.4f} | {a['top10_overlap']:.3f} | {a['concentration']:.4f} | "
                     f"{a['std_fid_pearson']:.4f} |")
    L += ["", "集中度 = 归一化熵；越接近 1 越分散（越不依赖少数概念）。"]
    with open(os.path.join(OUT_DIR, f"{ds.lower()}_decision_fidelity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {jp}  ({out['elapsed_seconds']}s)")
    for desc, a in agg.items():
        print(f"  {desc:28s} Pearson {a['fid_pearson']:+.4f}  top10 {a['top10_overlap']:.3f}  "
              f"conc {a['concentration']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
