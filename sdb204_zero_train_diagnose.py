"""SDB-204 v1.0 的零训练诊断（预注册见 RESULTS.md §3ab —— 定义与边界写在那里，本脚本只执行）。

两份词库对照：Original-204（训练在用）与 SDB-204 v1.0（冻结候选，未经 CLIP 语义验证）。全部描述性：
只用文本与源域数据（视觉统计只在 train_core），不使用目标域，不产生任何选择规则，SDB 不进入本轮训练。
  D1 类别语义保持 margin   （文本侧；margin<0 的"身份翻转"比例；8 条两库相同文本逐位一致）
  D2 跨类别一致性（可靠性）  （reliability.py 的 r_i；两库都在同一 CLIP 实例同一调用模式下现算，逐位可比）
  D3 近重复与冗余           （类平均方向 |cos|>0.9 的簇；最近邻分布；相对冗余度 c̃）
  D4 三种几何（并入 Stage 0C）（G_uniform / G_RW / G_group 的谱、有效秩、eps_dom、主轴夹角）
  D5 视觉扰动投影覆盖       （train_core 的 7 种光度扰动 Δu 在各库 Language-Full 椭球里的白化半径与值域外能量）
  D6 注入稳定性             （204 个描述符逐一精确复制 / 近重复注入后的首轴余弦、有效秩熵变化）

编码口径（2026-11-26 实测）：CLIP 文本编码的比特结果依赖 GPU 分配与调用历史（init 编码 / 同实例重编码 /
全新实例编码是三个稳定但不同的结果）。E 必须经 clip_cbm_orth 的同一构造取得（与训练/§3y 逐位一致）；
两库的 diffs/可靠性都在同一实例上现算以保证互相逐位可比，并与 Original 的现役缓存对比量化口径差。

用法: python sdb204_zero_train_diagnose.py [--selftest] [--skip_d6]
产物: experiment_results/DescriptorBank/sdb204_zero_train.{json,md} (md 报告在 §3ac 由脚本生成)
"""
import argparse
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

from descriptor_bank import load_descriptor_bank  # noqa: E402
from language_ellipsoid import build_ellipsoids, domain_directions, ln0, second_moments  # noqa: E402
from reliability import compute_descriptor_clusters, compute_descriptor_redundancy  # noqa: E402
from run_status import git_info  # noqa: E402
from stage1c_radius_diagnose import whiten_radius_sq  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "DescriptorBank")
SDB_PATH = os.path.join("prompts", "banks", "sdb-204-v1.json")
AUG_CACHE = "cache/CUB_train_aug.pt"
CLUSTER_T = 0.9                       # E3 / C-1 的既有阈值
INJECT_JITTER = 0.05                  # 近重复注入的方向抖动 (§3ab)
INJECT_SEED = 20261126
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]
SDB_CATS = ["绘画与艺术媒介", "线条与轮廓", "数字渲染与图形表现", "材质与纹理表现",
            "色彩与饱和度", "光照与对比度", "成像质量", "背景与环境"]


def q(v, p):
    return float(torch.quantile(v.double().reshape(-1), p))


def stat(v):
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    return {"n": int(v.numel()), "mean": float(v.mean()), "q10": q(v, 0.10), "median": q(v, 0.50),
            "q90": q(v, 0.90), "min": float(v.min()), "max": float(v.max())}


def identity_margins(t_emb, cls_emb):
    """margin = 与自己类参考向量的余弦 − 与最强竞争类参考向量的余弦。
    t_emb: (N·C, D)，按 N 个描述符 × C 个类的顺序；cls_emb: (C, D)。返回 (N·C,)。
    逐描述符分块计算（避免 (N·C, C) 的大矩阵），每块内对角线是自己的类。"""
    t = F.normalize(t_emb.float(), dim=-1)
    c = F.normalize(cls_emb.float(), dim=-1)
    C = c.size(0)
    assert t.size(0) % C == 0
    out = torch.empty(t.size(0), dtype=torch.float64)
    for i in range(t.size(0) // C):
        s = t[i * C:(i + 1) * C] @ c.T                              # (C, C): 行 = 描述符的类, 列 = 参考类
        own = s.diagonal().double()
        s2 = s.clone()
        s2.fill_diagonal_(float("-inf"))
        out[i * C:(i + 1) * C] = own - s2.max(dim=1).values.double()
    return out


def weighted_second_moments(U, w):
    """G_RW = Σ_i (p_i/N)·G_i, G_i = U_iᵀU_i/C, p = w/mean(w)（§3ab；权重均值 1 → 与等权 G 同尺度）。
    U: (N, C, K) 的域方向 (LN0 后); w: (N,)。"""
    N, C, K = U.shape
    ww = w.double() / w.double().mean()
    G = torch.zeros(K, K, dtype=torch.double)
    for i in range(N):
        G += (ww[i] / N) * (U[i].T @ U[i]) / C
    return G


def group_second_moments(U, w, assignment):
    """簇级等权几何: G = Σ_g (1/G_total) Σ_{i∈g} q_{i|g}·G_i, q_{i|g} = e^{r_i}/Σ_{j∈g} e^{r_j} (§3ab)。
    全部单元素簇时严格等于等权 G_uniform = UᵀU/(N·C)（自测 T3 钉住）。"""
    N, C, K = U.shape
    G_total = int(assignment.max().item()) + 1
    G = torch.zeros(K, K, dtype=torch.double)
    for g in range(G_total):
        members = torch.nonzero(assignment == g, as_tuple=True)[0]
        lg = w[members].double()
        qg = torch.exp(lg - lg.max())
        qg = qg / qg.sum()
        for k, i in enumerate(members.tolist()):
            G += (qg[k] / G_total) * (U[i].T @ U[i]) / C
    return G


def spectrum(G):
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    tot = float(ev.sum())
    nz = ev > float(ev[0]) * 1e-10
    return {"tr": tot, "lambda_max": float(ev[0]),
            "lambda_min_nonzero": (float(ev[nz][-1]) if bool(nz.any()) else None),
            "lambda_max_over_min_nonzero": (float(ev[0] / ev[nz][-1]) if bool(nz.any()) else None),
            "eff_rank_entropy_all": float(torch.exp(-(ev / tot * (ev / tot + 1e-300).log()).sum())),
            "energy_at_32": float(ev[:32].sum() / tot),
            "eff_rank_entropy_trunc32": float(torch.exp(
                -(ev[:32] / ev[:32].sum() * (ev[:32] / ev[:32].sum() + 1e-300).log()).sum()))}


def top_axes(G, k=32):
    _, vec = torch.linalg.eigh(G.double())
    return vec.flip(1)[:, :k]                                          # (K, k) 降序


def subspace_angles(V1, V2):
    """两个 (K, k) 正交标架的逐对主轴余弦（贪心配对的近似；只作描述）。"""
    M = (V1.T @ V2).abs()
    out, W = [], M.clone()
    for _ in range(min(V1.size(1), V2.size(1))):
        i, j = divmod(int(W.argmax()), W.size(1))
        out.append(float(M[i, j]))
        W[i, :] = -1
        W[:, j] = -1
    return out


def uniform_G(U_flat, N, C):
    """等权二阶矩: UᵀU/(N·C)（U_flat: (N·C, K) 展平的 LN0 域方向）。"""
    return U_flat.T @ U_flat / (N * C)


def inject_stats(G, U, kind, seed):
    """逐一注入（kind='exact': 复制该描述符的全部类方向; kind='near': u + 0.05·noise 再归一化）后重建 G：
    G2 = (N·G + U_iᵀU_i/C) / (N+1) —— 与"词库多一条描述符后重新取平均"一致（迹保持，比较才公平）。
    记录首轴余弦（注入前后）、有效秩熵变化、λmax 能量占比变化。N 次注入。"""
    N, C, K = U.shape
    g0 = torch.Generator().manual_seed(int(seed))
    base_sp, base_ax = spectrum(G), top_axes(G, 1)
    s0, lm0 = base_sp["eff_rank_entropy_all"], base_sp["lambda_max"] / base_sp["tr"]
    out = {"first_axis_cos": [], "eff_rank_drop": [], "lmax_share_gain": []}
    for i in range(N):
        if kind == "exact":
            row = U[i].clone()
        else:
            noise = torch.randn(U[i].shape, generator=g0, dtype=U.dtype)
            row = U[i] + float(INJECT_JITTER) * noise
            row = row / row.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        block = row.reshape(-1, K).T @ row.reshape(-1, K) / C
        G2 = (N * G + block) / (N + 1)
        sp, ax = spectrum(G2), top_axes(G2, 1)
        out["first_axis_cos"].append(float(abs(base_ax[:, 0] @ ax[:, 0])))
        out["eff_rank_drop"].append(s0 - sp["eff_rank_entropy_all"])
        out["lmax_share_gain"].append(sp["lambda_max"] / sp["tr"] - lm0)
    return {k: stat(v) for k, v in out.items()}


# ---------------- 自测 (合成数据) ----------------
def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    N, C, K = 12, 8, 30
    U = ln0(torch.randn(N * C, K, dtype=torch.float64)).reshape(N * C, K)
    U3 = U.reshape(N, C, K)
    G_u = uniform_G(U, N, C)
    check("T1 等权二阶矩 = UᵀU/(N·C)", float((G_u - U.T @ U / (N * C)).abs().max()) < 1e-14)
    w = torch.softmax(torch.randn(N) * 3, dim=0) * N
    G_rw = weighted_second_moments(U3, w)
    check("T2 RW 权重均值 1 → 与等权同尺度 (等权输入时严格相等; 随机权重时 tr 同量级)",
          abs(float(weighted_second_moments(U3, torch.ones(N)).trace() - G_u.trace())) < 1e-12
          and 0.2 < float(G_rw.trace() / G_u.trace()) < 5.0, f"{float(G_rw.trace() / G_u.trace()):.3f}")
    G_g1 = group_second_moments(U3, torch.ones(N), torch.arange(N))
    check("T3 全部单元素簇时 G_group ≡ G_uniform (预注册的等式)", float((G_g1 - G_u).abs().max()) < 1e-12)
    assign_pair = torch.tensor([0] * (N - 2) + [1, 1])
    G_g2 = group_second_moments(U3, torch.ones(N), assign_pair)
    check("T4 双元素簇的簇级等权 ≠ 等权几何 (去冗余生效)", float((G_g2 - G_u).abs().max()) > 1e-6)
    cls = F.normalize(torch.eye(4) * 3 + 0.1, dim=-1)
    t = torch.tensor([[3.0, 0.0, 0.0, 0.1], [0.0, 0.1, 3.0, 0.0], [0.0, 3.0, 0.0, 0.1], [0.1, 0.0, 0.0, 3.0]])
    m = identity_margins(t, cls)
    check("T5 margin: 行 k 的自身类是第 k 列 —— 样本 0/3 清晰 (margin>0.9), 样本 1/2 是身份翻转 (margin<0)",
          m[0] > 0.9 and m[1] < 0 and m[2] < 0 and m[3] > 0.9, f"{[round(float(x), 3) for x in m]}")
    sp = spectrum(G_u)
    check("T6 谱摘要: tr = sum(λ), 有效秩熵 ∈ [1, K]",
          abs(sp["tr"] - float(G_u.trace())) < 1e-12 and 1.0 <= sp["eff_rank_entropy_all"] <= K)
    V = top_axes(G_u, 8)
    check("T7 主轴标架正交; 同标架主轴余弦全为 1",
          float((V.T @ V - torch.eye(8, dtype=torch.float64)).abs().max()) < 1e-10 and min(subspace_angles(V, V)) > 1 - 1e-9)
    inj = inject_stats(G_u, U3, "exact", 1)
    inj_n = inject_stats(G_u, U3, "near", 1)
    check("T8 注入统计: N 项都记录, 首轴余弦 ≤ 1 (浮点容差), 有效秩熵不增 (精确注入使谱更集中; 近注入允许略增)",
          inj["first_axis_cos"]["n"] == N and inj["first_axis_cos"]["max"] <= 1.0 + 1e-9
          and inj["eff_rank_drop"]["min"] >= -1e-9 and inj_n["first_axis_cos"]["min"] <= 1.0 + 1e-9)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------- 真实数据 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip_d6", action="store_true", help="跳过注入稳定性 (408 次重建 G; 调试用)")
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    import clip
    from args import get_args
    from reliability import compute_domain_diffs_and_scores

    # ===== 0. 类别表 / 概念矩阵 / CLIP =====
    # E 必须来自与训练完全相同的编码路径: CLIP 文本编码的结果依赖 GPU 分配与调用历史 (同进程里 init 编码 /
    # 同实例重编码 / 全新实例编码是三个稳定但不同的口径, 2026-11-26 实测), 所以这里用 clip_cbm_orth 的同一
    # 构造 (抛一次性模型取 concept_embeddings) —— 它在训练、Stage 1C 与本脚本里都稳定复现 §3y 记录的 E_sha256。
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1", "--epochs", "1",
                "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt")]
    args = get_args()
    args.save_model = False
    args.device = str(device)
    ds = __import__("data").get_dataset_classes(args)[0]
    class_names = list(ds.classname2id.keys())
    concept_names = list(ds.concept2id.keys())
    with open(os.path.join("experiment_results", "Stage1C-Radius", "stage1c_radius_cub.json"), encoding="utf-8") as f:
        e_ref = json.load(f)["E_sha256"]
    from model.cbm_models import clip_cbm_orth
    model = clip_cbm_orth(args=args, class_names=class_names, concept_names=concept_names,
                          domain_diffs=ds.domain_diffs, weight_mode="none")
    clip_model = model.clip_model
    E = model.concept_embeddings.detach().double().cpu()
    got_e = hashlib.sha256(E.numpy().tobytes()).hexdigest()
    check("概念矩阵 E 与 §3y 记录逐位一致 (经 clip_cbm_orth 同一编码路径)", got_e == e_ref,
          f"{got_e[:16]} vs {e_ref[:16]}")
    with torch.no_grad():
        cls_emb = clip_model.encode_text(
            clip.tokenize([f"a photo of a {c}." for c in class_names]).to(device)).float().cpu()

    # ===== 1. 两库的域方向与可靠性 —— 两库都在同一实例、同一调用模式下现算 (逐位可比), 另与 Original 的
    # 现役缓存对比量化编码口径差 (不写入/不污染任何缓存) =====
    banks = {}
    for name, spec in (("original", "original"), ("sdb204", SDB_PATH)):
        b = load_descriptor_bank(spec)
        diffs, rel = compute_domain_diffs_and_scores(
            clip_model, b["source_text_prompts"], b["target_text_prompts"], class_names, device, verbose=False)
        banks[name] = {"bank": b, "diffs": diffs.double().cpu(), "rel": rel.double().cpu()}
        print(f"  [{name}] N={b['n_target']}, diffs {tuple(banks[name]['diffs'].shape)}, "
              f"rel 中位 {float(banks[name]['rel'].median()):.4f}", flush=True)
    cached = torch.load("cache/reliability/reliability_e3abc74fcd065d5f.pt", map_location="cpu", weights_only=False)
    reg = float((cached["scores"].double() - banks["original"]["rel"]).abs().max())
    check("现算的 Original 可靠性与现役缓存同量级 (编码口径差的量化, 见 regime_note)",
          0.0 <= reg < 0.05, f"最大差 {reg:.2e}")
    regime_note = ("CLIP 文本编码的比特结果依赖 GPU 分配/调用历史; 本脚本两库都在同一实例同一调用模式下现算, "
                   f"逐位可比; 现算 Original 与现役缓存的可靠性最大差 {reg:.2e} (编码口径差, 非同一口径的逐位比较)。")
    same_text = sorted(set(banks["original"]["bank"]["target_text_prompts"])
                       & set(banks["sdb204"]["bank"]["target_text_prompts"]))
    check("两库逐条相同的文本数与 GPT 报告一致 (8 条)", len(same_text) == 8, f"{len(same_text)} 条")

    # ===== 2. D1 类别语义保持 margin =====
    for name in banks:
        b = banks[name]
        N = b["bank"]["n_target"]
        Cn = len(class_names)
        with torch.no_grad():
            texts = [p.format(c) for p in b["bank"]["target_text_prompts"] for c in class_names]
            t_emb = []
            for i in range(0, len(texts), 1024):
                t_emb.append(clip_model.encode_text(clip.tokenize(texts[i:i + 1024]).to(device)).float().cpu())
            t_emb = torch.cat(t_emb)
        m = identity_margins(t_emb, cls_emb).reshape(N, Cn)
        b["margin"] = m
        b["margin_stat"] = stat(m.reshape(-1))
        b["margin_flip_rate"] = float((m < 0).double().mean())
        if name == "sdb204":
            cats = b["bank"]["meta"].get("descriptor_categories")
            cat_t = torch.tensor([SDB_CATS.index(x) for x in cats])
            b["margin_by_cat"] = {c: stat(m[cat_t == k].reshape(-1)) for k, c in enumerate(SDB_CATS)}
            b["flip_by_cat"] = {c: float((m[cat_t == k] < 0).double().mean()) for k, c in enumerate(SDB_CATS)}
        print(f"  [D1 {name}] margin 中位 {b['margin_stat']['median']:.4f}, "
              f"身份翻转 {100 * b['margin_flip_rate']:.2f}%", flush=True)
    oi = {p: i for i, p in enumerate(banks["original"]["bank"]["target_text_prompts"])}
    si = {p: i for i, p in enumerate(banks["sdb204"]["bank"]["target_text_prompts"])}
    dmax_m, dmax_r = 0.0, 0.0
    for p in same_text:
        dmax_m = max(dmax_m, float((banks["original"]["margin"][oi[p]] - banks["sdb204"]["margin"][si[p]]).abs().max()))
        dmax_r = max(dmax_r, abs(float(banks["original"]["rel"][oi[p]]) - float(banks["sdb204"]["rel"][si[p]])))
    check("8 条相同文本的 margin 与可靠性逐位一致 (管线确定性)", dmax_m < 1e-6 and dmax_r < 1e-9,
          f"margin 最大差 {dmax_m:.2e}, rel 最大差 {dmax_r:.2e}")

    # ===== 3. D2/D3: 可靠性分布与冗余聚类 =====
    for name in banks:
        b = banks[name]
        b["rel_stat"] = stat(b["rel"])
        if name == "sdb204":
            cats = b["bank"]["meta"].get("descriptor_categories")
            cat_t = torch.tensor([SDB_CATS.index(x) for x in cats])
            b["rel_by_cat"] = {c: float(b["rel"][cat_t == k].mean()) for k, c in enumerate(SDB_CATS)}
        d = b["diffs"]
        dirs = F.normalize(d.mean(dim=1), dim=-1)
        S = (dirs @ dirs.T).abs()
        S.fill_diagonal_(0)
        b["nn_cos"] = stat(S.max(dim=1).values)
        b["redundancy_stat"] = stat(compute_descriptor_redundancy(d))
        assign, ng = compute_descriptor_clusters(d, threshold=CLUSTER_T)
        sizes = torch.bincount(assign, minlength=ng)
        b["clusters"] = {"threshold": CLUSTER_T, "n_clusters": int(ng), "multi_member": int((sizes > 1).sum()),
                         "largest": int(sizes.max()),
                         "size_hist": {str(k): int(v) for k, v in enumerate(torch.bincount(sizes)) if v}}
        print(f"  [D3 {name}] 簇 {ng}, 多成员簇 {int((sizes > 1).sum())}, 最大簇 {int(sizes.max())}, "
              f"最近邻 cos 中位 {b['nn_cos']['median']:.4f}", flush=True)

    # ===== 4. D4: 三种几何 (Stage 0C 并入) =====
    geo = {}
    for name in banks:
        b = banks[name]
        N = b["bank"]["n_target"]
        U = domain_directions(b["diffs"], E)                        # (N·C, K) 已 LN0
        U3 = U.reshape(N, b["diffs"].size(1), U.size(1))
        assign, _ = compute_descriptor_clusters(b["diffs"], threshold=CLUSTER_T)
        Gs = {"uniform": uniform_G(U, N, b["diffs"].size(1)),
              "rw": weighted_second_moments(U3, b["rel"]),
              "group": group_second_moments(U3, b["rel"], assign)}
        Kd = U.size(1)
        geo[name] = {}
        for gname, G in Gs.items():
            sp = spectrum(G)
            geo[name][gname] = {"spectrum": sp,
                                "eps_dom_full": float(sp["tr"] * (Kd - 1)) ** 0.5,
                                "eta_mean": float(second_moments(U)[3]) if gname == "uniform" else None}
        geo[name]["axes"] = {
            "first_axis_cos_uniform_vs_rw": float(abs(top_axes(Gs["uniform"], 1)[:, 0] @ top_axes(Gs["rw"], 1)[:, 0])),
            "uniform_vs_rw_top32_min_cos": min(subspace_angles(top_axes(Gs["uniform"], 32), top_axes(Gs["rw"], 32))),
            "uniform_vs_group_top32_min_cos": min(subspace_angles(top_axes(Gs["uniform"], 32), top_axes(Gs["group"], 32))),
            "rw_vs_group_top32_min_cos": min(subspace_angles(top_axes(Gs["rw"], 32), top_axes(Gs["group"], 32)))}
        geo[name]["_G_uniform"] = Gs["uniform"]
        geo[name]["_U_shape"] = list(U.shape)
        print(f"  [D4 {name}] eff_rank uniform {geo[name]['uniform']['spectrum']['eff_rank_entropy_all']:.2f} / "
              f"rw {geo[name]['rw']['spectrum']['eff_rank_entropy_all']:.2f} / "
              f"group {geo[name]['group']['spectrum']['eff_rank_entropy_all']:.2f}; "
              f"eps_dom_full {geo[name]['uniform']['eps_dom_full']:.2f}", flush=True)
    Vu = top_axes(geo["original"]["_G_uniform"], 32)
    Vs = top_axes(geo["sdb204"]["_G_uniform"], 32)
    cross = subspace_angles(Vu, Vs)
    geo["cross_bank"] = {"first_axis_cos": cross[0], "top32_min_pair_cos": min(cross),
                         "top32_median_pair_cos": float(np.median(cross)),
                         "uniform_tr_ratio_sdb_over_orig": geo["sdb204"]["uniform"]["spectrum"]["tr"]
                                                          / geo["original"]["uniform"]["spectrum"]["tr"],
                         "eff_rank_ratio": geo["sdb204"]["uniform"]["spectrum"]["eff_rank_entropy_all"]
                                           / geo["original"]["uniform"]["spectrum"]["eff_rank_entropy_all"]}
    print(f"  [D4 跨库] 首轴 cos {cross[0]:.4f}, top32 最小配对 cos {min(cross):.4f}", flush=True)

    # ===== 5. D5: 视觉扰动投影覆盖 (train_core) =====
    aug = torch.load(AUG_CACHE, map_location="cpu", weights_only=False)
    tr_idx = torch.as_tensor(aug["split"]["train_core"], dtype=torch.long)
    u_clean = ln0(aug["clean_feats"][tr_idx].double() @ E.T)
    DU = {n: ln0(aug["aug_feats"][n][tr_idx].double() @ E.T) - u_clean for n in AUGS}
    for name in banks:
        built = build_ellipsoids(geo[name]["_G_uniform"], r=geo[name]["_U_shape"][1] - 1, seed=0, with_flat=False)
        L = built["language"]
        cov = {}
        for n in AUGS:
            sq, resid = whiten_radius_sq(L, DU[n])
            cov[n] = {"eps_vis_median": float(torch.sqrt(q(sq, 0.5))), "eps_vis_q25": float(torch.sqrt(q(sq, 0.25))),
                      "eps_vis_q75": float(torch.sqrt(q(sq, 0.75))), "resid_mean": float(resid.mean())}
        banks[name]["vis_coverage"] = cov
        print(f"  [D5 {name}] Δu 白化半径中位 (7 扰动均值): "
              f"{float(np.mean([cov[n]['eps_vis_median'] for n in AUGS])):.2f}, 值域外能量均值 "
              f"{float(np.mean([cov[n]['resid_mean'] for n in AUGS])):.4f}", flush=True)

    # ===== 6. D6 注入稳定性 =====
    if not a.skip_d6:
        for name in banks:
            N = banks[name]["bank"]["n_target"]
            U = domain_directions(banks[name]["diffs"], E).reshape(N, banks[name]["diffs"].size(1), -1)
            banks[name]["inject_exact"] = inject_stats(geo[name]["_G_uniform"], U, "exact", INJECT_SEED)
            banks[name]["inject_near"] = inject_stats(geo[name]["_G_uniform"], U, "near", INJECT_SEED)
            print(f"  [D6 {name}] 精确注入首轴余弦最差 {banks[name]['inject_exact']['first_axis_cos']['min']:.4f}, "
                  f"有效秩熵下降中位 {banks[name]['inject_exact']['eff_rank_drop']['median']:.4f}", flush=True)

    # ===== 7. 头响应 (词库无关的参照量, s0 头 seed 1) =====
    heads_resp = {}
    for arm in ("ce_a0", "full_a1", "full_a2"):
        h = torch.load(f"logs/heads/CUB_s0_{arm}_seed1.pt", map_location="cpu", weights_only=False)
        W, lnw, lnb, linb = (h["classifier"]["2.weight"], h["classifier"]["1.weight"],
                             h["classifier"]["1.bias"], h["classifier"]["2.bias"])
        A, cc = W * lnw[None, :], W @ lnb + linb
        lg = u_clean @ A.T + cc
        per = []
        for n in AUGS:
            dl = DU[n] @ A.T
            per.append((dl - dl.gather(1, lg.argmax(1, keepdim=True))).max(1).values)
        heads_resp[arm] = stat(torch.stack(per).mean(1))
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": str(device),
           "note": "SDB-204 v1.0 零训练诊断 (预注册 §3ab); 全部描述性, 不产生选择规则, 不使用目标域; 视觉统计只在 train_core",
           "banks": {"original": {"bank_sha256": banks["original"]["bank"]["bank_sha256"]},
                     "sdb204": {"bank_sha256": banks["sdb204"]["bank"]["bank_sha256"],
                                "file_sha256": banks["sdb204"]["bank"]["meta"]["file_sha256"]}},
           "E_sha256": got_e, "n_classes": len(class_names), "same_texts": len(same_text),
           "determinism": {"margin_max_diff": dmax_m, "rel_max_diff": dmax_r}, "regime_note": regime_note,
           "d1": {k: {kk: v[kk] for kk in ("margin_stat", "margin_flip_rate", "margin_by_cat", "flip_by_cat") if kk in v}
                  for k, v in banks.items()},
           "d2": {k: {"rel_stat": v["rel_stat"], "rel_by_cat": v.get("rel_by_cat")} for k, v in banks.items()},
           "d3": {k: {kk: v[kk] for kk in ("nn_cos", "redundancy_stat", "clusters") if kk in v} for k, v in banks.items()},
           "d4": {k: {kk: vv for kk, vv in geo[k].items() if not kk.startswith("_")} for k in ("original", "sdb204")}
                 | {"cross_bank": geo["cross_bank"]},
           "d5": {k: v.get("vis_coverage") for k, v in banks.items()},
           "d6": {k: {kk: v[kk] for kk in ("inject_exact", "inject_near") if kk in v} for k, v in banks.items()},
           "head_response_reference": heads_resp,
           "head_response_note": "词库无关的参照量 (现有 s0 头对视觉 Δu 的每扰动平均响应)",
           "all_checks_passed": ok, "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "sdb204_zero_train.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {OUT_DIR}/sdb204_zero_train.json  (耗时 {res['elapsed_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
