"""Stage 0C 的 L1 梯度比补充（外审第十一轮 P0 第 4 项；§3af 是先写的描述性补充计划）。
零训练：3 个 P1 初始化头 × train_core 前 256 固定批次，测量三种描述符加权几何
（等权 / 可靠性加权 / 簇级等权，与 §3ac 的 G_uniform/G_RW/G_group 同一套权重构造）的
L1 正则（|h(方向集)| 的加权均值）对分类头的初始梯度比与夹角，两库（Original-204 / SDB-204）对照。
回答 Stage 0C 的原始问题：标量加权在低有效秩方向分布上的梯度是否"触顶"。

第十二轮 P0-B 的修正（2026-11-27）：
  1. 直接比较加权梯度与等权梯度 —— 夹角余弦 cos(g_w, g_uniform) 与相对差 ‖g_w−g_uniform‖/‖g_uniform‖。
     此前只报"范数比"与"对 CE 梯度的夹角"，两者相近不能推出梯度相同。
  2. 权重均值检查改真实张量断言（此前直接传入 True）。
  3. §3af 与脚本、结果同时提交，属于提前计划的描述性补充实验，不是完整预注册验证；原有数字全部保留，
     新增字段以 `cos_vs_uniform_*` / `rel_diff_vs_uniform_*` 命名。

用法: python stage0c_l1_gradients.py [--selftest]
产物: experiment_results/DescriptorBank/stage0c_l1_gradients.json
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

from descriptor_bank import load_descriptor_bank  # noqa: E402
from reliability import attach_domain_reliability, compute_descriptor_clusters  # noqa: E402
from run_status import git_info  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "DescriptorBank")
NEW_JSON = os.path.join(OUT_DIR, "stage0c_l1_gradients.json")
SDB_PATH = os.path.join("prompts", "banks", "sdb-204-v1.json")
CLIP_PATH = os.path.abspath("clip_weights/ViT-L-14.pt")
CLUSTER_T = 0.9
SEEDS = (1, 2, 3)
GEOMS = ("uniform", "rw", "group")


def _sha(t):
    return hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()


def geom_weights(rel, assignment):
    """与 §3ac 的 G_RW / G_group 同一套权重 (均为均值 1, 与等权同尺度):
    rw: softmax(r)·N; group: 簇级等权 (每簇总权重 1/G), 簇内可靠性份额 q_{i|g} = softmax(r)|g; 再整体归一到均值 1。"""
    N = rel.numel()
    w_rw = torch.softmax(rel.double(), dim=0) * N
    G_total = int(assignment.max().item()) + 1
    w_group = torch.zeros(N, dtype=torch.float64)
    for g in range(G_total):
        members = torch.nonzero(assignment == g, as_tuple=True)[0]
        lg = rel[members].double()
        qg = torch.exp(lg - lg.max())
        qg = qg / qg.sum()
        w_group[members] = qg / G_total
    w_group = w_group / w_group.mean()
    return {"rw": w_rw, "group": w_group}


def weighted_l1(responses, w):
    """responses: (N, ...) 的 |h|; w: (N,) 均值 1 的权重 → 加权均值 (与等权 mean 同尺度)。"""
    N = responses.size(0)
    return (w.to(responses.device).reshape(-1, *([1] * (responses.dim() - 1))) * responses.abs()).sum(0) / (N * float(w.mean()))


def flat_grad(fn, params):
    """把标量 fn() 对 params 的梯度拼成一个向量 (None 的位置补零, 与 reg_ratio 的做法一致)。
    用于直接比较两种加权正则的梯度方向与幅值 (外审第十二轮 P0-B 第 1、2 项)。"""
    raw = torch.autograd.grad(fn(), params, retain_graph=True, allow_unused=True)
    return torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(raw, params)])


def cos_and_rel(a, b):
    """两个梯度向量的夹角余弦与相对差 ‖a−b‖/‖b‖。"""
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    rel = float((a - b).norm() / b.norm().clamp_min(1e-12))
    return float((a @ b) / denom), rel


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    rel = torch.rand(12)
    assign = torch.tensor([0] * 10 + [1, 1])
    w = geom_weights(rel, assign)
    check("W1 rw 权重均值 1", abs(float(w["rw"].mean()) - 1.0) < 1e-12)
    check("W2 group 权重均值 1 (归一后)", abs(float(w["group"].mean()) - 1.0) < 1e-12)
    tot0 = float(w["group"][:10].sum())
    tot1 = float(w["group"][10:].sum())
    check("W3 group: 每簇的总权重相等 (簇级等权的定义; 簇大小可以不同)",
          abs(tot0 - tot1) < 1e-9, f"簇0 总 {tot0:.4f}, 簇1 总 {tot1:.4f}")
    resp = torch.rand(12, 5, 3)
    w1 = torch.ones(12)
    diff = weighted_l1(resp, w1) - resp.abs().mean(dim=0)
    check("W4 等权权重下 weighted_l1 == 逐类 mean(|r|) (逐元素相等)",
          float(diff.abs().max()) < 1e-12)
    # 第十二轮 P0-B 新增助手的最小自测
    a, b, o = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0]), torch.tensor([2.0, -1.0, 0.0])
    c1, r1 = cos_and_rel(a, b)
    c2, _ = cos_and_rel(a, o)
    check("G1 cos_and_rel: 同向缩放 cos=1、相对差=0.5 (‖a−2a‖/‖2a‖)",
          abs(c1 - 1.0) < 1e-6 and abs(r1 - 0.5) < 1e-6, f"cos={c1:.6f} rel={r1:.6f}")
    check("G2 cos_and_rel: 正交向量 cos=0", abs(c2) < 1e-6, f"cos={c2:.2e}")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    check("G3 flat_grad == 手算梯度 (f = Σx², grad = 2x)",
          torch.allclose(flat_grad(lambda: (x ** 2).sum(), [x]), 2 * x.detach()))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [detail]" if detail else ""), flush=True)

    from stage1c_radius_diagnose import build_session, SEEDS

    # ===== 1. 三个 P1 初始化会话 (Original 词库的 dataset 提供 diffs; 分类头初始状态与固定批次同前) =====
    init_sess = {}
    for s in SEEDS:
        with open(f"logs/results/CUB_p1_dose_a1_seed{s}.json", encoding="utf-8") as f:
            r = json.load(f)
        init_sess[s] = build_session(r["run_config"], s)
    E = init_sess[SEEDS[0]].model.concept_embeddings.detach().double().cpu()
    e_sha = _sha(E)
    with open(os.path.join("experiment_results", "Stage1C-Radius", "stage1c_radius_cub.json"), encoding="utf-8") as f:
        e_ref = json.load(f)["E_sha256"]
    check("E 与 §3y 记录逐位一致", e_sha == e_ref, f"{e_sha[:16]}")

    # ===== 2. 两库的 diffs (Original 来自会话 dataset; SDB 经 attach 同一路径/缓存) =====
    banks = {"original": {"diffs": init_sess[SEEDS[0]].train_dataset.domain_diffs.double().cpu()}}
    b = load_descriptor_bank(SDB_PATH)
    holder = type("H", (), {"concept2id": init_sess[SEEDS[0]].train_dataset.concept2id})()
    import clip
    clip_model, _ = clip.load(CLIP_PATH, device=device)
    attach_domain_reliability(holder, clip_model, b["source_text_prompts"], b["target_text_prompts"],
                              list(init_sess[SEEDS[0]].train_dataset.classname2id.keys()), device,
                              weight_strategy="prior_residual", verbose=True, clip_path=CLIP_PATH)
    banks["sdb204"] = {"diffs": holder.domain_diffs.double().cpu(), "rel": holder.reliability_scores.double().cpu()}
    banks["original"]["rel"] = init_sess[SEEDS[0]].train_dataset.reliability_scores.double().cpu()

    # ===== 3. 每库 × 每几何: 加权 L1 的初始梯度比、与 CE 的夹角, 以及与等权梯度的直接余弦/相对差 =====
    # 第十二轮 P0-B: 梯度范数与对 CE 的夹角相近, 不等于两种加权的梯度相同 —— 直接比较 g_w 与 g_uniform。
    res = {}
    for bk in banks:
        d = banks[bk]["diffs"]
        rel = banks[bk]["rel"]
        assign, ng = compute_descriptor_clusters(d, threshold=CLUSTER_T)
        w_all = geom_weights(rel, assign)
        P = torch.as_tensor(domain_directions_for(d, E).reshape(d.size(0), -1, E.size(0)), dtype=torch.float32)
        W = {"uniform": torch.ones(d.size(0), dtype=torch.float64)}
        W.update(w_all)
        per_seed = {s: {} for s in SEEDS}
        for s in SEEDS:
            sess = init_sess[s]
            x, y = sess._diag_batch()
            sess.model.eval()
            params = [pp for pp in sess.model.classifier.parameters() if pp.requires_grad]
            _, lg, _ = sess.model(x)
            g_ce = flat_grad(lambda: F.cross_entropy(lg, y), params)
            for g in GEOMS:
                w = W[g]

                def reg_fn(w=w):
                    rr = sess.model.classifier[1:](P.to(sess.model.device))
                    return weighted_l1(rr.abs(), w).mean()
                g_w = flat_grad(reg_fn, params)
                per_seed[s][g] = {"g": g_w, "ratio": float(g_w.norm() / g_ce.norm().clamp_min(1e-12)),
                                  "cos_ce": cos_and_rel(g_w, g_ce)[0]}
        geo = {}
        for g in GEOMS:
            ratio = [per_seed[s][g]["ratio"] for s in SEEDS]
            cos_ce = [per_seed[s][g]["cos_ce"] for s in SEEDS]
            cos_u, rel_u = [], []
            for s in SEEDS:
                cu, ru = cos_and_rel(per_seed[s][g]["g"], per_seed[s]["uniform"]["g"])
                cos_u.append(cu)
                rel_u.append(ru)
            w = W[g]
            geo[g] = {"ratio_mean": float(np.mean(ratio)), "ratio_per_seed": ratio,
                      "cos_mean": float(np.mean(cos_ce)), "cos_per_seed": cos_ce,
                      "cos_vs_uniform_mean": float(np.mean(cos_u)), "cos_vs_uniform_per_seed": cos_u,
                      "rel_diff_vs_uniform_mean": float(np.mean(rel_u)), "rel_diff_vs_uniform_per_seed": rel_u,
                      "w_mean": float(w.mean()), "w_min": float(w.min()), "w_max": float(w.max()),
                      "n_clusters": int(ng), "w_effective_rank": float(
                          torch.exp(-(w / w.sum() * (w / w.sum() + 1e-300).log()).sum()))}
        res[bk] = {"geoms": geo, "diffs_sha256": hashlib.sha256(d.numpy().tobytes()).hexdigest()[:16]}
        print(f"  [{bk}] " + "；".join(f"{g} 比值 {geo[g]['ratio_mean']:.4f}" for g in GEOMS), flush=True)
        print(f"  [{bk}] 与等权梯度: " + "；".join(
            f"{g} cos {geo[g]['cos_vs_uniform_mean']:.6f} / 相对差 {geo[g]['rel_diff_vs_uniform_mean']:.4f}"
            for g in GEOMS), flush=True)

    check("等权与 rw/group 的权重均值都是 1 (同尺度可比; 真实张量断言, 第十二轮 P0-B 修正)",
          all(abs(res[bk]["geoms"][g]["w_mean"] - 1.0) < 1e-9 for bk in res for g in GEOMS),
          "；".join(f"{bk}: " + "/".join(f"{g} {res[bk]['geoms'][g]['w_mean']:.12f}" for g in GEOMS) for bk in res))
    check("等权自比较的余弦为 1、相对差为 0 (对照的定义域自检)",
          all(abs(res[bk]["geoms"]["uniform"]["cos_vs_uniform_mean"] - 1.0) < 1e-9
              and res[bk]["geoms"]["uniform"]["rel_diff_vs_uniform_mean"] < 1e-6 for bk in res))
    prev_sha = None
    if os.path.exists(NEW_JSON):                                  # 记录本版替换掉的上一版内容摘要 (审计用)
        with open(NEW_JSON, "rb") as f:
            prev_sha = hashlib.sha256(f.read()).hexdigest()
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": str(device),
           "E_sha256": e_sha,
           "note": "Stage 0C 的 L1 梯度比补充 (外审第十一轮 P0 第 4 项; §3af 为提前写下的描述性计划, "
                   "本脚本与结果同提交, 不构成完整预注册验证)。三种几何的权重构造与 §3ac 的 G_RW/G_group 同族; "
                   "第十二轮 P0-B 新增 cos_vs_uniform_* 与 rel_diff_vs_uniform_* (直接比较加权梯度与等权梯度), "
                   "并把权重均值检查改为真实张量断言。全部描述性。",
           "supersedes_sha256": prev_sha,
           "banks": res, "all_checks_passed": ok, "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {NEW_JSON}  (耗时 {out['elapsed_seconds']}s)")
    return 0 if ok else 1


def domain_directions_for(d, E):
    from language_ellipsoid import domain_directions
    return domain_directions(d, E)


if __name__ == "__main__":
    sys.exit(main())
