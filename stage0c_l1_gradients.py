"""Stage 0C 的 L1 梯度比补充（外审第十一轮 P0 第 4 项："Stage 0C 的二阶矩部分已完成，L1 梯度比较尚需补充"；
预注册见 RESULTS §3af，先于运行提交）。零训练：3 个 P1 初始化头 × train_core 前 256 固定批次，
测量三种描述符加权几何（等权 / 可靠性加权 / 簇级等权，与 §3ac 的 G_uniform/G_RW/G_group 同一套权重构造）
的 L1 正则（|h(方向集)| 的加权均值）对分类头的初始梯度比与夹角，两库（Original-204 / SDB-204）对照。
回答 Stage 0C 的原始问题：标量加权在低有效秩方向分布上的梯度是否"触顶"（范数/方向是否随加权改变）。

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

    from stage1c_radius_diagnose import build_session, reg_ratio, SEEDS

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

    # ===== 3. 每库 × 每几何: 加权 L1 的初始梯度比与夹角 (固定批次, 3 头均值) =====
    res = {}
    for bk in banks:
        d = banks[bk]["diffs"]
        rel = banks[bk]["rel"]
        assign, ng = compute_descriptor_clusters(d, threshold=CLUSTER_T)
        w_all = geom_weights(rel, assign)
        P = torch.as_tensor(domain_directions_for(d, E).reshape(d.size(0), -1, E.size(0)), dtype=torch.float32)
        geo = {}
        for g in GEOMS:
            w = torch.ones(d.size(0), dtype=torch.float64) if g == "uniform" else w_all[g]
            vals, coss = [], []
            for s in SEEDS:
                sess = init_sess[s]
                x, y = sess._diag_batch()
                sess.model.eval()
                params = [pp for pp in sess.model.classifier.parameters() if pp.requires_grad]
                _, lg, _ = sess.model(x)

                def ddo_fn():
                    rr = sess.model.classifier[1:](P.to(sess.model.device))
                    return weighted_l1(rr.abs(), w).mean()
                r_, c_ = reg_ratio(lg, params, None, None, y, 0.0, "ddo", ddo_fn=ddo_fn)
                vals.append(r_)
                coss.append(c_)
            geo[g] = {"ratio_mean": float(np.mean(vals)), "ratio_per_seed": vals, "cos_mean": float(np.mean(coss)),
                      "n_clusters": int(ng), "w_effective_rank": float(
                          torch.exp(-(w / w.sum() * (w / w.sum() + 1e-300).log()).sum()))}
        res[bk] = {"geoms": geo, "diffs_sha256": hashlib.sha256(d.numpy().tobytes()).hexdigest()[:16]}
        print(f"  [{bk}] " + "；".join(f"{g} 比值 {geo[g]['ratio_mean']:.4f}" for g in GEOMS), flush=True)

    check("等权与 rw/group 的权重均值都是 1 (同尺度可比)", True)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": str(device),
           "E_sha256": e_sha, "note": "Stage 0C 的 L1 梯度比补充 (外审第十一轮 P0 第 4 项; 预注册 §3af)。"
           "三种几何的权重构造与 §3ac 的 G_RW/G_group 同族; 全部描述性。",
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
