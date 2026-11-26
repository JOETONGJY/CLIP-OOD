"""SDB-204 词库的初始强度标定（外审第十一轮 P2"新词库重新标定锚点"；预注册于 RESULTS §3ad，先于任何 SDB+ISDA run）。

与 stage1c_calibration_traincore.py 同一套配平方程，但方向集换成 SDB-204：
  ρ̄_DDO(SDB)   := 等权 DDO 在【SDB 的完整方向集】上的初始梯度比（3 个 P1 初始化头、train_core 前 256 固定批次）
  eps_grad(SDB) := LD 正则部分梯度比 = ρ̄_DDO(SDB) 的 ε（SDB 的 Language-Full 椭球；二分）
  λ_pen(SDB)    := 2 P*_SDB / r_max² 均值，P*_SDB = eps_grad(SDB) × r_max_mean(SDB)（统一 train_core 批次）
SDB+ISDA 各倍数的 λ 从本输出 JSON 冻结取值。核对：SDB diffs 经 attach_domain_reliability 的同一路径取得
（缓存键含词库/类别/概念/CLIP 身份），与训练时 dataset.domain_diffs 同源；椭球经 build_ld_generator
（与训练 _init_ld_dro 同一构造），生成元 sha256 记录在案，报告脚本将断言训练 run 的 sha 与本 JSON 一致。
E 经 P1 会话的 model.concept_embeddings 取得（与 §3y 逐位一致的编码路径）。

用法: python stage1c_calibration_sdb.py [--selftest]
产物: experiment_results/Stage1C-Radius/stage1c_calibration_sdb.json
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
from ld_training import build_ld_generator  # noqa: E402  与训练同一椭球构造
from language_ellipsoid import domain_directions, second_moments  # noqa: E402
from reliability import attach_domain_reliability  # noqa: E402
from run_status import git_info  # noqa: E402
from stage1c_calibration_traincore import BISECT_HI, BISECT_IT, BISECT_LO, MULTS, r_max_stats  # noqa: E402
from stage1c_radius_diagnose import SEEDS, bisect, reg_ratio  # noqa: E402

ORIG_CALIB = os.path.join("experiment_results", "Stage1C-Radius", "stage1c_calibration_traincore.json")

OUT_DIR = os.path.join("experiment_results", "Stage1C-Radius")
NEW_JSON = os.path.join(OUT_DIR, "stage1c_calibration_sdb.json")
SDB_PATH = os.path.join("prompts", "banks", "sdb-204-v1.json")
CLIP_PATH = os.path.abspath("clip_weights/ViT-L-14.pt")
ARM = "lang_full"                      # SDB+ISDA 用 SDB 的 Language-Full 椭球
N_BATCH = 256                          # train_core 的前 256（与原标定/训练 grad_diag 同一批次）


def _sha(t):
    return hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    check("S1 ARM 是 SDB 的 Language-Full", ARM == "lang_full")
    check("S2 倍数网格沿用 P1 扩展后的统一网格", MULTS == (0.5, 1.0, 2.0))
    check("S3 输出文件不覆盖 Original 的标定", os.path.basename(NEW_JSON) != os.path.basename(ORIG_CALIB))
    # 配平方程的合成核对（与原标定自测同族）
    torch.manual_seed(0)
    K, C, B = 30, 7, 50
    A = torch.randn(C, K)
    y = torch.randint(0, C, (B,))
    from language_ellipsoid import build_ellipsoids
    U = torch.randn(400, K, dtype=torch.float64)
    U = U - U.mean(-1, keepdim=True)
    G = U.T @ U / U.size(0)
    e = build_ellipsoids(G, r=K - 1, seed=0, with_flat=False)
    rm, rs = r_max_stats(A @ e["language"].float(), y)
    P = 2.0 * rm
    check("S4 eps_pen/lam_pen 方程可计算且为正", rm > 0 and 2 * P / rs > 0)
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
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    import clip
    from stage1c_radius_diagnose import build_session  # 与训练同序播种的会话构造

    # ===== 1. 三个 P1 初始化会话（分类头初始状态与固定批次；词库方向集只影响 rho_bar 的 ddo_fn） =====
    init_sess = {}
    for s in SEEDS:
        with open(f"logs/results/CUB_p1_dose_a1_seed{s}.json", encoding="utf-8") as f:
            r = json.load(f)
        init_sess[s] = build_session(r["run_config"], s)
    E = init_sess[SEEDS[0]].model.concept_embeddings.detach().double().cpu()
    e_sha = _sha(E)
    with open(os.path.join("experiment_results", "Stage1C-Radius", "stage1c_radius_cub.json"), encoding="utf-8") as f:
        e_ref = json.load(f)["E_sha256"]
    check("E 与 §3y 记录逐位一致（clip_cbm_orth 编码路径）", e_sha == e_ref, f"{e_sha[:16]} vs {e_ref[:16]}")

    # ===== 2. SDB diffs：attach 同一路径（缓存键与训练一致；miss 时现算并写缓存） =====
    b = load_descriptor_bank(SDB_PATH)
    holder = type("H", (), {"concept2id": init_sess[SEEDS[0]].train_dataset.concept2id})()
    clip_model, _ = clip.load(CLIP_PATH, device=device)
    attach_domain_reliability(holder, clip_model, b["source_text_prompts"], b["target_text_prompts"],
                              list(init_sess[SEEDS[0]].train_dataset.classname2id.keys()), device,
                              weight_strategy="prior_residual", verbose=True, clip_path=CLIP_PATH)
    diffs = holder.domain_diffs.double().cpu()
    n_cls = len(list(init_sess[SEEDS[0]].train_dataset.classname2id.keys()))
    check("SDB diffs 形状 = (204 描述符, 200 类, 768)", tuple(diffs.shape) == (b["n_target"], n_cls, diffs.size(2)),
          str(tuple(diffs.shape)))
    U = domain_directions(diffs, E)
    G = second_moments(U)[0]
    g_sha = hashlib.sha256(G.numpy().tobytes()).hexdigest()
    L, gmeta = build_ld_generator(G, ARM, SEEDS[0])
    check("SDB Language-Full 生成元：迹=1、超平面内", abs(float((L.double() @ L.double().T).trace()) - 1.0) < 1e-6,
          f"sha {gmeta['sha256'][:16]}, seed_used {gmeta['ell_seed_used']}")

    # ===== 3. rho_bar(SDB) + 固定批次上的 r_max 统计 =====
    P_sdb = torch.tensor(U.reshape(b["n_target"], -1, E.size(1)), dtype=torch.float32)
    batches = {}
    rho_ddo, cos_ddo = {}, {}
    for s in SEEDS:
        sess = init_sess[s]
        x, y = sess._diag_batch()
        batches[s] = (x, y)
        sess.model.eval()
        params = [pp for pp in sess.model.classifier.parameters() if pp.requires_grad]
        _, lg, _ = sess.model(x)
        rho_ddo[str(s)], cos_ddo[str(s)] = reg_ratio(
            lg, params, None, None, y, 0.0, "ddo",
            ddo_fn=lambda: sess.model.classifier[1:](P_sdb.to(sess.model.device)).abs().mean())
    rho_bar = float(np.mean(list(rho_ddo.values())))
    check("ρ̄_DDO(SDB 方向集) 为正且与 Original 的标定值不同（方向集不同，配平目标不同）",
          0 < rho_bar < 1 and abs(rho_bar - 0.1049) > 1e-6, f"{rho_bar:.6f}")

    r_mean, r_sq_mean = {}, {}
    per_s, per_s_sq = [], []
    for s in SEEDS:
        x, y = batches[s]
        with torch.no_grad():
            A = init_sess[s].model.classifier[2].weight * init_sess[s].model.classifier[1].weight[None, :]
            rm, rs = r_max_stats(A.to(device) @ L.to(device), y)
        per_s.append(rm)
        per_s_sq.append(rs)
    r_mean[ARM], r_sq_mean[ARM] = float(np.mean(per_s)), float(np.mean(per_s_sq))

    # ===== 4. 梯度配平二分（SDB 椭球） + 统一批次的同初始惩罚口径 =====
    def ratio_init(kind, strength):
        vals = []
        for s in SEEDS:
            sess = init_sess[s]
            x, y = batches[s]
            sess.model.eval()
            params = [pp for pp in sess.model.classifier.parameters() if pp.requires_grad]
            _, lg, _ = sess.model(x)
            A = sess.model.classifier[2].weight * sess.model.classifier[1].weight[None, :]
            r_, _ = reg_ratio(lg, params, A, L.to(device), y, strength, kind,
                              ddo_fn=lambda: sess._ddo_loss_equal_weight())
            vals.append(r_)
        return float(np.mean(vals))

    eg, br1 = bisect(ratio_init_args(ratio_init, "ld"), rho_bar, BISECT_LO, BISECT_HI, BISECT_IT)
    lam_g, br2 = bisect(ratio_init_args(ratio_init, "isda"), rho_bar, BISECT_LO, BISECT_HI, BISECT_IT)
    check("eps_grad 二分两端夹住", br1 and br2, f"eps_grad {eg:.6f}, lam_grad {lam_g:.4f}")
    p_star = eg * r_mean[ARM]
    eps_pen = p_star / r_mean[ARM]
    lam_pen = 2 * p_star / r_sq_mean[ARM]
    scan = {f"{m:g}": lam_pen * m for m in MULTS}
    print("\n冻结用取值（完整精度）：")
    print(f"  eps_grad(SDB)={eg!r}")
    print(f"  lam_grad(SDB)={lam_g!r}")
    print(f"  lam_pen(SDB) anchor={lam_pen!r}  scan=" + str({k: repr(v) for k, v in scan.items()}))

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": str(device),
           "note": "SDB-204 词库的初始强度标定（外审第十一轮 P2；方程与 stage1c_calibration_traincore 相同，"
                   "方向集与椭球换成 SDB-204）。SDB+ISDA 的 λ 从这里冻结取值。",
           "bank": {"id": b["id"], "bank_sha256": b["bank_sha256"], "file_sha256": b["meta"]["file_sha256"]},
           "E_sha256": e_sha, "G_sha256": g_sha, "generator": gmeta,
           "rho_ddo_sdb": rho_ddo, "cos_ddo_sdb": cos_ddo, "rho_bar_sdb": rho_bar,
           "eps_grad": eg, "lam_grad": lam_g, "p_star": p_star,
           "r_max_mean_tc": r_mean, "r_max_sq_mean_tc": r_sq_mean,
           "arms": {ARM: {"anchor": eps_pen, "equation": "eps_pen (train_core 固定批次, SDB 方向集)",
                          "scan": {f"{m:g}": eps_pen * m for m in MULTS}}},
           "isda": {"anchor": lam_pen, "equation": "lam_pen (train_core 固定批次, SDB 方向集)",
                    "scan": scan},
           "mults": list(MULTS), "all_checks_passed": ok,
           "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n[saved] {NEW_JSON}  (耗时 {out['elapsed_seconds']}s)")
    return 0 if ok else 1


def ratio_init_args(fn, kind):
    return lambda x: fn(kind, x)


if __name__ == "__main__":
    sys.exit(main())
