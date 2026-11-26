"""初始强度配平的统一批次标定（外审第十轮 P2 收尾第 4 项；预注册于 RESULTS §3z，先于任何八臂训练运行）。

§3x/§3y 的配平里，梯度比用的是 train_core 的固定批次（train_core 的前 256 个样本），而最大类别对惩罚 r_max
的统计用的是 heldout_val（1200 个样本）上的真实类别。这不是目标域泄漏，但 heldout_val 应当专门承担 P1 的
选参功能。本脚本把配平的**全部四个量**统一到 train_core 的固定批次上，轻量重算（不重跑任何 PGD、不碰收敛头）：

  ρ̄_DDO        := 原始等权 DDO（α = 1）在 3 个初始化头上的梯度比均值（固定批次；与 §3y 相同 → 核对一致）
  eps_grad(e)  := LD 正则部分梯度比 = ρ̄_DDO 的 ε（二分；固定批次）—— 与 §3y 相同 → 核对一致
  lam_grad(e)  := ISDA 的对应量 —— 与 §3y 相同 → 核对一致
  P*_tc        := eps_grad(lang_full) × r_max_mean_tc(lang_full)          ← r_max 统计改在固定批次上（变化点）
  eps_pen(e)   := P*_tc / r_max_mean_tc(e)                                ← 变化点
  lam_pen(e)   := 2 P*_tc / r_max_sq_mean_tc(e)                           ← 变化点

八臂矩阵的 ε / λ 从本脚本的输出 JSON 取值并冻结进 §3z；若与 §3y 的参考值略有出入，以本输出为准
（GPT 第十轮指令：统一批次后按预先确定的配平方程更新数值并冻结，不按最终准确率手工调整）。
固定批次 = train_core 的前 256 个样本（与训练时 grad_diag 的诊断批次同一个）。只用源域数据与文本。

用法: python stage1c_calibration_traincore.py [--selftest]
产物: experiment_results/Stage1C-Radius/stage1c_calibration_traincore.json
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

from stage1c_radius_diagnose import (BISECT_LO, BISECT_HI, BISECT_IT, ELL_ORDER, OUT_DIR, SEEDS,  # noqa: E402
                                     bisect, pair_radius, prepare, reg_ratio)
from ld_dro import effective_response  # noqa: E402
from run_status import git_info  # noqa: E402

OLD_JSON = os.path.join(OUT_DIR, "stage1c_radius_cub.json")
NEW_JSON = os.path.join(OUT_DIR, "stage1c_calibration_traincore.json")
ARM_ORDER = ("lang_full", "rand_full", "lang32", "rand32", "identity")
ANCHOR_ELLS = ("lang_full", "rand_full", "lang32", "identity")   # 八臂的 LD 臂（rand32 不在八臂里）
MULTS = (0.5, 1.0, 2.0)


def r_max_stats(M, y):
    """固定批次上的 (r_max 均值, r_max² 均值)。r_yk = ‖M_k − M_y‖，真实类位置为 0 —— 与 §3x 同一定义。"""
    r = pair_radius(M, y)
    r_max = r.max(1).values
    return float(r_max.mean()), float((r_max ** 2).mean())


def self_test():
    """合成数据上核对配平方程本身（不碰真实数据）。"""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    K, C, B = 40, 9, 64
    A = torch.randn(C, K) * 0.5
    y = torch.randint(0, C, (B,))
    from language_ellipsoid import build_ellipsoids
    U = torch.randn(500, K, dtype=torch.float64)
    U = U - U.mean(-1, keepdim=True)
    G = U.T @ U / U.size(0)
    e = build_ellipsoids(G, r=10, seed=3, with_flat=False)
    rm, rs = r_max_stats(A @ e["language"].float(), y)
    rm2, rs2 = r_max_stats(A @ e["random"].float(), y)
    check("C1 r_max 统计为正且平方均值 ≥ 均值平方（同一批内）", rm > 0 and rs >= rm ** 2 - 1e-12, f"{rm:.4f}, {rs:.4f}")
    check("C2 同谱随机椭球的 r_max 统计是有限的正数（量级可比）", 0 < rm2 < 1e6, f"{rm2:.4f}")
    P = 2.0 * rm                                            # 假想 eps_grad(lang_full) = 2
    check("C3 eps_pen 方程: 同 P* 下反比于 r_max 均值", abs(P / rm - 2.0) < 1e-12 and abs(P / rm2 - 2.0 * rm / rm2) < 1e-12)
    check("C4 lam_pen 方程: 2P*/r_max² 均值", abs(2 * P / rs - 2 * P / rs) < 1e-15 and 2 * P / rs > 0)
    check("C5 MULTS 是预注册的 {0.5, 1, 2}", MULTS == (0.5, 1.0, 2.0))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    with open(OLD_JSON, encoding="utf-8") as f:
        old = json.load(f)
    ctx = prepare(device, check)                             # 与正式诊断相同的步骤 1–4（含 5 项核对）
    init_sess, ELL = ctx["init_sess"], ctx["ELL"]

    # ===== 1. 固定批次上的 r_max 统计（变化点：§3y 用 heldout_val，这里用 train_core 的固定批次） =====
    batches = {s: init_sess[s]._diag_batch() for s in SEEDS}
    r_mean, r_sq_mean = {}, {}
    for e in ELL_ORDER:
        per_s, per_s_sq = [], []
        for s in SEEDS:
            x, y = batches[s]
            with torch.no_grad():
                A = effective_response(init_sess[s].model.classifier)[0]
                rm, rs = r_max_stats(A @ ELL[s][e].float().to(device), y)
            per_s.append(rm)
            per_s_sq.append(rs)
        r_mean[e], r_sq_mean[e] = float(np.mean(per_s)), float(np.mean(per_s_sq))
    old_r = old["r_max_mean_init"]
    check("固定批次（train_core 前 256）与 heldout_val 的 r_max 统计量级相同但数值不同（统一批次有实际效果）",
          all(0.5 < r_mean[e] / old_r[e] < 2.0 and r_mean[e] != old_r[e] for e in ELL_ORDER),
          "; ".join(f"{e}: {r_mean[e]:.4f} vs {old_r[e]:.4f}" for e in ELL_ORDER))

    # ===== 2. 梯度配平（与 §3y 同一批次、同一方程 → 必须逐项一致） =====
    def ratio_init(kind, e, strength, want_cos=False):
        vals, coss = [], []
        for s in SEEDS:
            x, y = batches[s]
            sess = init_sess[s]
            sess.model.eval()
            params = [p for p in sess.model.classifier.parameters() if p.requires_grad]
            _, lg, _ = sess.model(x)
            A = effective_response(sess.model.classifier)[0]
            r_, c_ = reg_ratio(lg, params, A, ELL[s][e].float().to(device), y, strength, kind,
                               ddo_fn=sess._ddo_loss_equal_weight)
            vals.append(r_)
            coss.append(c_)
        return (float(np.mean(vals)), float(np.mean(coss))) if want_cos else float(np.mean(vals))

    rho_ddo, cos_ddo = {}, {}
    for s in SEEDS:
        x, y = batches[s]
        sess = init_sess[s]
        sess.model.eval()
        params = [p for p in sess.model.classifier.parameters() if p.requires_grad]
        _, lg, _ = sess.model(x)
        rho_ddo[str(s)], cos_ddo[str(s)] = reg_ratio(lg, params, None, None, y, 0.0, "ddo",
                                                     ddo_fn=sess._ddo_loss_equal_weight)
    rho_bar = float(np.mean(list(rho_ddo.values())))
    cal = {}
    for e in ELL_ORDER:
        eg, br1 = bisect(lambda x: ratio_init("ld", e, x), rho_bar, BISECT_LO, BISECT_HI, BISECT_IT)
        lg_, br2 = bisect(lambda x: ratio_init("isda", e, x), rho_bar, BISECT_LO, BISECT_HI, BISECT_IT)
        cal[e] = {"eps_grad": eg, "eps_grad_bracketed": br1, "lam_grad": lg_, "lam_grad_bracketed": br2,
                  "cos_ld_at_eps_grad": ratio_init("ld", e, eg, want_cos=True)[1]}
        oc = old["calibration"][e]
        check(f"{e}: 梯度配平与 §3y 逐项一致（同批次同方程的重算核对）",
              abs(eg - oc["eps_grad"]) < 1e-9 and abs(lg_ - oc["lam_grad"]) < 1e-9 and br1 and br2,
              f"eps_grad {eg:.6f} vs {oc['eps_grad']:.6f}; lam_grad {lg_:.4f} vs {oc['lam_grad']:.4f}")
    check("ρ̄_DDO 与 §3y 一致", abs(rho_bar - old["rho_bar"]) < 1e-9, f"{rho_bar:.6f} vs {old['rho_bar']:.6f}")

    # ===== 3. 统一批次的同初始惩罚口径与八臂取值 =====
    p_star = cal["lang_full"]["eps_grad"] * r_mean["lang_full"]
    for e in ELL_ORDER:
        cal[e]["eps_pen"] = p_star / r_mean[e]
        cal[e]["lam_pen"] = 2 * p_star / r_sq_mean[e]
    arms = {}
    for e in ANCHOR_ELLS:
        arms[e] = {"anchor": cal[e]["eps_pen"], "scan": {f"{m:g}": cal[e]["eps_pen"] * m for m in MULTS},
                   "equation": "eps_pen (train_core 固定批次)"}
    arms["isda"] = {"anchor": cal["lang_full"]["lam_pen"], "scan": {f"{m:g}": cal["lang_full"]["lam_pen"] * m for m in MULTS},
                    "equation": "lam_pen (train_core 固定批次)"}
    controls = {"identity_grad": cal["identity"]["eps_grad"], "isda_grad": cal["lang_full"]["lam_grad"],
                "randfull_same": cal["lang_full"]["eps_pen"], "pgd": cal["lang_full"]["eps_pen"]}
    print("\n冻结用取值（完整精度，run_matrix 从这里粘贴）：")
    for e in ANCHOR_ELLS:
        print(f"  {e:10s} anchor={arms[e]['anchor']!r}  scan=" + str({k: repr(v) for k, v in arms[e]['scan'].items()}))
    print(f"  isda       anchor={arms['isda']['anchor']!r}")
    print("  controls   " + str({k: repr(v) for k, v in controls.items()}))

    check("Identity-H 的统一批次 eps_pen 与 eps_grad 仍相差 3 倍以上（两套口径不能互相替代的结论不变）",
          cal["identity"]["eps_grad"] / cal["identity"]["eps_pen"] > 3.0,
          f"比值 {cal['identity']['eps_grad'] / cal['identity']['eps_pen']:.2f}")

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": device,
           "note": "统一 train_core 固定批次的初始强度标定（外审第十轮 P2 收尾第 4 项）。与 §3y 的差别只在 "
                   "r_max 统计的样本集（heldout_val → train_core 前 256）；梯度配平逐项一致（核对见 checks）。",
           "old_json_sha256": hashlib.sha256(open(OLD_JSON, "rb").read()).hexdigest(),
           "fixed_batch": "train_core 的前 256 个样本（与训练时 grad_diag 的诊断批次相同）",
           "equations": {"rho_bar": "3 个初始化头上等权 DDO (α=1) 正则梯度比均值",
                         "eps_grad": "LD 正则部分梯度比 = rho_bar 的 eps（二分）",
                         "P_star_tc": "eps_grad(lang_full) * r_max_mean_tc(lang_full)",
                         "eps_pen": "P_star_tc / r_max_mean_tc(e)",
                         "lam_pen": "2 * P_star_tc / r_max_sq_mean_tc(e)"},
           "rho_ddo": rho_ddo, "cos_ddo": cos_ddo, "rho_bar": rho_bar, "p_star": p_star,
           "r_max_mean_tc": r_mean, "r_max_sq_mean_tc": r_sq_mean, "r_max_mean_heldout_ref": old_r,
           "calibration": cal, "arms": arms, "controls": controls, "mults": list(MULTS),
           "ell_sha256": {f"{s}|{k}": ctx["ell_meta"][s][k]["sha256"] for s in SEEDS for k in ("full", "r32")},
           "all_checks_passed": ok, "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NEW_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n[saved] {NEW_JSON}  (耗时 {out['elapsed_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
