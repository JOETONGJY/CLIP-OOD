"""任意数据集的初始强度配平（第十二轮 P4 前置；方程与 §3z / stage1c_calibration_traincore.py 完全相同）。

CUB 的标定脚本把 run_config、旧 JSON 与 train_core 批次都写死成 CUB；P4 要在 AwA2 / LADV / PACS 上
“重新标定 λ，不直接照搬 CUB 数值”，因此这里把同一套方程参数化：

  ρ̄_DDO   := 3 个初始化头上、原始等权 DDO (α=1) 的正则梯度比均值（train_core 前 256 固定批次）
  eps_grad := LD 正则部分梯度比 = ρ̄_DDO 的 ε（log-ε 二分）
  lam_grad := ISDA 的对应 λ
  P*_tc    := eps_grad(lang_full) × r_max_mean_tc(lang_full)
  eps_pen  := P*_tc / r_max_mean_tc(e)
  lam_pen  := 2 P*_tc / r_max_sq_mean_tc(e)

会话来自 CUB 的 P1 run_config（同一套协议），只覆盖 dataset / 划分缓存 / run_tag；分类头处于初始化状态。
只用源域数据与文本，不读目标域。

用法: python stage1c_calibration_any.py --dataset AWA2 [--selftest]
产物: experiment_results/Stage1C-Radius/stage1c_calibration_{dataset}.json
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

from language_ellipsoid import domain_directions, second_moments  # noqa: E402
from ld_dro import effective_response  # noqa: E402
from ld_training import build_ld_generator  # noqa: E402
from run_status import git_info  # noqa: E402
from stage1c_radius_diagnose import BISECT_HI, BISECT_IT, BISECT_LO, build_session, bisect, pair_radius, reg_ratio  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1C-Radius")
BASE_RUN = "logs/results/CUB_p1_dose_a1_seed1.json"
SEEDS = (1, 2, 3)
ELLS = ("lang_full", "rand_full", "lang32", "rand32", "identity")
MULTS = (0.5, 1.0, 2.0, 4.0, 8.0)


def r_max_stats(M, y):
    r = pair_radius(M, y)
    r_max = r.max(1).values
    return float(r_max.mean()), float((r_max ** 2).mean())


def self_test():
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
    rm2, _ = r_max_stats(A @ e["random"].float(), y)
    check("C1 r_max 统计为正、平方均值 ≥ 均值平方", rm > 0 and rs >= rm ** 2 - 1e-12)
    check("C2 同谱随机椭球的统计量级可比", 0 < rm2 < 1e6)
    P = 2.0 * rm
    check("C3 eps_pen = P*/r_max_mean", abs(P / rm - 2.0) < 1e-12 and abs(P / rm2 - 2.0 * rm / rm2) < 1e-12)
    check("C4 lam_pen = 2P*/r_max_sq_mean 为正", 2 * P / rs > 0)
    check("C5 MULTS 覆盖 {0.5,1,2,4,8}", MULTS == (0.5, 1.0, 2.0, 4.0, 8.0))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    ds = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else ""
    assert ds, "--dataset 必填 (AWA2 / LADV / PACS)"
    cache = f"cache/{ds}_train_aug.pt"
    assert os.path.exists(cache), f"缺少 {cache}（先运行 build_split_aug_cache.py --dataset {ds}）"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""), flush=True)

    with open(BASE_RUN, encoding="utf-8") as f:
        cfg0 = json.load(f)["run_config"]
    sess = {}
    for s in SEEDS:
        c = dict(cfg0)
        c.update(dataset=ds, train_subset_cache=cache, source_val_cache=cache,
                 run_tag=f"calib_{ds.lower()}_seed{s}", seed=s)
        sess[s] = build_session(c, s)
    E = sess[SEEDS[0]].model.concept_embeddings.detach().double().cpu()
    K = int(E.size(0))
    diffs = sess[SEEDS[0]].train_dataset.domain_diffs.double().cpu()
    G = second_moments(domain_directions(diffs, E))[0]
    print(f"[{ds}] K={K} C={diffs.size(1)} N_desc={diffs.size(0)} tr(G)={float(torch.trace(G)):.4f}")

    ELL, ELL_META = {}, {}
    for s in SEEDS:
        ELL[s], ELL_META[s] = {}, {}
        for e in ELLS:
            L, meta = build_ld_generator(G, e, s)
            ELL[s][e], ELL_META[s][e] = L.to(device), meta

    batches = {s: sess[s]._diag_batch() for s in SEEDS}
    r_mean, r_sq = {}, {}
    for e in ELLS:
        m, q = [], []
        for s in SEEDS:
            x, y = batches[s]
            with torch.no_grad():
                A = effective_response(sess[s].model.classifier)[0]
                rm, rs = r_max_stats(A @ ELL[s][e].float().to(device), y)
            m.append(rm)
            q.append(rs)
        r_mean[e], r_sq[e] = float(np.mean(m)), float(np.mean(q))

    def ratio_init(kind, e, strength, want_cos=False):
        vals, coss = [], []
        for s in SEEDS:
            x, y = batches[s]
            m = sess[s]
            m.model.eval()
            params = [p for p in m.model.classifier.parameters() if p.requires_grad]
            _, lg, _ = m.model(x)
            A = effective_response(m.model.classifier)[0]
            r_, c_ = reg_ratio(lg, params, A, ELL[s][e].float().to(device), y, strength, kind,
                               ddo_fn=m._ddo_loss_equal_weight)
            vals.append(r_)
            coss.append(c_)
        return (float(np.mean(vals)), float(np.mean(coss))) if want_cos else float(np.mean(vals))

    rho, cos_ddo = {}, {}
    for s in SEEDS:
        x, y = batches[s]
        m = sess[s]
        m.model.eval()
        params = [p for p in m.model.classifier.parameters() if p.requires_grad]
        _, lg, _ = m.model(x)
        rho[str(s)], cos_ddo[str(s)] = reg_ratio(lg, params, None, None, y, 0.0, "ddo",
                                                 ddo_fn=m._ddo_loss_equal_weight)
    rho_bar = float(np.mean(list(rho.values())))
    check("ρ̄_DDO 在 (0.02, 0.4) 内（与 CUB 的 0.1049 同量级的合理范围）", 0.02 < rho_bar < 0.4, f"{rho_bar:.6f}")

    cal = {}
    for e in ELLS:
        eg, b1 = bisect(lambda x: ratio_init("ld", e, x), rho_bar, BISECT_LO, BISECT_HI, BISECT_IT)
        lg_, b2 = bisect(lambda x: ratio_init("isda", e, x), rho_bar, BISECT_LO, BISECT_HI, BISECT_IT)
        cal[e] = {"eps_grad": eg, "eps_grad_bracketed": b1, "lam_grad": lg_, "lam_grad_bracketed": b2,
                  "cos_ld_at_eps_grad": ratio_init("ld", e, eg, want_cos=True)[1]}
        check(f"{e}: 梯度配平被二分夹住", b1 and b2, f"eps_grad {eg:.4f}, lam_grad {lg_:.4f}")

    p_star = cal["lang_full"]["eps_grad"] * r_mean["lang_full"]
    for e in ELLS:
        cal[e]["eps_pen"] = p_star / r_mean[e]
        cal[e]["lam_pen"] = 2 * p_star / r_sq[e]
    arms = {e: {"anchor": cal[e]["eps_pen"], "scan": {f"{m:g}": cal[e]["eps_pen"] * m for m in MULTS},
                "equation": "eps_pen (train_core 固定批次)"} for e in ("lang_full", "rand_full", "lang32", "identity")}
    arms["isda"] = {"anchor": cal["lang_full"]["lam_pen"],
                    "scan": {f"{m:g}": cal["lang_full"]["lam_pen"] * m for m in MULTS},
                    "equation": "lam_pen (train_core 固定批次)"}
    arms["isda_rand"] = {"anchor": cal["rand_full"]["lam_pen"],
                         "scan": {f"{m:g}": cal["rand_full"]["lam_pen"] * m for m in MULTS},
                         "equation": "lam_pen (随机谱椭球, train_core 固定批次)"}
    arms["isda_identity"] = {"anchor": cal["identity"]["lam_pen"],
                             "scan": {f"{m:g}": cal["identity"]["lam_pen"] * m for m in MULTS},
                             "equation": "lam_pen (各向同性椭球, train_core 固定批次)"}
    print(f"  ρ̄_DDO={rho_bar!r}; λ_pen(lang)={arms['isda']['anchor']!r}; λ_pen(rand)={arms['isda_rand']['anchor']!r}; "
          f"λ_pen(identity)={arms['isda_identity']['anchor']!r}")

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "device": device, "dataset": ds,
           "note": "第十二轮 P4 的初始强度标定（方程与 §3z / traincore 版完全相同；数据集不同，λ 必须重算）",
           "base_run_config": BASE_RUN, "split_cache": cache,
           "split_cache_sha256": hashlib.sha256(open(cache, "rb").read()).hexdigest(),
           "K": K, "C": int(diffs.size(1)), "N_desc": int(diffs.size(0)),
           "tr_G": float(torch.trace(G)), "rho_ddo": rho, "cos_ddo": cos_ddo, "rho_bar": rho_bar, "p_star": p_star,
           "r_max_mean_tc": r_mean, "r_max_sq_mean_tc": r_sq, "calibration": cal, "arms": arms,
           "mults": list(MULTS),
           "ell_sha256": {f"{s}|{e}": ELL_META[s][e]["sha256"] for s in SEEDS for e in ELLS},
           "all_checks_passed": ok, "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"stage1c_calibration_{ds}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[saved] {path}  ({out['elapsed_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
