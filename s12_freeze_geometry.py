"""第十二轮 P1：冻结 Language / Random-SpecMatched / Identity 三个椭球的方向与种子（预注册 §3ag）。

为什么需要单独一步：外审第十二轮要求“同谱随机矩阵使用预先冻结的多个方向种子，禁止事后筛选”。
`build_ld_generator` 的随机帧由 (G, 训练 seed) 决定：每个训练 seed 用自己的一帧（种子候选 s/s+10/s+20/s+30，
取第一个通过“随机首轴与语言首轴夹角 < 0.5”自检的），三个训练 seed 因此覆盖三个不同的随机子空间。
本脚本在**任何 s12 run 之前**把三帧的 sha256 / 实际种子 / 谱摘要写进 manifest 并提交；run_matrix 在导入期
断言 manifest 与标定文件一致，报告脚本再逐个断言每个 run 的 generator sha256 就是冻结的那一帧。

交叉核对（证明新臂用的随机帧与第十轮已用的完全同一帧）：从 `CUB_s1c_randfull_x1_seed{s}.json` 与
`CUB_s1c_isda_x1_seed{s}.json` 的 ld_config.generator.sha256 逐 seed 比对。

用法: python s12_freeze_geometry.py [--selftest]
产物: experiment_results/Stage1C-Training/s12_geometry_manifest.json
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

import torch  # noqa: E402

from language_ellipsoid import domain_directions, second_moments  # noqa: E402
from ld_training import build_ld_generator  # noqa: E402
from run_status import git_info  # noqa: E402

CALIB = os.path.join("experiment_results", "Stage1C-Radius", "stage1c_calibration_traincore.json")
OUT_DIR = os.path.join("experiment_results", "Stage1C-Training")
OUT_JSON = os.path.join(OUT_DIR, "s12_geometry_manifest.json")
ELLS = ("lang_full", "rand_full", "identity")
SEEDS = (1, 2, 3)


def _sha(t):
    return hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    torch.manual_seed(0)
    U = torch.randn(800, 64, dtype=torch.float64)
    U = U - U.mean(-1, keepdim=True)
    G = second_moments(U)[0]
    a, ma = build_ld_generator(G, "rand_full", 1)
    b, mb = build_ld_generator(G, "rand_full", 1)
    c, mc = build_ld_generator(G, "rand_full", 2)
    check("F1 同一 seed 的随机帧可复现 (sha256 相同)", ma["sha256"] == mb["sha256"] and torch.equal(a, b))
    check("F2 不同 seed 的随机帧不同", ma["sha256"] != mc["sha256"] and not torch.equal(a, c))
    check("F3 记录的 ell_seed_used 与候选顺序一致", ma["ell_seed_used"] in (1, 11, 21, 31)
          and mc["ell_seed_used"] in (2, 12, 22, 32), f"{ma['ell_seed_used']} / {mc['ell_seed_used']}")
    l1, ml1 = build_ld_generator(G, "lang_full", 1)
    l2, ml2 = build_ld_generator(G, "lang_full", 2)
    check("F4 language 不依赖训练 seed", ml1["sha256"] == ml2["sha256"] and torch.equal(l1, l2))
    i1, mi1 = build_ld_generator(G, "identity", 1)
    i2, mi2 = build_ld_generator(G, "identity", 2)
    check("F5 identity 不依赖训练 seed", mi1["sha256"] == mi2["sha256"] and torch.equal(i1, i2))
    check("F6 三臂的生成元宽度/迹一致 (同秩同迹的对照)", a.size(1) == l1.size(1)
          and abs(float((i1.double() @ i1.double().T).trace()) - 1.0) < 1e-6)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return self_test()
    from stage1c_radius_diagnose import build_session, SEEDS as DIAG_SEEDS

    assert tuple(DIAG_SEEDS) == SEEDS, f"诊断脚本的 seeds {DIAG_SEEDS} 与本节 {SEEDS} 不一致"
    with open(CALIB, encoding="utf-8") as f:
        cal = json.load(f)
    t0 = time.time()

    # ===== 1. 与训练完全相同的 G (来自 P1 会话的 dataset 与模型) =====
    sess = {}
    for s in SEEDS:
        with open(f"logs/results/CUB_p1_dose_a1_seed{s}.json", encoding="utf-8") as f:
            r = json.load(f)
        sess[s] = build_session(r["run_config"], s)
    E = sess[SEEDS[0]].model.concept_embeddings.detach().double().cpu()
    e_sha = _sha(E)
    with open(os.path.join("experiment_results", "Stage1C-Radius", "stage1c_radius_cub.json"), encoding="utf-8") as f:
        e_ref = json.load(f)["E_sha256"]
    assert e_sha == e_ref, f"E 与 §3y 记录不一致: {e_sha[:16]} vs {e_ref[:16]}"
    U = domain_directions(sess[SEEDS[0]].train_dataset.domain_diffs.double().cpu(), E)
    G = second_moments(U)[0]
    g_sha = _sha(G)          # G 本身不单独冻结: 第 3 步用生成元 sha256 与第十轮的 run 逐 seed 对齐 (等价且更强)

    # ===== 2. 三帧 + 三臂生成元 =====
    frames = {}
    for s in SEEDS:
        frames[str(s)] = {}
        for name in ELLS:
            L, meta = build_ld_generator(G, name, s)
            frames[str(s)][name] = {"sha256": meta["sha256"], "r": meta["r"],
                                    "ell_seed_used": meta["ell_seed_used"],
                                    "spectrum": meta["spectrum"]}
    rand_shas = {s: frames[str(s)]["rand_full"]["sha256"] for s in SEEDS}
    assert len(set(rand_shas.values())) == 3, "三个训练 seed 的随机帧必须互不相同"
    for name in ("lang_full", "identity"):
        assert len({frames[str(s)][name]["sha256"] for s in SEEDS}) == 1, f"{name} 不应依赖训练 seed"

    # ===== 3. 与第十轮已用的帧交叉核对 (证明不是新造的一套) =====
    xcheck = {}
    for s in SEEDS:
        for name, tag in (("rand_full", f"CUB_s1c_randfull_x1_seed{s}"), ("lang_full", f"CUB_s1c_isda_x1_seed{s}")):
            p = f"logs/results/{tag}.json"
            with open(p, encoding="utf-8") as f:
                old = json.load(f)["ld_config"]["generator"]["sha256"]
            same = old == frames[str(s)][name]["sha256"]
            xcheck[tag] = {"old_sha256": old, "new_sha256": frames[str(s)][name]["sha256"], "same": same}
            assert same, f"{tag}: 生成元与第十轮不一致 ({old[:16]} vs {frames[str(s)][name]['sha256'][:16]})"

    # ===== 4. 冻结的 λ (来自统一批次标定, 不在本脚本重算) =====
    def lam_pen(name):
        return float(cal["calibration"][name]["lam_pen"])
    lam = {name: lam_pen(name) for name in ELLS}
    mults = ["0.5", "1", "2", "4", "8"]
    grid = {name: {m: lam[name] * float(m) for m in mults} for name in ELLS}
    same_lam = {"value": grid["lang_full"]["4"],
                "note": "Language-ISDA 4× 的 λ；随机/Identity 臂在同一数值下的对照 (外审第十二轮 P1 第 1 项)"}
    warm = {"lang_full_x4": grid["lang_full"]["4"], "rand_full_x8": grid["rand_full"]["8"],
            "identity_x8": grid["identity"]["8"], "epochs": 20}

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "note": "第十二轮 P1 的几何冻结 (预注册 §3ag)：随机谱椭球的三个训练 seed 各一帧，禁止事后筛选；"
                   "λ 取自统一批次标定 stage1c_calibration_traincore.json，不在本脚本重算。",
           "E_sha256": e_sha, "G_sha256": g_sha, "calibration_sha256": _sha_file(CALIB),
           "frames": frames, "lambda_pen": lam, "lambda_grid": grid, "same_lambda": same_lam, "warmup": warm,
           "cross_check": xcheck, "all_checks_passed": True,
           "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[E] {e_sha[:16]} | [G] {g_sha[:16]} | 与 §3y 一致")
    for s in SEEDS:
        f = frames[str(s)]
        print(f"  seed {s}: random seed_used={f['rand_full']['ell_seed_used']} sha {f['rand_full']['sha256'][:16]} | "
              f"lang {f['lang_full']['sha256'][:16]} | identity {f['identity']['sha256'][:16]}")
    print(f"  λ_pen: lang {lam['lang_full']:.6f} | rand {lam['rand_full']:.6f} | identity {lam['identity']:.6f}")
    print(f"  same-λ 4× = {same_lam['value']:.6f}")
    print(f"[saved] {OUT_JSON}  (耗时 {out['elapsed_seconds']}s)")
    return 0


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
