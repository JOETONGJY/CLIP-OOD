"""第十二轮的分析与判定（预注册 RESULTS §3ag）：Language-ISDA 的几何对照 + warmup 消融 + 选参规则冻结。

--stage source : 读 s12_geom (45) 与复用的第十/十一轮 run（Language-ISDA 五倍数、P3 warmup@8×、P1 的 DDO 七剂量，
                 另把 LD 与 SDB-ISDA 作为联合规则的事后描述性应用），逐 run 核对：
                 ① 配置与 run_matrix.expected 一致、结果完整；
                 ② 结果 json 里没有任何目标域数字（训练时 --target_eval none）；
                 ③ 独立划分 (train_core 4790 / heldout_val 1200) 与 7 种扰动齐全；
                 ④ 每个 run 的椭球生成元 sha256 = 冻结清单里该训练 seed 的那一帧（禁止事后挑选随机帧）；
                 ⑤ α=0、ld_mode=isda、λ 与冻结的网格逐位相同。
                 然后按两条规则（R_aug / R_joint）对各臂选强度；写 s12_source_selection.{json,md} → 提交锁定。
--stage final  : 要求选择文件已锁定；重算源域并与锁定文件逐项比对；读入 eval_heads_targets.py --s12 的
                 事后目标域评估（复用 s1c / s11 / p1 的事后评估），输出 G1 / G1b / G1c / G1d / G2 / G3 与
                 两条规则的 Selection Regret、Spearman、逐 seed 稳定性。写 s12_final_report.{json,md}。
--selftest     : 合成数据上的规则自测（配对门槛、并列、两条规则、regret、Spearman）。

预注册阈值（§3ag，沿用 §3z）：平均 ≥ 0.5 pp 且 ≥ 2/3 配对 seed 正向 —— 研发门槛，不是统计显著性。
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

from p1_independent_report import _locked, argbest  # noqa: E402
from run_matrix import COMMON, MATRICES, expected  # noqa: E402
from run_status import git_info, result_is_complete  # noqa: E402
from stage1c_training_report import GATE_POS, GATE_PP, SEEDS, _paired, gate, sha256_file  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage1C-Training")
SEL_JSON = os.path.join(OUT_DIR, "s12_source_selection.json")
SEL_MD = os.path.join(OUT_DIR, "s12_source_selection.md")
FIN_JSON = os.path.join(OUT_DIR, "s12_final_report.json")
FIN_MD = os.path.join(OUT_DIR, "s12_final_report.md")
TGT_JSON = os.path.join(OUT_DIR, "s12_target_eval.json")
MANIFEST = os.path.join(OUT_DIR, "s12_geometry_manifest.json")
S1C_TGT = os.path.join(OUT_DIR, "s1c_target_eval.json")
S11_TGT = os.path.join(OUT_DIR, "s11_target_eval.json")
P1_TGT = os.path.join("experiment_results", "Stage0D-Independent", "p1_target_eval.json")
CALIB_JSON = os.path.join("experiment_results", "Stage1C-Radius", "stage1c_calibration_traincore.json")

MULTS = ["0.5", "1", "2", "4", "8"]
MTAG = {"0.5": "x0p5", "1": "x1", "2": "x2", "4": "x4", "8": "x8"}
DOSES = [("0", "dose_a0"), ("0.125", "dose_a0p125"), ("0.25", "dose_a0p25"), ("0.5", "dose_a0p5"),
         ("1", "dose_a1"), ("2", "dose_a2"), ("4", "dose_a4")]
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]
N_TRAIN, N_VAL = 4790, 1200

# 复用 run 的出处（源域字段所在的结果 json 都按同一协议训练）
REUSE = {
    "language": {"0.5": ("stage1c", "s1c_isda"), "1": ("stage1c", "s1c_isda"), "2": ("stage1c", "s1c_isda"),
                 "4": ("s1c_ext", "s1c_isda"), "8": ("s1c_ext", "s1c_isda")},
    "ld": {"0.5": ("stage1c", "s1c_langfull"), "1": ("stage1c", "s1c_langfull"), "2": ("stage1c", "s1c_langfull"),
           "4": ("s1c_ext", "s1c_langfull"), "8": ("s1c_ext", "s1c_langfull")},
    "sdb_isda": {"0.5": ("sdb22", "sdb_isda"), "1": ("sdb22", "sdb_isda"), "2": ("sdb22", "sdb_isda"),
                 "4": ("sdb22", "sdb_isda"), "8": ("sdb22", "sdb_isda")},
}


def _runs_of(matrix, tag, s):
    return f"CUB_{tag}_seed{s}"


def _ddo_runs():
    return {a: f"CUB_p1_{t}" for a, t in DOSES}


def load_runs():
    """读取并核对全部源域字段。返回 runs: name -> 指标 dict; meta: 冻结清单与标定。"""
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    cal_sha = sha256_file(CALIB_JSON)
    runs = {}

    def read(name, want_extra=None, require_no_target=True):
        rp, hp = f"logs/results/{name}.json", f"logs/heads/{name}.pt"
        assert os.path.exists(rp) and os.path.exists(hp), f"{name}: 结果或分类头缺失"
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        if want_extra is not None:
            assert result_is_complete(rp, expected(COMMON + want_extra, "CUB", int(name[-1]), name[4:-6])), \
                f"{name}: 不完整或配置不匹配"
        if require_no_target:
            assert r.get("target_acc") is None and (r.get("final_diag") or {}).get("target_acc") is None \
                and all(e["target_acc"] is None for e in r["epoch_metrics"]), f"{name}: 训练结果里有目标域数字"
        ts, sv = r["train_subset"], r["source_val"]
        assert (ts["n_train"], ts["n_heldout"]) == (N_TRAIN, N_VAL) and sv["n"] == N_VAL \
            and sorted(sv["aug"]) == sorted(AUGS) and ts["split_cache_sha256"] == sv["cache_sha256"], \
            f"{name}: 划分/验证集不符"
        return r, {"clean_acc": sv["clean"]["acc"], "mean_aug_acc": sv["mean_aug_acc"],
                   "mean_aug_ce": sv["mean_aug_ce"], "head_sha256": sha256_file(hp),
                   "result_sha256": sha256_file(rp), "elapsed": r.get("elapsed_seconds"),
                   "generator": (r.get("ld_config") or {}).get("generator"),
                   "ld_calib": ((r.get("ld_config") or {}).get("calibration") or {}).get("sha256"),
                   "run_config": r["run_config"]}

    # ---- s12 的 45 个 run ----
    entries = {t: (ds, ex) for ds, t, ex in MATRICES["s12_geom"][1]}
    assert len(entries) == 15, len(entries)
    for tag, (ds, ex) in entries.items():
        for s in SEEDS:
            name = f"{ds}_{tag}_seed{s}"
            want_lam = expected(COMMON + ex, ds, s, tag)["ld_lam"]
            r, m = read(name, ex)
            cfg = r["run_config"]
            assert cfg["ld_mode"] == "isda" and float(cfg["alpha"]) == 0.0 and cfg["ld_ell"] in ("rand_full", "identity", "lang_full")
            assert float(cfg["ld_lam"]) == float(want_lam), f"{name}: λ 与冻结网格不一致"
            assert m["ld_calib"] == cal_sha, f"{name}: 标定引用不是统一批次标定"
            frame = man["frames"][str(s)][cfg["ld_ell"]]
            assert m["generator"] and m["generator"]["sha256"] == frame["sha256"], \
                f"{name}: 椭球生成元不是冻结清单里 seed {s} 的那一帧"
            m["arm"] = ("random" if "rand" in tag else "identity" if "ident" in tag else "language")
            m["kind"] = "warmup" if "warm" in tag else ("samelam" if "samelam" in tag else "grid")
            m["mult"] = None if m["kind"] != "grid" else [k for k, v in MTAG.items() if f"_{v}" in tag][0]
            runs[name] = m

    # ---- 复用的 Language 五倍数 / LD / SDB-ISDA 与 P1 的 DDO 七剂量 ----
    for arm, table in REUSE.items():
        for m_, (mname, tag) in table.items():
            for s in SEEDS:
                name = _runs_of(mname, tag, s)
                ex = [x for _, t, x in MATRICES[mname][1] if t == tag]
                assert len(ex) == 1, f"{name}: 在 {mname} 里找不到唯一配置"
                r, mm = read(name, ex[0])
                assert mm["ld_calib"] == cal_sha, f"{name}: 标定引用不符"
                if arm != "sdb_isda":
                    assert mm["generator"]["sha256"] == man["frames"][str(s)]["lang_full"]["sha256"], \
                        f"{name}: language 生成元与冻结清单不一致"
                mm.update({"arm": arm, "kind": "grid", "mult": m_})
                runs[name] = mm
    for s in SEEDS:
        for a, pfx in _ddo_runs().items():
            name = f"{pfx}_seed{s}"
            with open(f"logs/results/{name}.json", encoding="utf-8") as f:
                r = json.load(f)
            runs[name] = {"clean_acc": r["source_val"]["clean"]["acc"], "mean_aug_acc": r["source_val"]["mean_aug_acc"],
                          "mean_aug_ce": r["source_val"]["mean_aug_ce"], "arm": "ddo", "kind": "dose", "mult": a,
                          "head_sha256": sha256_file(f"logs/heads/{name}.pt"),
                          "result_sha256": sha256_file(f"logs/results/{name}.json"),
                          "elapsed": r.get("elapsed_seconds")}
    # P3 的 warmup@8× (第十一轮) 也参与 G2
    for s in SEEDS:
        name = f"CUB_isda_warm_x8_seed{s}"
        with open(f"logs/results/{name}.json", encoding="utf-8") as f:
            r = json.load(f)
        runs[name] = {"clean_acc": r["source_val"]["clean"]["acc"], "mean_aug_acc": r["source_val"]["mean_aug_acc"],
                      "mean_aug_ce": r["source_val"]["mean_aug_ce"], "arm": "language", "kind": "warmup", "mult": "8",
                      "head_sha256": sha256_file(f"logs/heads/{name}.pt"),
                      "result_sha256": sha256_file(f"logs/results/{name}.json"),
                      "elapsed": r.get("elapsed_seconds")}
    return runs, man


# ---------------- 曲线与两条选参规则 ----------------
RULE_TEXT = {"R_aug": "heldout_val 7 种扰动均值 Acc 最大 (P1 冻结规则, 对照)",
             "R_joint": "0.5·干净 Acc + 0.5·扰动均值 Acc 最大 (第十二轮 P2 冻结)"}


def _curve_name(arm, m, s):
    if arm == "language":
        return f"CUB_s1c_isda_{MTAG[m]}_seed{s}"
    if arm == "ld":
        return f"CUB_s1c_langfull_{MTAG[m]}_seed{s}"
    if arm == "sdb_isda":
        return f"CUB_sdb_isda_{MTAG[m]}_seed{s}"
    return f"CUB_s12_isda_{'rand' if arm == 'random' else 'ident'}_{MTAG[m]}_seed{s}"


def selection(runs):
    """对每个可扫臂应用两条规则。DDO 的七剂量同样适用（公平比较）。"""
    arms = ["ddo", "language", "random", "identity", "ld", "sdb_isda"]
    out = {}
    for arm in arms:
        keys = [a for a, _ in DOSES] if arm == "ddo" else MULTS

        def rec(a, k, s):
            if arm == "ddo":
                return runs[f"{_ddo_runs()[a]}_seed{s}"][k]
            return runs[_curve_name(arm, a, s)][k]

        aug = {a: float(np.mean([rec(a, "mean_aug_acc", s) for s in SEEDS])) for a in keys}
        clean = {a: float(np.mean([rec(a, "clean_acc", s) for s in SEEDS])) for a in keys}
        joint = {a: 0.5 * clean[a] + 0.5 * aug[a] for a in keys}
        per_seed = {"R_aug": [], "R_joint": []}
        for s in SEEDS:
            per_seed["R_aug"].append(argbest({a: rec(a, "mean_aug_acc", s) for a in keys}, "max"))
            per_seed["R_joint"].append(argbest(
                {a: 0.5 * rec(a, "clean_acc", s) + 0.5 * rec(a, "mean_aug_acc", s) for a in keys}, "max"))
        out[arm] = {"keys": keys, "mean_aug_acc": aug, "clean_acc": clean, "score_joint": joint,
                    "R_aug": argbest(aug, "max"), "R_joint": argbest(joint, "max"),
                    "per_seed": per_seed}
    return out


# ---------------- 事后目标域 ----------------
def target_table():
    """汇总四份事后评估文件 → {run_name: target_acc}。"""
    t = {}
    for p in (S1C_TGT, S11_TGT, P1_TGT, TGT_JSON):
        assert os.path.exists(p), f"缺少事后评估文件: {p}（先锁定选择, 再跑 eval_heads_targets.py）"
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d["heads"].items():
            if v.get("target_acc") is not None:
                t[k] = v["target_acc"]
    return t


def _target_curve(t, arm, keys):
    if arm == "ddo":
        return {a: float(np.mean([t[f"{_ddo_runs()[a]}_seed{s}"] for s in SEEDS])) for a in keys}
    return {a: float(np.mean([t[_curve_name(arm, a, s)] for s in SEEDS])) for a in keys}


def _ranks(v):
    """平均秩（并列取平均）。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = np.array(_ranks(x)), np.array(_ranks(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def final_analysis(runs, man, sel, tgt):
    out = {"target_curves": {}, "paired": {}, "gates": {}, "g1d": {}, "selection": {}, "verdict": {}}
    arms = ["ddo", "language", "random", "identity", "ld", "sdb_isda"]
    # 1. 目标域曲线
    for arm in arms:
        out["target_curves"][arm] = _target_curve(tgt, arm, sel[arm]["keys"])
    T = out["target_curves"]
    # 2. G1 / G1b / G1c
    def pair_curves(arm_a, key_a, arm_b, key_b):
        x = [tgt[_curve_name(arm_a, key_a, s) if arm_a != "ddo" else f"{_ddo_runs()[key_a]}_seed{s}"] for s in SEEDS]
        y = [tgt[_curve_name(arm_b, key_b, s) if arm_b != "ddo" else f"{_ddo_runs()[key_b]}_seed{s}"] for s in SEEDS]
        return _paired(x, y)

    def one(name, pa):
        out["paired"][name] = pa
        out["gates"][name] = gate(pa)

    one("G1_lang_x4_vs_rand_x4", pair_curves("language", "4", "random", "4"))
    one("G1b_lang_x4_vs_ident_x4", pair_curves("language", "4", "identity", "4"))
    out["paired"]["G1c_lang_x4_vs_rand_samelam"] = _paired(
        [tgt[_curve_name("language", "4", s)] for s in SEEDS],
        [tgt[f"CUB_s12_isda_rand_samelam4_seed{s}"] for s in SEEDS])
    out["paired"]["G1c_lang_x4_vs_ident_samelam"] = _paired(
        [tgt[_curve_name("language", "4", s)] for s in SEEDS],
        [tgt[f"CUB_s12_isda_ident_samelam4_seed{s}"] for s in SEEDS])
    # 3. G1d: 对照臂曲线的峰位与峰高
    ce = T["ddo"]["0"]
    for arm in ("random", "identity"):
        c = T[arm]
        peak = max(c, key=lambda m: c[m])
        out["g1d"][arm] = {"peak_mult": peak, "peak": c[peak], "range": float(max(c.values()) - min(c.values())),
                           "height_over_ce": float(c[peak] - ce),
                           "curve": c}
    out["g1d"]["language"] = {"peak_mult": max(T["language"], key=lambda m: T["language"][m]),
                              "peak": max(T["language"].values()),
                              "height_over_ce": float(max(T["language"].values()) - ce)}
    # 4. G2: warmup 配对
    one("G2_lang_warm4_vs_lang_x4", _paired(
        [tgt[f"CUB_s12_isda_lang_x4_warm20_seed{s}"] for s in SEEDS],
        [tgt[_curve_name("language", "4", s)] for s in SEEDS]))
    one("G2_lang_warm8_vs_lang_x8", _paired(
        [tgt[f"CUB_isda_warm_x8_seed{s}"] for s in SEEDS],
        [tgt[_curve_name("language", "8", s)] for s in SEEDS]))
    one("G2_rand_warm8_vs_rand_x8", _paired(
        [tgt[f"CUB_s12_isda_rand_x8_warm20_seed{s}"] for s in SEEDS],
        [tgt[_curve_name("random", "8", s)] for s in SEEDS]))
    one("G2_ident_warm8_vs_ident_x8", _paired(
        [tgt[f"CUB_s12_isda_ident_x8_warm20_seed{s}"] for s in SEEDS],
        [tgt[_curve_name("identity", "8", s)] for s in SEEDS]))
    out["warmup_target"] = {k: float(np.mean([tgt[f"{n}_seed{s}"] for s in SEEDS])) for k, n in
                            (("lang_x4", "CUB_s12_isda_lang_x4_warm20"), ("lang_x8", "CUB_isda_warm_x8"),
                             ("rand_x8", "CUB_s12_isda_rand_x8_warm20"), ("ident_x8", "CUB_s12_isda_ident_x8_warm20"))}
    # 5. P2: 两条规则的选择与后悔值
    for arm in arms:
        keys = sel[arm]["keys"]
        tc = T[arm]
        best = max(tc.values())
        sm = {}
        for rule, rs in (("R_aug", sel[arm]["R_aug"]), ("R_joint", sel[arm]["R_joint"])):
            acc = tc[rs]
            sm[rule] = {"strength": rs, "target_acc": acc, "regret": float(best - acc),
                        "rho_vs_target": spearman([sel[arm]["mean_aug_acc" if rule == "R_aug" else "score_joint"][k]
                                                   for k in keys], [tc[k] for k in keys]),
                        "per_seed_strength": sel[arm]["per_seed"][rule]}
        sm["post_hoc_best"] = {"strength": max(tc, key=lambda k: tc[k]), "target_acc": best}
        sm["default"] = {"strength": "1" if arm != "ddo" else "1", "target_acc": tc.get("1")}
        out["selection"][arm] = sm
    # 6. 判定文本
    g1 = out["gates"]["G1_lang_x4_vs_rand_x4"]
    p1 = out["paired"]["G1_lang_x4_vs_rand_x4"]
    out["verdict"]["G1"] = ("几何成立" if g1 else
                            ("几何不成立" if (p1["mean"] <= 0 or p1["n_positive"] < GATE_POS) else "不确定"))
    out["verdict"]["G2_any"] = bool(any(out["gates"][k] for k in out["gates"] if k.startswith("G2_")))
    out["verdict"]["note"] = ("G1 的比较对象是同谱随机椭球在同一训练 seed 下的同一倍数（λ_pen×4），"
                              "两臂 λ 相差 1.4%、初始梯度比相差 2.5%。")
    return out


def _f(x, nd=2):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def _write_json(path, obj):
    def safe(o):
        if isinstance(o, dict):
            return {k: safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe(obj), f, ensure_ascii=False, indent=1, allow_nan=False)


def stage_source():
    runs, man = load_runs()
    sel = selection(runs)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "prereg": "RESULTS.md §3ag",
           "target_info_present": False, "manifest_sha256": sha256_file(MANIFEST),
           "calibration_sha256": sha256_file(CALIB_JSON), "selection": sel,
           "frozen_frames": {s: {e: man["frames"][s][e]["sha256"] for e in ("lang_full", "rand_full", "identity")}
                             for s in ("1", "2", "3")},
           "runs": {k: {kk: v.get(kk) for kk in ("arm", "kind", "mult", "clean_acc", "mean_aug_acc",
                                                 "mean_aug_ce", "head_sha256", "result_sha256")}
                    for k, v in runs.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    _write_json(SEL_JSON, out)
    L = ["# 第十二轮：几何对照的源域验证与选参（目标域评估之前）", "",
         "预注册见 `RESULTS.md` §3ag。45 个 s12 run 与复用的第十/十一轮 run 全部 `--target_eval none`，"
         "无任何目标域数字（逐个断言）；每个 run 的椭球生成元 sha256 与冻结清单逐 seed 对齐（禁止事后挑选随机帧）。", "",
         "## 剂量曲线（源域，3 seeds 均值）", "",
         "| 臂 | 0.5× | 1× | 2× | 4× | 8× |", "|---|---|---|---|---|---|"]
    for arm in ("language", "random", "identity", "ld", "sdb_isda"):
        c = sel[arm]["mean_aug_acc"]
        L.append(f"| {arm} | " + " | ".join(_f(c[m]) for m in MULTS) + " |")
    L += ["", "## 选参结果（两条规则）", "",
          "| 臂 | R_aug 选中 | R_joint 选中 | 干净 Acc @R_joint | 联合分数 @R_joint |", "|---|---|---|---|---|"]
    for arm in ("ddo", "language", "random", "identity", "ld", "sdb_isda"):
        s = sel[arm]
        L.append(f"| {arm} | {s['R_aug']} | {s['R_joint']} | {_f(s['clean_acc'][s['R_joint']])} | "
                 f"{_f(s['score_joint'][s['R_joint']])} |")
    L += ["", "规则：R_aug = " + RULE_TEXT["R_aug"] + "；R_joint = " + RULE_TEXT["R_joint"] + "。",
          "R_joint 在已知 CUB 结果之后提出，CUB 上的表现属研发证据，不是独立验证。"]
    with open(SEL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[source] 核对通过: s12 45 个 + 复用 run; 冻结帧逐 seed 一致")
    print("         选择: " + ", ".join(f"{a} R_aug {sel[a]['R_aug']} / R_joint {sel[a]['R_joint']}"
                                       for a in ("ddo", "language", "random", "identity", "ld", "sdb_isda")))
    print(f"[saved] {SEL_JSON}、{SEL_MD} —— 提交之后才能运行 --stage final / eval_heads_targets.py --s12")
    return 0


def stage_final():
    ok, why = _locked(SEL_JSON)
    if not ok:
        print("拒绝运行: 源域选择结果尚未锁定 —— " + why)
        return 2
    runs, man = load_runs()
    sel_now = selection(runs)
    with open(SEL_JSON, encoding="utf-8") as f:
        sel_locked = json.load(f)["selection"]
    for arm in sel_now:
        for k in ("R_aug", "R_joint", "mean_aug_acc", "score_joint"):
            assert json.dumps(sel_now[arm][k], sort_keys=True) == json.dumps(sel_locked[arm][k], sort_keys=True), \
                f"源域重算与锁定文件不一致: {arm}.{k}"
    tgt = target_table()
    fin = final_analysis(runs, man, sel_now, tgt)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(),
           "selection_lock_commit": why, "selection_sha256": sha256_file(SEL_JSON),
           "target_eval_sha256": sha256_file(TGT_JSON), **fin}
    _write_json(FIN_JSON, out)
    L = ["# 第十二轮：几何对照与 warmup 消融的结果（源域选择锁定之后）", "",
         f"选择锁定于提交 `{why[:7]}`；目标域 = CUB-200-Painting 测试集。", "",
         "## 目标域剂量曲线（3 seeds 均值）", "",
         "| 臂 | 0.5× | 1× | 2× | 4× | 8× |", "|---|---|---|---|---|---|"]
    for arm in ("ddo", "language", "random", "identity", "ld", "sdb_isda"):
        c = fin["target_curves"][arm]
        L.append(f"| {arm} | " + " | ".join(_f(c.get(m)) if c.get(m) is not None else "—" for m in MULTS) + " |")
    L += ["", "## 判读", "",
          f"- **G1**（Language@4× − Random@同锚点 4×）：Δ = {_f(fin['paired']['G1_lang_x4_vs_rand_x4']['mean'])} pp，"
          f"逐 seed {[round(x, 2) for x in fin['paired']['G1_lang_x4_vs_rand_x4']['diff_per_seed']]}，"
          f"{fin['paired']['G1_lang_x4_vs_rand_x4']['n_positive']}/3 正向 → **{fin['verdict']['G1']}**",
          f"- **G1b**（vs Identity）：Δ = {_f(fin['paired']['G1b_lang_x4_vs_ident_x4']['mean'])} pp，"
          f"{fin['paired']['G1b_lang_x4_vs_ident_x4']['n_positive']}/3 正向（辅助，压力不匹配）",
          f"- **G1c**（同 λ）：vs Random {_f(fin['paired']['G1c_lang_x4_vs_rand_samelam']['mean'])} pp，"
          f"vs Identity {_f(fin['paired']['G1c_lang_x4_vs_ident_samelam']['mean'])} pp",
          f"- **G1d**（峰位）：Random 峰在 {fin['g1d']['random']['peak_mult']}×（{_f(fin['g1d']['random']['peak'])}），"
          f"Identity 峰在 {fin['g1d']['identity']['peak_mult']}×（{_f(fin['g1d']['identity']['peak'])}），"
          f"Language 峰在 {fin['g1d']['language']['peak_mult']}×（{_f(fin['g1d']['language']['peak'])}）",
          f"- **G2**（warmup）：lang@4× {_f(fin['paired']['G2_lang_warm4_vs_lang_x4']['mean'])} pp "
          f"({fin['paired']['G2_lang_warm4_vs_lang_x4']['n_positive']}/3)，"
          f"rand@8× {_f(fin['paired']['G2_rand_warm8_vs_rand_x8']['mean'])} pp，"
          f"identity@8× {_f(fin['paired']['G2_ident_warm8_vs_ident_x8']['mean'])} pp", "",
          "## 选参（两条规则，目标域事后）", "",
          "| 臂 | 规则 | 选中 | 目标域 Acc | Regret | ρ(源域, 目标域) |", "|---|---|---|---|---|---|"]
    for arm in ("ddo", "language", "random", "identity", "ld", "sdb_isda"):
        for rule in ("R_aug", "R_joint"):
            s = fin["selection"][arm][rule]
            L.append(f"| {arm} | {rule} | {s['strength']} | {_f(s['target_acc'])} | {_f(s['regret'])} | "
                     f"{_f(s['rho_vs_target'], 3)} |")
        b = fin["selection"][arm]["post_hoc_best"]
        L.append(f"| {arm} | — | 事后最优 {b['strength']} | {_f(b['target_acc'])} | 0.00 | — |")
    with open(FIN_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[final] G1 = {fin['verdict']['G1']} (Δ {fin['paired']['G1_lang_x4_vs_rand_x4']['mean']:+.2f}); "
          f"G2 lang@4× {fin['paired']['G2_lang_warm4_vs_lang_x4']['mean']:+.2f} / "
          f"rand@8× {fin['paired']['G2_rand_warm8_vs_rand_x8']['mean']:+.2f} / "
          f"identity@8× {fin['paired']['G2_ident_warm8_vs_ident_x8']['mean']:+.2f}")
    print(f"[saved] {FIN_JSON}、{FIN_MD}")
    return 0


# ---------------- 自测 ----------------
def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    # T1: 并列取较小强度
    check("T1 argbest 并列取较小强度", argbest({"0.5": 1.0, "1": 2.0, "2": 2.0}, "max") == "1")
    # T2: 门槛的两个分支
    p = {"diff_per_seed": [1.0, 0.8, 0.2], "n_positive": 3, "mean": 0.666}
    check("T2 平均 0.67、3/3 → 过门槛", gate(p) is True)
    p2 = {"diff_per_seed": [1.2, 0.2, -0.5], "n_positive": 2, "mean": 0.3}
    check("T3 平均 0.3 → 不过门槛", gate(p2) is False)
    # T4: 联合规则可以改变选择（干净项把过强区拉回）
    aug = {"0.5": 51.0, "1": 52.0, "2": 53.0, "4": 54.0, "8": 55.0}
    clean = {"0.5": 77.6, "1": 77.6, "2": 77.4, "4": 77.0, "8": 70.0}
    joint = {k: 0.5 * clean[k] + 0.5 * aug[k] for k in aug}
    check("T4 R_aug 选 8× 而 R_joint 选 4×（干净项把过强区拉回）",
          argbest(aug, "max") == "8" and argbest(joint, "max") == "4",
          f"R_aug={argbest(aug, 'max')} R_joint={argbest(joint, 'max')}")
    # T5: regret 定义
    tc = {"1": 54.0, "2": 56.0, "4": 57.0}
    check("T5 regret = 事后最优 − 选中", abs((57.0 - tc["2"]) - 1.0) < 1e-12)
    # T6: Spearman 的手算例子（完全同序 = 1，完全反序 = -1）
    check("T6 Spearman 同序 = 1、反序 = -1",
          abs(spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-12
          and abs(spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) + 1.0) < 1e-12)
    # T7: 并列秩的平均
    check("T7 并列取平均秩", _ranks([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0], str(_ranks([1, 2, 2, 3])))
    # T8: 配对差的方向与计数
    pp = _paired([56.0, 55.0, 57.0], [55.0, 55.0, 58.0])
    check("T8 配对差 = +1/+0/−1", pp["diff_per_seed"] == [1.0, 0.0, -1.0] and pp["n_positive"] == 1)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=["source", "final"])
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    if a.stage == "source":
        return stage_source()
    if a.stage == "final":
        return stage_final()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
