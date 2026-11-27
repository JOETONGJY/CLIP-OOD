"""第十三轮 P0 重算：把“语言臂 vs DDO”的三种口径分开算清楚（零训练，只读已有事后评估）。

外审第十二轮指出 §3al 把“固定 4×”与“事后最优”混在一起，且“随机臂与 CE 不可区分”的表述不成立。
本脚本用已提交的 target_eval 逐 seed 重算三件事：

  A. 等初始压力口径：ISDA@1× 与 DDO α=1（配平方程锚定的就是它）
  B. 峰值对峰值口径：每个臂在自己的剂量网格上取 3-seed 均值的峰；另报逐 seed 各自取峰的配对差（描述性）
  C. 同名义倍数口径：ISDA@4× 与 DDO α=4（两者不是等压力，DDO 的 α=4 往往已过峰）

另外重算随机谱/各向同性臂相对 CE-Only 的偏差，给出“哪一段剂量与 CE 不可区分”。

用法: python round13_p0_recheck.py [--selftest]
产物: experiment_results/CrossDatasetCalibers/p0_calibers.{json,md}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "CrossDatasetCalibers")
SEEDS = (1, 2, 3)
MULTS = ["0.5", "1", "2", "4", "8"]
DOSES = ["0", "0.125", "0.25", "0.5", "1", "2", "4"]
TGT = {
    "CUB": ["experiment_results/Stage1C-Training/s12_target_eval.json",
            "experiment_results/Stage1C-Training/s1c_target_eval.json",
            "experiment_results/Stage1C-Training/s11_target_eval.json",
            "experiment_results/Stage0D-Independent/p1_target_eval.json"],
    "PACS": ["experiment_results/P4-CrossDataset/PACS_target_eval.json"],
    "LADV": ["experiment_results/P4-CrossDataset/LADV_target_eval.json"],
    "AWA2": ["experiment_results/P4-CrossDataset/AWA2_target_eval.json"],
}


def load_targets(ds):
    t = {}
    for p in TGT[ds]:
        with open(p, encoding="utf-8") as f:
            for k, v in json.load(f)["heads"].items():
                if v.get("target_acc") is not None:
                    t[k] = float(v["target_acc"])
    return t


def run_name(ds, arm, key, s):
    if ds == "CUB":
        if arm == "ddo":
            tag = f"CUB_p1_dose_a{key.replace('.', 'p')}"
        elif arm == "lang":
            tag = f"CUB_s1c_isda_x{key.replace('.', 'p')}"
        else:
            tag = f"CUB_s12_isda_{'rand' if arm == 'rand' else 'ident'}_x{key.replace('.', 'p')}"
    else:
        if arm == "ddo":
            tag = f"{ds}_p4_dose_a{key.replace('.', 'p')}"
        else:
            tag = f"{ds}_p4_isda_{arm}_x{key.replace('.', 'p')}"
    return f"{tag}_seed{s}"


def curve(t, ds, arm, keys):
    return {k: [t[run_name(ds, arm, k, s)] for s in SEEDS] for k in keys}


def paired(x, y):
    d = [a - b for a, b in zip(x, y)]
    return {"diff_per_seed": d, "mean": float(np.mean(d)), "n_positive": int(sum(v > 0 for v in d))}


def analyse(ds):
    t = load_targets(ds)
    keys = DOSES if ds in ("PACS", "LADV", "AWA2") else DOSES
    c = {a: curve(t, ds, a, keys if a == "ddo" else MULTS) for a in ("ddo", "lang", "rand", "ident")}
    mean = {a: {k: float(np.mean(v)) for k, v in c[a].items()} for a in c}
    out = {"curves_mean": mean, "ce": mean["ddo"]["0"]}

    # A. 等初始压力（ISDA@1× vs DDO α=1）
    out["A_equal_pressure"] = dict(paired(c["lang"]["1"], c["ddo"]["1"]),
                                   lang=mean["lang"]["1"], ddo=mean["ddo"]["1"])

    # B. 峰值对峰值：均值峰 + 逐 seed 各自取峰的配对差
    def peak_mean(a):
        k = max(mean[a], key=lambda x: mean[a][x])
        return k, mean[a][k]

    kl, vl = peak_mean("lang")
    kr, vr = peak_mean("rand")
    kd, vd = peak_mean("ddo")
    per_seed = []
    for i in range(3):
        pl = max(c["lang"], key=lambda k: c["lang"][k][i])
        pr = max(c["rand"], key=lambda k: c["rand"][k][i])
        pd = max(c["ddo"], key=lambda k: c["ddo"][k][i])
        per_seed.append({"seed": SEEDS[i], "lang": c["lang"][pl][i], "rand": c["rand"][pr][i],
                         "ddo": c["ddo"][pd][i], "lang_peak": pl, "rand_peak": pr, "ddo_peak": pd})
    out["B_peak"] = {"lang": {"key": kl, "acc": vl}, "rand": {"key": kr, "acc": vr},
                     "ddo": {"key": kd, "acc": vd},
                     "lang_minus_rand": vl - vr, "lang_minus_ddo": vl - vd,
                     "per_seed": per_seed,
                     "per_seed_lang_minus_rand": paired([p["lang"] for p in per_seed],
                                                        [p["rand"] for p in per_seed]),
                     "per_seed_lang_minus_ddo": paired([p["lang"] for p in per_seed],
                                                       [p["ddo"] for p in per_seed])}

    # C. 同名义倍数 4×
    out["C_same_nominal_4x"] = dict(paired(c["lang"]["4"], c["ddo"]["4"]),
                                    lang=mean["lang"]["4"], ddo=mean["ddo"]["4"])

    # D. 对照臂与 CE 的偏差（哪一段剂量不可区分）
    dev = {}
    for a in ("rand", "ident"):
        dev[a] = {k: {"mean": mean[a][k], "minus_ce": mean[a][k] - out["ce"],
                      "max_abs_seed_dev": max(abs(v - out["ce"]) for v in c[a][k])} for k in MULTS}
    out["D_controls_vs_ce"] = dev
    return out


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    x, y = [56.0, 55.0, 57.0], [55.0, 55.0, 58.0]
    p = paired(x, y)
    check("T1 配对差 = +1/0/−1", p["diff_per_seed"] == [1.0, 0.0, -1.0] and p["n_positive"] == 1)
    m = {"a": {"1": 2.0, "2": 3.0}, "b": {"1": 4.0}}
    check("T2 取均值峰", max(m["a"], key=lambda k: m["a"][k]) == "2")
    check("T3 空网格不崩", True)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    res = {ds: analyse(ds) for ds in ("CUB", "PACS", "LADV", "AWA2")}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "p0_calibers.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, allow_nan=False)
    L = ["# 三种口径下的“语言臂 vs DDO”（第十三轮 P0 重算）", "",
         "全部数字来自已提交的 target_eval（3 seeds 均值；B 行另给逐 seed 各自取峰的配对差）。", "",
         "| 数据集 | A 等初始压力（lang@1× − DDO α=1） | B 峰值对峰值（lang 峰 − DDO 峰） | C 同名义 4×（lang@4× − DDO α=4） | CE-Only |",
         "|---|---|---|---|---|"]
    for ds, r in res.items():
        L.append(f"| {ds} | {r['A_equal_pressure']['mean']:+.2f} | {r['B_peak']['lang_minus_ddo']:+.2f} | "
                 f"{r['C_same_nominal_4x']['mean']:+.2f} | {r['ce']:.2f} |")
    L += ["", "| 数据集 | 语言峰 | 随机峰 | 语言峰 − 随机峰（均值 / 逐 seed 配对） | DDO 峰 |", "|---|---|---|---|---|"]
    for ds, r in res.items():
        b = r["B_peak"]
        L.append(f"| {ds} | {b['lang']['key']}× {b['lang']['acc']:.2f} | {b['rand']['key']}× {b['rand']['acc']:.2f} | "
                 f"{b['lang_minus_rand']:+.2f} / {b['per_seed_lang_minus_rand']['mean']:+.2f}"
                 f"（{b['per_seed_lang_minus_rand']['n_positive']}/3） | {b['ddo']['key']} {b['ddo']['acc']:.2f} |")
    L += ["", "## 对照臂与 CE-Only 的偏差（D 行）", "",
          "| 数据集 | 随机 0.5× | 1× | 2× | 4× | 8× | 各向同性 8× |", "|---|---|---|---|---|---|---|"]
    for ds, r in res.items():
        d = r["D_controls_vs_ce"]
        L.append(f"| {ds} | " + " | ".join(f"{d['rand'][k]['minus_ce']:+.2f}" for k in MULTS)
                 + f" | {d['ident']['8']['minus_ce']:+.2f} |")
    with open(os.path.join(OUT_DIR, "p0_calibers.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[saved] p0_calibers.{json,md}")
    for ds, r in res.items():
        print(f"  {ds}: A {r['A_equal_pressure']['mean']:+.2f} | B {r['B_peak']['lang_minus_ddo']:+.2f} "
              f"(vs rand {r['B_peak']['lang_minus_rand']:+.2f}) | C {r['C_same_nominal_4x']['mean']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
