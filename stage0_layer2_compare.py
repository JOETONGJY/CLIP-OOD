"""对比 stage0_layer2_cub.json 的两个版本 (外审第八轮 P0 收尾: 证明重构没有改变计算)。

背景: stage0_layer2_diagnose.py 在 5e9f1ff 之后做了重构 (逐样本循环 -> 向量化累加), 并把回归的控制变量
从未投影范数 ‖w‖ 换成投影范数 ‖Hw‖ (未投影口径保留为敏感性分析, 键名带 rawnorm)。本脚本核对:
  C1 0D 的全部数字 (干净 / 各扰动 / 均值的 Acc 与 CE) 逐 seed 完全相同
  C2 新版的未投影口径 == 旧版主口径 (逐 seed 逐口径, 容差 1e-9) —— 重构与向量化没有改变计算
  C3 新版主口径 (投影范数) 对旧版主口径的差异 (只报告)
  C4 判定标签新旧对照; 换控制变量后 ΔR² 与系数的符号全部不变
每个版本可以给 git 提交 (从仓库对象读取), 也可以给 JSON 文件路径 (没有仓库时用)。
用法:
  python stage0_layer2_compare.py 5e9f1ff                 # 旧版取自提交, 新版取自工作区
  python stage0_layer2_compare.py <旧.json> <新.json>     # 两个都是文件
  python stage0_layer2_compare.py --selftest
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REL = "experiment_results/Stage0-Diagnostics/stage0_layer2_cub.json"
SCHEMES = ("fixed", "own_avg", "own_sample")
OD_KEYS = ("clean_acc", "clean_ce", "avg_aug_acc", "avg_aug_ce")
TOL = 1e-9


def load(spec, repo):
    """spec 是已存在的文件路径则读文件; 否则当作 git 提交, 从仓库对象读取 (不经工作区)。返回 (dict, 来源说明)"""
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            return json.load(f), spec
    r = subprocess.run(["git", "-C", repo, "show", spec + ":" + REL], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("读不到版本 %s: %s" % (spec, r.stderr.decode("utf-8", "replace").strip()))
    return json.loads(r.stdout.decode("utf-8")), "%s:%s" % (spec, REL)


def compare(old, new, verbose=True):
    """返回 (是否全部通过, 结果 dict)。verbose=False 时只返回结果, 用于自测。"""
    res = {"c0": sorted(old["arms"]) == sorted(new["arms"]), "c1": 0.0, "n_od": 0, "c2": 0.0, "n_c2": 0, "c2b": True,
           "c4": True, "rows": []}
    say = print if verbose else (lambda *a, **k: None)
    ok = [res["c0"]]
    if not res["c0"]:
        say("  BAD  C0 两版包含相同的臂")
        return False, res
    tags = sorted(new["arms"])
    for t in tags:
        for po, pn in zip(old["arms"][t]["per_seed"], new["arms"][t]["per_seed"]):
            assert po["seed"] == pn["seed"], "两版的 seed 顺序不同"
            for k in OD_KEYS:
                res["c1"] = max(res["c1"], abs(po[k] - pn[k]))
                res["n_od"] += 1
            for k in ("aug_accs", "aug_ces"):
                assert sorted(po[k]) == sorted(pn[k])
                for an in po[k]:
                    res["c1"] = max(res["c1"], abs(po[k][an] - pn[k][an]))
                    res["n_od"] += 1
            for sch in SCHEMES:
                row = (t, po["seed"], sch, po["delta_r2_" + sch], pn["delta_r2_rawnorm_" + sch], pn["delta_r2_" + sch],
                       po["pcr_coef_" + sch], pn["pcr_coef_rawnorm_" + sch], pn["pcr_coef_" + sch])
                res["rows"].append(row)
                res["c2"] = max(res["c2"], abs(row[3] - row[4]), abs(row[6] - row[7]))
                res["n_c2"] += 1
                res["c2b"] &= po["n_pairs_" + sch] == pn["n_pairs_" + sch]
    ok.append(res["c1"] == 0.0)
    ok.append(res["c2"] < TOL)
    ok.append(res["c2b"])
    res["c4"] = all((r[3] > 0) == (r[5] > 0) and (r[6] < 0) == (r[8] < 0) for r in res["rows"])
    res["biggest"] = max(max(abs(r[5] - r[3]), abs(r[8] - r[6])) for r in res["rows"])
    if not verbose:
        return all(ok) and res["c4"], res

    def mark(name, cond, detail=""):
        say(("  OK   " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))
    mark("C0 两版包含相同的臂", res["c0"], "%d 个" % len(tags))
    mark("C1 0D 的全部数字逐 seed 完全相同 (%d 个数)" % res["n_od"], res["c1"] == 0.0, "最大差 %g" % res["c1"])
    mark("C2 新版未投影口径 == 旧版主口径 (%d 组 ΔR² 与系数, 逐 seed 逐口径)" % res["n_c2"], res["c2"] < TOL,
         "最大差 %.2e" % res["c2"])
    mark("C2b 配对数逐 seed 相同", res["c2b"])
    say("\n  %-9s %-5s %-11s %14s %14s %9s | %9s %9s %9s" % (
        "臂", "seed", "口径", "ΔR² 旧=未投影", "ΔR² 新=投影", "差", "系数 旧", "系数 新", "差"))
    for t, s, sch, o_d, r_d, p_d, o_c, r_c, p_c in res["rows"]:
        if sch == "fixed" or t in ("ce_a0", "full_a1", "full_a2"):
            say("  %-9s %-5d %-11s %14.5f %14.5f %+9.5f | %+9.4f %+9.4f %+9.4f" % (t, s, sch, o_d, p_d, p_d - o_d, o_c, p_c, p_c - o_c))
    say("\n  控制变量换成投影范数后, 全部 %d 组中 ΔR² 与系数的最大变化: %.4f" % (len(res["rows"]), res["biggest"]))

    def mean_of(src, t, key):
        v = [p[key] for p in src["arms"][t]["per_seed"]]
        return sum(v) / len(v)
    say("\n  %-22s %12s %12s %14s | %9s %9s" % ("臂", "ΔR² 旧(固定)", "ΔR² 新(固定)", "ΔR² 新(逐样本)", "系数 旧", "系数 新"))
    for t in tags:
        say("  %-22s %12.4f %12.4f %14.4f | %+9.3f %+9.3f" % (
            t, mean_of(old, t, "delta_r2_fixed"), mean_of(new, t, "delta_r2_fixed"), mean_of(new, t, "delta_r2_own_sample"),
            mean_of(old, t, "pcr_coef_fixed"), mean_of(new, t, "pcr_coef_fixed")))
    say("\n  旧版 0B 判定:", old.get("verdict_0b_layer2", {}).get("verdict"))
    say("  新版 0B 判定:", new.get("verdict_0b_layer2", {}).get("label"), "| 敏感性口径:", new.get("verdict_0b_layer2_sensitivity"))
    say("  旧版 0D 判定:", old.get("verdict_0d", {}).get("acc_optimal"), old.get("verdict_0d", {}).get("ce_optimal"),
        "| 新版:", new.get("verdict_0d", {}).get("acc_optimal"), new.get("verdict_0d", {}).get("ce_optimal"))
    mark("C4 换控制变量后 ΔR² 的符号与系数的符号全部不变", res["c4"])
    return all(ok) and res["c4"], res


def self_test():
    """合成一对版本: 未投影口径与旧版相同、投影口径略有不同但同号 → 通过; 改动一个 0D 数字 → C1 失败。"""
    old = {"arms": {}, "verdict_0b_layer2": {"verdict": "x"}, "verdict_0d": {}}
    for i, t in enumerate(("ce_a0", "full_a1")):
        per = []
        for s in (1, 2):
            p = {"seed": s, "clean_acc": 0.8 + 0.01 * i, "clean_ce": 0.5, "avg_aug_acc": 0.6, "avg_aug_ce": 2.0,
                 "aug_accs": {"a": 0.5, "b": 0.7}, "aug_ces": {"a": 2.0, "b": 2.2}}
            for sch in SCHEMES:                                  # 三个口径给不同的值
                p["delta_r2_" + sch] = 0.01 * (SCHEMES.index(sch) + 1)
                p["pcr_coef_" + sch] = 0.1 * (SCHEMES.index(sch) + 1)
                p["n_pairs_" + sch] = 10
            per.append(p)
        old["arms"][t] = {"per_seed": per}
    new = json.loads(json.dumps(old))
    for t in new["arms"]:
        for p in new["arms"][t]["per_seed"]:
            for sch in SCHEMES:
                p["delta_r2_rawnorm_" + sch] = p["delta_r2_" + sch]
                p["pcr_coef_rawnorm_" + sch] = p["pcr_coef_" + sch]
                p["delta_r2_" + sch] += 1e-4                      # 投影口径略有不同, 同号
                p["pcr_coef_" + sch] += 1e-3
    ok1, r1 = compare(old, new, verbose=False)
    bad = json.loads(json.dumps(new))
    bad["arms"]["full_a1"]["per_seed"][1]["aug_accs"]["a"] += 0.01
    ok2, r2 = compare(old, bad, verbose=False)
    bad2 = json.loads(json.dumps(new))
    bad2["arms"]["ce_a0"]["per_seed"][0]["delta_r2_rawnorm_fixed"] += 1e-3
    ok3, r3 = compare(old, bad2, verbose=False)
    good = ok1 and r1["c1"] == 0.0 and r1["c2"] == 0.0 and (not ok2) and (not ok3)
    print(("  OK  " if good else "  BAD  ") + "自测: 一致的版本对通过; 0D 数字被改 → 不通过; 未投影口径被改 → 不通过",
          "[%s / %s / %s]" % (ok1, ok2, ok3))
    print("SELFTEST:", "PASS" if good else "FAIL")
    return 0 if good else 1


def main():
    argv = sys.argv[1:]
    if "--selftest" in argv:
        return self_test()
    repo = argv[argv.index("--repo") + 1] if "--repo" in argv else "."
    specs = [a for a in argv if not a.startswith("--") and a != repo] or ["HEAD"]
    old, src_old = load(specs[0], repo)
    if len(specs) > 1:
        new, src_new = load(specs[1], repo)
    elif os.path.isfile(os.path.join(repo, REL)):
        with open(os.path.join(repo, REL), encoding="utf-8") as f:
            new, src_new = json.load(f), os.path.join(repo, REL) + " (工作区)"
    else:
        new, src_new = load("HEAD", repo)
    print("旧版:", src_old, "\n新版:", src_new)
    ok, _ = compare(old, new)
    print("\nCOMPARE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
