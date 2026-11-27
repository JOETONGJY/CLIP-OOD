"""用已保存的分类头做事后目标域评估 (P1, 2026-11-26, 外审第九轮; 预注册见 RESULTS.md §3v)。

为什么需要它: P1 的 21 个 run 以 --target_eval none 训练 —— 训练期间不读目标域特征缓存, 结果 json 里
没有任何目标域数字。源域选择规则与选择结果锁定 (提交) 之后, 才用 --save_head 保存的分类头把目标域精度算出来。
"选择在先、目标域评估在后"的先后顺序由 git 提交记录保证: 评估 P1 的分类头之前, 本脚本要求
experiment_results/Stage0D-Independent/p1_source_selection.json 已被 git 跟踪且没有未提交的修改, 否则拒绝运行。

评估走的就是训练管线自己的 TrainingSession._evaluate: 用结果 json 里记录的 run_config 重建与训练时同一个模型
(同 args => 同概念矩阵 E、同 clip_cbm_orth 结构), 把分类头参数换成保存的那一份, 在
cache/<ds>_target_test.pt 上评估。不另写一套前向计算。

等价性验证 (--validate_stage0, 与 P1 无关, 可以在任何时候运行): Stage 0 的 27 个分类头是 final_only 协议
训练的, 它们的结果 json 里记着训练结束时管线自己算出的目标域 Acc / CE。用本脚本重算必须完全相同 ——
否则说明"分类头 + 缓存 + 重建模型"不能复现管线内的评估, 事后评估的结果就不可信。

用法:
  python eval_heads_targets.py --validate_stage0                    # 等价性验证 (27 个 Stage 0 分类头)
  python eval_heads_targets.py [--seeds 1 2 3]                      # P1: 矩阵 p1 的 7 个剂量 (先检查选择已锁定)
  python eval_heads_targets.py --stage1c [--seeds 1 2 3]            # Stage 1C 八臂 (预注册 §3z; 门是 s1c 的选择文件)
  python eval_heads_targets.py --s11 [--seeds 1 2 3]                # 第十一轮 (预注册 §3ad; 门是 s11 的选择文件)
  python eval_heads_targets.py --s12 [--seeds 1 2 3]                # 第十二轮 (预注册 §3ag; 门是 s12 的选择文件)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from main_cached import CachedTrainingSession, FeatureDataset  # noqa: E402
from run_matrix import DOSE  # noqa: E402
from run_status import git_info  # noqa: E402

OUT_DIR = os.path.join("experiment_results", "Stage0D-Independent")
SELECT_JSON = os.path.join(OUT_DIR, "p1_source_selection.json")
SEL1C_JSON = os.path.join("experiment_results", "Stage1C-Training", "s1c_source_selection.json")
SEL11_JSON = os.path.join("experiment_results", "Stage1C-Training", "s11_source_selection.json")
SEL12_JSON = os.path.join("experiment_results", "Stage1C-Training", "s12_source_selection.json")
# 不影响模型结构的配置项: 这些不同也共用同一个重建的模型 (只换分类头)
KEY_SKIP = {"run_tag", "seed", "alpha", "epochs", "lr", "weight_decay", "target_eval", "save_head", "grad_diag",
            "save_model", "overwrite_run", "train_subset_cache", "source_val_cache", "device", "wandb"}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def selection_is_locked(path=SELECT_JSON):
    """源域选择结果是否已锁定: 文件存在、被 git 跟踪、且相对 HEAD 没有未提交的修改。返回 (是否锁定, 说明)。"""
    if not os.path.exists(path):
        return False, f"选择结果文件不存在: {path}"
    rel = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__))).replace(os.sep, "/")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], capture_output=True)
    if tracked.returncode != 0:
        return False, f"{rel} 还没有被 git 跟踪 (未提交)"
    st = subprocess.run(["git", "status", "--porcelain", "--", rel], capture_output=True).stdout.decode("utf-8").strip()
    if st:
        return False, f"{rel} 有未提交的修改: {st.strip()}"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.decode().strip()
    return True, head


class _HeadEvalSession(CachedTrainingSession):
    """只重建数据 loader 与模型 (不建优化器); 目标域 loader 总是实际读取, 与 run_config 里 target_eval 的取值无关。"""
    def _init_components(self):
        self._prepare_datasets()
        if self.target_test_loader is None:                       # run_config 里是 --target_eval none
            self.target_test_loader = DataLoader(
                FeatureDataset(os.path.join("cache", f"{self.args.dataset}_target_test.pt")),
                batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        self._init_model()
        self._init_loss_fns()


def model_key(cfg):
    return json.dumps(sorted((k, str(v)) for k, v in cfg.items() if k not in KEY_SKIP), ensure_ascii=False)


class HeadEvaluator:
    """按 run_config 重建模型 (结构相同的 run 共用), 逐个换上保存的分类头再评估。"""
    def __init__(self):
        self._sessions, self.target_meta = {}, None

    def session(self, cfg):
        key = model_key(cfg)
        if key not in self._sessions:
            torch.manual_seed(int(cfg.get("seed", 0)))
            # attach 的缓存键含 CLIP 身份, 身份从命令行取 —— 评估进程的 argv 没有 --CLIP_type,
            # 会让键少一段 (值相同、键不同)。临时补上, 使评估与训练用同一个缓存文件。
            argv0 = sys.argv[:]
            if "--CLIP_type" not in sys.argv and cfg.get("CLIP_type"):
                sys.argv += ["--CLIP_type", str(cfg["CLIP_type"])]
            try:
                s = _HeadEvalSession(argparse_namespace(cfg))
            finally:
                sys.argv[:] = argv0
            if self.target_meta is None:
                p = os.path.join("cache", f"{cfg['dataset']}_target_test.pt")
                self.target_meta = {"path": p, "sha256": sha256_file(p), "n": len(s.target_test_loader.dataset)}
            self._sessions[key] = s
        return self._sessions[key]

    def evaluate(self, head_path, result_path=None):
        head = torch.load(head_path, map_location="cpu", weights_only=False)
        s = self.session(head["run_config"])
        s.model.classifier.load_state_dict(head["classifier"], strict=True)
        t = s._evaluate(s.target_test_loader, "target")
        src = s._evaluate(s.source_test_loader, "source")
        out = {"run_name": head["run_name"], "alpha": head["run_config"].get("alpha"), "seed": head["run_config"].get("seed"),
               "head_sha256": sha256_file(head_path), "head_git": head.get("git"),
               "target_acc": t["target_acc"], "target_ce": t["target_cls_loss"],
               "source_test_acc": src["source_acc"], "source_test_ce": src["source_cls_loss"]}
        if result_path and os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                r = json.load(f)
            out["result_sha256"] = sha256_file(result_path)
            out["recorded"] = {k: (r.get("final_diag") or {}).get(k) for k in ("target_acc", "target_ce", "source_acc", "source_ce")}
        return out


def argparse_namespace(cfg):
    import argparse
    return argparse.Namespace(**dict(cfg))


def validate_stage0(seeds):
    """27 个 Stage 0 分类头: 本脚本重算的目标域 / 源测试集 Acc 与 CE 必须与结果 json 里管线自己记录的完全相同。"""
    from run_matrix import MATRICES
    ev, ok = HeadEvaluator(), True
    n, n_exact, worst = 0, 0, 0.0
    for ds, tag, _ in MATRICES["stage0"][1]:
        for s in seeds:
            run = f"{ds}_{tag}_seed{s}"
            hp, rp = f"logs/heads/{run}.pt", f"logs/results/{run}.json"
            if not (os.path.exists(hp) and os.path.exists(rp)):
                print(f"  BAD  {run}: 分类头或结果 json 不存在")
                ok = False
                continue
            o = ev.evaluate(hp, rp)
            rec = o["recorded"]
            diffs = {k: abs(o[a] - rec[k]) for k, a in (("target_acc", "target_acc"), ("target_ce", "target_ce"),
                                                        ("source_acc", "source_test_acc"), ("source_ce", "source_test_ce"))}
            exact = all(v == 0.0 for v in diffs.values())
            n += 1
            n_exact += exact
            worst = max([worst] + list(diffs.values()))
            ok &= exact
            print(f"  {'OK  ' if exact else 'BAD '} {run}: 目标域 {o['target_acc']:.4f}% (记录 {rec['target_acc']:.4f}%) | "
                  f"最大差 {max(diffs.values()):.3e}")
    print(f"等价性验证: {n_exact} / {n} 个分类头逐位相同 (最大绝对差 {worst:.3e})")
    print("VALIDATE:", "PASS" if ok and n > 0 else "FAIL")
    return 0 if ok and n > 0 else 1


def eval_stage1c(seeds):
    """Stage 1C 八臂 (预注册 RESULTS §3z) 的事后目标域评估: 与 P1 同一套流程, 门是 s1c 的源域选择文件。
    CE-Only / DDO α=1 / α=2 三个基线不在这里重评 —— 复用 P1 的事后评估 (p1_target_eval.json)。"""
    from run_matrix import MATRICES
    locked, why = selection_is_locked(SEL1C_JSON)
    if not locked:
        print("拒绝运行: Stage 1C 的源域选择结果尚未锁定 —— " + why)
        print("必须先运行 stage1c_training_report.py --stage source 并提交其输出, 再做事后目标域评估。")
        return 2
    ev = HeadEvaluator()
    heads = {}
    for ds, tag, _ in MATRICES["stage1c"][1]:
        for s in seeds:
            run = f"{ds}_{tag}_seed{s}"
            heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(SEL1C_JSON, encoding="utf-8") as f:
        sel = json.load(f)

    def same(r):
        v, w = heads[r], sel["runs"].get(r, {})
        return bool(w) and w.get("head_sha256") == v["head_sha256"] and w.get("result_sha256") == v["result_sha256"]

    mismatched = [r for r in sorted(set(heads) | set(sel["runs"])) if r not in heads or not same(r)]
    if mismatched:
        print("拒绝: 以下 run 的分类头或结果 json 与源域选择锁定时的不一致 (或数量不同): " + ", ".join(mismatched))
        return 3
    os.makedirs(os.path.dirname(SEL1C_JSON), exist_ok=True)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": sha256_file(SEL1C_JSON), "target_cache": ev.target_meta, "heads": heads}
    out_path = os.path.join(os.path.dirname(SEL1C_JSON), "s1c_target_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"选择锁定于提交 {why[:7]}; 评估了 {len(heads)} 个分类头; 目标域 {ev.target_meta['n']} 个样本")
    for ds, tag, _ in MATRICES["stage1c"][1]:
        accs = [heads[f"{ds}_{tag}_seed{s}"]["target_acc"] for s in seeds]
        print(f"  {tag:22s} 目标域 Acc: " + " / ".join(f"{x:.2f}" for x in accs))
    print(f"[saved] {out_path}")
    return 0


def _extra_verifiable(run, sel):
    """锁后新增的 run (P3 warmup): 其分类头 sha256 必须能用锁定时已知的规则核对 —— 这里核对它引用的
    标定文件与词库身份已在锁定清单的同臂 run 里出现过 (同 λ 由 run_matrix 的导入自检钉住)。"""
    return run.startswith("CUB_isda_warm_")


def eval_s11(seeds):
    """第十一轮 (预注册 §3ad) 的事后目标域评估: 门是 s11 的源域选择文件; 覆盖 s1c_ext + sdb22
    (+ s11_warmup 矩阵, 在 P3 的 warmup 矩阵注册之后重跑本命令即可补评)。"""
    from run_matrix import MATRICES
    locked, why = selection_is_locked(SEL11_JSON)
    if not locked:
        print("拒绝运行: 第十一轮的源域选择结果尚未锁定 —— " + why)
        print("必须先运行 round11_report.py --stage source 并提交其输出, 再做事后目标域评估。")
        return 2
    ev = HeadEvaluator()
    heads = {}
    mats = [("s1c_ext", MATRICES["s1c_ext"][1]), ("sdb22", MATRICES["sdb22"][1])]
    if "s11_warmup" in MATRICES:
        mats.append(("s11_warmup", MATRICES["s11_warmup"][1]))
    for mname, entries in mats:
        for ds, tag, _ in entries:
            for s in seeds:
                run = f"{ds}_{tag}_seed{s}"
                heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(SEL11_JSON, encoding="utf-8") as f:
        sel = json.load(f)

    def same(r):
        v, w = heads.get(r), sel["runs"].get(r, {})
        return bool(w) and v is not None and w.get("head_sha256") == v["head_sha256"]             and w.get("result_sha256") == v["result_sha256"]

    # 只核对"锁定清单内的"已评估 run (清单还含 stage1c 的 57 个出处记录, 本命令不评估它们);
    # 锁定之后新增的 run (P3 的 warmup, §3ad 预注册) 允许出现在 extra, 数值逐个核对 sha256
    missing = [r for r in sorted(heads) if r in sel["runs"] and not same(r)]
    bad_extra = [r for r in sorted(heads) if r not in sel["runs"] and not _extra_verifiable(r, sel)]
    extra = [r for r in sorted(heads) if r not in sel["runs"]]
    not_evaluated = [r for r in sorted(sel["runs"]) if r not in heads]
    if missing or bad_extra:
        print("拒绝: 已评估 run 与锁定不一致: " + ", ".join((missing + bad_extra)[:8]))
        return 3
    os.makedirs(os.path.dirname(SEL11_JSON), exist_ok=True)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": sha256_file(SEL11_JSON), "target_cache": ev.target_meta, "heads": heads,
           "note": "extra(不在锁定清单) = " + (", ".join(extra) if extra else "无")
                   + "; 未评估(锁定清单内的 stage1c 旧 run) = " + str(len(not_evaluated)) + " 个"}
    out_path = os.path.join(os.path.dirname(SEL11_JSON), "s11_target_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"选择锁定于提交 {why[:7]}; 评估了 {len(heads)} 个分类头; 目标域 {ev.target_meta['n']} 个样本")
    for mname, entries in mats:
        accs = {f"{ds}_{tag}": [heads[f"{ds}_{tag}_seed{s}"]["target_acc"] for s in seeds] for ds, tag, _ in entries}
        for k, v in accs.items():
            print(f"  {k:24s} " + " / ".join(f"{x:.2f}" for x in v))
    print(f"[saved] {out_path}")
    return 0


def eval_s12(seeds):
    """第十二轮（预注册 §3ag）的事后目标域评估：门是 s12 的源域选择文件；
    覆盖 s12_geom 的全部 45 个分类头（几何对照 36 + 同 λ 6 + warmup 3）。"""
    from run_matrix import MATRICES
    locked, why = selection_is_locked(SEL12_JSON)
    if not locked:
        print("拒绝运行: 第十二轮的源域选择结果尚未锁定 —— " + why)
        print("必须先运行 round12_report.py --stage source 并提交其输出, 再做事后目标域评估。")
        return 2
    ev = HeadEvaluator()
    heads = {}
    entries = MATRICES["s12_geom"][1]
    for ds, tag, _ in entries:
        for s in seeds:
            run = f"{ds}_{tag}_seed{s}"
            heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(SEL12_JSON, encoding="utf-8") as f:
        sel = json.load(f)

    def same(r):
        v, w = heads.get(r), sel["runs"].get(r, {})
        return bool(w) and v is not None and w.get("head_sha256") == v["head_sha256"] \
            and w.get("result_sha256") == v["result_sha256"]

    missing = [r for r in sorted(heads) if not same(r)]
    if missing:
        print("拒绝: 已评估 run 与锁定不一致: " + ", ".join(missing[:8]))
        return 3
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": sha256_file(SEL12_JSON), "target_cache": ev.target_meta, "heads": heads}
    out_path = os.path.join(os.path.dirname(SEL12_JSON), "s12_target_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"选择锁定于提交 {why[:7]}; 评估了 {len(heads)} 个分类头; 目标域 {ev.target_meta['n']} 个样本")
    for ds, tag, _ in entries:
        accs = [heads[f"{ds}_{tag}_seed{s}"]["target_acc"] for s in seeds]
        print(f"  {tag:28s} " + " / ".join(f"{x:.2f}" for x in accs))
    print(f"[saved] {out_path}")
    return 0


def eval_p4(ds, seeds):
    """第十二轮 P4（预注册 §3ai）：某个数据集的 72 个分类头。门是该数据集的源域选择文件。"""
    from run_matrix import MATRICES
    out_dir = os.path.join("experiment_results", "P4-CrossDataset")
    sel_json = os.path.join(out_dir, f"{ds}_source_selection.json")
    locked, why = selection_is_locked(sel_json)
    if not locked:
        print(f"拒绝运行: {ds} 的源域选择尚未锁定 —— " + why)
        print(f"必须先运行 p4_report.py --dataset {ds} --stage source 并提交其输出。")
        return 2
    ev = HeadEvaluator()
    heads = {}
    entries = MATRICES[f"p4_{ds.lower()}"][1]
    for _, tag, _ex in entries:
        for s in seeds:
            run = f"{ds}_{tag}_seed{s}"
            heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(sel_json, encoding="utf-8") as f:
        sel = json.load(f)

    def same(r):
        v, w = heads.get(r), sel["runs"].get(r, {})
        return bool(w) and v is not None and w.get("head_sha256") == v["head_sha256"] \
            and w.get("result_sha256") == v["result_sha256"]

    missing = [r for r in sorted(heads) if not same(r)]
    if missing:
        print("拒绝: 已评估 run 与锁定不一致: " + ", ".join(missing[:8]))
        return 3
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "dataset": ds,
           "selection_lock_commit": why, "selection_sha256": sha256_file(sel_json),
           "target_cache": ev.target_meta, "heads": heads}
    out_path = os.path.join(out_dir, f"{ds}_target_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"[{ds}] 选择锁定于 {why[:7]}; 评估 {len(heads)} 个头; 目标域 {ev.target_meta['n']} 个样本")
    for _, tag, _ex in entries:
        accs = [heads[f"{ds}_{tag}_seed{s}"]["target_acc"] for s in seeds]
        print(f"  {tag:28s} " + " / ".join(f"{x:.2f}" for x in accs))
    print(f"[saved] {out_path}")
    return 0


def eval_stage1b_c2(seeds):
    """第十五轮 P2（预注册 §3as）：Bias-Free × L1 的 12 个 run。门 = stage1b_c2_source.json。"""
    from run_matrix import MATRICES
    out_dir = os.path.join("experiment_results", "Stage1B-L2")
    sel_json = os.path.join(out_dir, "stage1b_c2_source.json")
    locked, why = selection_is_locked(sel_json)
    if not locked:
        print("拒绝运行: stage1b_c2 的源域文件尚未锁定 —— " + why)
        return 2
    ev = HeadEvaluator()
    heads = {}
    for _, tag, _ex in MATRICES["stage1b_c2"][1]:
        for s in seeds:
            run = f"CUB_{tag}_seed{s}"
            heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(sel_json, encoding="utf-8") as f:
        sel = json.load(f)

    def same(r):
        v, w = heads.get(r), sel["runs"].get(r, {})
        return bool(w) and v is not None and w.get("head_sha256") == v["head_sha256"]             and w.get("result_sha256") == v["result_sha256"]

    missing = [r for r in sorted(heads) if not same(r)]
    if missing:
        print("拒绝: 已评估 run 与锁定不一致: " + ", ".join(missing[:8]))
        return 3
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": sha256_file(sel_json), "target_cache": ev.target_meta, "heads": heads}
    out_path = os.path.join(out_dir, "stage1b_c2_target_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"选择锁定于 {why[:7]}; 评估 {len(heads)} 个分类头; 目标域 {ev.target_meta['n']} 个样本")
    for _, tag, _ex in MATRICES["stage1b_c2"][1]:
        accs = [heads[f"CUB_{tag}_seed{s}"]["target_acc"] for s in seeds]
        print(f"  {tag:18s} " + " / ".join(f"{x:.2f}" for x in accs))
    print(f"[saved] {out_path}")
    return 0


def eval_stage1b(seeds):
    """第十三轮 Stage 1B（预注册 §3ao）：24 个 L2 run 的事后目标域评估。门 = stage1b_source.json。"""
    from run_matrix import MATRICES
    out_dir = os.path.join("experiment_results", "Stage1B-L2")
    sel_json = os.path.join(out_dir, "stage1b_source.json")
    locked, why = selection_is_locked(sel_json)
    if not locked:
        print("拒绝运行: Stage 1B 的源域文件尚未锁定 —— " + why)
        return 2
    ev = HeadEvaluator()
    heads = {}
    for _, tag, _ex in MATRICES["stage1b"][1]:
        for s in seeds:
            run = f"CUB_{tag}_seed{s}"
            heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(sel_json, encoding="utf-8") as f:
        sel = json.load(f)

    def same(r):
        v, w = heads.get(r), sel["runs"].get(r, {})
        return bool(w) and v is not None and w.get("head_sha256") == v["head_sha256"] \
            and w.get("result_sha256") == v["result_sha256"]

    missing = [r for r in sorted(heads) if not same(r)]
    if missing:
        print("拒绝: 已评估 run 与锁定不一致: " + ", ".join(missing[:8]))
        return 3
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": sha256_file(sel_json), "target_cache": ev.target_meta, "heads": heads}
    out_path = os.path.join(out_dir, "stage1b_target_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"选择锁定于 {why[:7]}; 评估 {len(heads)} 个分类头; 目标域 {ev.target_meta['n']} 个样本")
    for _, tag, _ex in MATRICES["stage1b"][1]:
        accs = [heads[f"CUB_{tag}_seed{s}"]["target_acc"] for s in seeds]
        print(f"  {tag:22s} " + " / ".join(f"{x:.2f}" for x in accs))
    print(f"[saved] {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate_stage0", action="store_true")
    ap.add_argument("--stage1c", action="store_true")
    ap.add_argument("--s11", action="store_true")
    ap.add_argument("--s12", action="store_true")
    ap.add_argument("--stage1b", action="store_true")
    ap.add_argument("--stage1b_c2", action="store_true")
    ap.add_argument("--p4", choices=["AWA2", "LADV", "PACS"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    if a.validate_stage0:
        return validate_stage0(a.seeds)
    if a.stage1c:
        return eval_stage1c(a.seeds)
    if a.s11:
        return eval_s11(a.seeds)
    if a.s12:
        return eval_s12(a.seeds)
    if a.stage1b:
        return eval_stage1b(a.seeds)
    if a.stage1b_c2:
        return eval_stage1b_c2(a.seeds)
    if a.p4:
        return eval_p4(a.p4, a.seeds)

    locked, why = selection_is_locked()
    if not locked:
        print("拒绝运行: 源域选择结果尚未锁定 —— " + why)
        print("必须先运行 p1_independent_report.py --stage source 并提交其输出, 再做事后目标域评估。")
        return 2
    ev = HeadEvaluator()
    heads = {}
    for _, tag in DOSE:
        for s in a.seeds:
            run = f"CUB_p1_{tag}_seed{s}"
            heads[run] = ev.evaluate(f"logs/heads/{run}.pt", f"logs/results/{run}.json")
    with open(SELECT_JSON, encoding="utf-8") as f:
        sel = json.load(f)
    # 事后评估用的分类头必须就是源域选择时用的那一份, 结果 json 也没有被改动过 (逐个按 sha256 核对)
    def same(r):
        v, w = heads[r], sel["runs"].get(r, {})
        return bool(w) and w.get("head_sha256") == v["head_sha256"] and w.get("result_sha256") == v["result_sha256"]
    mismatched = [r for r in sorted(set(heads) | set(sel["runs"])) if r not in heads or not same(r)]
    if mismatched:
        print("拒绝: 以下 run 的分类头或结果 json 与源域选择锁定时的不一致 (或数量不同): " + ", ".join(mismatched))
        return 3
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": git_info(), "selection_lock_commit": why,
           "selection_sha256": sha256_file(SELECT_JSON), "target_cache": ev.target_meta, "heads": heads}
    with open(os.path.join(OUT_DIR, "p1_target_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"选择锁定于提交 {why[:7]}; 评估了 {len(heads)} 个分类头; 目标域 {ev.target_meta['n']} 个样本")
    for _, tag in DOSE:
        accs = [heads[f"CUB_p1_{tag}_seed{s}"]["target_acc"] for s in a.seeds]
        print(f"  {tag:11s} 目标域 Acc: " + " / ".join(f"{x:.2f}" for x in accs))
    print(f"[saved] {OUT_DIR}/p1_target_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
