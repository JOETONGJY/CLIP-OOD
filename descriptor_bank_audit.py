"""描述符库现状审查 (2026-11-26, 外审第九轮 P4 准备; 说明见 doc/描述符库接口.md)。

只读审查 + 一次真实数据的端到端核对, 不训练、不生成新词库:
  A  训练管线实际使用的词库 (data/__init__.py 是否经由统一入口; 默认值)
  B  prompts/ 下各词库文件: 条数、是否通过校验、身份
  C  CLIP 权重身份 (进入缓存键)
  D  CUB + Original-204: 新旧缓存键、缓存文件、认领的旧缓存与旧文件逐位相同
  E  用现役 CLIP 权重重算一遍 CUB 的域方向与可靠性分数, 与缓存比对 (只此一次, 约一两分钟)
  F  JSON 冻结格式与 Python 列表给出同一份词库 (身份与缓存键都相同)

用法: python descriptor_bank_audit.py [--skip_recompute]
产物: experiment_results/DescriptorBank/descriptor_bank_audit.md (本文件由脚本生成)
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch

import descriptor_bank as db
import reliability as rel
from run_status import git_info

OUT_DIR = os.path.join("experiment_results", "DescriptorBank")
CACHE_DIR = "cache/reliability"
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))


def clip_path_from_argv():
    a = sys.argv
    return os.path.abspath(a[a.index("--CLIP_type") + 1]) if "--CLIP_type" in a else os.path.abspath("clip_weights/ViT-L-14.pt")


def bank_rows():
    rows = []
    for p in sorted(glob.glob("prompts/*.py")) + sorted(glob.glob("prompts/banks/*.json")):
        try:
            b = db.load_descriptor_bank(p)
            rows.append((p.replace(os.sep, "/"), b["n_source"], b["n_target"], "通过", b["bank_sha256"]))
        except ValueError as e:
            n_src = n_tgt = None
            if p.endswith(".py"):
                import importlib.util
                so = importlib.util.spec_from_file_location("audit_" + hashlib.md5(p.encode()).hexdigest()[:8], p)
                m = importlib.util.module_from_spec(so)
                so.loader.exec_module(m)
                n_src, n_tgt = len(getattr(m, "source_text_prompts", [])), len(getattr(m, "target_text_prompts", []))
            rows.append((p.replace(os.sep, "/"), n_src, n_tgt, "拒绝: " + str(e).split(": ", 1)[-1][:70], ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--CLIP_type", default="clip_weights/ViT-L-14.pt")
    ap.add_argument("--skip_recompute", action="store_true")
    a = ap.parse_args()
    clip_file = os.path.abspath(a.CLIP_type)
    L = ["# 描述符库现状审查（CUB；" + time.strftime("%Y-%m-%d") + "）", "",
         "由 `descriptor_bank_audit.py` 生成（代码版本 `" + str((git_info() or {}).get("commit", "?"))[:7] + "`）。只读审查，不训练，不生成新词库。", ""]

    # ---- A ----
    src = open("data/__init__.py", encoding="utf-8").read()
    via_loader = "load_descriptor_bank(" in src and "from prompts.prompt200new import" not in src
    from args import get_args
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1", "--CLIP_type", clip_file, "--epochs", "1"]
    default_bank = getattr(get_args(), "descriptor_bank", None)
    check("A 训练管线经由统一入口取词库, 不再硬编码导入; 默认为 original", via_loader and default_bank == "original",
          f"统一入口={via_loader}, 默认={default_bank}")
    L += ["## A. 训练实际使用的词库", "",
          f"- `data/__init__.py` 经由 `descriptor_bank.load_descriptor_bank` 取模板：{via_loader}；`--descriptor_bank` 的默认值：`{default_bank}`（Original-204）。", ""]

    # ---- B ----
    rows = bank_rows()
    L += ["## B. prompts/ 下的词库文件", "", "| 文件 | 源模板数 | 目标模板数 | 校验 | 词库身份 (sha256 前 16 位) |", "|---|---|---|---|---|"]
    for p, ns, nt, verdict, sha in rows:
        L.append(f"| `{p}` | {ns} | {nt} | {verdict} | {sha[:16]} |")
    orig = db.load_descriptor_bank("original")
    check("B Original-204 通过校验: 1 个源模板 + 204 个目标模板", orig["n_source"] == 1 and orig["n_target"] == 204)
    L.append("")
    L += [f"- Original-204 的身份：源 `{orig['src_sha256'][:16]}`，目标 `{orig['tgt_sha256'][:16]}`，词库 `{orig['bank_sha256'][:16]}`；"
          f"文件 sha256 `{orig['meta']['file_sha256'][:16]}`。", ""]

    # ---- C ----
    clip_id = rel.clip_file_identity(clip_file)
    check("C CLIP 权重身份可计算 (sha256 前 16 位, 进入缓存键)", bool(clip_id) and len(clip_id) == 16, f"{os.path.basename(clip_file)} -> {clip_id}")
    L += ["## C. CLIP 权重身份", "", f"- `{os.path.basename(clip_file)}`（{os.path.getsize(clip_file) // 1_000_000} MB）：sha256 前 16 位 `{clip_id}`。进入新的缓存键。", ""]

    # ---- D ----
    sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1", "--CLIP_type", clip_file, "--epochs", "1"]
    args = get_args()
    args.save_model = False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    from data import get_dataset_classes
    ds = get_dataset_classes(args)[0]
    class_names, concept_names = list(ds.classname2id.keys()), list(ds.concept2id.keys())
    legacy_key, key = db.cache_key(orig["source_text_prompts"], class_names, orig["target_text_prompts"], concept_names, clip_id)
    new_p, legacy_p = os.path.join(CACHE_DIR, f"reliability_{key}.pt"), os.path.join(CACHE_DIR, f"reliability_{legacy_key}.pt")
    check("D CUB + Original-204 的新键缓存文件存在, 且键与数据集对象上记录的一致",
          os.path.exists(new_p) and getattr(ds, "reliability_cache_key", None) == key, f"新键 {key}")
    new = torch.load(new_p, map_location="cpu", weights_only=False)
    meta = new.get("meta") or {}
    adopted = os.path.exists(legacy_p)
    if adopted:
        old = torch.load(legacy_p, map_location="cpu", weights_only=False)
        same = torch.equal(old["diffs"], new["diffs"]) and torch.equal(old["scores"], new["scores"])
        check("D2 认领自旧缓存的数值与旧文件逐位相同", same and meta.get("legacy_adopted") is True, f"旧键 {legacy_key}")
    else:
        check("D2 (没有旧缓存文件; 新缓存由重算产生)", meta.get("legacy_adopted") is False, f"旧键 {legacy_key}")
    check("D3 缓存文件记录了词库身份 / 类别表 / 概念表 / CLIP 身份",
          meta.get("bank_sha256") == orig["bank_sha256"] and meta.get("n_classes") == len(class_names)
          and meta.get("n_concepts") == len(concept_names) and meta.get("clip_identity") == clip_id, json.dumps(meta, ensure_ascii=False)[:200])
    check("D4 domain_diffs 的形状 = (204, 200, 768)", tuple(ds.domain_diffs.shape) == (204, len(class_names), 768), str(tuple(ds.domain_diffs.shape)))
    L += ["## D. CUB + Original-204 的可靠性缓存", "",
          f"- 旧键 `{legacy_key}`；新键 `{key}`（旧键的三个文本序列 + 概念名 {len(concept_names)} 个 + CLIP 身份 `{clip_id}`）。",
          f"- 新键缓存文件：`{os.path.basename(new_p)}`；" + ("认领自旧键文件，数值与旧文件逐位相同。" if adopted and same else "由重算产生。"),
          f"- 缓存内的记录：`{json.dumps(meta, ensure_ascii=False)}`", ""]

    # ---- E ----
    if not a.skip_recompute:
        import clip
        model, _ = clip.load(clip_file, device=args.device)
        diffs, scores = rel.compute_domain_diffs_and_scores(model, orig["source_text_prompts"], orig["target_text_prompts"],
                                                             class_names, args.device, verbose=False)
        d_diff = float((diffs.cpu().double() - new["diffs"].double()).abs().max())
        d_score = float((scores.cpu().double() - new["scores"].double()).abs().max())
        bitwise = d_diff == 0.0 and d_score == 0.0
        check("E 用现役 CLIP 权重重算的域方向 / 可靠性分数与缓存一致 (最大绝对差 < 1e-5)",
              d_diff < 1e-5 and d_score < 1e-5, f"域方向 {d_diff:.2e}, 分数 {d_score:.2e}, 逐位相同={bitwise}")
        L += ["## E. 一次重算核对", "",
              f"- 用现役 CLIP 权重重算 204 对模板的域方向与可靠性分数，与缓存的最大绝对差：域方向 {d_diff:.2e}，分数 {d_score:.2e}；"
              + ("**逐位相同**。" if bitwise else "不是逐位相同。"), ""]

    # ---- F ----
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "orig_as_json.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"id": "original-as-json", "version": "audit", "source_text_prompts": orig["source_text_prompts"],
                       "target_text_prompts": orig["target_text_prompts"]}, f, ensure_ascii=False)
        bj = db.load_descriptor_bank(p)
    check("F 同样的文本写成 JSON 冻结格式: 词库身份相同, 缓存键相同", bj["bank_sha256"] == orig["bank_sha256"]
          and db.cache_key(bj["source_text_prompts"], class_names, bj["target_text_prompts"], concept_names, clip_id)[1] == key)
    L += ["## F. JSON 冻结格式", "", "- 同样的文本写成 JSON 冻结格式后，词库身份与缓存键都与 Python 列表版本相同（文本才是身份，文件格式不是）。", "",
          "## 结论", "", "全部核对项通过：" + str(OK), ""]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "descriptor_bank_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("AUDIT:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
