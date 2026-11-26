"""描述符库 (descriptor bank) 的统一加载入口 (2026-11-26, 外审第九轮 P4 准备; 说明见 doc/描述符库接口.md)。

现状: 训练管线一直硬编码使用 prompts/prompt200new.py 的两个列表 —— 1 个源模板 + 204 个目标模板,
      每个模板恰好含一个 {} (类别名占位符)。下文称它为 Original-204。**Original-204 保持不变。**

本模块只做一件事: 把"用哪一份词库"变成一个可以显式指定、并且能被完整记录的输入。
  python main_cached.py ... --descriptor_bank original                 # 默认; 等价于原来的硬编码导入
  python main_cached.py ... --descriptor_bank prompts/banks/sdb-204.json     # 冻结格式的词库文件
  python main_cached.py ... --descriptor_bank prompts/my_bank.py             # 含两个列表的 Python 文件

冻结格式 (JSON; 供 SDB-204 / Original-Rephrased 等新词库使用; 新词库在收到并冻结之前不训练):
  {
    "id": "sdb-204", "version": "v1", "created": "2026-11-..", "generator": "...",
    "generation_record": "生成记录 (完整对话 / 种子 / 规则) 的位置或摘要",
    "source_text_prompts": ["a photo of a {}."],
    "target_text_prompts": ["an ink wash painting depicting a {}.", ...],       # 顺序即最终顺序
    "descriptor_categories": ["绘画与艺术媒介", ...],                           # 可选; 与 target 等长
    "notes": "..."
  }
词库的身份 = 两个列表的完整文本 (含顺序) 的 sha256 —— bank_sha256; 它进入可靠性缓存的键。

规则 (外审第九轮):
  - 不自行根据已知的目标域准确率编写专门的词库; 不使用 descriptor_selection.py 里"真实目标域偏移解释力"作为筛选依据。
  - Original-Rephrased 对照 (只改措辞与模板、保持原语义) 已预留位置: prompts/banks/original_rephrased.json, 尚未提供。
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(ROOT, "prompts", "prompt200new.py")   # Original-204: 当前正式词库, 不修改 (绝对路径: 不依赖工作目录)
RESERVED = {"original_rephrased": "Original-Rephrased: 只改措辞与模板、保持原语义的对照 (尚未提供)"}


def text_sha256(texts):
    """一段文本列表的身份: 逐条用 NUL 连接后的 sha256 (含顺序、含完整文本)"""
    return hashlib.sha256("\x00".join(list(texts)).encode("utf-8")).hexdigest()


def validate_bank(source_text_prompts, target_text_prompts, where="词库"):
    """每条模板必须是非空字符串、恰好含一个 {} 占位符 (类别名), 且没有重复。任何一条不满足就抛错。"""
    for name, texts in (("source_text_prompts", source_text_prompts), ("target_text_prompts", target_text_prompts)):
        if not isinstance(texts, (list, tuple)) or len(texts) == 0:
            raise ValueError(f"{where}: {name} 必须是非空列表")
        seen = set()
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"{where}: {name}[{i}] 不是非空字符串: {t!r}")
            if t.count("{}") != 1 or "{" in t.replace("{}", "") or "}" in t.replace("{}", ""):
                raise ValueError(f"{where}: {name}[{i}] 必须恰好含一个 {{}} 占位符 (且不含其它花括号): {t!r}")
            if t in seen:
                raise ValueError(f"{where}: {name}[{i}] 重复出现: {t!r}")
            seen.add(t)


def load_descriptor_bank(spec="original"):
    """spec: "original" (Original-204) | 词库文件路径 (.json 冻结格式 / .py 含两个列表的模块) | 预留 id。
    返回 dict: id, path, source_text_prompts, target_text_prompts, n_source, n_target,
               src_sha256, tgt_sha256, bank_sha256 (= sha256(src_sha256 + tgt_sha256)), meta。"""
    spec = (spec or "original").strip()
    if spec == "original":
        path = ORIGINAL
    elif spec in RESERVED:
        p = os.path.join(ROOT, "prompts", "banks", spec + ".json")
        if not os.path.exists(p):
            raise ValueError(f"词库 {spec!r} ({RESERVED[spec]}) 尚未提供: 找不到 {p}")
        path = p
    else:
        path = spec if os.path.isabs(spec) else os.path.join(ROOT, spec)
    if not os.path.exists(path):
        raise ValueError(f"找不到词库文件: {path} (--descriptor_bank 取 'original' 或一个 .json / .py 文件的路径)")
    try:
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:                                          # Windows: 词库文件与仓库不在同一盘符
        rel = path.replace(os.sep, "/")
    meta = {"path": rel}
    if spec == "original":
        meta["id"] = "original"                                 # 词库 id 是 original; 文件名只是它的位置
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        src, tgt = d["source_text_prompts"], d["target_text_prompts"]
        validate_bank(src, tgt, where=meta["path"])
        cats = d.get("descriptor_categories")
        if cats is not None and len(cats) != len(tgt):
            raise ValueError(f"{path}: descriptor_categories 的长度 ({len(cats)}) 与 target_text_prompts ({len(tgt)}) 不一致")
        meta.update({k: d.get(k) for k in ("id", "version", "created", "generator", "generation_record", "notes")})
        meta["n_categories"] = len(set(cats)) if cats else None
        meta["file_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    elif path.lower().endswith(".py"):
        import importlib.util
        mod_name = "descriptor_bank_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = mod
        spec_obj.loader.exec_module(mod)                       # 执行该文件里的赋值 (与原来的 import 语义一致)
        src, tgt = mod.source_text_prompts, mod.target_text_prompts
        validate_bank(src, tgt, where=meta["path"])
        meta["file_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    else:
        raise ValueError(f"不认识的词库文件类型 (只支持 .json / .py): {path}")
    src_sha, tgt_sha = text_sha256(src), text_sha256(tgt)
    meta.update({"format": "json" if path.lower().endswith(".json") else "python", "src_sha256": src_sha,
                 "tgt_sha256": tgt_sha, "bank_sha256": text_sha256([src_sha, tgt_sha])})
    return {"id": meta.get("id") or os.path.splitext(os.path.basename(path))[0], "path": meta["path"],
            "source_text_prompts": list(src), "target_text_prompts": list(tgt),
            "n_source": len(src), "n_target": len(tgt), "src_sha256": src_sha, "tgt_sha256": tgt_sha,
            "bank_sha256": meta["bank_sha256"], "meta": meta}


def cache_key(source_text_prompts, class_names, target_text_prompts, concept_names=None, clip_identity=None):
    """可靠性缓存 (cache/reliability) 的键。
    legacy_key: 只含 三个文本序列 —— 2026-11-26 之前的键, 保留用来认领旧的缓存文件。
    key: 另加概念表 (名称与顺序) 与 CLIP 权重身份。完整文本、顺序、类别表都在键里, 新旧词库不会串用。"""
    core = "\x00".join(list(source_text_prompts)) + "\x00" + "\x00".join(list(class_names)) + "\x00" \
        + "\x00".join(list(target_text_prompts))
    legacy = hashlib.md5(core.encode("utf-8")).hexdigest()[:16]
    extra = "\x00" + "\x00".join(list(concept_names or [])) + "\x00" + str(clip_identity or "")
    return legacy, hashlib.md5((core + extra).encode("utf-8")).hexdigest()[:16]


def self_test():
    import tempfile
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    b = load_descriptor_bank("original")
    check("B1 Original-204: 1 个源模板 + 204 个目标模板, 每条恰好一个 {}", b["n_source"] == 1 and b["n_target"] == 204)
    # Original-204 的身份 (2026-11-26 钉死): 两个列表的完整文本按 NUL 连接后的 sha256。有人改动了 prompts/prompt200new.py,
    # 这里就会报出来 —— 正式词库必须保持不变 (外审第九轮)。
    PIN = {"src": "e8e367e769f86c27d7507382b76c671667da9fa8ae030159a57681a812b0fa33",
           "tgt": "1f55c2a9855f277fe9da6c60ff15a774f827397b156c11eb29014ef7e52f60ee",
           "bank": "bc4ba710ad72ff0fee814a9303d5fed1731ed02dcdecf1d7829f63f8f94c82de"}
    check("B2 Original-204 的文本身份与钉死的记录一致 (正式词库未被改动)",
          b["src_sha256"] == PIN["src"] and b["tgt_sha256"] == PIN["tgt"] and b["bank_sha256"] == PIN["bank"],
          b["tgt_sha256"][:16])
    try:
        load_descriptor_bank(os.path.join("prompts", "prompt200.py"))
        check("B2b 旧的 prompt200.py (210 条) 含重复条目, 校验器拒绝加载它", False)
    except ValueError as e:
        check("B2b 旧的 prompt200.py (210 条) 含重复条目, 校验器拒绝加载它", "重复出现" in str(e), str(e)[-40:])
    from prompts import prompt200new
    check("B3 加载结果与原来的硬编码导入逐条相同 (文本与顺序)", list(prompt200new.source_text_prompts) == b["source_text_prompts"]
          and list(prompt200new.target_text_prompts) == b["target_text_prompts"])
    b2 = load_descriptor_bank(os.path.join("prompts", "prompt200new.py"))
    check("B4 用文件路径加载与用 'original' 加载是同一份词库", b2["bank_sha256"] == b["bank_sha256"] and b2["path"] == b["path"])
    check("B4b 记录的路径是相对仓库根目录的正斜杠形式", b["path"] == "prompts/prompt200new.py", b["path"])
    cwd0 = os.getcwd()
    try:
        os.chdir(tempfile.gettempdir())                          # 训练可以从任何工作目录启动
        b3 = load_descriptor_bank("original")
        check("B4c 换一个工作目录也能加载 Original-204 (路径不依赖当前目录)", b3["bank_sha256"] == b["bank_sha256"])
    finally:
        os.chdir(cwd0)
    # ---- 校验规则 ----
    bad = [("a {}.", ["b {}."], "不是列表而是一个字符串"), (["a {} and {}."], ["b {}."], "两个占位符"), (["a photo."], ["b {}."], "没有占位符"),
           (["a {x} {}."], ["b {}."], "多余花括号"), (["a {}."], ["b {}.", "b {}."], "重复"), ([], ["b {}."], "空列表"),
           (["a {}.", ""], ["b {}."], "空字符串")]
    for i, (s, t, why) in enumerate(bad):
        try:
            validate_bank(s, t)
            check(f"B5.{i} 拒绝不合格的词库: {why}", False)
        except ValueError:
            check(f"B5.{i} 拒绝不合格的词库: {why}", True)
    check("B5b 元组也可以 (内部转成列表, 不改变内容)", load_descriptor_bank("original")["target_text_prompts"]
          == list(tuple(load_descriptor_bank("original")["target_text_prompts"])))
    # ---- JSON 冻结格式 ----
    with tempfile.TemporaryDirectory() as d:
        good = {"id": "toy", "version": "v1", "generated": "x", "source_text_prompts": ["a photo of a {}."],
                "target_text_prompts": ["a painting of a {}.", "a sketch of a {}."],
                "descriptor_categories": ["绘画", "线条"]}
        p = os.path.join(d, "toy.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(good, f, ensure_ascii=False)
        bj = load_descriptor_bank(p)
        check("B6 JSON 词库: 加载、计数、分类数、身份", bj["n_target"] == 2 and bj["meta"]["n_categories"] == 2
              and bj["id"] == "toy" and len(bj["bank_sha256"]) == 64 and bj["bank_sha256"] == load_descriptor_bank(p)["bank_sha256"])
        with open(p, "w", encoding="utf-8") as f:
            json.dump({**good, "descriptor_categories": ["只有一类"]}, f, ensure_ascii=False)
        try:
            load_descriptor_bank(p)
            check("B7 类别标注长度与描述符数不一致时拒绝", False)
        except ValueError:
            check("B7 类别标注长度与描述符数不一致时拒绝", True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({**good, "target_text_prompts": list(reversed(good["target_text_prompts"]))}, f, ensure_ascii=False)
        check("B8 顺序不同即身份不同 (bank_sha256 含顺序)", load_descriptor_bank(p)["bank_sha256"] != bj["bank_sha256"])
    try:
        load_descriptor_bank("original_rephrased")
        check("B9 预留的 Original-Rephrased 尚未提供: 报错而不是静默回退", False)
    except ValueError as e:
        check("B9 预留的 Original-Rephrased 尚未提供: 报错而不是静默回退", "尚未提供" in str(e))
    try:
        load_descriptor_bank("prompts/no_such_bank.json")
        check("B10 文件不存在时报错", False)
    except ValueError:
        check("B10 文件不存在时报错", True)
    # ---- 缓存键 ----
    s, t, cls = ["a photo of a {}."], ["a painting of a {}."], ["cat", "dog", "bird"]
    l1, k1 = cache_key(s, cls, t, ["fur", "wing"], "clipA")
    l2, k2 = cache_key(s, cls, t, ["fur", "wing"], "clipA")
    check("B11 相同输入 → 相同的键", (l1, k1) == (l2, k2))
    check("B12 旧键只含三个文本序列 (与 2026-11-26 之前的实现一致)",
          l1 == hashlib.md5(("a photo of a {}.\x00cat\x00dog\x00bird\x00a painting of a {}.").encode("utf-8")).hexdigest()[:16])
    check("B13 概念表不同 / CLIP 不同 / 词库不同 → 新键不同 (旧键只在词库或类别表变化时才变)",
          cache_key(s, cls, t, ["fur", "beak"], "clipA")[1] != k1 and cache_key(s, cls, t, ["fur", "wing"], "clipB")[1] != k1
          and cache_key(s, cls, ["a sketch of a {}."], ["fur", "wing"], "clipA")[1] != k1
          and cache_key(s, cls, t, ["fur", "wing"], "clipB")[0] == l1)
    check("B14 类别名顺序不同 → 键不同", cache_key(s, ["dog", "cat", "bird"], t, ["fur", "wing"], "clipA")[1] != k1)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
