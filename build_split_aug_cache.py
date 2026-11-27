"""为任意数据集构建 P1 风格的“独立源域划分 + 增强特征”缓存（第十二轮 P4 的前置）。

CUB 的这一份（cache/CUB_train_aug.pt）是 P1 时做的，脚本没有入库；P4 要在 AwA2 / LADV / PACS 上
“独立划分源域训练集和验证集”，因此把同一套构造参数化：

  - 划分：按类分层 80/20（seed 42，与 CUB 相同），逐类独立打乱（numpy default_rng，不与 torch 全局随机数耦合）
  - 特征：只对 heldout_val 的样本编码（clean + 7 种扰动）；train_core 的行填 0。
    `main_cached._evaluate_source_val` 只按 heldout 下标取值，`_make_train_subset` 只读 labels 与 split，
    所以填 0 不影响任何被使用的数值。这一点写在 meta 里，避免后人误读。
  - 归一化：与 CUB 相同 —— 编码器原始输出按 float32 做 L2 归一化后存 fp16（与训练缓存“未归一化”不同，
    见 RESULTS §3u 的说明）。
  - 扰动参数与 CUB 的 meta 完全一致：solarize 128 / equalize / posterize 3 / autocontrast / grayscale /
    blur(kernel 7, sigma 4.0 固定) / colorjitter(0.4,0.4,0.4,0.2，逐样本 seed = 20261126 + index)。

用法:
  python build_split_aug_cache.py --dataset PACS --smoke           # 只做前 40 个 heldout 样本
  python build_split_aug_cache.py --dataset AwA2
  python build_split_aug_cache.py --dataset LADV
  python build_split_aug_cache.py --dataset PACS
产物: cache/{dataset}_train_aug.pt、cache/{dataset}_train_aug_meta.json（不覆盖已存在的文件）
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
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from run_status import git_info  # noqa: E402

SEED_BASE = 20261126
AUGS = ["solarize", "equalize", "posterize", "autocontrast", "grayscale", "blur", "colorjitter"]


def perturb(name, pil, idx):
    if name == "solarize":
        return ImageOps.solarize(pil, threshold=128)
    if name == "equalize":
        return ImageOps.equalize(pil)
    if name == "posterize":
        return ImageOps.posterize(pil, 3)
    if name == "autocontrast":
        return ImageOps.autocontrast(pil)
    if name == "grayscale":
        return pil.convert("L").convert("RGB")
    if name == "blur":
        return T.GaussianBlur(kernel_size=7, sigma=4.0)(pil)          # 固定 sigma, 消除随机性
    if name == "colorjitter":
        torch.manual_seed(SEED_BASE + int(idx))                        # 逐样本固定种子
        return T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)(pil)
    raise ValueError(name)


def stratified_split(labels, ratio=0.8, seed=42):
    """按类分层。逐类用独立 rng，类别顺序稳定；返回 (train_core, heldout_val) 的升序 LongTensor。"""
    y = labels.numpy()
    tr, va = [], []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng = np.random.default_rng(int(seed) + int(c))
        perm = rng.permutation(idx)
        n_val = max(1, int(round((1.0 - ratio) * len(idx))))
        va.extend(perm[:n_val].tolist())
        tr.extend(perm[n_val:].tolist())
    return torch.tensor(sorted(tr), dtype=torch.long), torch.tensor(sorted(va), dtype=torch.long)


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD  ") + name + ("  [%s]" % detail if detail else ""))

    lab = torch.tensor([0] * 10 + [1] * 9 + [2] * 5)
    tr, va = stratified_split(lab)
    check("S1 划分不相交、并集为全体、保持升序",
          len(set(tr.tolist()) & set(va.tolist())) == 0 and len(tr) + len(va) == len(lab)
          and tr.tolist() == sorted(tr.tolist()) and va.tolist() == sorted(va.tolist()))
    ratio = []
    for c in (0, 1, 2):
        m = lab == c
        ratio.append(len(set(va.tolist()) & set(torch.nonzero(m).view(-1).tolist())) / int(m.sum()))
    check("S2 逐类留出比例接近 20%", all(abs(r - 0.2) <= 0.11 for r in ratio), str([round(r, 3) for r in ratio]))
    check("S3 同 seed 可复现、换 seed 不同", torch.equal(va, stratified_split(lab)[1])
          and not torch.equal(va, stratified_split(lab, seed=7)[1]))
    img = Image.new("RGB", (64, 64), (120, 80, 40))
    outs = [np.asarray(perturb(a, img, 3)).mean() for a in AUGS]
    outs2 = [np.asarray(perturb(a, img, 3)).mean() for a in AUGS]
    check("S4 七种扰动逐个可复现（同 index）", np.allclose(outs, outs2))
    check("S5 grayscale 三通道相等", np.asarray(perturb("grayscale", img, 0))[..., 0].std() == 0)
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="只编码前 40 个 heldout 样本（其余行保持 0）")
    a = ap.parse_args()
    if a.selftest:
        return self_test()
    assert a.dataset, "--dataset 必填"

    # 用原管线构建数据集（保证 annos / preprocess / 类别表与训练一致）
    sys.argv = [sys.argv[0], "--dataset", a.dataset, "--CLIP_type",
                os.path.abspath(os.path.join("clip_weights", "ViT-L-14.pt")), "--epochs", "1"]
    from args import get_args
    from data import get_dataset_classes
    import clip

    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = get_dataset_classes(args)[0]
    labels = torch.load(os.path.join("cache", f"{a.dataset}_train.pt"), map_location="cpu")["labels"].long()
    assert len(ds) == len(labels), f"数据集 {len(ds)} 与缓存 {len(labels)} 长度不一致"
    # 抽样核对“annos 顺序 = 缓存顺序”（两处标签都对得上才继续）
    rng = np.random.default_rng(0)
    probe = rng.choice(len(labels), size=min(64, len(labels)), replace=False)
    bad = [int(i) for i in probe if int(ds.annos[int(i)].split(",")[1]) != int(labels[int(i)])]
    assert not bad, f"annos 与缓存的标签顺序不一致（前几个: {bad[:5]}）"

    tr, va = stratified_split(labels, a.ratio, a.seed)
    print(f"[{a.dataset}] N={len(labels)} → train_core {len(tr)} / heldout_val {len(va)}")
    use = va[:40] if a.smoke else va

    N, D = len(labels), 768
    clean = torch.zeros(N, D, dtype=torch.float16)
    aug = {k: torch.zeros(N, D, dtype=torch.float16) for k in AUGS}
    clip_model, preprocess = clip.load(os.path.abspath(os.path.join("clip_weights", "ViT-L-14.pt")),
                                       device=args.device)
    clip_model.eval()
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for s in range(0, len(use), a.batch):
            chunk = use[s:s + a.batch].tolist()
            pils, ids = [], []
            for i in chunk:
                line = ds.annos[i]
                p = os.path.join(ds.data_root, line.split(",")[0])
                try:
                    pils.append(Image.open(p).convert("RGB"))
                    ids.append(i)
                except OSError:
                    continue
            if not pils:
                continue
            f0 = clip_model.encode_image(torch.stack([preprocess(p) for p in pils]).to(args.device)).float()
            f0 = (f0 / f0.norm(dim=-1, keepdim=True)).half().cpu()
            for j, i in enumerate(ids):
                clean[i] = f0[j]
            for name in AUGS:
                fk = clip_model.encode_image(
                    torch.stack([preprocess(perturb(name, p, i)) for p, i in zip(pils, ids)]).to(args.device)).float()
                fk = (fk / fk.norm(dim=-1, keepdim=True)).half().cpu()
                for j, i in enumerate(ids):
                    aug[name][i] = fk[j]
            done += len(ids)
            if (s // a.batch) % 10 == 0:
                print(f"  {done}/{len(use)}  {time.time() - t0:.0f}s", flush=True)

    out = os.path.join("cache", f"{a.dataset}_train_aug.pt")
    if os.path.exists(out) and not a.smoke:
        print(f"[拒绝覆盖] {out} 已存在；如需重建请先手动删除")
        return 1
    torch.save({"clean_feats": clean, "aug_feats": aug, "labels": labels,
                "split": {"train_core": tr, "heldout_val": va}}, out)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": a.dataset,
            "n_samples": N, "n_classes": int(labels.max()) + 1,
            "feature_dim": D, "feature_dtype": "float16",
            "normalization": "L2-normalized (unit norm per feature vector)",
            "computed_rows": "heldout_val" if not a.smoke else "heldout_val 的前 40 个（冒烟）",
            "zero_rows": "train_core（以及冒烟时未编码的 heldout 行）填 0；只被 heldout 下标读取",
            "augmentations": AUGS, "split": {"ratio": a.ratio, "seed": a.seed, "method": "stratified by class",
                                             "n_train_core": len(tr), "n_heldout_val": len(va)},
            "git": git_info(), "elapsed_seconds": round(time.time() - t0, 1),
            "cache_sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
    with open(os.path.join("cache", f"{a.dataset}_train_aug_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[done] {out}  ({meta['elapsed_seconds']}s, {len(use)} 个样本 × 8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
