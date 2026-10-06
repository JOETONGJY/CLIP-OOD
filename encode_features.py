"""
一次性 CLIP 图像特征编码（特征缓存加速, 2026-09 本地实验用）

原理: CLIP 全程冻结, 数据集 preprocess 无增广(Resize+CenterCrop+Normalize 确定性变换),
      因此"先编码全部图片存特征"与"每次 forward 现场编码"数学等价。
      缓存后训练只需过 7k 参数的分类头, GPU 几乎零负载。

用法:
  python encode_features.py --dataset CUB  --CLIP_type "F:/CLIP-OOD/clip_weights/ViT-L-14.pt" --batch_size 64
  python encode_features.py --dataset AWA2 --CLIP_type "F:/CLIP-OOD/clip_weights/ViT-L-14.pt" --batch_size 64

产物: cache/{dataset}_{split}.pt  每个 split 一个文件, 含 feats(fp16,N,768)/labels/attrs
"""
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from args import get_args
from data import get_dataset_classes
import clip


def encode_split(dataset, clip_model, device, batch_size, out_path):
    if os.path.exists(out_path):
        print(f"[skip] {out_path} 已存在")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels, attrs = [], [], []
    with torch.no_grad():
        for images, labels_b, attrs_b in tqdm(loader, desc=os.path.basename(out_path)):
            f = clip_model.encode_image(images.to(device))  # fp16 输出, 与原 forward 一致
            feats.append(f.cpu())
            labels.append(torch.as_tensor(labels_b) if not torch.is_tensor(labels_b) else labels_b)
            attrs.append(torch.as_tensor(attrs_b) if not torch.is_tensor(attrs_b) else attrs_b)
    torch.save({
        'feats': torch.cat(feats).cpu(),   # fp16 (N, 768), encode_image 原始输出(未归一化)
        'labels': torch.cat([l.view(-1) for l in labels]).long(),
        'attrs': torch.cat([a.view(a.size(0), -1) for a in attrs]).float(),
    }, out_path)
    d = torch.load(out_path)
    print(f"[done] {out_path}: feats{tuple(d['feats'].shape)} labels{tuple(d['labels'].shape)}")


def main():
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    os.makedirs("cache", exist_ok=True)

    # 按原管线构建三个数据集(保证 annos/preprocess 与训练完全一致)
    train_dataset, _, source_test_dataset, _, target_test_dataset, _ = get_dataset_classes(args)

    # 释放各数据集内部持有的 CLIP 实例(已用不到, 省显存)
    for ds in (train_dataset, source_test_dataset, target_test_dataset):
        if getattr(ds, "clip_model", None) is not None:
            ds.clip_model = None
    torch.cuda.empty_cache()

    # 自己加载一份 CLIP 用于编码
    clip_model, _ = clip.load(args.CLIP_type, device=device)
    clip_model.eval()

    splits = [
        ("train", train_dataset),
        ("source_test", source_test_dataset),
        ("target_test", target_test_dataset),
    ]
    for name, ds in splits:
        out_path = os.path.join("cache", f"{args.dataset}_{name}.pt")
        encode_split(ds, clip_model, device, args.batch_size, out_path)

    print("全部编码完成。后续训练用 main_cached.py。")


if __name__ == "__main__":
    main()
