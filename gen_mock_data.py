"""
生成 mock CUB 数据集（用于 smoke test，不验证精度只验证流程）:
- 20 个类, 每类 15 张训练图 + 5 张测试图 (source)
- 20 个类, 每类 5 张 painting 图 (target)
- 图片用随机噪声 + 每类固定颜色偏移（让分类器有东西可学）
- 生成对应的标注 txt（与 cub_train.txt / cub_test.txt / cubp_test.txt 格式一致）
"""
import os
import random
from PIL import Image
import numpy as np

random.seed(42)
np.random.seed(42)

MOCK_ROOT = "/home/llms/CLIP-OOD-main/mock_data"
SRC_IMG_DIR = os.path.join(MOCK_ROOT, "CUB_200_2011", "images")
# 注意: 原版 CUB-200-Painting 的图片直接在根目录下 (无 images/ 层)
TGT_IMG_DIR = os.path.join(MOCK_ROOT, "CUB-200-Painting")

num_classes = 20
train_per_class = 15
test_per_class = 5
tgt_per_class = 5

# 每类一个固定的颜色偏移 (模拟类别可区分性)
class_tints = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_classes)]

for c in range(num_classes):
    cls_dir = os.path.join(SRC_IMG_DIR, f"{c+1:02d}_mock_bird")
    os.makedirs(cls_dir, exist_ok=True)
    tgt_dir = os.path.join(TGT_IMG_DIR, f"{c+1:02d}_mock_bird")
    os.makedirs(tgt_dir, exist_ok=True)

train_lines, test_lines, tgt_lines = [], [], []

for c in range(num_classes):
    tint = class_tints[c]
    # source train
    for i in range(train_per_class):
        arr = np.random.randint(0, 120, (64, 64, 3), dtype=np.uint8)
        arr = arr.astype(float) * 0.5 + np.array(tint) * 0.5
        img = Image.fromarray(arr.astype(np.uint8))
        fname = f"{c+1:02d}_mock_bird/train_{i:03d}.jpg"
        img.save(os.path.join(SRC_IMG_DIR, fname))
        # 格式: img_path,cls_label,top_x,top_y,btm_x,btm_y (label 从 1 开始)
        train_lines.append(f"{fname},{c+1},0,0,64,64")
    # source test
    for i in range(test_per_class):
        arr = np.random.randint(0, 120, (64, 64, 3), dtype=np.uint8)
        arr = arr.astype(float) * 0.5 + np.array(tint) * 0.5
        img = Image.fromarray(arr.astype(np.uint8))
        fname = f"{c+1:02d}_mock_bird/test_{i:03d}.jpg"
        img.save(os.path.join(SRC_IMG_DIR, fname))
        test_lines.append(f"{fname},{c+1},0,0,64,64")
    # target (painting 风格: 低分辨率+强色块)
    for i in range(tgt_per_class):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        arr = arr.astype(float) * 0.3 + np.array(tint) * 0.7
        img = Image.fromarray(arr.astype(np.uint8)).resize((64, 64), Image.BILINEAR)
        fname = f"{c+1:02d}_mock_bird/tgt_{i:03d}.jpg"
        img.save(os.path.join(TGT_IMG_DIR, fname))
        # target 格式: img_path,cls_label
        tgt_lines.append(f"{fname},{c+1}")

with open(os.path.join(MOCK_ROOT, "cub_train.txt"), "w") as f:
    f.write("\n".join(train_lines) + "\n")
with open(os.path.join(MOCK_ROOT, "cub_test.txt"), "w") as f:
    f.write("\n".join(test_lines) + "\n")
with open(os.path.join(MOCK_ROOT, "cubp_test.txt"), "w") as f:
    f.write("\n".join(tgt_lines) + "\n")

# classes.txt: 每类类名必须唯一 (解析时去掉前 4 字符 "001.")
# "001.mock_bird_01" -> "mock bird 01"
with open(os.path.join(MOCK_ROOT, "CUB_200_2011", "classes.txt"), "w") as f:
    lines = [f"{c+1} {c+1:03d}.mock_bird_{c+1:02d}" for c in range(num_classes)]
    f.write("\n".join(lines) + "\n")

print("Mock 数据集生成完成:")
print(f"  类别数: {num_classes}")
print(f"  source train: {len(train_lines)} 张")
print(f"  source test: {len(test_lines)} 张")
print(f"  target test: {len(tgt_lines)} 张")
print(f"  位置: {MOCK_ROOT}")
