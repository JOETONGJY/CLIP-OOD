import re
import matplotlib.pyplot as plt

# ====== 修改这里：你的日志文件路径 ======
file_path = r"C:\Users\pc\Desktop\毕业设计\实验\4.22-awa2-clsloss\新建 文本文档.txt"   # 或 .md

# ====== 输出图片名称 ======
output_png = "accuracy_curve.png"

# ====== 读取日志 ======
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# ====== 正则提取 ======
pattern = re.findall(
    r"Epoch\s+(\d+)\s+Summary:.*?"
    r"Train Acc:\s*([\d.]+)%.*?"
    r"Source Acc:\s*([\d.]+)%.*?"
    r"Target Acc:\s*([\d.]+)%",
    text,
    re.S
)

epochs = []
train_acc = []
source_acc = []
target_acc = []

for ep, tr, src, tgt in pattern:
    epochs.append(int(ep))
    train_acc.append(float(tr))
    source_acc.append(float(src))
    target_acc.append(float(tgt))

# ====== 检查数据 ======
if not epochs:
    raise ValueError("没有提取到数据，请检查日志格式！")

# ====== 画图 ======
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, source_acc, label="Source Acc")
plt.plot(epochs, target_acc, label="Target Acc")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")

plt.legend()
plt.grid(True)
plt.tight_layout()

# ====== 保存图片 ======
plt.savefig(output_png, dpi=300)

print(f"图已保存为: {output_png}")

# ====== 显示 ======
plt.show()
