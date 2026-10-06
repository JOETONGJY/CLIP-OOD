"""
诊断: 图像特征 z 在 domain subspace B 上的能量占比
(解释 C4 为何降 1%: 若 z 在 B 上的投影能量极小, 抑制几乎不改变特征,
 C4 与 DDO 的差异主要来自双头/双DDO结构而非投影本身)
"""
import torch
from args import get_args
from reliability import compute_domain_diffs_and_scores
from prompts.prompt200new import source_text_prompts, target_text_prompts
import clip
from data.CUB.cub_data import Processed_CUB_Dataset  # 只为拿 classname 顺序? 不, 直接读 classes.txt

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device

# 类名(与 cub_data 相同解析)
with open("data/CUB/CUB_200_2011/classes.txt") as f:
    class_names = [x.split(" ")[1][4:].replace("_", " ").lower().rstrip() for x in f.readlines()]

model, _ = clip.load(args.CLIP_type, device=device)
model.eval()

diffs, scores = compute_domain_diffs_and_scores(
    model, source_text_prompts, target_text_prompts, class_names, device)

dirs = diffs.mean(dim=1).float()
dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
U, S, Vh = torch.linalg.svd(dirs, full_matrices=False)
ev = S ** 2
cum = torch.cumsum(ev, 0) / ev.sum()
tau_t = torch.tensor(0.8, device=cum.device)
m = int(torch.searchsorted(cum, tau_t).item()) + 1
B = Vh[:m].T.contiguous()  # (768, m)
print(f"m(tau=0.8) = {m}")
B = B.to(device)

# 图像特征在 B 上的能量
d = torch.load("cache/CUB_train.pt", map_location="cpu")
z = d["feats"].float().to(device)
z = z / z.norm(dim=-1, keepdim=True)
proj = (z @ B) @ B.T
energy = (proj * proj).sum(-1) / (z * z).sum(-1)   # z 归一化后分母=1
cos = (proj * z).sum(-1) / (proj.norm(dim=-1) * z.norm(dim=-1))
print(f"z 在 B 上能量占比: mean={energy.mean():.4f}  p90={energy.quantile(0.9):.4f}  max={energy.max():.4f}")
print(f"投影方向与 z 的余弦: mean={cos.mean():.4f}")

# target 域(油画)特征同样算一遍
dt = torch.load("cache/CUB_target_test.pt", map_location="cpu")
zt = dt["feats"].float().to(device)
zt = zt / zt.norm(dim=-1, keepdim=True)
projt = (zt @ B) @ B.T
energyt = (projt * projt).sum(-1) / (zt * zt).sum(-1)
print(f"target(painting) 能量占比: mean={energyt.mean():.4f}")
print(f"source-target 投影能量差(域信号强度): {energyt.mean()-energy.mean():+.4f}")
