"""六数据集能量占比统一分析: 每个数据集的 source/target 图像特征
在"该数据集类名诱导的语言域子空间 B(τ=0.8)"上的能量占比"""
import os
import torch
import clip
from args import get_args
from reliability import compute_domain_diffs_and_scores
from prompts.prompt200new import source_text_prompts, target_text_prompts

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
model, _ = clip.load(args.CLIP_type, device=device)
model.eval()

CONFIG = {
    "CUB":  {"classes": ("data/CUB/CUB_200_2011/classes.txt", "idname_lower"),
             "src": "cache/CUB_train.pt", "tgt": "cache/CUB_target_test.pt", "tgt_name": "painting"},
    "AwA2": {"classes": ("data/awa2/classes.txt", "name_asis"),
             "src": "cache/AWA2_train.pt", "tgt": "cache/AWA2_target_test.pt", "tgt_name": "clipart"},
    "LADA": {"classes": ("data/LAD_animal/class.txt", "idname"),
             "src": "cache/LADA_train.pt", "tgt": "cache/LADA_target_test.pt", "tgt_name": "sculpture"},
    "LADV": {"classes": ("data/LAD_vehicle/class.txt", "idname"),
             "src": "cache/LADV_train.pt", "tgt": "cache/LADV_target_test.pt", "tgt_name": "3D"},
    "PACS": {"classes": ("data/pacs/classes.txt", "idname_asis"),
             "src": "cache/PACS_train.pt", "tgt": "cache/PACS_target_test.pt", "tgt_name": "art_painting"},
    "OfficeHome": {"classes": ("data/pacs/officehome_classes.txt", "idname_asis_us"),
                   "src": "cache/OfficeHome_train.pt", "tgt": "cache/OfficeHome_target_test.pt", "tgt_name": "Art"},
}

def parse_classes(path, style):
    names = []
    for x in open(path, encoding="utf-8"):
        x = x.strip()
        if not x:
            continue
        if style == "idname_lower":
            names.append(x.split(" ")[1][4:].replace("_", " ").lower())
        elif style == "idname":
            names.append(x.split(" ")[1].rstrip())
        elif style == "name_asis":
            names.append(x.rstrip())
        else:  # idname_asis / idname_asis_us
            parts = x.split(" ", 1)
            n = parts[1] if len(parts) > 1 else parts[0]
            if style == "idname_asis_us":
                n = n.replace("_", " ")
            names.append(n.rstrip())
    return names

print(f'{"数据集":12s} {"m":>4s} {"source能量":>10s} {"target能量":>10s} {"target−source":>12s}')
results = {}
for ds, cfg in CONFIG.items():
    try:
        names = parse_classes(*cfg["classes"])
        diffs, _ = compute_domain_diffs_and_scores(
            model, source_text_prompts, target_text_prompts, names, device, verbose=False)
        dirs = diffs.mean(dim=1).float()
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        U, S, Vh = torch.linalg.svd(dirs, full_matrices=False)
        ev = S ** 2
        cum = torch.cumsum(ev, 0) / ev.sum()
        tau_t = torch.tensor(0.8, device=cum.device)
        m = int(torch.searchsorted(cum, tau_t).item()) + 1
        B = Vh[:m].T.to(device)
        row = {}
        for key, path in [("src", cfg["src"]), ("tgt", cfg["tgt"])]:
            if not os.path.exists(path):
                row[key] = None
                continue
            z = torch.load(path, map_location="cpu")["feats"].float().to(device)
            z = z / z.norm(dim=-1, keepdim=True)
            proj = (z @ B) @ B.T
            row[key] = float((proj * proj).sum(-1).mean().item())
        d_ts = (row["tgt"] - row["src"]) if (row["src"] is not None and row["tgt"] is not None) else None
        results[ds] = (row, m, d_ts)
        print(f'{ds:12s} {m:4d} {row["src"]:10.4f} {row["tgt"]:10.4f} {d_ts:+12.4f}')
    except Exception as e:
        print(f"{ds}: 失败 {e}")
torch.save(results, "downloads/energy_all_datasets.pt")
print("已存 downloads/energy_all_datasets.pt")
