"""P0 升级版: Text–Visual Domain Geometry Diagnosis (不预设错位)

视觉侧用 class-conditional shift (专家指定):
  Δ_c = μ_{target,c} − μ_{source,c}  (逐类, 从特征缓存直接计算)
  D_real = [Δ_1..Δ_C] 归一化 → SVD → B_real
语言侧: B_text = SVD(204 prompt 类平均方向, τ=0.8)

四指标:
  1. Principal Angles   cosθ_i = σ_i(SVD(B_text^T B_real))
  2. Overlap            ‖B_text^T B_real‖_F^2 / m
  3. R_shift            ‖P_text D_real‖_F^2 / ‖D_real‖_F^2  (cross reconstruction)
  4. ER = R_shift / E_base  (enrichment ratio; E_base = 源域普通特征在 B_text 的能量占比)

B_real 只用于诊断 (事后分析), 不进任何训练 —— 严格 DG 不受影响。
用法: python modality_misalignment2.py
"""
import csv
import glob
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

DATASETS = {
    "CUB":        ("data/CUB/CUB_200_2011/classes.txt", "idname_lower",
                   "cache/CUB_train.pt", "cache/CUB_target_test.pt"),
    "AwA2":       ("data/awa2/classes.txt", "idname",
                   "cache/AWA2_train.pt", "cache/AWA2_target_test.pt"),
    "LADA":       ("data/LAD_animal/class.txt", "idname",
                   "cache/LADA_train.pt", "cache/LADA_target_test.pt"),
    "LADV":       ("data/LAD_vehicle/class.txt", "idname",
                   "cache/LADV_train.pt", "cache/LADV_target_test.pt"),
    "PACS":       ("data/pacs/classes.txt", "idname_asis",
                   "cache/PACS_train.pt", "cache/PACS_target_test.pt"),
    "OfficeHome": ("data/pacs/officehome_classes.txt", "idname_asis_us",
                   "cache/OfficeHome_train.pt", "cache/OfficeHome_target_test.pt"),
    "DomainNet":  ("data/domainnet/dn_classes.txt", "idname",
                   "cache/DomainNet_train.pt", "cache/DomainNet_target_test.pt"),
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
        else:
            parts = x.split(" ", 1)
            n = parts[1] if len(parts) > 1 else parts[0]
            if style == "idname_asis_us":
                n = n.replace("_", " ")
            names.append(n.rstrip())
    return names


def svd_subspace(rows, tau=0.8, cap=None):
    """rows: (n, 768) 已归一化 → 返回 B (768, m) 及 m"""
    U, S, Vh = torch.linalg.svd(rows, full_matrices=False)
    ev = S ** 2
    cum = torch.cumsum(ev, 0) / ev.sum()
    m = int(torch.searchsorted(cum, torch.tensor(tau, device=cum.device)).item()) + 1
    if cap:
        m = min(m, cap)
    return Vh[:m].T.contiguous(), m


@torch.no_grad()
def class_conditional_real(src_cache, tgt_cache):
    """逐类均值差 → D_real (C,768) 归一化; 返回 None 若某类在 target 无样本"""
    s = torch.load(src_cache, map_location="cpu")
    t = torch.load(tgt_cache, map_location="cpu")
    sf, sl = s["feats"].float(), s["labels"]
    tf, tl = t["feats"].float(), t["labels"]
    n_cls = int(max(sl.max(), tl.max())) + 1
    deltas = []
    for c in range(n_cls):
        ms = sf[sl == c]
        mt = tf[tl == c]
        if len(ms) == 0 or len(mt) == 0:
            continue
        d = mt.mean(0) - ms.mean(0)
        deltas.append(d / (d.norm() + 1e-8))
    if len(deltas) < 5:
        return None, 0
    return torch.stack(deltas).to(device), len(deltas)


def main():
    rows_out = []
    print(f'{"数据集":12s} {"m_t":>4s} {"m_r":>4s} {"C_eff":>5s} {"angles(前4)":>28s} {"overlap":>8s} '
          f'{"R_shift":>8s} {"E_base":>7s} {"ER":>6s}')
    for ds, (clsfile, style, src, tgt) in DATASETS.items():
        try:
            if not (os.path.exists(src) and os.path.exists(tgt)):
                print(f"{ds}: 缓存缺失, 跳过")
                continue
            names = parse_classes(clsfile, style)
            diffs, _ = compute_domain_diffs_and_scores(
                model, source_text_prompts, target_text_prompts, names, device, verbose=False)
            dirs = diffs.mean(dim=1).float()
            dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
            B_text, m_t = svd_subspace(dirs)

            D_real, C_eff = class_conditional_real(src, tgt)
            if D_real is None:
                print(f"{ds}: 类条件 shift 不可用")
                continue
            m_cap = min(32, C_eff - 1)
            B_real, m_r = svd_subspace(D_real, cap=m_cap)

            # 1) principal angles
            sv = torch.linalg.svdvals(B_text.T @ B_real)
            # 2) overlap
            m_min = min(m_t, m_r)
            overlap = (torch.norm(B_text.T @ B_real, p="fro") ** 2 / m_min).item()
            # 3) R_shift: 文本子空间对真实类条件迁移的重构比
            proj = (B_text @ B_text.T) @ D_real.T
            r_shift = (torch.norm(proj, p="fro") ** 2 / torch.norm(D_real.T, p="fro") ** 2).item()
            # 4) E_base → ER
            z = torch.load(src, map_location="cpu")["feats"].float().to(device)
            z = z / z.norm(dim=-1, keepdim=True)
            e_base = (((z @ B_text) @ B_text.T * z).sum(-1)).mean().item()
            er = r_shift / e_base

            angs = " ".join(f"{s:.3f}" for s in sv[:4].tolist())
            print(f"{ds:12s} {m_t:4d} {m_r:4d} {C_eff:5d} {angs:>28s} {overlap:8.3f} "
                  f"{r_shift:8.3f} {e_base:7.3f} {er:6.2f}")
            rows_out.append([ds, m_t, m_r, C_eff,
                             ";".join(f"{s:.4f}" for s in sv[:8].tolist()),
                             f"{overlap:.4f}", f"{r_shift:.4f}", f"{e_base:.4f}", f"{er:.3f}"])
        except Exception as e:
            print(f"{ds}: 失败 {e}")

    with open("downloads/modality_diagnosis_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "m_text", "m_real", "C_effective", "principal_angles_8",
                    "overlap", "R_shift", "E_base", "ER"])
        w.writerows(rows_out)
    print("完成 → downloads/modality_diagnosis_v2.csv")


if __name__ == "__main__":
    main()
