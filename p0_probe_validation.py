"""P0 验证1: 固定概念投影到底损失了多少判别信息 + 子空间随机对照

三个对照 (相同训练划分、相同优化条件):
  A. Linear Probe  — 输入原始 CLIP 特征 z (768维) → 线性头
  B. CBM Probe     — 输入概念激活 a = z·Eᵀ (312维) → 线性头
  C. Adapted CBM   — 校准后概念激活 (本验证先不做, 留接口)

附加严谨性修正 (专家指出):
  - 域子空间能量 3-9% 必须与随机 32 维子空间对照 (期望 32/768=4.17%)
  - 补测: 图像特征在概念矩阵 E 张成子空间中的能量

协议: 源域验证集选模型, 目标域仅最终测试 (回应专家的协议批评)
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip as clip_lib
from args import get_args

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
torch.manual_seed(0); np.random.seed(0)

DS = os.environ.get("PROBE_DS", "CUB")
CONCEPTS = {"CUB": "data/CUB/cub_concepts.txt", "AWA2": "data/awa2/awa_concept.txt"}
CKPT = {"CUB": "best_CUB.pth", "AWA2": "best_AWA2.pth"}

def main():
    print(f"===== P0 验证1: {DS} =====")
    train = torch.load(f"cache/{DS}_train.pt", map_location="cpu")
    src = torch.load(f"cache/{DS}_source_test.pt", map_location="cpu")
    tgt = torch.load(f"cache/{DS}_target_test.pt", map_location="cpu")

    # 概念嵌入
    model, _ = clip_lib.load(args.CLIP_type, device=device)
    model.eval()
    concepts = [l.strip() for l in open(CONCEPTS[DS], encoding="utf-8") if l.strip()]
    with torch.no_grad():
        E = model.encode_text(clip_lib.tokenize(concepts, truncate=True).to(device)).float()
        E = E / E.norm(dim=-1, keepdim=True)
    del model; torch.cuda.empty_cache()
    K = len(concepts)
    print(f"K = {K} 概念")

    def prep(d):
        z = d["feats"].float().to(device)
        z = z / z.norm(dim=-1, keepdim=True)
        return z, d["labels"].to(device)

    z_tr, y_tr = prep(train); z_sv, y_sv = prep(src); z_tg, y_tg = prep(tgt)

    # ===== 附加诊断: 子空间能量对照 (专家修正) =====
    print("\n===== 子空间能量对照 =====")
    # 1) 概念矩阵 E 张成的子空间
    # E 是 (K, 768) 行向量组; 其张成子空间的正交基
    U_e, _, _ = torch.linalg.svd(E.T, full_matrices=False)  # (768, rank)
    P_e = U_e @ U_e.T
    e_concept_src = ((z_sv @ P_e) * z_sv).sum(-1).mean().item()
    e_concept_tgt = ((z_tg @ P_e) * z_tg).sum(-1).mean().item()
    print(f"图像特征在概念子空间 span(E) 能量: src {e_concept_src:.4f} | tgt {e_concept_tgt:.4f}  (rank={U_e.shape[1]})")

    # 2) 域子空间 B (从 checkpoint 提取) + 随机对照
    if os.path.exists(CKPT[DS]):
        state = torch.load(CKPT[DS], map_location="cpu")
        if "domain_basis" in state:
            B = state["domain_basis"].float().to(device)  # (768, m)
            m = B.shape[1]
            e_dom_src = ((z_sv @ B) ** 2).sum(-1).mean().item()
            e_dom_tgt = ((z_tg @ B) ** 2).sum(-1).mean().item()
            print(f"图像特征在域子空间 B 能量:  src {e_dom_src:.4f} | tgt {e_dom_tgt:.4f}  (m={m})")
            # 随机 m 维正交子空间对照 (100 次平均)
            rand_es, rand_et = [], []
            g = torch.Generator(device='cpu').manual_seed(0)
            for _ in range(100):
                G = torch.randn(768, m, generator=g)
                Qr, _ = torch.linalg.qr(G)
                Qr = Qr.to(device)
                rand_es.append(((z_sv @ Qr) ** 2).sum(-1).mean().item())
                rand_et.append(((z_tg @ Qr) ** 2).sum(-1).mean().item())
            print(f"随机 {m} 维子空间对照 (100次):  src {np.mean(rand_es):.4f}±{np.std(rand_es):.4f} | tgt {np.mean(rand_et):.4f}±{np.std(rand_et):.4f}")

    # ===== 主对照: Linear Probe vs CBM Probe =====
    print("\n===== Linear Probe 对照 (源域选 epoch, 目标域仅测试) =====")

    def train_probe(X_tr, y_tr, X_val, y_val, X_tg, y_tg, dim, tag, epochs=100):
        """线性头; 每 epoch 在源域验证集上看, 记录源域最优时对应的目标域成绩"""
        head = nn.Linear(dim, int(y_tr.max()) + 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        n = len(X_tr)
        best_val, tg_at_best = 0.0, 0.0
        for ep in range(epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                loss = F.cross_entropy(head(X_tr[idx]), y_tr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                val_acc = (head(X_val).argmax(1) == y_val).float().mean().item()
                if val_acc > best_val:
                    best_val = val_acc
                    tg_at_best = (head(X_tg).argmax(1) == y_tg).float().mean().item()
        return best_val * 100, tg_at_best * 100

    C = int(y_tr.max()) + 1

    # A: Linear Probe (原始 z, 768 维)
    a_val, a_tg = train_probe(z_tr, y_tr, z_sv, y_sv, z_tg, y_tg, 768, "Linear(z)")
    print(f"A. Linear Probe (z, 768d):    源域val {a_val:.2f} | 目标域 {a_tg:.2f}")

    # B: CBM Probe (概念激活, K 维)
    a_tr = z_tr @ E.T; a_sv = z_sv @ E.T; a_tg = z_tg @ E.T
    b_val, b_tg = train_probe(a_tr, y_tr, a_sv, y_sv, a_tg, y_tg, K, "CBM(a)")
    print(f"B. CBM Probe (a=zEᵀ, {K}d):  源域val {b_val:.2f} | 目标域 {b_tg:.2f}")

    gap_src = float(a_val) - float(b_val)
    gap_tgt = float(a_tg) - float(b_tg)
    a_val, a_tg, b_val, b_tg = float(a_val), float(a_tg), float(b_val), float(b_tg)
    print(f"\n概念投影信息损失: 源域 {gap_src:+.2f} | 目标域 {gap_tgt:+.2f}")
    print("(若 B 明显低于 A, 固定概念投影确实损失判别信息 → 支持概念校准路线)")

    out = {"dataset": DS, "K": K,
           "linear_val": a_val, "linear_tgt": a_tg,
           "cbm_val": b_val, "cbm_tgt": b_tg,
           "gap_src": gap_src, "gap_tgt": gap_tgt,
           "energy_concept_src": e_concept_src, "energy_concept_tgt": e_concept_tgt}
    os.makedirs("downloads", exist_ok=True)
    json.dump(out, open(f"downloads/probe_{DS}.json", "w"), indent=2)
    print(f"\n已保存 → downloads/probe_{DS}.json")

if __name__ == "__main__":
    main()
