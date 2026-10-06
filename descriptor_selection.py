"""算法 A: 可靠性-多样性联合域描述符选择 (Reliability-Diversity Descriptor Selection)

流水线:
  ① 构造候选池 (M≈600, 模板法诚实标注: 200+ 风格名词 × 3 模板)
  ② 全部候选计算可靠性 r_i 与类平均方向 d_i (与主方法同一度量)
  ③ 贪心选择 N=200: 逐个加入使 J(S) 增益最大的候选
       J(S) = mean_{i∈S}(r_i) − λ_rel · mean_{i≠j∈S}(A_ij)
  ④ P0 前置检验: B_selected vs B_original 对 B_real 的解释力对比
  ⑤ 输出 prompts/selected_descriptors.txt 供训练 (--prompt_file)

用法: python descriptor_selection.py CUB   (逐数据集; 可加 AWA2 LADV)
"""
import os
import sys

import torch
import clip
from args import get_args
from reliability import compute_domain_diffs_and_scores

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = args.device
model, _ = clip.load(args.CLIP_type, device=device)
model.eval()

# ============ 候选池: 风格名词表 (构造式, 论文中如实标注构造方式) ============
STYLE_NOUNS = [
    # 绘画媒介 (经典)
    "oil painting", "watercolor painting", "gouache painting", "acrylic painting",
    "tempera painting", "fresco painting", "encaustic painting", "ink wash painting",
    "sumi-e painting", "impressionist painting", "expressionist painting",
    "surrealist painting", "cubist painting", "baroque painting", "renaissance painting",
    "romantic painting", "abstract painting", "minimalist painting", "pointillist painting",
    "fauvist painting", "ukiyo-e painting", "persian miniature", "byzantine icon",
    "pop art painting", "op art painting", "pixel art", "digital painting",
    "matte painting", "plein air painting", "vanitas painting", "trompe-l'oeil painting",
    # 素描/制图
    "pencil sketch", "charcoal sketch", "graphite drawing", "conté drawing",
    "crayon drawing", "pastel drawing", "chalk drawing", "silverpoint drawing",
    "pen and ink drawing", "technical illustration", "architectural blueprint",
    "engineering diagram", "patent drawing", "anatomical illustration",
    "botanical illustration", "scientific diagram", "wireframe sketch",
    "gesture drawing", "figure study", "cartoon sketch", "caricature",
    "comic strip panel", "manga illustration", "anime illustration",
    "storyboard sketch", "doodle", "scribble", "line art", "coloring book page",
    # 版画
    "woodcut print", "linocut print", "etching", "engraving", "aquatint print",
    "drypoint print", "lithograph", "screen print", "serigraph", "risograph print",
    "letterpress print", "block print", "monotype print", "mezzotint",
    "vintage poster", "advertising poster", "propaganda poster", "movie poster",
    "concert poster", "travel poster", "tapestry", "embroidery", "cross-stitch",
    # 摄影处理
    "black and white photograph", "sepia photograph", "vintage photograph",
    "daguerreotype", "tintype photograph", "polaroid photo", "lomography photo",
    "infrared photograph", "x-ray photograph", "long exposure photograph",
    "double exposure photograph", "tilt-shift photograph", "macro photograph",
    "aerial photograph", "underwater photograph", "night photograph",
    "high contrast photograph", "overexposed photograph", "blurred photograph",
    "grainy film photo", "cyanotype print", "photogram", "radiograph",
    "hdr photograph", "bokeh photograph", "analog film photo", "disposable camera photo",
    # 数字/计算机
    "3D render", "cgi render", "voxel art", "low poly render", "isometric render",
    "wireframe render", "raytraced render", "clay render", "toon shaded render",
    "cel shaded render", "glitch art", "vaporwave artwork", "synthwave artwork",
    "circuit board pattern", "holographic image", "vector illustration",
    "flat design illustration", "neumorphic design", "papercraft design",
    # 雕塑/三维实体
    "marble sculpture", "bronze sculpture", "clay sculpture", "wood carving",
    "stone carving", "ice sculpture", "sand sculpture", "wire sculpture",
    "paper sculpture", "origami model", "papier-mâché sculpture",
    "ceramic figurine", "porcelain figurine", "jade carving", "ivory carving",
    "plaster cast", "resin model", "clay figurine", "terracotta statue",
    "totem pole", "moai statue", "bust sculpture", "relief carving",
    # 工艺/材质
    "stained glass window", "mosaic", "collage", "photo collage", "mixed media artwork",
    "decoupage", "quilling", "beadwork", "macramé", "knitted toy",
    "felt craft", "leather craft", "metal engraving", "wood burning art",
    "scratchboard art", "linocut", "rubber stamp art", "wax seal impression",
    "cake decoration", "sugar sculpture", "soap carving", "balloon animal",
    "lego model", "claymation still", "stop-motion puppet", "action figure",
    "miniature diorama", "dollhouse miniature", "model train scene",
    # 纹理/表面
    "watercolor texture", "oil paint texture", "canvas texture", "paper texture",
    "rust texture", "marble texture", "wood grain texture", "fabric texture",
    "denim texture", "silk texture", "velvet texture", "burlap texture",
    "concrete texture", "plaster texture", "foil texture", "neon sign",
    "chalkboard drawing", "whiteboard drawing", "sidewalk chalk art",
    "cave painting", "petroglyph", "hieroglyphic", "ancient rune",
    "constellation chart", "star map", "tarot card", "trading card",
    "postage stamp", "currency engraving", "album cover", "book cover illustration",
    "fairytale illustration", "children's book illustration", "art nouveau poster",
    "art deco poster", "constructivist poster", "swiss design poster",
    "brutalist graphic", "psychedelic poster", "graffiti", "street art mural",
    "spray paint art", "airbrush artwork", "stencil art", "sticker art",
]

TEMPLATES = ["a {} of a {}.", "a {}-style {}."]
# 注: 候选模板与主方法一致地以 "a ... of a {}." 为主, 变体模板提供表达多样性

N_SELECT = 200
LAMBDA_REL = 0.3      # J(S) 中多样性项相对权重 (归一化目标下的经验值)
SRC_PROMPT = "a photo of a {}."

DS_CLASSES = {
    "CUB":  ("data/CUB/CUB_200_2011/classes.txt", "idname_lower"),
    "AWA2": ("data/awa2/classes.txt", "idname"),
    "LADA": ("data/LAD_animal/class.txt", "idname"),
    "LADV": ("data/LAD_vehicle/class.txt", "idname"),
    "PACS": ("data/pacs/classes.txt", "idname_asis"),
}


def parse_classes(path, style):
    names = []
    for x in open(path, encoding="utf-8"):
        x = x.strip()
        if not x:
            continue
        if style == "idname_lower":
            names.append(x.split(" ")[1][4:].replace("_", " ").lower())
        else:
            names.append(x.split(" ")[1].rstrip())
    return names


def build_pool():
    pool = []
    for noun in STYLE_NOUNS:
        art = "an" if noun[0] in "aeiou" else "a"
        pool.append(f"{art} {noun} of a {{}}.")
        pool.append(f"{art} {noun}-style {{}}.")
    return pool


@torch.no_grad()
def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "CUB"
    clsfile, style = DS_CLASSES[ds]
    names = parse_classes(clsfile, style)
    pool = build_pool()
    print(f"[{ds}] 候选池 {len(pool)} 条, 类别 {len(names)}")

    # ① 逐候选计算可靠性 + 类平均方向 (复用主方法度量)
    from reliability import compute_domain_diffs_and_scores as crds
    # crds 接口按 prompt 列表逐个算; 池子大 → 分批
    diffs_list, scores_list = [], []
    B = 64
    for i in range(0, len(pool), B):
        batch = pool[i:i + B]
        d, s = crds(model, [SRC_PROMPT], batch, names, device, verbose=False)
        diffs_list.append(d.cpu())
        scores_list.append(s.cpu())
    diffs = torch.cat(diffs_list)          # (M, K, 768)
    r = torch.cat(scores_list)             # (M,)
    dirs = diffs.mean(dim=1)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    A = (dirs @ dirs.T).clamp(min=0)
    A.fill_diagonal_(0)
    print(f"[{ds}] 可靠性范围 [{r.min():.3f}, {r.max():.3f}] 均值 {r.mean():.3f}")

    # ② 贪心选择 (归一化目标: 提升均值质量, 压低成对冗余)
    M = len(pool)
    S = []
    in_S = torch.zeros(M, dtype=torch.bool)
    sum_r, sum_A = 0.0, 0.0
    for step in range(N_SELECT):
        n = len(S)
        # 边际增益: 加 j 后 J = (sum_r + r_j)/(n+1) - λ * (sum_A*2 + A_self)/(n(n+1))
        col_sum = A[:, ~in_S].sum(dim=1) if n == 0 else None
        best_j, best_gain = -1, -1e18
        for j in range(M):
            if in_S[j]:
                continue
            new_n = n + 1
            new_sum_r = sum_r + r[j].item()
            if n == 0:
                pen = 0.0
            else:
                cross = (A[j, in_S].sum()).item()
                new_sum_A = sum_A + cross          # 成对和(单侧)
                pen = LAMBDA_REL * new_sum_A / (new_n * (new_n - 1) / 2)
            gain = new_sum_r / new_n - pen
            if gain > best_gain:
                best_gain, best_j = gain, j
        S.append(best_j)
        in_S[best_j] = True
        sum_r += r[best_j].item()
        if n > 0:
            sum_A += A[best_j, in_S].sum().item() - 0  # 不含自身
        if (step + 1) % 50 == 0:
            sel_A = A[np.array(S)][:, np.array(S)]
            m = (sel_A.sum() - sel_A.trace()) / (len(S) * (len(S) - 1))
            print(f"  step {step+1}: J 内部质量 {r[in_S].mean():.3f}, 内部平均冗余 {m:.3f}")

    import numpy as np
    S = np.array(S)
    selected = [pool[j] for j in S]
    sel_r = r[S]
    orig_mask = torch.tensor([p in set(pool) for p in selected])
    sel_A = A[S][:, S]
    inner_A = (sel_A.sum() - sel_A.trace()) / (N_SELECT * (N_SELECT - 1))
    # 原始 204 的冗余对照
    from prompts.prompt200new import target_text_prompts
    orig_idx = [pool.index(p) for p in target_text_prompts if p in pool]
    if orig_idx:
        oA = A[orig_idx][:, orig_idx]
        orig_inner = (oA.sum() - oA.trace()) / (len(orig_idx) * (len(orig_idx) - 1))
        orig_r = r[orig_idx]
        print(f"[对照] 原始204(池内命中{len(orig_idx)}): 质量 {orig_r.mean():.3f}, 冗余 {orig_inner:.3f}")
    print(f"[选择] N=200: 质量 {sel_r.mean():.3f}, 冗余 {inner_A:.3f}")

    os.makedirs("prompts", exist_ok=True)
    out = f"prompts/selected_descriptors_{ds}.txt"
    open(out, "w", encoding="utf-8").write("\n".join(selected))
    torch.save({"selected_idx": S, "r": r, "dirs": dirs, "A": A, "pool": pool},
               f"cache/descriptor_selection_{ds}.pt")
    print(f"已写出 {out} 与 cache/descriptor_selection_{ds}.pt")

    # ③ P0 前置检验: B_selected vs B_original 对 B_real 的解释力
    try:
        from modality_misalignment2 import svd_subspace, class_conditional_real
        src_c, tgt_c = {
            "CUB": ("cache/CUB_train.pt", "cache/CUB_target_test.pt"),
            "AWA2": ("cache/AWA2_train.pt", "cache/AWA2_target_test.pt"),
            "LADA": ("cache/LADA_train.pt", "cache/LADA_target_test.pt"),
            "LADV": ("cache/LADV_train.pt", "cache/LADV_target_test.pt"),
            "PACS": ("cache/PACS_train.pt", "cache/PACS_target_test.pt"),
        }[ds]
        D_real, C_eff = class_conditional_real(src_c, tgt_c)
        if D_real is not None:
            B_sel, m1 = svd_subspace(dirs[S].to(device), tau=0.8)
            from prompts.prompt200new import target_text_prompts as ORIG
            od, os_ = crds(model, [SRC_PROMPT], ORIG, names, device, verbose=False)
            od = od.mean(dim=1).float()
            od = od / (od.norm(dim=-1, keepdim=True) + 1e-8)
            B_ori, m2 = svd_subspace(od, tau=0.8)
            def pr(B, D):
                proj = (B @ B.T) @ D.T
                return (torch.norm(proj, p='fro')**2 / torch.norm(D.T, p='fro')**2).item()
            print(f"[P0检验] R_selected→real = {pr(B_sel, D_real):.4f} (m={m1})  "
                  f"vs  R_original→real = {pr(B_ori, D_real):.4f} (m={m2})")
    except Exception as e:
        print(f"[P0检验跳过] {e}")


if __name__ == "__main__":
    main()
