"""方案 A1: 局部语义关系矩阵约束 (语义锚定低秩校准·最小版)

动机 (E2 旧版 + F4 决策规则): P4 校准提升精度但概念-标注相关性下降——
Frobenius 锚定 ‖ΔE‖²_F 只约束总体修改量, 不约束概念间语义结构。

本模块在损失内部施加"局部 Gram 关系约束": 对原始词典 E 中互为 top-k 邻居的
概念对 (i,j), 约束校准后 E' 的对应余弦相似度不远离原值:

    L_rel = (1/|N|) Σ_{(i,j)∈N} (cos(E'_i, E'_j) − cos(E_i, E_j))²

设计决策 (外审意见落实):
  * 归一化只在损失内部进行 —— 不修改模型前向的 E' (不 L2 归一化), 保证消融干净;
  * 仅约束 top-k 局部邻居 (E 的原始语义邻域) —— 允许全局几何旋转,
    但禁止局部邻域结构被破坏 (注意: Gram 约束对正交旋转不变, 它保"关系"不保"对应",
    对应关系由 A2 的响应锚定补足);
  * 无新参数、无人工标注依赖、不改冻结 CLIP。

用法 (在 clip_cbm_subspace 上启用):
    from model.semantic_anchor import SemanticAnchor
    anchor = SemanticAnchor(model.concept_embeddings, topk=5).to(model.device)
    E_eff = model._effective_E()
    loss_rel = anchor(E_eff)
    total_loss += lambda_rel * loss_rel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAnchor(nn.Module):
    """局部语义关系约束: E' 相对 E 的 top-k 邻域余弦结构保持"""

    def __init__(self, E, topk=5):
        super().__init__()
        E_ref = F.normalize(E.detach().float(), dim=-1)          # (K, 768)
        sim = E_ref @ E_ref.T
        sim.fill_diagonal_(float('-inf'))
        neighbors = sim.topk(topk, dim=1).indices                # (K, topk)
        # 原始邻域余弦值 (常数目标)
        old_sim = (E_ref[:, None, :] * E_ref[neighbors]).sum(-1)  # (K, topk)
        self.register_buffer('E_ref', E_ref)
        self.register_buffer('neighbors', neighbors)
        self.register_buffer('old_sim', old_sim)

    def forward(self, E_cal):
        """E_cal: 未归一化的校准词典 (模型前向原样); 归一化仅在此内部"""
        E_new = F.normalize(E_cal.float(), dim=-1)
        new_sim = (E_new[:, None, :] * E_new[self.neighbors]).sum(-1)  # (K, topk)
        return F.mse_loss(new_sim, self.old_sim)

    @torch.no_grad()
    def neighborhood_stability(self, E_cal, ks=(5, 10)):
        """E3 指标: 校准前后 top-k 邻居重合率 (逐概念均值)
        注意: k>topk 时锚只存了 topk 个参考邻居, 重合率分母仍为 k —— 结果解释要小心;
        传 ks ⊆ {1..topk} 才是严格的"锚定邻域重合率"。"""
        E_new = F.normalize(E_cal.float(), dim=-1)
        sim_new = E_new @ E_new.T
        sim_new.fill_diagonal_(float('-inf'))
        out = {}
        for k in ks:
            new_nb = sim_new.topk(k, dim=1).indices               # (K, k)
            ref_nb = self.neighbors[:, :min(k, self.neighbors.size(1))]
            overlap = torch.zeros(E_new.size(0), device=E_new.device)
            for r in range(E_new.size(0)):
                overlap[r] = len(set(new_nb[r].tolist()) & set(ref_nb[r].tolist())) / k
            out[f'top{k}_overlap'] = float(overlap.mean())
        return out
