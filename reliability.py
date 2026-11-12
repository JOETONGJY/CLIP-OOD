"""
Reliability-Weighted DDO: 可复用的可靠性计算与权重生成模块。

设计目标
--------
把"可靠性分数计算 + 权重生成"从各数据集 Dataset.__init__ 中剥离出来，
所有数据集 (CUB / AWA2 / LADA / LADV / ...) 共用同一份逻辑，
切换权重策略 (手调二值 / 连续软加权 / 可学习 logits) 只改本文件一处。

使用方式
--------
    from reliability import attach_domain_reliability

    # 在 Dataset.__init__ 中, 算完 classname2id 之后:
    attach_domain_reliability(
        self, clip_model, src_dm_texts, tgt_dm_texts,
        class_names=list(self.classname2id.keys()),
        device=device,
        weight_strategy="prior_residual",  # "binary"|"soft"|"prior_residual"
        binary_high=1.8, binary_low=0.75,  # 仅 "binary" 时生效
    )

    # 之后 self.domain_diffs / self.domain_weights / self.reliability_scores 自动可用
    # prior_residual 策略下, domain_weights = reliability_scores (作为 prior 透传给 model)
"""
from typing import List, Optional
import torch
from tqdm import tqdm


# ============================================================
# 1. 核心: 计算 domain_diffs + reliability_scores (纯函数, 无副作用)
# ============================================================
def compute_domain_diffs_and_scores(
    clip_model,
    src_dm_texts: List[str],
    tgt_dm_texts: List[str],
    class_names: List[str],
    device,
    verbose: bool = True,
):
    """
    对每个 (source_prompt, target_prompt) pair:
      - 编码所有类别的 source/target 文本嵌入
      - 做差并归一化得到域迁移方向 normed_diffs
      - 用类间方向一致性衡量 reliability_score

    Returns
    -------
    domain_diffs : torch.Tensor, shape (num_prompts, num_classes, feat_dim)
    reliability_scores : torch.Tensor, shape (num_prompts,)
    """
    diffs_list = []
    scores_list = []

    # 延迟 import: 避免模块加载时触发 utils.py -> import clip (clip 库需要 GPU 环境)
    from utils import get_domain_text_embs

    iterator = zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)
    if verbose:
        iterator = tqdm(iterator, total=len(tgt_dm_texts), desc="Computing Domain Diffs + Reliability")

    for src_prompt, tgt_prompt in iterator:
        if verbose:
            tqdm.write(f"{tgt_prompt} - {src_prompt}")

        source_embeddings, target_embeddings = get_domain_text_embs(
            clip_model,
            [src_prompt],
            [tgt_prompt],
            class_names,
            device,
        )

        # normalize embeddings
        source_embeddings = source_embeddings / (source_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
        target_embeddings = target_embeddings / (target_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # raw diff -> normalized diff (域迁移方向)
        raw_diffs = target_embeddings.float() - source_embeddings.float()
        if raw_diffs.norm() == 0:
            print(f"Warning: zero diff detected for '{tgt_prompt}' - '{src_prompt}'")
        normed_diffs = raw_diffs / (raw_diffs.norm(dim=-1, keepdim=True) + 1e-8)

        # reliability score: 类间迁移方向的平均 cosine similarity
        # 同一 prompt 在不同类别上产生的迁移方向越一致, reliability 越大
        cos_sim = torch.matmul(normed_diffs, normed_diffs.T)  # (num_classes, num_classes)
        mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
        reliability_score = cos_sim[mask].mean()

        diffs_list.append(normed_diffs)
        scores_list.append(reliability_score)

    domain_diffs = torch.stack(diffs_list, dim=0).to(device)
    reliability_scores = torch.stack(scores_list).to(device)

    return domain_diffs, reliability_scores


# ============================================================
# 2. 权重生成策略
# ============================================================
def make_weights_binary(scores: torch.Tensor, high: float = 1.8, low: float = 0.75) -> torch.Tensor:
    """
    [旧策略, 保留用于复现] median 二值化加权。
    问题: 丢失连续信息, high/low 需手调, 跨数据集不泛化。
    """
    threshold = scores.median()
    return torch.where(
        scores >= threshold,
        torch.tensor(high, device=scores.device, dtype=scores.dtype),
        torch.tensor(low, device=scores.device, dtype=scores.dtype),
    )


def make_weights_soft(scores: torch.Tensor) -> torch.Tensor:
    """
    [方案 1] 连续软加权, 零超参。
    softmax 归一后乘 N 使均值为 1, 保持 orth_loss 整体量级。
    """
    return torch.softmax(scores, dim=0) * len(scores)


def make_weights_prior_residual(scores: torch.Tensor) -> torch.Tensor:
    """
    [专家建议: Prior + Residual] 返回 reliability_scores 原值作为 prior。
    model 侧会构造 logits = prior(detach) + residual(Parameter),
    residual 初始化为 0, 训练初期 logits=prior, 随训练逐步修正。
    这里只负责透传 scores, 可学习性在 model/cbm_models.py 中实现。
    """
    return scores.clone().detach()


# 旧名保留向后兼容 (已被 prior_residual 替代)
make_weights_learnable_init = make_weights_prior_residual


# ============================================================
# 2b. C-1 描述符冗余度 (2026-11-09, 方案 C: 去重复计权)
# ============================================================
def compute_descriptor_redundancy(
    domain_diffs: torch.Tensor,
    topk: int = 0,
    relative: bool = True,
) -> torch.Tensor:
    """
    描述符冗余度 c_j: 方向 j 与全部其他方向的方向相似度之和。
    冗余度高 = 该方向的表达被很多其他描述符重复 → 在 DDO 中应降低单独话语权。

    Parameters
    ----------
    domain_diffs : (N, C, D) 域迁移方向 (已逐类归一化)
    topk : int, 0=全矩阵求和; >0 时每行只累加相似度最高的 k 个方向
           (外审建议: 避免大量弱相关方向累积成过大的冗余惩罚)
    relative : bool, True (2026-11-12 修正) 时 c_j 除以中位数, 得到相对冗余度
           —— 旧版(绝对量)的缺陷: CLIP 文本方向天然挤在一起, 绝对 c 基数 ~72,
              注入一个重复描述符仅增加 1/72 的惩罚, 公式对真重复钝化。
           相对量下天然数据 c̃≈1 (惩罚≈0, 严格退化回 RW-DDO),
           注入重复时重复对 c̃ 明显上升, 惩罚具有分辨力。

    Returns
    -------
    redundancy : (N,) relative=True 时 c̃_j 以中位数为 1; False 时绝对量 ∈ [1, N]
    """
    with torch.no_grad():
        dirs = domain_diffs.mean(dim=1).float()               # (N, D) 类平均方向
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        # 绝对余弦: DDO 正交约束中 d 与 -d 是同一一维子空间
        A = (dirs @ dirs.T).abs()
        A.fill_diagonal_(1.0)
        if topk and topk > 0:
            # 保留每行 top-k (含自身), 其余置 0 —— 稀疏化冗余统计
            kth = A.topk(min(topk, A.size(0)), dim=1).values[:, -1:]
            A = torch.where(A >= kth, A, torch.zeros_like(A))
        c = A.sum(dim=1).clamp_min(1e-8)
        if relative:
            c = c / c.median().clamp_min(1e-8)
        return c


# ============================================================
# 3. 对外统一入口: 一次性挂载到 Dataset 上
# ============================================================
def attach_domain_reliability(
    dataset,
    clip_model,
    src_dm_texts: Optional[List[str]],
    tgt_dm_texts: Optional[List[str]],
    class_names: List[str],
    device,
    weight_strategy: str = "soft",
    binary_high: float = 1.8,
    binary_low: float = 0.75,
    verbose: bool = True,
):
    """
    把 domain_diffs / domain_weights / reliability_scores 挂到 dataset 上。
    若 src_dm_texts 或 tgt_dm_texts 为 None, 则不挂载 (与原逻辑一致)。

    Parameters
    ----------
    dataset : Dataset
        被 patch 的 dataset 对象, 会设置三个属性:
          - dataset.domain_diffs       (num_prompts, num_classes, feat_dim)
          - dataset.reliability_scores (num_prompts,)
          - dataset.domain_weights     (num_prompts,)  根据 strategy 生成
    weight_strategy : str
        "binary"         : 旧的手调二值化 (复现用)
        "soft"           : 连续软加权 (方案 1, 零超参)
        "prior_residual" : 透传 scores 作为 prior, model 侧构造 prior+residual (专家建议主方法)
        "learnable"      : 同 "prior_residual" 的旧名, 向后兼容
    """
    if src_dm_texts is None or tgt_dm_texts is None:
        return

    # ---- 缓存: 大类数数据集 (如 DomainNet 345 类) 需 ~3 小时, 必须缓存 ----
    import hashlib, os
    cache_key = hashlib.md5(
        ("\u0000".join(sorted(class_names)) + "\u0000" + "\u0000".join(tgt_dm_texts)
         ).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = "cache/reliability"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"reliability_{cache_key}.pt")

    if os.path.exists(cache_path):
        if verbose:
            print(f"[reliability cache] hit: {cache_path}")
        cached = torch.load(cache_path, map_location=device)
        domain_diffs = cached["diffs"].to(device)
        reliability_scores = cached["scores"].to(device)
    else:
        if verbose:
            print(f"[reliability cache] miss, computing... ({len(class_names)} classes)")
        domain_diffs, reliability_scores = compute_domain_diffs_and_scores(
            clip_model, src_dm_texts, tgt_dm_texts, class_names, device, verbose=verbose
        )
        torch.save(
            {"diffs": domain_diffs.cpu(), "scores": reliability_scores.cpu()},
            cache_path,
        )
        if verbose:
            print(f"[reliability cache] saved: {cache_path}")
        domain_diffs = domain_diffs.to(device)
        reliability_scores = reliability_scores.to(device)

    dataset.domain_diffs = domain_diffs
    dataset.reliability_scores = reliability_scores
    # C-1: 冗余度始终预计算并挂载 (开销极小), 是否生效由 model 侧 --redundancy 开关决定
    dataset.descriptor_redundancy = compute_descriptor_redundancy(
        domain_diffs, topk=getattr(dataset, "redundancy_topk", 0)
    )

    if weight_strategy == "binary":
        dataset.domain_weights = make_weights_binary(reliability_scores, binary_high, binary_low)
    elif weight_strategy == "soft":
        dataset.domain_weights = make_weights_soft(reliability_scores)
    elif weight_strategy in ("prior_residual", "learnable"):
        dataset.domain_weights = make_weights_prior_residual(reliability_scores)
    else:
        raise ValueError(
            f"Unknown weight_strategy: {weight_strategy}. "
            f"Expected one of: 'binary', 'soft', 'prior_residual', 'learnable'."
        )
