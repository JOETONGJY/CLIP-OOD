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

    domain_diffs, reliability_scores = compute_domain_diffs_and_scores(
        clip_model, src_dm_texts, tgt_dm_texts, class_names, device, verbose=verbose
    )

    dataset.domain_diffs = domain_diffs
    dataset.reliability_scores = reliability_scores

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
