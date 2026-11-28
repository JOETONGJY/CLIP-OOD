
import torch
import torchvision
import torch.nn as nn
import clip
import numpy as np
from tqdm import tqdm
import torch.nn.init as init

class Standard_cbm(nn.Module):
    def __init__(self, init_with_class_embedding=False):
        super(Standard_cbm, self).__init__()
        # clip model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, preprocess = clip.load('ViT-L/14', device=device)
        for p in self.clip_model.parameters(): p.requires_grad = False

        self.concept_embeddings = nn.Parameter(torch.empty(len(rival_atributes), 768))
        # 使用 Xavier 均匀初始化
        init.xavier_uniform_(self.concept_embeddings)

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len(rival_atributes)),
            nn.Linear(len(rival_atributes), 10)
        )

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        dev = visual_features.device
        concept_activations = visual_features @ self.concept_embeddings.T  # (bs,312)
        return concept_activations, self.classifier(concept_activations)

class clipzs(nn.Module):
    def __init__(self, args, prompts, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clipzs, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        class_texts = [[prompt.format(c) for prompt in prompts] for c in class_names]
        class_embeddings = []
        for texts in class_texts:
            texts_ = clip.tokenize(texts).to(self.device)
            class_embedding = self.clip_model.encode_text(texts_).float()
            class_embedding /= class_embedding.norm()
            class_embedding = class_embedding.mean(0)[None]
            class_embeddings.append(class_embedding)
        self.class_embeddings = torch.cat(class_embeddings)
        self.class_embeddings /= self.class_embeddings.norm()

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        sim = visual_features @ self.class_embeddings.T  # (bs,312)
        return sim


class clip_mlp(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_mlp, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)
        # classifer
        self.classifier = nn.Linear(feature_dim, len(class_names))

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)

        return None, self.classifier(visual_features), None

    def extract_cls_concept(self):

        return {}, {}

class clip_mlp_orth(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_mlp_orth, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)
        # classifer
        self.mapper = nn.Linear(feature_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, len(class_names))


    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        auxiliary_visual_features = self.mapper(visual_features)
        regularizer = self.mapper(self.diffs)

        return None, self.classifier(auxiliary_visual_features), regularizer

    def extract_cls_concept(self):
        return {}, {}

class clip_cbm(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_cbm, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len(concept_names)),
            nn.Linear(len(concept_names), len(class_names)))

        if init_with_class_embedding == True:
            with torch.no_grad():
                class_embeddings = self.clip_model.encode_text(clip.tokenize(class_names).to(device)).float()
                self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                self.classifier[2].weight = nn.Parameter(class_embeddings @ self.concept_embeddings.T)


    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        concept_activations = visual_features @ self.concept_embeddings.T  # (bs,312)
        return concept_activations, self.classifier(concept_activations), self.diffs

    def extract_cls_concept(self):
        asso_mat_last = self.classifier[2].weight
        topk_res_last = {}
        topk_res = {}
        # import pdb; pdb.set_trace()
        concept_names = np.array(self.concept_names)
        for i, cls_name in enumerate(self.class_names):

            topk_res_last[cls_name] = np.unique(concept_names[asso_mat_last[i].topk(10)[1].cpu().detach().numpy()]).tolist()

        return {},topk_res

class clip_cbm_orth(nn.Module):
    # 改动时间26.4.29-----------------------------------
    def __init__(
            self,
            args,
            class_names,
            concept_names,
            domain_diffs,
            domain_weights=None,
            weight_mode="prior_residual",
            init_with_class_embedding=False,
            descriptor_redundancy=None,
            use_redundancy=False,
            freeze_delta_mask=None,
            grouped_weights=False,
            descriptor_clusters=None,
            ca_prior=None,
            ca_beta=1.0,
            ddo_dirs='full',
            ddo_k=32,
            ddo_pca_src='ln0',
            ddo_dirs_seed=0
    ):
        # def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_cbm_orth, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        # C-1: 去冗余开关与冗余度向量 (buffer, 不参与梯度)
        self.use_redundancy = bool(use_redundancy)
        if descriptor_redundancy is not None:
            self.register_buffer('descriptor_redundancy',
                                 descriptor_redundancy.clone().float().detach())
        self.num_prompts = domain_diffs.size(0)

        # ===== 权重设计 (专家建议: Prior + Residual) =====
        # domain_weights 传入的是 reliability_scores (数据集侧统一 prior_residual 策略透传原始分数)
        # weight_mode 控制使用方式 (对应论文 4 组核心实验):
        #   "none"           : 不加权                     -> Original DDO (baseline)
        #   "fixed"          : softmax(scores)*N, 固定     -> Soft Reliability (无 residual, 无 KL)
        #   "residual"       : prior=0, 仅 residual 学习   -> Residual Only (无先验)
        #   "prior_residual" : prior=scores + residual    -> 主方法
        self.weight_mode = weight_mode

        if domain_weights is None or weight_mode == "none":
            self.reliability_prior = None
            self.domain_residual = None
        elif weight_mode == "fixed":
            # 固定软权重: softmax(scores)*N, buffer, 不可学习
            scores = domain_weights.clone().float().to(self.device).detach()
            fixed_w = torch.softmax(scores, dim=0) * self.num_prompts
            self.register_buffer('reliability_prior', scores)   # 仍保存 scores 供日志对比
            self.fixed_weights = fixed_w
            self.domain_residual = None
        elif weight_mode == "ca_fixed":
            # CA-Fixed (2026-11-24): 类别感知固定权重 w_{i,c} = N·softmax_i(r_i + β·Î_{i,c})
            # Î 尺度匹配到 r 的 std (用户要求: 控制交互项尺度与 softmax 集中度);
            # β=0 严格退化回 fixed 模式 (RW-DDO 固定权重)。无可学习残差, KL=0。
            if ca_prior is None:
                raise ValueError("ca_fixed 需要 --ca_prior_file 提供 (204, C) 交互矩阵")
            self.ca_beta = float(ca_beta)
            scores = domain_weights.clone().float().to(self.device).detach()
            ca = ca_prior.float().to(self.device)
            C = domain_diffs.size(1)
            if ca.shape != (self.num_prompts, C):
                raise ValueError(f"ca_prior 形状 {tuple(ca.shape)} != ({self.num_prompts}, {C})")
            scale = scores.std() / ca.std().clamp_min(1e-8)
            logits_ca = scores.unsqueeze(1) + self.ca_beta * ca * scale
            w_ca = torch.softmax(logits_ca, dim=0) * self.num_prompts   # 每类和 = N
            self.register_buffer('reliability_prior', scores)
            self.register_buffer('fixed_weights_ca', w_ca)
            n_eff = (1.0 / (torch.softmax(logits_ca, 0) ** 2).sum(0))
            print(f"[CA-Fixed] β={self.ca_beta} scale={scale:.3f} "
                  f"N_eff/类: min={n_eff.min():.1f} mean={n_eff.mean():.1f} max={n_eff.max():.1f}")
            self.domain_residual = None
        else:
            # "prior_residual" / "residual" / "pr_graph"
            scores = domain_weights.clone().float().to(self.device).detach()
            if weight_mode == "residual":
                scores = torch.zeros_like(scores)  # Residual Only: 先验置零
            # prior: 固定 buffer, detach, 不参与梯度 (作为归纳偏置)
            self.register_buffer('reliability_prior', scores)
            # ===== E3 方案 B (2026-11-21): 簇级分组加权 =====
            # 近重复描述符 (|cos|>θ 同簇) 共享话语权: 簇级先验=簇内 r 均值 (成员数无关,
            # 注入复制不改变簇相对话语权; 单元素簇复制下相对份额严格不变),
            # 残差升至簇级 (G 维); 全单元素簇时严格退化回 RW-DDO。
            self.grouped_weights = bool(grouped_weights) and descriptor_clusters is not None
            if self.grouped_weights:
                if use_redundancy:
                    raise ValueError("grouped_weights 与 C-1 redundancy 互斥 (机制上相互替代)")
                assignment, n_groups = descriptor_clusters
                # 注册为 buffer: 簇成员映射随 state_dict 保存/恢复 (外审 2026-11-22 建议)
                self.register_buffer('cluster_assignment',
                                     assignment.long().to(self.device).clone())
                self.num_groups = int(n_groups)
                counts = torch.bincount(self.cluster_assignment, minlength=self.num_groups)
                cluster_prior = torch.zeros(self.num_groups, device=self.device)
                cluster_prior.scatter_add_(0, self.cluster_assignment, scores)
                cluster_prior = cluster_prior / counts.to(self.device).clamp_min(1).float()
                self.register_buffer('cluster_prior', cluster_prior)
                self.cluster_residual = nn.Parameter(
                    torch.zeros(self.num_groups, device=self.device))
                n_multi = int((counts > 1).sum().item())
                print(f"[E3] grouped weights: {self.num_prompts} prompts -> {self.num_groups} clusters "
                      f"(multi-member: {n_multi})")
                self.domain_residual = None   # 残差升至簇级, 不再逐描述符
            else:
                # residual: 可学习 Parameter, 初始为 0 (训练初期 logits=prior)
                self.domain_residual = nn.Parameter(
                    torch.zeros_like(scores)
                )
            # --- Graph Laplacian (仅 pr_graph 模式构造): 描述符关系结构约束残差修正一致性 ---
            # A_ij = max(0, cos(d_i, d_j)), d_i 为类平均域方向 (与第四章共享同一 D)
            # F1-补丁 v2 (2026-11-20, 外审意见): 上一版把 assert 放在了 else 分支内,
            # 导致 prior_residual 正常路径也触发断言失败 —— 改为条件构造:
            # 仅 weight_mode == "pr_graph" 时才构造 L_G, 其他模式一律不构造。
            if weight_mode == "pr_graph":
                with torch.no_grad():
                    dirs = self.diffs.mean(dim=1).float()
                    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
                    A = (dirs @ dirs.T).clamp(min=0)
                    A.fill_diagonal_(0)
                    deg = A.sum(1)
                    L_G = torch.diag(deg) - A
                    self.register_buffer('desc_laplacian', L_G)

        # E1 干预 (2026-11-21): 冻结象限掩码 (LongTensor 索引); apply_delta_freeze() 在每个 step 后回零
        if freeze_delta_mask is not None:
            self.register_buffer(
                'delta_freeze_mask',
                freeze_delta_mask.to(device=self.device).long().clone())

        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)
        # 改动时间26.4.29-----------------------------------
        self.domain_concept_projection = self.diffs @ self.concept_embeddings.T

        # ===== DDO-Subspace (2026-11-25): 训练时约束的方向集可替换 =====
        # 原始完整方向集 (N 个描述符 x C 个文本类别) 始终保留在 domain_concept_projection_full,
        # 供统一评价指标 ddo_loss_equal_weight() 使用; ddo_dirs != 'full' 时只替换训练用的方向集。
        # 不消耗全局随机数 (随机标架用独立 generator) => 同 seed 下各臂的分类头初始化完全相同。
        self.domain_concept_projection_full = self.domain_concept_projection
        self.ddo_dirs_info = {'ddo_dirs': ddo_dirs, 'pairing': 'none'}
        if ddo_dirs != 'full':
            if weight_mode not in ('none',):
                raise ValueError("ddo_dirs 变体仅在等权 DDO (weight_mode='none') 下定义")
            self.domain_concept_projection = self._build_direction_set(
                ddo_dirs, int(ddo_k), ddo_pca_src, int(ddo_dirs_seed))
        # 实际构造的方向集的行数 / 形状 / 内容摘要 (写入结果 json): 不能只凭命令行参数反推模型约束的是哪组方向。
        # 完整方向集是三维的 (N 个描述符 x C 个类别 x 312), 所以行数按"方向向量个数"计, 不用 size(0)
        P = self.domain_concept_projection
        self.ddo_dirs_info.update({'rows': int(P.numel() // P.shape[-1]), 'shape': list(P.shape),
                                   'sha256': self._directions_sha256(P)})
        if ddo_dirs != 'full':
            print(f"[DDO-Subspace] 训练用方向集: {self.ddo_dirs_info}")

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len(concept_names)),
            nn.Linear(len(concept_names), len(class_names)))

        if init_with_class_embedding == True:
            with torch.no_grad():
                class_embeddings = self.clip_model.encode_text(clip.tokenize(class_names).to(device)).float()
                self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                self.classifier[2].weight = nn.Parameter(class_embeddings @ self.concept_embeddings.T)

        # DDO-L2 闭式 (外审第十五轮 P1) 与 Centered-L2 (外审第十六轮 P2): 只在显式开启时预计算。
        # 守卫 (外审第十六轮 P1): 闭式二阶矩公式只对等权、冻结 E、支持的方向集成立 —— 其余配置拒绝启动。
        _l2c_on = getattr(args, "ddo_norm", "l1") == "l2" and getattr(args, "ddo_l2_closed", False)
        _l2cen_on = getattr(args, "ddo_norm", "l1") == "l2c"
        if _l2c_on or _l2cen_on:
            if weight_mode != "none":
                raise ValueError("闭式路径只支持等权 (weight_mode='none'); 加权方向集的闭式未定义, 不静默忽略")
            if ddo_dirs not in ("full", "full_bf"):
                raise ValueError("闭式路径目前支持 --ddo_dirs full / full_bf")
            if float(getattr(args, "beta", 0.0) or 0.0) != 0.0 or int(getattr(args, "cal_rank", 0) or 0) != 0:
                raise ValueError("闭式路径要求 beta=0 (属性损失) 与 cal_rank=0 (冻结 E)")
            if getattr(args, "use_redundancy", False) or getattr(args, "grouped_weights", False):
                raise ValueError("闭式路径不支持 redundancy / grouped 加权")
            if getattr(args, "ld_mode", "none") != "none":
                raise ValueError("闭式路径与 ld_mode 互斥")
            self._build_l2_closed_buffers()
            self.l2_closed_skip_forward = True     # forward 不再逐方向计算显式响应 (§3av)
            print(f"[DDO-L2 {'closed' if _l2c_on else 'Centered'}] G {tuple(self.l2_G.shape)}, "
                  f"mu {tuple(self.l2_mu.shape)} (buffers, 不参与梯度); forward 跳过显式响应")

    def get_domain_weights(self):
        """
        计算最终权重, 返回 (weights, residual_softmax):
          - weights: 作用于 regularizer 的权重 (均值=1)
          - residual_softmax: softmax(residual), 供 KL 正则使用 (fixed/none 模式为 None, KL 恒 0)

        weight_mode 行为:
          "none"           -> (None, None)                    不加权
          "fixed"          -> (softmax(scores)*N, None)       固定软权重
          "residual"       -> (softmax(0+δ)*N, softmax(δ))    无先验
          "prior_residual" -> (softmax(r+δ)*N, softmax(δ))    主方法
        """
        if self.weight_mode == "none" or self.reliability_prior is None:
            return None, None
        if self.weight_mode == "fixed":
            return self.fixed_weights, None
        if self.weight_mode == "ca_fixed":
            return self.fixed_weights_ca, None    # (204, C), forward 按 dim 分支

        # E3 方案 B: 簇级分组加权
        # w_i = Q_g(i) * q_i|g, Q_g = softmax(mean_r_g + Δ_g) * G (簇总话语权与成员数无关),
        # q_i|g = softmax(r) 簇内份额; 最后校准 mean(w)=1。全单元素簇时严格退化回 RW-DDO。
        if getattr(self, 'grouped_weights', False):
            logits_g = self.cluster_prior + self.cluster_residual          # (G,)
            Q = torch.softmax(logits_g, dim=0) * self.num_groups
            r = self.reliability_prior
            # 簇内 softmax(r): 全局 shift (r 范围 ~[0.5,0.95], 无下溢风险), 兼容所有 torch 版本
            e = torch.exp(r - r.max())
            denom = torch.zeros(self.num_groups, device=r.device, dtype=r.dtype)
            denom.scatter_add_(0, self.cluster_assignment, e)
            q_in = e / denom[self.cluster_assignment]                      # Σ_{i∈g} q = 1
            w = Q[self.cluster_assignment] * q_in
            w = w * (self.num_prompts / w.sum())
            return w, torch.softmax(self.cluster_residual, dim=0)          # KL 约束簇级残差

        logits = self.reliability_prior + self.domain_residual
        weights = torch.softmax(logits, dim=0) * self.num_prompts  # 均值=1, 保量级
        # C-1 (方案 C): 去重复计权 —— logits 减 log(c_j), 冗余方向降话语权
        # 全正交时 c_j=1, log(c_j)=0, 严格退化回 RW-DDO; 开关由 args.redundancy 控制
        if getattr(self, 'use_redundancy', False) and getattr(self, 'descriptor_redundancy', None) is not None:
            logits = logits - torch.log(self.descriptor_redundancy.clamp_min(1e-8))
            weights = torch.softmax(logits, dim=0) * self.num_prompts
        # residual 的 softmax q, 用于 KL 正则: D_KL(q ‖ uniform) = log N - H(q)
        # 注意方向: q 在前 (正向 KL), 即 "q 到 uniform 的距离", 约束 residual 不偏离 uniform 太远
        residual_softmax = torch.softmax(self.domain_residual, dim=0)
        return weights, residual_softmax

    def get_prior_weights(self):
        """先验权重 (residual=0 时的权重), 供日志/Δw 分析。none 模式返回 None。"""
        if self.weight_mode == "none" or self.reliability_prior is None:
            return None
        return torch.softmax(self.reliability_prior, dim=0) * self.num_prompts

    def apply_delta_freeze(self):
        """E1 干预: 每个 optimizer.step 后将冻结象限的 δ 强制回零 (权重退化为先验)。无掩码时 no-op。"""
        mask = getattr(self, 'delta_freeze_mask', None)
        if mask is not None and self.domain_residual is not None:
            with torch.no_grad():
                self.domain_residual.data[mask] = 0.0

    def get_grouped_prior_weights(self):
        """E3 grouped: 簇级残差 Δ_g=0 时的完整初始权重
        w^(0)_i = N·softmax_g(R_g)·q_{i|g}·(N/Σw), 与 get_domain_weights 同构但残差置零。
        (2026-11-23 外审修正: 旧日志字段混入残差且未乘簇内份额, 名实不符;
         Δw_structure = w^(0) − RW先验权重, Δw_learn = w^(1) − w^(0) 由此可分离)
        仅 grouped_weights 模式有效, 其余返回 None。"""
        if not getattr(self, 'grouped_weights', False):
            return None
        with torch.no_grad():
            a = self.cluster_assignment
            Q0 = torch.softmax(self.cluster_prior, dim=0) * self.num_groups
            r = self.reliability_prior
            e = torch.exp(r - r.max())                       # 全局 shift, 与主分支一致
            den = torch.zeros(self.num_groups, device=r.device, dtype=r.dtype)
            den.scatter_add_(0, a, e)
            q_in = e / den[a]
            w = Q0[a] * q_in
            return w * (self.num_prompts / w.sum())

    def _build_l2_closed_buffers(self):
        """DDO-L2 的二阶矩闭式 (外审第十五轮 P1; §3ar): 预计算 ū 的 G 与 μ。
        full (LN-affine 路径): 响应 = A·LN0(p) + c → ū = LN0(P); full_bf (无偏置 ln-free 路径):
        P 已 LN0, 响应 = A·P → ū = P。G = ūᵀū/M (K,K), μ = ū.mean(0) (K,)。
        数值等价性 (损失与梯度) 见 stage1b_l2_closed.py 的真实数据核对。"""
        P = self.domain_concept_projection
        P2 = P.reshape(-1, P.size(-1)).double()
        if getattr(self, "_ddo_dirs_ln_free", False):
            ubar = P2                                   # full_bf: 已经 LN0
        else:
            ubar = (P2 - P2.mean(-1, keepdim=True)) / torch.sqrt(
                P2.var(-1, unbiased=False, keepdim=True) + 1e-5)
        self.register_buffer("l2_G", (ubar.T @ ubar / ubar.size(0)).float())
        self.register_buffer("l2_mu", ubar.mean(0).float())
        self._l2_closed_ready = True

    def ddo_l2_closed_loss(self):
        """(1/C_cls)·[tr(A G Aᵀ) + 2 cᵀAμ + ‖c‖²]（Bias-Free 时 c=0 → tr(AGAᵀ)/C_cls）。
        数值细节：先把 A 右乘 H_K（每个权重行减去其在特征维上的均值 —— LN0 的不可辨识方向，
        G·1=0 使该投影成为恒等变换），再在 float64 下计算；不投影会有 float32 灾难性消去
        (§3ar 实测 4e-3/1.2e-1)。**注意（外审第十六轮 P0）：这是特征维投影，不是类别中心化**；
        类别中心化是左乘 H_C（Ã_c = A_c − Ā），见 ddo_l2c_closed_loss。"""
        assert getattr(self, "_l2_closed_ready", False), "--ddo_l2_closed 需要先 _build_l2_closed_buffers"
        ln, lin = self.classifier[1], self.classifier[2]
        A = (lin.weight * ln.weight[None, :]).double()
        A = A - A.mean(-1, keepdim=True)                    # 右乘 H_K: LN0 不可辨识方向 (恒等投影, 去消去)
        G = self.l2_G.double()
        C_cls = A.size(0)
        if getattr(self, "_ddo_dirs_ln_free", False):
            # full_bf: 损失路径无偏置 (F.linear(x·γ, W, None)), 闭式 = tr(ÂGÂᵀ)/C —— 不能加 c 项!
            return (((A @ G) * A).sum() / C_cls).float()
        c = (lin.weight @ ln.bias + lin.bias).double()
        mu = self.l2_mu.double()
        out = ((A @ G) * A).sum() / C_cls + 2 * (c @ (A @ mu)) / C_cls + (c * c).sum() / C_cls
        return out.float()

    def ddo_l2c_closed_loss(self):
        """Centered-L2 (外审第十六轮 P2; §3aw): R_Cent = (1/C_cls)·tr(Ã G Ãᵀ)，
        Ã = H_C·A（类别中心化：Ã_c = A_c − Ā，左乘 H_C —— 与上面的特征维投影是两种不同投影）。
        性质：对公共 logit 平移（每类行加同一向量）不变；对特征维 1_K 偏移不变（G·1=0）；
        与 Pairwise-L2 关系 R_Pairwise = (2C/(C−1))·R_Cent；与 DDO-L2 分解 R_L2 = R_Cent + ĀᵀGĀ（E7）。"""
        assert getattr(self, "_l2_closed_ready", False), "Centered-L2 需要先 _build_l2_closed_buffers"
        ln, lin = self.classifier[1], self.classifier[2]
        A = (lin.weight * ln.weight[None, :]).double()
        A = A - A.mean(-1, keepdim=True)                    # 右乘 H_K: 数值稳定化 (恒等投影)
        At = A - A.mean(0, keepdim=True)                    # 左乘 H_C: 类别中心化 (定义)
        G = self.l2_G.double()
        return (((At @ G) * At).sum() / At.size(0)).float()

    def ddo_loss_equal_weight(self):
        """统一的等权 DDO 评价指标 (2026-11-25): 在原始完整方向集上不加权地计算 |h(P)| 的均值。
        与训练时的 weight_mode / 加权 / α 无关, 用于不同方法之间的横向比较。
        返回带梯度的张量; 只在评估时由调用方包在 no_grad 里。"""
        P = getattr(self, 'domain_concept_projection_full', self.domain_concept_projection)
        return self.classifier[1:](P).abs().mean()

    @staticmethod
    def _directions_sha256(p):
        """方向张量的内容摘要 (float32 连续内存的 sha256)"""
        import hashlib
        return hashlib.sha256(p.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    @staticmethod
    def _principal_rows(X, k):
        """X (n, d) 的前 k 个右奇异向量 (按行, 奇异值降序) 与全部奇异值的平方。
        用 float64 的 Gram 矩阵特征分解 —— 与准备分析脚本 ddo_subspace_spectra.svd_rows 是同一个算法。
        (此前用 float32 GPU 的 torch.linalg.svd, 向量的正交性只有约 1e-4, 2026-11-25 由断言发现;
        两者给出同一个子空间, 逐方向一致到 1e-3 以内。)"""
        X = X.double()
        if X.size(0) >= X.size(1):
            lam, vec = torch.linalg.eigh(X.T @ X)             # 升序
            s = lam.clamp_min(0).sqrt().flip(0)
            V = vec.flip(1)                                   # 列向量, 降序
        else:
            lam, U = torch.linalg.eigh(X @ X.T)
            s = lam.clamp_min(0).sqrt().flip(0)
            V = (X.T @ U.flip(1)) / s.clamp_min(1e-30)
        return V.T[:k].float(), (s ** 2)

    def _build_direction_set(self, ddo_dirs, k, pca_src, seed):
        """DDO-Subspace: 构造训练时约束的方向集 (M, 312), 每行是分类头 LayerNorm 之前的概念空间方向。

        ddo_dirs:
          'mean'      204 个描述符的类平均方向经概念投影          (M = 204, 不成对)
          'pca'       主方向, ± 成对                               (M = 2k)
                      pca_src='ln0'     : 对 LN0(D E^T) 做 SVD —— 分类头实际看到的约束输入 (无 affine)
                      pca_src='concept' : 对 D E^T 做 SVD —— 未经 LayerNorm 的概念投影
                      pca_src='mean'    : 对归一化类平均方向 (204x768) 做 SVD 得到 B (与第四章同一构造),
                                          再经 E 映射到概念空间 —— 两章机制关联实验用的"同一个 B"
          'random'    概念空间零均值超平面内的随机正交标架, ± 成对  (M = 2k)
          'full_sym'  原始完整方向集 + 其相反数                     (M = 2NC)
        LayerNorm 会逐行归一化, 所以行的尺度无关, 只有方向有意义。

        2026-11-25 外审后的两点修正:
          - 谱能量占比是奇异值平方的累计 (L2 口径)。DDO 实际优化的是 L1 响应, 二者没有等价关系,
            这里的占比只描述矩阵本身的谱, 不代表"保留了同样比例的约束信息"。
          - ± 成对不是简单地把方向复制一份: 分类头带偏置 b 时 (|a+b| + |-a+b|)/2 = max(|a|, |b|)。
            所以相对完整方向集, pca / random 臂同时改变了方向空间、方向分布和符号分布三个因素;
            'full_sym' 是只改符号分布的对照臂。
        """
        E = self.concept_embeddings
        with torch.no_grad():
            if ddo_dirs == 'mean':
                dirs = self.diffs.mean(dim=1)
                dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
                P = dirs @ E.T                                        # (204, 312)
                self.ddo_dirs_info.update({'pairing': 'none'})
            elif ddo_dirs == 'full_sym':
                P = torch.cat([self.domain_concept_projection_full, -self.domain_concept_projection_full], 0)
                self.ddo_dirs_info.update({'pairing': 'plus_minus'})
            elif ddo_dirs == 'pca':
                A = (self.diffs @ E.T).reshape(-1, E.size(0))       # (N*C, 312)
                if pca_src == 'ln0':
                    A = (A - A.mean(-1, keepdim=True)) / torch.sqrt(A.var(-1, unbiased=False, keepdim=True) + 1e-5)
                    X = A
                elif pca_src == 'concept':
                    X = A
                elif pca_src == 'mean':
                    dirs = self.diffs.mean(dim=1)
                    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
                    X = dirs
                else:
                    raise ValueError(f"未知 ddo_pca_src='{pca_src}' (ln0 / concept / mean)")
                V, ev = self._principal_rows(X, k)
                cum = torch.cumsum(ev, 0) / ev.sum()
                if pca_src == 'ln0':
                    # LN0 的行都在零均值超平面里, 其右奇异向量也应如此 (与 random 标架的构造空间一致)
                    assert float(V.mean(-1).abs().max()) < 1e-4, "ln0 主方向不在零均值超平面内"
                assert float((V @ V.T - torch.eye(k, device=V.device)).abs().max()) < 1e-4, "主方向不正交"
                P = (V @ E.T) if pca_src == 'mean' else V              # 'mean' 的主方向在 768 维, 经 E 映射
                P = torch.cat([P, -P], 0)                             # ± 成对
                self.ddo_dirs_info.update({'pairing': 'plus_minus',
                                           'pca_src': pca_src, 'k': k,
                                           # 奇异值平方的累计占比 (谱能量), 不是 L1 约束的覆盖率
                                           'cum_spectral_energy_at_k': round(float(cum[min(k, cum.numel()) - 1]), 4)})
            elif ddo_dirs in ('projected', 'residual', 'split'):
                # 外审第三轮 (2026-11-25): 分离子空间截断与方向分布重构。
                # u = LN0(A) 的行 (分解在 LN 后、affine 前的统一空间); P_k = V V^T;
                # projected 约束 P u, residual 约束 (I-P) u, split 对两个分量分别求损失后相加
                # (由 _ddo_split_scale=2 与外层 abs().mean() 实现)。注意 |A u| != |A P u| + |A (I-P) u|,
                # 两个分量可能相互抵消或互补 —— 这正是该对照要回答的问题。
                # 方向不重新逐行归一化 (外审要求保留投影幅值; 损失路径见 forward 的 _ddo_dirs_ln_free 分支)。
                if pca_src != 'ln0':
                    raise ValueError("projected / residual / split 仅支持 pca_src='ln0' (分解与约束都在 LN0 后的空间)")
                A = (self.diffs @ E.T).reshape(-1, E.size(0))       # (N*C, 312)
                U = (A - A.mean(-1, keepdim=True)) / torch.sqrt(A.var(-1, unbiased=False, keepdim=True) + 1e-5)
                V, ev = self._principal_rows(U, k)
                cum = torch.cumsum(ev, 0) / ev.sum()
                assert float(V.mean(-1).abs().max()) < 1e-4, "ln0 主方向不在零均值超平面内"
                assert float((V @ V.T - torch.eye(k, device=V.device)).abs().max()) < 1e-4, "主方向不正交"
                Proj = V.T @ V                                       # (312, 312) 正交投影
                assert float((Proj @ Proj - Proj).abs().max()) < 1e-4, "投影算子不幂等"
                Um, Ur = U @ Proj, U - U @ Proj
                if ddo_dirs == 'projected':
                    P = Um                                           # (N*C, 312), 保留投影幅值
                elif ddo_dirs == 'residual':
                    P = Ur
                else:
                    P = torch.cat([Um, Ur], 0)                       # (2*N*C, 312)
                    self._ddo_split_scale = 2.0                      # abs().mean() x 2 = 两个分量损失之和
                self._ddo_dirs_ln_free = True
                self.ddo_dirs_info.update({'pairing': 'none', 'k': k,
                                           'loss_path': 'affine_linear (LN0 已施加, 不再逐行归一化)',
                                           'bias': '保留 (含 W·beta + b, 与 full 的有效响应一致)',
                                           'cum_spectral_energy_at_k': round(float(cum[min(k, cum.numel()) - 1]), 4)})
            elif ddo_dirs in ('full_bf', 'split_bf'):
                # 外审第四轮 (2026-11-25): split 的 |A·u_p + c| + |A·u_r + c| 把有效偏置 c = W·beta + b 计了两次,
                # split 与 full 的差异里混入了"截距重复处理"这个因素。不用"偏置只计一次"的各种 split (如
                # |A·u_p + c| + |A·u_r|): 偏置分给哪个分量没有自然依据, 会引入不对称性 (外审第五轮更正了我
                # "它恒等于 full"的说法 —— 只有先把两分量相加再加偏置才恒等于 full)。
                # 正确的配对是 bias-free 的 split |A·u_p| + |A·u_r| 对 bias-free 的 full |A·u|:
                # full_bf  = 完整 LN0 方向 (不分解), 无偏置
                # split_bf = 两分量分别求损失后相加, 无偏置
                A = (self.diffs @ E.T).reshape(-1, E.size(0))       # (N*C, 312)
                U = (A - A.mean(-1, keepdim=True)) / torch.sqrt(A.var(-1, unbiased=False, keepdim=True) + 1e-5)
                if ddo_dirs == 'full_bf':
                    P = U
                    self.ddo_dirs_info.update({'pairing': 'none'})
                else:
                    V, ev = self._principal_rows(U, k)
                    Proj = V.T @ V
                    P = torch.cat([U @ Proj, U - U @ Proj], 0)
                    self._ddo_split_scale = 2.0
                    self.ddo_dirs_info.update({'pairing': 'none', 'k': k})
                self._ddo_dirs_ln_free = True
                self._ddo_bias_free = True
                self.ddo_dirs_info.update({'loss_path': 'affine_linear (LN0 已施加, 不再逐行归一化)',
                                           'bias': '无 (bias-free: 不含 W·beta + b)'})
            elif ddo_dirs == 'random':
                g = torch.Generator().manual_seed(seed)              # 独立 generator, 不动全局随机数
                R = torch.randn(E.size(0), k, generator=g)
                R = R - R.mean(dim=0, keepdim=True)                  # 先中心化: 落在零均值超平面 (外审)
                R = torch.linalg.qr(R)[0].T                          # (k, 312) 超平面内的随机正交标架
                assert float(R.mean(-1).abs().max()) < 1e-4, "随机标架不在零均值超平面内"
                assert float((R @ R.T - torch.eye(k)).abs().max()) < 1e-4, "随机标架不正交"
                P = torch.cat([R, -R], 0)
                self.ddo_dirs_info.update({'pairing': 'plus_minus', 'k': k, 'seed': seed,
                                           'frame': 'centered_orthonormal'})
            else:
                raise ValueError(f"未知 ddo_dirs='{ddo_dirs}' (full / mean / pca / random / full_sym)")
        return P.float().to(self.device)

    @staticmethod
    def _affine_linear(x, ln, lin, with_bias=True):
        """只做 LayerNorm 的 affine + Linear, 不做逐行归一化: W·(gamma⊙x + beta) + b。
        用于 projected / residual / split 臂 —— 方向已经过 LN0, 再过分类头的 LayerNorm 会把投影幅值
        重新归一化 (外审明确要求避免: 低能量残差会被人为放大)。与 full 的有效响应
        W·diag(gamma)·LN0(u) + W·beta + b 中 gamma/beta/b 的处理一致。
        with_bias=False 时只算方向响应 W·(gamma⊙x), 不含 W·beta + b —— 用于 bias-free 对照:
        split 的 |A·u_p + c| + |A·u_r + c| 把有效偏置 c 计了两次 (外审第四轮指出), 要去掉这个混杂,
        正确的配对是 bias-free 的 split |A·u_p| + |A·u_r| 对 bias-free 的 full |A·u|。
        注意 nn.Linear 自带 bias, 所以无偏置路径必须用 F.linear(x·gamma, W, bias=None),
        不能只去掉 beta (tests_projected_dirs.py 的 BF1/BF2/BF5 钉住这一点)。"""
        import torch.nn.functional as F
        if with_bias:
            return lin(x * ln.weight + ln.bias)
        return F.linear(x * ln.weight, lin.weight, None)

    # 改动时间26.4.29-----------------------------------
    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)

        concept_activations = visual_features @ self.concept_embeddings.T

        if getattr(self, "l2_closed_skip_forward", False):
            # 闭式路径 (外审第十六轮 P1; §3av): 不逐方向计算显式响应 —— 训练损失由 _ddo_norm_loss 的
            # 闭式分支给出 (reg_loss=None 不等于跳过损失); 统一 L1 评价 (ddo_full_eval / ddo_eq) 仍在
            # 诊断阶段显式计算。前向省掉 (40800×312)@(312×200) 的响应矩阵。
            self._reg_unweighted = None
            return concept_activations, self.classifier(concept_activations), None

        if getattr(self, '_ddo_dirs_ln_free', False):
            # projected / residual / split / bias-free full: 方向已 LN0, 绕过分类头的 LayerNorm (保留投影幅值)
            regularizer = self._affine_linear(self.domain_concept_projection,
                                              self.classifier[1], self.classifier[2],
                                              with_bias=not getattr(self, '_ddo_bias_free', False))
            if getattr(self, '_ddo_split_scale', 1.0) != 1.0:
                regularizer = regularizer * self._ddo_split_scale   # split: abs().mean() x 2 = 两个分量之和
        else:
            regularizer = self.classifier[1:](
                self.domain_concept_projection
            )
        # 这里改动的原因是避免每次forward都算遍，速度慢了25倍
        # regularizer = self.classifier[1:](
        #     self.diffs @ self.concept_embeddings.T
        # )

        weights, _ = self.get_domain_weights()
        # 2026-11-25: 记录加权之前的 DDO 响应, 供统一评价口径 (等权 DDO 损失) 使用
        self._reg_unweighted = regularizer.detach()
        if weights is not None:
            if weights.dim() == 2:        # CA-Fixed: (204, C) 逐类别加权
                regularizer = regularizer * weights.unsqueeze(-1)
            else:
                regularizer = regularizer * weights.view(-1, 1, 1)

        return concept_activations, self.classifier(concept_activations), regularizer
    # 改动时间26.4.29-----------------------------------
    # def forward(self, images):
    #     visual_features = self.clip_model.encode_image(images).float()
    #     visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
    #     concept_activations = visual_features @ self.concept_embeddings.T  # (bs,312)
    #     regularizer = self.classifier[1:](self.diffs @ self.concept_embeddings.T)
    #     return concept_activations, self.classifier(concept_activations), regularizer

    def extract_cls_concept(self):
        asso_mat_last = self.classifier[2].weight
        topk_res_last = {}
        topk_res = {}
        # import pdb; pdb.set_trace()
        concept_names = np.array(self.concept_names)
        for i, cls_name in enumerate(self.class_names):
            topk_res_last[cls_name] = np.unique(concept_names[asso_mat_last[i].topk(10)[1].cpu().detach().numpy()]).tolist()
        return {},topk_res


class clip_cbm_subspace(nn.Module):
    """
    第四章: Domain Subspace Guided Invariant Classification (专家定稿版, 2026-08 第四轮咨询)

    主线: Domain Prompt Structure → Adaptive Domain Subspace → Soft Suppression
          → Invariant Classification Head

    关键设计:
      1. B 从 domain prompt 类平均方向的 SVD 得到, m = m(τ) energy-threshold 自适应:
             m(τ) = min{ m : Σ_{i≤m} σ_i² / Σ σ_i² ≥ τ }     (默认 τ=0.8)
         完全由文本得到, 不使用 target image (DG 严格设定)
      2. 软抑制 (不用硬删除, 因 principal directions 可能携带 class 信息):
             z_inv = z − γ·BBᵀz,  γ = σ(a) 为 learnable scalar
      3. 投影放在 CLIP image embedding 层 (专家方案A), 之后进 concept bottleneck:
             z → z_inv → concept → invariant head
      4. 双头: 原始头 h_o(z) 仅作辅助监督; 测试只用 h_inv(z_inv)
         (避免 "提升来自 ensemble" 的质疑)
      5. DDO 等权正交正则保留 (第三章 weight_mode="none" 行为), 第四章模块叠加其上

    ch4_mode 五组消融 (专家实验矩阵):
      "ddo"      : 无 subspace 无新头 (基线)
      "hardproj" : B + γ≡1 固定, 单头在 z_inv 上
      "softproj" : B + γ=σ(a) 可学习, 单头在 z_inv 上
      "headonly" : 无投影, 双头都在原始 z 上 (证明"仅加头没用")
      "full"     : B + γ 可学习 + 双头 (主方法)
    """

    def __init__(
            self,
            args,
            class_names,
            concept_names,
            domain_diffs,
            tau=0.8,
            ch4_mode="full",
            eta_ddo_inv=1.0,
            domain_weights=None,
            weight_mode="none",
            init_with_class_embedding=False,
            cal_rank=0,
            descriptor_redundancy=None,
            use_redundancy=False
    ):
        super(clip_cbm_subspace, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.num_prompts = domain_diffs.size(0)
        self.ch4_mode = ch4_mode
        self.tau = tau
        # P4: 低秩概念校准 E' = E + U Vᵀ (V 零初始化 ⇒ 初始行为与冻结 E 逐位一致)
        self.cal_rank = int(cal_rank)
        # C-1: 去冗余开关与冗余度向量 (buffer, 不参与梯度)
        self.use_redundancy = bool(use_redundancy)
        if descriptor_redundancy is not None:
            self.register_buffer('descriptor_redundancy',
                                 descriptor_redundancy.clone().float().detach())
        # DDO 正则对 invariant head 的约束权重 (专家第五轮: 测试头必须被 DDO 约束)
        # L_DDO = ‖h_o[1:](proj)‖ + eta_ddo_inv · ‖h_inv[1:](proj)‖, 默认 1.0 对称
        self.eta_ddo_inv = eta_ddo_inv

        # ===== 2×2 联合实验接口: 第三章 importance 加权 (复用 clip_cbm_orth 机制) =====
        # weight_mode="none" → 纯第四章 (DDO 等权); "prior_residual" → C3+C4 组合
        # 常规第四章实验不传 domain_weights 即可保持等权
        self.c3_weight_mode = weight_mode
        if domain_weights is None or weight_mode == "none":
            self.reliability_prior = None
            self.domain_residual = None
            self.fixed_weights = None
        else:
            scores = domain_weights.clone().float().to(self.device).detach()
            if weight_mode == "fixed":
                self.fixed_weights = torch.softmax(scores, dim=0) * self.num_prompts
                self.register_buffer('reliability_prior', scores)
                self.domain_residual = None
            elif weight_mode not in ("residual", "prior_residual", "pr_graph"):
                # 2026-11-25 守卫: 未实现的 weight_mode 不能静默落入 prior_residual 分支
                # (ca_fixed / grouped 仅在 clip_cbm_orth 下实现)
                raise ValueError(
                    f"clip_cbm_subspace 不支持 weight_mode='{weight_mode}' "
                    f"(仅 none/fixed/residual/prior_residual/pr_graph)")
            else:
                if weight_mode == "residual":
                    scores = torch.zeros_like(scores)
                self.register_buffer('reliability_prior', scores)
                self.domain_residual = nn.Parameter(torch.zeros_like(scores))

        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings (与 clip_cbm_orth 相同)
        attr_texts = clip.tokenize(concept_names).to(self.device)
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)

        # ===== P4: 低秩概念校准参数 (E' = E + U Vᵀ, cal_rank>0 时启用) =====
        # V 零初始化 ⇒ ΔE=0 ⇒ 初始行为与冻结 E 逐位一致 (不伤基线起步)
        if self.cal_rank > 0:
            K = self.concept_embeddings.size(0)
            self.cal_U = nn.Parameter(0.02 * torch.randn(K, self.cal_rank, device=self.device))
            self.cal_V = nn.Parameter(torch.zeros(self.cal_rank, self.concept_embeddings.size(1),
                                                  device=self.device))
        # 类平均 prompt 方向 → SVD → m = m(τ) → B = V[:, :m]
        if ch4_mode in ("hardproj", "softproj", "full", "full_e", "full_m", "full_m7", "full_mc", "full_align"):
            with torch.no_grad():
                dirs = self.diffs.mean(dim=1).float()               # (N, D) 类平均
                dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
                U, S, Vh = torch.linalg.svd(dirs, full_matrices=False)
                ev = S ** 2
                cum_ev = torch.cumsum(ev, dim=0) / ev.sum()
                tau_t = torch.tensor(float(tau), device=cum_ev.device)
                self.subspace_dim = int(torch.searchsorted(cum_ev, tau_t).item()) + 1
                B = Vh[:self.subspace_dim].T.contiguous()           # (D, m), 列正交
                self.register_buffer('domain_basis', B)
                print(f"[Ch4] domain subspace: tau={tau} -> m={self.subspace_dim}, "
                      f"B shape={tuple(B.shape)}")
        else:
            self.domain_basis = None
            self.subspace_dim = 0

        # ===== ③ Soft suppression: γ =====
        # hardproj: γ≡1; softproj/full: γ=σ(a) learnable (初始 σ(0)=0.5); 其余 γ≡0
        self.gate_type = {"full_e": "energy", "full_m": "mlp", "full_m7": "mlp_zd", "full_mc": "mlp"}.get(ch4_mode, "scalar")
        if ch4_mode == "hardproj":
            self.gamma_logit = None
            self.fixed_gamma = 1.0
        elif ch4_mode in ("softproj", "full", "full_align"):
            self.gamma_logit = nn.Parameter(torch.zeros(1))
            self.fixed_gamma = None
        elif ch4_mode in ("full_e", "full_m", "full_m7", "full_mc", "full_align"):
            # 样本自适应域抑制门控 (SADS): gamma(x) = sigma(gate([B^T z; e]))
            self.gamma_logit = None
            self.fixed_gamma = None
            self.last_gamma_mean = None
            if self.gate_type == "energy":
                # C4-B: 仅由域能量比 e 决定 gamma(x), 初始 w=0,b=0 => gamma=0.5
                self.gate_w = nn.Parameter(torch.zeros(1))
                self.gate_b = nn.Parameter(torch.zeros(1))
            elif self.gate_type == "mlp_zd":
                # 探索变体 full_m7: 门控输入为完整域分量 z_d (768 维)
                self.gate_mlp = nn.Sequential(
                    nn.Linear(self.clip_model.visual.output_dim, 128), nn.GELU(),
                    nn.Linear(128, 1))
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.zeros_(self.gate_mlp[-1].bias)
            else:
                # C4-C: 轻量 MLP 门控 [B^T z; e] -> gamma(x), 末层零初始化 => 初始 0.5
                self.gate_mlp = nn.Sequential(
                    nn.Linear(self.subspace_dim + 1, 128), nn.GELU(),
                    nn.Linear(128, 1))
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.zeros_(self.gate_mlp[-1].bias)
        else:
            self.gamma_logit = None
            self.fixed_gamma = 0.0

        # ===== ④ heads =====
        def make_classifier():
            return nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(len(concept_names)),
                nn.Linear(len(concept_names), len(class_names)))

        self.classifier = make_classifier()          # 原始头 (辅助)
        if ch4_mode in ("headonly", "full", "full_e", "full_m", "full_m7", "full_mc", "full_align"):
            self.inv_classifier = make_classifier()  # invariant 头 (测试主输出)
        else:
            self.inv_classifier = None

        # ===== 算法 C (full_align): 语言锚定 + 视觉校准的残差子空间 =====
        # B_adapt = orth(B_text + ΔB);  ΔB 零初始化 ⇒ 初始 B_adapt = B_text
        if ch4_mode == "full_align":
            import os
            aug_path = f"cache/aug_subspace_{args.dataset}.pt"
            assert os.path.exists(aug_path), (
                f"full_align 需要 {aug_path}, 请先运行 precompute_aug_subspace.py {args.dataset}")
            aug = torch.load(aug_path, map_location="cpu")
            B_aug = aug["B_aug"].float().to(self.device)          # (768, m_aug) 列正交
            self.register_buffer("aug_basis", B_aug)
            self.delta_B = nn.Parameter(torch.zeros_like(self.domain_basis))  # (768, m_text)

        # DDO 等权正交正则的投影 (与 clip_cbm_orth weight_mode="none" 一致)
        self.domain_concept_projection = self.diffs @ self.concept_embeddings.T

        if init_with_class_embedding == True:
            with torch.no_grad():
                class_embeddings = self.clip_model.encode_text(
                    clip.tokenize(class_names).to(self.device)).float()
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                self.classifier[2].weight = nn.Parameter(
                    class_embeddings @ self.concept_embeddings.T)

    def get_gamma(self):
        """当前抑制强度 γ (自适应门控返回最近一个 batch 的样本均值)"""
        if self.gamma_logit is not None:
            return torch.sigmoid(self.gamma_logit)
        if self.gate_type != "scalar" and getattr(self, "last_gamma_mean", None) is not None:
            return self.last_gamma_mean
        return self.fixed_gamma

    def _effective_E(self):
        """P4 低秩概念校准: 返回 E' = E + U Vᵀ 并缓存锚定范数; cal_rank=0 时原样返回冻结 E"""
        if self.cal_rank > 0:
            dE = self.cal_U @ self.cal_V
            self._cal_norm = torch.norm(dE, p="fro")
            return self.concept_embeddings + dE
        return self.concept_embeddings

    def _domain_projection(self, E):
        """DDO 正则的域-概念投影: 校准开启时随 E' 动态计算, 否则用 init 预计算的冻结投影"""
        if self.cal_rank > 0:
            return self.diffs @ E.T
        return self.domain_concept_projection

    def suppress_adaptive(self, z):
        """样本自适应域抑制 (SADS): q=B^T z, e=||q||/||z||, gamma(x)=sigma(gate([q;e]))"""
        q = z @ self.domain_basis                               # (bs, m)
        e = q.norm(dim=-1) / (z.norm(dim=-1) + 1e-8)            # 域能量比 (bs,)
        if self.gate_type == "energy":
            gamma = torch.sigmoid(self.gate_w * e.unsqueeze(-1) + self.gate_b)   # (bs,1)
        elif self.gate_type == "mlp_zd":
            gamma = torch.sigmoid(self.gate_mlp(q @ self.domain_basis.T))        # (bs,1)
        else:
            gamma = torch.sigmoid(self.gate_mlp(torch.cat([q, e.unsqueeze(-1)], dim=-1)))
        self.last_gamma_mean = gamma.detach().mean()
        return z - gamma * q @ self.domain_basis.T

    def get_domain_weights(self):
        """第三章 importance 加权 (2×2 联合实验用, 机制同 clip_cbm_orth):
        weight_mode="none" → (None, None) 等权; 其余 → (weights, softmax(residual))"""
        if self.c3_weight_mode == "none" or self.reliability_prior is None:
            return None, None
        if self.c3_weight_mode == "fixed":
            return self.fixed_weights, None
        logits = self.reliability_prior + self.domain_residual
        # C-1 (方案 C): 去重复计权 —— 与 clip_cbm_orth 同一机制
        if getattr(self, 'use_redundancy', False) and getattr(self, 'descriptor_redundancy', None) is not None:
            logits = logits - torch.log(self.descriptor_redundancy.clamp_min(1e-8))
        weights = torch.softmax(logits, dim=0) * self.num_prompts
        return weights, torch.softmax(self.domain_residual, dim=0)

    def get_prior_weights(self):
        """先验权重 (residual=0 时), 供日志/Δw 分析"""
        if self.c3_weight_mode == "none" or self.reliability_prior is None:
            return None
        return torch.softmax(self.reliability_prior, dim=0) * self.num_prompts

    def get_alignment_losses(self):
        """full_align 的两个投影差正则 (anchor / visual); 其他模式返回 None"""
        if self.ch4_mode != "full_align":
            return None
        return self._anchor_loss, self._visual_loss

    def suppress(self, z):
        """z_inv = z − γ·BBᵀz (无 subspace 或 γ=0 时原样返回)"""
        if self.domain_basis is None:
            return z
        gamma = self.get_gamma()
        if isinstance(gamma, float) and gamma == 0.0:
            return z
        return z - gamma * (z @ self.domain_basis) @ self.domain_basis.T

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()
        z = visual_features / visual_features.norm(dim=-1, keepdim=True)

        if self.ch4_mode == "full_align":
            B_adapt = torch.linalg.qr(self.domain_basis + self.delta_B)[0]   # (768, m) 列正交
            gamma = self.get_gamma()
            z_inv = z - gamma * (z @ B_adapt) @ B_adapt.T
            # 两个投影差正则 (带梯度, 供 main.py 取用)
            P_a = B_adapt @ B_adapt.T
            with torch.no_grad():
                P_t = self.domain_basis @ self.domain_basis.T
                P_v = self.aug_basis @ self.aug_basis.T
                self._align_anchor_ref = P_t
                self._align_visual_ref = P_v
            self._anchor_loss = torch.norm(P_a - self._align_anchor_ref, p="fro") ** 2
            self._visual_loss = torch.norm(P_a - self._align_visual_ref, p="fro") ** 2
        elif self.gate_type != "scalar":
            z_inv = self.suppress_adaptive(z)
        else:
            z_inv = self.suppress(z)

        # concept bottleneck (P4: 校准开启时投影走 E')
        E_eff = self._effective_E()
        concept_activations = z @ E_eff.T
        concept_activations_inv = z_inv @ E_eff.T

        logits_ori = self.classifier(concept_activations)

        if self.inv_classifier is not None:
            logits_inv = self.inv_classifier(concept_activations_inv)
            cls_preds = (logits_ori, logits_inv)      # 双头: 训练算双损失, 评估取 inv
            main_concept = concept_activations_inv
        else:
            cls_preds = self.classifier(concept_activations_inv)  # 单头在 z_inv 上
            main_concept = concept_activations_inv

        # DDO 等权正交正则: 两头分别约束 (按设计公式 ‖h_o[1:](proj)‖ + eta·‖h_inv[1:](proj)‖)
        # 注意 abs 在求和之前逐头施加 —— 若先求和再取 abs (|A+B|), 两头域响应可互相抵消
        # (h_o=-h_inv 时损失为 0 但两头都未被正交约束), 这是 2026-09-11 排查发现的实现偏差,
        # 修复后 headonly/full 重新实验 (修复前结果存于 logs/results_preddofix/)。
        ddo_proj = self._domain_projection(E_eff)
        regularizer = torch.abs(self.classifier[1:](ddo_proj))
        if self.inv_classifier is not None:
            reg_inv = torch.abs(self.inv_classifier[1:](ddo_proj))
            regularizer = regularizer + self.eta_ddo_inv * reg_inv
        # 2026-11-25: 记录加权之前的 DDO 响应, 供统一评价口径 (等权 DDO 损失) 使用
        self._reg_unweighted = regularizer.detach()

        # 2×2 联合实验: 第三章 importance 加权 (C3+C4 组合时启用)
        weights, _ = self.get_domain_weights()
        if weights is not None:
            if weights.dim() == 2:        # CA-Fixed: (204, C) 逐类别加权
                regularizer = regularizer * weights.unsqueeze(-1)
            else:
                regularizer = regularizer * weights.view(-1, 1, 1)

        return main_concept, cls_preds, regularizer

    def ddo_loss_equal_weight(self):
        """统一的等权 DDO 评价指标 (2026-11-25): 原始完整方向集、等权、仅主分类头。
        与训练时的 weight_mode / 双头系数 / P4 校准状态无关 (始终用冻结 E 的原始投影),
        用于不同方法之间的横向比较。"""
        return self.classifier[1:](self._domain_projection(self.concept_embeddings)).abs().mean()

    def extract_cls_concept(self):
        asso_mat = (self.inv_classifier or self.classifier)[2].weight
        concept_names = np.array(self.concept_names)
        topk_res = {}
        for i, cls_name in enumerate(self.class_names):
            topk_res[cls_name] = np.unique(
                concept_names[asso_mat[i].topk(10)[1].cpu().detach().numpy()]).tolist()
        return {}, topk_res


