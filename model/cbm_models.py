
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
            init_with_class_embedding=False
    ):
        # def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_cbm_orth, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
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
        else:
            # "prior_residual" 或 "residual"
            scores = domain_weights.clone().float().to(self.device).detach()
            if weight_mode == "residual":
                scores = torch.zeros_like(scores)  # Residual Only: 先验置零
            # prior: 固定 buffer, detach, 不参与梯度 (作为归纳偏置)
            self.register_buffer('reliability_prior', scores)
            # residual: 可学习 Parameter, 初始为 0 (训练初期 logits=prior)
            self.domain_residual = nn.Parameter(
                torch.zeros_like(scores)
            )

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

        logits = self.reliability_prior + self.domain_residual
        weights = torch.softmax(logits, dim=0) * self.num_prompts  # 均值=1, 保量级
        # residual 的 softmax q, 用于 KL 正则: D_KL(q ‖ uniform) = log N - H(q)
        # 注意方向: q 在前 (正向 KL), 即 "q 到 uniform 的距离", 约束 residual 不偏离 uniform 太远
        residual_softmax = torch.softmax(self.domain_residual, dim=0)
        return weights, residual_softmax

    def get_prior_weights(self):
        """先验权重 (residual=0 时的权重), 供日志/Δw 分析。none 模式返回 None。"""
        if self.weight_mode == "none" or self.reliability_prior is None:
            return None
        return torch.softmax(self.reliability_prior, dim=0) * self.num_prompts

    # 改动时间26.4.29-----------------------------------
    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)

        concept_activations = visual_features @ self.concept_embeddings.T

        regularizer = self.classifier[1:](
            self.domain_concept_projection
        )
        # 这里改动的原因是避免每次forward都算遍，速度慢了25倍
        # regularizer = self.classifier[1:](
        #     self.diffs @ self.concept_embeddings.T
        # )

        weights, _ = self.get_domain_weights()
        if weights is not None:
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
            init_with_class_embedding=False
    ):
        super(clip_cbm_subspace, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.num_prompts = domain_diffs.size(0)
        self.ch4_mode = ch4_mode
        self.tau = tau
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

        # ===== ① Domain Structure Discovery + ② Adaptive Subspace =====
        # 类平均 prompt 方向 → SVD → m = m(τ) → B = V[:, :m]
        if ch4_mode in ("hardproj", "softproj", "full"):
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
        if ch4_mode == "hardproj":
            self.gamma_logit = None
            self.fixed_gamma = 1.0
        elif ch4_mode == "softproj" or ch4_mode == "full":
            self.gamma_logit = nn.Parameter(torch.zeros(1))
            self.fixed_gamma = None
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
        if ch4_mode in ("headonly", "full"):
            self.inv_classifier = make_classifier()  # invariant 头 (测试主输出)
        else:
            self.inv_classifier = None

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
        """当前抑制强度 γ"""
        if self.gamma_logit is not None:
            return torch.sigmoid(self.gamma_logit)
        return self.fixed_gamma

    def get_domain_weights(self):
        """第三章 importance 加权 (2×2 联合实验用, 机制同 clip_cbm_orth):
        weight_mode="none" → (None, None) 等权; 其余 → (weights, softmax(residual))"""
        if self.c3_weight_mode == "none" or self.reliability_prior is None:
            return None, None
        if self.c3_weight_mode == "fixed":
            return self.fixed_weights, None
        logits = self.reliability_prior + self.domain_residual
        weights = torch.softmax(logits, dim=0) * self.num_prompts
        return weights, torch.softmax(self.domain_residual, dim=0)

    def get_prior_weights(self):
        """先验权重 (residual=0 时), 供日志/Δw 分析"""
        if self.c3_weight_mode == "none" or self.reliability_prior is None:
            return None
        return torch.softmax(self.reliability_prior, dim=0) * self.num_prompts

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

        z_inv = self.suppress(z)

        # concept bottleneck
        concept_activations = z @ self.concept_embeddings.T
        concept_activations_inv = z_inv @ self.concept_embeddings.T

        logits_ori = self.classifier(concept_activations)

        if self.inv_classifier is not None:
            logits_inv = self.inv_classifier(concept_activations_inv)
            cls_preds = (logits_ori, logits_inv)      # 双头: 训练算双损失, 评估取 inv
            main_concept = concept_activations_inv
        else:
            cls_preds = self.classifier(concept_activations_inv)  # 单头在 z_inv 上
            main_concept = concept_activations_inv

        # DDO 等权正交正则: 两头都约束 (专家第五轮检查点)
        # 测试头是 inv_classifier, DDO 必须至少直接约束它, 否则训练目标与测试头逻辑断裂。
        # 双头模式: L_DDO = ‖h_o[1:](proj)‖ + eta_ddo_inv · ‖h_inv[1:](proj)‖
        # 单头模式 (ddo/hardproj/softproj): 唯一头即测试头, 直接过它 (原实现已一致)
        # 注: CLIP frozen 的 CBM 中两头无共享可学习参数, original head 的梯度不流向
        #     测试路径; 其作用是 (a) headonly 消融的结构基础 (b) 无投影参照分类器
        #     (source acc 对比可作为 "投影代价" 的分析素材)。
        regularizer = self.classifier[1:](self.domain_concept_projection)
        if self.inv_classifier is not None:
            reg_inv = self.inv_classifier[1:](self.domain_concept_projection)
            regularizer = regularizer + self.eta_ddo_inv * reg_inv

        # 2×2 联合实验: 第三章 importance 加权 (C3+C4 组合时启用)
        weights, _ = self.get_domain_weights()
        if weights is not None:
            regularizer = regularizer * weights.view(-1, 1, 1)

        return main_concept, cls_preds, regularizer

    def extract_cls_concept(self):
        asso_mat = (self.inv_classifier or self.classifier)[2].weight
        concept_names = np.array(self.concept_names)
        topk_res = {}
        for i, cls_name in enumerate(self.class_names):
            topk_res[cls_name] = np.unique(
                concept_names[asso_mat[i].topk(10)[1].cpu().detach().numpy()]).tolist()
        return {}, topk_res


