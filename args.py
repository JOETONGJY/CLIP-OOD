import argparse


def get_args():
    parser = argparse.ArgumentParser(description="domain-invariant CBM.")

    # HYPERPARAMETERS ##
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', action="store_true", default=False)
    # parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1, help="orthogonality loss weight")
    parser.add_argument("--beta", type=float, default=0, help="concept supervision weight")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="KL reg weight for domain residual (prior_residual strategy)")
    parser.add_argument('--lambda_anchor', type=float, default=0.1,
                        help='projector-diff anchor weight: ||P_adapt-P_text||_F^2 (full_align)')
    parser.add_argument('--lambda_visual', type=float, default=0.1,
                        help='visual calibration weight: ||P_adapt-P_aug||_F^2 (full_align)')
    parser.add_argument('--protect_k', type=int, default=0,
                        help='SPDA: 类别判别子空间维数; 0=按 tau_c 能量阈值自适应')
    parser.add_argument('--cal_rank', type=int, default=0,
                        help='P4 低秩概念校准: ΔE=U Vᵀ 的秩 r; 0=关闭 (冻结 E)')
    parser.add_argument('--lambda_cal', type=float, default=0.01,
                        help='P4 概念校准锚定权重: lambda_cal * ||ΔE||_F^2')
    parser.add_argument('--lambda_rel', type=float, default=0.0,
                        help='A1 语义关系约束权重: top-k 邻域 Gram 结构保持; 0=关闭')
    parser.add_argument('--anchor_topk', type=int, default=5,
                        help='A1: 语义邻域 top-k')
    parser.add_argument('--redundancy', action="store_true", default=False,
                        help='C-1 去冗余 RW-DDO: logits 减 log(c_j), 冗余描述符降话语权')
    parser.add_argument('--redundancy_topk', type=int, default=0,
                        help='C-1: 冗余统计稀疏化 top-k; 0=全矩阵')
    parser.add_argument('--freeze_delta_file', type=str, default='',
                        help='E1 干预: 冻结象限掩码文件 (含 "indices" LongTensor); 空=关闭。'
                             '掩码内描述符的 residual 每个 step 后强制回零, 权重退化为先验')
    parser.add_argument('--grouped_weights', action="store_true", default=False,
                        help='E3 方案 B: 近重复簇级分组加权 (簇级先验+残差, 抑制重复计权:'
                             '单元素簇复制下相对份额严格不变, 全单元素簇时严格退化回 RW-DDO)')
    parser.add_argument('--cluster_threshold', type=float, default=0.9,
                        help='E3: 方向 |cos| 聚簇阈值 (仅依据离线相似度分布选择, 不按目标精度调)')
    parser.add_argument('--inject_file', type=str, default='',
                        help='E3 扰动实验: 注入描述符列表 json (含 "prompts"); 空=关闭。'
                             '仅实验驱动层使用, 在数据集构建前 patch target_text_prompts')
    parser.add_argument('--cls_rank', type=int, default=16,
                        help='M2 class-conditional selection: low-rank correction rank r')
    parser.add_argument('--lambda_cls_sparse', type=float, default=0.01,
                        help='M2 concept selection sparsity reg weight')
    parser.add_argument('--lambda_graph', type=float, default=0.1,
                        help='graph Laplacian reg weight on residual (pr_graph mode)')
    parser.add_argument('--weight_mode', type=str, default='prior_residual',
                        choices=['none', 'fixed', 'residual', 'prior_residual', 'pr_graph', 'ca_fixed'],
                        help='none=Original DDO; fixed=Soft Reliability; residual=Residual Only; prior_residual=full method; pr_graph=+graph Laplacian; ca_fixed=类别感知固定权重(CA-Fixed)')
    parser.add_argument('--ca_prior_file', type=str, default='',
                        help='CA-Fixed: 类别交互矩阵文件 (含 "interaction" (204,C) 张量); 空=关闭')
    parser.add_argument('--ca_beta', type=float, default=1.0,
                        help='CA-Fixed: 交互项强度 β (尺度自动匹配到 r 的 std; β=0 退化回 fixed)')
    ## Chapter 4: Domain Subspace Guided Invariant Classification ##
    parser.add_argument('--ch4_mode', type=str, default='full',
                        choices=['ddo', 'hardproj', 'softproj', 'headonly', 'full', 'full_e', 'full_m', 'full_m7', 'full_mc', 'full_align'],
                        help='Ch4 ablation modes (only used with --CBM_type clip_cbm_subspace)')
    parser.add_argument('--cons_weight', type=float, default=0.0,
                        help='dual-head consistency loss weight: L += w * KL(p_inv || p_orig); full_mc 默认 0.1')
    parser.add_argument('--gate_type', type=str, default='scalar',
                        choices=['scalar', 'energy', 'mlp'],
                        help='C4 suppression gate: scalar=global learnable gamma; energy=energy-conditioned gamma(x); mlp=MLP-gated gamma(x)')
    parser.add_argument('--tau', type=float, default=0.8,
                        help='energy threshold for adaptive subspace dim m(tau)')
    parser.add_argument('--run_tag', type=str, default='',
                        help='覆盖结果文件名中的方法 tag (如 tau 扫描: full_tau06); 留空则按模式自动生成')
    parser.add_argument('--lambda_inv', type=float, default=1.0,
                        help='loss weight of invariant head (aux ori head has weight 1)')
    parser.add_argument('--eta_ddo_inv', type=float, default=1.0,
                        help='DDO regularizer weight on invariant head (both heads constrained when >0)')
    parser.add_argument('--use_c3_weights', action="store_true", default=False,
                        help='combine Ch3 importance weighting into subspace model (2x2 joint experiment: C3+C4)')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")

    ## Setting ##
    parser.add_argument('--class_avg_concept', action="store_true", default=False)  # CUB, RIVAL10

    ## DATA/MODEL PATH ##
    parser.add_argument('--dataset', type=str, default='CUB')  # CUB, AWA2, LADA, LADV
    parser.add_argument('--data_dir', type=str, default='./data')  # CUB, AWA2, LADA, LADV
    parser.add_argument('--CLIP_type', type=str, default='ViT-L/14') # ViT-B/32, ViT-L/14
    parser.add_argument('--CBM_type', type=str, default='clip_cbm') # clip_cbm, cliplp
    parser.add_argument('--model_save_path', type=str, default=r'logs')
    parser.add_argument('--prompt_type', type=str, default=r'origin') # origin, others
    parser.add_argument("--save_model", action="store_true", default=True)
    # 2026-11-25: store_true + default=True 意味着命令行无法关闭保存 —— 11-20 的
    # 1-epoch smoke 正是这样覆盖了 best_CUB.pth。补一个显式关闭开关 (默认行为不变)。
    parser.add_argument("--no_save_model", dest="save_model", action="store_false",
                        help="不写 best_<dataset>*.pth (smoke/诊断 run 应始终使用)")
    # ---- 2026-11-25 实验可靠性修复 (外审建议) ----
    parser.add_argument("--overwrite_run", action="store_true", default=False,
                        help="覆盖保护: 同名结果 json / 权重日志 / 检查点已存在时默认拒绝启动; "
                             "加上本开关才覆盖 (会先删除该 run 的旧结果与权重日志)")
    parser.add_argument('--target_eval', type=str, default='per_epoch',
                        choices=['per_epoch', 'final_only'],
                        help='目标域评估时机: per_epoch=每个 epoch 都评估 (历史默认); '
                             'final_only=目标域不参与训练与选模, 仅在最后一个 epoch 评估一次 (严格 final 协议; '
                             'main_cached 入口在此模式下把目标域特征缓存延迟到最终评估时才读入)')
    parser.add_argument('--grad_diag', action="store_true", default=False,
                        help='在固定 epoch 与固定批次上测量 CE 与本模型正则项、等权 DDO 损失对分类头的梯度范数比'
                             ' (含乘以 alpha 之后的实际范数比) 与夹角 (只读梯度, 不影响训练)')
    # ---- DDO-Subspace (2026-11-25): 训练时约束的方向集; 统一评价指标始终用原始完整方向集 ----
    parser.add_argument('--ddo_dirs', type=str, default='full',
                        choices=['full', 'mean', 'pca', 'random', 'full_sym'],
                        help='full=原始全部类别条件方向(默认); mean=204 个描述符类平均方向; '
                             'pca=主方向(±成对); random=零均值超平面内的同维随机正交标架(±成对); '
                             'full_sym=完整方向集+其相反数 (只改符号分布的对照臂)。仅 weight_mode=none 下可用')
    parser.add_argument('--ddo_pca_src', type=str, default='ln0',
                        choices=['ln0', 'concept', 'mean'],
                        help='pca 的分解对象: ln0=LN0(D E^T) (分类头实际约束输入); concept=D E^T; '
                             'mean=归一化类平均方向 (与第四章同一子空间 B, 经 E 映射)')
    parser.add_argument('--ddo_k', type=int, default=32, help='DDO-Subspace: pca / random 的维数 k')
    parser.add_argument('--ddo_dirs_seed', type=int, default=0,
                        help='DDO-Subspace: 随机标架的种子 (与训练 seed 相互独立; 同一标架的不同剂量实验之间保持不变)')
    parser.add_argument("--model_selection", type=str, default='final',
                        choices=['final', 'target'],
                        help='F1 选模协议: final=固定epoch/严格DG(默认); target=按target_acc选best(历史对照)')
    args = parser.parse_args()

    return args