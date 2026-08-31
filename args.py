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
    parser.add_argument('--weight_mode', type=str, default='prior_residual',
                        choices=['none', 'fixed', 'residual', 'prior_residual'],
                        help='none=Original DDO; fixed=Soft Reliability; residual=Residual Only; prior_residual=full method')
    ## Chapter 4: Domain Subspace Guided Invariant Classification ##
    parser.add_argument('--ch4_mode', type=str, default='full',
                        choices=['ddo', 'hardproj', 'softproj', 'headonly', 'full'],
                        help='Ch4 ablation modes (only used with --CBM_type clip_cbm_subspace)')
    parser.add_argument('--tau', type=float, default=0.8,
                        help='energy threshold for adaptive subspace dim m(tau)')
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
    args = parser.parse_args()

    return args