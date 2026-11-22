"""
δ–ℓ 耦合诊断 (2026-11-20, 外审驱动): RW-DDO 的可学习残差是否存在 "weight avoidance"
(残差倾向于下调高 DDO 损失描述符的权重, 从而逃避困难约束)?

理论背景: L_DDO = Σ_i p_i ℓ_i, p = softmax(r+δ) ⇒ ∂L/∂δ_i = p_i(ℓ_i - Σ_j p_j ℓ_j),
即梯度下降天然倾向于下调 ℓ_i 高于加权平均的描述符的权重。
本脚本不预设结论, 只量化:
  (1) 静态层: 收敛时 corr(δ, ℓ), corr(w, ℓ), corr(r, ℓ) —— 检查点自洽 (δ 与分类器同源)
  (2) 动态层: 每 5 epoch 保存分类器快照 => 同一 epoch 的 corr(δ(t), ℓ(t)) 真实曲线
      (不重建历史分类器, 遵守 "不要用最终 ℓ 伪造 ℓ(t)" 的审查意见)
  (3) 参照层: 原始主表三次 run (CUB_prior_residual_seed1/2/3) 的 δ(t) 动态

实现: 复用 CachedTrainingSession (与主表同一条缓存管线, 逐位等价), 但
  -- save_model 强制关闭 (不覆盖任何 best_*.pth; 2026-11-20 已发现 smoke test
     覆盖收敛检查点的事故, 本脚本杜绝再发生)
  -- model_selection=final => 结果写入 *_sel_seed*.json, 与主表 json 零冲突
  -- 分类器快照存独立诊断目录

用法:
  python weight_loss_coupling_diagnosis.py --dataset CUB --seed 1 \
      --CLIP_type "F:/CLIP-OOD/clip_weights/ViT-L-14.pt"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 支持 python -I 直跑
# GBK 控制台保险: 在 wandb 安装 stdout 代理之前重配置为 utf-8 (wandb 代理持有同一对象, 就地生效)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import glob
import json
import random

import numpy as np
import torch

DIAG_DIR = os.path.join("experiment_results", "07_权重逃逸诊断")
SNAP_DIR = os.environ.get("DIAG_SNAP_DIR", os.path.join(DIAG_DIR, "cls_snapshots"))
SNAP_EVERY = 5   # 分类器快照间隔 (epoch)


class DiagSession:
    """包装 CachedTrainingSession: 每 SNAP_EVERY 个 epoch 额外保存分类器快照 + 真实同 epoch ℓ"""

    def __init__(self, args, snapshot_dir=None):
        from main_cached import CachedTrainingSession
        self.args = args
        self.snapshot_dir = snapshot_dir or SNAP_DIR
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.session = CachedTrainingSession(args)
        self._orig_log = self.session._log_domain_weights
        self.session._log_domain_weights = self._log_and_snapshot

    def _log_and_snapshot(self, epoch):
        out = self._orig_log(epoch)
        if epoch % SNAP_EVERY == 0 or epoch == self.args.epochs - 1:
            model = self.session.model
            snap = {"epoch": epoch,
                    "cls": {k: v.detach().cpu().clone()
                            for k, v in model.classifier.state_dict().items()}}
            # 真实同 epoch ℓ_i = |h_t(proj_i)|.mean (E1 象限定义与动态层都用)
            if hasattr(model, 'domain_concept_projection') and \
                    getattr(model, 'domain_residual', None) is not None:
                with torch.no_grad():
                    reg_t = model.classifier[1:](model.domain_concept_projection)
                    snap["ell"] = reg_t.abs().mean(dim=(1, 2)).cpu()
            torch.save(snap, os.path.join(self.snapshot_dir, f"epoch_{epoch:04d}.pt"))
        return out

    def run(self):
        self.session.run()
        return self.session


# ---------------------------------------------------------------- 数值工具
def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def neff(w):
    p = np.asarray(w, dtype=np.float64)
    p = p / p.sum()
    return float(1.0 / np.sum(p ** 2))


# ---------------------------------------------------------------- 主流程
def main():
    from args import get_args
    args = get_args()

    # ---- 强制安全开关: 诊断 run 不得触碰任何正式产物 ----
    args.save_model = False          # 不写 best_*.pth (2026-11-20 覆盖事故的直接教训)
    args.wandb = False
    args.model_selection = "final"   # => 结果 json 自动带 _sel 后缀, 与主表零冲突
    args.CBM_type = "clip_cbm"       # 第三章模型 (clip_cbm_orth)
    args.weight_mode = "prior_residual"
    if args.epochs < 20:
        args.epochs = 100            # 诊断需要收敛轨迹, 拒绝 smoke 短跑

    os.makedirs(DIAG_DIR, exist_ok=True)
    run_tag = f"{args.dataset}_prior_residual_sel_seed{args.seed}"
    print(f"[diag] run={run_tag}  epochs={args.epochs}  save_model={args.save_model}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    session = DiagSession(args).run()
    model = session.model
    device = args.device

    with torch.no_grad():
        # ===== 静态层: 收敛状态下, δ 与分类器天然自洽 (同一 state) =====
        proj = model.domain_concept_projection.to(device).float()     # (204, C, 312)
        reg = model.classifier[1:](proj)                              # 与 forward 完全一致
        ell = reg.abs().mean(dim=(1, 2)).cpu().numpy()                # ℓ_i = |h(proj_i)|.mean
        delta = model.domain_residual.detach().cpu().numpy()
        prior = model.reliability_prior.detach().cpu().numpy()
        w_final, _ = model.get_domain_weights()
        w_final = w_final.detach().cpu().numpy()
        w_prior = model.get_prior_weights().detach().cpu().numpy()
        red = None
        if getattr(model, "descriptor_redundancy", None) is not None:
            red = model.descriptor_redundancy.detach().cpu().numpy()
        delta_weight = w_final - w_prior

        # 描述符文本 (与 domain_diffs 顺序一致)
        from prompts.prompt200new import target_text_prompts
        texts = list(target_text_prompts)
        assert len(texts) == len(delta), f"prompt 数不匹配: {len(texts)} vs {len(delta)}"

        stats = {
            "run": run_tag,
            "epochs": args.epochs,
            "pearson_delta_ell": pearson(delta, ell),
            "spearman_delta_ell": spearman(delta, ell),
            "pearson_w_ell": pearson(w_final, ell),
            "pearson_wprior_ell": pearson(w_prior, ell),
            "pearson_dw_ell": pearson(delta_weight, ell),
            "pearson_r_ell": pearson(prior, ell),
            "pearson_r_delta": pearson(prior, delta),
            "pearson_dw_r": pearson(delta_weight, prior),
            "neff_final": neff(w_final),
            "neff_prior": neff(w_prior),
            "delta_norm": float(np.linalg.norm(delta)),
            "ell_mean": float(ell.mean()),
            "ell_std": float(ell.std()),
        }

        # ===== 动态层 1: 本 run 的 δ(t) 与同 epoch ℓ(t) (真实分类器快照) =====
        snap_files = sorted(glob.glob(os.path.join(SNAP_DIR, "epoch_*.pt")))
        dynamic_same_epoch = []
        for f in snap_files:
            snap = torch.load(f, map_location="cpu", weights_only=False)
            e = snap["epoch"]
            proj_cpu = proj.detach().cpu().float()
            ln = torch.nn.LayerNorm(proj_cpu.size(-1))
            lin = torch.nn.Linear(proj_cpu.size(-1), reg.size(-1))
            ln.load_state_dict({"weight": snap["cls"]["1.weight"], "bias": snap["cls"]["1.bias"]})
            lin.load_state_dict({"weight": snap["cls"]["2.weight"], "bias": snap["cls"]["2.bias"]})
            with torch.no_grad():
                r_t = lin(ln(proj_cpu)).abs().mean(dim=(1, 2)).numpy()
            wlog = os.path.join("logs", "weights", run_tag, f"epoch_{e:04d}.pt")
            d_t = torch.load(wlog, map_location="cpu", weights_only=False)["residual"].numpy()
            dynamic_same_epoch.append({
                "epoch": int(e),
                "corr_delta_ell_same_epoch": pearson(d_t, r_t),
                "delta_norm": float(np.linalg.norm(d_t)),
                "ell_mean": float(r_t.mean()),
            })

        # ===== 动态层 2: δ(t) 全分辨率 (本 run + 原始主表三次 run 参照) =====
        wlog_base = os.path.join("logs", "weights")
        delta_traj = {}
        for tag in [run_tag, f"{args.dataset}_prior_residual_seed1",
                    f"{args.dataset}_prior_residual_seed2", f"{args.dataset}_prior_residual_seed3"]:
            files = sorted(glob.glob(os.path.join(wlog_base, tag, "epoch_*.pt")))
            if not files:
                continue
            traj, ent = [], []
            for f in files:
                d = torch.load(f, map_location="cpu", weights_only=False)
                traj.append(float(np.linalg.norm(d["residual"].numpy())))
                ent.append(d.get("weight_entropy", float("nan")))
            delta_traj[tag] = {"delta_norm": traj, "entropy": ent,
                               "corr_delta_t_ell_final": [
                                   pearson(torch.load(f, map_location="cpu", weights_only=False)["residual"].numpy(), ell)
                                   for f in files]}

        # ===== 逐描述符明细 (按 δ 排序 Top/Bottom) =====
        order = np.argsort(delta)
        rows = []
        for i in order:
            rows.append({
                "idx": int(i), "text": texts[i],
                "reliability": round(float(prior[i]), 4),
                "delta": round(float(delta[i]), 4),
                "ell": round(float(ell[i]), 4),
                "w_final": round(float(w_final[i]), 4),
                "dw": round(float(delta_weight[i]), 4),
            })

        result = {
            "static": stats,
            "dynamic_same_epoch": dynamic_same_epoch,
            "delta_traj": delta_traj,
            "per_prompt": rows,
        }
        out_json = os.path.join(DIAG_DIR, f"diag_{run_tag}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

        # ---- 控制台摘要 (任何编码下都不致命) ----
        try:
            print("\n===== delta-ell 耦合诊断摘要 =====")
            for k, v in stats.items():
                print(f"  {k:24s} {v}")
            print("  --- 同 epoch corr(delta(t), ell(t)) 演化 ---")
            for d in dynamic_same_epoch:
                print(f"  epoch {d['epoch']:3d}  corr={d['corr_delta_ell_same_epoch']:+.4f}"
                      f"  |delta|={d['delta_norm']:.4f}  mean_ell={d['ell_mean']:.4f}")
            print(f"  [saved] {out_json}")
        except UnicodeEncodeError:
            pass  # 结果已落盘, 打印失败不影响实验


if __name__ == "__main__":
    main()
