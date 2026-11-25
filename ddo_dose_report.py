"""
DDO-Dose 分析 (2026-11-25): 剂量曲线、配对差值、梯度诊断、统一评价口径下的 CE-DDO 权衡。

读取 run_matrix.py dose / weights 两个矩阵的结果 json (logs/results/CUB_<tag>_seed<k>.json)。
统一评价口径 (外审要求): 横向比较一律用 final_diag 里的量 ——
  source_ce = 最终模型在固定源域验证集 (source_test) 上的平均 CE
  ddo_eq    = 最终模型在【原始完整方向集】上的等权 DDO 损失 |h(P)| 的均值 (与各方法的加权方式、alpha 无关)
  target_acc= 最终模型的目标域精度 (严格 final 协议, 训练期间未访问目标域)
不使用各方法训练时自己的加权损失。

剂量曲线只作描述与机制分析, 不依据目标域精度挑选 alpha (3 seeds, 仅 CUB)。
产物: experiment_results/DDO-Dose/dose_cub.json、dose_cub.md、fig_dose_curve.png、fig_tradeoff.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

OUT_DIR = os.path.join("experiment_results", "DDO-Dose")
SEEDS = (1, 2, 3)
DOSE = [(0.0, "dose_a0"), (0.125, "dose_a0p125"), (0.25, "dose_a0p25"), (0.5, "dose_a0p5"),
        (1.0, "dose_a1"), (2.0, "dose_a2"), (4.0, "dose_a4")]
VARIANTS = [("w_rwddo_a1", "RW-DDO (prior_residual, KL 0.01)", "RW-DDO"),
            ("w_rwfixed_a1", "RW-Fixed (固定先验, 无残差)", "RW-Fixed"),
            ("w_grouped_a1", "grouped RW-DDO (簇级残差)", "grouped RW-DDO"),
            ("w_cafixed_a1", "CA-Fixed beta=10 (旧交互评分)", "CA-Fixed (old score)")]
# 剂量 x 加权交互: RW-DDO 与等权 DDO 在同一 alpha 下逐 seed 配对
WD = [(0.5, "dose_a0p5", "w_rwddo_a0p5"), (1.0, "dose_a1", "w_rwddo_a1"),
      (2.0, "dose_a2", "w_rwddo_a2"), (4.0, "dose_a4", "w_rwddo_a4")]
GD_TAGS = ["before_epoch1", "before_epoch10", "before_epoch30", "before_epoch50", "final"]


def load(tag, seed):
    p = os.path.join("logs", "results", f"CUB_{tag}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def runs(tag):
    return [j for j in (load(tag, s) for s in SEEDS) if j]


def vals(tag, get):
    return np.array([get(j) for j in runs(tag)])


def ms(x):
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def paired(tag_a, tag_b, get):
    """逐 seed 配对差 a - b (只取两边都有的 seed)"""
    d = []
    for s in SEEDS:
        ja, jb = load(tag_a, s), load(tag_b, s)
        if ja and jb and ja["seed"] == jb["seed"]:
            d.append(get(ja) - get(jb))
    return d


def gd(j, tag, key):
    for g in j.get("grad_diag") or []:
        if g["tag"] == tag:
            return g.get(key)
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    get_t = lambda j: j["final_diag"]["target_acc"]
    get_s = lambda j: j["final_diag"]["source_acc"]
    get_ce = lambda j: j["final_diag"]["source_ce"]
    get_r = lambda j: j["final_diag"]["ddo_eq"]
    get_te = lambda j: j["final_diag"]["target_ce"]

    # ---------- 表 1: 剂量曲线 ----------
    dose_rows = []
    for a, tag in DOSE:
        rs = runs(tag)
        if not rs:
            continue
        t = vals(tag, get_t); s_ = vals(tag, get_s); ce = vals(tag, get_ce); r = vals(tag, get_r); te = vals(tag, get_te)
        row = {"alpha": a, "tag": tag, "n_seeds": len(rs),
               "target_acc": {"seeds": [round(float(v), 3) for v in t], "mean": round(ms(t)[0], 3), "sd": round(ms(t)[1], 3)},
               "source_acc": {"mean": round(ms(s_)[0], 3), "sd": round(ms(s_)[1], 3)},
               "source_ce": {"mean": round(ms(ce)[0], 4), "sd": round(ms(ce)[1], 4)},
               "target_ce": {"mean": round(ms(te)[0], 4), "sd": round(ms(te)[1], 4)},
               "ddo_eq_final": {"mean": round(ms(r)[0], 5), "sd": round(ms(r)[1], 5)},
               "ddo_train_last_epoch": {"mean": round(float(np.mean([j["epoch_metrics"][-1]["ddo"] for j in rs])), 5)},
               "grad": {}}
        for g in GD_TAGS:
            ratio = [gd(j, g, "grad_norm_ratio_ddo_over_ce") for j in rs]
            cos = [gd(j, g, "grad_cos_ce_ddo") for j in rs]
            if all(v is not None for v in ratio):
                row["grad"][g] = {"ratio": round(float(np.mean(ratio)), 5), "ratio_sd": round(float(np.std(ratio, ddof=1)), 5) if len(ratio) > 1 else 0.0,
                                  "cos": round(float(np.mean(cos)), 4), "cos_sd": round(float(np.std(cos, ddof=1)), 4) if len(cos) > 1 else 0.0}
        dose_rows.append(row)
    base_tag = "dose_a1"
    for row in dose_rows:
        d = paired(row["tag"], base_tag, get_t)
        if d:
            row["target_diff_vs_alpha1"] = {"per_seed": [round(v, 3) for v in d], "mean": round(float(np.mean(d)), 3),
                                            "n_positive": int(sum(v > 0 for v in d)), "n": len(d)}

    # ---------- 表 2: 加权变体 (统一口径) ----------
    var_rows = []
    for tag, name, _short in VARIANTS:
        rs = runs(tag)
        if not rs:
            continue
        t = vals(tag, get_t); ce = vals(tag, get_ce); r = vals(tag, get_r); s_ = vals(tag, get_s)
        d = paired(tag, base_tag, get_t)
        d_rw = paired(tag, "w_rwddo_a1", get_t) if tag != "w_rwddo_a1" else []
        row = {"tag": tag, "name": name, "n_seeds": len(rs),
               "target_acc": {"seeds": [round(float(v), 3) for v in t], "mean": round(ms(t)[0], 3), "sd": round(ms(t)[1], 3)},
               "source_acc_mean": round(ms(s_)[0], 3), "source_ce_mean": round(ms(ce)[0], 4),
               "ddo_eq_final_mean": round(ms(r)[0], 5),
               "grad_ratio_init": round(float(np.mean([gd(j, "before_epoch1", "grad_norm_ratio_ddo_over_ce") for j in rs])), 5),
               "grad_ratio_final": round(float(np.mean([gd(j, "final", "grad_norm_ratio_ddo_over_ce") for j in rs])), 5),
               "grad_cos_final": round(float(np.mean([gd(j, "final", "grad_cos_ce_ddo") for j in rs])), 4)}
        if d:
            row["target_diff_vs_alpha1"] = {"per_seed": [round(v, 3) for v in d], "mean": round(float(np.mean(d)), 3),
                                            "n_positive": int(sum(v > 0 for v in d)), "n": len(d)}
        if d_rw:
            row["target_diff_vs_rwddo"] = {"per_seed": [round(v, 3) for v in d_rw], "mean": round(float(np.mean(d_rw)), 3),
                                           "n_positive": int(sum(v > 0 for v in d_rw)), "n": len(d_rw)}
        var_rows.append(row)

    # ---------- 表 3: 把加权变体放到等权剂量曲线上 ----------
    # 曲线: 各 alpha 的 (ddo_eq 均值, source_ce 均值 / target 均值); 在 ddo_eq 上线性插值,
    # 得到"若该变体只是换了一个工作点, 曲线上同 ddo_eq 处应有什么取值", 残差 = 实际 - 曲线。
    curve = sorted([(row["ddo_eq_final"]["mean"], row["source_ce"]["mean"], row["target_acc"]["mean"], row["alpha"])
                    for row in dose_rows], key=lambda z: z[0])
    xs = np.array([c[0] for c in curve]); ce_c = np.array([c[1] for c in curve]); t_c = np.array([c[2] for c in curve])
    trade = {"curve_x_ddo_eq": [round(float(v), 5) for v in xs], "curve_source_ce": [round(float(v), 4) for v in ce_c],
             "curve_target_acc": [round(float(v), 3) for v in t_c], "curve_alpha": [c[3] for c in curve],
             "curve_target_sd": [next(r["target_acc"]["sd"] for r in dose_rows if r["alpha"] == c[3]) for c in curve],
             "variants": []}
    lo, hi = xs.min(), xs.max()
    for row in var_rows:
        x = row["ddo_eq_final_mean"]
        in_range = bool(lo <= x <= hi)
        ce_hat = float(np.interp(x, xs, ce_c)) if in_range else None
        t_hat = float(np.interp(x, xs, t_c)) if in_range else None
        trade["variants"].append({
            "tag": row["tag"], "name": row["name"], "short": _short, "x_ddo_eq": x, "within_curve_range": in_range,
            "source_ce": row["source_ce_mean"], "curve_source_ce_at_x": round(ce_hat, 4) if ce_hat is not None else None,
            "source_ce_residual": round(row["source_ce_mean"] - ce_hat, 4) if ce_hat is not None else None,
            "target_acc": row["target_acc"]["mean"], "curve_target_acc_at_x": round(t_hat, 3) if t_hat is not None else None,
            "target_acc_residual": round(row["target_acc"]["mean"] - t_hat, 3) if t_hat is not None else None})
    # ---------- 表 3b: 剂量 x 加权交互 (RW-DDO 与等权 DDO 同 alpha 逐 seed 配对) ----------
    wd_rows = []
    for a, eq_tag, rw_tag in WD:
        if not runs(rw_tag):
            continue
        t_eq = vals(eq_tag, get_t); t_rw = vals(rw_tag, get_t)
        d = paired(rw_tag, eq_tag, get_t)
        x = float(np.mean(vals(rw_tag, get_r))); ce_rw = float(np.mean(vals(rw_tag, get_ce)))
        in_r = bool(lo <= x <= hi)
        ce_hat = float(np.interp(x, xs, ce_c)) if in_r else None
        t_hat = float(np.interp(x, xs, t_c)) if in_r else None
        wd_rows.append({"alpha": a, "equal_mean": round(float(np.mean(t_eq)), 3), "rwddo_mean": round(float(np.mean(t_rw)), 3),
                        "rwddo_seeds": [round(float(v), 3) for v in t_rw],
                        "diff_per_seed": [round(v, 3) for v in d], "diff_mean": round(float(np.mean(d)), 3),
                        "n_positive": int(sum(v > 0 for v in d)), "n": len(d),
                        "rwddo_ddo_eq": round(x, 5), "equal_ddo_eq": round(float(np.mean(vals(eq_tag, get_r))), 5),
                        "rwddo_source_ce": round(ce_rw, 4),
                        "curve_source_ce_at_x": round(ce_hat, 4) if ce_hat is not None else None,
                        "source_ce_residual": round(ce_rw - ce_hat, 4) if ce_hat is not None else None,
                        "target_acc_residual": round(float(np.mean(t_rw)) - t_hat, 3) if t_hat is not None else None})

    # 历史参照 (不同评估协议: 每 epoch 评估目标域; 此处仅作旁注, 不参与比较)
    result = {"dataset": "CUB", "seeds": list(SEEDS), "n_seeds_found": {r["tag"]: r["n_seeds"] for r in dose_rows + var_rows},
              "protocol": "main_cached.py, clip_cbm, batch 256, 100 epochs, final 协议, 训练期间不访问目标域 (--target_eval final_only)",
              "dose": dose_rows, "variants": var_rows, "tradeoff": trade, "rwddo_by_alpha": wd_rows}
    with open(os.path.join(OUT_DIR, "dose_cub.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # ---------- Markdown ----------
    L = ["# DDO-Dose：正则强度扫描（CUB，3 seeds，2026-11-25）", "",
         "**配置**：第三章入口 `clip_cbm`、等权 DDO（`weight_mode=none`）、batch 256、100 epochs、严格 final 协议；"
         "训练期间不访问目标域（`--target_eval final_only`），仅在最后一个 epoch 评估一次。"
         "所有 α 的其余配置完全一致。α=1 即第三章的等权 DDO 基线（本次重跑）。", "",
         "## 1. 剂量曲线", "",
         "| α | 目标域精度（逐 seed） | 均值 ± sd | 相对 α=1（逐 seed） | 均值 | 源域精度 | 源域验证 CE | 最终等权 DDO 损失 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in dose_rows:
        t = r["target_acc"]; d = r.get("target_diff_vs_alpha1")
        seeds_s = " / ".join(f"{v:.2f}" for v in t["seeds"])
        if d:
            pair_s = " / ".join(f"{v:+.2f}" for v in d["per_seed"]); mean_s = f"{d['mean']:+.2f}"
        else:
            pair_s, mean_s = "— (基准)", "—"
        L.append(f"| {r['alpha']} | {seeds_s} | **{t['mean']:.2f}** ± {t['sd']:.2f} | {pair_s} | {mean_s} | "
                 f"{r['source_acc']['mean']:.2f} | {r['source_ce']['mean']:.4f} | {r['ddo_eq_final']['mean']:.4f} |")
    L += ["", "配对差值按相同 seed 相减（正数表示该 α 优于 α=1）。", "", "## 2. 梯度诊断", "",
          "固定批次（训练集前 256 个样本）上，等权 DDO 损失与分类损失对分类头参数的梯度：范数比 ‖∇DDO‖/‖∇CE‖ 与夹角余弦。下表为 3 seeds 均值。", "",
          "| α | " + " | ".join(f"{g.replace('before_epoch', 'epoch ')} 比 / cos" for g in GD_TAGS) + " |",
          "|---|" + "---|" * len(GD_TAGS)]
    for r in dose_rows:
        cells = []
        for g in GD_TAGS:
            v = r["grad"].get(g)
            cells.append(f"{v['ratio']:.3f} / {v['cos']:+.3f}" if v else "—")
        L.append(f"| {r['alpha']} | " + " | ".join(cells) + " |")
    L += ["", "## 3. 加权变体（统一评价口径）", "",
          "这些 run 与剂量扫描共用同一套配置（同协议、同 batch、同 seed），只是加权方式不同。", "",
          "| 方法 | 目标域精度（逐 seed） | 均值 ± sd | 相对等权 α=1（逐 seed） | 均值 | 源域验证 CE | 最终等权 DDO 损失 | 初始梯度比 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in var_rows:
        t = r["target_acc"]; d = r.get("target_diff_vs_alpha1")
        seeds_s = " / ".join(f"{v:.2f}" for v in t["seeds"])
        if d:
            pair_s = " / ".join(f"{v:+.2f}" for v in d["per_seed"]); mean_s = f"{d['mean']:+.2f}"
        else:
            pair_s, mean_s = "—", "—"
        L.append(f"| {r['name']} | {seeds_s} | **{t['mean']:.2f}** ± {t['sd']:.2f} | {pair_s} | {mean_s} | "
                 f"{r['source_ce_mean']:.4f} | {r['ddo_eq_final_mean']:.4f} | {r['grad_ratio_init']:.3f} |")
    L += ["", "## 4. CE–DDO 权衡：加权变体是否只是换了工作点", "",
          "横轴为最终等权 DDO 损失（约束被满足的程度，越小越强），纵轴分别为源域验证 CE 与目标域精度。"
          "等权剂量曲线给出这两个量随 α 的变化；把每个加权变体按其自身的等权 DDO 损失放到曲线上，"
          "残差 = 实际值 − 曲线在同一 DDO 损失处的插值。残差接近 0 表示该变体落在等权 DDO 的工作点轨迹上。", "",
          "| 方法 | 等权 DDO 损失 | 源域 CE（实际 / 曲线） | 源域 CE 残差 | 目标域精度（实际 / 曲线） | 目标域精度残差 |",
          "|---|---|---|---|---|---|"]
    for v in trade["variants"]:
        if v["within_curve_range"]:
            L.append(f"| {v['name']} | {v['x_ddo_eq']:.4f} | {v['source_ce']:.4f} / {v['curve_source_ce_at_x']:.4f} | "
                     f"{v['source_ce_residual']:+.4f} | {v['target_acc']:.2f} / {v['curve_target_acc_at_x']:.2f} | {v['target_acc_residual']:+.2f} |")
        else:
            L.append(f"| {v['name']} | {v['x_ddo_eq']:.4f} | 超出曲线范围 | — | 超出曲线范围 | — |")
    L += ["", "## 4b. 剂量 × 加权交互：RW-DDO 与等权 DDO 在同一 α 下逐 seed 配对", "",
          "| α | 等权 DDO | RW-DDO（逐 seed） | RW-DDO 均值 | RW-DDO − 等权（逐 seed） | 均值 | 正向 seed 数 | RW-DDO / 等权 的最终等权 DDO 损失 | 源域 CE 残差（对曲线） |",
          "|---|---|---|---|---|---|---|---|---|"]
    for w in wd_rows:
        L.append(f"| {w['alpha']} | {w['equal_mean']:.2f} | {' / '.join(f'{v:.2f}' for v in w['rwddo_seeds'])} | {w['rwddo_mean']:.2f} | "
                 f"{' / '.join(f'{v:+.2f}' for v in w['diff_per_seed'])} | {w['diff_mean']:+.2f} | {w['n_positive']}/{w['n']} | "
                 f"{w['rwddo_ddo_eq']:.4f} / {w['equal_ddo_eq']:.4f} | "
                 f"{w['source_ce_residual']:+.4f}" if w['source_ce_residual'] is not None else
                 f"| {w['alpha']} | {w['equal_mean']:.2f} | {' / '.join(f'{v:.2f}' for v in w['rwddo_seeds'])} | {w['rwddo_mean']:.2f} | — | 超出曲线范围 |")
    L += ["", "## 5. 读法与限制", "",
          "- 3 个 seed、仅 CUB。表中给出逐 seed 数值，便于核对均值是否由单个 seed 驱动。",
          "- 剂量曲线只作描述：**不依据目标域精度挑选 α**。若日后要用 α≠1 的取值，必须有只依赖源域的选择规则。",
          "- α=1 的等权基线、RW-DDO 等加权变体都是本次在 `--target_eval final_only` 下重跑的；"
          "与此前逐 epoch 评估目标域的历史 run（如 E1 的 RW-DDO 56.02）数值不完全可比，因为训练期间的随机数消耗不同。",
          "- 曲线插值是线性的，曲线上相邻点间距不大；残差的量级应与 seed 间波动（约 ±0.3–0.5）一起看。",
          ""]
    with open(os.path.join(OUT_DIR, "dose_cub.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # ---------- 图 (英文标注, 避免字体缺字) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        al = [r["alpha"] for r in dose_rows]
        tm = [r["target_acc"]["mean"] for r in dose_rows]; ts = [r["target_acc"]["sd"] for r in dose_rows]
        sm = [r["source_acc"]["mean"] for r in dose_rows]; ss = [r["source_acc"]["sd"] for r in dose_rows]
        xpos = list(range(len(al)))
        fig, ax = plt.subplots(2, 1, figsize=(7.2, 7.6))
        ax[0].errorbar(xpos, tm, yerr=ts, marker="o", capsize=3, color="#1f4e9e", label="target-domain accuracy (final model)")
        ax[0].errorbar(xpos, sm, yerr=ss, marker="s", capsize=3, color="#b8860b", label="source-domain accuracy (final model)")
        for i, r in enumerate(dose_rows):
            ax[0].annotate(f"{tm[i]:.2f}", (xpos[i], tm[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax[0].set_xticks(xpos); ax[0].set_xticklabels([str(a) for a in al])
        ax[0].set_xlabel("DDO weight alpha (0 = CE only)"); ax[0].set_ylabel("accuracy (%)")
        ax[0].set_title("DDO dose-response, CUB, mean +/- sd over 3 seeds (strict final protocol)")
        ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax2 = ax[1]
        ax2.plot(xs, t_c, "-o", color="#1f4e9e", label="equal-weight DDO dose curve")
        for c, r in zip(curve, dose_rows):
            ax2.annotate(f"a={c[3]}", (c[0], c[2]), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#1f4e9e")
        marks = ["D", "^", "v", "P"]
        short = {tag: s for tag, _n, s in VARIANTS}          # tag -> 图例用的英文短名
        for v, m in zip(trade["variants"], marks):
            if v["within_curve_range"]:
                ax2.plot(v["x_ddo_eq"], v["target_acc"], m, ms=9, label=short[v["tag"]])
        if wd_rows:
            ax2.plot([w["rwddo_ddo_eq"] for w in wd_rows], [w["rwddo_mean"] for w in wd_rows], "D--", ms=7,
                     color="#c0392b", label="RW-DDO at alpha 0.5/1/2/4")
        ax2.set_xscale("log"); ax2.set_xlabel("equal-weight DDO loss of the final model (log scale; smaller = stronger)")
        ax2.set_ylabel("target-domain accuracy (%)")
        ax2.set_title("Weighting variants on the equal-weight dose curve", fontsize=11)
        ax2.grid(alpha=0.3); ax2.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_dose_curve.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(xs, ce_c, "-o", color="#1f4e9e", label="equal-weight DDO dose curve")
        for c in curve:
            ax.annotate(f"a={c[3]}", (c[0], np.interp(c[0], xs, ce_c)), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#1f4e9e")
        for v, m in zip(trade["variants"], marks):
            if v["within_curve_range"]:
                ax.plot(v["x_ddo_eq"], v["source_ce"], m, ms=9, label=short[v["tag"]])
        if wd_rows:
            xr = [w["rwddo_ddo_eq"] for w in wd_rows]; yr = [w["rwddo_source_ce"] for w in wd_rows]
            ax.plot(xr, yr, "D--", ms=7, color="#c0392b", label="RW-DDO at alpha 0.5/1/2/4")
        ax.set_xscale("log"); ax.set_xlabel("equal-weight DDO loss of the final model (log scale; smaller = stronger constraint)")
        ax.set_ylabel("CE on the held-out source split (final model)")
        ax.set_title("CE vs. equal-weight DDO loss: do weighting variants leave the dose curve?")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_tradeoff.png"), dpi=130)
        plt.close(fig)
        print("[fig] saved fig_dose_curve.png, fig_tradeoff.png")
    except Exception as e:  # 图是辅助, 失败不影响数据产物
        print("[fig] skipped:", repr(e))

    print("\n".join(L))
    print(f"\n[saved] {os.path.join(OUT_DIR, 'dose_cub.json')} / dose_cub.md")


if __name__ == "__main__":
    main()
