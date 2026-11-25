"""
语言域椭球的构造 (主线 B 公共底座; 规格见 doc/主线B实施规格.md §2、§6)。Stage 0B / 0C / 1C 共用。

记号: u_{i,c} = LN0(E d_{i,c}) 是域方向 (N 个描述符 x C 个文本类别), 恒位于零均值超平面 H (1^T u = 0)。
  G      = E[u u^T]   非中心化二阶矩 (C 细节 1: u 是 target - photo 的差分, 均值 mu 就是"离开照片域"的公共迁移方向;
                            按惯例中心化会抹掉这条轴 —— 所以默认用 G, 另给 Centered Cov 对照)
  Sigma  = L L^T      椭球 {L q : |q| <= eps} 的生成元 (G3: 低秩生成式, 避开奇异协方差求逆)

三 / 四个对照椭球, 迹都归一化为 1:
  identity   H / (K-1)                          各向同性, 限制在超平面内
  language   V Lam V^T                          语言二阶矩 (可截断到 r 维: 先截断再迹归一化)
  random     Q Lam Q^T                          同谱随机旋转: 与 language 完全相同的截断谱 / 秩 / 迹, 只有特征向量方向不同
  flat       (可选) Q_r Q_r^T / r               随机低秩平谱 (G3: 区分"低秩本身"与"非均匀谱")

构造在 float64 下完成, 转成 float32 交付。全部可复现 (随机正交基用独立 generator), 并给出 sha256 摘要。
用法 (自检): python language_ellipsoid.py
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

EPS_LN = 1e-5


def ln0(x, eps=EPS_LN):
    """无 affine 的 LayerNorm (与 nn.LayerNorm 的归一化部分一致: 有偏方差) —— 与训练实现同一公式"""
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)


def domain_directions(diffs, E):
    """训练时 DDO 实际约束的方向 u = LN0(D E^T), 展平成 (N*C, K) 的 float64 矩阵。
    diffs: (N, C, 768) 逐类单位方向; E: (K, 768) 归一化的概念矩阵。与 clip_cbm_orth 的 pca_src='ln0' 分支同一构造。"""
    A = (diffs.double() @ E.double().T).reshape(-1, E.size(0))
    return ln0(A)


def second_moments(U):
    """(G 非中心化, G_cov 中心化, mu 均值, eta_mean 均值能量占比)"""
    G = U.T @ U / U.size(0)
    mu = U.mean(0)
    Uc = U - mu
    G_cov = Uc.T @ Uc / U.size(0)
    eta = float(mu @ mu / (G.diagonal().sum()))
    return G, G_cov, mu, eta


def _random_onb_in_hyperplane(K, r, seed):
    """零均值超平面 H 内的随机正交基 (K, r), 满足 1^T Q = 0、Q^T Q = I。独立 generator, 不动全局随机数。"""
    g = torch.Generator().manual_seed(int(seed))
    Q = torch.randn(K, r, generator=g, dtype=torch.float64)
    Q = Q - Q.mean(0, keepdim=True)                      # 中心化: 落入 H
    Q = torch.linalg.qr(Q)[0]
    assert float(Q.mean(0).abs().max()) < 1e-8, "随机基不在零均值超平面内"
    assert float((Q.T @ Q - torch.eye(r, dtype=torch.float64)).abs().max()) < 1e-8, "随机基不正交"
    return Q


def build_ellipsoids(G, r=32, seed=0, with_flat=True):
    """由二阶矩 G (K, K) 构造各对照椭球的生成元 L (K, r_i), 使 tr(L L^T) = 1。

    language: G 的前 r 个特征对 -> 先截断、再迹归一化 (C 细节 3.1)
    random:   同一个谱 Lam_lang, 换成 H 内的随机正交基
    identity: H / (K-1), 迹为 1 (满秩)
    flat:     r 维随机平谱 (迹 1) —— 只用来区分"低秩本身"
    返回 {"language": L, "random": L, "identity": L, "flat": L|None, "spectrum": {...}, "sha256": {...}}
    """
    K = G.size(0)
    G = G.double()
    evals, evecs = torch.linalg.eigh(G)                   # 升序
    evals, evecs = evals.flip(0), evecs.flip(1)           # 降序
    evals = evals.clamp_min(0)
    r = int(min(r, K - 1))                                # H 的维数是 K-1, 至多取到 K-1
    lam = evals[:r]
    V = evecs[:, :r]
    # 特征向量应落在 H 内 (G 的列空间在 H 中); 数值上做一次投影清理
    assert float(V.mean(0).abs().max()) < 1e-6, "G 的特征向量不在零均值超平面内 (G 不是由超平面内的方向生成的?)"
    scale = float(lam.sum()) ** -0.5                      # 迹归一化: tr(L L^T) = sum(lam) * scale^2 = 1
    L_lang = V * lam.sqrt() * scale
    Q = _random_onb_in_hyperplane(K, r, seed)
    L_rand = Q * lam.sqrt() * scale                       # 同谱、同秩、同迹
    L_id = torch.eye(K, dtype=torch.float64) - torch.ones(K, K, dtype=torch.float64) / K
    L_id = L_id / (K - 1) ** 0.5                          # H / sqrt(K-1): L L^T = H/(K-1), 迹 = 1
    out = {"language": L_lang, "random": L_rand, "identity": L_id, "flat": None}
    if with_flat:
        Qf = _random_onb_in_hyperplane(K, r, seed + 1)
        out["flat"] = Qf / r ** 0.5
    total = float(evals.sum())
    out["spectrum"] = {
        "r": r,
        "top_eigenvalues": [round(float(v), 6) for v in lam[:8]],
        "energy_captured_by_r": float(lam.sum() / total),
        "effective_rank_entropy_all": float(torch.exp(-(evals / total * (evals / total + 1e-300).log()).sum())),
        "effective_rank_entropy_trunc": float(torch.exp(-(lam / lam.sum() * (lam / lam.sum() + 1e-300).log()).sum())),
        "lambda_max_over_lambda_min_trunc": float(lam[0] / lam[-1].clamp_min(1e-300)),
    }
    out["sha256"] = {k: hashlib.sha256(v.float().numpy().tobytes()).hexdigest()[:16]
                     for k, v in out.items() if torch.is_tensor(v)}
    # ---- 自检: 谱 / 迹 / 秩 / 超平面 ----
    for name in ("language", "random", "identity", "flat"):
        Lx = out[name]
        if Lx is None:
            continue
        S = Lx @ Lx.T
        assert abs(float(torch.trace(S)) - 1.0) < 1e-9, f"{name}: 迹不是 1"
        assert float(Lx.mean(0).abs().max()) < 1e-8, f"{name}: 不在零均值超平面内"
        assert float((S - S.T).abs().max()) < 1e-10, f"{name}: 不对称"
        ev = torch.linalg.eigvalsh(S).clamp_min(0)
        assert float(ev[-1]) <= 1.0 + 1e-9, f"{name}: 特征值超过 1"
        if name in ("language", "random"):
            assert torch.allclose(torch.linalg.eigvalsh(S).flip(0), torch.linalg.eigvalsh(L_lang @ L_lang.T).flip(0),
                                  atol=1e-9), f"{name}: 谱与 language 不一致"
    # language 与 random 的主轴必须不同 (否则对照无意义)
    cos_principal = float(abs(V[:, 0] @ Q[:, 0]))
    assert cos_principal < 0.5, f"random 的首主轴与 language 的首主轴夹角余弦过大: {cos_principal:.3f}"
    out["spectrum"]["cos_first_axis_language_vs_random"] = round(cos_principal, 6)
    return out


def self_test():
    """合成数据上的自检 + 真实 CUB 域方向上的谱概况 (只用文本侧数据, 不涉及图像)"""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("  OK  " if cond else "  BAD ") + name + (f"  [{detail}]" if detail else ""))

    torch.manual_seed(0)
    # ---- 合成: 已知谱 ----
    K, N = 60, 4000
    U = torch.randn(N, K, dtype=torch.float64) * torch.linspace(4, 0.1, K)
    U = U - U.mean(-1, keepdim=True)                      # 超平面内
    G, G_cov, mu, eta = second_moments(U)
    check("S1 G 对称、半正定", float((G - G.T).abs().max()) < 1e-10 and float(torch.linalg.eigvalsh(G).min()) > -1e-10)
    check("S2 tr(G) = tr(G_cov) + |mu|^2 (非中心化 = 中心化 + 均值)", abs(float(torch.trace(G) - torch.trace(G_cov) - mu @ mu)) < 1e-9)
    check("S3 eta_mean = |mu|^2 / tr(G)", abs(eta - float(mu @ mu / torch.trace(G))) < 1e-12)
    e = build_ellipsoids(G, r=10, seed=3)
    for name in ("language", "random", "identity", "flat"):
        check(f"S4 {name}: tr=1、在超平面内", abs(float(torch.trace(e[name] @ e[name].T)) - 1) < 1e-9
              and float(e[name].mean(0).abs().max()) < 1e-8)
    check("S5 random 与 language 谱相同", torch.allclose(torch.linalg.eigvalsh(e["random"] @ e["random"].T),
                                                  torch.linalg.eigvalsh(e["language"] @ e["language"].T), atol=1e-10))
    check("S6 可复现 (同 seed 同摘要)", build_ellipsoids(G, 10, 3)["sha256"]["random"] == e["sha256"]["random"])
    check("S7 不同 seed 的 random 不同", build_ellipsoids(G, 10, 4)["sha256"]["random"] != e["sha256"]["random"])
    check("S8 r 截断的能量占比在 (0,1] 内且随 r 增大", build_ellipsoids(G, 5, 0)["spectrum"]["energy_captured_by_r"]
          <= build_ellipsoids(G, 20, 0)["spectrum"]["energy_captured_by_r"] <= 1 + 1e-12)
    # 先截断再归一化: 截断谱的迹为 1, 而不是原谱的迹
    check("S9 先截断再迹归一化 (截断后的迹为 1, 不是原 G 的迹)", abs(float(torch.trace(e["language"] @ e["language"].T)) - 1) < 1e-9
          and abs(float(torch.trace(G)) - 1) > 1e-3)
    print("SELFTEST:", "PASS" if ok else "FAIL")

    # ---- 真实 CUB 域方向 (文本侧) ----
    if "--real" in sys.argv:
        from args import get_args
        sys.argv = [sys.argv[0], "--dataset", "CUB", "--seed", "1",
                    "--CLIP_type", os.path.abspath("clip_weights/ViT-L-14.pt"), "--epochs", "1"]
        args = get_args()
        args.save_model = False
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = __import__("data").get_dataset_classes(args)[0]
        diffs = ds.domain_diffs.double().cpu()
        import clip
        clip_model, _ = clip.load(os.path.abspath("clip_weights/ViT-L-14.pt"), device=args.device)
        with torch.no_grad():
            E = clip_model.encode_text(clip.tokenize(list(ds.concept2id.keys())).to(args.device)).double().cpu()
            E = E / E.norm(dim=-1, keepdim=True)
        del clip_model
        U = domain_directions(diffs, E)
        G, G_cov, mu, eta = second_moments(U)
        print(f"\n真实 CUB 域方向: 形状 {tuple(U.shape)}, 行范数均值 {float(U.norm(dim=-1).mean()):.4f}")
        print(f"均值能量占比 eta_mean = {eta:.4f}   (非中心化 tr(G) = {float(torch.trace(G)):.4f}; 中心化 tr = {float(torch.trace(G_cov)):.4f})")
        print(f"行范数 (零均值 => 应为 sqrt(K)): {float(U.norm(dim=-1).mean()):.4f} vs sqrt(K) = {U.size(1) ** 0.5:.4f}")
        for r in (8, 32, 64, 128, 311):
            e = build_ellipsoids(G, r=r, seed=0)
            s = e["spectrum"]
            print(f"  r={r:3d}: 截断捕获能量 {s['energy_captured_by_r']:.4f} | 截断后有效秩 {s['effective_rank_entropy_trunc']:.2f} | "
                  f"λmax/λmin {s['lambda_max_over_lambda_min_trunc']:.3g} | 首轴余弦(language vs random) {s['cos_first_axis_language_vs_random']:.3f}")
        print(f"  全谱有效秩: {build_ellipsoids(G, 8)['spectrum']['effective_rank_entropy_all']:.2f}")
        ec = torch.linalg.eigvalsh(G_cov).flip(0).clamp_min(0)
        print(f"  中心化协方差 G_cov 的谱: 首 4 个特征值 { [round(float(v), 4) for v in ec[:4]] }, tr = {float(ec.sum()):.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
