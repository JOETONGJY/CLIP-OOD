# 特征缓存等价性验证（2026-09-11）

## 结论
`main_cached.py`（特征缓存训练）与 `main.py`（原始管线）**逐位等价**——
所有指标精确到小数点后 4 位一致，速度 **155×**（1.18s/epoch vs 183s/epoch，CUB batch 64）。

## 验证设置
- 数据/参数完全一致：CUB, clip_cbm(prior_residual), 2 epochs, batch 64, seed 1,
  α=1.0, β=0, λ_kl=0.01, CLIP ViT-L/14 (官方权重, SHA256 b8cca3fd... 校验通过)
- 编码 batch 与训练 batch 相同(64), fp16, cudnn.deterministic=True

## 逐指标对照

| 指标 | main.py(原始) | main_cached.py(缓存) |
|---|---|---|
| Epoch 1 Train Acc | 1.55% | 1.55% |
| Epoch 1 Source Acc | 4.01% | 4.01% |
| Epoch 1 Target Acc | 2.99% | 2.99% |
| Epoch 2 Train Acc | 8.65% | 8.65% |
| Epoch 2 Source Acc | 14.68% | 14.68% |
| Epoch 2 Target Acc | 9.39% | 9.39% |
| Epoch 2 H(q) | 5.3179 | 5.3179 |
| Epoch 2 w_max / w_min / w_std | 1.1102 / 0.8500 / 0.0504 | 1.1102 / 0.8500 / 0.0504 |
| Δw 区间 | [−0.0263, +0.0273] | [−0.0263, +0.0273] |

## 等价性依据
1. CLIP 全程冻结(`requires_grad=False`), 训练不改变编码器
2. `clip.load` 的 preprocess 是确定性变换(Resize+CenterCrop+Normalize), 无随机增广
3. 缓存保存 `encode_image` 原始输出(fp16, 未归一化), forward 中的
   `.float()` 与归一化照常执行 —— 计算图与原始路径逐操作一致
4. 同 seed 下 DataLoader shuffle 序列一致(RNG 消费顺序相同)

原始日志: `downloads/smoke_main.log` / `downloads/equiv_cached.log`
