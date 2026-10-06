# 修复记录：双头 DDO 正则"先求和后取绝对值"实现偏差（2026-09-11）

## 问题
`model/cbm_models.py` `clip_cbm_subspace.forward` 中双头 DDO 正则原实现：

```python
regularizer = self.classifier[1:](self.domain_concept_projection)
if self.inv_classifier is not None:
    reg_inv = self.inv_classifier[1:](self.domain_concept_projection)
    regularizer = regularizer + self.eta_ddo_inv * reg_inv   # ← 先求和
# main.py: orth_loss = torch.abs(reg_loss).mean()            # ← 后取 abs
```

等效于 `|h_o(proj) + η·h_inv(proj)|`，而设计公式（交接文档 §5.5 与代码注释）为
`‖h_o[1:](proj)‖ + η·‖h_inv[1:](proj)‖`（逐头范数再求和）。

**后果**：两头域响应可符号相反对消——取 `h_o = −h_inv` 时正则损失为 0，
但两个头都没有被真正正交约束，DDO 对双头模式形同虚设。

## 定位证据
CUB 消融（修复前，3 seeds）中掉分组恰为全部双头模式：
- headonly（双头、无投影）56.14→55.66（−0.48）
- full（双头+投影）56.14→55.11（−1.03）
- c3+full 56.14→55.16（−0.98）
而单头模式不受影响（softproj 56.12≈DDO）。

## 修复
逐头取绝对值后再求和（单头路径数学不变；main.py 的外层 abs 对非负和无影响）：

```python
regularizer = torch.abs(self.classifier[1:](self.domain_concept_projection))
if self.inv_classifier is not None:
    reg_inv = torch.abs(self.inv_classifier[1:](self.domain_concept_projection))
    regularizer = regularizer + self.eta_ddo_inv * reg_inv
```

## 影响范围与处置
- 受影响：ch4_mode ∈ {headonly, full} 及 C3+C4 组合
- 不受影响：ddo / softproj / hardproj / 第三章全部（clip_cbm_orth 未改动）
- 修复前结果完整备份：本目录 `results_preddofix/`（论文可作"问题定位→修复"证据链）
- 修复后重跑：headonly / full / c3+full × 3 seeds（结果见 RESULTS.md §5）
