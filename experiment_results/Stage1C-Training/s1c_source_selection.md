# Stage 1C 八臂训练：源域验证与强度选择（目标域评估之前）

预注册见 `RESULTS.md` §3z。57 个 LD run 与 P1 的 21 个 run 一样以 `--target_eval none` 训练（train_core 4790 / heldout_val 1200；训练期间不读目标域特征缓存），结果 json 中没有任何目标域数字（脚本已逐个断言）。

- 选择规则 = P1 冻结的规则：**heldout_val 7 种光度扰动的均值 Acc，3 seeds 均值最大，并列取较小强度**。P1 只验证过 DDO 剂量选择，LD 强度选择是同一规则的第一次新验证。
- 强度锚点：统一批次标定 `stage1c_calibration_traincore.json`（sha256 `75a49e9fc881fa98`）。
- P1 基线复用的配置一致性核对：通过（CE-Only / DDO α=1 / α=2 与 LD run 在两个配置共有的全部键上逐项一致）。

## 扫描臂（锚点 × {0.5, 1, 2}）

| 臂 | 椭球 | 锚点 | 强度 | 源域增强 Acc（主指标，3 seeds 均值） | 逐 seed | 干净 Acc | 扰动均值 CE |
|---|---|---|---|---|---|---|---|
| Language-Full | lang_full | 5.3459（eps_pen (train_core 固定批次)） | 0.5× | **50.18** | 49.92 / 50.49 / 50.14 | 75.97 | 2.3795 |
| Language-Full | lang_full | 5.3459（eps_pen (train_core 固定批次)） | 1× | **51.08** | 50.96 / 51.19 / 51.07 | 76.78 | 2.3455 |
| Language-Full | lang_full | 5.3459（eps_pen (train_core 固定批次)） | 2× | **52.29** **← 选中** | 52.00 / 52.44 / 52.42 | 77.28 | 2.3153 |
| Random-Full | rand_full | 5.3395（eps_pen (train_core 固定批次)） | 0.5× | **48.64** **← 选中** | 48.46 / 48.67 / 48.79 | 75.00 | 2.4523 |
| Random-Full | rand_full | 5.3395（eps_pen (train_core 固定批次)） | 1× | **48.61** | 48.40 / 48.67 / 48.76 | 74.61 | 2.4574 |
| Random-Full | rand_full | 5.3395（eps_pen (train_core 固定批次)） | 2× | **48.44** | 48.24 / 48.52 / 48.56 | 74.56 | 2.4678 |
| Language-32 | lang32 | 5.2849（eps_pen (train_core 固定批次)） | 0.5× | **50.21** | 50.00 / 50.50 / 50.12 | 75.94 | 2.3780 |
| Language-32 | lang32 | 5.2849（eps_pen (train_core 固定批次)） | 1× | **51.13** | 50.94 / 51.33 / 51.11 | 76.69 | 2.3424 |
| Language-32 | lang32 | 5.2849（eps_pen (train_core 固定批次)） | 2× | **52.43** **← 选中** | 52.13 / 52.68 / 52.49 | 77.47 | 2.3094 |
| Identity-H | identity | 8.9724（eps_pen (train_core 固定批次)） | 0.5× | **48.81** | 48.46 / 48.96 / 49.01 | 74.97 | 2.4469 |
| Identity-H | identity | 8.9724（eps_pen (train_core 固定批次)） | 1× | **48.83** **← 选中** | 48.57 / 48.87 / 49.05 | 74.86 | 2.4515 |
| Identity-H | identity | 8.9724（eps_pen (train_core 固定批次)） | 2× | **48.49** | 48.31 / 48.43 / 48.74 | 74.47 | 2.4788 |
| Language-ISDA | isda | 123.4769（lam_pen (train_core 固定批次)） | 0.5× | **51.14** | 50.99 / 51.29 / 51.15 | 76.75 | 2.3511 |
| Language-ISDA | isda | 123.4769（lam_pen (train_core 固定批次)） | 1× | **52.33** | 52.15 / 52.45 / 52.39 | 77.50 | 2.3227 |
| Language-ISDA | isda | 123.4769（lam_pen (train_core 固定批次)） | 2× | **53.60** **← 选中** | 53.32 / 53.79 / 53.70 | 77.81 | 2.3296 |

## 固定强度对照臂（不扫描）

| 臂 | 源域增强 Acc（3 seeds 均值） | 逐 seed |
|---|---|---|
| Identity-H 梯度配平对照（ε_grad） | 47.22 | 47.17 / 46.99 / 47.51 |
| Language-ISDA 梯度配平对照（λ_grad） | 51.87 | 51.68 / 51.99 / 51.95 |
| Random-Full 同半径对照（ε 与 Language-Full ×1 逐位相同） | 48.61 | 48.40 / 48.67 / 48.76 |
| Language-PGD 锚点（ε 与 Language-Full ×1 逐位相同，PGD-5×1） | 50.40 | 50.12 / 50.57 / 50.51 |

## 各扫描臂的选择结果

- **Language-Full：强度 = 2× 锚点**（ε/λ = 10.6917）；逐 seed 的选择：['2', '2', '2']；辅助指标：干净 Acc → 2×，扰动均值 CE → 2×
- **Random-Full：强度 = 0.5× 锚点**（ε/λ = 2.6698）；逐 seed 的选择：['0.5', '0.5', '0.5']；辅助指标：干净 Acc → 0.5×，扰动均值 CE → 0.5×
- **Language-32：强度 = 2× 锚点**（ε/λ = 10.5698）；逐 seed 的选择：['2', '2', '2']；辅助指标：干净 Acc → 2×，扰动均值 CE → 2×
- **Identity-H：强度 = 1× 锚点**（ε/λ = 8.9724）；逐 seed 的选择：['1', '0.5', '1']；辅助指标：干净 Acc → 0.5×，扰动均值 CE → 0.5×
- **Language-ISDA：强度 = 2× 锚点**（ε/λ = 246.9538）；逐 seed 的选择：['2', '2', '2']；辅助指标：干净 Acc → 2×，扰动均值 CE → 1×

## 平均训练成本（秒 / run，含最终评估；PGD 的搜索开销单独可见）

- Identity-H 梯度配平对照（ε_grad）：86 / 54 / 36
- Language-ISDA 梯度配平对照（λ_grad）：36 / 36 / 36
- Random-Full 同半径对照（ε 与 Language-Full ×1 逐位相同）：36 / 36 / 36
- Language-PGD 锚点（ε 与 Language-Full ×1 逐位相同，PGD-5×1）：43 / 42 / 42
- 扫描臂的逐 run 耗时见结果 json 的 elapsed_seconds