# 描述符库现状审查（CUB；2026-11-26）

由 `descriptor_bank_audit.py` 生成（代码版本 `7ded084`）。只读审查，不训练，不生成新词库。

## A. 训练实际使用的词库

- `data/__init__.py` 经由 `descriptor_bank.load_descriptor_bank` 取模板：True；`--descriptor_bank` 的默认值：`original`（Original-204）。

## B. prompts/ 下的词库文件

| 文件 | 源模板数 | 目标模板数 | 校验 | 词库身份 (sha256 前 16 位) |
|---|---|---|---|---|
| `prompts/prompt.py` | 1 | 130 | 通过 | 2d7ac19b6c82a2a4 |
| `prompts/prompt200.py` | 1 | 210 | 拒绝: target_text_prompts[57] 重复出现: 'a quickdraw of a {}.' |  |
| `prompts/prompt200new.py` | 1 | 204 | 通过 | bc4ba710ad72ff0f |

- Original-204 的身份：源 `e8e367e769f86c27`，目标 `1f55c2a9855f277f`，词库 `bc4ba710ad72ff0f`；文件 sha256 `ce8d5e8114ed1b70`。

## C. CLIP 权重身份

- `ViT-L-14.pt`（932 MB）：sha256 前 16 位 `b8cca3fd41ae0c99`。进入新的缓存键。

## D. CUB + Original-204 的可靠性缓存

- 旧键 `3ea993d3a8ced9f7`；新键 `e3abc74fcd065d5f`（旧键的三个文本序列 + 概念名 312 个 + CLIP 身份 `b8cca3fd41ae0c99`）。
- 新键缓存文件：`reliability_e3abc74fcd065d5f.pt`；认领自旧键文件，数值与旧文件逐位相同。
- 缓存内的记录：`{"key": "e3abc74fcd065d5f", "legacy_key": "3ea993d3a8ced9f7", "bank_sha256": "bc4ba710ad72ff0fee814a9303d5fed1731ed02dcdecf1d7829f63f8f94c82de", "n_source": 1, "n_target": 204, "n_classes": 200, "class_names_sha256": "cb595aba63ffdf3eac258453361980ba474dd307f4d851e4d39df008e075af8a", "n_concepts": 312, "concept_names_sha256": "8a17d9f36606bc4e12039e7c125528d4f7522374724c9e02524fb92a5a172367", "clip_identity": "b8cca3fd41ae0c99", "clip_file": "ViT-L-14.pt", "created": "2026-11-26 12:18:15", "git": {"commit": "7ded084c73743b9c3f5cbac72765ed019aa6acaf", "dirty": true}, "legacy_adopted": true}`

## E. 一次重算核对

- 用现役 CLIP 权重重算 204 对模板的域方向与可靠性分数，与缓存的最大绝对差：域方向 0.00e+00，分数 0.00e+00；**逐位相同**。

## F. JSON 冻结格式

- 同样的文本写成 JSON 冻结格式后，词库身份与缓存键都与 Python 列表版本相同（文本才是身份，文件格式不是）。

## 结论

全部核对项通过：True
