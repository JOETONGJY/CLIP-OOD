"""实验完成状态与代码版本 (2026-11-25, 外审建议的最小工程修复)。

不依赖 torch / wandb, 可被训练入口与流水线脚本 (run_pending.py / run_pipeline.py) 直接 import。
"""
import json
import os
import subprocess


def load_result(path):
    """读取结果 json; 不存在或不可解析时返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def result_is_complete(path, expected_config=None):
    """结果是否代表一次完整且配置匹配的训练。

    新格式 json (2026-11-25 起带 status 字段): 要求 status == 'completed' 且
        completed_epochs == run_config.epochs。
    旧格式 json (无 status 字段): 目标准确率 > 0 视为完成 (旧的失败 run 会留下 0.0 的结果)。
    expected_config: 可选 dict, 逐项与 json 的 run_config (或顶层同名字段) 比对,
        任一项不一致即视为不是所要的实验, 返回 False。
    """
    r = load_result(path)
    if not isinstance(r, dict):
        return False
    if "status" in r:
        cfg = r.get("run_config") or {}
        if r.get("status") != "completed":
            return False
        if r.get("completed_epochs") != cfg.get("epochs"):
            return False
    else:
        if not (float(r.get("target_acc") or 0.0) > 0.0):
            return False
    if expected_config:
        cfg = r.get("run_config") or {}
        for k, want in expected_config.items():
            got = cfg.get(k, r.get(k))
            if got != want and str(got) != str(want):
                return False
    return True


def git_info(cwd=None):
    """当前代码版本: {"commit": <40位哈希>, "dirty": <工作区是否有未提交改动>}; 取不到返回 None"""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd,
                                             stderr=subprocess.DEVNULL).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return None


def jsonable(v):
    """把 argparse 取值转成可写入 json 的基本类型 (torch.device 等一律转成字符串)"""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return v
    return str(v)
