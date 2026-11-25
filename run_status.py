"""实验完成状态、运行锁与代码版本 (2026-11-25, 外审建议的最小工程修复)。

不依赖 torch / wandb, 可被训练入口与流水线脚本 (run_pending.py / run_pipeline.py / run_matrix.py) 直接 import。
运行锁的持有者检测优先用 psutil (可识别 PID 复用), 没有安装时退化为只看 PID 是否存在。
"""
import json
import os
import subprocess
import time


def load_result(path):
    """读取结果 json; 不存在或不可解析时返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _same(got, want):
    """配置项比对: 相等, 或两边都能读成数值且数值相等 (命令行里的 '0' 与 json 里的 0.0 是同一个值)"""
    if got == want:
        return True
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return str(got) == str(want)


def result_is_complete(path, expected_config=None):
    """结果是否代表一次完整且配置匹配的训练。

    新格式 json (2026-11-25 起带 status 字段): 要求 status == 'completed'、epochs 是有效正整数、且
        completed_epochs == run_config.epochs (0 == 0 / None == None 不算完成)。
    旧格式 json (无 status 字段): 目标准确率 > 0 视为完成 (旧的失败 run 会留下 0.0 的结果)。
    expected_config: 可选 dict, 逐项与 json 的 run_config (或顶层同名字段) 比对,
        任一项不一致即视为不是所要的实验, 返回 False。
    """
    r = load_result(path)
    if not isinstance(r, dict):
        return False
    cfg = r.get("run_config") or {}
    if "status" in r:
        if r.get("status") != "completed":
            return False
        if not _is_pos_int(cfg.get("epochs")) or r.get("completed_epochs") != cfg.get("epochs"):
            return False
    else:
        if not (float(r.get("target_acc") or 0.0) > 0.0):
            return False
    if expected_config:
        for k, want in expected_config.items():
            if not _same(cfg.get(k, r.get(k)), want):
                return False
    return True


def _pid_exists(pid):
    """不用 psutil 时判断进程是否存在 (Windows: OpenProcess; 其他: kill(pid, 0))"""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k32.OpenProcess(0x00100000, False, int(pid))          # SYNCHRONIZE
        if not h:
            return ctypes.get_last_error() == 5                   # ACCESS_DENIED: 进程存在但无权打开
        k32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _holder_running(info):
    """锁的持有者是否仍在运行: 有 psutil 时同时核对进程创建时间 (PID 被别的进程复用不算)"""
    pid = info.get("pid")
    if not _is_pos_int(pid):
        return False
    try:
        import psutil
    except ImportError:
        return _pid_exists(pid)
    try:
        return (psutil.pid_exists(pid)
                and psutil.Process(pid).create_time() <= float(info.get("start_time") or 0) + 1.0)
    except psutil.Error:
        return False


class RunLock:
    """运行锁 (2026-11-25, 外审要求: 防止两个进程同时跑同一个 run / 同一个实验矩阵而互相覆盖产物)。

    锁文件以 O_CREAT|O_EXCL 独占创建, 内容为 {pid, start_time, time, what}。已有锁且持有者仍在运行、
    且最近 stale_seconds 秒内有过心跳时拒绝启动 (RuntimeError); 否则视为残留锁并接管
    (先原子改名再删除, 两个进程同时接管时只有一个成功)。长任务应在每个阶段调用 heartbeat()。
    用法: with RunLock(path, what="run_name=...") as lock: ...
    """

    def __init__(self, path, stale_seconds=1800, what=None):
        self.path, self.stale_seconds, self.what = path, stale_seconds, what
        self._start = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        info = {"pid": os.getpid(), "start_time": time.time(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "what": self.what}
        while True:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(info, f)
                self._start = info["start_time"]
                return self
            except FileExistsError:
                old = load_result(self.path) or {}
                age = time.time() - os.path.getmtime(self.path)
                if _holder_running(old) and age < self.stale_seconds:
                    raise RuntimeError(
                        f"拒绝启动: {self.what or self.path} 正在被另一个进程运行 "
                        f"(pid={old.get('pid')}, 启动于 {old.get('time')}; 锁文件 {self.path})")
                stale = f"{self.path}.stale.{os.getpid()}"
                try:
                    os.replace(self.path, stale)          # 原子接管; 别的进程刚接管则这里落空
                except FileNotFoundError:
                    pass
                else:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass

    def heartbeat(self):
        os.utime(self.path, None)

    def __exit__(self, *exc):
        cur = load_result(self.path) or {}
        if cur.get("pid") == os.getpid() and cur.get("start_time") == self._start:
            # 只删除自己的锁 (若因长时间无心跳被接管, 文件已属于别的进程)
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
        return False


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
