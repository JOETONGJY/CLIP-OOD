@echo off
chcp 65001 >nul
title LanCE 待补实验管线(断点续跑: tau扫描 + 双头消融跨数据集)
cd /d F:\CLIP-OOD
D:\anaconda3\envs\trl4060\python.exe -u run_pending.py
echo.
echo ===== 管线已结束, 窗口可关闭 =====
pause
