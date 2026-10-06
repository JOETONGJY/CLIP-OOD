@echo off
chcp 65001 >nul
title LanCE 实验管线(断点续跑)
cd /d F:\CLIP-OOD
D:\anaconda3\envs\trl4060\python.exe -u run_pipeline.py
echo.
echo ===== 管线已结束, 窗口可关闭 =====
pause
