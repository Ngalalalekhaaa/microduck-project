# Microduck 训练结果

报告生成时间：2026-09-27T20:23:24.361750+08:00

流水线状态：**completed**。

使用官方 XL330 模型，两版独立随机初始化；每版目标 10 轮、2048 个环境，种子 42。
齿隙版为每个驱动关节增加 ±1° 的被动机械齿隙。

评估每条命令 64 个环境、10 秒，固定头部和身体命令为零，关闭外部推力，
保留上游启动/重置随机化。存活率按首回合统计；速度误差只统计第 1 秒后、
首次失败前的样本，需结合存活率一起看。本报告不代表实机验证。

| 训练模型 | 测试环境 | 命令 vx,vy,wz | 10 秒存活率 | 实际平均 vx,vy,wz | 速度绝对误差 vx,vy,wz |
|---|---|---|---:|---|---|

## 模型与原始证据

### flat

- 任务：`Mjlab-Velocity-Flat-MicroDuck`
- 状态：completed
- 训练日志：[flat.log](/home/luckysir/microduck/runs/20260927_201530_benchmark/flat.log)

### backlash

- 任务：`Mjlab-Velocity-Flat-Backlash-MicroDuck`
- 状态：completed
- 训练日志：[backlash.log](/home/luckysir/microduck/runs/20260927_201530_benchmark/backlash.log)

## 训练曲线

![训练曲线](/home/luckysir/microduck/runs/20260927_201530_benchmark/training_curves.png)

当前产物是仿真训练结果。HLS3915 参数文件尚未提供，不能把本次 XL330 策略视为 HLS3915 适配版本。
