# Microduck 本地工作区

**2026-09-28：四版模型均已完成训练、ONNX 导出和交叉评估。**
HD-1910 队列于 06:35 结束。当前优先候选为 HD-1910 平地训练版；低速、后退、横移和直行质量仍需改进。
详见 [四版训练验收与模型/视频](reports/四版训练验收.md)。

- [6 个仓库的作用及关系](reports/项目分析.md)
- [仓库版本清单](reports/repositories.json)
- [训练前资源检查](reports/preflight_before_training.json)
- [2048 环境性能测试](reports/benchmark.json)
- [已安装的依赖版本](reports/installed_requirements.txt)
- [正式训练启动信息](runs/active_run.json)
- [训练总日志](runs/train_supervisor.log)
- [250 轮检查点的早期评估与视频](reports/250轮初步评估.md)
- [HD-1910 / Radxa 直连 IMU 部署评估](reports/HD1910部署评估.md)
- [原版速度跟踪诊断与官方模型对照](reports/速度跟踪诊断.md)

流水线已生成 `reports/训练结果.md`、`reports/HD1910训练结果.md` 和相应运行目录中的 `training_curves.png`。
生成过程记录在 `runs/final_report.log`；实际训练是否完成以状态文件为准。

## HD-1910 接续训练（2026-09-28）

用户已授权前两版 XL330 任务结束后，继续训练 HD-1910 平地和齿隙两版。
代码位于隔离工作树 `microduck_rl_hd1910/`，复用 `microduck_rl/.venv`，
通过 `PYTHONPATH` 选择源码，不重复安装 CUDA 或修改原版训练源码。

- 两版独立随机初始化：每版 4000 轮、2048 环境、种子 42，每 250 轮保存。
- 使用公开 1910 BAM M6 参数、P=5、7.4–8.0V 电压随机范围；齿隙假设 ±1°。
  这些不是用户机器的实测参数，整机质量惯量和运行时仍需校准。
- 队列先等待原版两版的训练、导出、评估全部完成并退出，然后运行 HD-1910
  GPU 集成/站姿检查、64 环境×5 轮短训练及导出评估、2048 环境×10 轮容量验证。
  全部通过才进入正式训练。失败会停止并记录原因。
- 启动前要求至少 3 GiB 空间，训练/导出/评估中低于 2.5 GiB 会停止当前阶段。
- HD-1910 正式模型会在普通和齿隙环境各测试一次，包含 0.15 和 0.30 m/s 前进，
  以及站立、后退、侧移、转向，保留视频和 ONNX 数值一致性结果。

查看进度仍使用 `python3 scripts/status.py`，它也显示 HD-1910 队列。
队列状态为 `runs/hd1910_queue.json`，总日志为 `runs/hd1910_queue.log`。
正式产物为 `runs/<排队时间>_hd1910_train/`，结果报告为 `reports/HD1910训练结果.md`。
进程可在退出终端后继续；电脑需保持开机，重启不会自动恢复进程。

源码与验证说明：[HD-1910 训练说明](microduck_rl_hd1910/docs/hd1910-training.md)。
队列运行期间不要修改其训练源码和执行脚本；队列会核对源码指纹并在变化后停止。

已完成的 XL330 平地模型在 0.15 m/s 前进评估中基本原地站立。
后续上游 CPU 推理路径对照发现：本次及官方历史 walking 模型在 0.3/0.4 m/s 指令下均可前进，
但低速响应弱、实际速度偏低。因此不能将低速测试推广为整版不会走；详见速度跟踪诊断。
训练轮数、回报和存活率不能替代速度跟踪及视频验收。

## 本次训练

使用官方 XL330 BAM M6 模型，从随机初始化开始，分别训练：

| 任务 | 平地 | 每个舵机额外机械齿隙 | 迭代 / 并行环境 / 种子 |
|---|---|---|---|
| `Mjlab-Velocity-Flat-MicroDuck` | 是 | 无 | 4000 / 2048 / 42 |
| `Mjlab-Velocity-Flat-Backlash-MicroDuck` | 是 | ±1°，总计 2° | 4000 / 2048 / 42 |

每轮每环境采样 24 个控制步，每版共 196,608,000 个环境步。
训练使用上游默认奖励、课程与随机化，50 Hz，61 维观测、14 维动作。
这是 XL330 仿真基线。用户已确认实机使用 HD-1910、Radxa Zero 3W、HAT 和直连的
LSM6DSV16XTR；当前任务尚未切换为 HD-1910，不应将其产物视为已适配实机的行走策略。
需要的训练与运行时改动见上方部署评估。

正式运行按顺序训练两版；每版完成后通过官方脚本导出包含归一化的 ONNX，
再在普通/齿隙环境中各评估一次。每种环境包含站立、前进、后退、侧移、
转向 5 组命令，每组 64 个环境、10 秒。评估关闭外部推力并固定头部/身体命令，
保留启动/重置随机化；结果区分首回合存活率和失败前速度误差，避免用自动重置掩盖摔倒。

`runs/*_train/status.json` 中的 `completed` 代表流水线成功结束；
步态质量应查看 `evaluation.json` 和 `forward.mp4`，不能只凭训练退出码判断。
短训练检查点在 `local_*_smoke/`，仅用于链路测试。

## 查看进度

```bash
cd /home/luckysir/microduck
python3 scripts/status.py
tail -f runs/train_supervisor.log
```

每版训练的详细日志路径会打印在上述状态中。
也可以启动本地 TensorBoard：

```bash
microduck_rl/.venv/bin/tensorboard --logdir microduck_rl/logs/rsl_rl --host 127.0.0.1
```

## 文件位置

- 模型检查点：`microduck_rl/logs/rsl_rl/local_{flat,backlash}_train/<时间和运行名>/model_*.pt`
- 完整训练配置：同目录 `params/env.yaml`、`params/agent.yaml`
- 正式产物：`runs/<启动时间>_train/{flat,backlash}/policy.onnx`
- 交叉评估：上述产物目录内 `eval_{flat,backlash}/evaluation.json`
- 同任务前进视频：对应 `eval_*/forward.mp4`

## 复现

已有训练运行时不要重复启动，以免争抢显存。

```bash
# 空间检查：最低保留 3 GiB
python3 scripts/preflight.py

# 两版都先做小规模检查
python3 scripts/train_pair.py --phase smoke --num-envs 64 --iterations 5

# 正式训练 + 标准导出 + 交叉评估
python3 scripts/train_pair.py --phase train --num-envs 2048 --iterations 4000 --finalize
```

首轮依赖使用 `microduck_rl/uv.lock` 锁定安装。环境位于 `microduck_rl/.venv`，
已复用本机其他环境中版本一致的 CUDA 二进制硬链接，节省约 4.08 GiB。
不需要激活或修改原 Conda 环境。普通命令可直接使用 `.venv/bin/python` 或 `.venv/bin/train`。

### 从检查点继续

在 `microduck_rl/` 下执行（将目录名和检查点换成实际路径；选择对应 task / experiment）：

```bash
.venv/bin/train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.logger tensorboard \
  --agent.experiment-name local_flat_train --agent.run-name continued \
  --agent.resume True --agent.load-run '<原运行目录名>' \
  --agent.load-checkpoint model_1000.pt --agent.max-iterations 1000
```

此处 `max-iterations` 是继续执行的轮数，加载会恢复课程计数器。
跨任务加载不等于同任务续训，本次两版均独立从头训练。

## 已完成的启动检查

- PyTorch 2.9.1 / CUDA 12.8 GPU 运算通过；MuJoCo 3.10.0、mjlab 1.3.0、Warp 1.12.0。
- 上游测试 224 项通过、1 项跳过；另 1 项 PATH 测试修正启动环境后复验通过。
- 两版各 64 环境 × 5 轮短训练成功，`nan_state` 指标均为 0。
- 两版 ONNX 均通过结构检查和 CPU 推理，接口 `[1,61] → [1,14]`。
- 评估脚本在短训练模型上正确记录摔倒；ONNX 与检查点推理对比通过。

6 个仓库都是浅克隆（当前工作文件完整）。CAD Releases 的大型 SolidWorks ZIP
不包含在 Git 克隆内；下载入口见 CAD 仓库 README。截图中的 `HLS3915.zip` 未提供实际文件。
