# HD1910 HOME 校正训练

## 无电流输入的两种训练目标（2026-10-31 更新）

旧的电流输入训练已停止，最后保存 `model_1200.pt`。以下两个任务从头训练，均不读取
实时电流或实时受力作为 actor 输入；训练奖励仍可使用仿真特权信息。

| 目标 | 任务 | Actor 输入 | 主要奖励 |
| --- | --- | --- | --- |
| 理想 HOME 位置 | `Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910` | 34×8=272 | 真实关节角接近 HOME |
| 理想 HOME 受力比例 | `Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910` | 48×8=384 | 关节驱动力矩大小比例接近参考向量 |

34 维基础帧依次为编码器角减 HOME（14）、重力方向（3）、陀螺仪（3）、实际已下发的
HOME 补偿（14）。第二种额外加入固定的 14 维目标比例。目标比例是任务参数，
不是当前实测力；此次只训练一个参考目标，不代表可任意替换目标后泛化。
输出都为 14 个增量，20 Hz、每步最多 0.25°、累计最多 ±8°、前 0.5 秒保持；
每回合 10 秒、随机零偏 ±3°，10% 回合显式使用全零偏差。

```bash
cd /home/luckysir/microduck/microduck_rl_hd1910
../microduck_rl/.venv/bin/python scripts/train_home_calibration.py \
  --no-current --objective pose --phase train --num-envs 1024 --iterations 3000

../microduck_rl/.venv/bin/python scripts/train_home_calibration.py \
  --objective force-ratio --phase train --num-envs 1024 --iterations 3000
```

每个入口都先通过名义 HOME 物理稳定性检查、64 环境/5 轮 PPO 冒烟训练和
包含归一化的 ONNX 一致性检查，再开始正式训练；不会加载旧的 392 维权重。
两个任务分别保存在 `logs/rsl_rl/microduck_hd1910_home_calibration_no_current/`
和 `logs/rsl_rl/microduck_hd1910_home_force_ratio_no_current/`。

本机查看：两种训练曲线共用 `http://127.0.0.1:6006`，位置目标仿真在
`http://127.0.0.1:8080`，受力比例目标仿真在 `http://127.0.0.1:8081`。
仿真页面的 Checkpoints → Use Latest 用于加载该任务新保存的模型。
这是仿真效果预览；训练中导出的早期模型不代表校正已成功。

第二种参考由 `scripts/capture_home_force_reference.py` 在名义仿真中采集：
用真实角度反馈把关节稳定在理想 HOME 附近后采样，先对每个关节的
`qfrc_actuator` 取绝对值并求均值，再归一化。通过姿态、双支撑、有限总力矩等
检查后才使用 `tasks/home_force_reference.json`。这是关节驱动力矩比例，
不是足底力，也不声称是全部外部关节载荷。训练的比例奖励保留支撑和非零载荷约束。

训练与部署以严格版本合同区分 272、384 和旧 392 维模型。真机仍读取电流做过流保护，
但两种新模型都无需电流符号/比例映射；其 actor 输入只来自已有编码器、IMU 和命令。
必须区分接口验证与效果验证：这些信号不能保证唯一恢复所有关节零偏，比例一致也
不保证姿态一致。最终需比较同一测试条件下的模型与零动作、真实姿态误差、比例误差
及站立失败率。下面保留旧电流方案的设计记录，不能用于新模型的输入维度。

## 旧电流方案记录（已停止）

任务：`Mjlab-HomeCalibration-Flat-MicroDuck-HD1910`。

使用本仓库现有 MuJoCo/mjlab、HD1910 BAM 和 PPO 环境，从随机权重开始训练。
训练目标是实际机械关节角接近 `HOME_FRAME`，机器人双脚落地自主承重。
真实角度、零偏和接触力只进入奖励或 Critic；Actor 从始至终只接收模拟的真机传感器与已知命令。

## 启动

在这台训练电脑复用已有 Python 3.12 环境，不需要重新安装 CUDA 或依赖：

```bash
cd /home/luckysir/microduck/microduck_rl_hd1910
../microduck_rl/.venv/bin/python scripts/train_home_calibration.py \
  --phase train --num-envs 1024 --iterations 3000
```

入口依次执行：实际 CUDA 运算检查 → 4 环境物理预检 → 64 环境、5 轮 PPO
短训练 → 归一化 ONNX 数值一致性检查 → 从随机权重开始正式训练。
任何前置检查失败都会停止。正式训练完成后，再做导出检查和 nominal/train/stress
三个分布各 32 环境、8 秒的独立评估。短训练权重不用于初始化正式训练。

只运行前置验证：

```bash
../microduck_rl/.venv/bin/python scripts/train_home_calibration.py --phase smoke
```

CPU 只用于验证代码流程，入口不允许用它启动正式长训练：

```bash
../microduck_rl/.venv/bin/python scripts/train_home_calibration.py --phase smoke --device cpu
```

2026-10-31 首次设置时出现的驱动 `cuInit=999` 故障已在用户重启后恢复。
随后修复了入口传递 `--gpu-ids 0` 导致的 Tyro 列表解析错误：CUDA 使用 mjlab
默认的 `[0]`，CPU 短训练显式传递 `--gpu-ids None`。上面的启动命令无需改变。

已完成的 CPU 验证：58 项配置/动作/观测/导出契约测试；4 环境 3 秒 HOME 保持，
最大躯干倾角 2.65°；64 环境、5 轮 PPO 短训练，未出现 `nan_state`；128 组
checkpoint/ONNX 对照，最大输出差 `2.38e-7`。

重启后已通过 GPU 上的物理预检、64 环境 5 轮 PPO 短训练及 ONNX 数值对照
（128 组输入，最大输出差 `5.14e-7`）；另完成 1024 环境 2 轮容量检查，
没有显存不足或 `nan_state`。新增 3 项启动器回归测试通过，使用实际安装的
Tyro/TrainConfig 验证 CPU 和 CUDA 参数解析。GPU 短训练报告在
`logs/home_calibration_checks/20261031_220413_813591/`。
这些结果验证训练流程，不代表校正策略已收敛；3000 轮正式训练尚未启动。

## 默认配置

| 项目 | 设置 |
|---|---|
| 模型 | 当前 Microduck walk 几何、14 个身体关节、HD1910 公共 BAM M6 |
| 理想位置 | 当前 `HOME_FRAME` / STAND2，保持原模型参考角度 |
| 零偏 | 每个关节独立均匀采样 ±3°，一回合内固定 |
| 初始角度 | `真实角 = HOME − 零偏 + 小扰动`，上报角度接近 HOME |
| 位置观测 | `上报角 = 真实角 + 零偏`，4096 刻度量化 |
| 物理目标 | `HOME + 累计命令修正 − 零偏` |
| 控制频率 | 20 Hz；物理步长 5 ms，每控制步 10 个物理步 |
| 动作 | 14 维归一化增量，逐步最多 ±0.25°，累计最多 ±8° |
| 稳定等待 | 每次重置后 0.5 秒保持当前 HOME 命令，忽略策略调整 |
| 回合 | 10 秒；倾角超过 25°、躯干低于 0.085 m 或非有限状态时结束 |
| Actor | 49 维一帧 × 8 帧 = 392 维，按时间从旧到新排列 |
| Critic | 当前传感器帧 + 真实角度误差、零偏、速度、重力、足部接触力 |
| PPO | 32 步/轮、网络 256/128/64、观测归一化、种子 42 |
| 正式训练 | 默认 1024 环境、3000 轮，每 100 轮保存 |

Actor 的一帧依次为：

1. 上报关节角减 HOME：14 维，rad。
2. 带符号电流除以 `max(所有关节电流绝对值之和, 0.02 A)`：14 维。
3. 所有关节电流绝对值之和：1 维，A。
4. IMU 重力方向：3 维。
5. IMU 角速度：3 维，rad/s。
6. 已知的累计命令修正：14 维，rad。

重置后用第一帧填满历史；同一控制步 Actor/Critic 使用同一次噪声采样。
Critic 使用真实角度不会改变 Actor 的输入，也不会把未知零偏通过初始编码器读数直接暴露给 Actor。
既有 61 维行走任务、模型权重与部署配置不变；新任务采用独立的校正策略接口。

## 物理与测量假设

电流输入暂由 BAM 的电机力矩除以 `kt` 生成，方向对应模型关节的正方向，
再叠加每颗舵机的增益、偏置、噪声和量化。它是**仿真的电流代理量**，
不是已经验证过的 HD1910 `load` 寄存器或实测相电流模型。
6.5 mA 量化步长参考当前驱动实现，不能据此宣称已完成硬件单位校验。

训练同时随机化质量与惯量、质心、足底摩擦、执行器响应、供电、延迟和 IMU 误差。
这些范围是初版实验设置，具体数值见配置文件；它们不是这台机器的辨识结果。
`nominal` 去掉物理/测量扰动但保留 ±3° 零偏；`stress` 扩大干扰，零偏范围仍为 ±3°。

奖励包括真实 HOME 跟踪、真实角度误差、躯干倾斜、动作/运动成本、双脚接触和失败成本。
不把受力比例匹配当作训练成功条件。对于弱可观测的偏航等关节，仍需通过评估判断
受力、IMU 和短时历史是否足够，不能用训练正常结束代替校正精度验收。

累计动作还可能补偿重力静差、摩擦和机构响应。因此它是 HOME 位置命令补偿候选，
不能未经独立几何验证就整体写入永久软件零位。

## 输出和检查

正式结果：`logs/rsl_rl/microduck_hd1910_home_calibration/<时间与名称>/`。
短训练：`logs/rsl_rl/microduck_hd1910_home_calibration_smoke/`。
前置检查报告：`logs/home_calibration_checks/<时间>/`。

- `model_*.pt`：Actor、Critic、优化器和归一化统计。
- `home_calibration.onnx`：最近保存的 Actor，已经包含观测归一化。
- `home_calibration_contract.json`：关节顺序、帧顺序、量纲、控制周期和动作后处理。
- `params/env.yaml`、`params/agent.yaml`：本次训练的配置快照。
- `evaluation_{nominal,train,stress}.json`：正式训练完成后的真实角度和首次回合评估。

TensorBoard 中查看 `Home/joint_mae_deg`、`Home/joint_rmse_deg`、`Home/joint_max_deg`
及 `Episode_Reward/true_home_tracking`。这些训练期数值含探索动作；最终应看独立评估，
比较零动作基线与策略的逐关节误差和存活率。首次失败后冻结该环境的统计，
避免自动重置掩盖摔倒。初步预检只检查 HOME 站立稳定性，不保证精确几何 HOME 已可达。

```bash
../microduck_rl/.venv/bin/tensorboard \
  --logdir logs/rsl_rl/microduck_hd1910_home_calibration --host 127.0.0.1
```

单独评估某个 checkpoint（替换实际路径）：

```bash
PYTHONPATH="$PWD/src" ../microduck_rl/.venv/bin/python scripts/check_home_calibration.py \
  --checkpoint logs/rsl_rl/microduck_hd1910_home_calibration/<运行目录>/model_2999.pt \
  --profile train --num-envs 32 --seconds 8 --output home_evaluation.json
```

标准导出器同样支持该任务，读取 checkpoint 时会选择 HOME 专用元数据：

```bash
PYTHONPATH="$PWD/src" ../microduck_rl/.venv/bin/python scripts/export.py \
  Mjlab-HomeCalibration-Flat-MicroDuck-HD1910 \
  --checkpoint-file <checkpoint绝对路径> --num-envs 1 --onnx-file home_calibration.onnx
```

继续同一个 HOME 训练（不要加载 61D 行走 checkpoint）：

```bash
PYTHONPATH="$PWD/src" ../microduck_rl/.venv/bin/python -m mjlab_microduck.train_cli \
  Mjlab-HomeCalibration-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 1024 --agent.resume True \
  --agent.load-run '<原运行目录名>' --agent.load-checkpoint '<checkpoint文件名>' \
  --agent.max-iterations 1000
```

`max-iterations` 是继续执行的轮数。改变 Actor 历史长度或网络结构后不能直接恢复旧权重。
ONNX 的 14 个输出是增量控制信号，必须执行契约中的累计与限幅，不能作为现有行走策略的
绝对位置偏移直接发送。当前工作范围是训练与离线评估，未接入真机自动标定/零位写回。
