# Microduck 训练与真机部署命令

这份命令表对应本工作区在 2026-10-01 的代码。**训练在带 NVIDIA GPU 的电脑上运行；真机部署在 Radxa Zero 3W 上运行。** 仓库中的部署模型用于复现实验，现有真机测试尚未证明机器人能持续前进。

## 目录和模型

| 目录 | 内容 | 运行位置 |
| --- | --- | --- |
| `microduck_rl/` | 原版 XL330 仿真任务 | 训练电脑 |
| `microduck_rl_hd1910/` | HD-1910 平地与假设齿隙任务 | 训练电脑 |
| `scripts/train_pair.py` | 两版顺序训练、导出 ONNX、交叉评估 | 训练电脑 |
| `deploy_hd1910/` | 本机训练的 HD-1910 平地和齿隙部署代码 | Radxa |
| `deploy_xgoduck_40000/` | 用户提供的 XgoDuck 40000 轮 ONNX 独立部署代码 | Radxa |
| `servo_web_hd1910/` | 舵机网页调试台 | Radxa |

`deploy_hd1910/models/hd1910_flat.onnx` 和 `hd1910_backlash.onnx` 是本机训练的两版；后者使用每关节 ±1° 的**仿真假设**。`deploy_xgoduck_40000/models/2026-09-24_20-26-30_xgoduck.onnx` 是用户另行训练的模型；本仓库没有它完整的训练配置和检查点，不能仅凭 ONNX 复现那 40000 轮。

## 一、电脑上训练

在原训练机执行；`microduck_rl/.venv` 是安装好的 CUDA/MuJoCo/mjlab 环境，**GitHub 上传副本没有包含虚拟环境**。首次在新电脑上训练，需按 [`microduck_rl/README.md`](microduck_rl/README.md) 安装依赖，并先核对 `nvidia-smi`。下面的 `train_pair.py` 会使用 `microduck_rl/.venv`，根据 `--robot` 选择对应源码。

```bash
cd /home/luckysir/microduck
python3 scripts/preflight.py
```

`preflight.py` 检查 GPU 与可用空间；默认至少需要保留 3 GiB。正式训练、导出、评估期间，`train_pair.py` 会在剩余空间低于 2.5 GiB 时停止当前阶段。每版 4000 轮、2048 个并行环境会消耗数小时及大量空间；不要同时启动多条 GPU 训练命令。

### 原版 XL330：平地与齿隙

```bash
cd /home/luckysir/microduck
python3 scripts/train_pair.py --robot xl330 --phase smoke --num-envs 64 --iterations 5 --finalize
python3 scripts/train_pair.py --robot xl330 --phase train --num-envs 2048 --iterations 4000 --finalize
```

### HD-1910：平地与齿隙

```bash
cd /home/luckysir/microduck
python3 scripts/train_pair.py --robot hd1910 --phase smoke --num-envs 64 --iterations 5 --finalize
python3 scripts/train_pair.py --robot hd1910 --phase train --num-envs 2048 --iterations 4000 --finalize
```

以上每条 `train_pair.py` 默认依次运行 `flat` 与 `backlash`。`--finalize` 会导出包含观测归一化的 ONNX，并在平地/齿隙环境中交叉评估；同任务评估还会保存视频。只训练其中一版时，加 `--variant flat` 或 `--variant backlash`，例如：

```bash
python3 scripts/train_pair.py --robot hd1910 --variant flat --phase train --num-envs 2048 --iterations 4000 --finalize
```

这些命令会**新开训练**，不会自动从已有检查点续训。新结果放在 `runs/<时间>_train/` 或 `runs/<时间>_hd1910_train/`。每次运行的 `status.json` 显示 `completed` 才表示训练、导出和评估流程都结束；行走效果还需查看 `evaluation.json` 与 `forward.mp4`。

### 查看结果、手动评估、续训

```bash
cd /home/luckysir/microduck
python3 scripts/status.py
```

已有 HD-1910 正式产物位于 `runs/20260928_001943_hd1910_train/{flat,backlash}/policy.onnx`；相邻 `eval_flat/` 和 `eval_backlash/` 含评估结果。需要单独评估一个检查点时，指定**与训练模型相同的 task**：

```bash
cd /home/luckysir/microduck
PYTHONPATH="$PWD/microduck_rl_hd1910/src" \
microduck_rl/.venv/bin/python scripts/evaluate.py \
  --task Mjlab-Velocity-Flat-MicroDuck-HD1910 \
  --checkpoint /绝对路径/model_3999.pt \
  --onnx runs/20260928_001943_hd1910_train/flat/policy.onnx \
  --output runs/manual_hd1910_flat_eval --commands stand forward forward_fast backward left turn --video
```

续训要用原任务、原运行目录和真实检查点名。下面以 HD-1910 平地为例；把尖括号内容替换成实际目录/文件名：

```bash
cd /home/luckysir/microduck
PYTHONPATH="$PWD/microduck_rl_hd1910/src" \
microduck_rl/.venv/bin/train Mjlab-Velocity-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 2048 --agent.logger tensorboard \
  --agent.experiment-name local_hd1910_flat_train \
  --agent.run-name continued --agent.resume True \
  --agent.load-run '<原运行目录名>' \
  --agent.load-checkpoint model_3999.pt \
  --agent.max-iterations 1000
```

`--agent.max-iterations 1000` 表示在该检查点后继续运行 1000 轮。更完整的 HD-1910 配置与前置验证见 [`microduck_rl_hd1910/docs/hd1910-training.md`](microduck_rl_hd1910/docs/hd1910-training.md)。

## 二、Radxa 真机准备

**以下命令在已装配的机器人上执行。** 本机使用 1–14 号 HD-1910 舵机、`/dev/ttyS2`、I²C3 上的 LSM6DSV16XTR。首次部署需安装 Python 依赖并完成逐关节编号、方向、直腿姿态和 IMU 安装方向核对：

```bash
cd ~/deploy_hd1910
bash install.sh
./run.sh doctor
./run.sh scan
./run.sh inspect
```

首次标定流程与各命令的含义见 [`deploy_hd1910/README.md`](deploy_hd1910/README.md) 和 [`deploy_hd1910/docs/直腿标定到HOME.md`](deploy_hd1910/docs/直腿标定到HOME.md)。真机配置 `robot.json` 是逐台标定的文件，本 GitHub 仓库不包含它。**克隆仓库后不能跳过标定直接运行策略。** 已标定的 Radxa 上保留自己的 `~/deploy_hd1910/robot.json`；不要用 `robot.example.json` 覆盖。

从新电脑向现有 Radxa 传部署源码时，先把仓库克隆到电脑，再复制目录；保留 Radxa 原来的标定配置和虚拟环境。示例地址请按当前热点分配的 IP 修改：

```bash
scp -r deploy_hd1910 deploy_xgoduck_40000 wmh@<Radxa-IP>:~/
```

此 `scp -r` 适合首次创建目标目录。更新已有部署时，先备份板上 `robot.json`，再核对将要覆盖的文件；不要直接用仓库副本覆盖整个已标定目录。XgoDuck 目录使用自己的模型与 HOME，首次安装还需复用已安装的 `.venv`、复制已核对的标定配置，把 `robot.json` 的 `model` 指向 `models/2026-09-24_20-26-30_xgoduck.onnx`，然后运行 `scripts/verify_profile.py`。当前已配置的 Radxa 上，这些步骤已经完成。

### 只读检查与网页调试台

```bash
cd ~/deploy_hd1910
./run.sh doctor
./run.sh inspect
```

网页调试台在 Radxa 上运行：

```bash
cd ~/servo_web_hd1910
bash run.sh
```

在同一局域网电脑浏览 `http://<Radxa-IP>:8080`。网页和行走程序都要独占串口；行走包装器会检查并临时停止已由其管理的网页进程。网页的标定/扭矩操作详见 [`servo_web_hd1910/README.md`](servo_web_hd1910/README.md)。

## 三、Radxa 真机行走命令

两套试机入口都会先缓慢回各自模型的 HOME，之后等待操作员输入 `WALK` 才开始策略。回 HOME 时托住躯干和头部、双脚悬空；完成后落脚、扶稳、保持静止，再输入 `WALK`。`--seconds` 是策略运行时长，最大 60 秒；`--vx` 是目标前进速度，范围 0～0.3 m/s；`--ramp-seconds 3` 使速度指令在 3 秒内渐增。`Ctrl+C`、正常完成和当前程序捕获的错误都会尝试卸力，操作员需持续扶护并能断开舵机电源。

### 本机训练的 HD-1910 平地模型

```bash
cd ~/deploy_hd1910
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/home_then_policy.py \
  --move --seconds 60 --vx 0.2 --ramp-seconds 3 \
  --max-target-jump-deg 45
```

当前 `robot.json` 的 `model` 应为 `models/hd1910_flat.onnx`。`--max-target-jump-deg 45` 只调整**单周期目标变化**上限，不改变绝对关节角度限制。切换为本机齿隙模型时，先备份 `robot.json`，将 `model` 改为 `models/hd1910_backlash.onnx`，再用同一入口。齿隙模型尚未完成充分真机验证。

### 用户训练的 XgoDuck 40000 轮模型

```bash
cd ~/deploy_xgoduck_40000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/home_then_policy.py \
  --move --seconds 60 --vx 0.1 --ramp-seconds 3
```

本部署使用 XgoDuck 的髋/踝 ±24° HOME；与本机训练的 HD-1910 HOME 不同。`scripts/verify_profile.py` 在运动前检查模型哈希、标定与 HOME 对应关系。用户此前为诊断角度限制运行过 `--log-only-target-angles`：该选项让策略目标范围和单周期跳变**只记录、不拦截**，实测约 0.21 秒后右髋偏航目标增长到约 142°，实际角度触发停止。此选项仅用于有外部支撑的故障复现，不作为默认行走命令。

两套部署的 HOME 恢复均可根据当前关节位置延长时间，已取消原先的单次 80°距离限制；单圈编码器、通信与动作停止检查仍生效。回 HOME 成功的打印只表示目标插值完成，应查看 `target_reached` 和 `errors_ticks` 确认实际误差。

### 查看运行记录

```bash
ls -lt ~/deploy_hd1910/logs/motion_*.jsonl | head
ls -lt ~/deploy_xgoduck_40000/logs/motion_*.jsonl | head
```

每次运行会保存逐帧 `motion_*.jsonl` 和同名 `.session.json`；HOME 恢复另有 `supported_home_*.json`。报告与已知真机问题见 [`reports/Radxa舵机标定进度.md`](reports/Radxa舵机标定进度.md)。旧模型在 0.05～0.1 m/s 出现原地站立、较高速度试机出现目标突变；XgoDuck 模型在起步阶段出现髋偏航目标持续增长。**这些结果尚不能证明实机实现持续行走。**
