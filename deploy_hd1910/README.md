# HD-1910 Microduck：Radxa 首次试机包

本文件夹可单独用 `scp -r` 复制。包含 Python CPU 运行程序、两份实际训练的 ONNX、
舵机协议、LSM6DSV16X I²C/SPI 驱动、标定工具和姿态图片。无需复制训练环境或安装 CUDA。

**这是首次台架试机程序。已做本机 CPU/模拟接口测试，尚未连接你的 Radxa 和机器人。**
模型在仿真中能前进，但低速、后退、横移和直行仍有不足，不能据此宣称已经通过真机行走验收。
默认使用表现较好的 `models/hd1910_flat.onnx`。备用齿隙版是 `models/hd1910_backlash.onnx`。

**0.2.1 已加入你提供的 1–14 号关节映射；导入后仍需逐颗实测方向。**
保留 0.2.0 的直腿参考标定和缓慢移到训练 HOME 功能。
完整摆法、前提与命令见 [直腿标定到 HOME](docs/直腿标定到HOME.md)。
使用 `calibrate --pose straight`、只读 `home-plan --from-pose straight`，再显式执行
`home --from-pose straight --move-seconds 8 --seconds 3 --enable-motion`。

## 1. 复制和安装

在当前训练电脑执行，替换用户名和地址：

```bash
scp -r /home/luckysir/microduck/deploy_hd1910 用户名@RADXA_IP:~/
ssh 用户名@RADXA_IP
```

进入 Radxa 后执行：

```bash
cd ~/deploy_hd1910
bash install.sh
./run.sh init
./run.sh map-ids --profile servo_ids.user.json
./run.sh doctor
./run.sh scan
./run.sh inspect
```

安装脚本需要网络，先检查剩余空间和 64 位 Python，再创建当前文件夹内的 `.venv`。
固定的 ONNX Runtime 1.24.4 支持本包所核对的 Linux ARM64 Python 3.11–3.14，
至少需要 Python 3.11；较旧系统不要把系统 Python 强行覆盖成新版本。
不修改系统 Python、启动配置或开机服务。若提示缺少 venv，Debian/Ubuntu 可先运行：

```bash
sudo apt-get update
sudo apt-get install python3-venv
```

如果之前创建环境时出现 `ensurepip is not available`，安装对应版本的 venv 包后重试。
例如 Python 3.13 使用 `sudo apt-get install python3.13-venv`。
0.2.2 的安装脚本会补齐不完整环境中的 pip；旧版脚本可先运行
`python3 -m venv .venv`，再运行 `bash install.sh`。这两条命令不加 sudo，无需删除目录。

`doctor` 保存 `logs/doctor.json`；`inspect` 保存 `logs/servos.json`。遇到识别失败，先保留这两份输出，
不要盲目改设备树或更换接线。有关总线号和排针，见 [Radxa 与 IMU 接线](docs/Radxa与IMU接线.md)。

## 2. 先读数，确认硬件

`robot.json` 是你的机器配置，初始零位和方向为空。`init` 使用官方参考 ID，随后必须运行上面的
`map-ids`，导入你提供的编号：左腿 1–5、颈和头 6–9、右腿 10–14。
已有配置也用同一条导入命令：保留串口、IMU 等设置，不向硬件写入 ID。
发生 ID 变化的关节会清空方向、零位和映射确认，全局零位标定也会失效；
重复导入完全相同的 ID 会保留已有标定。编号来自你的记录，方向仍需实测。
默认串口 `/dev/ttyS2`、波特率 1000000
来自官方安装参考；实际设备以 `doctor` 和 `scan` 为准，可用 `nano robot.json` 修改。

- 需要 14 个不同 ID 的策略舵机。嘴部 ID 34 不参与策略，本程序不向它发送动作。
- 如果只有一个出厂 ID，先逐颗分配 ID；同时连接同 ID 的舵机无法通过软件区分。
  此包不会自动改 ID 或波特率。`scan --all --baudrate 1000000` 可只读扫描完整 ID 范围。
- 模式应为 `mode=4`（HD-1910 纯位置 PD），`angle_resolution=1`。
  程序读回不符会停止，不会自动重写 EEPROM 模式。
- `inspect` 会记录现有模式、增益、phase、分辨率、速度/加速度设置、电压和温度。
- 不要同时运行官方 robotd、servo-web 或其它占用舵机串口的程序。
  先用 `systemctl is-active robotd` 和 `sudo fuser /dev/ttyS2` 检查。
  若确实是 robotd 占用试机串口，可 `sudo systemctl stop robotd`，结束后按需要恢复。

串口/I²C权限不足时，查看 `ls -l /dev/ttyS2 /dev/i2c-*` 的所属组。
在系统确实使用 dialout/i2c 组的情况下，运行下面命令并重新 SSH 登录：

```bash
sudo usermod -aG dialout,i2c "$USER"
```

没有这些组时不要照抄；使用设备实际所属组。临时以 root 运行整个程序会产生 root 所有的配置和日志，
不建议作为默认安装方式。

## 3. 标定舵机零位与方向

详细命令和每个关节的正方向，见 [零位标定操作步骤](docs/零位标定操作步骤.md)。

先阅读 [姿态与零位](docs/姿态与零位.md)，对照其中真实训练模型的正面、侧面、背面图片。
这里有三个不同概念：**舵机绝对刻度、机器人关节 q=0、策略默认 HOME**。

托住机器人，主动卸力：

```bash
./run.sh relax
./run.sh watch --seconds 30
```

手动轻微转动一个关节，观察哪个 ID 的读数变化。按姿态文档的关节正方向转动，
记录刻度是增大还是减小，再保存这个关节的映射。例如，**仅在实测吻合时**：

```bash
./run.sh map-joint --joint left_hip_yaw --id 1 --direction -1
```

其余 13 个关节依次处理；不能因为别人的机器全为 −1 就直接照填。
`map-joint` 只写本地配置，每次变更会使旧零位标定失效。

将所有关节摆到文档里的 **模型 ZERO 姿态**，固定住，执行：

```bash
./run.sh calibrate --pose zero
```

这是 ZERO 流程。如果选择更容易用轴心对齐的直腿参考，请使用
[直腿标定到 HOME](docs/直腿标定到HOME.md) 中的 `calibrate --pose straight`；该姿态不是全零。

若有夹具/量角方法能准确复现训练 HOME，也可选择 `calibrate --pose home`。
**这个命令只读取当前角度，无法判断你的物理摆姿是否准确。** 不要把任意稳定姿态登记为 HOME。
保存值只进入 `robot.json`，不调用硬件位置校准、不写 EEPROM、不默认零点为 2048。
已有配置修改前自动保留 `.bak.*` 文件。

保存后可运行 `./run.sh watch --angles --seconds 30` 查看模型角度，或加 `--id 8` 查看头偏航。
该检查不需要 IMU，不驱动舵机；按独立机械基准回零和量取小角度，检查正负号和角度变化是否对应。

## 4. 检查 IMU 安装方向

```bash
./run.sh imu --seconds 20
```

开始时将机身保持静止约 2 秒，测量陀螺仪偏置，然后依次轻轻前倾、后倾、左倾、右倾。
程序会自动查找可访问的 `/dev/i2c-*`，仅检查 0x6A/0x6B 地址的身份寄存器。
只有恰好一个 `WHO_AM_I=0x70` 时才自动选用。无设备或多个匹配会给出错误。

机身坐标为 X 向前、Y 向左、Z 向上。正确直立时 `gravity ≈ [0,0,-1]`。
进一步核对倾斜方向，见接线文档。确认安装恰好与机身轴一致后可保存：

```bash
./run.sh imu-map --axes=+x,+y,+z
```

这只是轴一致时的例子。其它安装方向需要不同映射；仅仅竖直读数正确还不能确定 X/Y 方向。
小的安装倾角可在 `robot.json` 填入完整 3×3 正交旋转矩阵，不要改策略观测去补错误方向。

## 5. 不发送动作，检查真实循环

```bash
./run.sh check --seconds 10
```

此命令读取舵机和 IMU、构造 61 维观测并进行 ONNX 推理，**不写目标、不写增益、不上力**。
托住机器人，放到接近 HOME 的姿态，使输出更有参考意义。
日志应连续完整，典型控制用时应低于 20 ms，无丢包、过期 IMU 或持续超时。
每次会话另保存 `.session.json`，即使启动或回 HOME 失败，也记录停止原因及最后一帧读数。
输出目标检查错误时，先检查零位和方向，保留日志；不要直接扩大角度限制。

## 6. 先 HOME，再短时策略试机

放在可支撑机身的台架上，腿脚有活动空间，准备随时断开舵机电源。
先手动摆到接近 HOME，不要从折叠或倒地姿态启动。

```bash
./run.sh home --enable-motion --seconds 3
```

程序从当前角缓慢进入 HOME（约 3 秒），再保持 3 秒，最后卸力。**卸力时要托住。**
它会在本次运行临时设置 RAM P=5、D=0、I=0，退出后尝试恢复原值；不修改 EEPROM。
运行 P=5 与训练设置一致，但实际 HD 固件响应仍需用这一小范围试验确认。

HOME 对齐且反馈正常后，先测试零速度策略，再试前进：

```bash
./run.sh run --enable-motion --seconds 5 --vx 0
./run.sh run --enable-motion --seconds 5 --vx 0.15
```

`Ctrl+C`、`SIGTERM` 或输入 `q` 后回车会停止并尝试卸力。每次最长 30 秒，默认不会后台运行或开机自启。
若试机后支撑检查和零速度稳定性合格，可在明确控制场地内逐步增加前进指令；
仿真中 0.15 m/s 可能响应偏弱，不能为了“让它动”跳过标定或直接猛加速度。

切换备用模型：编辑 `robot.json` 的 `model` 为 `models/hd1910_backlash.onnx`。
无需改变关节顺序、零位或 IMU 轴映射。

## 控制行为和边界

- 动作按训练约定 `目标关节角 = HOME + 原始 action`；不新增动作低通、动作缩放或归一化。
- `home` 只执行姿态过渡和保持，不加载/推理策略。直腿入口只对已记录且实时接近的参考开放，
  不放宽 `run` 的启动姿态限制；回HOME途中检查跟踪误差和反馈跳变。
- 关节速度用相邻校准角度的有限差分，避免猜飞特 `phase` 对反馈速度单位的影响。
- IMU 使用 120 Hz 原始数据及软件互补滤波，不是原版小板的 SFLP；该差异需实机验证。
- 第一次目标写入本身也按“可能运动”处理。HD 的实际运行项目报告过关扭矩后写目标仍拖动的现象，
  因此 read-only 命令完全不发目标，运动命令的首个目标是刚读到的当前位置。
- 对应传感器断连、异常值、过期、倾斜、电压/温度异常、目标超限、角度跳变与超时，程序停止并尝试卸力。
- 首次试机保守地拒绝超出模型关节范围的**目标角**。训练中的低增益控制允许一定目标超调，
  因而正常策略也可能触发此限制；这是停止条件，不是已证明策略失效，更不能自动解除限制。
- 软件看门狗不能处理系统卡死、断线后无法发包或舵机固件故障；不能把 SSH/软件停止当作独立硬件急停。
- 本包保留已有速度/加速度/电流上限，不照搬来源中含义不同的 0 值；这些设置应通过 `inspect`
  记录并在小范围 HOME 测试中确认。不会自动改保护阈值。

详细来源见 [NOTICE](NOTICE.md)，本机验证见 [验证记录](docs/验证记录.md)。
