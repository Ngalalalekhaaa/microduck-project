# HD-1910 Microduck：Radxa 首次试机包

**2026-11-09：**[传感器与扬声器统一目录](../sensors/README.md)：IMX219 摄像头、HAT 语音/音乐播放、VL53L8CX 测距工具。扬声器音乐已试听通过；测距工具尚未实机验证。

**2026-11-05：**[IMX219 摄像头独立部署](../sensors/IMX219/README.md)，包含官方 ISP/IQ、自动曝光抓帧回传和完整问题记录；目录可单独交给另一台 Radxa。

**2026-10-30：新增按装配几何标定。** 全新舵机从脚向上装配的完整流程、14关节 HOME 和45°孔位记录见[从脚装配与几何标定](docs/从脚装配与几何标定.md)。`assembly-template` 生成空白表，`assembly-calibrate --profile 表.json` 离线预览，追加 `--apply` 保存并备份软件标定；不写舵机。`home-plan/home --from-pose assembly`、支撑 HOME 脚本和 Xbox A 已兼容该参考，不要求再做 straight 标定。工厂中位孔型相位不能从八孔等分或旧装配图推定，未知参数保持空白。

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

- 需要 14 个不同 ID 的策略舵机。嘴部不参与策略；启用独立的 `jaw` 配置后，通过 X 键控制。
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
IMU 读取失败、电压下降的分析命令及测量边界见 [IMU供电诊断](docs/IMU供电诊断.md)。
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

### 软件电压下限调整为 4V（2026-10-02）

默认配置的舵机电压范围为 **4.0～8.4V**，低于4.0V或高于8.4V时停止并卸力。
已有真机的 `robot.json` 需单独更新。上传 `scripts/apply_voltage4.py` 后，
在没有运动任务运行时执行：

```bash
cd ~/deploy_hd1910
sudo systemctl stop microduck-xbox.service
.venv/bin/python scripts/apply_voltage4.py --apply
sudo systemctl start microduck-xbox.service
```

更新脚本先备份原配置，再只修改 `control.servo_voltage_range`；模型、舵机标定、
IMU及其他控制设置均保留。重复执行不会重复备份。不加 `--apply` 只预览，
脚本不访问串口，不驱动舵机。此修改不增加新的电流阈值或4～5V告警。

## 电脑终端数字控制（无需手柄）

2026-10-31 新增。通过普通交互式 SSH 登录 Radxa 后执行：

```bash
cd ~/deploy_hd1910
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/home_then_policy.py \
  --move --keyboard --seconds 0 --vx 0 \
  --max-vx 0.3 --max-vy 0.3 --max-wz 1.0 \
  --log-only-target-angles
```

先托稳躯干和头部、双脚悬空回 HOME，再让双脚落地、保持静止，输入 `WALK` 回车。
IMU 校偏与 HOME 对齐完成后，从零速度启动模型；看到数字控制就绪提示后输入以下命令，
直接按键，无需回车。方向以机器人自身为准。

| 输入 | 速度指令 |
| --- | --- |
| `1` | 前进，最高 0.3 m/s |
| `j` | 前进 0.15 m/s |
| `k` | 前进 0.20 m/s |
| `l` | 前进 0.25 m/s |
| `2` | 后退，最高 0.3 m/s |
| `3` | 向左侧移，最高 0.3 m/s |
| `4` | 向右侧移，最高 0.3 m/s |
| `5` | 向左旋转，最高 1.0 rad/s |
| `6` | 向右旋转，最高 1.0 rad/s |
| `0` | 将三轴速度指令归零，继续运行模型，保持上力 |
| `q` | 退出并卸力，先托住机器人 |

duck1 的 `robot.json`、`robot.home_actions_20261101.json` 和
`robot.home_actions_com_20261101.json` 自 2026-11-01 增加六个行走模型选择：

| 单键 | 行走模型 |
| --- | --- |
| `a` | 默认，齿隙 30000：`hd1910_backlash_30000.onnx` |
| `b` | 摩擦齿隙 5000：`hd1910_friction_backlash_5000.onnx` |
| `c` | 摩擦齿隙 4000：`hd1910_friction_backlash_4000.onnx` |
| `d` | 齿隙 9729：`hd1910_backlash_9729.onnx` |
| `e` | 原始齿隙 3999：`hd1910_backlash.onnx` |
| `f` | 行走 1849：`hd1910_walk_1849.onnx` |

六个模型启动时校验 SHA256 并预加载。按字母后使用当前模型停步 0.6 秒，再切换并保持零速度；
重新按 `1`–`6` 行走。切换保留前一周期原始动作历史，不对动作增加滤波。
坐姿、低头抬头、滚翻、起立等待或另一次切换期间，字母请求会提示未执行，完成后需重新按键。
数字动作键继续使用原配置：`7` 坐下/站起、`8` 站起、`9` 低头抬头、`r` 单次前滚翻。
键盘 `9` 完成后回到刚才选择的行走模型，以零速度恢复 3 秒，再接受新方向键；
期间输入不排队，看到恢复完成提示后重新按 `1`–`6` 或 `j/k/l`。
这一恢复过程不调用坐站模型，前一周期原始动作历史照常保留。
所有坐下站起模型也可直接按字母选择，选择之后坐下和站起都使用 `7`：

| 单键 | 坐下站起模型 |
| --- | --- |
| `g` | 原版 1699：`hd1910_sitstand_1699.onnx` |
| `h` | 摩擦版 2598：`hd1910_sitstand_friction_2598.onnx` |
| `i` | 重心版 2598：`hd1910_sitstand_com_2598.onnx` |
| `m` | 旧版 20261015：`hd1910_sitstand_20261015.onnx` |
| `n` | 强化随机化 4999 实验版：`hd1910_sitstand_robust_4999.onnx` |

五份坐站模型均在启动时校验并预加载。字母选择不触发坐下/站起，行走速度指令会归零；
按 `7` 坐下，再按 `7` 用同一模型站起。坐着时需先按 `7` 站起才能换其他坐站模型。
动作或起立等待期间的选择不排队；站立时切换坐站权重后适应 3 秒，再按 `7` 或方向键。
当前两份微调 HOME 配置默认仍选 `n`，`robot.json` 默认仍选 `m`，本次只增加可选模型。
`n` 此前仿真评估没有完成坐下任务，保留为明确标记的实验选项。
这些默认值描述当前 duck1；逐台标定的 `robot.home_actions*.json` 不在公共仓库中。
[keyboard_models.example.json](keyboard_models.example.json) 只包含公共模型路径、哈希和键位，
可将其中模型字段合并到自己的已标定配置；它不是完整的机器人配置，不包含舵机零位。
`keyboard_sitstand_default` 对应的模型必须与该配置的 `sitstand_model` 相同。
这些字母只在策略已启动的键盘模式生效，HOME 完成后仍需输入 `walk` 并回车。
使用当前 HOME 和动作配置的完整入口为：

```bash
cd /home/duck1/deploy_hd1910
./run.sh relax
./walk_from_tuned_home.sh \
  --config "$PWD/robot.home_actions_com_20261101.json" \
  --move --keyboard --seconds 0 --vx 0
```

方向指令持续到下一条命令，方向切换采用与手柄相同的速度变化率上限；`0` 立即归零速度指令。
启动前输入的数字不排队执行；不认识的非空输入会归零速度并提示。
此模式不使用 5 秒自动归零，也不使用固定 `--vx` 或 `--ramp-seconds`；
`--max-vx/--max-vy/--max-wz` 分别指定前后、侧移和旋转速度。
前进四档为 `1=0.30`、`j=0.15`、`k=0.20`、`l=0.25 m/s`，直接按键，无需回车。
显式设置更低的 `--max-vx` 时，各档不超过该上限。后退最多 0.3 m/s，侧移最多 0.3 m/s，旋转最多 1 rad/s。
`Ctrl+C`、终端断开或控制故障仍会停止并卸力；`0` 停步后模型仍在运行，实际站立表现需观察。

duck1 的策略连续超时门槛通过 `control.policy_overrun_threshold_s=0.030` 设置：
连续 5 轮工作耗时超过 30 ms 才触发此项停机，任一轮不超过 30 ms 就重置计数。
这允许本次日志中的连续约 21 ms 用时，调度目标仍为 50 Hz，不保证每轮都能达到 20 ms。
单轮策略硬限 80 ms、采样间隔 100 ms、IMU 数据年龄 100 ms 及舵机反馈检查继续有效。
未配置该字段时保留原 20 ms 门槛；纯 HOME 保持不采用这个策略放宽值。
会话日志记录实际门槛，连续超时故障会记录门槛和次数。

策略目标的角度告警与编码转换在每轮只计算一次，发包复用该轮结果；发包前仍检查停止请求和时限。
诊断没有手柄时的读取、推理和散热情况，可运行旧遥控器基准的零指令模式：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/benchmark_control_readonly.py \
  --seconds 30 --no-gamepad --telemetry \
  --output logs/control-readonly-telemetry.json
```

此工具禁止写舵机，只测读取、推理和序列化；结果不代表上力行走的完整周期。
输出文件逐轮保存 CPU 频率、温度和降频设备状态，以区分通信延迟与运行环境变化。

## 开机自动等待 Xbox：日常使用不需要电脑

Radxa 安装 `microduck-xbox.service` 后，开机自动等待已配对手柄，
手柄可以在主板启动前或启动后开机。等 Xbox 灯常亮，再使用下面的按键。
服务启动不会自动给舵机上力或移动关节；需要新按下 A。

1. 托稳躯干和头部、双脚悬空，短按 **A**：回训练 HOME，通常至少 6 秒，较远姿态自动延长。
2. HOME 完成并反馈到位后，双脚落地、机身静止，短按 **菜单键（≡）**：
   IMU 静止校偏约 3 秒，对齐 HOME 约 1 秒，然后运行策略。
3. **B**：停止当前流程、舵机卸力，松开 B 后可重新按 A。
   这不会断开电池电源，也不会关闭 Radxa。

配置并标定嘴部后，短按 **X** 张开嘴巴，再短按一次闭合；长按只切换一次。
等待 A、回 HOME、HOME 保持和行走期间均可使用。校偏时收到的 X 指令在校偏完成后执行。
嘴部使用 `robot.json` 中独立的 `jaw` 配置，与身体共用同一条串口，策略仍控制原来的14个关节。
嘴部开合默认用0.8秒平滑过渡，首次 X 才上力；B、手柄断线或控制故障同时卸力嘴部。
2026-10-02 实机记录：嘴部15号，闭合344刻度，张开51刻度，X事件码307。
两位置的舵机编码器角度分别为30.234°和4.482°，相差25.752°；这不是嘴巴开口的几何角度。

回 HOME、校偏和策略运行期间，A/菜单键的提前按下不会留到下一阶段执行。
未到位时菜单键无效；可以按 B 复位后重新按 A。
HOME 已到位、正在等待菜单键时，再按 A 可重新回 HOME。
手柄断线时流程停止并尝试卸力；重连后重新等待新的 A。
策略默认持续运行，直到按 B 或故障检查触发停止，不再有 60 秒到时停止。

默认选择 **摩擦随机化 5000 轮齿隙模型**，`models/hd1910_friction_backlash_5000.onnx`。
前进最大0.3 m/s、后退最大0.2 m/s、左右 ±0.3 m/s、转向 ±1 rad/s。
正常使用只需主板供电和蓝牙手柄；不依赖热点、SSH 或电脑。

以下是首次安装命令，不需要每次执行：

```bash
cd ~/deploy_hd1910
sudo python3 scripts/install_xbox_service.py --user "$(whoami)" --start
```

检查与维护命令：

```bash
systemctl status microduck-xbox --no-pager
journalctl -u microduck-xbox -n 50 --no-pager
# 先扶住机器人；停止服务会卸力。停止后可运行手动入口或网页调试台。
sudo systemctl stop microduck-xbox
sudo systemctl start microduck-xbox
```

模型、最长时间和前后基础速度上限保存在 `/etc/default/microduck-xbox`；后退另外限制到0.2 m/s：
`MICRODUCK_VARIANT=backlash`（或 `flat`）、`MICRODUCK_SECONDS=0`（不限时；正数表示运行秒数）、
`MICRODUCK_MAX_VX=0.3`。修改后重启服务生效。

指定其它兼容的 ONNX 模型时，在同一文件中设置
`MICRODUCK_MODEL=models/hd1910_backlash_30000.onnx`。
该设置优先于 `MICRODUCK_VARIANT`，启动时先校验模型，再只更新
`robot.json` 的 `model`；原舵机标定、IMU、控制参数、按键流程以及服务的
时长和前后限速配置继续使用。删除 `MICRODUCK_MODEL` 后恢复原 5000 轮选择流程。

手动运行指定模型（先停止开机服务以避免同时控制）：

```bash
sudo systemctl stop microduck-xbox
cd ~/deploy_hd1910
bash xbox_model_hd1910.sh models/hd1910_backlash_30000.onnx 0 0.3
```

`0` 表示不限时；末尾参数为前后基础速度上限，后退不会超过0.2 m/s。左右和转向上限沿用当前包装脚本的
0.3 m/s 和 1 rad/s。模型导出与归一化一致性检查记录在
`hd1910_backlash_30000_manifest.json` 和 `models/hd1910_backlash_30000.parity.json`。

## Xbox 手柄手动启动入口

Radxa 上配对并信任手柄后，先检查输入数据；这条命令只打开手柄输入设备：

```bash
cd ~/deploy_hd1910
.venv/bin/python scripts/test_xbox.py --seconds 30
```

两只摇杆分别推过各方向后松开，中心应显示 `command_vx_vy_wz: [0,0,0]`。
蓝牙连接成功与摇杆数据正常是两项检查。若尚未收到输入报告，不能开始运动。
`--list` 可列出输入设备；通常无需固定 `/dev/input/event4`，程序自动寻找 Xbox。

```bash
# 平地版 9750 轮，60 秒；前后速度上限 0.3 m/s
bash xbox_walk.sh flat 60
# 齿隙版 9729 轮，60 秒；前后速度上限 0.3 m/s
bash xbox_walk.sh backlash 60 0.3
```

上述 Xbox 入口也使用 A → HOME → 菜单键 → 校偏/策略，B 卸力后等待 A。
若开机服务正在运行，先托住机器人并停止服务，再运行手动入口。
开始时松开两只摇杆；完成 IMU 校偏后，再推动摇杆。

| 手柄操作 | 输入给策略的命令 |
| --- | --- |
| 左摇杆前／后 | 前进最大0.3 m/s，后退最大0.2 m/s |
| 左摇杆左／右 | 向左／右横移，±0.3 m/s |
| 右摇杆左／右 | 左／右转向，±1 rad/s，约 ±57.3°/s |
| 松开摇杆 | 速度命令归零，策略继续维持站立 |
| A 键 | 空闲时回 HOME；HOME 到位后可重新回 HOME |
| 菜单键（≡） | HOME 到位后，开始校偏和策略 |
| B 键（任意阶段） | 结束当前流程、舵机卸力，重新等待 A；卸力前托住躯干 |
| Ctrl+C 或终端输入 q 回车 | 停止并卸力 |

断开手柄或输入线程过期时，当前按键流程停止并尝试卸力。
开机服务自动等待重连，手动入口则需重新运行脚本。
摇杆有 12% 中心死区；速度命令在约 1 秒内由零增至上限，运行开始另有 3 秒渐增。
模型动作没有新增缩放或滤波，仍按 50 Hz 推理。
每 0.5 秒终端显示 `cmd=[vx vy wz]`，日志记录实际传入策略的命令。
这些是期望速度，真实速度与转向效果仍需实机测量。

本入口使用已有标定、IMU 映射及 HOME。目标角范围和目标跳变沿用仅记录模式；
实际反馈、姿态、电压、温度、通信和时序检查沿用当前部署配置。
手柄输入依赖 `evdev>=1.9,<2`；首次在 Debian 编译时需要 `python3-dev`、`gcc`、`linux-libc-dev`。

## IMU 辅助检查 HOME 与站姿微调

单个躯干 IMU 只能测机身倾斜，无法唯一确定 14 个关节的安装零点。
编码器反馈只能验证已有标定下的跟踪误差。头颈角度、膝弯曲程度、脚间距离和机身高度仍需几何参考。
训练 HOME 的躯干重力目标为 `[0,0,-1]`；脚部参考架存在约 ±5° 侧倾，不能强制把两脚参考架都水平当成 HOME。

只读取关节和 IMU，输出机身 roll/pitch 与逐关节跟踪误差：

```bash
cd ~/deploy_hd1910
.venv/bin/python scripts/imu_home_assist.py
```

自动慢速回现有 HOME，再做小范围机身倾斜微调：

```bash
.venv/bin/python scripts/imu_home_assist.py --move --level
```

1. 启动前托住躯干和头部，双脚悬空；程序先调用现有慢速 HOME 入口。
2. 出现提示后双脚落到平整、不打滑的地面，自然承重。旁边扶护，但不要把机身固定住；输入 `GROUND`。
3. 程序分别探测左右踝俯仰、左右髋侧倾的组合响应，再逐步减小机身前后/左右倾斜。
   探测为 ±1.5°，每次调整不超过 0.5°，每个参数累计不超过 ±4°；速度不超过 1°/s。
4. 结束后保持当前姿态上力，报告保存为 `logs/imu_home_*.json`，含原姿态、探测响应、调整量和最终姿态。
   Ctrl+C、SSH断开或错误会尝试卸力，需要旁边扶护。

IMU 必须牢固装在躯干且安装方向已核对；IMU 自身装歪与机身倾斜无法自行区分。
身体被托死、脚打滑、响应不稳定时会停止。只读模式不会发舵机写指令。

该工具不更改 `robot.json`、舵机内部零点、IMU 坐标映射或策略。
小倾角不代表真实的 14 关节已匹配仿真，微调报告也不是新的机械零位标定。
现有行走入口仍会回训练 HOME，这次微调不会自动加入策略动作或观测。
恢复原训练 HOME（先托住躯干/头部、双脚悬空）：

```bash
.venv/bin/python scripts/imu_home_assist.py --move
```


## Y 键站立／坐下（2026-10-17）

在原有 A → HOME → 菜单键 → IMU 校偏 → 策略流程启动后，短按 Y 坐下，再短按 Y 站起。
行走中第一次按 Y 会先给行走策略零速度，停步0.6秒后切到站坐策略；坐下期间摇杆不触发行走。
站起由同一份站坐权重执行，至少等待3秒，然后推动摇杆可恢复原行走策略；摇杆居中时保持站立。
长按Y只切换一次；启动、HOME和校偏期间提前按下的Y不会在策略开始后执行。
X嘴部开合、B卸力及原有启动流程继续使用当前部署逻辑。Y仅在策略运行期间生效。

`robot.json` 新增 `sitstand_model: "models/hd1910_sitstand_20261015.onnx"`；原有 `model` 仍为行走模型。
站坐ONNX内含观测归一化，输入61维、输出14关节，`obs[48]` 的0/1分别表示站立/坐下。
两份模型在IMU启动前加载，控制循环不再现场读取模型；切换时保留上一帧原始动作历史。
日志新增 `policy_mode`（walking/stopping/sit/stand）和会话 `policy_transitions`。
站坐策略的小幅目标超调若越过现有标定可表示的0..4095刻度，会在边界饱和；超过边界5度则停止。
这只处理站坐目标，不改标定、不做刻度取模，上一帧动作仍保存网络原始输出。
将 `sitstand_model` 设为null可关闭Y站坐功能。此次更新仅增加策略/代码和该字段，保留已有标定、IMU、嘴部、
速度、电压、保护阈值及行走模型。站坐已通过仿真、无硬件单元测试；真机动作效果须现场扶护验证。

## 强化学习 HOME 校正（2026-10-31）

### 2026-11-01：固定评估模型，站立微调后悬空跟随保存目标

`fine_tune_best_home.sh` 固定加载 `models/pose/selected/` 的模型和匹配合同，
选择依据见该目录的 `selection.json`。它不会随训练自动换模型；“best”指本轮候选比较
和已有真机证据支持的选择，不表示已经证明真实机械 HOME 最优。
本轮比较 model300、model600、model1100，选择 model600：相同 P8 仿真、8个环境、
10秒、随机±3°零偏下，两种评估设置的位置误差比 model300 低约9%–12%，
首回合存活数没有退步。model600 尚未进行真机动作测试；model300 的已有真机结果
不能当作 model600 的实测结果。

以下在机器人上的同一个终端分步执行。先托稳躯干、双脚离地，卸力后回到当前标定 HOME：

```bash
cd /home/duck1/deploy_hd1910/home_calibration_no_current_20261031
../run.sh relax
./hold_calibrated_home.sh --move
```

`relax` 关闭14个身体舵机的扭矩并核对读回，并不切断电池供电。
将双脚放到平整地面，自行站稳后，运行10秒模型微调，并显式保存这一次报告：

```bash
RUN_REPORT="/home/duck1/deploy_hd1910/logs/home_calibration_pose_$(date +%Y%m%d_%H%M%S).json"
./fine_tune_best_home.sh --standing-confirmed --seconds 10 --report "$RUN_REPORT"
```

只有该次微调正常完成后，托住躯干、让双脚离地，再执行10秒固定目标跟随：

```bash
./follow_saved_home.sh --source-report "$RUN_REPORT" --supported-confirmed --move --seconds 10
```

悬空阶段不运行模型、不使用 IMU，目标固定为上述成功报告内14个关节的最后目标。
它会验证报告、标定映射、目标及上力状态；不会静默改用更早的成功报告。
若仍保持这组目标和 P8，舵机内部位置环继续追随，脚本采样10秒；
`mode=observe_existing_goal`、`goal_writes=0` 是这种正常情况。
若已经全部卸力，则设置易失 RAM P8，预加载当前位置、回读后上力，缓慢移到保存目标；
每次最多2刻度，动作至少6秒，总过程10秒。当前位置离目标超过15°、混合上力，
或已上力却持有另一组目标时会拒绝；后一种情况可托稳后执行 `../run.sh relax` 再重试。
不要在这两个阶段之间再回 HOME，否则会覆盖刚才的目标。

完成后输出 `completed` 和 `target_reached`：前者表示流程结束并验证目标仍保持，
后者要求14个关节都在保存目标的1°内且观测持续至少0.5秒。
两阶段均保存所有关节的最终位置、目标及误差 JSON/CSV；悬空报告路径会在终端打印。
悬空阶段电流只记录，电压、温度、舵机状态、通信时效及位置限位仍检查；
接管后故障会尝试卸力并验证，因此整个阶段需持续托稳。
仅悬空跟随命令去掉 `--move` 可只读预览；模型入口追加 `--mode inspect` 为离线检查，
追加 `--mode monitor` 为只读采样。到达保存目标不等于真实机械 HOME 标定已正确，
也不会把临时目标写进 `robot.json` 或 EEPROM。

2026-11-01：无电流发布目录的回 HOME 和微调入口默认使用 RAM **P=8、D=0、I=0**。
原行走仍从原 `robot.json` 使用 P=5。这些无电流模型的训练基准仍为 P=5，P=8 是用户
指定的运行参数覆盖，报告会分别记录，不代表模型已在 P=8 下重新训练。
首次从 P=5 切换时先托稳躯干、双脚离地，在发布目录执行 `../run.sh relax`，再执行
`./hold_calibrated_home.sh --move`；成功后放到地面自行站稳，再运行微调命令。
持位脚本只在全部舵机卸力时设置 RAM 增益，已上力但档位不符时拒绝，不会自动卸力。
重复同一 P8 流程不需要为改增益再次卸力。两入口均可显式追加 `--ram-p 5` 使用原档位。

每次微调正常结束或异常停止后，终端输出 `end_joint_positions`，运行 JSON 增加
`end_joint_positions`，并写同名 `*.positions.csv`，按模型顺序记录14个关节的ID、名称、
实测刻度/角度、相对当前HOME角度、目标读回、跟踪误差、电流、电压和温度。
异常时先执行原来的停发/卸力处理，再尝试独立读取舵机位置；读取失败会把最后有效采样
标为 `last_observed_sample`、`fresh_readback=false`，完全无读数则保留14行空值。
CSV包含读取来源、采样时间、样本年龄和捕获时的退出状态。角度来自现有软件标定，
不是独立测得的机械零偏；报告不自动写回 `robot.json`。

新版提供无实时电流输入的两个任务，独立部署目录为
`/home/duck1/deploy_hd1910/home_calibration_no_current_20261031`。
位置目标模型输入 272 维；受力比例目标模型输入 384 维，其中额外的目标比例来自
模型合同中的固定参考值，不是实时电流或受力读数。两种模型都不要求 `--current-map`，
无电流发布目录的模型微调入口按用户要求默认 `--current-check log-only`：电流仍真实
读取和记录，超过原 800 mA 软件阈值不再因此中止。此选项仅影响电流数值阈值，
温度、电压、IMU、跟踪误差、关节限位、通信与舵机故障状态检查继续执行；
不会修改舵机内部限流寄存器，也不会把电流加入模型输入。
日志记录 `current_check: log-only` 和 `current_stop_ma: null`。如需恢复软件电流
中止检查，可在命令末尾指定 `--current-check stop`；直接调用 Python CLI 默认仍为 stop。
无电流发布目录的 `hold_calibrated_home.sh` 按用户要求沿用原有回 HOME 的
`--log-only-feedback` 行为：当前直腿参考标定下，电流、温度、电压、位置跟踪和
最终到位误差只记录，不因这些软件阈值中止；通信时效、舵机状态报错、目标步长、
硬件位置范围、目标回读与配置检查仍执行。装配参考标定仍使用严格反馈检查。
`holding: true` 表示目标已保持，是否实际到位需同时查看 `target_reached` 和误差。

```bash
ssh duck1@10.142.40.165
cd ~/deploy_hd1910/home_calibration_no_current_20261031
# 托住躯干、双脚离地，成功后保持现有标定 HOME
./hold_calibrated_home.sh --move
# 放到地面、确认能自行站稳后，选择一种模型
./fine_tune_home.sh --standing-confirmed --seconds 10
# 或者：先重新回到 HOME，再试受力比例目标模型
./fine_tune_force_ratio.sh --standing-confirmed --seconds 10
```

两个微调命令分别加载 `models/pose/latest/`、`models/force_ratio/latest/` 的 ONNX
及匹配合同；同一时间只能运行一种。模型不会在控制期间自动更新。
微调默认运行 10 秒（包含最初 0.5 秒持位等待），正常结束保持最终目标；异常仍可能
提前结束。无需设置此前首次短试使用的 `--seconds 0.75`。
`--mode inspect` 可覆盖动作模式，进行不访问硬件的模型检查；`--mode monitor`
仅采样并计算候选动作，不控制舵机。没有实际 HOME 持位时可能跳过候选推理。
训练刚启动时导出的早期快照只能用于接口验证，校准效果仍需独立评估。
脚本源位于 `scripts/home_no_current/`，发布时复制到 release 根目录。
下文的 392 维、电流映射要求属于保留的旧模型版本。

入口 `scripts/home_calibration.py` 使用独立的 HOME ONNX（392 输入、14 输出），
不是 61 维行走模型。20 Hz、8 帧历史、前 0.5 秒等待、每步最多 0.25°、累计最多 8°，
与训练动作语义一致；真实关节角和未知零偏不作为部署输入。输出仅是本次 HOME
命令的补偿候选，不自动写入 `robot.json` 或舵机 EEPROM。

先离线检查模型接口，此模式不打开串口或 IMU：

```bash
.venv/bin/python scripts/home_calibration.py --mode inspect \
  --model models/home_calibration.onnx \
  --contract models/home_calibration_contract.json \
  --write-current-template current_map.unverified.json
```

模板按已确认的关节 ID/方向生成，但所有 `verified` 初始为 `false`。
电流寄存器 69–70 的现有解码为 bit15 符号-幅值、6.5 mA/单位；需要核对各关节的
实际极性、单位、静态偏置与训练的带符号电机电流是否一致。
`current_sign` **已包含模型关节轴方向**，程序不会再乘一次编码器方向。
映射公式为 `I = current_sign * current_scale_a * (decoded_raw - current_zero_raw)`。
支持明确指定的符号-幅值或二进制补码；只有无符号幅值时只能记录诊断，不能驱动模型校正。
不要通过把 `verified` 改成 `true` 来代替实际核验，也不用 load/位置误差伪造电流方向。

只读采集传感器，并在输入适用时计算建议动作：

```bash
.venv/bin/python scripts/home_calibration.py --mode monitor \
  --model models/home_calibration.onnx --contract models/home_calibration_contract.json \
  --current-map current_map.unverified.json --seconds 10
```

只读模式禁止所有舵机寄存器写入，包括扭矩开关；IMU 会初始化并开始采样。
IMU 使用已有安装旋转和配置的陀螺偏置（未配置时为零），不会在受载状态重新校零。
观测里的已知命令来自真实 goal 回读，未发送的建议动作不会累计成虚假的已执行命令。
没有电流映射或已有目标超过 HOME±8°时，仅记录传感器并注明跳过推理。

待模型评估合格、电流映射完成核验，并已按现有流程使机器人双脚站稳保持 HOME 后，
才运行动作模式：

```bash
.venv/bin/python scripts/home_calibration.py --mode run \
  --model models/home_calibration.onnx --contract models/home_calibration_contract.json \
  --current-map current_map.verified.json --standing-confirmed --seconds 10
```

动作模式要求全部 14 颗舵机已经使能、mode=4、resolution=1、RAM P5/D0/I0，
已有目标等于标定配置的 HOME，实测姿态接近 HOME 且稳定。程序不自动起身、不使能
舵机、不改增益，也不自动暂停其他控制服务；发现总线占用就退出。部署时通过
`--config /home/duck1/deploy_hd1910/robot.json` 读取现有配置并共享该目录的操作锁。

正常完成后保留最后目标和扭矩，日志中记录命令补偿与目标回读。Ctrl+C 或软件故障
默认停止后续目标并保留已有持位，反馈已丢失时不会声称持位已验证；明确过温、
过流、过大倾角/角速度会尝试卸力并记录是否确认成功。也可显式使用
`--fault-action release` 让接管后的软件故障尝试卸力。前置检查失败始终零电机写入。
软件 watchdog 受串口与进程调度约束，不是硬件断电保护。每次最多 10 秒，不自动重启回合。

### 真机分两步：先保持当前标定 HOME，再启动模型微调

已部署目录 `/home/duck1/deploy_hd1910/home_calibration_rl_20261031` 中提供两个独立入口。
先登录机器人，托住躯干和头部、双脚离地，再执行第一条动作命令：

```bash
ssh duck1@10.142.40.165
cd ~/deploy_hd1910/home_calibration_rl_20261031
./hold_calibrated_home.sh --move
```

它读取原目录的 `robot.json`，从当前位置缓慢回到当前标定 HOME。全部卸力时使用
易失 RAM P5/D0/I0 后上力；如果已经上力，只接受全部关节均使用这组增益。
混合上力状态或已上力但增益不符会在写入前退出。回位保留电流、电压、温度和跟踪误差
检查，预加载目标经回读核对后才使能舵机。成功输出 `"holding": true` 后命令退出，
舵机继续保持 HOME；不带 `--move` 时只读预览。

此时将双脚放到平整、不打滑的地面，确认机器人能自行站稳、无需手持续承重。
第一步只是位置持位，没有主动平衡控制；两步之间没有自动倒计时。
放好后，在同一目录执行第二条命令：

```bash
./fine_tune_home.sh --standing-confirmed
```

第二条使用 `models/latest/home_calibration.onnx` 和同目录匹配的契约，运行最多 10 秒，
正常结束保持校正后的目标，不改标定零点。`latest` 是手动同步的快照，不会在运行中
跟随训练自动换模型；再次更新时需要同步 ONNX 与契约两个文件。
电流映射仍必须真实核验：默认 `current_map.duck1.json` 未核验时会在接触硬件前拒绝
微调，并保留第一步的持位。核验后可用 `--current-map /绝对路径/已核验映射.json`
指定对应文件。不能仅修改 `verified` 标志代替实测。

两条命令共用原部署目录的操作锁；其他控制进程占用设备时退出，不会自动停止它们。
第一步执行中中断或故障会尝试卸力；完成后等待放置期间仅由舵机内部控制器持位。
第二步的停止与故障行为见上文。重新运行第二步前需要先恢复 HOME 目标，不能从上次
已有补偿直接开始新回合。

输出为 `logs/home_calibration_*.json` 和逐帧 `*.samples.jsonl`，包含模型 SHA256、
契约、传感器、动作和退出状态。真机只能看到带现有零偏的编码器角，无法用这些日志
直接证明真实 HOME 误差已消除，最终精度仍需独立几何测量。
