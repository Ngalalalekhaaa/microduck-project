# Microduck 传感器与扬声器

摄像头、扬声器和激光测距的独立部署代码统一放在本目录。每个子目录包含自己的脚本、依赖说明和测试步骤，不需要训练环境、模型或舵机标定文件。

| 目录 | 硬件与连接 | 验证状态（2026-11-09） |
| --- | --- | --- |
| [IMX219](IMX219/README.md) | IMX219 摄像头 → Radxa CSI | 已验证抓帧、ISP 与曝光收敛；标准光照画质尚未验收 |
| [Speaker](Speaker/README.md) | 扬声器 → HAT J1；TLV320AIC3104 + PAM8406 | 音乐已由使用者确认音量够用、声音清楚；中文语音播放程序成功，尚待单独试听确认 |
| [VL53L8CX](VL53L8CX/README.md) | 测距模块 → HAT J5 / I²C3 | 按 ST ULD 与手册准备；J5 已损坏，未连接、未读到距离 |

测试不需要启动行走或 Xbox 控制程序。`camera/IMX219` 在仓库中保留为兼容符号链接，新命令统一使用 `sensors/IMX219`。

## 取得代码并上传

在电脑 Bash 终端执行；将 `RADXA_HOST` 改为自己的登录地址：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Ngalalalekhaaa/microduck-project.git ~/microduck-sensors-source
git -C ~/microduck-sensors-source sparse-checkout set sensors
cd ~/microduck-sensors-source
RADXA_HOST='你的用户名@你的Radxa_IP'
ssh "$RADXA_HOST" 'mkdir -p ~/sensors'
scp sensors/README.md "$RADXA_HOST:~/sensors/"
scp -r sensors/IMX219 sensors/Speaker sensors/VL53L8CX "$RADXA_HOST:~/sensors/"
```

这里复制源码和自带语音，不改变启动配置、不重启、不启动硬件。相机的厂商安装包需要按 [IMX219 步骤](IMX219/README.md) 下载并核验；已部署好的相机和声卡无需重复安装。

## 播放“你好，我是 microduck”

已启用 `aic3104` 声卡的 Radxa，在 SSH 终端执行：

```bash
bash ~/sensors/Speaker/scripts/play.sh
```

这会设置已验证的音量并播放自带 WAV，无需网络或语音服务。等价的完整底层命令为：

```bash
set -e
amixer -q -c aic3104 cset 'name=PCM Playback Volume' 116,116
amixer -q -c aic3104 cset 'name=Line DAC Playback Volume' 118,118
amixer -q -c aic3104 cset 'name=Line Playback Volume' 0,0
amixer -q -c aic3104 cset 'name=Line Playback Switch' on,on
aplay -D plughw:CARD=aic3104,DEV=0 \
  ~/sensors/Speaker/audio/hello_microduck.wav
```

语音文件峰值已调整为 0.60，并在头尾留有静音。不要直接把全部增益调到最大；任意其他 WAV 的响度可能不同。新板安装声卡、播放音乐和生成其他语句见 [Speaker 说明](Speaker/README.md)。

## 激光测距：先准备，修好 J5 后再测试

现在可以在 Radxa 安装编译工具并编译，以下命令不会访问 I²C：

```bash
sudo apt-get update
sudo apt-get install -y build-essential
cd ~/sensors/VL53L8CX
make
./build/vl53l8cx_test --help
```

**仅在 J5 修复、模块供电和使能接法确认后执行：**

```bash
cd ~/sensors/VL53L8CX
sudo ./build/vl53l8cx_test probe --bus /dev/i2c-3 --address 0x29
sudo ./build/vl53l8cx_test range --bus /dev/i2c-3 --address 0x29 --frames 10 --hz 5
```

`probe` 核对芯片身份；`range` 上传 ST 固件、初始化并读取 8×8 距离矩阵，单位为 mm。不能用一个 `i2cget` 命令代替完整初始化。上述是待实测命令，不是已经获得距离数据的记录。

HAT J5 按引脚编号为 **1=GND、2=3.3V、3=SDA、4=SCL**；软件默认使用 Linux 7 位地址 `0x29`。用户的红色转接板虽有电平转换芯片，但尚无准确载板原理图，VIN 允许范围、LPn 与接口选择脚的默认电平仍需确认，不能直接套裸芯片供电。详见 [VL53L8CX 接线前提与来源](VL53L8CX/README.md)。

## 摄像头

已部署好的相机，在电脑上执行（不是在 Radxa 本机执行）：

```bash
cd ~/microduck-sensors-source
bash sensors/IMX219/scripts/capture_frame.sh "$RADXA_HOST"
```

照片、原始帧和诊断记录写入 `sensors/IMX219/captures/`。首次接入和安装仍按 [IMX219 完整部署步骤](IMX219/README.md) 执行。

## 本次整理的验证记录（2026-11-10）

- 摄像头迁移后 20 项离线回归通过，包含设备树合并检查；旧入口的帮助命令正常。
- 扬声器 8 项测试在电脑与 Radxa 通过，包含 WAV 格式/峰值、音频生成和启动配置保留。新目录下的问候语已通过 `aplay` 播放，退出码为 0；实际听感仍以试听为准。
- 音频安装器在现有 Radxa 的预览通过；本次没有应用启动配置或重启。
- VL53L8CX 在电脑和 Radxa 均编译通过，传输与命令行模拟测试通过。未对实物执行 `probe` 或 `range`，未取得真实距离数据。
