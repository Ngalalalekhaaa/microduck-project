# HAT 扬声器与语音播放

本目录可独立复制到 Radxa 的 `~/sensors/Speaker`。默认播放“你好，我是 microduck。”，音频文件已随代码保存，播放不需要联网、TTS 服务或 GStreamer。

适用硬件为 Radxa Zero 3W、MicroDuck HAT C1 的 TLV320AIC3104 + PAM8406 音频通路。2026-11-09 已在 vendor 6.1.115 内核上完成驱动安装和旋律试听，用户确认音量够用、声音清楚。语音 WAV 已完成设备播放，程序成功退出；尚未记录用户对语音的试听反馈。移植后的脚本另有无硬件测试，见下文。

## 直接播放

在已经完成部署的机器人上执行：

```bash
bash ~/sensors/Speaker/scripts/play.sh
```

默认 PCM 音量为 `116`，Line DAC 为 `118`，Line 输出增益为 `0 dB`。这是本次旋律试听使用的设置；语音波形峰值为满量程的 `0.60`，旋律峰值为 `0.40`。它们不是声压测量值，也不是相同的主观响度。

完整的底层播放命令如下，四条 `amixer` 命令必须成功后再播放：

```bash
set -e
amixer -q -c aic3104 cset 'name=PCM Playback Volume' 116,116
amixer -q -c aic3104 cset 'name=Line DAC Playback Volume' 118,118
amixer -q -c aic3104 cset 'name=Line Playback Volume' 0,0
amixer -q -c aic3104 cset 'name=Line Playback Switch' on,on
aplay -D plughw:CARD=aic3104,DEV=0 \
  "$HOME/sensors/Speaker/audio/hello_microduck.wav"
```

其他测试方式：

```bash
# 连续说两遍，两次间隔 1 秒
bash ~/sensors/Speaker/scripts/play.sh --repeat 2

# 约 27 秒《小星星》，纯 Python 生成，随后通过 ALSA 播放
python3 ~/sensors/Speaker/scripts/play.py --melody

# 播放自己的 PCM 16-bit WAV；可用 --pcm 降低音量
bash ~/sensors/Speaker/scripts/play.sh /path/to/voice.wav --pcm 104

# 只检查 WAV 和打印将执行的命令，不操作声卡
bash ~/sensors/Speaker/scripts/play.sh --dry-run
```

脚本每次都设置上述四个播放控制项，不改麦克风配置，不写入 ALSA 永久存档。自定义 WAV 建议为 48 kHz 双声道、左右声道相同；HAT C1 的 J1 使用右声道，纯左声道文件可能无声。`--pcm` 允许 `0..116`，默认值采用已验证档位。

## J1 接线

以 **HAT 元件面朝上、40 针长排母在上、大舵机连接器在左** 的方向看 J1：左侧 1 脚为 `SPK+`，接扬声器红线；右侧 2 脚为 `SPK−`，接黑线。两线连接无源扬声器，本次使用 4 Ω / 3 W 扬声器。`SPK−` 是桥接功放输出，**不能接到 GND**。方向改变时应按板子引脚编号判断，不要只记照片左右。

## 新系统首次启用音频

现有机器人已经启用声卡，可直接播放，无需重复安装或重启。新系统先检查：

```bash
aplay -l
amixer -c aic3104 info
```

如没有 `aic3104`，安装工具并预览音频配置：

```bash
sudo apt update
sudo apt install -y python3 alsa-utils device-tree-compiler kmod dkms build-essential
python3 ~/sensors/Speaker/scripts/setup_audio.py
```

安装器只支持已核对的 Radxa Zero 3、vendor 6.1 内核、单启动项 `/boot/extlinux/extlinux.conf` 布局。若内核没有 AIC3x 驱动，需要**运行中内核完全匹配的 headers**；先确认 `/lib/modules/$(uname -r)/build` 存在。不要为了驱动构建盲目换内核。

预览通过后应用：

```bash
sudo python3 ~/sensors/Speaker/scripts/setup_audio.py --apply
```

安装器保留摄像头、UART、J5 等现有 overlay，仅补齐音频需要的 I2C3 M0 与 AIC3104 overlay；写入前备份文件，先构建驱动再改启动配置，**不自动重启**。已有 `microduck-j5-i2c` 会直接复用。若必须新增 I2C3 M0，随附 overlay 会禁用占用原 M1 引脚的 FUSB302 节点，改变 USB-C PD 协商配置；适用当前 HAT 供电方案。其他启动布局或冲突的声卡 overlay 会中止，需先核对。

需重启时先停止会驱动机器人动作的服务并让机器人有支撑或断开舵机电源，再手动执行 `sudo reboot`。重启后用 `aplay -l` 确认卡名，执行播放命令即可；本模块不会启动遥控或舵机服务。

## 生成其他中文语音（可选）

此步骤需要互联网，文本会发给 Microsoft Edge 在线语音服务。使用 [edge-tts 项目](https://github.com/rany2/edge-tts)的命令行客户端；与播放已有 WAV 无关。本次实际使用 `edge-tts==7.3.1`。

```bash
sudo apt install -y python3-venv ffmpeg
python3 -m venv ~/sensors/Speaker/.venv
~/sensors/Speaker/.venv/bin/pip install 'edge-tts==7.3.1'
PATH="$HOME/sensors/Speaker/.venv/bin:$PATH" \
  python3 ~/sensors/Speaker/scripts/generate_speech.py \
  --text '你好，我是 microduck。' \
  --voice zh-CN-XiaoxiaoNeural --rate=-5% \
  --output /tmp/microduck-new-greeting.wav
bash ~/sensors/Speaker/scripts/play.sh /tmp/microduck-new-greeting.wav
```

生成器拒绝覆盖已有输出。它通过 ffmpeg 解码为 48 kHz 双声道 PCM 16-bit，再将峰值归一到 `0.60`，首尾各加 0.5 秒静音。不对极短的问候语使用整段 `loudnorm`，避免本次测试曾遇到的输出过小问题。`audio/provenance.json` 记录随附语音的生成参数和 SHA-256。

## 检查与排查

无硬件检查：

```bash
python3 -m unittest discover -s ~/sensors/Speaker/tests -v
python3 ~/sensors/Speaker/scripts/setup_audio.py --help
bash ~/sensors/Speaker/scripts/play.sh --dry-run
```

检查音频卡和输出路由：

```bash
aplay -l
amixer -c aic3104 cget 'name=PCM Playback Volume'
amixer -c aic3104 cget 'name=Right DAC Mux'
amixer -c aic3104 cget 'name=Right Line Mixer DACR1 Switch'
amixer -c aic3104 cget 'name=Line Playback Switch'
```

本次设备路由为右 DAC `DAC_R1`、右 Line Mixer 的 `DACR1` 开启。若只在首尾听到“啵”声，先确认播放的实际卡名、文件右声道是否有数据及四项音量设置。不要把 `SPK−` 改接地来排查。命令无报错代表 ALSA 接收了音频，是否实际出声仍需听音确认。

## 文件与来源

- `scripts/play.sh` / `play.py`：播放与纯 Python 旋律测试。
- `scripts/generate_speech.py`：可选在线语音生成、峰值归一化。
- `scripts/setup_audio.py`：从已验证的 `deploy_hd1910/media/setup_radxa.py` 提取的音频独立安装器；不安装或要求摄像头。
- `audio/hello_microduck.wav`：本次请求生成的语音；`provenance.json` 记录参数。
- `vendor/aic3x-dkms`：AIC3x 内核驱动，保留 GPL-2.0 许可证及源文件版权头。
- `vendor/*.dts`：来自本项目已验证的 upstream 快照，保留 Apache-2.0 许可证；原始来源和摘要见 `vendor/sources.json`。

安装器的无硬件测试检查启动参数和已有 overlay 的保留、重复执行不重复添加、冲突拒绝；这些测试不等于在新的硬件或内核上验证过驱动。
