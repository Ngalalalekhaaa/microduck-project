# IMX219 摄像头：另一台 Microduck 的部署全流程

整理日期：2026-11-05。**把本 `IMX219` 目录完整交给同事即可使用，不依赖父级机器人代码。**
本方案完成 CSI 接入、ISP 白平衡/颜色处理、有限轮自动曝光、单帧拍摄、原始数据与 PNG 回传。
不需要训练环境、ONNX 模型或舵机标定文件。

已在一台 **Radxa ZERO 3W / RK3566 / aarch64 / Armbian Trixie / `6.1.115-vendor-rk35xx`** 上验证出帧、ISP 与曝光收敛。
摄像头为 **IMX219，Camera V2 兼容模组**；不同镜头、IR 滤光片仍需实拍验收。
**目前没有完成标准光照下的色彩、清晰度验收，不能把能出图等同于最终画质合格。**

## 1. 准备硬件、系统和连接

| 项目 | 要求 / 下载入口 |
|---|---|
| 主板 | Radxa ZERO 3W/3E，RK3566；本次真机是 3W |
| 摄像头 | [原采购 IMX219 模组](https://detail.tmall.com/item.htm?id=775872575316&skuId=5474618334309)，不是 USB 相机、OV5647 或 IMX708 |
| 排线 | 常见模组端 15Pin/1.0mm，Radxa 端 22Pin/0.5mm；[原采购排线](https://item.taobao.com/item.htm?id=678057486394&skuId=6265661507552)。按实物核对触点方向 |
| 接线图 | [Radxa 官方摄像头说明](https://docs.radxa.com/en/zero/zero3/accessories/camera) |
| 系统 | 已能运行 Microduck 的 vendor 6.1 系统优先；[Armbian 下载](https://armbian.com/boards/radxa-zero3)、[历史镜像](https://rsync.armbian.com/archive/radxa-zero3/archive/) |
| 尚未装系统 | [整机从零手册第 2、3 节](https://github.com/Ngalalalekhaaa/microduck-project/blob/main/README-%E4%BB%8E%E9%9B%B6%E5%A4%8D%E7%8E%B0.md)：包含镜像、刷卡、板卡批次引导、联网；[刷卡工具 Etcher](https://etcher.balena.io/) |
| 电脑 | 下文按 Ubuntu/Debian 或 WSL 的 Bash 终端写；能通过 SSH 访问 Radxa；Python ≥ 3.9 |

本包不刷系统、不升级内核。**主线 6.18、其他 SoC、仅有 `armbianEnv.txt` 的启动方式不能直接套用**。
已验证的启动方式是单启动项 `/boot/extlinux/extlinux.conf`，指定 `/boot/…dtb` 和已有 overlays。
官方 Radxa OS 页面中的 `rsetup` 适用于其系统，本包使用经核对的 extlinux 流程，二者不要混着改。

接线顺序：机器人停止运动、保持支撑 → 关机并断开电池和 USB → 打开 CSI 锁扣 → 排线平直插到底、核对触点朝向 → 扣好两端锁扣 → 上电。
相机接主板 CSI，不接 HAT 的 J5 或舵机接口。

在 **Radxa 控制台**确认 IP、SSH（已有 SSH 可跳过）：

```bash
whoami
hostname -I
sudo systemctl enable --now ssh
```

记录自己的用户名与 IP。下面用变量 `RADXA_HOST`，请替换成同事那台机器的真实地址，没有内置账号或密码。

## 2. 在电脑取得代码和依赖

先装电脑工具：

```bash
sudo apt-get update
sudo apt-get install -y openssh-client python3 ffmpeg git ca-certificates device-tree-compiler
```

**方式 A：接收交付压缩包（推荐）。** 将 `.tar.gz` 和同名 `.sha256` 放在同一目录，在该目录执行：

```bash
# 将文件名替换为实际收到的文件名
CAMERA_ARCHIVE=microduck-camera-IMX219-20261105.tar.gz
sha256sum -c "$CAMERA_ARCHIVE.sha256"
mkdir -p ~/microduck-camera
test ! -e ~/microduck-camera/IMX219 && tar -xzf "$CAMERA_ARCHIVE" -C ~/microduck-camera
cd ~/microduck-camera/IMX219
sha256sum -c SHA256SUMS
python3 scripts/fetch_assets.py --check
```

已有同名目录时，改用新的空目录解压，保留之前照片和配置。

**方式 B：从仓库仅下载摄像头目录。** 在电脑执行（目标目录须尚不存在）：

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Ngalalalekhaaa/microduck-project.git ~/microduck-camera-source
git -C ~/microduck-camera-source sparse-checkout set camera
cd ~/microduck-camera-source/camera/IMX219
python3 scripts/fetch_assets.py
```

它下载两个固定版本 `.deb`、Sony 数据手册、对应内核驱动参考，并核对 SHA256。
压缩包已经缓存它们；Git 不存这些厂商二进制。网络失败时可在有网络的电脑下载后重新打包。
这些缓存不包含全部 apt 系统依赖，首次安装仍需要 Radxa 能访问 apt 源。

## 3. 从电脑上传到另一台 Radxa

保持电脑当前目录为 `IMX219`：

```bash
RADXA_HOST='你的用户名@你的Radxa_IP'   # 必须修改
ssh "$RADXA_HOST" 'mkdir -p ~/camera/IMX219'
scp README.md "$RADXA_HOST:~/camera/IMX219/"
scp -r scripts vendor docs references packages "$RADXA_HOST:~/camera/IMX219/"
ssh "$RADXA_HOST"
```

终端提示密码时输入那台主板的密码。不必复制任何人的 `robot.json` 或 Wi-Fi 配置。
下面第 4、5 节都在 **Radxa 的 SSH 终端**执行。

## 4. 安装工具、启用 CSI

```bash
cd ~/camera/IMX219
cat /proc/device-tree/model
uname -m
uname -r
cat /etc/os-release
sudo apt-get update
sudo apt-get install -y python3 v4l-utils psmisc gcc libc6-dev \
  device-tree-compiler libdrm2 ca-certificates
sudo usermod -aG video "$(id -un)"
mkdir -p logs
set -o pipefail
python3 scripts/enable_camera.py 2>&1 | tee logs/overlay-preview.txt
```

预览必须通过。它合并校验已有设备树和新增 IMX219 overlay，检查 UART2、J5/I²C3 的配置没有改变。
显示缺 overlay、多个启动项、冲突相机或不支持的系统时，先按错误处理；不要删掉检查强行运行。
摄像头已经启用时不会重复添加；新板没有 J5 overlay 也不要求凭空加上。

```bash
sudo python3 scripts/enable_camera.py --apply 2>&1 | tee logs/overlay-install.txt
```

应用会在 `/var/backups/microduck-camera-boot/时间/` 保存原 `extlinux.conf` 和差异。
脚本不自动重启、不控制舵机。若这次新增了 overlay，在机器人停止运动、支撑稳定的前提下执行：

```bash
sudo reboot
```

等待主板恢复，在**电脑**重连（同一电脑终端中的变量仍有效；新终端需重新设置）：

```bash
ssh "$RADXA_HOST"
```

若 overlay 原本已启用，仍应退出 SSH 再登录一次，使 `video` 组权限生效。

## 5. 在 Radxa 安装 ISP 与官方 IQ 参数

```bash
cd ~/camera/IMX219
mkdir -p logs
id
v4l2-ctl --list-devices
python3 scripts/doctor.py --output logs/before-isp.json
python3 scripts/fetch_assets.py --check
python3 scripts/setup_isp.py --package-dir packages
```

应找到 `rkisp_mainpath` 和 IMX219 子设备。最后一条仅展示计划。
安装前关闭其他正在取流的应用；若已有 `mediad`，先停止它再使用本包单帧入口。
**本包单帧曝光与持续推流的曝光控制不能同时运行**。

```bash
systemctl is-active mediad
# 仅在显示 active 时执行；未安装此服务可跳过
sudo systemctl stop mediad
```

`stop` 不改变开机设置：重启后 `mediad` 可能再次启动，拍摄前需再次检查。
如果这台机器以后长期只使用本包单帧方案，可自行执行 `sudo systemctl disable mediad`；
恢复上游实时推流时需先恢复其配套 ISP/曝光配置，再启用 `mediad`，不能直接共用两套曝光控制。

```bash
set -o pipefail
sudo python3 scripts/setup_isp.py --apply --package-dir packages \
  2>&1 | tee logs/isp-install.txt
systemctl is-enabled rkaiq_3A
systemctl is-active rkaiq_3A
systemctl status rkaiq_3A --no-pager
python3 scripts/doctor.py --output logs/after-isp.json
```

正常应为 `enabled` / `active`。安装内容为：

- 固定版本 Radxa RK3566/3568 ISP 引擎、官方 `imx219_rpi-camera-v2_default.json`；安装前核对包 SHA256。
- 适配 vendor 内核 ioctl 的 shim；`rkaiq_3A` 使用直接受 systemd 管理的进程，日志进入 journal。
- 引擎启动前固定传感器为 **1920×1080**；开机自动启动 ISP，未拍照时等待开流。
- IQ 中只改两处枚举兼容名称、关闭引擎 AE：**抓帧脚本管曝光，ISP 管白平衡和颜色**。没有手调 RGB、CCM 或 Gamma 曲线。

系统配置备份位置：`/var/backups/microduck-camera-isp/时间/`，安装时会打印。
配置清单与实际 IQ 哈希保存在该目录。当前摄像头方案只需上述依赖，不需要安装音频 DKMS。

若报 `Existing rkaiq/system service drop-ins`，说明这台板子有其他服务覆盖配置，例如上游 `robot.conf`。
执行 `systemctl cat rkaiq_3A.service` 查看安装器列出的文件。先确定保留原媒体栈还是迁移；
迁移时应备份并逐项处理这些配置，再重跑安装。安装器会在改动前停止，不自动删除别人的配置。
不能仅因服务显示 active 就跳过此问题；后加载的配置可能覆盖本包的 shim 和启动钩子。

## 6. 在电脑拍一帧并自动取回

退出 SSH 回到电脑：

```bash
exit
```

在电脑 `IMX219` 目录运行（解压方式 A 的目录如下；方式 B 使用其对应目录）：

```bash
cd ~/microduck-camera/IMX219
RADXA_HOST='你的用户名@你的Radxa_IP'   # 使用同事的实际地址
bash scripts/capture_frame.sh "$RADXA_HOST"
```

按提示完成 SSH/SCP 认证。约数秒至十几秒后打印保存目录，默认在电脑本目录的 `captures/日期/…/`。
每次是新目录。相机拍摄期间保持静止；不要同时运行 `cheese`、旧录像脚本或另一份抓帧程序。

需要稍亮的测光目标，或指定保存位置：

```bash
bash scripts/capture_frame.sh --bright "$RADXA_HOST"
bash scripts/capture_frame.sh "$RADXA_HOST" "$HOME/Pictures/microduck-$(date +%Y%m%d_%H%M%S)"
```

| 文件 | 内容 |
|---|---|
| `frame.png` | 电脑上直接打开的照片，默认 640×360，保持 16:9，不作离线手动调色 |
| `frame.nv12` | 从 ISP 输出的未压缩 NV12；**不是传感器 Bayer RAW** |
| `capture.json` | 成功状态、曝光/增益、每轮测光、AE 收敛、裁切比例、IQ 与帧哈希 |
| `capture.log` | 板上取流和控制命令完整日志；失败时也尽量取回 |
| `ssh.log`、`transfer.log` | 连接、传输日志 |

程序在开流后强制写入曝光，等待新帧迭代测光；1080p 曝光上限 1760 行、模拟增益上限 8 倍。
结束后停流、恢复原控制值并删除大块临时序列；板端 `/tmp/microduck-camera-…/` 保留单帧和日志供追溯。
脚本会校验传输后的原始帧 SHA256，再按协商的 full/limited range 转成 PNG。

`success=true` 表示成功出帧；`ae_converged=true` 才表示本轮达到设定测光目标。
单次有限轮 AE 并非所有光线下都必然收敛，暗室、背光、直射亮斑可能触发警告。
不要把 `--get-ctrl` 数值当作传感器寄存器的实时读回值。

## 7. 验收、问题记录与回退

首次验收：相机固定，镜头朝向约 0.5–1 米外有细节的室内场景；确认镜头无遮挡、灯光均匀。
可以让普通白纸与场景同时入镜，粗查是否偏绿、是否过曝，但不要用手电直射纸面。
**白纸用于验收白平衡，不是每次拍摄的必备物，也不能替代色卡与完整 ISP 标定。**
如果与标准 Camera V2 的镜头/滤光片不同，官方 IQ 不保证完全匹配，需单独调试。

排障先收集只读信息（Radxa）：

```bash
cd ~/camera/IMX219
python3 scripts/doctor.py --output "logs/doctor-$(date +%Y%m%d_%H%M%S).json"
sudo journalctl -k -b --no-pager > logs/kernel-current-boot.txt
sudo journalctl -u rkaiq_3A -b --no-pager > logs/rkaiq-current-boot.txt
```

电脑取回板端诊断：

```bash
mkdir -p ./logs/radxa
scp -r "$RADXA_HOST:~/camera/IMX219/logs/." ./logs/radxa/
```

完整原因和对应处理见 **[问题与排查](docs/问题与排查.md)**，包括：极暗但能出帧、开流覆盖曝光、偏绿、IQ/ABI 不匹配、`libdrm2` 缺失、服务假活、摄像头占用、EIO/NACK、格式与权限问题。

如果仅需停止本包 ISP，先结束采集，再执行 `sudo systemctl disable --now rkaiq_3A`。
需还原启动配置时，选择**本次安装输出的**备份，确认期间没有其他启动配置更新，再执行：

```bash
# Radxa；将时间目录换成实际备份，勿照抄占位文字
BOOT_BACKUP=/var/backups/microduck-camera-boot/实际时间目录
sudo diff -u "$BOOT_BACKUP/extlinux.conf" /boot/extlinux/extlinux.conf
sudo cp -a "$BOOT_BACKUP/extlinux.conf" /boot/extlinux/extlinux.conf
sudo reboot
```

ISP 安装备份用于逐项核对恢复 IQ、drop-in、模式脚本和 shim；不要整包覆盖 `/etc`。
它不是完整系统镜像，**不会自动回退之前软件包版本**。已有其他相机方案的机器应先保留其原配置和包版本。

## 8. 目录、历史记录和再次交付

```text
IMX219/
  README.md                 本说明
  scripts/                  启用 CSI、安装 ISP、只读诊断、拍照回传、下载、打包
  vendor/                   ioctl 兼容 C 源码、许可与来源
  docs/                     问题与排查、资料来源
  references/               来源/SHA 清单；下载后的 Sony PDF 与驱动参考
  packages/                 下载后的两份固定版本 ISP .deb
  tests/                    不连机器的自动检查
  evidence/                 去标识的实际验收摘要及本次包验证结果
  captures/                 以后拍摄的照片和单次日志（本地生成）
  archive/                  本机已有原始照片/日志/历史代码（不在 Git 和部署包）
  dist/                     生成的交付压缩包（不在 Git）
```

原始记录已集中到 `archive/raw-reports/`、`archive/captures/radxa_camera/`；旧工作区的
`reports/…camera…`、`videos/radxa_camera` 是兼容链接，仍指向同一批文件。
历史实验脚本、旧媒体代码和上游 mediad 源码快照在 `archive/legacy-code/`，仅供追溯，不作为安装入口。
仓库公开，不发布房间照片、原始主机日志、登录信息和机器标定；部署所需代码与问题结论均在公开目录中。

重新生成可转交包（电脑，本目录执行）：

```bash
python3 scripts/fetch_assets.py
python3 -m unittest discover -s tests -v
python3 scripts/package.py
```

把打印出的 `.tar.gz` 与 `.sha256` 一起交给同事即可。该包包含全部摄像头部署代码、说明和已缓存资料/ISP 包，
不包含整台 Microduck 的运动控制环境；同事已有的机器人部署继续使用自己的配置。
持续录像/实时 WebRTC 属于后续集成，上游入口和与本包 AE 的边界见 [资料来源](docs/资料来源.md)。
