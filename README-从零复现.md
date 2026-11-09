# Microduck 从零复现：下载、装机、HOME、训练与部署

**2026-11-09：**[传感器与扬声器统一目录](sensors/README.md)：IMX219 摄像头、HAT 语音/音乐播放、VL53L8CX 测距工具。扬声器音乐已试听通过；测距工具尚未实机验证。

**2026-11-05：**[IMX219 摄像头部署全流程](sensors/IMX219/README.md)，包含官方 ISP/IQ、自动曝光抓帧回传和完整问题记录；目录可单独交给另一台 Radxa。

核对日期：**2026-10-28**。适用：**Radxa Zero 3W + Robot HAT + 14 颗 HD-1910 + HAT J5 直连 LSM6DSV16XTR I²C IMU**；嘴部可另加 ID 15。

按章节顺序操作；只部署现成模型时可跳过第 17–18 节训练。已有机器从缺少的步骤继续，保留自己的 `robot.json`。本文核对了源码和下载入口，未重新刷卡、启动训练或驱动真机；跑通命令与稳定行走需分别验收。

> `~` / `$HOME` 是 Linux 家目录；**机器人 HOME 是训练默认关节姿态**。记录 HOME 的完整命令在 **第 10 节**，不需要把舵机全部调成 2048。

**2026-10-30 新机装配入口：**使用全新 HD-1910、按八孔45°档位从脚向上装配，见[从脚装配与几何标定](deploy_hd1910/docs/从脚装配与几何标定.md)。已知中位孔型相位和连接件基准后，可用 `assembly-template` / `assembly-calibrate` 直接生成软件标定和 HOME 目标，现有 Xbox 回位已兼容；不需要再摆直腿。该新入口尚不在原10月28日压缩包内，须使用当前新版代码。没有机械基准的数据不能直接填0。

2026-10-28 新增六模型测试与部署见 [六模型/Xbox 操作说明](deploy_hd1910/docs/20261028模型测试与Xbox部署.md)，录像统一保存在 [录像索引](videos/20261028_model_suite/README.md)。新模型更新包为 `microduck-model-suite-20261028.tar.gz`，与下面的从零复现基础包分开。

四份更新模型的十字键遥操入口见 [Xbox 四模型按键部署](deploy_hd1910/docs/Xbox四模型按键部署.md)：↑新旧行走、↓前滚翻、←摩擦坐站、→重心坐站；使用独立的 `microduck-xbox-four-actions-20261028.tar.gz` 更新包。

## 0. 先准备这些

| 物品 | 用途 |
| --- | --- |
| NVIDIA GPU 电脑 | 训练；本机基线 Ubuntu 22.04.5、RTX 4050 Laptop 6 GB，环境和训练建议空出 30 GB 以上 |
| Radxa Zero 3W、32 GB microSD、读卡器、USB 数据线 | 主板、刷系统；主板单独供电用 5 V |
| Robot HAT、HD-1910 ×14、匹配打印件/舵盘/紧固件 | 机械、电源与舵机总线；嘴部另算 |
| LSM6DSV16XTR 模块、线材、散热片/风扇 | IMU 与散热 |
| 合适的舵机电源、支撑架、飞特 USB 总线转接板 | 首次编号、标定、托住机器人试机 |
| 蓝牙 Xbox 手柄（可选）、同一 Wi-Fi/热点 | 手柄控制及电脑 SSH/SCP |

下载入口集中在这里；选择与系统架构匹配的安装包：

| 下载什么 | 网址 / 选择 |
| --- | --- |
| 训练电脑 Ubuntu | [Ubuntu 22.04 LTS](https://releases.ubuntu.com/22.04/)：Desktop **amd64** ISO；已有可用 Ubuntu/NVIDIA 环境不用重装 |
| 刷卡软件 | [balenaEtcher](https://etcher.balena.io/) 或 [Armbian Imager](https://github.com/armbian/imager/releases) |
| Radxa 系统 | [Armbian Zero 3](https://www.armbian.com/radxa-zero-3/)、[历史归档](https://rsync.armbian.com/archive/radxa-zero3/archive/)；本文硬件修复脚本基于 **26.2.1 / Trixie / vendor 6.1.115 / minimal** |
| V1.12 社区镜像说明 | [镜像说明](https://github.com/fanhao375/microduck-replica/blob/master/tools/radxa/镜像使用说明.md)、[发布页](https://github.com/fanhao375/microduck-replica/releases)；按板卡批次选，见第 2 节 |
| Python 环境管理器 uv | [官方安装页](https://docs.astral.sh/uv/getting-started/installation/)，第 17 节给命令；Python 3.12 由 uv 下载 |
| NVIDIA 驱动 | [Ubuntu 驱动安装说明](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)，第 17 节给命令 |
| 飞特 FD 调试软件 | [飞特官方 Gitee](https://gitee.com/ftservo/fddebug)；本源码包也带 `microduck-replica/tools/飞特/FD1.9.8.5(250729).7z`，Windows 解压运行 |
| USB 转接板驱动 | 先看芯片型号；[CH343 驱动](https://www.wch.cn/downloads/CH343SER_EXE.html) 用于 CH343，CH340 用 FD 仓库对应驱动，不能混装代替 |
| Windows 串口终端 | [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)：Serial、对应 COM、115200、8N1、无流控 |
| 飞特版机械资料 | [CAD 仓库](https://github.com/fanhao375/microduck-replica-cad)、[大型 CAD 下载](https://github.com/fanhao375/microduck-replica-cad/releases)、[打印件](https://github.com/fanhao375/microduck-replica/tree/master/print) |
| 电路与原版参考 | [HAT 电路](https://github.com/pollen-robotics/elec_RPI_Robot_HAT)、[原版 Microduck](https://github.com/pollen-robotics/microduck)、[原版训练](https://github.com/pollen-robotics/microduck_rl) |

**新电脑装 Ubuntu：**下载 ISO → Etcher 选择 ISO/安装 U 盘并写入 → 从 U 盘启动 → 按安装向导选择目标磁盘、创建账户、联网 → 重启进入 Ubuntu。写 U 盘和安装系统都会改目标磁盘，先确认盘内数据。打开终端 Ctrl+Alt+T，安装本文用到的基础命令：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssh-client python3 \
  python3-venv build-essential ripgrep xz-utils unzip nano
```

**命令位置约定：**标“电脑”的块在 Ubuntu 执行；标“Radxa”的块在 SSH 登录后执行。`cd` 切目录；`sudo` 临时提权；行尾 `\` 表示命令还没结束。命令出错先处理，不接着发运动命令。

| 位置 | 本机实例 | 统一写法 |
| --- | --- | --- |
| 电脑家目录 / 项目 | `/home/luckysir` / `/home/luckysir/microduck` | `~` / `~/microduck` |
| Radxa 家目录 / 部署 | `/home/wmh` / `/home/wmh/deploy_hd1910` | `~` / `~/deploy_hd1910` |
| 机器人标定及 HOME | `/home/wmh/deploy_hd1910/robot.json` | `~/deploy_hd1910/robot.json` |
| 训练 Python | 电脑项目内 | `~/microduck/microduck_rl/.venv/bin/python` |
| 推理 Python | Radxa 部署内 | `~/deploy_hd1910/.venv/bin/python` |

在任一终端确认自己在哪里：

```bash
whoami
hostname
printf '%s\n' "$HOME"
pwd
```

`exit` 退出 SSH 回电脑；运动命令均在 Radxa 执行，训练命令均在电脑执行。

整体顺序：**获取完整代码 → 系统与网络 → 装配/接线 → 驱动与权限 → 舵机编号/方向 → 直腿标定并记录 HOME → IMU → 回位/策略 → Xbox → 可选训练新模型 → 备份。**

## 1. 获取完整适配源码

**2026-11-01 更新：**[汇总仓库](https://github.com/Ngalalalekhaaa/microduck-project) 已同步当前训练源码、
部署入口和 ONNX。新电脑可直接克隆；目录已存在时另选空目录，保留原工作区：

```bash
git clone https://github.com/Ngalalalekhaaa/microduck-project.git "$HOME/microduck"
cd ~/microduck
test -f deploy_hd1910/xbox_model_hd1910.sh
test -f scripts/verify_hd1910_export.py
test -f microduck_rl/uv.lock
```

逐台标定配置、Wi-Fi 密码、虚拟环境和 `.pt` 训练检查点不在 Git 仓库中。
当前键盘键位和动作恢复流程见 [部署说明](deploy_hd1910/README.md#电脑终端数字控制无需手柄)。
下列压缩包仍是 **2026-10-28 历史快照**，用于复现该日期的基线；其后新增功能以 Git 仓库为准。

将这两个文件复制到**新电脑** `~/Downloads/`，执行：

```bash
cd ~/Downloads
sha256sum -c microduck-repro-20261028.tar.gz.sha256
# 仅首次解压；若 ~/microduck 已存在，先另选空目录，勿覆盖工作区。
test ! -e "$HOME/microduck" && tar -xzf microduck-repro-20261028.tar.gz -C "$HOME"
cd ~/microduck
sha256sum -c REPRO_SHA256SUMS
test -f deploy_hd1910/xbox_model_hd1910.sh
test -f scripts/verify_hd1910_export.py
test -f microduck_rl/uv.lock
```

两次校验应全部 `OK`；原工作区直接 `cd ~/microduck`，不用重新解压。包包含训练源码/锁文件、部署模型/脚本、舵机网页、I²C 修复工具和手册；不含虚拟环境、系统镜像、训练检查点、机器私有标定及日志。若需要复现历史某个外部 PT 的训练过程，还必须另有该 PT 和训练配置。

要重新制作当前源码快照，在**电脑**执行：

```bash
cd ~/microduck
python3 scripts/package_reproduction.py
```

原始公开项目只作参考，可另存空目录；不要覆盖本适配目录：

```bash
mkdir -p ~/microduck-upstream
cd ~/microduck-upstream
git clone https://github.com/fanhao375/microduck-replica-cad.git
git clone https://github.com/fanhao375/microduck-replica.git
git clone https://github.com/pollen-robotics/elec_RPI_Robot_HAT.git
```

## 2. 给 Radxa 安装系统

基线：**Armbian 26.2.1 Minimal / Debian 13 Trixie / vendor 6.1.115 / aarch64 / Python 3.13**。2026-10-28 已核验历史镜像和校验文件仍可下载；当前 Imager 默认版本可能不同。**电脑**下载：[官方镜像](https://rsync.armbian.com/archive/radxa-zero3/archive/Armbian_26.2.1_Radxa-zero3_trixie_vendor_6.1.115_minimal.img.xz)、[官方 SHA256](https://rsync.armbian.com/archive/radxa-zero3/archive/Armbian_26.2.1_Radxa-zero3_trixie_vendor_6.1.115_minimal.img.xz.sha)。

```bash
mkdir -p ~/Downloads/microduck-os
cd ~/Downloads/microduck-os
RADXA_IMAGE=Armbian_26.2.1_Radxa-zero3_trixie_vendor_6.1.115_minimal.img.xz
RADXA_ARCHIVE=https://rsync.armbian.com/archive/radxa-zero3/archive
curl -fL --retry 3 -O "$RADXA_ARCHIVE/$RADXA_IMAGE"
curl -fL --retry 3 -O "$RADXA_ARCHIVE/$RADXA_IMAGE.sha"
sha256sum -c "$RADXA_IMAGE.sha"
```

应显示 `OK`；当次官方 SHA256 为 `6612794106f6790f84d128cf17440b583cb2374ef5ea860126c848ea5c3c4c72`。

**普通匹配板卡：**下载镜像 → 若刷卡器要求先解压则 `xz -dk "$RADXA_IMAGE"` → Etcher 选择镜像和 microSD → Flash → 等待校验。先用 `lsblk -o NAME,SIZE,MODEL,MOUNTPOINTS` 确认卡，写卡会清空所选目标。[Radxa 刷卡说明](https://docs.radxa.com/en/zero/zero3/getting-started/install-os)

**V1.12 / AIC8800D80：**若官方引导不能启动，可用本源码包的脚本生成 bootfix 镜像。先取得第 1 节源码包；旧社区成品 `bootfix-20260920` 本次未在 Releases 找到，不依赖其仍可下载。以下在 Ubuntu 电脑操作，预留数 GB 镜像临时/输出空间：

```bash
sudo apt-get install -y curl xz-utils python3 util-linux dpkg
cd ~/microduck/microduck-replica/tools/radxa
# 创建自己的配置，以下密码占位文字必须改成你自己的密码。
cat > card.local.conf <<'CARD'
WIFI_SSID=''
USER_NAME='wmh'
ROOT_PASSWORD='请改成自己的root密码'
USER_PASSWORD='请改成自己的wmh密码'
CARD
chmod 600 card.local.conf
nano card.local.conf
sudo bash build-armbian-card.sh \
  "$HOME/Downloads/microduck-os/Armbian_26.2.1_Radxa-zero3_trixie_vendor_6.1.115_minimal.img.xz" \
  card.local.conf
```

`WIFI_SSID=''` 表示先不写 Wi-Fi，开机后按第 3 节连接。脚本自动下载并校验匹配的 Radxa U-Boot；只生成输入镜像所在目录的 `…minimal-鸭子卡.img`，再用 Etcher 写此文件。**V1.12J/其他批次可能还需匹配 DTB，不能直接套用这个脚本。**

插卡，用 **5 V IN / USB_OTG 的 USB 2.0 口**供电并等待启动。**bootfix 镜像**可由此口接数据线到电脑，Windows 用 PuTTY 打开新 COM，115200 / 8N1 / 无流控；USB 3.0 Host 口不提供此控制台。**原版镜像未必启用 USB 串口**，首次用 HDMI 显示器+USB 键盘登录，或使用 Imager 支持的网络/账户预设。bootfix 在 Ubuntu 电脑可用：

```bash
sudo apt-get install -y screen
ls /dev/ttyACM* /dev/ttyUSB*
sudo screen /dev/ttyACM0 115200
```

设备名按实际输出填写；screen 退出为 Ctrl+A、K、Y。自己制作的镜像先按 `card.local.conf` 的账户/密码及屏幕提示登录；若仍进入未初始化的 Armbian 首次设置，旧基线入口为 `root / 1234`，完成改密和建用户。**改成自己的密码并创建普通用户 `wmh`**；若首次设置未创建，在管理员账户执行：

```bash
sudo adduser wmh
sudo usermod -aG sudo,dialout wmh
```

主板装上 HAT 后改用 Wi-Fi 登录，避免 HAT 与电脑 USB 同时供 5 V。

## 3. 联网、找到 IP、SSH 登录

在 **Radxa 控制台**执行 `systemctl is-active NetworkManager`。如果运行且有 `nmcli`：

```bash
sudo nmcli --ask device wifi connect "你的热点名" ifname wlan0
ip -4 a show wlan0
sudo systemctl enable --now ssh
```

若 Minimal 基线没有 NetworkManager，改用下面这一套；**两套选一，不同时管理 wlan0**：

```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
```

填写以下内容，改 SSID/密码；nano 按 Ctrl+O、回车保存，Ctrl+X 退出：

```ini
ctrl_interface=DIR=/run/wpa_supplicant GROUP=netdev
update_config=1
country=CN
network={
    ssid="你的热点名"
    psk="你的热点密码"
    key_mgmt=WPA-PSK
    pairwise=CCMP
    ieee80211w=0
    scan_ssid=1
}
```

启用无线和 DHCP 获取地址：

```bash
sudo chmod 600 /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
sudo mkdir -p /etc/systemd/network
sudo tee /etc/systemd/network/25-wlan0.network >/dev/null <<'NETWORK'
[Match]
Name=wlan0

[Network]
DHCP=ipv4
NETWORK
sudo chmod 644 /etc/systemd/network/25-wlan0.network
sudo systemctl enable --now systemd-networkd wpa_supplicant@wlan0
sudo networkctl reload
sudo networkctl reconfigure wlan0
sudo wpa_cli -i wlan0 status
ip -4 a show wlan0
ip -4 route
sudo systemctl enable --now ssh
```

预期 `wpa_state=COMPLETED` 且 wlan0 有 IPv4。电脑连接同一热点，地址从上述输出或热点“已连接设备”读取。**电脑**登录：

```bash
read -rp 'Radxa 当前 IP：' RADXA_IP
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 wmh@"$RADXA_IP"
```

首次核对主机指纹后输入 `yes` 和账户密码；输入密码时不会显示字符。新终端要重新设置 `RADXA_IP`。**登录后的 Radxa**核验：

```bash
whoami
cat /etc/os-release
uname -m
uname -r
python3 --version
df -h ~
```

预期用户 `wmh`、架构 `aarch64`，内核/系统与第 2 节匹配。换热点只改网络/IP，不重装；热点客户端隔离可能阻止 SSH。已配对的 Xbox 本地控制无需热点。

## 4. 传送主板运行包

**电脑**：将 IP 替换成第 3 节查到的当前地址；下面命令不复制电脑的 Python 环境或标定：

```bash
cd ~/microduck
read -rp 'Radxa 当前 IP：' RADXA_IP
tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='logs' --exclude='poses' --exclude='robot.json*' --exclude='*.pid' \
  --exclude='.git' --exclude='directions.json' \
  -czf microduck-radxa.tar.gz deploy_hd1910 servo_web_hd1910 radxa_j5_i2c_fix
sha256sum microduck-radxa.tar.gz > microduck-radxa.tar.gz.sha256
scp microduck-radxa.tar.gz microduck-radxa.tar.gz.sha256 wmh@"$RADXA_IP":~/
```

**已有机器更新前**：先托稳、按 B、退出网页/手动程序；有 Xbox 服务就停止并备份本机配置：

```bash
sudo systemctl stop microduck-xbox.service
cd ~/deploy_hd1910
cp -p robot.json "robot.json.backup.$(date +%Y%m%d_%H%M%S)"
```

新装跳过上述备份块。然后在 **Radxa** 解包：

```bash
cd ~
sha256sum -c microduck-radxa.tar.gz.sha256
tar -xzf microduck-radxa.tar.gz
```

运行目录为 `~/deploy_hd1910`、`~/servo_web_hd1910`、`~/radxa_j5_i2c_fix`。包中没有 `robot.json`，已有标定可保留。旧 `microduck_radxa_20261007.tar.gz` 不含后续全部改动，本流程不再依赖它。

## 5. 装配、供电和接线

按第 0 节飞特版 CAD/装配资料打印、装舵盘和连杆；核对左右镜像、转轴、紧固与线缆活动空间。电气测试先卸载或支撑关节，再装成整机。

| 接线 | 要点 |
| --- | --- |
| 电源/2S 电池 → HAT | 先核对舵机与 HAT 额定电压；2S 标称 7.4 V、满电 8.4 V |
| HAT → Radxa | 降压后 **5 V**，不可把舵机电压接入主板 5 V |
| HAT J5 → IMU | **3.3 V、GND、SDA、SCL** 按丝印/原理图对应；固定在躯干刚性结构，不能随头转 |
| HAT → 舵机 | VCC、GND、DATA；左右腿、头颈三支路共享 DATA、每颗 ID 唯一 |
| UART / I²C | 本基线 UART2 为主板物理针 8/10，J5 SDA/SCL 为物理针 3/5；经 HAT 半双工接口连接，不把裸 UART TX/RX 直接短到 DATA |

舵机线束逐颗串接，但供电是并联。电源显示 **CC** 代表到达限流并可能降压，动作时需查看舵机端电压；设定电压不等于负载端电压。装散热片/风扇。历史上某台短接 R33 的排障结果不作为新机必做步骤。

## 6. 安装 Radxa 运行依赖

**Radxa：**

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev gcc linux-libc-dev \
  device-tree-compiler i2c-tools psmisc bluez
cd ~/deploy_hd1910
bash install.sh
./run.sh init
./run.sh map-ids --profile servo_ids.user.json
./run.sh self-test
```

`install.sh` 在主板建立 CPU 推理 `.venv`；`init` 不覆盖已有配置；`map-ids` 只导入软件 ID 表；`self-test` 不驱动硬件。预期模型接口 `[1,61] → [1,14]`。部署要求 64 位 Python ≥3.11，本基线为 3.13；训练另用 3.12，不能跨机器复制 `.venv`。

## 7. 给普通账户设备权限

**Radxa：**

```bash
sudo groupadd -f i2c
sudo groupadd -f input
sudo usermod -aG dialout,i2c,input wmh
sudo tee /etc/udev/rules.d/99-microduck-i2c.rules >/dev/null <<'RULE'
KERNEL=="i2c-[0-9]*", GROUP="i2c", MODE="0660"
RULE
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=i2c-dev
exit
```

重新从电脑 SSH 登录后检查：

```bash
id
ls -l /dev/i2c-* /dev/ttyS2
```

`dialout` 访问串口，`i2c` 访问传感器，`input` 访问手柄事件。组权限需要新登录会话才能生效。没有 `/dev/ttyS2` 时，继续第 8 节；权限拒绝与没有检测到设备是两类问题。

## 8. 启用 UART2 与 HAT J5 的 I²C

### 8.1 先检查，不重复改已正常的机器

```bash
cat /boot/armbianEnv.txt
cat /boot/extlinux/extlinux.conf
ls /boot/dtb/rockchip/overlay/*uart2*m0* /boot/dtb/rockchip/overlay/*dwc3*peripheral*
ls -l /dev/ttyS* /dev/i2c-*
```

本次已调通的启动配置含：

```ini
overlays=uart2-m0 dwc3-peripheral
```

extlinux 的 `fdtoverlays` 一行包含：

```text
/boot/dtb/rockchip/overlay/rk3568-uart2-m0.dtbo /boot/dtb/rockchip/overlay/rk3568-dwc3-peripheral.dtbo
```

若新装的**相同版本/启动布局**尚未包含它们，先备份，再用编辑器追加，保留其他 overlay 和启动参数：

```bash
sudo cp -p /boot/armbianEnv.txt /boot/armbianEnv.txt.before-microduck
sudo cp -p /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.before-microduck
sudo nano /boot/armbianEnv.txt
sudo nano /boot/extlinux/extlinux.conf
```

不要把 DTBO 路径单独当命令执行。在 extlinux 中它们属于同一行 `fdtoverlays ...`。如果目录、overlay 文件或启动布局不同，停止套用本节模板，按该系统的设备树配置方式处理。

### 8.2 J5 I²C 修复

当前板子原来启用的 I²C3 是 M1 引脚组，而 J5 用 M0。修复工具 1.1.0 只针对上述基线：

```bash
cd ~/radxa_j5_i2c_fix
python3 enable_j5.py --version
python3 enable_j5.py
```

第二条只做预演：应显示 I²C3 → GPIO1_A0/A1、400 kHz，并预览启动文件差异。**预演通过后**：

```bash
sudo python3 enable_j5.py --apply
sudo reboot
```

应用会备份启动配置，并打印回退路径。它会停用 FUSB302 的 USB-C PD 协商，适用于当前 HAT 提供 5 V 的接法。应用失败时先处理输出，不继续重启。

重启后重新 SSH，再执行：

```bash
sudo sh -c 'grep -H -E "fe5c0000|i2c3|pin (32|33|109|110) " /sys/kernel/debug/pinctrl/*/pinmux-pins'
cd ~/deploy_hd1910
./run.sh doctor
```

成功时 pin 32/33 属于 `i2c3m0-xfer`；`doctor` 能发现 LSM6DSV16X：地址通常 0x6A 或 0x6B，WHO_AM_I 为 112 / 0x70。本机之前为 I²C 3、0x6A。只出现 `/dev/i2c-3` 不能证明 J5 正确复用或传感器已经应答。

## 9. 设置并核对每个舵机 ID

**新机 duck1 编号（2026-10-31 用户确认）：**左腿从髋到脚为 1–5；颈俯仰、头俯仰、头偏航、头侧倾为 6–9；嘴为 10；右腿从髋到脚为 11–15。身体策略仍是 14 个关节，不含嘴。专用编号表为 [`servo_ids.duck1.json`](deploy_hd1910/servo_ids.duck1.json)，在 Radxa 导入时使用：

```bash
cd ~/deploy_hd1910
./run.sh map-ids --profile servo_ids.duck1.json
```

这只保存软件映射，不修改硬件 ID；ID 变更会使原零位标定失效，IMU 配置保留。嘴部 10 号独立标定开闭端点，完成前不启用。`supported_home.py` 应使用按配置读取 14 个身体 ID 的版本。

**duck1 本轮选择直接 HOME 标定：**摆放依据与命令见[直接 HOME 的机械依据](deploy_hd1910/docs/姿态与零位.md#duck1直接摆-home-的机械依据)。使用 `calibrate --pose home`，之后使用不带 `--from-pose` 的 `home-plan/home`；现有 `supported_home.py` 仍只支持 straight/assembly，不能因为已修正 ID 限制就直接用于 HOME 标定记录。下文第10节直腿路线保留作其他机器的参考。

**下面的右腿 10–14、嘴 15 编号表，以及第 10.1 节的旧机方向命令属于旧机记录，不能直接用于 duck1。** 新机应按关节名读取 `robot.json` 对应的实际 ID，逐关节实测方向；不能用关节序号直接充当舵机 ID。

新舵机先用第 0 节 **Windows FD + 对应 USB 总线转接板**逐颗编号；软件 `map-ids` 不会改舵机硬件 ID：

1. 一次只接一颗舵机，转接板与外部舵机电源按说明连接并共地。
2. Windows 设备管理器确认 COM；解压 FD，选 COM、1000000 baud，打开并搜索。搜不到时核对该舵机出厂波特率。
3. 读取型号/寄存器，按下表写唯一 ID；本部署用 `hls_2`、波特率 1000000、位置模式 4、分辨率 1、Ki=0。按对应型号资料写入并读回。
4. 断电重连、重新搜索，确认新 ID 和参数仍在，贴关节标签；重复 14 颗后再接总线。舵盘安装时保留关节行程，核对单圈硬件限位。

FD 没有对应型号寄存器定义时先取匹配资料，不套用 XL330/其他 STS 的写寄存器命令。

| ID | 关节 | 配置名 |
| --- | --- | --- |
| 1 | 左髋偏航，整腿转向 | `left_hip_yaw` |
| 2 | 左髋侧倾，整腿左右摆 | `left_hip_roll` |
| 3 | 左髋俯仰，大腿前后摆 | `left_hip_pitch` |
| 4 | 左膝，小腿弯伸 | `left_knee` |
| 5 | 左踝，脚掌前后翘 | `left_ankle` |
| 6 | 颈俯仰 | `neck_pitch` |
| 7 | 头俯仰 | `head_pitch` |
| 8 | 头偏航 | `head_yaw` |
| 9 | 头侧倾 | `head_roll` |
| 10 | 右髋偏航 | `right_hip_yaw` |
| 11 | 右髋侧倾 | `right_hip_roll` |
| 12 | 右髋俯仰 | `right_hip_pitch` |
| 13 | 右膝 | `right_knee` |
| 14 | 右踝 | `right_ankle` |
| 15（可选） | 嘴 | 不属于 14 关节行走策略 |

**先让串口只归一个程序使用：**

```bash
systemctl is-active microduck-xbox.service
systemctl is-active robotd
sudo fuser -v /dev/ttyS2
```

若已有 Xbox 服务，先托稳、按 B 后停止；若官方 `robotd` 占用串口，也停止它：

```bash
sudo systemctl stop microduck-xbox.service
sudo systemctl stop robotd
```

不存在这些服务时无需执行对应停止命令。网页前台程序用 Ctrl+C 退出。若 `serial-getty@ttyS2` 占用同一 UART，关闭该串口登录服务：

```bash
sudo systemctl disable --now serial-getty@ttyS2.service
```

读取确认：

```bash
cd ~/deploy_hd1910
./run.sh scan --all --baudrate 1000000
./run.sh inspect
```

`scan --all` 查询 ID 0–253，不改 ID；`inspect` 读取身份、模式、位置、电压等，保存 `logs/servos.json`。应能稳定读到所有身体 ID 1–14。

## 10. 关节方向与直腿参考标定

### 10.1 卸力后核对方向

托住躯干和头部、让双脚离地，再执行：

```bash
cd ~/deploy_hd1910
./run.sh relax
./run.sh pose
./run.sh watch --seconds 30
```

`relax` 关闭身体 14 颗扭矩；`pose` 显示模型姿态与正转轴说明；`watch` 只读刻度。轻轻转动单一关节，确认编号和模型正方向对应刻度增加还是减少。单独看一颗：

```bash
./run.sh watch --id 5 --seconds 20
```

你这台此前实测：1–4、6–13 的方向为 −1，5 和 14 为 +1。**以下命令仅用于同样装配并已核对的首次配置；已有标定不要重复执行，因为 `map-joint` 会使零位标定失效。**

```bash
./run.sh map-joint --joint left_hip_yaw --id 1 --direction -1
./run.sh map-joint --joint left_hip_roll --id 2 --direction -1
./run.sh map-joint --joint left_hip_pitch --id 3 --direction -1
./run.sh map-joint --joint left_knee --id 4 --direction -1
./run.sh map-joint --joint left_ankle --id 5 --direction 1
./run.sh map-joint --joint neck_pitch --id 6 --direction -1
./run.sh map-joint --joint head_pitch --id 7 --direction -1
./run.sh map-joint --joint head_yaw --id 8 --direction -1
./run.sh map-joint --joint head_roll --id 9 --direction -1
./run.sh map-joint --joint right_hip_yaw --id 10 --direction -1
./run.sh map-joint --joint right_hip_roll --id 11 --direction -1
./run.sh map-joint --joint right_hip_pitch --id 12 --direction -1
./run.sh map-joint --joint right_knee --id 13 --direction -1
./run.sh map-joint --joint right_ankle --id 14 --direction 1
```

### 10.2 手工摆直腿参考姿态

扭矩关闭、外部支撑下摆放：

1. 躯干固定，髋偏航的转轴竖直；两腿朝机身前方，不内八/外八。
2. 髋侧倾摆正，腿不向侧面倾斜。
3. 从正侧面看，髋俯仰轴心 H、膝轴心 K、踝轴心 A 上下竖直对齐。
4. 保持轴心对齐，脚部参考平面水平。
5. 颈根部与头俯仰轴心上下对齐；头平视，不侧歪。

确认的是转轴中心在侧视图中的投影，不是舵机外壳边缘；左右方向允许有模型本身的结构错位。

在本项目模型中，这个直腿姿态的髋俯仰与膝角：左侧约 **−58.411873°**，右侧约 **+58.411873°**，其他参考角为 0。这一关系要求真实连杆与模型几何匹配。详细图解见 [直腿标定到 HOME](deploy_hd1910/docs/直腿标定到HOME.md)。

保持不动，记录标定：

```bash
./run.sh calibrate --pose straight
./run.sh home-plan --from-pose straight --move-seconds 8
./run.sh watch --angles --seconds 20
```

- `calibrate` 读取多组当前位置，求软件零位，保存 `robot.json`；不写舵机 EEPROM。
- `home-plan` 只读并列出 HOME 目标和角度差，不执行运动。
- `watch --angles` 显示模型角与 HOME 偏差。

当前回 HOME 包装脚本需要有效的**直腿参考标定**。只在网页里“保存姿态”不会自动生成这项部署标定。

### 10.3 标定数学关系

设当前刻度为 c、软件零位为 z、方向为 s=±1：

```text
q = s × (c − z) × 2π / 4096
z = c参考 − s × q参考 × 4096 / 2π
cHOME = z + s × qHOME × 4096 / 2π
```

因此，无需知道舵盘安装时的刻度，也不要求把所有读数改为 2048。2048 是编码器中点，不等于各个关节的模型零位；HOME 更不是“全回 2048”。

若真实读数超出 4095，应先核对模式、分辨率、编码器中心或多圈状态，不能随便 `%4096` 后发运动。本机先前做过的硬件重定中心不作为新机器必做步骤。

### 10.4 记录 HOME 目标位置：完整命令

第 10.2 节 `calibrate --pose straight` 已把直腿实测刻度、软件零位和计算出的 HOME 刻度一起写入 **`robot.json`**。保持这份直腿标定；**不要再执行 `calibrate --pose home`**，否则会覆盖现有包装脚本需要的直腿记录。

**Radxa**：保存带时间戳的配置及 HOME 表，每行含关节、ID、方向、零位、直腿刻度、HOME 目标刻度：

```bash
cd ~/deploy_hd1910
mkdir -p logs
cp -p robot.json "robot.json.home.$(date +%Y%m%d_%H%M%S).json"
.venv/bin/python - <<'PYHOME'
import csv, json
from datetime import datetime
from pathlib import Path
cfg = json.loads(Path("robot.json").read_text())
cal = cfg["calibration"]
assert cal["pose"] == "straight", "先完成直腿标定"
path = Path("logs") / ("home-targets-" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
with path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["joint", "id", "direction", "zero_tick", "straight_tick", "home_tick"])
    for j, ref, home in zip(cfg["joints"], cal["captured_ticks"], cal["home_ticks"]):
        w.writerow([j["name"], j["id"], j["direction"], j["zero_tick"], ref, home])
print(path.resolve())
print(path.read_text())
PYHOME
```

### 10.5 实际回 HOME，并记录实际位置

关闭网页及其他串口程序，托住躯干/头部、双脚悬空。下面第一组先卸力、再只读预览；确认目标与机械活动空间合理再执行第二组：

```bash
cd ~/deploy_hd1910
./run.sh relax
.venv/bin/python scripts/supported_home.py --target home --soft-gains
```

**第一次从已摆好的直腿参考回位**，沿用附加反馈检查：

```bash
.venv/bin/python scripts/supported_home.py --target home --soft-gains --move
```

该路径要求当前接近直腿参考（约 ±5°），计划 12 秒回 HOME，再约 1 秒检查到位；成功后保持扭矩。若因当前姿态不接近直腿而拒绝，重新卸力、托住并摆回已标定的参考姿态，勿以重标零位消除报错。

后续从其他姿态进行受支撑恢复，现有日常包装器采用下列路径：

```bash
./run.sh relax
.venv/bin/python scripts/supported_home.py \
  --target home --soft-gains --move --log-only-feedback
```

该路径最低约 6 秒、随位移延长；`--log-only-feedback` 将附加电流/温度/电压/跟踪阈值改成仅记录，设备错误、数据时效和硬件范围仍检查。首次标定优先用前面的直腿路径。

到位后先保持支撑，记录**实际**位置：

```bash
./run.sh watch --angles --seconds 5 \
  | tee "logs/home-feedback-$(date +%Y%m%d_%H%M%S).txt"
./run.sh inspect
cp -p logs/servos.json "logs/home-feedback-$(date +%Y%m%d_%H%M%S).json"
```

检查 `supported_home_*.json` 的 `target_reached`、`final_errors_ticks`，以及文本中的 `raw`、`q`、`HOME_delta`；14 颗都应有完整反馈，无 `error`。目标表记录“应到哪里”，反馈记录“实际到了哪里”。完成后托稳再 `./run.sh relax`。

备份到**电脑**，IP 使用第 4 节的当前值：

```bash
mkdir -p ~/microduck/backups
RADXA_BACKUP_DIR="$HOME/microduck/backups/radxa-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RADXA_BACKUP_DIR"
scp wmh@"$RADXA_IP":~/deploy_hd1910/robot.json "$RADXA_BACKUP_DIR/"
scp "wmh@$RADXA_IP:~/deploy_hd1910/logs/home-*" "$RADXA_BACKUP_DIR/"
```

其中最重要的是 `robot.json`：仅属于这一台机器。HOME CSV/反馈用于复查，不能替代它。

## 11. 可选网页调试台

停止其他串口程序，**Radxa**执行：

```bash
cd ~/servo_web_hd1910
bash install.sh
bash run.sh
```

电脑浏览器访问 `http://主板当前IP:8080`；默认查 1–14，要看嘴部改为 `bash run.sh --ids 1-15`。

本适配版支持读数、保存姿态、方向记录、当前位置扭矩保持/卸力；滑块运动、EEPROM/改 ID 等功能停用。姿态保存在 `~/servo_web_hd1910/poses/*.json`，不替代部署 `robot.json` 标定。用完 Ctrl+C 退出，再运行命令行/行走程序，避免抢占串口。

## 12. 验证 IMU 方向与只读策略

**Radxa**，先让躯干静止约 2 秒再观察：

```bash
cd ~/deploy_hd1910
./run.sh doctor
./run.sh imu --seconds 30
```

机身坐标：**X 前、Y 左、Z 上**。直立静止时重力约 `[0,0,-1]`、角速度约 `[0,0,0]` rad/s。移动整个躯干验证：

| 动作 | 正确趋势 |
| --- | --- |
| 前倾/前部下压 | gravity_x > 0；转动时 gyro_y > 0 |
| 后倾/前部抬起 | gravity_x < 0；转动时 gyro_y < 0 |
| 左倾/左肩低 | gravity_y > 0；转动时 gyro_x < 0 |
| 右倾/右肩低 | gravity_y < 0；转动时 gyro_x > 0 |
| 向左转 | 转动时 gyro_z > 0 |

倾斜后保持不动，角速度应重新近零，重力仍表示倾斜。**只有实际确认传感器三轴与机身同向，才执行这个示例**：

```bash
./run.sh imu-map --axes='+x,+y,+z'
./run.sh imu --seconds 30
./run.sh check --seconds 10
```

实际不同就换正确映射；如 `+z,+y,-x` 表示机身 X/Y/Z 分别取传感器 Z/Y/负 X。任意斜角安装需求旋转矩阵，不能简单交换轴。`imu-map` 会保存并标记验证完成；`check` 读传感器并运行模型，不发运动。

后面的 WALK/Menu 会重新静止校偏约 **3 秒**，保持机身不动；校偏不修改关节零位，也不能把任意倾斜姿态当成竖直。失败时看 IMU 原始值和日志。

## 13. 选择现成模型

| 模型（位于 `~/deploy_hd1910/models/`） | 含义 |
| --- | --- |
| `hd1910_flat.onnx` / `hd1910_backlash.onnx` | 最初本地平地/齿隙训练基线 |
| `hd1910_friction_flat_5000.onnx` / `hd1910_friction_backlash_5000.onnx` | 增强摩擦随机化的候选 |
| `hd1910_backlash_30000.onnx` | 后来提供的齿隙 PT 导出候选 |

轮数不代表质量排名；模型需要与 61 维观测、14 关节顺序、HOME 和动作缩放匹配。**Radxa** 选择器只校验模型并修改模型路径，保留零位，不驱动硬件：

```bash
cd ~/deploy_hd1910
ls -lh models/*.onnx
.venv/bin/python scripts/select_policy_model.py models/hd1910_backlash_30000.onnx 0 0.3
./run.sh self-test
```

后两个参数供入口校验：`0` 表示无限时长，`0.3` 表示预期手柄速度上限；实际运动时仍需给运行命令传同样参数。只回 HOME 见第 10.5 节。

## 14. 不带遥控器的定速真机命令

先选择需要的模型，再运行：

```bash
cd ~/deploy_hd1910
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/home_then_policy.py \
  --move --seconds 10 --vx 0.1 --ramp-seconds 3
```

参数解释：

| 参数 | 含义 |
| --- | --- |
| `--move` | 本次实际执行动作；没有它只预览 |
| `--seconds 10` | 首次策略运行 10 秒；验证后再延长，`0` 为持续运行 |
| `--vx 0.1` | 固定向前 0.1 m/s 目标；改成 0.2 即 0.2 m/s |
| `--ramp-seconds 3` | 用 3 秒把速度命令从零升到指定值 |
| 两个线程变量 | 限制数学库线程数量，减少 Radxa CPU 争抢 |

实际操作顺序：

1. 外部托住躯干与头部，双脚悬空，执行命令。
2. 回 HOME 完成、保持上力后，将双脚放在摩擦足够的平整地面，先保持不动。
3. 终端提示时输入 `walk` 或 `WALK`，回车。
4. 程序先约 3 秒 IMU 校偏，再约 1 秒对齐 HOME，之后运行策略。
5. Ctrl+C 或 `q` 回车停止；正常结束或停止后会卸力，需要接住机器人。

Xbox 包装入口会加 `--log-only-target-angles`，使目标角范围/跳变只记录、不拦截或裁剪；实际反馈检查仍存在。以上定速首次试机命令保留目标检查；若触发停止，先排查标定/模型和日志。

## 15. Xbox 配对与运行

### 15.1 配对和输入测试

**Radxa**：

```bash
sudo systemctl enable --now bluetooth
bluetoothctl
```

以下在 **bluetoothctl 提示符**输入；手柄开机后长按顶部配对键使灯快闪，扫描到 Xbox 的 MAC 后替换占位地址：

```text
power on
agent NoInputNoOutput
default-agent
scan on
connect XX:XX:XX:XX:XX:XX
pair XX:XX:XX:XX:XX:XX
trust XX:XX:XX:XX:XX:XX
info XX:XX:XX:XX:XX:XX
scan off
quit
```

本机 BLE 手柄采用先连接再配对；其他型号依实际配对提示。确认 Paired/Bonded/Trusted/Connected，再读输入；不要只看灯。[BlueZ 命令说明](https://manpages.debian.org/trixie/bluez/bluetoothctl.1.en.html)

```bash
cd ~/deploy_hd1910
.venv/bin/python scripts/test_xbox.py --list
.venv/bin/python scripts/test_xbox.py --seconds 30
```

这两条不访问舵机。拨动摇杆、按 A，B 或 Ctrl+C 退出；默认自动找设备，不写死 event 编号。配对失败查 `bluetoothctl show`、`bluetoothctl devices`、`journalctl -u bluetooth -n 100 --no-pager`。AIC8800 特殊问题见 [原版配对说明](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/pair-a-gamepad.md)；不要无依据照搬别台的 Privacy 配置。

### 15.2 手柄控制

先托稳、退出网页和手动程序；若已有自启服务，按 B 后停止它。**Radxa**：

```bash
sudo systemctl stop microduck-xbox.service
cd ~/deploy_hd1910
bash xbox_model_hd1910.sh models/hd1910_backlash_30000.onnx 0 0.3
```

新装尚无服务时跳过 stop 那行。参数顺序为 **模型、秒数、最大前后速度**；`0` 为持续运行，可先用 `10` 秒验证。该包装器会把策略目标角/跳变设成仅记录；实际反馈检查仍保留。

| 按键 | 动作 |
| --- | --- |
| A | 托住躯干/头部、双脚悬空时按，开始回 HOME；HOME 保持阶段可再按 A 回位 |
| Menu / ≡ | HOME 就绪、双脚落地、摇杆居中、机身静止后按；约 3 秒校偏和约 1 秒对齐后开始策略 |
| B | 请求停止、卸力，复位到等待新的 A；提前接住机器人 |
| 左摇杆前后/左右 | 前后速度 vx / 侧移 vy |
| 右摇杆左右 | 转向 wz |
| 松开摇杆 | 指令回零，策略仍平衡，不卸力 |
| X | 嘴动作，仅嘴部已配置/标定时可用 |
| Y | 策略阶段坐下/站起，需匹配 `sitstand_model` |
| RB | 俯身啄取，需另配 `ground_pick_model` 和坐站模型；新配置默认未启用 |

默认前进/后退 **0.3 m/s**、侧移 **±0.3 m/s**、转向 **±1 rad/s**、死区约 12%；旧文档后退 0.2 已过时。提前按 Menu 不会排队启动，每阶段要求新按键和摇杆居中；HOME 就绪容差约 8.6°，不代表标定精度。

## 16. 开机后只靠手柄启动，不用电脑发命令

首次安装服务前先停止动作、托住机器人。**Radxa：**

```bash
cd ~/deploy_hd1910
sudo python3 scripts/install_xbox_service.py --user "$(whoami)"
```

这会安装并启用 `microduck-xbox.service`；下面配置好再启动。服务等待手柄和新的 A 键，不自动运动。

已有 `/etc/default/microduck-xbox` 时安装器保留它。要明确改为 30000 轮模型，先按 B 并停止服务：

```bash
sudo systemctl stop microduck-xbox.service
sudo cp -p /etc/default/microduck-xbox \
  "/etc/default/microduck-xbox.backup.$(date +%Y%m%d_%H%M%S)"
sudo tee /etc/default/microduck-xbox >/dev/null <<'XBOX'
MICRODUCK_VARIANT=backlash
MICRODUCK_MODEL=models/hd1910_backlash_30000.onnx
MICRODUCK_SECONDS=0
MICRODUCK_MAX_VX=0.3
XBOX
```

`MICRODUCK_MODEL` 优先于 variant 默认模型；`SECONDS=0` 去掉运行时长限制。随后：

```bash
sudo systemctl start microduck-xbox.service
systemctl status microduck-xbox.service --no-pager
journalctl -u microduck-xbox.service -n 50 --no-pager
```

以后日常流程就是：**给主板/舵机供电 → 手柄开机并连接 → 托稳、脚悬空、按 A 回 HOME → 落脚静止 → 按 Menu 校偏并开始策略 → 用摇杆控制 → 按 B 卸力。**

查看实时日志：

```bash
journalctl -u microduck-xbox.service -f
```

这里 Ctrl+C 只是停止看日志，不停止机器人。要取消开机监听：

```bash
sudo systemctl disable --now microduck-xbox.service
```

停止、重启或禁用正在控制的服务可能卸力，先托稳、按 B。蓝牙已配对、模型已在本机时，整个手柄运行流程不需要手机热点。

## 17. 在训练电脑复现训练

### 17.1 安装 NVIDIA 驱动、uv 和锁定环境

**仅训练电脑执行**，先看显卡驱动：

```bash
nvidia-smi
```

已经正常显示显卡则跳过安装；未装驱动时用 Ubuntu 工具选择适配驱动。Secure Boot 开启时按安装提示完成 MOK 注册，重启后再次确认 `nvidia-smi`：[官方步骤](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)。

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

重启后打开**电脑终端**：

```bash
sudo apt-get install -y libegl1 libgl1 libglfw3 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.12
cd ~/microduck/microduck_rl
uv sync --frozen --python 3.12
cd ~/microduck
microduck_rl/.venv/bin/python - <<'PYGPU'
import torch, mujoco
from importlib.metadata import version
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("MuJoCo:", mujoco.__version__, "mjlab:", version("mjlab"), "Warp:", version("warp-lang"))
assert torch.cuda.is_available(), "CUDA 不可用；先检查驱动，不启动正式训练"
x = torch.ones(4, device="cuda")
print(torch.cuda.get_device_name(0), (x + x).cpu())
PYGPU
python3 scripts/preflight.py --reserve-gib 3
```

本锁文件为 **Python 3.12 / Torch 2.9.1（CUDA 12.8）/ MuJoCo 3.10.0 / mjlab 1.3.0 / Warp 1.12.0**。uv 会下载依赖；通常无需另装完整 CUDA Toolkit、Conda 或单独手装 MuJoCo。`nvidia-smi` 的 CUDA 字样代表驱动支持能力，不是项目 Torch 的运行库版本，以 Python 检查为准。

环境约 8 GB，安装缓存与训练检查点另外占空间，建议预留 30 GB 以上；`preflight` 的 3 GiB 仅是运行保留量。新终端找不到 uv 时重开终端或重执行 PATH 那行。

### 17.2 从头训练两版 HD-1910

先做短训练确认配置和导出能跑：

```bash
cd ~/microduck
python3 scripts/train_pair.py --robot hd1910 --phase smoke \
  --num-envs 64 --iterations 5 --finalize
```

再正式训练：

```bash
python3 scripts/train_pair.py --robot hd1910 --phase train \
  --num-envs 2048 --iterations 4000 --finalize
```

- 两版顺序执行，共用 GPU 和 `microduck_rl/.venv`，源码选择 `microduck_rl_hd1910/src`。
- `--variant flat` 或 `--variant backlash` 可只训一版。
- 4000 轮是 PPO 更新轮数，不是 4000 次完整走路；每轮各环境采集一段控制步。
- 每 250 轮保存检查点；编号从零开始，所以 4000 轮常见最终文件 `model_3999.pt`。
- `--finalize` 完成后导出 ONNX 并做评估，产物在 `runs/时间_hd1910_train/`。
- PT 与完整参数在 `microduck_rl_hd1910/logs/rsl_rl/<实验>/<运行>/`，其中 `params/env.yaml`、`params/agent.yaml` 记录配置；监督状态 `status.json` 和标准输出 `flat.log`、`backlash.log` 在 `runs/<时间>_hd1910_train/`。
- 监督脚本在训练、导出、评估中持续查磁盘，空闲低于 2.5 GiB 停当前阶段。

原版 XL330 的复现将 `--robot hd1910` 改为 `--robot xl330`。HD-1910 是 BAM 1910 动力学参数适配，不是只换文件名；既有电压、延迟随机化与齿隙假设尚不能等同于实测舵机辨识。

### 17.3 查看奖励曲线

```bash
cd ~/microduck
microduck_rl/.venv/bin/tensorboard \
  --logdir microduck_rl_hd1910/logs/rsl_rl --host 127.0.0.1 --port 6006
```

在训练电脑浏览器打开 `http://127.0.0.1:6006`。总奖励高不一定速度跟踪好；同时看实际 vx、目标误差、倒地率和视频。

### 17.4 指定检查点接续训练

先找实际目录：

```bash
cd ~/microduck
rg --files microduck_rl_hd1910/logs/rsl_rl -g 'model_*.pt'
```

以下示例续训平地版。把 `实际运行目录名` 和检查点换成上面真实输出：

```bash
cd ~/microduck/microduck_rl_hd1910
MUJOCO_GL=egl WANDB_MODE=disabled PYTHONPATH="$PWD/src" \
../microduck_rl/.venv/bin/train \
  Mjlab-Velocity-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 2048 --agent.logger tensorboard \
  --agent.experiment-name local_hd1910_flat_train \
  --agent.run-name continued --agent.resume True \
  --agent.load-run '^实际运行目录名$' --agent.load-checkpoint model_3999.pt \
  --agent.max-iterations 1000 --agent.save-interval 250 \
  --agent.upload-model False
```

这里的 cwd 很重要，日志按项目目录寻找；`1000` 是追加的更新次数，不是绝对终止编号。齿隙版同时改任务为 `Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910`、实验名为对应 backlash。

这条直接训练命令没有外层持续磁盘监督，应另行监控；从头训练优先用上面的监督脚本。

### 17.5 长时间运行和续训区别

首次训练在终端前台运行并保持电脑开机；关闭终端可能中断。需要断开终端后继续，可先安装 `tmux`：

```bash
sudo apt-get install -y tmux
tmux new -s microduck-train
```

在 tmux 内执行第 17.2 节正式训练；Ctrl+B 后按 D 脱离。以后 `tmux attach -t microduck-train` 返回，Ctrl+C 请求结束。睡眠/重启会中断训练，重启后需用第 17.4 节从已存检查点继续。

本地的 `train_hd1910_until_stopped.py` / `train_hd1910_robust.py` 另依赖历史运行目录或外部 PT，不能作为新电脑默认入口。本文提供的从头训练能复现基础两任务，不能承诺重现外部 30000/40000 轮模型的完整训练历史。

## 18. 导出 ONNX、检查一致性、仿真速度跟踪

### 18.1 导出

**电脑**，以齿隙任务为例。将 `CHECKPOINT` 改成真实绝对路径：

```bash
cd ~/microduck
read -rp '齿隙模型 PT 的绝对路径：' CHECKPOINT
test -f "$CHECKPOINT"
PYTHONPATH="$PWD/microduck_rl_hd1910/src" \
microduck_rl/.venv/bin/python microduck_rl_hd1910/scripts/export.py \
  Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910 \
  --checkpoint-file "$CHECKPOINT" \
  --onnx-file deploy_hd1910/models/hd1910_backlash_new.onnx \
  --num-envs 1 --device cuda:0
```

训练任务、观测顺序、动作尺度和 HOME 必须匹配。导出需包含观测归一化，不能只把 actor 单独导出。导出能加载不证明任意外部检查点就是这个训练环境。

### 18.2 验证 PT 与 ONNX 数值一致性

使用 18.1 同一个终端设置的 `CHECKPOINT`；换终端时重新赋值为同一个真实路径。

```bash
cd ~/microduck
PYTHONPATH="$PWD/microduck_rl_hd1910/src" \
microduck_rl/.venv/bin/python scripts/verify_hd1910_export.py \
  --task Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910 \
  --checkpoint "$CHECKPOINT" \
  --onnx deploy_hd1910/models/hd1910_backlash_new.onnx \
  --output deploy_hd1910/models/hd1910_backlash_new.parity.json
```

它比较相同输入下 PT 与 ONNX 输出，帮助排除导出差异；不能证明 sim-to-real 动力学正确。

### 18.3 MuJoCo 跟踪多种速度并录视频

```bash
cd ~/microduck
PYTHONPATH="$PWD/microduck_rl_hd1910/src" \
microduck_rl/.venv/bin/python scripts/evaluate.py \
  --task Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910 \
  --checkpoint "$CHECKPOINT" \
  --onnx deploy_hd1910/models/hd1910_backlash_new.onnx \
  --output runs/manual_backlash_speed_eval \
  --speeds -0.1 0 0.1 0.2 0.3 --num-envs 64 --seconds 10 --trace --video
```

在相同 HD-1910 仿真任务中测试后退、站立和不同前进速度，输出 `evaluation.json`、轨迹和视频。脚本关掉外部推扰用于固定指令比较，其余随机化以实际任务配置为准；不能将该结果称为真机测试。

### 18.4 把新模型传到主板

**电脑：**

```bash
cd ~/microduck
scp deploy_hd1910/models/hd1910_backlash_new.onnx \
  wmh@"$RADXA_IP":~/deploy_hd1910/models/
```

**Radxa，停原服务后：**

```bash
cd ~/deploy_hd1910
bash xbox_model_hd1910.sh models/hd1910_backlash_new.onnx 0 0.3
```

通过选择器保存模型路径，不覆盖本机零位。若用开机服务，要同时更新第 16 节的 `MICRODUCK_MODEL`；只运行一次手动命令不会自动修改服务模型。

## 19. 日志、故障定位、备份与关机

### 19.1 快速诊断命令

```bash
cd ~/deploy_hd1910
./run.sh doctor
.venv/bin/python -m json.tool robot.json
cat /etc/default/microduck-xbox
ls -lt logs | head
sudo fuser -v /dev/ttyS2
journalctl -u microduck-xbox.service -n 100 --no-pager
cat /sys/class/thermal/thermal_zone0/temp
df -h ~
free -h
```

最后的温度值通常是毫摄氏度，除以 1000 得到 °C；`df` 看磁盘，`free` 看内存。

| 现象/错误 | 首先解释什么 |
| --- | --- |
| `ensurepip is not available` | 系统缺 venv 包，补 `python3-venv` 再运行安装 |
| `Permission denied: /dev/i2c-*` | 设备权限/会话组权限问题，尚不能判定 IMU 接错 |
| `ServoTimeout` | 舵机回复没在期限内到齐，检查总线/供电/重复 ID/串口争用 |
| IMU 初始化或静止校偏失败 | 策略可能尚未进入执行阶段，先检查原始 IMU 和启动日志 |
| ONNX GPU discovery warning | 当前使用 CPU 推理，单独这个警告不等于运行失败 |
| `target_reached:false` | 回位插值完了但实际偏差超容差，要看是哪颗、偏多少 |
| 目标角/跳变超阈值 | 原来的软件目标检查截停；仅记录模式下不会由这两项停机 |
| 测得关节超模型范围 | 实际反馈检查，需查标定、方向、机械几何或真实动作 |
| 舵机电压低/CC | 动作时供电压降或限流；看各舵机、接头和电源输出 |
| 动一下后站着 | 可能策略输出趋于静止，也可能执行器未跟踪；用目标与实际反馈区分 |
| 脚打滑、走几步倒 | 地面摩擦、重心/配重、关节标定、动作跟踪和策略训练都需核对 |

当前模板的反馈电压范围是 **4.0–8.4 V**、温度上限 60°C、倾斜上限 35°；旧日志的 6.5 V 或 5 V 下限不代表当前配置。**以主板实际 `robot.json` 为准。** 软件容许低电压不保证舵机还具备训练假设的速度和力矩。

`--log-only-target-angles` 没有取消实际关节范围、0–4095 刻度范围、供电/温度、通信、IMU 数据时效和控制周期检查。停止后“软掉”是程序关闭扭矩的结果，不等于电源物理被切断。

新版对 IMU 启动失败有受支撑恢复/保持流程，避免旧版那样所有初始化异常都立刻卸力；具体行为仍看当前日志，不以旧 traceback 推断现在已经跑过策略。

### 19.2 保存复现材料

每台至少保存：第 10.4–10.5 节的 `robot.json`、HOME 表/反馈，实际 ONNX、源码包及 SHA256；训练另存 PT 与 `params/env.yaml`、`params/agent.yaml`。主板服务和设备树配置可一并打包，再由电脑取走：

```bash
# Radxa；在已安装第 16 节服务的机器执行
cd ~
sudo tar -czf microduck-system-config.tar.gz \
  /etc/default/microduck-xbox /etc/systemd/system/microduck-xbox.service \
  /etc/udev/rules.d/99-microduck-i2c.rules \
  /boot/armbianEnv.txt /boot/extlinux/extlinux.conf /boot/overlay-user
sudo chown "$(id -u):$(id -g)" microduck-system-config.tar.gz
```

```bash
# 电脑；RADXA_BACKUP_DIR 使用第 10.5 节创建的备份目录
scp wmh@"$RADXA_IP":~/microduck-system-config.tar.gz "$RADXA_BACKUP_DIR/"
```

新终端需重新设置 `RADXA_IP` 和备份目录。配置备份用于查阅/同机恢复，不直接覆盖另一台机器。

### 19.3 停机

先扶稳、按 B；有自启服务就先停止，手动程序/网页退出后再卸力：

```bash
sudo systemctl stop microduck-xbox.service
cd ~/deploy_hd1910
./run.sh relax
sudo poweroff
```

等待 Linux 关机后再切断电源。`poweroff` 关闭系统；直接断电可能使尚未落盘的配置或日志丢失。

## 20. 日常操作与验收

IMU 中途读失败时，见 [IMU供电诊断](deploy_hd1910/docs/IMU供电诊断.md)：包含直接执行的日志分析命令、3.3 V供电关系及实际测量方法。

日常只需：**供电 → 手柄开机 → 托稳悬空按 A → HOME 就绪后落地静止 → Menu → 摇杆 → 托稳按 B → 关机。**

一次复现完成应留下：系统/源码版本及校验文件、14 颗舵机身份与方向表、`robot.json`、HOME 目标/反馈、IMU 方向核验、模型及训练/评估结果。通信、自检和回 HOME 成功不等于稳定行走；机械、供电、摩擦、重心和策略还需实机验收。
