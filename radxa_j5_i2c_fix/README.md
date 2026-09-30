# Radxa Zero 3W：启用 HAT J5 的 I²C

修复工具版本：**1.1.0**。

旧版把 `rockchip,pins` 的排列顺序写死为 A0、A1，但 Armbian/Rockchip 6.1 源码实际先列
A1（SCL）、再列 A0（SDA），因此只读预演会误报“M0引脚不是GPIO1_A0/A1”。
1.1.0 改为验证两颗引脚及其功能的集合，接受两种排列，仍拒绝重复、额外、错误引脚和错误功能。
此次修正不改变官方 overlay 内容，也不交换实物的 SDA/SCL 接线。
来源：[Armbian 6.1 pinctrl](https://github.com/armbian/linux-rockchip/blob/rk-6.1-rkr5.1/arch/arm64/boot/dts/rockchip/rk3568-pinctrl.dtsi)。

适用于本次已核对的 Armbian 26.2.1 / vendor 6.1.115 配置：extlinux 单一 `armbian`
启动项，原有 overlay 为 `uart2-m0` 和 `dwc3-peripheral`。

用户提供的实时状态为 GPIO1_A0/A1 未分配，I2C3 使用 M1（GPIO3_B5/B6）。
HAT J5 需要将 I2C3 切换到 M0（GPIO1_A0/A1，主板物理针 3/5）。
本包采用官方 `i2c3-pihat.dts`，原样保留，来源和许可见 [NOTICE](NOTICE.md)。

## 作用与影响

- I2C3 改为 J5 使用，400 kHz。
- 停用 FUSB302 USB-C PD 控制器的驱动/协商。此机器人由电池经 HAT 提供 5 V，沿用该供电方式。
- 新 overlay 保存到 `/boot/overlay-user/microduck-j5-i2c.dtbo`。
- 在 extlinux 原 `fdtoverlays` 末尾添加该路径，同时在 `armbianEnv.txt` 设置
  `user_overlays=microduck-j5-i2c`，原有串口、USB、内核、根分区等设置保留。
- 不安装整套官方软件，不改舵机、模型和标定配置，不自动重启。

默认运行只检查：编译 overlay，按原顺序合并现有两个 overlay 和新增 overlay 到临时 DTB，
验证正确的 M0 引脚、400 kHz、FUSB302 disabled。`--apply` 也会重新完成这些检查。
遇到不符的启动布局、不同内容的同名 overlay 或预演失败时停止。

## 1. 电脑传输

在训练电脑新开终端（不是 Radxa SSH 窗口）：

```bash
cd /home/luckysir/microduck
scp radxa_j5_i2c_fix_v1.1.0.tar.gz wmh@10.146.187.157:~/
```

## 2. Radxa 检查并应用

在 Radxa SSH 窗口：

```bash
cd ~
tar -xzf radxa_j5_i2c_fix_v1.1.0.tar.gz
cd ~/radxa_j5_i2c_fix
sha256sum -c SHA256SUMS
python3 enable_j5.py --version
sudo apt-get update
sudo apt-get install -y device-tree-compiler
python3 enable_j5.py
```

应显示“设备树预演通过”和两个启动文件的修改预览。然后应用：

```bash
sudo python3 enable_j5.py --apply
```

应用会先在 `/boot/microduck-j5-backup-日期时间/` 备份两个原始配置，再安装已验证的 overlay
并原子更新配置。输出含准确的备份路径与两条回退命令。重复应用相同内容不会新增修改。

**只有显示“应用成功”或“配置已一致”后再重启。报错时保留完整输出，不继续重启。**
托稳机身，运行：

```bash
sudo reboot
```

SSH 此时断开是预期结果；IP 若变化，以热点设备列表为准。

## 3. 重启后检查

电脑重新连接：

```bash
ssh wmh@10.146.187.157
```

Radxa 执行：

```bash
sudo sh -c 'grep -H -E "fe5c0000|i2c3|pin (32|33|109|110) " /sys/kernel/debug/pinctrl/*/pinmux-pins'
cd ~/deploy_hd1910
./run.sh doctor
```

预期 GPIO1_A0/A1（pin 32/33）归 `fe5c0000.i2c` 使用，group 为 `i2c3m0-xfer`。
若 IMU 实际接线与模块模式正确，doctor 应找到地址 0x6A 或 0x6B、WHO_AM_I=112 的唯一设备。
只有恰好一个匹配项时再执行 `./run.sh imu --seconds 30`，开始约两秒保持机身静止。
若仍无应答，保留诊断结果，继续检查实体接线和模块配置。

## 回退

运行 `--apply` 时会打印两条带准确路径的 `sudo cp -p` 命令。执行它们后 `sudo reboot`，
恢复原 extlinux.conf 和 armbianEnv.txt 即可；不再引用的 DTBO 文件可留在原处。
如果无法登录，可将系统存储介质接入 Linux 电脑，在挂载的系统中，把
`boot/microduck-j5-backup-日期时间/` 下的 `extlinux.conf`、`armbianEnv.txt`
分别复制回 `boot/extlinux/extlinux.conf`、`boot/armbianEnv.txt`，安全卸载后再开机。

写入异常时脚本会尝试恢复所有配置；若有恢复失败，输出会明确给出备份目录并保留有效 DTBO，
避免仍引用它的启动项出现文件缺失。此时先处理错误，不重启。

## 验证范围

2026-09-29，1.1.0：23 项 unittest 测试通过，使用真实 dtc/fdtoverlay/fdtget 和临时模拟设备树。
先用旧实现复现官方 A1、A0 排列误拒绝，再确认新版接受两种排序。
重复引脚、错误复用功能、额外或缺失记录均拒绝，并在错误里显示实际读取值。
覆盖精确修改、幂等、冲突拒绝、缺失符号/节点、M0 选择、备份及失败恢复。
没有远程修改你的 Radxa，也未用这些测试宣称实体接线或 IMU 已验证成功。
运行时仍须在你的 Radxa 上完成实际 DTB 预演和重启后检查。

开发者复现（需安装 device-tree-compiler）：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
```
