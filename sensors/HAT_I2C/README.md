# HAT J6 软件 I²C

适用于 MicroDuck HAT **C1 默认电阻配置**（R18/R19 已装，R16/R17 未装）与 Radxa ZERO 3 的 vendor 6.1 内核。保留原有 J5 硬件 I²C3、音频、摄像头和 UART，新增 `/dev/i2c-10`，不需要改焊接电阻。不同 HAT 版本或改过跳线的板子不能直接套用。

## 接线

**以板子元件面朝向自己、40 针长排针在上方、舵机大插座在左侧为准，J6 从左到右是：**

```text
SCL       SDA       3.3V       GND
第4脚     第3脚     第2脚      第1脚
```

这与同一朝向下 J5 的顺序一致。电源和地由 PCB 固定走线决定；软件只指定两根信号线的 SDA/SCL 功能。

| J6 脚号 | 功能 | 40 针排针物理脚号 | Radxa GPIO |
|---|---|---|---|
| 1 | GND | 接地网络 | — |
| 2 | 3.3V | 3.3V 网络 | — |
| 3 | SDA | 7 | GPIO3_C4，bank 3 offset 20 |
| 4 | SCL | 29 | GPIO3_B3，bank 3 offset 11 |

断电后接线；J6 只提供 **3.3V**。泛用红色 VL53L8CX 转接板的 VIN、电平和 LPn / SPI_I2C_N 接法仍要按该转接板资料确认，不能仅凭芯片名称判断。接线与测距说明见 [VL53L8CX](../VL53L8CX/README.md)。

## 安装与检查

将本目录完整复制到机器人 `~/sensors/HAT_I2C`。依赖为 Python 3、`device-tree-compiler`、`kmod`，内核需要 `i2c-gpio` 模块。

```bash
cd ~/sensors/HAT_I2C
# 只预览并验证设备树，不修改系统
python3 scripts/setup_j6.py
# 备份后写入，脚本不自动重启
sudo python3 scripts/setup_j6.py --apply
```

脚本逐项对比合并前后设备树的全部旧属性，只允许新增 J6 节点、引脚组和总线别名；保留 extlinux 原有 overlays 和 armbianEnv 中包括 NUL 在内的其他字节。已有不同内容的同名 overlay、已占用的 i2c10、多个启动项均会拒绝覆盖。备份位于 `/var/backups/microduck-j6-i2c/`，写入失败会恢复本轮已修改文件。脚本不会扫描 I²C、操作舵机服务或更换内核。

应用前还需确认 GPIO3_C4 和 GPIO3_B3 没有被其他运行中的驱动占用；设备树预演不能替代这项检查。本次已通过 debugfs 确认两脚空闲。

机器人放稳并停止动作服务后重启。重启后检查：

```bash
ls -l /dev/i2c-10
cat /sys/class/i2c-adapter/i2c-10/name
readlink -f /sys/class/i2c-adapter/i2c-10/device/of_node
```

设备树节点应指向 `/i2c-j6`。设备文件出现只证明驱动注册成功，不代表已确认连接的传感器能通信。SDA/SCL 都以开漏方式驱动并允许读取 SCL；使用 HAT 现有 10kΩ 上拉。初始 `delay-us=10` 是保守设置，理论上限约 50kHz，实际速率依调度和硬件而变化，尚未经示波器测量。

## 后续测距测试

确认模块供电和控制脚正确、接线完成后，再执行：

```bash
cd ~/sensors/VL53L8CX
make
sudo ./build/vl53l8cx_test probe --bus /dev/i2c-10 --address 0x29
sudo ./build/vl53l8cx_test range --bus /dev/i2c-10 --address 0x29 \
  --frames 10 --hz 2 --init-timeout-ms 120000
```

软件 I²C 上传 VL53L8CX 固件比原硬件总线慢，因此提高初始化超时。测试阶段不推荐同时承担机器人实时姿态闭环。不要用总线扫描来判断所有设备是否正常，也不要把另一个总线的 SDA/SCL 并接到 J6。

## 验证与资料

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖启动项保留、重复字段拒绝、NUL 保留、写入回滚，以及真实 `dtc` / `fdtoverlay` 合并后的 GPIO、开漏标志和原设备树属性保留。传感器通信与速率须接实物后另行验证。

- [Linux 6.1 i2c-gpio binding](https://github.com/torvalds/linux/blob/v6.1/Documentation/devicetree/bindings/i2c/i2c-gpio.yaml)：使用 `GPIO_ACTIVE_HIGH | GPIO_OPEN_DRAIN`，数值 6；不使用旧的 `sda-open-drain` / `scl-open-drain` 属性。
- [Radxa ZERO 3 接口定义](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface)。
- HAT C1 原理图：仓库硬件目录 `elec_RPI_Robot_HAT/production/ASE01187-C1_elec_RPI_Robot_HAT_SCH.pdf`。

## 本机启用记录（2026-11-10）

已应用并重启验证：`/dev/i2c-10` 指向 `i2c-j6`，SDA/SCL 空闲逻辑电平均为高；原 I²C3、AIC3104 声卡及相机 ISP 服务正常。Xbox 服务恢复原开机启用设置，当前仍保持停止。5 项配置测试在电脑和 Radxa 均通过。尚未接传感器进行识别或测距，电压与速率未用仪器测量。详见 [验证记录](evidence/verification.json)。

新增固定 `i2c10` 别名后，内核动态编号的 HDMI DDC 总线由 `i2c-6` 变为 `i2c-11`；原硬件 I²C0/2/3 编号保留，HDMI 声卡仍正常识别。
