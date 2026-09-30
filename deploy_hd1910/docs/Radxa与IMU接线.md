# Radxa Zero 3W 与 LSM6DSV16X：先识别，再核对方向

这份说明适用于 **LSM6DSV16X/TR 直接通过 I²C 或 SPI 接入 Radxa**。先保留现有接线，用 `doctor` 识别。读不到设备时，先确认板型、供电、接口配置和实际导线去向；不要根据网上另一块板的照片盲目换针。

下面的命令在 `deploy_hd1910` 目录执行，假设已按主 README 完成 `install.sh`。这些 IMU 命令不使能舵机。核对姿态时应托稳机身，保持舵机卸力。

## 1. 排针、GPIO 名和 Linux 总线号是三个概念

Zero 3W 的扩展座是 **2 排 × 每排 20 针，共 40 针**。物理 pin 是插座上的位置编号；`GPIO4_B2` 是芯片信号名；`/dev/i2c-N` 是 Linux 暴露的软件设备。不能拿其中一个数字替代另一个。[Radxa 官方硬件接口说明](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface)

| 看到的标记 | 表示什么 | 怎么确认 |
| --- | --- | --- |
| Pin 27 | 排针的第 27 个物理位置 | 按对应板版本的官方排针图数位置 |
| GPIO4_B2 / I2C4_SDA_M0 | 这个位置可复用的芯片功能 | 查官方接口表及实际启用的设备树配置 |
| `/dev/i2c-4` | Linux 的某个 I²C 设备节点 | 看系统实际设备与芯片身份读取结果 |
| `0x6A`、`0x6B` | 一条 I²C 总线上的芯片地址 | 本程序在这两个地址读取 `WHO_AM_I` |

**Pin 1 的方向必须根据板上丝印、定位标记和官方图确认。** 从正面、背面或转过 180° 看板，照片中的左右会变化。不要以“USB 口朝左时最上面那颗”之类口述猜方向。型号、版本可在 [Radxa 硬件资料下载页](https://docs.radxa.com/en/zero/zero3/download) 对照原理图和元件位置图。

### 官方接口表提供的 I²C 位置

以下是核对接线时的参考，**不表示当前系统已经启用这些接口，也不要求把现有线改接到这里**：

| 官方表中的功能 | SDA：物理 pin / GPIO 名 | SCL：物理 pin / GPIO 名 |
| --- | --- | --- |
| I2C4_M0 | 27 / GPIO4_B2 | 28 / GPIO4_B3 |
| I2C5_M0 | 31 / GPIO3_B4 | 29 / GPIO3_B3 |

当前官方表将 pin 3、5 列为 GPIO1_A0/A1、UART3_RX/TX_M0，不能直接套用树莓派“3、5 默认就是 I²C”的说法。上述功能和电平以 [Radxa 当前接口页](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface) 为依据，核对日期为 2026-09-28。

**安装过 Microduck 官方 Linux 教程后，还可能有自定义配置。** Microduck 的 [`i2c3-pihat.dts`](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/deploy/audio/i2c3-pihat.dts) 将 pin 3/5 复用为 I2C3_M0，并描述 `/dev/i2c-3` 和 `/dev/i2c-pihat`。这解释了教程与通用接口表可能不同。先查看自己的实际系统；不要自行重装 overlay 或改 USB-C 控制器配置。我们的程序不改设备树，也不把硬件名称中的数字当成 Linux 总线号。

## 2. 供电与芯片型号

I²C 连接需要 SDA、SCL、电源和共同的 GND。LSM6DSV16X 裸芯片的 VDD 工作范围为 1.71–3.6 V，VDD_IO 为 1.08–3.6 V；使用 3.3 V 系统时，也应按具体小板的 `3V3`、`VCC`、`VIN` 标记和原理图核实供电。模块是否带稳压或电平转换不能仅凭外观判断。I²C 模式、地址选择和上拉也应按该模块说明确认。[ST 数据手册：电气参数与接口章节](https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf)

**舵机的 7.4 V 电源不能接入 IMU 或 GPIO；GPIO 信号线不能接 5 V。** Radxa 的 GPIO 使用 3.3 V 电平；主板的 5 V 电源入口并不表示信号脚可以承受 5 V。[Radxa GPIO 电压说明](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface#gpio-voltage)

另外，Pollen 原版 Robot HAT 的传感器原理图标注 **U11 = BMI088**。你提供的型号是 **LSM6DSV16XTR**，两者驱动不同；不能因为安装了 HAT 就认定上面是 LSM6DSV16X。[官方 HAT 原理图](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/blob/main/sensors.kicad_sch)

本部署程序按 LSM6DSV16X 实现，要求在 `0x0F` 身份寄存器读到 **`0x70`（十进制 112）**。它不会把 BMI088 当作兼容设备继续配置。`TR` 是同一芯片的包装型号后缀。[ST 数据手册](https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf)

## 3. I²C：先自动发现，不用先猜总线号

```bash
./run.sh init
./run.sh doctor
```

`init` 创建 `robot.json`，已有文件会保留。`doctor` 保存 `logs/doctor.json`，列出 `/dev/i2c-*`，仅在每条总线的 `0x6A` 和 `0x6B` 读取身份寄存器；不做全地址扫描，不写传感器配置。

看输出中的 `i2c`：

| 结果 | 含义与下一步 |
| --- | --- |
| `matches` 恰好一项，`who_am_i: 112` | 找到匹配芯片，可继续 IMU 读数 |
| `buses: []` | 当前没有可访问的 I²C 设备节点；先核对系统接口是否启用 |
| `errors` 含 `Permission denied` | 有设备节点，但当前账号没有访问权限；先按该系统设备权限设置处理 |
| 有总线、没有匹配项 | 核对实际芯片、供电、接线、模块接口模式和对应总线；某些地址无应答本身很常见 |
| `matches` 多于一项 | 需要按实际安装明确选择机身传感器，程序不会随意挑一颗 |

默认保留 `robot.json` 中的：

```json
"imu": {
  "transport": "i2c",
  "i2c_bus": null,
  "i2c_address": null,
  "mounting_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}
```

这里的 `null` 表示自动发现。只有一个匹配项时，`imu`、`check` 和后续运行会采用发现的总线与地址；自动选择不会覆盖配置文件。如果需要固定设备，将发现结果中的 `i2c_bus` 和 `i2c_address` 原样填入。地址也支持字符串 `"0x6A"` / `"0x6B"`；JSON 中不能直接写无引号的 `0x6A`。

```bash
./run.sh imu --seconds 30
```

启动先静止约 2 秒校准陀螺仪偏置，随后显示 `gyro(rad/s)` 和 `gravity`。程序会正常配置传感器的易失寄存器，默认 120 Hz、加速度 ±4 g、陀螺仪 ±500 °/s；不写 OTP，也不操作舵机。`gravity` 是单位重力方向，不是 m/s² 数值。

## 4. 验证安装轴：不能只看“放平时正常”

模型的机体坐标是：**X 向前、Y 向左、Z 向上**。传感器必须刚性固定在躯干上；如果它安装在会随头颈活动的位置，一个固定旋转矩阵不能把它一直当作躯干 IMU 使用。

`mounting_rotation` 的作用是：`机体向量 = 旋转矩阵 × 传感器向量`。默认单位矩阵只是暂用设置。静止、直立时应看到：

- `gyro` 接近 `[0, 0, 0]`。
- `gravity` 接近 `[0, 0, -1]`。
- 稍微倾斜并保持后，`gravity` 应保留倾角，不能仍强行显示竖直。

接着轻缓改变姿态，每次先停稳再读重力。校准的最初 2 秒保持静止，之后再做这些动作：

| 手动动作 | 正确映射后的重力变化 | 动作进行时的陀螺仪符号 |
| --- | --- | --- |
| 抬高机身前端，随后低下前端 | 抬高时 `gravity_x < 0`；低下时 `gravity_x > 0` | 抬高过程 `gyro_y < 0`，反向为正 |
| 下压机身左侧，随后下压右侧 | 左侧下压时 `gravity_y > 0`；右侧下压时 `< 0` | 左侧下压过程 `gyro_x < 0`，反向为正 |
| 机身保持竖直，向左转，再向右转 | `gravity` 大体保持 `[0,0,-1]` | 向左转过程 `gyro_z > 0`，向右转为负 |

这三组动作同时检查轴的排列和正负方向。仅静止竖直不能确定绕 Z 轴的安装角：传感器水平转了 90° 或 180°，重力仍可能看起来完全正确。方向判断以机器人自身的前、左、上为准。

确认后，用 `imu-map` 保存映射。三个参数依次表示 **机体 X、Y、Z 使用哪个传感器轴**。例如 `+z,+y,-x` 表示机体 X 取传感器 +Z，机体 Y 取 +Y，机体 Z 取 −X；这个例子不代表你的安装方式。

只有实际验证三个轴同向，才用下面的单位映射：

```bash
./run.sh imu-map --axes='+x,+y,+z'
./run.sh imu --seconds 30
```

实际安装不同就使用已验证的轴排列与符号。CLI 会拒绝重复轴和镜像变换，保存后将配置顶层 `imu_mount_verified` 设为 `true`。再次完成三组动作确认后，才继续主 README 的关节标定和动作步骤。若传感器与机体有斜角，简单的轴交换不够，需要经过验证的完整旋转矩阵。

陀螺仪静态校准只减去静止偏置，不会把初始倾角重定义为竖直。互补滤波会拒绝明显偏离 1 g 的加速度；持续线加速仍可能影响姿态，六轴 IMU 也不能提供绝对航向。CPU 模拟测试通过并不等于安装方向和行走中的姿态已得到实机验证。

## 5. SPI 备选

如果实际模块已接 SPI，先按主板和模块原理图确认 SCLK、MOSI、MISO、CS、电源与地，以及对应的 `/dev/spidevB.D`。当前驱动使用四线 SPI，支持 mode 0 或 3，默认 1 MHz。不要从排针编号推算 `B`、`D`。

I²C 安装依赖里包含 `smbus2`；SPI 还需要在部署虚拟环境安装 `spidev`：

```bash
.venv/bin/python -m pip install spidev
```

将 `robot.json` 的 `imu.transport` 改为 `"spi"`，增加已核实的整数 `spi_bus`、`spi_device`；可选 `spi_mode: 0` 和 `spi_max_speed_hz: 1000000`。SPI 不自动猜总线或片选。`doctor` 会列出 SPI 设备节点，随后运行 `imu` 才通过 SPI 核对芯片身份。接口配置和安装方向确认流程与上面相同。
