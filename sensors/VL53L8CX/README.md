# VL53L8CX 激光测距测试（Radxa Linux）

这是独立的 C 测试工具，包含 ST VL53L8CX ULD、固件和 Linux I²C 适配层，不需要编译整个 MicroDuck / Rust 工程。支持单个地址的器件识别、固件初始化、8×8 测距和 CSV 导出。

**2026-11-09 状态：本机编译与模拟测试通过；J5 连接器已脱落，未进行实物测距验证。以下 `probe`、`range` 命令只供修复后测试。**

## 接线前提

HAT C1 的 J5 引脚依据仓库中的 [C1 原理图](https://github.com/Ngalalalekhaaa/microduck-project/blob/main/elec_RPI_Robot_HAT/production/ASE01187-C1_elec_RPI_Robot_HAT_SCH.pdf) 第 5 页及 PCB 核对：

| J5 针号 | HAT 信号 | 模块对应信号 |
| --- | --- | --- |
| 1 | GND | GND |
| 2 | 3.3V | 仅在模块确认支持 3.3V 输入时接 VIN |
| 3 | SDA / IO2 | SDA/MOSI，须确认主控侧兼容 3.3V |
| 4 | SCL / IO3 | SCL/CLK，须确认主控侧兼容 3.3V |

**针号不等于从任意视角看到的左右顺序。** 请以 HAT 丝印、原理图和之前的接线标注图定位。J6 等接口使用不同 GPIO，不能只换一个插座就沿用 J5 配置。

用户的红色九针转接板虽有 WTXS0104E 电平转换芯片，但 VIN 供电范围、转换线路、LPn 上拉及 SPI_I2C_N/NCS 下拉尚未拿到卖家确认。先修复连接，再确认这些条件；本工具不能验证硬件供电是否安全。

ST [DS14161 数据手册](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf) 表 3 描述的是传感器芯片：I²C 模式需要 SPI_I2C_N 为低、LPn 在 IOVDD 电平为高，NCS 按手册下拉；MISO 不接。芯片 IOVDD 是 1.2V 或 1.8V，不能据此把转接板 LPn 直接接 3.3V。当前程序不配置任何 GPIO，也不改变器件地址。

## 安装和编译

在 Radxa 上进入本目录后执行（仓库可以放在任意位置）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential linux-libc-dev python3
make
./build/vl53l8cx_test --help
make test
```

`make test` 使用模拟的 I²C 与驱动，不打开 `/dev/i2c-*`，不需要传感器，也不需要 sudo。直接运行工具而不带子命令，也只显示帮助。

## 修复后的实物测试命令

本项目当前 Radxa 的 J5 配置为 `/dev/i2c-3`；其他系统的总线号可能不同。可用下面两个命令只查看设备节点和内核总线名称，不进行总线扫描：

```bash
ls -l /dev/i2c-*
cat /sys/class/i2c-dev/i2c-3/name
```

先确保其他程序没有操作同一个 VL53L8CX，然后在本目录执行：

```bash
# 仅识别指定地址，不扫描整条总线。
sudo ./build/vl53l8cx_test probe --bus /dev/i2c-3 --address 0x29

# 上传 ST 固件，采集 10 帧 8×8 数据，5 Hz，然后停止测距。
sudo ./build/vl53l8cx_test range --bus /dev/i2c-3 --address 0x29 --frames 10 --hz 5

# 记录 100 帧 CSV；状态说明和错误写入 stderr，不混入 CSV。
sudo ./build/vl53l8cx_test range --bus /dev/i2c-3 --address 0x29 --frames 100 --hz 5 --csv > ranging.csv
```

**Linux 使用 7 位地址 `0x29`。ST 文档的 `0x52` 是包含读写位位置的 8 位写地址，不能当作这里的默认地址。** `probe` 调用 ST `vl53l8cx_is_alive()` 检查 ID `F0/0C`，其中包含写寄存器页选择；它不是纯读取命令，也不证明测距功能已通过。

`range` 调用完整的 `vl53l8cx_init()`，包括固件下载；之后设置 8×8、autonomous 模式、5 ms 积分、指定帧率及最强目标，每区保留一个目标。以 `check_data_ready()` 轮询，再用 `get_ranging_data()` 取帧，最后 `stop_ranging()`。步骤依据 ST [UM3109](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf) 第 4、5 节。

按 Ctrl+C 可提前结束，程序会尝试停止测距并关闭设备；如果总线断线导致停止失败，会明确报告传感器状态未确认。不要与另一套 ToF 程序同时运行。工具的 advisory lock 只约束遵守同一锁的程序，不能阻止其他程序直接访问 I²C。

## 如何看结果

默认输出每帧的 8×8 距离矩阵（毫米），随后输出状态矩阵：

| 显示 | 判定 |
| --- | --- |
| 普通数字 | 检测到目标，`target_status=5` |
| `~` 加数字 | 检测到目标，状态 6 或 9，置信度较低 |
| `----` | 未检测到目标，或其他状态；不作为可靠距离 |

CSV 同时保留原始 `distance_mm`、`target_status`、`targets` 和 `quality`。不能忽略状态，直接把每个原始距离都当作有效测量。有效性解释依据 UM3109 第 5.5 节；本工具采用保守过滤。

矩阵按 ULD 区域编号 0–63 顺序显示，`row`/`column` 只是数组索引；不是已根据机器人安装方向校正的实际左右/上下。UM3109 第 2.2 节说明光学成像会翻转场景，实装后应移动目标确认方向。

## 参数、错误与实现范围

- `--hz` 允许 1–15（8×8 模式）；默认 5 Hz。
- `--chunk` 默认 1024 字节，可设 1–4096。读操作使用同一次 `I2C_RDWR` 内的寄存器地址写入与重复起始读取；16 位寄存器地址高字节先传，固件大块写入按递增地址分包。
- `--init-timeout-ms` 默认 30000；`--frame-timeout-ms` 默认 5000。初始化慢时可以明确增大前者。截止时间在平台回调中检查，单次 ioctl 的阻塞仍受 Linux 内核 I²C 驱动限制，程序不修改适配器的全局超时设置。
- I²C 失败后保留首个 errno 并停止继续传输。程序同时检查 ULD 状态和平台错误；开始测距后遇到失败也会尝试收尾。
- 退出码：0 成功、1 通信/驱动/超时失败、2 参数错误、130 为 Ctrl+C 中断。
- `Cannot open/use`：检查设备节点、权限和是否被内核驱动占用。本工具不使用 `I2C_SLAVE_FORCE`。
- `probe failed` / ID 不匹配：检查确认过的供电、电平、使能和总线映射，不要用遍历所有地址的方法碰运气。
- `init failed`：固件上传未完成或驱动检查失败，不应继续读取距离。
- 没有有效区域时，可在硬件修复后用距离约 20–100 cm 的平整目标检查朝向、遮挡与光照；这仍需要实物验证。

测试范围：本机 gcc 编译；模拟验证 32 KiB 固件分包、I²C 地址与寄存器字节序、重复起始读取、部分传输错误、超时、信号取消、两帧输出和测距收尾。没有声称这些模拟结果等同于 Radxa 与该红色模块的实测。

源码与版本记录见 [SOURCES.md](SOURCES.md)，许可证见 [LICENSE](LICENSE) 及 [vendor/LICENSE.txt](vendor/LICENSE.txt)。
