# 来源、版本与维护

核对日期：2026-11-09。

## ST 驱动

`vendor/vl53l8cx/` 三个文件原样复制自本仓库既有 `microduck/tof/vendor/vl53l8cx/`：

- `vl53l8cx_api.c`
- `vl53l8cx_api.h`
- `vl53l8cx_buffers.h`（包含固件与默认配置）

固定版本：`VL53L8CX_API_REVISION = VL53L8CX_1.2.1`。不宣称该版本是 ST 最新版本；本次保留既有固件与驱动配套关系，没有混搭不同版本的固件。`vendor/LICENSE.txt` 也原样保留；其中规定未随附其他包许可时采用 BSD-3-Clause。本目录 LICENSE 给出 BSD-3-Clause 完整文本并保留 ST 权利声明。

SHA-256：

```text
c036ad41c1f85d64935aab3244f920af9add290206ebcf3d2d10afa819c5469a  vendor/vl53l8cx/vl53l8cx_api.c
1515b53bc8d8aa5938a7809a32b70d2212c28cdc4b07109ed35a5d9c78eeaa12  vendor/vl53l8cx/vl53l8cx_api.h
392721459d81bd901d19809f6b03725f88826e8eff94854a36956cf8d5a9998a  vendor/vl53l8cx/vl53l8cx_buffers.h
```

`src/platform.h`、`src/platform.c` 为本测试包的 Linux 适配；`src/vl53l8cx_test.c` 为命令行应用。没有沿用原 Rust 工程的输出裁剪，特意保留 `nb_target_detected`、`target_status` 与 `distance_mm` 三者。

## 官方资料

- [ST DS14161 / VL53L8CX 数据手册，Rev 12](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)：表 3 的引脚/I²C 模式要求，第 7 节电气条件。
- [ST UM3109，Rev 12](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf)：第 4.1 节固件初始化，第 4.3–4.6 节分辨率/频率/模式/积分，第 5.3–5.5 节读取流程与状态过滤。
- [ST 官方 STM32duino VL53L8CX 库](https://github.com/stm32duino/VL53L8CX)：用于核对官方 API、默认地址和平台回调约定；在线版本可能比本包固定版本更新。本包不依赖 Arduino，也不在构建时下载源码。
- [本仓库 HAT C1 原理图](https://github.com/Ngalalalekhaaa/microduck-project/blob/main/elec_RPI_Robot_HAT/production/ASE01187-C1_elec_RPI_Robot_HAT_SCH.pdf)：J5 引脚对应。

红色九针转接板的品牌原理图仍未取得。以上 ST 芯片资料不能替代该转接板 VIN、电平转换与控制脚默认状态的确认。
