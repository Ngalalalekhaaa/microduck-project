# IMU 供电诊断

**供电/地线干扰仍是候选原因，尚未证实，约1 V压降不能单独解释故障。** 三次前滚翻出现 I²C 地址无应答，舵机反馈电压分别从 7.2 / 6.7 / 6.6 V 降到 6.1 / 5.6 / 5.6 V；但同日重心坐站的两次起立分别从约7.0 V降到5.1 V、5.3 V，IMU仍持续工作。另一次滚翻下降1.2 V后因头部限位停止，也没有IMU错误。静止及轻动线束测试17,800轮未复现。

详细对照已保存在电脑工作区 `reports/20261028_action_voltage_comparison/`。这些反例说明不能把舵机电压下降直接当成IMU3V3掉电的证明；仍需实际测量传感器端电源及通信波形。

## 电压关系

- I²C 是通信接口，没有统一的“供电必须 3.3 V”规定。这台 Radxa 的 GPIO 使用 **3.3 V 信号电平**，SDA/SCL 通过上拉形成高电平。
- LSM6DSV16X 裸芯片的 VDD 为 **1.71–3.6 V**、VDD_IO 为 **1.08–3.6 V**。模块的 VIN 能接多少伏还取决于小板是否有稳压、电平转换，不能据此给模块接 5 V。[ST 数据手册](https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf)
- 按仓库 HAT 和 Radxa Zero 3W V1.12 原理图：**电池 → HAT 的 AP63205 降压得到 5 V → Radxa 产生 3.3 V → 40针排针 → HAT J5/IMU**。实际板卡版本和 IMU 小板仍须核对。3.3 V 经过稳压，电池从 6.6 V 降到 5.6 V，不表示它会按比例降低；输入余量不足、瞬时大电流、线阻或地线扰动可能使其失稳。
- 实机 SARADC 的标度约为 1.8 V 满量程，尚无经过验证的 IMU 电源测量通道。ADC0 在官方图中用于 Recovery 按键。`regulator` 显示的 3300000 等设定值也不是电压表读数。**不能把 3.3 V 直接接入该 ADC。**

电路依据：[HAT 电源](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/blob/main/power.kicad_sch)、[HAT J5及上拉](https://github.com/pollen-robotics/elec_RPI_Robot_HAT/blob/main/sensors.kicad_sch)、[Radxa V1.12原理图](https://dl.radxa.com/zero3/docs/hw/3w/radxa_zero_3w_v1.12_schematic.pdf)、[AP63205 数据手册](https://www.diodes.com/assets/Datasheets/AP63200-AP63201-AP63203-AP63205.pdf)。

## 直接使用

SSH 登录后执行；手柄服务保持原来的启动方式。诊断程序只读取日志，不打开 I²C/舵机串口，不触发动作。

```bash
ssh wmh@10.142.40.157
cd ~/deploy_hd1910

# 分析最近一个已经结束的动作回合
.venv/bin/python scripts/imu_power_diagnose.py --latest

# 或：监听后续回合，动作结束后自动输出诊断；Ctrl+C退出监听
.venv/bin/python scripts/imu_power_diagnose.py --watch
```

结果保存在 **`~/deploy_hd1910/logs/imu_power_diagnosis/motion_时间/`**：

- `report.json`：停止原因、是否 I²C NACK、翻滚前/中的舵机反馈电压、结论及证据限制。
- `timeline.csv`：逐帧时间、模式和最低舵机反馈电压，可以用表格软件查看。

新增日志字段：`policy_start_monotonic_s` 记录策略计时起点；IMU 失败时 `imu_fault` 保存软件发现时间及本轮已经读到的舵机反馈。不会增加 I²C 访问、重试或延后原有失败卸力。旧日志仍可分析，但故障时刻只能用最后成功帧近似，不能恢复当时没有测量的 3.3 V。

查看 ADC 清单：`.venv/bin/python scripts/imu_power_diagnose.py --inventory`。此功能不会把未核实的通道解释成电源测量。

## 如何确认是电源问题

在 **IMU 小板端的电源与本地 GND** 同步测波形，优先测 VDD、VDD_IO（若为不同电源）和 HAT 的 5 V，再对照 SDA/SCL 的应答丢失。示波器能捕获比约50 Hz控制日志更短的跌落；普通万用表显示正常不能排除瞬时问题。接地方式须适合该电池系统，防止探头地夹短路其他节点。

如果电源在失败前明显跌落、越出工作范围，支持供电故障；若电源波形稳定而 SDA/SCL 异常，则继续检查上拉、线束、共地及400 kHz信号完整性。只有某一低速测点未见跌落，不能排除其他电源或更短毛刺。

已有外部采集数据可导入，但必须先完成真实测量与时间同步；程序不能替代这些硬件步骤。CSV 格式如下，**示例值不是实测结果**：

```csv
session,t,point,voltage_v
motion_时间.session.json,13.80,imu_vdd,3.30
motion_时间.session.json,13.82,imu_vdd,1.60
```

`session` 须与分析的文件名完全相同；`t` 对齐对应 `.jsonl` 的 `t`（秒），不是系统日期，也不是从SSH启动开始计时。新日志可用 `policy_start_monotonic_s` 同步同机采集；示波器需要实际同步标记，不能随意平移曲线。测点只支持 `imu_vdd`、`imu_vdd_io`、`hat_5v`，输入单位为伏。

```bash
.venv/bin/python scripts/imu_power_diagnose.py \
  --session logs/motion_时间.session.json --rail-csv /实际路径/rail.csv
```

程序比较故障参考时刻前0.1秒至后0.05秒的实测样本与对应芯片电源工作范围，并单独标记3.3 V测点低于3.0 V的压降。**3.0 V只是压降观察线，不是LSM6DSV16X最低工作电压。** 报告同时保留采样间隔及旧日志时间近似限制，不把相关性写成已确定根因。
