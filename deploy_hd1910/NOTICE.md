# 来源与已知差异

## 代码与模型

- 本包新增 Python 运行、标定、诊断和测试代码按 Apache-2.0 提供，全文见 `licenses/Apache-2.0.txt`。
- 训练框架与策略观测、HOME、关节顺序来源于
  [Pollen Robotics microduck_rl](https://github.com/pollen-robotics/microduck_rl/tree/cb70b792312d559a4da09064d92009079671815f)
  和本地 `microduck_rl_hd1910` 的 `7c29a4e627446516c0b437eb52a52149e4cf5cc2`。
  本包两份模型是本次自行训练的 4000 轮检查点导出，含观测归一化；来源及 SHA-256 见 `manifest.json`。
- HD1910 M6 参数来自
  [LuwuDynamics xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl/blob/326d77a1122870bdefa2c36403937502c958e69c/src/mjlab_microduck/robot/xgoduck/params/1910_m6.json)，
  不是本机实测辨识。
- FT-SCS 协议实现参考飞特公开协议及
  [fanhao375/microduck-replica](https://github.com/fanhao375/microduck-replica/tree/a23f4c18a4741a3cde5257e639960ea37bbadd8d)
  的 `tools/servo-web/feetech.py`、`tools/FeeTech_HD1910M_Servo/src/registers.rs`。
  这些工具代码使用 Apache-2.0；本包没有逐字复制上游说明文档。
- 关扭矩后写目标仍可能拖动的实机经验，以及速度字段使用情况，来自
  [LuwuDynamics Arduino 运行程序](https://github.com/LuwuDynamics/xgoduck_runtime_arduino/blob/a6dc9f98eae6f8cc8f2bac81a97dde5a7a8bd0d2/README.md)。
  本包没有移植其不同的关节排序或 ±24° HOME。
- LSM6DSV16X 寄存器/量程来自
  [ST 数据手册](https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf) 与
  [ST 官方驱动](https://github.com/STMicroelectronics/lsm6dsv16x-pid/tree/2808e5cd6b85f91b66758e1dd0faab5f043aba07)。
  相关上游许可见 `licenses/ST-driver-LICENSE.txt`。

## 图片

`docs/images` 是用原版 `robot_walk.xml` 静态渲染的 ZERO 和 HOME 参考图。
Pollen Robotics 的 Microduck 几何与网格使用 CC BY-NC-SA 声明；这些衍生图保留来源，
按 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 分享。
图中不是你的已组装机器人照片。

## 本包与原版运行时的区别

本包是独立的前台 Python 台架程序，使用本机 CPU ONNX Runtime、HD1910 总线与 Linux 直连 IMU。
它不安装官方 robotd、遥控器、视频、在线策略管理或开机服务。

策略的 61 维输入、14 维动作、50 Hz、HOME 和动作比例与训练匹配；
硬件执行仍有未标定差异：P/D 固件响应、速度/电流上限、软件姿态融合、角度差分速度、
通信延迟、质量/重心、电池压降和机械间隙。首次试机采用目标角拒绝边界，可能中止仿真中允许的目标超调。
本包不会自动把这些边界放宽成未经验证的实机设置。

姿态估计采用六轴软件互补滤波；无法用重力观测绝对航向，持续线加速度也可能影响重力估计。
输入只需要角速度和投影重力，不需要向模型提供绝对航向。

软件卸力和看门狗需要 Linux 与串口能工作。断线、总线系统调用卡死、操作系统卡死等情况下，
软件无法保证电机已停止；试机时应保留直接切断舵机供电的能力。
