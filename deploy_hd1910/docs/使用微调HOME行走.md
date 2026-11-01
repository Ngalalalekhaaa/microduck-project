# 使用地面微调结束时的位置行走

本入口采用 `home_calibration_pose_20261101_010812.json` 中10秒模型微调结束时的
`end_joint_positions.joints.actual_ticks`，作为行走的软件 HOME。
14个刻度依次为 `2045,2191,1620,1203,1696,1784,1781,2084,1988,2042,2056,2474,2760,1812`。
它使用地面模型阶段的实测位置；悬空阶段的读数、尚未到达的目标均未用于本配置。

配置为 `robot.home_ground_20261101_010812.json`。读数换算和动作目标换算都使用同一组
新零点；行走仍使用原 `hd1910_walk_1849.onnx` 和 P5/D0/I0。
这是用户选定的软件参考，未独立验证机械 HOME 精度。

在机器人终端执行。先托稳躯干与头部、双脚离地：

```bash
cd /home/duck1/deploy_hd1910
./run.sh relax
./walk_from_tuned_home.sh --move --keyboard --seconds 0 --vx 0
```

程序缓慢回到这组新 HOME 并持位。把双脚放到地面，扶稳并保持躯干静止，再输入 `WALK`
并回车。程序静止校偏约3秒，对齐新 HOME 约1秒，然后启动行走模型。
模型从零速度开始，输入数字后回车控制：`1`前进、`2`后退、`3`左移、`4`右移、
`5`左转、`6`右转、`0`停步保持模型、`q`退出并卸力。Ctrl+C也会停止并卸力。

Xbox入口也可使用这组参考：

```bash
./walk_from_tuned_home.sh --move --gamepad --gamepad-buttons --seconds 0 --vx 0
```

托稳后按A回新 HOME，双脚落地、松开摇杆后按菜单键启动，摇杆控制，B卸力。
所有新入口都固定加载该配置。原 `robot.json` 保留，可用原入口恢复原参考。
运行日志保留完整配置及其 `home_endpoint_reference` 来源。
