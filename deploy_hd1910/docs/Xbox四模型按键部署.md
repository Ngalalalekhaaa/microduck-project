# Xbox 十字键：四份更新模型

适用当前 **Radxa + HD-1910 的 `deploy_hd1910` Python 遥操**。原生 Rust `padd` 的十字键已有其他功能，本入口会拒绝与 `padd/robotd` 并行运行。安装包为 `microduck-xbox-four-actions-20261028.tar.gz`；此前的六模型包与录像保留。

后续头部反馈保护调整见 [滚翻头部俯仰范围](滚翻头部俯仰范围.md)：当前机器仅前滚翻的 ID 7 正向上限试调为 105°，对应补充包 `microduck-roulade-head-limit-20261028.tar.gz`；该调整尚未做真实机械安全范围测量。

## 按键

| 键位 | 动作 |
| --- | --- |
| 十字键 ↑ | 当前行走 ↔ 新行走 1849；先停步约 0.6 秒，站立保持 3 秒，再推左摇杆前后行走 |
| 十字键 ↓ | 单次前滚翻 1600；先停步，检测翻转完成且重新站稳后保持站立 |
| 十字键 ← | 摩擦坐站 2598：站着按则坐下，坐着再按则站起 |
| 十字键 → | 重心坐站 2598：站着按则坐下，坐着再按则站起 |
| A / Menu / B | 回 HOME / 校偏后启动策略 / 停止卸力 |
| Y / RB / LB / X | 当前坐站模型切换坐/站 / 原俯身 / 前滚翻别名 / 已标定的嘴动作 |

左右两键是**两套坐站权重**，动作类型相同。坐着按任一左右键，都会先用当前坐站模型站起；等 3 秒后再按另一方向，才能换另一套权重坐下。↑、↓仅在行走或站稳时接受，坐着时先站起。

方向键按一下触发一次，长按不重复；斜按/多方向不执行新动作。过渡、俯身、滚翻和起立等待期间的新方向键请求会丢弃，不会事后突然执行。原有 Y/RB/LB 与方向键同时按时，原有按键优先。支持蓝牙 `ABS_HAT0X/Y` 和 USB `BTN_DPAD_*` 两种输入。

## 1. 电脑传包

在电脑终端执行，`wmh` 替换为实际机器人用户名：

```bash
cd ~/microduck
sha256sum -c microduck-xbox-four-actions-20261028.tar.gz.sha256
read -r -p 'Radxa IP: ' ROBOT_IP
scp microduck-xbox-four-actions-20261028.tar.gz \
  microduck-xbox-four-actions-20261028.tar.gz.sha256 "wmh@$ROBOT_IP:~/"
ssh "wmh@$ROBOT_IP"
```

## 2. Radxa 安装

先接住机器人、按 B 卸力，退出已有手动控制。以下均在 **Radxa SSH 终端**执行：

```bash
# 只停止已安装的控制服务；不会禁用它们的开机配置。
for svc in microduck-xbox.service servo-web.service padd.service robotd.service; do
  if [ "$(systemctl show "$svc" -p LoadState --value 2>/dev/null)" = loaded ]; then
    sudo systemctl stop "$svc" || break
  fi
done
cd ~
sha256sum -c microduck-xbox-four-actions-20261028.tar.gz.sha256
# 首次解压；若已有同名目录，先移走旧目录再执行。
test ! -e microduck-xbox-four-actions-20261028 && \
  tar -xzf microduck-xbox-four-actions-20261028.tar.gz
cd ~/microduck-xbox-four-actions-20261028
python3 install.py --target ~/deploy_hd1910
```

最后一条只预览：检查模型、依赖、原标定与控制器占用。预览通过后安装：

```bash
python3 install.py --target ~/deploy_hd1910 --apply
cd ~/deploy_hd1910
.venv/bin/python scripts/model_suite.py check --config robot.json
```

安装保留原机器 `robot.json`、HOME/舵机零位/IMU 标定，更新前自动备份文件，失败回滚；不会启动服务或动作。若提示缺少依赖，先在现有环境安装 `~/microduck-xbox-four-actions-20261028/payload/requirements.txt`，然后重新预览。

## 3. 只读核对方向键

```bash
cd ~/deploy_hd1910
.venv/bin/python scripts/test_xbox.py --seconds 30
```

逐个按 ↑↓←→，输出 `dpad_pressed` 应分别显示相应箭头；规范后的按钮码为 544/545/546/547。按 B 或 Ctrl+C 退出。此测试不读取 IMU、不驱动舵机。

## 4. 启动四模型遥操

```bash
cd ~/deploy_hd1910
# 可先预览将启用哪些模型，不写配置、不运动
.venv/bin/python scripts/model_suite.py select --four-actions
# 启动，每轮策略最多60秒，结束后卸力并重新等待A
bash xbox_four_actions_hd1910.sh 60
```

按终端提示：**托住机身、双脚悬空按 A 回 HOME → HOME 就绪后落脚、摇杆居中并保持静止 → 按 Menu → 用十字键选动作 → 接住机器人后按 B 卸力**。新开进程从 `robot.json` 当前行走模型开始；↑切换仅在本次进程内生效，日志显示 `walking:original/updated`。

每轮结束或按 B 后会重新等待 A；要退出整个入口，用 Ctrl+C。

需要持续运行时：

```bash
bash xbox_four_actions_hd1910.sh 0
```

包装器固定最大前后速度为 **0.2 m/s**。新行走需将左摇杆向前推至接近满量程：本轮仿真中 0.1 m/s、后退、侧移和转向响应很弱。四模型入口会显式开启滚翻，并沿用既有 Xbox 目标角/跳变“仅记录”策略；单圈编码器、实际关节角、反馈与 IMU 时效等检查继续强制执行。滚翻未站稳、超时、反转过大或动作中断连会卸力。

本次完成离线按键、模型路由及原始动作历史验证；**未连接真机验证这些新键位**。已有模型仿真结果与录像位置见 [六模型测试说明](20261028模型测试与Xbox部署.md)。手动试用通过前，保持本入口手动启动；机器重启后，原开机服务仍按原设置运行。
