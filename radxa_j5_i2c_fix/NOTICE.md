# 来源

修复工具版本：1.1.0。该版本修正了引脚排列顺序的校验；官方 DTS 保持原样。

`i2c3-pihat.dts` 原样复制自 Pollen Robotics Microduck：

- 仓库：https://github.com/pollen-robotics/microduck
- 提交：`a9ec4b2079ef8ee7904014089c885bb07d57d63c`
- 文件：`deploy/audio/i2c3-pihat.dts`
- SHA-256：`39c89da24a5a717aff1716a0c095af676828c3a3d8ebafb808f77c518965cb92`
- 许可：Apache-2.0，见 LICENSE。

`enable_j5.py` 针对用户提供的 Armbian 26.2.1 / Radxa Zero 3W 启动布局编写。
它仅支持已核对的单一 extlinux armbian 启动项及 uart2-m0、dwc3-peripheral 两个已有 overlay。
Armbian user_overlays 加载规则来自：
https://github.com/armbian/build/blob/main/config/bootscripts/boot-rockchip64.cmd
