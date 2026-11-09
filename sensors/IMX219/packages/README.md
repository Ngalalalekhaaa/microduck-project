# ISP 软件包缓存

在 IMX219 目录运行 `python3 scripts/fetch_assets.py` 获取固定版本并校验 SHA256。
两份 `.deb` 的文件名、下载地址和哈希在 [sources.json](../references/sources.json)。

Radxa 执行 `sudo python3 scripts/setup_isp.py --apply --package-dir packages` 安装缓存包。
缓存只包含 ISP 引擎和 IQ；系统依赖仍需按主说明用 apt 安装，不是完整离线系统镜像。
