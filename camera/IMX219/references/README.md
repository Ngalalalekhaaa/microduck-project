# 厂家资料与固定依赖

联网执行 `python3 scripts/fetch_assets.py`（先进入 IMX219 目录），下载
[sources.json](sources.json) 中的两个 ISP 安装包、Sony 数据手册和对应内核驱动参考。
每份文件均校验 SHA256；相同名称但哈希不匹配时停止，不自动采用新版。
`python3 scripts/fetch_assets.py --check` 只检查本地文件，不联网。

Git 仓库保留来源清单，不重复发布厂商二进制/PDF。交付用本地压缩包已缓存这些文件。
资料版权归各原作者；更多链接、版本及授权说明见 [资料来源](../docs/资料来源.md)。

驱动源码仅供解释开流时重设曝光的原因，部署不重新编译内核。
