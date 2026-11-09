# 摄像头兼容入口

摄像头现统一放在 [sensors/IMX219](../sensors/IMX219/README.md)，与 [扬声器和 VL53L8CX](../sensors/README.md) 并列。

`camera/IMX219` 保留为指向新目录的符号链接，原本地命令仍可使用。新部署、Git sparse checkout 和文档链接请使用 `sensors/IMX219`。在不支持符号链接的平台上也直接使用新目录。
