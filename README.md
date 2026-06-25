# GPU Watch

同时看多台远程服务器的 GPU 状态。类似 [nvitop](https://github.com/XuehaiPan/nvitop)，但是跨机器的。

## 它能干什么

左边勾选服务器，右边实时显示每块 GPU 的利用率、显存、温度、功率。你自己的 GPU 进程会高亮出来，别人的进程按用户名合并显示，不用在一堆 PID 里找自己的。

远程服务器不需要装任何东西，有 NVIDIA 驱动和 Python 3 就行。数据全部通过 SSH 传输，不写文件。

## 安装

```bash
git clone https://github.com/GPIOX/gpuwatch
cd gpuwatch
uv sync
uv run gpuwatch
```

或者一行搞定：

```bash
uv tool install git+https://github.com/GPIOX/gpuwatch && gpuwatch
```

本地需要 Python 3.10 以上。远程服务器需要 NVIDIA 驱动和 Python 3，配置好 SSH 免密登录。

## 使用

```bash
gpuwatch
```

| 按键 | 作用 |
|------|------|
| `↑` `↓` | 在左侧服务器列表里移动 |
| `Space` | 勾选或取消当前服务器 |
| `r` | 强制刷新所有已选服务器 |
| `c` | 紧凑模式 |
| `q` | 退出 |

## 配置

启动时自动读 `~/.ssh/config`，把里面的 Host 都列出来（github.com 这类代码托管域名自动跳过）。

如果想自定义显示名或默认勾选某些服务器，可以建 `~/.config/gpuwatch/servers.yml`：

```yaml
refresh_seconds: 1.5
timeout_seconds: 5.0
servers:
  - host: two4090
    label: "2x RTX 4090"
    enabled: true
  - host: a100-server
    label: "8x A100"
```

## 原理

本地通过 SSH 把一段 Python 脚本发到远程服务器执行。脚本用 ctypes 直接调 NVIDIA 驱动的 C 库（`libnvidia-ml.so`）拿 GPU 数据，和 nvitop 一样的方式。结果以 JSON 格式从 stdout 返回来，本地解析后渲染。

GPU 进程信息优先用 NVML 接口拿。如果当前用户没权限看其他用户的进程，会自动换 `nvidia-smi` 来查。

整个过程不创建任何临时文件。

## License

MIT
