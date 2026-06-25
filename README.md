# GPU Watch

多服务器 GPU 实时监控 TUI 工具。类似 [nvitop](https://github.com/XuehaiPan/nvitop)，但可以同时监控多台远程服务器的 GPU 状态。

## 特性

- **远程零依赖**：通过 SSH 执行 NVML 探针，远程服务器只需 NVIDIA 驱动 + Python3
- **零磁盘 IO**：所有数据走 SSH stdin/stdout，内存解析，不写任何文件
- **交互式 TUI**：左侧 checkbox 勾选服务器，右侧实时显示 GPU 状态
- **进程高亮**：自己的 GPU 进程绿色高亮 + 完整命令行，其他用户按用户名聚合
- **nvtop 风格**：利用率条、GiB/MiB 内存单位、温度/功率颜色渐变
- **容错**：超时、离线、认证失败均有状态提示，不会因一台服务器挂了而崩溃

## 环境要求

**本地**：Python ≥ 3.10，`uv` 或 `pip`

**远程服务器**：
- NVIDIA 驱动（自带 `libnvidia-ml.so`）
- Python 3
- 可通过 SSH 免密登录

## 安装

```bash
git clone <repo-url>
cd CLI工具

# 方式一：uv（推荐）
uv sync
uv run gpuwatch

# 方式二：pip
pip install -e .
gpuwatch
```

## 使用

```bash
gpuwatch          # 或 python -m gpuwatch
```

| 按键 | 功能 |
|------|------|
| `↑` `↓` | 在左侧服务器列表中移动 |
| `Space` | 勾选/取消监控当前服务器 |
| `Tab` | 切换到右侧面板 |
| `r` | 强制刷新所有已选服务器 |
| `c` | 切换紧凑模式 |
| `q` | 退出 |

## 配置

### 自动发现

`gpuwatch` 自动读取 `~/.ssh/config`，列出所有 Host（github.com 等代码托管域名自动跳过）。

### 自定义配置（可选）

创建 `~/.config/gpuwatch/servers.yml`：

```yaml
refresh_seconds: 1.5
timeout_seconds: 5.0
servers:
  - host: two4090
    label: "2x RTX 4090"
    enabled: true          # 启动时自动勾选
  - host: a100-server
    label: "8x A100"
```

## 原理

```
本地 (Mac)                          远程 GPU 服务器
───────                             ──────────────
                                    python3 - < nvml_probe.py
ssh server 'python3 -' ──────────→   ├─ ctypes 加载 libnvidia-ml.so
  < nvml_probe.py                   ├─ nvmlDeviceGetUtilizationRates()
                                    ├─ nvmlDeviceGetMemoryInfo()
         ←  {"gpus":[{...}]}  ────  └─ JSON → stdout
         
本地 Textual TUI 渲染 ← 内存解析 JSON
```

- GPU 指标通过 NVML C 库直接获取（与 nvitop 同款方案）
- 进程信息优先用 NVML，权限不够时自动 fallback 到 `nvidia-smi`
- 全程零文件读写

## License

MIT
