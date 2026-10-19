# OSAgent Runtime：操作系统智能体开发运行平台

本项目参加第四届开放原子大赛·2026 开源行业解决方案创新赛，
**赛道一「开源基础软件与解决方案」**。

OSAgent Runtime 面向智能体开发与运行，提供有界规划、工具调用、事件轨迹和长期记忆。
其中，记忆模块将任务过程和业务知识保存为带证据、来源、版本和作用域的长期记忆，
通过 Agent 运行时、MCP 和 HTTP 接口为行业智能体提供记忆管理能力。
本次交付以**制造运维交接**作为公开仿真场景：跨班次保留设备维护知识，更新当前流程，
追溯被替代版本，并验证设备注销后的事实遗忘。

## 5 分钟复现

需要 Python 3.11+；支持 macOS、Linux、Windows。运行本地演示无需 API Key、模型下载或 Open WebUI。首次安装 Python 依赖需要网络；安装后该演示离线运行。完整测试还需要 `rg`（ripgrep）命令。
如已安装 uv，可用 `uv sync --frozen --extra dev` 复用锁定版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
os-agent track1 --scenario examples/track1/maintenance-handover.json --output artifacts/track1
python -m pytest -q
```

Windows PowerShell 使用 `py -m venv .venv`、`.venv\Scripts\Activate.ps1`，其余命令相同。
命令输出真实操作结果和检查计数；非通过状态返回非零退出码。
每次运行生成独立 `artifacts/track1/run-*/report.json` 和 `report.html`。
用浏览器打开 HTML 可查看每步输入、输出、耗时及验证证据，便于录制演示视频。
重复运行创建新的状态目录，不读取用户已有记忆。样例中的工厂、设备和审核单均为合成数据。

## 交付与评审入口

- [作品说明](doc/track1/solution.md)：背景、架构、创新点、应用场景、验证边界。
- [本次验证记录](doc/track1/validation.md)：实际测试结果、运行证据与未验证范围。
- [部署和演示](doc/track1/demo.md)：录屏步骤、扩展接入与故障排查。
- [评分映射与提交清单](doc/track1/submission.md)：三类材料、待填写信息和官方要求。
- [第三方与许可证](THIRD_PARTY_NOTICES.md)：依赖、前端、系统 SDK 和来源边界。
- [贡献指南](CONTRIBUTING.md)、[治理](GOVERNANCE.md)、[发展规划](ROADMAP.md)、[安全报告](SECURITY.md)。

可生成可复核的源码交付包：

```bash
python scripts/package-track1.py --output dist
```

脚本仅打包列明的源码、测试、配置示例和文档；排除虚拟环境、私有配置、运行状态、
模型及历史源码清单，生成逐文件 SHA-256。打包成功仅说明源码包完整，
不代表仓库链接、录屏、作者授权或报名审核已经完成。

## 架构

```text
行业智能体 / MCP Host / 可选 Open WebUI
                     ↓
          AgentLoop、工具、事件轨迹
                     ↓
       AgentMemoryModule 统一记忆接口
                     ↓
  Core 当前事实 / Archival 历史 / Recall 原始事件 / STM
                     ↓
  结构过滤 + 词法 / 可选语义检索 + RRF + 上下文压缩
                     ↓
     本地文件 / SQLite / 通用向量检索
```

核心实现位于 `src/os_agent/`，Open WebUI 接入位于 `integrations/`，
可选 Docker Compose 部署位于 `deploy/open-webui/`。
`os-agent --help`、`os-agent-mcp --help`、`os-agent-web --help` 可查看运行入口。
默认使用回环服务和有界工具权限；面向多租户生产部署仍需要身份、存储隔离和运维验收。

## 平台能力与验证边界

目标平台为 macOS、Linux、Windows，使用 Python 通用运行时。核心 Agent、文件工具、记忆、CLI、MCP、HTTP 和离线行业样例不依赖系统专用 SDK。
可选桌面输入依赖 Linux 的 xdotool/ydotool，未配置时保持关闭，不属于三平台核心验收。
可选浏览器功能需要单独安装浏览器与 Playwright。

公开演示使用显式结构化输入及词法检索；检查成功率不是自然语言抽取准确率、生产性能或官方比赛分数。
三平台 / Python 3.11、3.13 的 CI 会执行安装、完整回归、场景重放和打包；实际完成状态见 Actions。
当前本机验证范围和结果见 [验证记录](doc/track1/validation.md)。

## 开源与来源

自有代码按 [Apache-2.0](LICENSE) 提供；第三方组件遵守各自许可证。
原始目录是无 Git 历史的源码快照；改造后的版本已建立仓库：
https://github.com/ply-01/osagent-runtime 。目前为私有仓库，确认原代码公开授权后再开放评审访问。
正式报名名称为 OSAgent Runtime，Python 包和 CLI 沿用 os-agent-memory / os-agent，避免破坏兼容性。维护者须核实原始代码权属及
第三方引入来源，建立公开仓库、填写团队信息并补充演示视频链接后再提交。
项目不声明 Open WebUI、RRF、第三方模型或系统 SDK 为自主研发。
