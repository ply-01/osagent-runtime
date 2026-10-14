# 本次交付验证记录

日期：2026-10-14。运行平台：macOS arm64，Python 3.13.13。

## 实际运行结果

- 完整 pytest：375 passed，4 skipped，另有 30 个 subtests passed。
- 4 个跳过项是未包含 Open WebUI 上游/定制源码的前端检查，不代表前端验证通过。
- 公开制造运维场景：10/10 检查通过；输入为合成结构化数据，后端为真实记忆实现。
- 从源码 ZIP 解压到临时目录、创建新虚拟环境、安装依赖后：测试结果相同，场景仍 10/10 通过。
- 新 HTML 报告已在浏览器检查，显示总体通过、各步状态及可展开证据。
- CLI、MCP 和 HTTP 服务的 help 入口可运行。
- 无 LLM 的 CLI 实际执行目录列举和 README 文件读取，工具调用均成功。
- 正式 PDF 共 7 页，已提取文本并渲染检查中文、图表与命令排版。
- 演示视频为实际运行结果可视化，1280×720、30 fps、180 秒；全量解码 5400 帧通过，抽取解码画面检查。无音轨，提供字幕。

本次未验证 Windows/Linux 实机、麒麟 SDK、大规模并发、语义模型、真实客户环境或私有评测。
跨平台 CI 配置已提供，尚未在远程 CI 执行，不能当成实机通过记录。

## 复核入口

```bash
python -m pytest -q
os-agent track1 --scenario examples/track1/maintenance-handover.json --output artifacts/track1
python scripts/package-track1.py --output dist
```

工作目录保存了 `artifacts/track1-tests.log`、`artifacts/track1-tests.xml`、
`artifacts/release-tests.log`、`artifacts/track1-cli.json` 和 `artifacts/release-scenario.json`。
这些是本机运行产物，不默认混入源码包。收到源码包的评审应自行运行生成新的证据。
源码包中 SOURCE-MANIFEST.json 覆盖各文件 SHA-256，ZIP 同时检查 CRC 和解压后字节哈希。
