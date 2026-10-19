# 本次交付验证记录

日期：2026-10-19。目标平台：macOS、Linux、Windows。

## 实际运行结果

- 本机：macOS arm64，Python 3.13.13，327 passed、3 skipped、30 subtests passed。
- 3 个跳过项为未打包的可选 Open WebUI 上游源码检查，不计为前端通过。
- 公开制造运维场景：10/10 检查通过；合成输入，真实记忆实现；没有加载模型或请求 API。
- 无 LLM 的 CLI 实际执行目录列举和 README 文件读取，两个工具调用成功。
- CLI、MCP、HTTP 服务入口及相关回归通过。
- 可选通用评测流水线实际完成，默认只运行本次公开行业样例。
- 源码 ZIP 的逐文件 SHA-256 与 CRC 检查通过；重新安装和运行以该交付源码验证。
- 7 页作品介绍已重新渲染检查；3 分钟视频完整解码 5400 帧，1280×720，无音轨，提供字幕。

## 三平台远程验收

GitHub Actions 第 4 次流水线：六组任务全部成功，耗时 1 分 47 秒。

https://github.com/ply-01/osagent-runtime/actions/runs/39242910883

核心代码提交：39069ac08c6f60d250d429c3931ee746282d51b8。
此后提交仅同步验证说明、材料生成文字与源码哈希清单，不改变运行时。

| 平台 | Python | 验收范围 | 结果 |
|---|---|---|---|
| ubuntu-latest | 3.11 / 3.13 | 安装、完整回归、公开场景、打包 | 两组通过 |
| windows-latest | 3.11 / 3.13 | 安装、完整回归、公开场景、打包 | 两组通过 |
| macos-latest | 3.11 / 3.13 | 安装、完整回归、公开场景、打包 | 两组通过 |

## 复核入口

```bash
python -m pytest -q
os-agent track1 --scenario examples/track1/maintenance-handover.json --output artifacts/track1
python scripts/package-track1.py --output dist
```

当前测试数量反映退役系统专用能力后的保留功能，不与原 375 项机械比较。
本轮未验证真实客户、真实桌面控制闭环、语义模型质量、多进程并发和生产负载。
通过这些测试不能证明软件不存在任何缺陷；如有失败应保存原始结果并修复，不能改写报告。
