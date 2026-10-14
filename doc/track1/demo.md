# 部署与演示说明

## 本地演示

按 README 创建虚拟环境并安装。运行：

```bash
os-agent track1 --scenario examples/track1/maintenance-handover.json --output artifacts/track1
```

选择本次新生成的 `run-*` 目录，浏览器打开 `report.html`。
页面不连接外部服务。输入、输出和检查来自真实记忆模块；实际状态保存在该目录的 `memory/`。
重复运行不会修改旧目录。文件中 `generated_at` 是 UTC，平台信息标明运行机器。

## 建议录屏流程

赛道一官网未明确视频时长；以下是演示安排建议。

1. 说明制造运维跨班次知识失效问题，以及该案例使用合成数据。
2. 展示安装命令、样例 JSON 和运行命令；拍到退出状态及实际报告路径。
3. 展开持久化、流程更新、旧版审计、另一设备不受影响的输出证据。
4. 展示注销后的无召回和历史清理，说明删除范围仅限这次仿真运行。
5. 说明 AgentMemoryModule、MCP 接入以及可选麒麟适配，展示源码位置。
6. 说明当前限制、开源治理和后续真实试点计划。

录屏应展示功能与使用流程，上传后提供评审可访问的视频链接。
不要把 HTML 验证报告当成已经完成的演示视频。

## 可选服务入口

`os-agent-mcp --help`、`os-agent-web --help` 查看参数，默认保持回环绑定和服务端策略。
生产模型使用 config 中的 example 文件配置，不把真实凭据写入仓库。
Open WebUI 是可选组件，按 SUBMISSION-README 和上游固定版本单独获取；
银河麒麟部署按 deploy/galaxy-kylin-v11/README.md，目标机执行原生预检及向量 smoke。

## 故障排查

- `os-agent` 找不到：激活虚拟环境，或使用 `python -m os_agent`。
- 安装失败：检查 Python 版本及官方包源网络；该项目需要 MCP 2.x API，不应替换为不兼容 1.x。
- 样例路径找不到：在项目根目录运行，或传入绝对路径。
- 演示检查失败：保留新运行目录的 report.json，以非零退出状态报告失败，不能手改结果为成功。
- 麒麟预检在 macOS 失败：这是平台专属检查；通用赛道一入口是 track1。
