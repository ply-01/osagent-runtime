# 自有代码、第三方组件与许可证边界

本项目的自有代码采用 Apache-2.0 发布方案，见 LICENSE。
这是交付整理时选定的方案；原始快照没有作者清单或既有 LICENSE，
维护者必须在公开发布前核实贡献者授权和引入来源。本文件不是第三方代码的重新授权。

## 运行依赖

精确版本见 uv.lock；安装后的 dist-info 元数据与许可证随 Python 包保存。
下表列出直接依赖及可选依赖，传递依赖应随发行版清单一并审查。

| 组件 | 用途 | 上游许可 / 来源 |
|---|---|---|
| mcp 2.x | MCP 协议服务 | MIT，https://github.com/modelcontextprotocol/python-sdk |
| Starlette | HTTP 接口 | BSD-3-Clause，https://github.com/encode/starlette |
| Uvicorn | ASGI 服务 | BSD-3-Clause，https://github.com/encode/uvicorn |
| pdfplumber | PDF 提取 | MIT，https://github.com/jsvine/pdfplumber |
| pytest、httpx | 开发与测试 | MIT / BSD-3-Clause，https://github.com/pytest-dev/pytest 、https://github.com/encode/httpx |
| huggingface-hub、tokenizers、transformers | 可选模型接入 | Apache-2.0，https://github.com/huggingface |
| numpy | 可选向量运算 | BSD-3-Clause，https://github.com/numpy/numpy |
| onnxruntime | 可选本地推理 | MIT，https://github.com/microsoft/onnxruntime |
| torch | 可选本地推理 | BSD-3-Clause，https://github.com/pytorch/pytorch |
| playwright | 可选浏览器能力 | Apache-2.0，https://github.com/microsoft/playwright-python |
| hatchling | 打包构建 | MIT，https://github.com/pypa/hatch |

Python 安装依赖不等于源码包内包含其源代码。再分发含依赖的运行环境时应保留依赖许可证。
模型权重、训练数据和 API 服务条款独立于框架许可证；默认公开仿真不加载模型或请求 API。

## 可选前端与系统组件

- Open WebUI 固定上游 commit f9590b8017199e56d5e953657e6498e3cef1d246。
  来源 https://github.com/open-webui/open-webui 。其固定版本采用上游 Open WebUI License，
  含品牌相关条件，应保留上游许可、通知与标识，不能仅按本项目 Apache-2.0 分发。
  本包只提供自有接入与补丁；未包含上游完整前端。
- 银河麒麟 Embedding、向量引擎及系统 SDK 是操作系统提供的可选组件。
  权限与再分发由供应方授权决定。本包不包含授权 SDK、二进制库或模型。
- 旧评测入口可调用第三方系统或读取外部数据；相关包、数据和权利并非本项目原创。
  公开制造运维样例为本次编写的合成输入，不包含客户、个人或私有评测数据。

## 来源审查待办

原始源码 commit 声明仅来自旧清单，当前没有可核验的 Git 历史。
`mem0_*`、上游参考代码、前端补丁等由维护者确认实际来源；
如果含第三方改写或复制代码，补充原始作者、许可证与具体文件，不能用本文件抹去原始许可。
公开仓库发布前完成该审查，并填写维护者与贡献者信息。
