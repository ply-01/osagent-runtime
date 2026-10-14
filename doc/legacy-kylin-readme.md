# OS Agent Memory

OSagent 是面向银河麒麟桌面操作系统 V11 的本地长期记忆智能体。项目已经从第一阶段骨架发展为
包含 Agent 运行时、分层记忆、混合检索、MCP、定制 Open WebUI 和麒麟原生向量桥接的完整系统。

它关注三件事：

- 让 agent 能根据用户输入规划任务、调用工具、返回结果。
- 把每次运行过程记录成结构化事件和 session trace，方便调试、回放和后续 subagent 审核。
- 从事件中自动抽取候选记忆，逐步沉淀项目知识、用户偏好和长期画像。

当前项目定位、能力边界和关键数据见 [项目总览](doc/project-overview.md)；完整技术细节见
[实现技术方案](doc/submission/2026-osagent-implementation-technical-solution.md)。

## 当前能力

- `AgentLoop`：有边界的主循环，负责上下文构建、规划、工具执行、响应生成和事件记录。
- `SimplePlanner`：不依赖大模型的规则型 planner，用于测试和兜底。
- `LLMPlanner`：接入 OpenAI-compatible API 后，由大模型生成结构化 `TaskPlan`。
- `PlanValidator`：在执行前校验、规范化、去重 planner 输出，并记录诊断事件。
- 工具系统：支持文件/PDF、受控 Shell、记忆 CRUD、插件生命周期和可选桌面后端能力。
- 事件日志：每轮运行写入 `data/events/YYYY-MM-DD.jsonl`。
- Session 持久化：每个 session 单独写入 `data/sessions/<session_id>.jsonl`。
- 结构化记忆：用 MemoryFrame、FactCard、状态槽和版本链管理偏好、事实、作用域和时态。
- 分层存储：STM、Session 摘要、跨 Session 全局摘要、Core、Archival 和 Recall 协同工作。
- 混合检索：结构过滤后融合 lexical、semantic、Session、Archival 和原始事件候选，并使用 RRF 排序。
- 上下文压缩：按查询蒸馏召回证据，避免原始事件无限进入模型窗口。
- 原生向量：支持银河麒麟系统 Embedding 和 C++ 向量引擎 bridge，提供严格预检与 CRUD smoke。
- 多入口：通过 MCP、Open WebUI Filter 和受控移动 API 复用同一记忆后端。
- Hook 机制：支持在工具调用前拦截、工具结果后修改。
- Skill 声明：从 `skills/*/SKILL.md` 加载项目能力说明。

## 项目结构

```text
src/os_agent/
  agent/        agent loop、planner、hook、上下文构建
  browser/      奇安信浏览器后端能力
  desktop/      受控桌面输入与截图适配
  events/       事件 schema、每日事件日志、session trace
  llm/          OpenAI-compatible LLM provider
  memory/       结构化记忆、分层存储、混合检索、压缩与维护
  plugins/      插件目录与生命周期
  remote/       移动指令协议、身份绑定和持久准入
  skills/       SKILL.md 加载器
  subagents/    可独立调用的记忆子 Agent
  tools/        文件、shell、memory 等工具

native/         银河麒麟 C++ vector bridge
integrations/   Open WebUI Filter 与接入组件
deploy/         银河麒麟 V11 和 Open WebUI 部署资产
vendor/         固定版本的 Open WebUI 源码与补丁
benchmarks/     冻结评测、公开数据适配和对照流水线
skills/         项目本地 skill 声明
config/         LLM 配置示例和本地配置
data/events/    运行时事件日志，默认不提交
data/sessions/  按 session 保存的运行轨迹，默认不提交
memory/         Markdown 记忆库
doc/            稳定架构、评测、答辩与部署文档
tests/          单元测试
```

## 安装

建议使用 Python 3.11 或更高版本。

```powershell
pip install -e .
```

`-e` 是 editable install。安装后：

- 修改 `src/os_agent/` 下的代码会立刻生效，无需重新安装。
- 系统会注册全局命令 `os-agent`，可在任意目录直接调用。

如果暂时不安装，也可以在当前 PowerShell 设置：

```powershell
$env:PYTHONPATH = "src"
python -m os_agent run "请总结 README.md"
```

## 命令行接口

安装后提供 `os-agent` 命令；运行 `os-agent --help` 可查看完整子命令列表。

### `os-agent run` — 运行 Agent

```powershell
os-agent run "请总结 README.md"                          # 基本用法
os-agent run "列出文件" --session-id demo-01              # 指定 session
os-agent run "快速测试" --no-llm                          # 关闭 LLM，使用规则 planner
os-agent run "任务" --workspace D:\project --max-steps 3  # 指定工作区和最大步数
os-agent run "任务" -v                                    # verbose 模式，打印诊断信息
```

### `os-agent config` — 管理 LLM 配置

```powershell
os-agent config show                                      # 查看当前配置（文件 + 环境变量）
os-agent config set --api-key sk-xxx --model deepseek-v4-flash  # 设置 API key 和模型
os-agent config set --base-url https://api.deepseek.com --timeout 120  # 设置地址和超时
```

配置文件默认存储在 `config/llm.local.json`，会自动在项目根目录查找，无需在项目目录下运行。

### `os-agent memory` — 管理记忆

```powershell
os-agent memory list                                      # 列出全部记忆（含 L1/L2/L3 分层和权重）
os-agent memory pending                                   # 查看待人工确认的记忆
os-agent memory maintain --max-sections-per-file 60       # 压缩去重 + 归档旧记忆
```

### `os-agent session` — 查看运行记录

```powershell
os-agent session list                                     # 列出所有 session
os-agent session show demo-01                             # 查看某个 session 的摘要
os-agent session show demo-01 --json --limit 20           # 输出原始 JSON 事件
```

### `os-agent subagent` — 运行独立记忆 Subagent

```powershell
os-agent subagent serve --host 127.0.0.1 --port 8765 --root data/subagents/native-memory
os-agent subagent health --url http://127.0.0.1:8765
```

服务使用独立 Core / Archival 存储并支持跨进程重启持久化。默认只允许绑定回环地址，
避免无认证接口被直接暴露到网络。

### 向后兼容

`python -m os_agent` 仍然可用，与 `os-agent` 命令完全等价：

```powershell
python -m os_agent run "请总结 README.md"
python -m os_agent config show
python -m os_agent memory list
```

## 配置大模型

项目支持 OpenAI-compatible chat completions API。DeepSeek 也可以按这个方式接入。

复制示例配置：

```powershell
Copy-Item config/llm.example.json config/llm.local.json
```

然后编辑 `config/llm.local.json`：

```json
{
  "api_key": "sk-your-api-key-here",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "embedding_backend": "openai",
  "embedding_model": "text-embedding-3-small",
  "embedding_base_url": "https://api.openai.com/v1",
  "timeout_seconds": 60,
  "embedding_timeout_seconds": 60,
  "temperature": 0.0
}
```

`config/*.local.json` 已经被 `.gitignore` 忽略，适合放本机秘钥。不要把真实 API key 写进 README、测试或公开文档。

也可以用环境变量覆盖配置：

```powershell
$env:OS_AGENT_API_KEY = "sk-..."
$env:OS_AGENT_BASE_URL = "https://api.deepseek.com"
$env:OS_AGENT_MODEL = "deepseek-v4-flash"
```

如果没有配置 API key，agent 会自动退回 `SimplePlanner` 和模板响应。

如果要使用离线多语言语义检索，安装可选依赖并建立本机配置：

```powershell
pip install -e ".[local-embeddings]"
Copy-Item config/embedding.local.example.json config/embedding.local.json
$env:OS_AGENT_EMBEDDING_CONFIG = "config/embedding.local.json"
```

本地后端默认示例为固定 revision 的 ONNX Runtime qint8
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)，
使用 E5 的 `query:` / `passage:` 前缀、mean pooling 和 L2 归一化。首次召回会将模型下载到
Hugging Face 用户缓存；将 `embedding_local_files_only` 改为 `true` 后可强制离线运行。
需要完整 PyTorch/Transformers 后端时，安装 `.[local-embeddings-torch]` 并把 backend 改为
`local_transformers`。

### 银河麒麟比赛生产配置

比赛生产链路必须使用银河麒麟系统 Embedding 接口，不能把上述 ONNX 后端当作正式替代。
在银河麒麟桌面操作系统 V11 上复制示例配置并执行预检：

```bash
cp config/embedding.kylin.example.json config/embedding.local.json
python -m os_agent competition preflight
python -m os_agent competition vector-smoke
```

`kylin_system` provider 通过 `ctypes` 调用 `libkysdk-coreai-embedding` 的 C ABI，复用一个
系统 session，并在模型初始化、结果复制和资源释放失败时显式报错。该后端是必需后端：SDK
或 runtime 不可用时不会静默降级为词法检索或其他 embedding provider。

预检会检查银河麒麟桌面 V11、Embedding 配置/动态库/runtime socket、系统向量数据库
动态库/runtime socket 和 OS Agent 的系统向量集成。系统向量数据库 C++ bridge 已接入事实与
session 语义索引；预检会检查 bridge 工件是否安装，部署后仍应执行一次真实的写入、检索和
删除 smoke test，避免仅凭系统包存在就声称启用了原生加速。`vector-smoke` 使用系统
Embedding 生成同维向量，再通过系统向量 SDK 完成 upsert、候选过滤、更新、Top-K 和 delete
验证；测试行会在退出前删除。机器相关的动态库位置可通过
`OS_AGENT_EMBEDDING_LIBRARY_PATH` 指定。

原生 bridge 安装时会针对目标机实际动态库自动探测两种 libstdc++ C++ ABI，并用 `ldd -r`
拒绝仍有未解析 SDK 符号的候选。服务能力接口会区分 bridge 工件“存在”和“可加载”，ABI
不一致时明确显示错误，不会把 fallback 误报成麒麟原生加速。

完整赛题要求和实施计划见：

- [2026 OS Agent 赛题要求](doc/competition/2026-youth-innovation-os-agent-requirements.md)
- [2026 OS Agent 记忆赛题改进计划](goals/2026-os-agent-competition.md)

### 银河麒麟桌面操作系统 V11 + Open WebUI 部署

当前部署目标固定为银河麒麟桌面操作系统 V11，采用宿主机原生 OSagent 服务和 Docker
Open WebUI 的组合。原生服务可直接访问当前桌面用户的 Kylin AI Runtime socket；WebUI
通过 host network 访问仅绑定在 `127.0.0.1` 的鉴权记忆接口。安装、systemd 用户服务、
专用 Compose、SSH 转发和预检说明见 [Galaxy Kylin V11 部署指南](deploy/galaxy-kylin-v11/README.md)。

OSagent 还提供受控插件市场和默认注册的 Agent MCP。插件市场可导入固定版本的 OpenAI
官方插件目录；HTTPS 包安装要求哈希校验，本地导入包要求固定源目录、路径边界和符号链接
检查，两种安装都需要显式开关。插件目录位于 Open WebUI 的工作区页面，负责搜索、安装、
启用、禁用和卸载。已安装 bundle 中的有界 `SKILL.md` 可作为声明式能力读取；不携带静态
凭据的远程 HTTPS MCP 声明可由插件页接入 Open WebUI。OAuth 2.1 凭据由 Open WebUI 按
用户保存，OSagent 不读取 token。命令型、本地进程型、HTTP 明文或内嵌 header/env 的连接
不会自动接入，下载包代码也不会被导入或执行。

安装与可调用是两个独立状态。对 Linear 这类远程 MCP 插件，安装后在
**工作区 -> 插件** 点击“连接”，完成服务方 OAuth 授权；授权返回后页面会进入对话并选中
对应的 MCP 工具。以后可直接点击“在对话中使用”。只有显示“可使用”的插件才具备真实工具
调用条件，是否实际调用仍取决于当前模型的 tool calling 能力和用户对该对话的工具授权。

当前版本已经移除聊天输入框旁的 OSagent 扩展控制面板，不再在对话框内展示 Computer Use、
文件权限或插件调用开关。Agent MCP 服务仍由部署配置全局注册；文件访问和可选 Computer Use
后端能力由认证后的服务端策略控制，前端不能自行放宽。Computer Use 仍缺少可靠视觉理解闭环，
不应宣传为通用自主桌面控制。历史 V11 键鼠链路验证见
[奇安信浏览器 Computer Use 验收报告](doc/evaluation/2026-09-03-galaxy-kylin-qianxin-computer-use.md)。

Open WebUI 的完整前后端源码固定在 `vendor/open-webui`（v0.11.0），OSagent
专用工作区页面及 MCP 启动自恢复补丁位于 `vendor/patches/open-webui/`。OSagent 自身的后端、MCP
工具和 Kylin 网关仍分别维护在 `src/`、`deploy/` 与 `integrations/`，便于升级
上游而不混淆运行时边界。

上游全仓类型检查存在大量既有诊断。OSagent 不通过关闭类型检查掩盖它们，也不在 fork
内大面积修改无关上游文件；改动面检查和基线处置见
[Open WebUI 类型检查策略](doc/open-webui-typecheck-policy.md)。

本地 UI 预览：

```powershell
cd vendor/open-webui
npm ci --force
npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

然后访问 `http://127.0.0.1:4173/workspace/plugins`。预览默认把 `/osagent/*`
请求转发到银河麒麟 V11 的 VirtualBox 端口 `33001`，因此插件目录和生命周期操作
可以直接使用虚拟机中的 OSagent。如果端口不同，可设置
`OSAGENT_PROXY_TARGET=http://127.0.0.1:<port>`。

语义检索会用同一个 provider 为 Core fact 和确定性的 session 摘要建向量，再与 Core lexical、
Archival 和原始事件 lexical 结果通过 RRF 合并。远程后端仍可使用 OpenAI-compatible
`/embeddings` 接口。未配置 `embedding_model` 时，上述长期记忆语义分支保持关闭；Markdown
向量检索仍可使用原有本地 hashing fallback。文档向量持久化到
`data/vector-cache/memory-documents.sqlite3`；query 向量仅保存在进程内，默认不提交 Git。
session embedding 文本会优先保留偏好信号，但召回返回完整的有界 session 摘要。
索引刷新后，Core 与 session 文档向量会以内容快照绑定的 float32 矩阵驻留内存，查询直接进行
矩阵相似度计算；内容改变时自动重建，SQLite 继续承担跨进程持久缓存。

## 查看 Session Trace

每次运行都会把事件按 session 写到：

```text
data/sessions/<session_id>.jsonl
```

列出已有 session：

```powershell
python -m os_agent session list
```

查看某个 session：

```powershell
python -m os_agent session show task-plan-smoke
```

输出原始 JSON 事件，方便后续 replay 或 subagent 审核：

```powershell
python -m os_agent session show task-plan-smoke --json
```

## 记忆系统

记忆系统采用 Mem0 风格的 **Core / Archival / Recall** 三层架构，默认开启。

完整的使用、配置与排障说明见 [Core 与 Markdown 记忆系统指南](doc/memory-system-guide.md)。

### 整体数据流

```text
AgentLoop 执行工具 / 收到用户消息
  → [session 结束] ExtractUpdatePipeline 批量处理事件
  → [Pipeline] 归档完整原始事件，并提取结构化 FactCard (entity + attribute + value)
  → [Pipeline] Core 插入/更新（同属性值变 → 版本+1、旧版归档）
  → [同步] 将 Core 当前事实 upsert 到 Markdown（可读镜像）
  → [检索] RecallStrategy: Core / session 摘要 / Archival / 原始事件并行候选 → RRF → distill
```

### Mem0 三层架构

| 层 | 存储 | 职责 |
|----|------|------|
| **Core** | `core_facts.json` | 结构化事实卡片，版本化管理，只保留最新活跃版本 |
| **Archival** | `archive_facts.json` + `raw_events.jsonl` | 过期/被替代的旧版事实 + 全量事件日志 |
| **Recall** | 检索策略（不是存储） | 合并 Core lexical/semantic、Archival 和原始事件候选，再做蒸馏 |

Core 是运行时唯一事实源。Markdown 只保存 Core 当前活跃事实的可读镜像，使用
`core_key` 和 `core_fact_id` 做关联：Core 更新会替换旧镜像，Core 删除会移除镜像，
旧版本保留在 Archival。默认不把 Markdown 作为独立检索源，因此不会出现两套记忆
各自参与决策的问题；`use_mem0=False` 时仍保留旧版 Markdown-first 流程。

隐式偏好先以 `preference_observation` 保存。只有来自至少两个独立 session、主题和正负极性
一致且关键词证据重合的 observation 才会晋升为 `preference_reinforced`；一次性意图、冲突证据
和混合偏好默认不进入稳定召回。最终 distillation 对同一 session 默认保留最多两个结果槽位，
减少重复片段挤占上下文。

### 统一记忆模块接口

现有记忆能力已经抽离为 `AgentMemoryModule`，对外提供稳定接口，避免其他 agent 直接依赖底层 Markdown 或 Mem0 文件实现：

```python
from os_agent.memory import AgentMemoryModule, RememberRequest, ForgetRequest, UpdateRequest

memory = AgentMemoryModule("memory")
memory.remember(RememberRequest(scope="ltm", content="...", entity="user", attribute="preference"))
memory.update(UpdateRequest(scope="ltm", entity="user", attribute="preference", value="..."))
memory.forget(ForgetRequest(scope="ltm", entity="user", attribute="preference"))
```

默认工具注册表也暴露了 tool-use 友好的工具名：

```text
memory.remember
memory.forget
memory.update
memory.recall
subagent.memory_search  ← 动态调用 Memory Subagent（需配置启用）
subagent.memory_manage  ← 对原生 Subagent 显式 remember/update/forget
```

其中 `ltm` 使用 Core / Archival / Recall 长期记忆，`stm` 使用轻量短期记忆存储。后续要接入任意 agent 或 `touse` wrapper 时，可以优先调用这层接口，而不是直接操作底层存储文件。

自然语言写入会先通过实体、作用域、属性和时态对齐器寻找同一逻辑事实；不同写法会共享
版本链，作用域不同或只有模糊值相似时则保持独立，避免把不相关偏好错误合并。检索 planner
会对 Unicode、中文问句和通用并列结构做有界归一化/分解，仍保留原始 query，并不读取测评
样例或训练固定答案。

移动端可以通过原生服务的 `/v1/mobile/commands` 提交指令，再轮询对应状态接口；银河麒麟
网关为手机使用单独的 `OS_AGENT_MOBILE_API_KEY`，不会把内部服务密钥暴露给客户端。完整的
端口、令牌和防火墙配置见 [Galaxy Kylin V11 部署指南](deploy/galaxy-kylin-v11/README.md)。
该静态移动密钥只适合单用户本地验证，不允许作为多客户共享凭据。商业化账号、二维码绑定、
设备出站 Relay、设备私钥和逐请求所属关系校验见
[手机端账号与电脑设备绑定方案](doc/commercial-mobile-device-linking.md)。

### 记忆抽取与偏好规范化

| # | 触发条件 | 记忆类型 | 置信度 | 写入文件 |
|---|---------|---------|--------|---------|
| 1 | `filesystem.read_file` 成功 | knowledge | 0.55 | project-knowledge.md |
| 2 | `filesystem.list_dir` 成功 | knowledge | 0.50 | project-knowledge.md |
| 3 | `shell.run_safe` 成功 | tool_history | 0.55 | tool-history.md |
| 4 | 任何工具**执行失败** | tool_history | 0.60 | tool-history.md |
| 5 | 用户偏好声明、配置或重复行为 | preference | 0.62-0.72 | profile.md |
| 6 | 用户消息匹配安全词 | safety | 0.55 | safety.md（首次） |

偏好抽取会区分陈述、疑问、否定和引用，并将偏好规范到文档格式、保存位置、界面主题、工具参数等稳定属性。属性不同的偏好可以共存；只有同一属性的新值才形成版本更新。重复工具参数默认至少观察三次才推断为行为偏好，敏感参数不会进入聚合证据。

召回会过滤通用英文填充词、规范轻量词形，并使用中文字符 unigram/bigram lexical 候选；配置 `embedding_model` 后会将 Core fact 与 session 摘要放入同一语义空间，再与 Archival/原始事件候选用 RRF 合并。session 摘要只抽取运行时事件中的用户消息、助手消息和工具结果，不读取评测标签。`ForgetRequest(hard=True)` 可物理清除 Core、Archival、原始事件归档、Markdown 镜像、来源事件和派生缓存。

安全触发词示例：`不要`、`禁止`、`千万别`、`don't`、`never`、`avoid`

### 记忆分层检索（时间衰减权重）

检索不再单纯按关键词匹配数排序，而是加入访问频率和时效性的考量：

```text
有效权重 = 加权分数 × 0.85^距上次访问天数

L1-hot:  有效权重 ≥ 3.0  → 第一层检索，优先返回
L2-warm: 有效权重 ≥ 0.5  → L1 不足时补充
L3-cold: 有效权重 < 0.5  → 最后兜底

每次检索命中：加权分数先衰减再加 1.0
```

效果：最近频繁访问的记忆优先返回，久不访问的记忆自然退化。

衰减率 0.85/天 ≈ 半衰期约 4.3 天（7 天不访问，权重减至约 0.32）。

### Priority 分页

默认分页仍按 hot/warm/cold 层级稳定展示。agent 构建运行时上下文时会使用 `priority` 分页，把当前任务 query 纳入评分：

```text
priority = 0.45 × relevance + 0.25 × recency + 0.20 × importance + 0.10 × layer
```

- `relevance`：当前任务 query 与记忆标题/内容的关键词或向量相似度。
- `recency`：最近命中时间的时间衰减分数。
- `importance`：优先读取记忆 metadata 中的 `importance`，其次根据 `confidence` 和记忆类型推断。
- `layer`：hot/warm/cold 的基础层级偏置。

`priority` 分页会按页保留 hot/warm/cold 的基础配额，避免常用记忆垄断全部上下文，也避免冷但强相关的记忆长期沉底。

### 记忆维护

```powershell
# 查看全部记忆（含分层和权重）
python -m os_agent memory list

# 只看待确认
python -m os_agent memory pending

# 压缩去重 + 归档旧记忆
python -m os_agent memory maintain --max-sections-per-file 80
```

查看稳定记忆分页和层级：

```powershell
python -m os_agent memory page --page 1 --page-size 10
python -m os_agent memory page --source project-knowledge.md --page 2
python -m os_agent memory page --layer hot
python -m os_agent memory page --strategy priority --query "planner memory"
python -m os_agent memory layers
python -m os_agent memory page --json
```

搜索稳定记忆：

```powershell
python -m os_agent memory search "embedding retrieval"
python -m os_agent memory search "embedding retrieval" --mode keyword
python -m os_agent memory search "embedding retrieval" --mode vector
python -m os_agent memory search "embedding retrieval" --mode hybrid --json
```

`vector` 模式会优先使用配置好的 semantic embedding provider；未配置或调用失败时回退到本地 hashing vectorizer。本地 hashing 只提供向量接口和余弦相似度排序，不等价于真正语义 embedding。

记忆分层索引存储在 `memory/.tier_index.json`，维护时会自动清理已删除的记忆条目。

### 记忆存储结构

```text
memory/
├── profile.md               ← preference 稳定记忆
├── project-knowledge.md     ← knowledge 稳定记忆
├── workflows.md             ← workflow 记忆（预留）
├── tool-history.md          ← tool_history 稳定记忆
├── safety.md               ← safety 稳定记忆
├── .tier_index.json         ← 分层访问索引（自动维护）
├── pending-confirmation/    ← 待确认 / 冲突记忆
└── archive/                 ← 维护时归档的旧记忆
```

### 记忆覆盖情况

| 记忆类型 | 抽取规则 | 写入路径 | 检索可见 |
|---------|---------|---------|---------|
| knowledge | 规则 1-2（读文件、列目录） | project-knowledge.md | ✅ |
| tool_history | 规则 3-4（shell 结果、工具失败） | tool-history.md | ✅ |
| preference | 规则 5（用户偏好检测） | profile.md 或 pending | ✅ |
| safety | 规则 6（安全约束检测） | safety.md 或 pending | ✅ |
| workflow | 预留（待多步分析） | workflows.md | ✅（空） |

### 动态调用记忆 Subagent

OSagent 现在支持将记忆 subagent 作为工具动态调用，而不是每次启动都固定调用。主 agent 可以根据查询复杂度智能选择：

- **简单查询** → 使用本地 `memory.recall`（快速，~44ms）
- **复杂查询** → 调用 `subagent.memory_search`（强大，~200ms）

原生 subagent 是一个真实的独立 HTTP 进程，拥有自己的 Core / Archival、自动会话抽取和
显式记忆管理接口。先在一个终端启动服务：

```powershell
os-agent subagent serve --host 127.0.0.1 --port 8765 --root data/subagents/native-memory
```

再在运行主 agent 的终端启用客户端：

```powershell
$env:OS_AGENT_MEMORY_SUBAGENT = "native"
$env:OS_AGENT_MEMORY_SUBAGENT_MODE = "tool"  # tool / automatic / hybrid
$env:OS_AGENT_NATIVE_SUBAGENT_URL = "http://127.0.0.1:8765"

os-agent run "找到所有关于项目配置的跨会话记忆"
```

**工作原理**：
1. Agent 检测到原生 subagent 后注册 `subagent.memory_search` 和 `subagent.memory_manage`
2. Planner 根据本地/跨会话职责边界决定检索、写入、更新或忘记
3. 每轮结束通过 `/v1/memory/observe` 自动持久化用户与助手消息
4. Subagent 离线时主流程记录失败事件并继续运行

**使用场景**：
- 跨 session 的复杂语义检索
- 大规模记忆库（10k+ facts）
- 多 agent 共享同一记忆服务
- 独立部署、重启后继续使用同一记忆库

详细配置、API 和使用示例见 [Subagent 动态调用指南](doc/subagent-tool-guide.md)。

默认 `tool` 模式只允许 LLM 显式调用，不会在规划前预调用。传统固定调用可设为
`automatic`；需要固定召回与动态工具并存时可设为 `hybrid`。

### 让 Codex 作为 MCP 宿主调用 OSagent

项目提供 `os-agent-mcp`，将 agent 运行、记忆召回以及受控的记忆变更暴露为 MCP 工具。
显式记忆变更默认关闭；只有启动时传入 `--allow-memory-mutations` 才能直接写入、更新或软删除记忆。
`os_agent_run` 仍保留 AgentLoop 原有的事件记录和自动记忆行为。

```powershell
pip install -e .
codex mcp add osagent -- python -m os_agent.mcp_server --workspace D:\OSagent --memory-root D:\OSagent\memory --no-llm
codex mcp list
```

这里的 `--no-llm` 让外层 Codex 负责问题分解和 query 改写，OSagent 负责确定性的本地执行与记忆检索。
完整配置、工具边界和公平测试口径见 [Codex 调用 OSagent](doc/codex-mcp-testing.md)。

## 开发和测试

运行全部测试：

```powershell
python -m unittest -q
```

或带 verbose：

```powershell
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前自动化测试包含 192 个 pytest 用例和 30 个参数化子测试，覆盖：

- LLM 配置加载和环境变量覆盖
- agent loop 事件写入
- session trace 写入和 CLI 查看
- LLM planner 和规则 planner
- planner 根据观察继续规划
- hook 拦截和结果修改
- 文件工具 workspace 边界
- Markdown 记忆 keyword/semantic-vector/hybrid 检索、多层分页、去重和维护
- 记忆 6 条抽取规则单元和集成测试（test_memory_hook.py, 27 个）
- 记忆分层检索和时间衰减测试（test_tiered_memory.py, 28 个）
- Memory Subagent HTTP 调用、CRUD、重启持久化和失败容错
- 动态 Subagent 工具注册、当前 session 绑定和参数回退
- 真实 LLM + 独立进程端到端验收（`benchmarks/native_subagent_e2e_eval.py`）

## 目前边界

这个项目还处于第一阶段。已经有 runtime、planner、tool、event、session、memory 的骨架，但还没有完成：

- 真正的 session replay executor
- 更完善的 embedding provider 管理和向量索引持久化
- 成熟的用户画像构建器
- 记忆审核 subagent
- 真实 GUI 自动化
- 更强的代码修改和补丁规划能力

后续比较自然的方向是：让 agent 从用户输入和工具结果中持续沉淀记忆，再通过 profile builder 形成可解释、可追溯、可压缩的用户画像。

## 最近更新

### 2026-08-08 — 动态 Subagent 工具调用

- 新增 `CallMemorySubagentTool`：将 Memory Subagent 包装为可调用工具
- Agent Loop 自动注册 `subagent.memory_search` 工具（当 subagent 可用时）
- LLM 可以根据查询复杂度自主决定何时调用 subagent（智能决策）
- 支持本地 memory + 远程 subagent 的混合策略
- 完整的测试覆盖（10 个 subagent 相关测试全部通过）
- 新增文档：[Subagent 动态调用指南](doc/subagent-tool-guide.md)
- 向后兼容：保留原有固定调用模式

### 2026-07-20 — Mem0 风格三层记忆架构

- 引入 Mem0 风格的 Core / Archival / Recall 三层记忆架构
- **Core** (`core_facts.json`)：结构化事实卡片（entity + attribute + value + version），自动去重、版本管理
- **Archival** (`archive_facts.json`)：过期/被替代的旧版事实 + 原始事件日志，按需深度检索
- **Recall**：Core 优先 → Archival 回退的级联检索策略
- **ExtractUpdatePipeline**：session 结束后批量处理事件，提取结构化事实，自动版本递增
- 同属性值不变 → 置信度自动强化；值变化 → 旧版归档、新版上位
- 默认开启 `--mem0` 模式，无需额外参数
- 新增 CLI 命令：`os-agent core list/search/stats`、`os-agent pipeline run/status`
- 旧版 Markdown 记忆降级为 Core 的可读镜像，不再作为独立检索源

### 2026-07-20 — CLI 独立命令行 + PDF 工具

- 注册 `os-agent` 全局命令（`pyproject.toml` `[project.scripts]`）
- CLI 重构为 argparse 子命令树：`run` / `config` / `memory` / `session`
- 新增 `config show/set` 子命令，LLM 配置无需手动编辑 JSON
- 新增 `--version`、`--max-steps`、`--verbose` 选项
- 新增 `filesystem.read_pdf` 工具（基于 pdfplumber），支持从 PDF 提取文本
- SimplePlanner 支持 `.pdf` 文件自动路由到 `read_pdf` 工具
- SimplePlanner 文件路径提取修复：不再误匹配中文字符
- LLM 配置文件跨目录查找：任意目录下运行 `os-agent` 都能找到项目配置
- 修复 `config set` 只传部分参数时错误覆写全部字段的 bug
- 保持 `python -m os_agent` 完全向后兼容
