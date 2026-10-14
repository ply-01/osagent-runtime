# 2026-09-10 银河麒麟 V11 记忆与原生向量评测报告

## 1. 结论

OSagent 在银河麒麟桌面操作系统 V11 虚拟机上完成两轮冻结 500/10k 协议复测。两轮均强制使用 `KylinSystemEmbeddingProvider + KylinVectorEngine`，且在开始前通过 768 维 write/search/filter/update/delete 往返验证；两次评测进程退出码均为 0。

第二轮四项比赛指标为：偏好提取准确率 `90.0%`、知识 Recall@5 `74.0%`、10k 完整在线检索 p95 `844.928ms`、自然语言冲突处理正确率 `20.0%`，即 `1/4` 达标。第一、二轮的质量结果完全一致；完整请求 p95 分别为 `916.724ms` 和 `844.928ms`。

C++ 原生向量搜索阶段的 p95 为 `99.424-110.427ms`，但比赛延迟口径是包含 Query Embedding、结构化过滤、词法召回、融合及原生搜索在内的完整 Recall 请求，不能用约 100ms 的组件延迟替代 `844.928-916.724ms` 的端到端指标。

与首轮“系统 Embedding + NumPy 检索”的 `6174.272ms` 相比，第二轮原生完整请求 p95 降低 `86.32%`；但仍高于内部 `500ms` 阈值。当前性能瓶颈已经从向量搜索转移到 Query Embedding 和其余召回流程。

## 2. 数据与公平性

| 项目 | 固定值 |
|---|---|
| 冻结 stage | `~/.cache/osagent-v11-eval-20260909` |
| 数据目录 | `data/benchmarks/kylin-memory-eval-v5` |
| 协议标签 | `KylinMemoryEval-v1` |
| Manifest SHA-256 | `ce40257f57463a9222912032de97e4f2df8b0f2b7a792fe2d2ccafeaaf357256` |
| Seed commitment | `a5cec6883ffaf72e8601d6d275235d846d6a73ff0bfbaa178a8465a4bc688a8c` |
| 数据规模 | 偏好 500；事实 10,000；Query 500；冲突 500 |
| 检索设置 | Top-5，单 worker |
| 标签隔离 | `labels_visible_to_agent=false`；`labels_loaded_after_prediction=true` |

`v5` 是冻结评测包/修订目录名，`KylinMemoryEval-v1` 是结果中的协议标签，二者不是同一版本字段。输入身份以 manifest 哈希为准。

本次工作只读取聚合 JSON、运行状态和资源计时，不读取或发布 Query、标签、答案、失败句子及样例明细；没有依据本轮数据修改生产记忆规则，也没有训练模型。

## 3. 环境与原生链路

| 项目 | 实测配置 |
|---|---|
| 操作系统 | 银河麒麟桌面操作系统 V11，x86_64 |
| Python | 3.12.3 |
| Embedding | `ensemble-embd_gte-base_uint8-text`，768 维 |
| Python Provider | `KylinSystemEmbeddingProvider` |
| 向量后端 | `KylinVectorEngine` |
| C ABI bridge | `~/.local/lib/libosagent_kylin_vector_bridge.so` |
| 虚拟机资源 | 4 vCPU，8 GiB 内存 |

目标机 runtime 与 client 包存在 ABI 不一致。两轮使用同一兼容 preload 拉起系统 runtime，再由 Python 通过稳定 C ABI 调用 C++ client。该路径已完成真实 CRUD 和完整评测接线，但兼容库仍是部署时的临时适配层，不等同于厂商包 ABI 已统一。

第二轮在虚拟机重启后重新执行。系统 runtime 未自动启动，因此先恢复 runtime 和 embedding socket，再运行 CRUD smoke 与冻结评测。复测结束时 runtime、socket、V11 OSagent、V11 Open WebUI 和 Windows 转发入口均处于可用状态，三个 HTTP 入口均返回 200。

## 4. 对比设计

| 运行 | Embedding | 向量检索 | 用途 |
|---|---|---|---|
| V11 初始轮 | 麒麟系统模型 | NumPy | 对照原生 bridge 接入前的完整路径 |
| V11 原生第 1 轮 | 麒麟系统模型 | C++ `KylinVectorEngine` | 原生完整评测 |
| V11 原生第 2 轮 | 麒麟系统模型 | C++ `KylinVectorEngine` | 重启后的独立复测 |

三轮使用相同 manifest、样本规模、Top-K 和评分器。初始轮与原生轮的主要变量是向量检索后端；两轮原生结果用于观察稳定性，不以最佳轮代替真实区间。

## 5. 四项比赛指标

| 指标 | 内部阈值 | V11 NumPy 初始轮 | 原生第 1 轮 | 原生第 2 轮 | 第二轮通过 |
|---|---:|---:|---:|---:|---|
| 偏好提取准确率 | >=85% | 90.0% | 90.0% | 90.0% | 是 |
| 知识 Recall@5 | >=85% | 75.6% | 74.0% | 74.0% | 否 |
| 10k 完整在线检索 p95 | <=500ms | 6174.272ms | 916.724ms | 844.928ms | 否 |
| 自然语言冲突正确率 | >=88% | 20.0% | 20.0% | 20.0% | 否 |

第二轮结论：`1/4` 达标。原生 bridge 带来显著端到端加速，但不改变偏好与冲突评分；Recall@5 相对初始轮下降 `1.6` 个百分点，不能宣称质量提升。

## 6. 原生分阶段性能

### 6.1 在线请求

| 阶段 | 原生第 1 轮 p50 / p95 | 原生第 2 轮 p50 / p95 |
|---|---:|---:|
| 完整 Recall 请求 | 604.237 / 916.724ms | 581.447 / 844.928ms |
| Query Embedding | 263.176 / 488.040ms | 248.724 / 456.325ms |
| C++ 原生向量搜索 | 56.291 / 99.424ms | 63.888 / 110.427ms |
| 其余 Recall 流程 | 265.731 / 385.233ms | 255.167 / 369.859ms |

第二轮完整请求 mean/p99/max 为 `533.167/1146.308/1444.188ms`；第一轮为 `565.694/1072.253/7717.210ms`。各阶段分位数来自不同请求，不能直接相加。

第二轮相对第一轮的完整请求 p95 改善 `7.83%`。两轮 C++ search p95 区间为 `99.424-110.427ms`，说明原生搜索本身已不再是主要延迟来源；下一步应优先减少 Query Embedding 调用次数、复用查询向量，并压缩结构过滤与多路融合开销。

### 6.2 冷索引与资源

| 指标 | 原生第 1 轮 | 原生第 2 轮 |
|---|---:|---:|
| 冷索引总时间 | 1960993.915ms | 1151501.133ms |
| Document Embedding | 1733994.612ms | 917693.938ms |
| 非 Embedding 索引 | 226999.303ms | 233807.195ms |
| 整轮墙钟 | 37:31.92 | 23:46.45 |
| 峰值常驻内存 | 584920 KiB | 584584 KiB |
| Query Embedding 文本数 | 1791 | 1791 |
| 原生 search 调用数 | 2394 | 2394 |

第二轮冷索引相对初始 NumPy 轮的 `1125460.519ms` 慢 `2.31%`，可视为同一量级。冷索引主要由 Document Embedding 决定，C++ 检索后端不能消除首次文档向量化成本。

## 7. 质量分项

| 指标 | V11 NumPy 初始轮 | 原生第 1 轮 | 原生第 2 轮 |
|---|---:|---:|---:|
| Recall@5 | 75.6% | 74.0% | 74.0% |
| Complete Recall@5 | 74.2% | 72.2% | 72.2% |
| MRR | 70.42% | 72.92% | 72.92% |
| Exact | 100% | 100% | 100% |
| English paraphrase | 100% | 100% | 100% |
| Cross-lingual | 62% | 57% | 57% |
| Chinese paraphrase | 64% | 63% | 63% |
| Multi-fact | 52% | 50% | 50% |
| 结构化冲突 | 100% | 100% | 100% |
| 自然语言冲突 | 20% | 20% | 20% |

两轮原生质量结果完全一致，说明结果可重复。精确匹配和英文改写稳定；跨语言、中文语义改写和多事实召回仍是主要损失来源。结构化冲突满分而自然语言冲突偏低，瓶颈位于自然语言结构化与语义属性对齐，而非底层版本存储。

## 8. 完整性记录

| 文件或组件 | SHA-256 |
|---|---|
| 原生第 1 轮聚合 JSON | `0b5ff4cecb9da688dc311c453621580600023d327fee060638de85f070be7a86` |
| 原生第 1 轮 time | `c57cc4107cb81f3e80b99e05fd4834a744a81680fe611f8fd102b49816c9c34d` |
| 原生第 1 轮 status | `8d44a0730942b551f4388d9304ee2be9abcdd613417eb52ec9d40436b8b156d4` |
| 原生第 2 轮聚合 JSON | `c296f2cd187dd40084c2d3f2cc4d9c5e12e0ebc648aeb53deea718ba4134f686` |
| 原生第 2 轮 time | `bb084332cda5c81ee5878fff6bdd4ae435683ca3f5fe69bc3944232f0a893705` |
| 原生第 2 轮 status | `bd4128d1d7889956d47c6e659b981a7518a8098fb27678a337c7401d25367232` |
| 评测 runner | `740e0dc57eaec55e92fecb75ee187d49535bf0fd711a20b9aae04243daafc5a5` |
| C ABI bridge | `19e5b980905939b6dad6305b08dff01eddcd52a365cd7d9eb288058abac14646` |
| ABI 兼容库 | `5ad166fbc607075cac698eca1b7579ce71a6a395a5b010ed1ca906de0b9b7735` |

本地归档文件：

- `results/kylin-memory-eval-v5-v11-native-vector-2026-09-10.json`
- `results/kylin-memory-eval-v5-v11-native-vector-2026-09-10.time.txt`
- `results/kylin-memory-eval-v5-v11-native-vector-2026-09-10.status.txt`
- `results/kylin-memory-eval-v5-v11-native-vector-rerun-2026-09-10.json`
- `results/kylin-memory-eval-v5-v11-native-vector-rerun-2026-09-10.time.txt`
- `results/kylin-memory-eval-v5-v11-native-vector-rerun-2026-09-10.status.txt`

不归档评测 stdout/stderr，避免潜在输入文本进入项目材料。

## 9. 后续优化顺序

1. 合并并缓存重复 Query Embedding，减少当前每轮 `1791` 个文本的系统模型调用。
2. 对结构过滤、词法召回、RRF 和结果组装做批量化与候选集裁剪，将“其余 Recall”p95 从约 `370ms` 继续压低。
3. 在公开开发集上评估通用跨语言模型、Query 分解和 reranker；候选必须按同协议 A/B 且不能读取隐藏样例。
4. 对自然语言结构化提取采用 schema 约束与独立校准集，保留结构化事实版本治理，不增加面向句式的规则。
5. 将 runtime 与 ABI 兼容启动纳入 systemd 用户服务，验证重启自恢复，再作为生产部署闭环验收。
