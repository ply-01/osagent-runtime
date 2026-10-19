# 通用记忆质量评估工具

本次比赛的必跑验收入口为：

```bash
os-agent track1 --scenario examples/track1/maintenance-handover.json --output artifacts/track1
python -m pytest -q
```

`python benchmarks/memory_eval_pipeline.py` 默认只执行同一公开制造运维场景并归档实际结果，不请求模型、不下载外部数据。配置见 config/memory_eval_pipeline.json。
其余保留脚本是可选的通用质量或外部系统对比工具，需要自行提供合法数据与依赖；没有本轮实测证据时不能作为参赛性能或优势结论。
