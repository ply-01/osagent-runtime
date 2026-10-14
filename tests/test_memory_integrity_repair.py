"""Independent message-boundary and history API regression tests."""
from pathlib import Path

from os_agent.memory.memory_frame import MemoryFrameParser
from os_agent.mcp_server import OSAgentMCPConfig, OSAgentMCPService


def test_fence_is_not_an_attribute():
    parser = MemoryFrameParser()
    plain = parser.parse("我偏好深色主题")
    fenced = parser.parse("```markdown\n我偏好深色主题\n```")
    assert plain == fenced


def test_question_and_single_action_do_not_become_facts():
    parser = MemoryFrameParser()
    for text in ("我默认使用什么格式？请查一下。", "请创建一个 CSV 文件", "```markdown\n主题是什么？请回答。\n```"):
        assert parser.parse(text) is None


def test_project_slots_are_stable_and_scoped():
    parser = MemoryFrameParser()
    first = parser.parse("请记住：灯塔项目的默认编辑器是工具甲。")
    second = parser.parse("配置调整：灯塔项目的默认编辑器现在改为工具乙。")
    other = parser.parse("请记住：港湾项目的默认编辑器是工具丙。")
    assert first and second and other
    assert first.should_persist and second.should_persist
    assert first.attribute == second.attribute == other.attribute
    assert first.storage_entity == second.storage_entity != other.storage_entity
    assert (first.value, second.value) == ("工具甲", "工具乙")


def test_history_returns_actual_versions_without_mutation(tmp_path: Path):
    service = OSAgentMCPService(OSAgentMCPConfig.from_values(
        workspace=tmp_path, memory_root=tmp_path / "memory", use_llm=False,
        allow_memory_mutations=True,
    ))
    service.remember(content="Editor configuration", entity="lighthouse", attribute="editor", value="cedar")
    service.update(content="Editor configuration updated", entity="lighthouse", attribute="editor", value="birch")
    before = (tmp_path / "memory" / "archive_facts.json").read_bytes()
    result = service.recall(query="Previous configuration", entity="lighthouse", attribute="editor", history=True)
    assert [fact["value"] for fact in result["records"]] == ["birch", "cedar"]
    assert [fact["status"] for fact in result["records"]] == ["active", "superseded"]
    assert result["records"][0]["supersedes"] == result["records"][1]["fact_id"]
    assert (tmp_path / "memory" / "archive_facts.json").read_bytes() == before
