"""Lightweight project skill registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Skill:
    """A declarative skill loaded from a local SKILL.md file."""

    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    path: Path | None = None
    body: str = ""


class SkillRegistry:
    """Load simple Pi-style skills from a directory tree."""

    def __init__(self, root: Path | str = "skills") -> None:
        self.root = Path(root)
        self._skills: dict[str, Skill] = {}

    def load(self) -> None:
        self._skills.clear()
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*/SKILL.md")):
            skill = self._load_skill(path)
            if skill.name in self._skills:
                raise ValueError(f"duplicate skill: {skill.name}")
            self._skills[skill.name] = skill

    def list(self) -> list[Skill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def names(self) -> list[str]:
        return sorted(self._skills)

    def tools_for_skill(self, name: str) -> list[str]:
        return list(self._skills[name].tools)

    def _load_skill(self, path: Path) -> Skill:
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = self._split_frontmatter(text)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description", "")
        tools = self._split_csv(metadata.get("tools", ""))
        return Skill(
            name=name,
            description=description,
            tools=tools,
            path=path,
            body=body.strip(),
        )

    def _split_frontmatter(self, text: str) -> tuple[dict[str, str], str]:
        if not text.startswith("---\n"):
            return {}, text
        end = text.find("\n---\n", 4)
        if end == -1:
            return {}, text
        raw = text[4:end]
        body = text[end + len("\n---\n") :]
        metadata: dict[str, str] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
        return metadata, body

    def _split_csv(self, value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]
