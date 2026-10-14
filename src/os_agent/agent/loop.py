"""Phase-one Pi-style agent loop with Mem0-style memory architecture."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from os_agent.agent.context_builder import ContextBuilder
from os_agent.agent.hooks import AgentHook
from os_agent.agent.llm_planner import LLMPlanner
from os_agent.agent.plan import PlanStep, TaskPlan
from os_agent.agent.plan_validator import PlanValidator
from os_agent.agent.planner import SimplePlanner
from os_agent.events.logger import EventLogger
from os_agent.events.schemas import (
    AgentEvent,
    AgentRequest,
    AgentResponse,
    ToolCall,
    ToolResult,
)
from os_agent.events.session import SessionRecorder
from os_agent.llm.provider import LLMError, OpenAICompatibleProvider
from os_agent.memory.hook import MemoryHook
from os_agent.memory.markdown_store import MarkdownMemoryStore
from os_agent.memory.mem0_pipeline import ExtractUpdatePipeline
from os_agent.memory.memory_config import MemoryConfig
from os_agent.memory.module import AgentMemoryModule
from os_agent.memory.query_rewriter import QueryRewriter
from os_agent.skills.registry import SkillRegistry
from os_agent.subagents.memory import MemorySubagent, build_memory_subagent_from_env
from os_agent.tools.builtins import build_default_registry
from os_agent.tools.registry import ToolRegistry
from os_agent.tools.subagent import CallMemorySubagentTool, ManageMemorySubagentTool


class AgentLoop:
    """A bounded, observable agent loop with Mem0-style memory architecture.

    Memory flow:
      Session events → ExtractUpdatePipeline → CoreMemoryStore (structured)
                                              ↓
                         ArchivalMemoryStore (superseded + raw history)
                                              ↓
                         RecallStrategy (Core-first → Archival fallback)
    """

    def __init__(
        self,
        workspace: Path | str = ".",
        registry: ToolRegistry | None = None,
        event_logger: EventLogger | None = None,
        memory_store: MarkdownMemoryStore | None = None,
        skill_registry: SkillRegistry | None = None,
        planner: SimplePlanner | None = None,
        llm_planner: LLMPlanner | None = None,
        session_recorder: SessionRecorder | None = None,
        hooks: list[AgentHook] | None = None,
        use_llm: bool = True,
        max_steps: int = 5,
        use_mem0: bool = True,
        memory_subagent: MemorySubagent | None = None,
        memory_subagent_mode: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.event_logger = event_logger or EventLogger()
        self.session_recorder = session_recorder or SessionRecorder(
            self.event_logger.root.parent / "sessions"
        )
        self.memory_store = memory_store or MarkdownMemoryStore()
        self.registry = registry or build_default_registry(
            self.workspace,
            memory_root=self.memory_store.root,
        )
        self.memory_hook = MemoryHook(self.memory_store)
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_registry.load()
        # Mem0-style stores
        self.use_mem0 = use_mem0
        self.memory_module = (
            AgentMemoryModule(
                self.memory_store.root,
                markdown_store=self.memory_store,
            )
            if use_mem0
            else None
        )
        self.core_memory = self.memory_module.core if self.memory_module else None
        self.archival_memory = self.memory_module.archival if self.memory_module else None
        self.pipeline = (
            ExtractUpdatePipeline(self.memory_module.core, self.memory_module.archival, self.memory_store)
            if self.memory_module
            else None
        )
        self.context_builder = ContextBuilder(self.memory_store)
        self.planner = planner or SimplePlanner()
        self.llm_planner = llm_planner or LLMPlanner(OpenAICompatibleProvider())
        self.plan_validator = PlanValidator()
        self.hooks = hooks or []
        self.use_llm = use_llm
        self.max_steps = max_steps
        self.memory_config = MemoryConfig.global_config()
        self.query_rewriter = QueryRewriter()
        self.memory_subagent = memory_subagent or build_memory_subagent_from_env()
        configured_mode = (
            memory_subagent_mode
            or os.getenv("OS_AGENT_MEMORY_SUBAGENT_MODE")
            or "tool"
        ).strip().lower()
        if configured_mode not in {"tool", "automatic", "hybrid"}:
            raise ValueError(
                "memory_subagent_mode must be 'tool', 'automatic', or 'hybrid'"
            )
        self.memory_subagent_mode = configured_mode

    def run(self, user_input: str, session_id: str | None = None) -> AgentResponse:
        request = AgentRequest(
            content=user_input,
            session_id=session_id or f"session-{uuid4().hex[:12]}",
        )

        if (
            self.memory_subagent is not None
            and self.memory_subagent_mode in {"tool", "hybrid"}
        ):
            if CallMemorySubagentTool.name in self.registry.names():
                subagent_tool = self.registry.get(CallMemorySubagentTool.name)
                if isinstance(subagent_tool, CallMemorySubagentTool):
                    subagent_tool.subagent = self.memory_subagent
                    subagent_tool.session_id = request.session_id
                    subagent_tool.default_query = user_input
            else:
                self.registry.register(
                    CallMemorySubagentTool(
                        self.memory_subagent,
                        request.session_id,
                        user_input,
                    )
                )
            manager = getattr(self.memory_subagent, "manage", None)
            if callable(manager):
                if ManageMemorySubagentTool.name in self.registry.names():
                    manage_tool = self.registry.get(ManageMemorySubagentTool.name)
                    if isinstance(manage_tool, ManageMemorySubagentTool):
                        manage_tool.subagent = self.memory_subagent
                        manage_tool.session_id = request.session_id
                else:
                    self.registry.register(
                        ManageMemorySubagentTool(
                            self.memory_subagent,
                            request.session_id,
                        )
                    )

        self._emit(
            "agent_start",
            request,
            {"user_input": user_input, "workspace": str(self.workspace)},
        )
        self._emit_message(
            request=request,
            role="user",
            content=user_input,
            turn_index=1,
        )
        user_msg_event = AgentEvent(
            event_type="message_start",
            session_id=request.session_id,
            task_id=request.task_id,
            payload={"role": "user", "content": user_input},
        )
        user_candidates = []
        if not self.use_mem0:
            user_candidates = self.memory_hook.observe(user_msg_event)
        user_memory_titles = [c.title for c in user_candidates]
        self._emit("turn_start", request, {"turn_index": 1, "reason": "initial"})
        skills = self.skill_registry.list()
        subagent_context, subagent_memories = "", []
        if self.memory_subagent_mode in {"automatic", "hybrid"}:
            subagent_context, subagent_memories = self._recall_memory_subagent(
                user_input,
                request,
            )
        # Mem0-style recall: Core → Archival cascade
        mem0_context = ""
        if self.use_mem0 and self.memory_module is not None:
            recall_query = user_input
            if bool(self.memory_config.query_rewrite_enabled):
                recall_query = self.query_rewriter.rewrite(user_input)
            mem0_context = self.memory_module.recall_context_block(recall_query)
        # Legacy keyword retrieval (backward compat)
        include_legacy = bool(self.memory_config.recall_include_legacy_markdown_context)
        retrieved = self.memory_store.retrieve(user_input) if include_legacy else []
        context = self.context_builder.build(
            self.registry.names(),
            skills,
            query=user_input if include_legacy else "",
            skip_markdown=not include_legacy,
        )
        if mem0_context:
            context = mem0_context + "\n\n" + context
        if subagent_context:
            context = subagent_context + "\n\n" + context
        self._emit(
            "context_built",
            request,
            {
                "chars": len(context),
                "skills": [skill.name for skill in skills],
                "retrieved_memories": [
                    {
                        "source": item.source,
                        "title": item.title,
                        "score": item.score,
                    }
                    for item in retrieved
                ]
                + subagent_memories,
            },
        )

        results, tool_memory_titles, last_turn_index = self._execute_steps(user_input, context, request)
        memory_titles = user_memory_titles + tool_memory_titles

        response_text = self._respond(user_input, context, results, memory_titles, request)
        self._emit_message(
            request=request,
            role="assistant",
            content=response_text,
            turn_index=last_turn_index,
        )
        response = AgentResponse(
            content=response_text,
            session_id=request.session_id,
            task_id=request.task_id,
            steps=len(results),
            memory_candidates=memory_titles,
            retrieved_memories=[item.title for item in retrieved]
            + [item["title"] for item in subagent_memories],
        )
        self._emit(
            "agent_end",
            request,
            {
                "response": response.content,
                "steps": response.steps,
                "memory_candidates": response.memory_candidates,
                "retrieved_memories": response.retrieved_memories,
                "turns": last_turn_index,
            },
        )
        # Mem0 pipeline: batch-process session events into Core facts
        if self.use_mem0 and self.pipeline is not None:
            session_events = self.session_recorder.read(request.session_id)
            if session_events:
                pipeline_results = self.pipeline.process_session(
                    session_events,
                    session_id=request.session_id,
                )
                self.memory_module.refresh_semantic_indexes()
                if pipeline_results:
                    response.memory_candidates.extend(
                        f"[Core] {f.entity}.{f.attribute} = {f.value}"[:120]
                        for f in pipeline_results
                    )
        self._observe_memory_subagent(
            request=request,
            user_input=user_input,
            assistant_output=response_text,
        )
        return response

    def _recall_memory_subagent(
        self,
        query: str,
        request: AgentRequest,
    ) -> tuple[str, list[dict[str, object]]]:
        if self.memory_subagent is None:
            return "", []
        try:
            hits = self.memory_subagent.recall(query, session_id=request.session_id)
            context = self.memory_subagent.render_context(hits)
        except Exception as exc:  # noqa: BLE001 - optional subagents must fail open
            self._emit(
                "memory_subagent_recall_failed",
                request,
                {
                    "subagent": self.memory_subagent.name,
                    "error": str(exc),
                    "fail_open": True,
                },
            )
            return "", []
        memories = [
            {
                "source": self.memory_subagent.name,
                "title": f"{hit.kind}:{hit.memory_id}",
                "score": hit.score,
            }
            for hit in hits
        ]
        self._emit(
            "memory_subagent_recall_succeeded",
            request,
            {
                "subagent": self.memory_subagent.name,
                "hits": len(hits),
                "context_chars": len(context),
            },
        )
        return context, memories

    def _observe_memory_subagent(
        self,
        *,
        request: AgentRequest,
        user_input: str,
        assistant_output: str,
    ) -> None:
        if self.memory_subagent is None:
            return
        try:
            result = self.memory_subagent.observe_turn(
                session_id=request.session_id,
                user_input=user_input,
                assistant_output=assistant_output,
            )
        except Exception as exc:  # noqa: BLE001 - optional subagents must fail open
            self._emit(
                "memory_subagent_observe_failed",
                request,
                {
                    "subagent": self.memory_subagent.name,
                    "error": str(exc),
                    "fail_open": True,
                },
            )
            return
        self._emit(
            "memory_subagent_observe_succeeded",
            request,
            {
                "subagent": self.memory_subagent.name,
                "result": result,
            },
        )

    def _execute_steps(
        self,
        user_input: str,
        context: str,
        request: AgentRequest,
    ) -> tuple[list[ToolResult], list[str], int]:
        plan = self._create_plan(user_input, context, request)
        turn_index = 1
        self._emit_plan(request, plan, step_index=1, turn_index=turn_index)

        results: list[ToolResult] = []
        memory_titles: list[str] = []
        executed: set[tuple[str, str]] = set()
        step_index = 1
        calls = plan.tool_calls()
        while calls and len(results) < self.max_steps:
            call = calls.pop(0)
            self._mark_step_running(plan, call)
            key = self._call_key(call)
            if key in executed:
                self._emit(
                    "tool_call_skipped",
                    request,
                    {
                        **self._tool_call_payload(call),
                        "reason": "duplicate_call",
                        "turn_index": turn_index,
                    },
                )
                self._mark_step_done(plan, call, "duplicate skipped")
                continue
            executed.add(key)
            self._emit(
                "tool_call",
                request,
                {**self._tool_call_payload(call), "turn_index": turn_index},
            )
            blocked_reason = self._before_tool_call(request, call, turn_index)
            if blocked_reason:
                self._emit(
                    "tool_call_blocked",
                    request,
                    {
                        **self._tool_call_payload(call),
                        "turn_index": turn_index,
                        "reason": blocked_reason,
                    },
                )
                result = ToolResult(
                    tool_name=call.tool_name,
                    status="error",
                    input=call.arguments,
                    output={},
                    error=f"blocked by hook: {blocked_reason}",
                )
            else:
                result = self.registry.execute(call)
            result = self._after_tool_result(request, call, result, turn_index)
            self._mark_step_done(plan, call, self._tool_result_observation(result))
            results.append(result)
            event = self._emit(
                "tool_result",
                request,
                {
                    **self._tool_result_payload(result),
                    "tool_call_id": call.call_id,
                    "turn_index": turn_index,
                },
            )
            candidates = []
            if not self.use_mem0:
                candidates = self.memory_hook.observe(event)
            memory_titles.extend(candidate.title for candidate in candidates)

            step_index += 1
            self._emit(
                "turn_end",
                request,
                {
                    "turn_index": turn_index,
                    "tool_call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "status": result.status,
                    "error": result.error,
                },
            )
            if calls or len(results) >= self.max_steps:
                if calls and len(results) < self.max_steps:
                    turn_index += 1
                    self._emit(
                        "turn_start",
                        request,
                        {"turn_index": turn_index, "reason": "queued_tool_call"},
                    )
                continue
            follow_context = self._context_with_observations(context, results)
            follow_plan = self._revise_plan(
                user_input=user_input,
                plan=plan,
                context=follow_context,
                results=results,
                executed_keys=executed,
                request=request,
            )
            if self._should_stop(follow_plan, results):
                self._emit(
                    "plan_stopped",
                    request,
                    {
                        "plan": follow_plan.to_dict(),
                        "reason": "planner_stop",
                        "step_index": step_index,
                        "turn_index": turn_index,
                    },
                )
                break
            follow_up = [
                call for call in follow_plan.tool_calls() if self._call_key(call) not in executed
            ]
            if follow_up:
                turn_index += 1
                self._emit(
                    "turn_start",
                    request,
                    {"turn_index": turn_index, "reason": "plan_revision"},
                )
                self._emit_plan(
                    request,
                    follow_plan,
                    step_index=step_index,
                    turn_index=turn_index,
                )
                calls.extend(follow_up)
        return results, memory_titles, turn_index

    def _create_plan(
        self,
        user_input: str,
        context: str,
        request: AgentRequest,
    ) -> TaskPlan:
        self._last_plan_used_llm = False
        if self.use_llm and self.llm_planner.enabled:
            try:
                plan = self.llm_planner.create_plan(
                    user_input=user_input,
                    context=context,
                    tool_names=self.registry.names(),
                    max_steps=self.max_steps,
                    tool_descriptions=self.registry.descriptions(),
                )
                self._last_plan_used_llm = True
                self._emit(
                    "llm_plan_succeeded",
                    request,
                    {"plan": plan.to_dict()},
                )
                validated = self._validate_plan(
                    plan=plan,
                    request=request,
                    phase="initial",
                    max_steps=self.max_steps,
                    executed_keys=set(),
                )
                if (
                    not validated.has_pending_tool_calls()
                    and plan.has_pending_tool_calls()
                ):
                    self._emit(
                        "llm_plan_rejected",
                        request,
                        {
                            "reason": "no_executable_steps_after_validation",
                            "fallback": "simple_planner",
                        },
                    )
                else:
                    return validated
            except LLMError as exc:
                self._emit(
                    "llm_plan_failed",
                    request,
                    {"error": str(exc), "fallback": "simple_planner"},
                )
        plan = self.planner.create_plan(user_input, context)
        return self._validate_plan(
            plan=plan,
            request=request,
            phase="initial_fallback",
            max_steps=self.max_steps,
            executed_keys=set(),
        )

    def _revise_plan(
        self,
        user_input: str,
        plan: TaskPlan,
        context: str,
        results: list[ToolResult],
        executed_keys: set[tuple[str, str]],
        request: AgentRequest,
    ) -> TaskPlan:
        remaining_steps = max(self.max_steps - len(results), 0)
        if remaining_steps <= 0:
            return TaskPlan(
                objective=plan.objective,
                steps=[],
                confidence=plan.confidence,
                done=True,
                planner=plan.planner,
            )
        if self.use_llm and self.llm_planner.enabled:
            try:
                revised = self.llm_planner.revise_plan(
                    user_input=user_input,
                    previous_plan=plan,
                    context=context,
                    results=results,
                    tool_names=self.registry.names(),
                    max_steps=remaining_steps,
                    tool_descriptions=self.registry.descriptions(),
                )
                self._emit(
                    "llm_plan_revised",
                    request,
                    {"plan": revised.to_dict()},
                )
                validated = self._validate_plan(
                    plan=revised,
                    request=request,
                    phase="revision",
                    max_steps=remaining_steps,
                    executed_keys=executed_keys,
                )
                if validated.has_pending_tool_calls() or revised.done:
                    return validated
                self._emit(
                    "llm_plan_revision_rejected",
                    request,
                    {
                        "reason": "no_executable_steps_after_validation",
                        "fallback": "simple_planner",
                    },
                )
            except LLMError as exc:
                self._emit(
                    "llm_plan_revision_failed",
                    request,
                    {"error": str(exc), "fallback": "simple_planner"},
                )
        revised = self.planner.revise_plan(
            user_input=user_input,
            plan=plan,
            results=results,
            executed_keys=executed_keys,
            context=context,
        )
        return self._validate_plan(
            plan=revised,
            request=request,
            phase="revision_fallback",
            max_steps=remaining_steps,
            executed_keys=executed_keys,
        )

    def _should_stop(self, plan: TaskPlan, results: list[ToolResult]) -> bool:
        if plan.planner == "llm" and self.use_llm and self.llm_planner.enabled:
            return self.llm_planner.should_stop(plan, results)
        return self.planner.should_stop(plan, results)

    def _validate_plan(
        self,
        plan: TaskPlan,
        request: AgentRequest,
        phase: str,
        max_steps: int,
        executed_keys: set[tuple[str, str]],
    ) -> TaskPlan:
        result = self.plan_validator.validate(
            plan=plan,
            available_tools=self.registry.names(),
            max_steps=max_steps,
            executed_keys=executed_keys,
        )
        self._emit(
            "plan_validated",
            request,
            {
                **result.to_dict(),
                "phase": phase,
                "planner": plan.planner,
                "plan": result.plan.to_dict(),
            },
        )
        return result.plan

    def _emit_plan(
        self,
        request: AgentRequest,
        plan: TaskPlan,
        step_index: int,
        turn_index: int,
    ) -> None:
        self._emit(
            "plan_created",
            request,
            {
                "plan": plan.to_dict(),
                "calls": [self._tool_call_payload(call) for call in plan.tool_calls()],
                "planner": plan.planner,
                "step_index": step_index,
                "turn_index": turn_index,
            },
        )

    def _respond(
        self,
        user_input: str,
        context: str,
        results: list[ToolResult],
        memory_titles: list[str],
        request: AgentRequest,
    ) -> str:
        if self.use_llm and self.llm_planner.enabled:
            try:
                text = self.llm_planner.respond(
                    user_input=user_input,
                    context=context,
                    results=results,
                    memory_titles=memory_titles,
                )
                self._emit("llm_response_succeeded", request, {"chars": len(text)})
                return text
            except LLMError as exc:
                self._emit(
                    "llm_response_failed",
                    request,
                    {"error": str(exc), "fallback": "template_response"},
                )
        return self._compose_response(user_input, results, memory_titles)

    def _emit(self, event_type: str, request: AgentRequest, payload: dict) -> AgentEvent:
        event = AgentEvent(
            event_type=event_type,
            session_id=request.session_id,
            task_id=request.task_id,
            payload=payload,
        )
        self.event_logger.append(event)
        self.session_recorder.append(event)
        return event

    def _emit_message(
        self,
        request: AgentRequest,
        role: str,
        content: str,
        turn_index: int,
    ) -> None:
        message_id = f"msg-{uuid4().hex[:12]}"
        payload = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "chars": len(content),
            "turn_index": turn_index,
        }
        self._emit("message_start", request, payload)
        self._emit("message_end", request, payload)

    def _before_tool_call(
        self,
        request: AgentRequest,
        call: ToolCall,
        turn_index: int,
    ) -> str | None:
        for hook in self.hooks:
            decision = hook.before_tool_call(request, call, turn_index)
            if decision and decision.block:
                return decision.reason or "blocked"
        return None

    def _after_tool_result(
        self,
        request: AgentRequest,
        call: ToolCall,
        result: ToolResult,
        turn_index: int,
    ) -> ToolResult:
        current = result
        for hook in self.hooks:
            patched = hook.after_tool_result(request, call, current, turn_index)
            if patched is not None:
                current = patched
        return current

    def _tool_call_payload(self, call: ToolCall) -> dict:
        return {
            "tool_call_id": call.call_id,
            "tool_name": call.tool_name,
            "arguments": call.arguments,
            "reason": call.reason,
        }

    def _tool_result_payload(self, result: ToolResult) -> dict:
        return {
            "tool_name": result.tool_name,
            "status": result.status,
            "input": result.input,
            "output": result.output,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }

    def _compose_response(
        self,
        user_input: str,
        results: list[ToolResult],
        memory_titles: list[str],
    ) -> str:
        lines = [
            "OSAgent Runtime 已完成本轮任务。",
            "",
            f"用户输入: {user_input}",
            "",
            "工具执行:",
        ]
        for result in results:
            lines.append(
                f"- {result.tool_name}: {result.status}, {result.duration_ms}ms"
            )
            if result.tool_name == "filesystem.list_dir" and result.ok:
                entries = result.output.get("entries", [])
                names = ", ".join(item["name"] for item in entries[:12])
                lines.append(f"  目录样例: {names}")
            if result.tool_name == "filesystem.read_file" and result.ok:
                content = str(result.output.get("content", ""))
                lines.append(f"  文件摘要: {self._brief(content)}")
            if result.tool_name == "filesystem.read_pdf" and result.ok:
                pages = result.output.get("pages", [])
                total = result.output.get("total_pages", len(pages))
                chars = result.output.get("chars", 0)
                lines.append(f"  PDF 共 {total} 页，提取 {chars} 字符")
                # show first page excerpt
                if pages:
                    first_text = pages[0].get("text", "")
                    lines.append(f"  首页预览: {self._brief(first_text)}")
            if result.error:
                lines.append(f"  错误: {result.error}")
        if memory_titles:
            lines.extend(["", "记忆更新:"])
            lines.extend(f"- {title}" for title in memory_titles)
        return "\n".join(lines)

    def _brief(self, text: str) -> str:
        clean = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return clean[:600] if clean else "(empty file)"

    def _context_with_observations(
        self,
        context: str,
        results: list[ToolResult],
    ) -> str:
        observations = [
            f"- {result.tool_name}: {result.status}, error={result.error or ''}"
            for result in results
        ]
        return context + "\n\n# Observations\n" + "\n".join(observations)

    def _call_key(self, call: ToolCall) -> tuple[str, str]:
        return (call.tool_name, repr(sorted(call.arguments.items())))

    def _mark_step_running(self, plan: TaskPlan, call: ToolCall) -> None:
        step = self._find_step(plan, call)
        if step:
            step.status = "running"

    def _mark_step_done(self, plan: TaskPlan, call: ToolCall, observation: str) -> None:
        step = self._find_step(plan, call)
        if step:
            step.status = "done"
            step.observation = observation

    def _find_step(self, plan: TaskPlan, call: ToolCall) -> PlanStep | None:
        key = self._call_key(call)
        for step in plan.steps:
            if step.tool_call and self._call_key(step.tool_call) == key:
                return step
        return None

    def _tool_result_observation(self, result: ToolResult) -> str:
        if result.ok:
            return f"{result.tool_name} succeeded in {result.duration_ms}ms"
        return f"{result.tool_name} failed: {result.error or 'unknown error'}"
