"""
ReAct Agent Core
─────────────────
Implements the Reasoning + Acting loop (ReAct pattern).

Each agent step:
  1. THINK  — reason about current state and decide what to do next
  2. ACT    — call a tool from the registry
  3. OBSERVE — receive the tool's result
  4. REFLECT — evaluate whether the goal is met or another step is needed

The loop continues until:
  - The agent produces a FINAL_ANSWER
  - Max steps is reached (safety cap)
  - A CRISIS is detected (immediate escalation)

This is model-agnostic: the reasoning step uses a rule-based planner
by default but can be swapped for any LLM that implements the
`ReasoningBackend` protocol (OpenAI, Anthropic, Ollama, etc.).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from agents.tools import Tool, ToolResult, TOOL_REGISTRY, get_tool
from utils.logger import get_logger

log = get_logger(__name__)

MAX_STEPS = 6   # hard cap to prevent infinite loops


# ─────────────────────────────────────────────────────────────────────────────
# Step types
# ─────────────────────────────────────────────────────────────────────────────

class StepType(str, Enum):
    THINK = "THINK"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    REFLECT = "REFLECT"
    FINAL_ANSWER = "FINAL_ANSWER"
    CRISIS_ESCALATE = "CRISIS_ESCALATE"


@dataclass
class AgentStep:
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[ToolResult] = None


@dataclass
class AgentTrace:
    """Complete trace of a single agent run — useful for debugging and audit."""
    agent_name: str
    user_message: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    is_crisis: bool = False
    total_steps: int = 0

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.total_steps += 1

    def to_summary(self) -> str:
        lines = [f"[{self.agent_name}] Trace ({self.total_steps} steps):"]
        for i, s in enumerate(self.steps, 1):
            if s.step_type == StepType.THINK:
                lines.append(f"  {i}. THINK: {s.content[:120]}…")
            elif s.step_type == StepType.ACT:
                lines.append(f"  {i}. ACT: {s.tool_name}({list((s.tool_args or {}).keys())})")
            elif s.step_type == StepType.OBSERVE:
                lines.append(f"  {i}. OBSERVE: {s.content[:120]}…")
            elif s.step_type == StepType.REFLECT:
                lines.append(f"  {i}. REFLECT: {s.content[:120]}…")
        lines.append(f"  → ANSWER: {self.final_answer[:200]}…")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Backend Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class ReasoningBackend(Protocol):
    """
    Any object implementing this protocol can drive the THINK step.
    Swap in an LLM client here for richer reasoning.
    """
    async def think(
        self,
        agent_name: str,
        user_message: str,
        observations: List[str],
        available_tools: List[str],
        context: Dict[str, Any],
    ) -> "ThinkOutput":
        ...


@dataclass
class ThinkOutput:
    reasoning: str
    next_action: str          # tool name OR "FINAL_ANSWER" OR "CRISIS_ESCALATE"
    tool_args: Dict[str, Any]
    is_done: bool


# ─────────────────────────────────────────────────────────────────────────────
# Rule-Based Reasoning Backend (default — no LLM required)
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedReasoner:
    """
    Deterministic planner that decides the next tool based on what
    observations are already available.
    """

    async def think(
        self,
        agent_name: str,
        user_message: str,
        observations: List[str],
        available_tools: List[str],
        context: Dict[str, Any],
    ) -> ThinkOutput:

        already_run = {s["tool"] for s in context.get("tools_run", [])}

        # ── Always start with NLP analysis ──────────────────────────────────
        if "analyze_message" not in already_run:
            return ThinkOutput(
                reasoning="I need to understand the user's emotional state and intent before acting.",
                next_action="analyze_message",
                tool_args={"text": user_message},
                is_done=False,
            )

        analysis = context.get("analysis", {})
        crisis_detected = analysis.get("crisis_detected", False)
        crisis_severity = analysis.get("crisis_severity", "low")

        # ── Crisis → immediate risk assessment ──────────────────────────────
        if crisis_detected and "assess_risk" not in already_run:
            return ThinkOutput(
                reasoning="Crisis signals detected. I must perform a risk assessment immediately.",
                next_action="assess_risk",
                tool_args={"text": user_message},
                is_done=False,
            )

        if crisis_detected and crisis_severity in ("critical", "high"):
            return ThinkOutput(
                reasoning="Risk is critical/high. Escalate to crisis resources immediately.",
                next_action="CRISIS_ESCALATE",
                tool_args={},
                is_done=True,
            )

        intent = analysis.get("intent", "vent")
        emotion = analysis.get("emotion", "neutral")

        # ── Fetch history for personalisation ───────────────────────────────
        if "get_user_history" not in already_run and context.get("user_id"):
            return ThinkOutput(
                reasoning="I should check the user's history to personalise my response.",
                next_action="get_user_history",
                tool_args={"user_id": context["user_id"], "db": "__db__"},
                is_done=False,
            )

        # ── Track progress if user checks in ────────────────────────────────
        if intent == "check_in" and "track_progress" not in already_run:
            return ThinkOutput(
                reasoning="User is checking in. I should review their progress trend.",
                next_action="track_progress",
                tool_args={"user_id": context.get("user_id"), "db": "__db__"},
                is_done=False,
            )

        # ── Generate coping plan for active distress ─────────────────────────
        if intent in ("vent", "seek_advice") and emotion in ("sadness", "fear", "anger"):
            if "generate_coping_plan" not in already_run:
                return ThinkOutput(
                    reasoning=f"User is experiencing {emotion} and needs active coping strategies.",
                    next_action="generate_coping_plan",
                    tool_args={
                        "emotion": emotion,
                        "stress_level": context.get("stress_level", 5),
                        "topics": context.get("recent_topics", []),
                    },
                    is_done=False,
                )

        # ── Get resource recommendations ──────────────────────────────────────
        if "recommend_resources" not in already_run:
            return ThinkOutput(
                reasoning="I have enough context to generate personalised resource recommendations.",
                next_action="recommend_resources",
                tool_args={
                    "user_id": context.get("user_id"),
                    "analysis_dict": analysis,
                    "db": "__db__",
                    "user_topic_weights": context.get("user_topic_weights", {}),
                    "user_modality_prefs": context.get("user_modality_prefs", {}),
                    "recent_topics": context.get("recent_topics", []),
                    "stress_level": context.get("stress_level", 5),
                },
                is_done=False,
            )

        # ── All needed tools have run → compile final answer ─────────────────
        return ThinkOutput(
            reasoning="I have all the information I need. Time to compile the response.",
            next_action="FINAL_ANSWER",
            tool_args={},
            is_done=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Abstract ReAct agent. Subclasses override `compile_answer()` to
    assemble the final response from collected observations.
    """

    name: str = "BaseAgent"
    tools: List[str] = []            # tool names this agent is allowed to use

    def __init__(self, reasoner: Optional[ReasoningBackend] = None):
        self._reasoner = reasoner or RuleBasedReasoner()

    async def run(
        self,
        user_message: str,
        context: Dict[str, Any],
    ) -> AgentTrace:
        """
        Execute the ReAct loop.

        Parameters
        ----------
        user_message : the raw user message this turn
        context      : shared context dict (user_id, db session, profile data, etc.)
        """
        trace = AgentTrace(agent_name=self.name, user_message=user_message)
        observations: List[str] = []
        context.setdefault("tools_run", [])

        log.info(f"[{self.name}] Starting ReAct loop for message: '{user_message[:60]}…'")

        for step_num in range(MAX_STEPS):
            # ── THINK ────────────────────────────────────────────────────────
            think_out = await self._reasoner.think(
                agent_name=self.name,
                user_message=user_message,
                observations=observations,
                available_tools=self.tools or list(TOOL_REGISTRY.keys()),
                context=context,
            )
            think_step = AgentStep(StepType.THINK, content=think_out.reasoning)
            trace.add_step(think_step)
            log.debug(f"[{self.name}] THINK: {think_out.reasoning}")

            # ── Check terminal states ────────────────────────────────────────
            if think_out.next_action == "FINAL_ANSWER":
                answer = self.compile_answer(context, observations)
                trace.final_answer = answer
                trace.add_step(AgentStep(StepType.FINAL_ANSWER, content=answer))
                log.info(f"[{self.name}] FINAL_ANSWER reached at step {step_num + 1}.")
                break

            if think_out.next_action == "CRISIS_ESCALATE":
                trace.is_crisis = True
                answer = self.compile_crisis_answer(context, observations)
                trace.final_answer = answer
                trace.add_step(AgentStep(StepType.CRISIS_ESCALATE, content=answer))
                log.warning(f"[{self.name}] CRISIS_ESCALATE triggered.")
                break

            # ── ACT ──────────────────────────────────────────────────────────
            tool_name = think_out.next_action
            tool = get_tool(tool_name)
            if tool is None:
                log.warning(f"[{self.name}] Tool '{tool_name}' not found. Stopping.")
                break

            act_step = AgentStep(
                StepType.ACT,
                content=f"Calling {tool_name}",
                tool_name=tool_name,
                tool_args=think_out.tool_args,
            )
            trace.add_step(act_step)

            # Inject db from context (marked as "__db__" sentinel in tool_args)
            resolved_args = {
                k: context.get("db") if v == "__db__" else v
                for k, v in think_out.tool_args.items()
            }

            result = await tool.fn(**resolved_args)

            # ── OBSERVE ──────────────────────────────────────────────────────
            observation = result.to_observation()
            observations.append(observation)
            trace.add_step(AgentStep(StepType.OBSERVE, content=observation, tool_result=result))
            log.debug(f"[{self.name}] OBSERVE: {observation[:200]}")

            # Update shared context with structured outputs
            context["tools_run"].append({"tool": tool_name, "success": result.success})
            self._update_context(tool_name, result, context)

        else:
            # Max steps hit — compile whatever we have
            trace.final_answer = self.compile_answer(context, observations)
            trace.add_step(AgentStep(StepType.FINAL_ANSWER, content=trace.final_answer))
            log.warning(f"[{self.name}] Max steps ({MAX_STEPS}) reached.")

        log.info(f"[{self.name}] Completed. Steps={trace.total_steps} Crisis={trace.is_crisis}")
        return trace

    def _update_context(self, tool_name: str, result: ToolResult, context: Dict[str, Any]) -> None:
        """Merge structured tool outputs back into the shared context."""
        if not result.success:
            return
        if tool_name == "analyze_message":
            context["analysis"] = result.output
        elif tool_name == "assess_risk":
            context["risk_assessment"] = result.output
        elif tool_name == "get_user_history":
            context["user_history"] = result.output
        elif tool_name == "generate_coping_plan":
            context["coping_plan"] = result.output
        elif tool_name == "recommend_resources":
            context["recommendations"] = result.output
        elif tool_name == "track_progress":
            context["progress"] = result.output

    def compile_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        """Override in subclasses. Default: join observations."""
        return "\n\n".join(observations[-3:])

    def compile_crisis_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        risk = context.get("risk_assessment", {})
        return risk.get("escalation_message", (
            "I'm very concerned about your safety right now. "
            "Please call or text 988 (Suicide & Crisis Lifeline) immediately. "
            "If you're in immediate danger, call 911."
        ))
