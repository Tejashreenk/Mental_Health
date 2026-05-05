"""
Specialist Agents
──────────────────
Each agent is an expert in a specific mental health domain.
They all inherit BaseAgent but override:
  - `tools`          — the subset of tools they are permitted to use
  - `compile_answer` — domain-specific response assembly

Specialist agents:
  CrisisInterventionAgent  — immediate safety, de-escalation, hotline referrals
  CBTAgent                 — cognitive restructuring, thought challenging
  MindfulnessAgent         — breathing, grounding, meditation exercises
  GriefAgent               — loss, bereavement, stages of grief
  ProgressAgent            — tracks improvement, celebrates wins, flags declines
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from agents.react_agent import BaseAgent, RuleBasedReasoner, ThinkOutput
from agents.tools import ToolResult
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Crisis Intervention Agent
# ─────────────────────────────────────────────────────────────────────────────

class CrisisInterventionAgent(BaseAgent):
    name = "CrisisInterventionAgent"
    tools = ["analyze_message", "assess_risk", "recommend_resources"]

    def __init__(self):
        super().__init__(reasoner=_CrisisReasoner())

    def compile_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        risk = context.get("risk_assessment", {})
        recs = context.get("recommendations", [])

        lines = [
            "I hear you, and I want you to know that you matter.",
            "",
            risk.get("escalation_message", "Please reach out to a crisis counsellor right now."),
        ]

        if recs:
            lines += ["", "**Immediate support resources:**"]
            for r in recs[:3]:
                lines.append(f"• **{r['title']}**")
                if r.get("url"):
                    lines.append(f"  {r['url']}")

        lines += [
            "",
            "You don't have to face this alone. A trained counsellor is available 24/7.",
            "Would you be willing to reach out to one of these resources right now?",
        ]
        return "\n".join(lines)

    def compile_crisis_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        return self.compile_answer(context, observations)


class _CrisisReasoner(RuleBasedReasoner):
    """Crisis agent always runs risk assessment first, then recommends crisis resources."""

    async def think(self, agent_name, user_message, observations, available_tools, context):
        already_run = {s["tool"] for s in context.get("tools_run", [])}

        if "analyze_message" not in already_run:
            return ThinkOutput(
                reasoning="Analysing message for crisis signals.",
                next_action="analyze_message",
                tool_args={"text": user_message},
                is_done=False,
            )
        if "assess_risk" not in already_run:
            return ThinkOutput(
                reasoning="Running targeted risk assessment.",
                next_action="assess_risk",
                tool_args={"text": user_message},
                is_done=False,
            )
        if "recommend_resources" not in already_run:
            analysis = context.get("analysis", {})
            return ThinkOutput(
                reasoning="Fetching crisis and support resources.",
                next_action="recommend_resources",
                tool_args={
                    "user_id": context.get("user_id"),
                    "analysis_dict": analysis,
                    "db": "__db__",
                    "user_topic_weights": {},
                    "user_modality_prefs": {},
                    "recent_topics": ["crisis"],
                    "stress_level": 10,
                },
                is_done=False,
            )
        return ThinkOutput(reasoning="All crisis steps complete.", next_action="FINAL_ANSWER",
                           tool_args={}, is_done=True)


# ─────────────────────────────────────────────────────────────────────────────
# CBT Agent
# ─────────────────────────────────────────────────────────────────────────────

class CBTAgent(BaseAgent):
    name = "CBTAgent"
    tools = ["analyze_message", "get_user_history", "generate_coping_plan", "search_resources", "recommend_resources"]

    def compile_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        analysis = context.get("analysis", {})
        coping_plan = context.get("coping_plan", {})
        recs = context.get("recommendations", [])
        emotion = analysis.get("emotion", "difficult feelings")

        openers = [
            f"It sounds like you're dealing with {emotion}. That's really tough, and it takes courage to talk about it.",
            f"I can see you're experiencing {emotion}. You're not alone in this.",
            f"Thank you for sharing. What you're feeling — {emotion} — makes complete sense given what you're going through.",
        ]
        lines = [random.choice(openers), ""]

        if coping_plan.get("plan_steps"):
            lines.append("**Here's a CBT-based plan for right now:**")
            for step in coping_plan["plan_steps"]:
                lines.append(step)
            lines.append("")

        if recs:
            lines.append("**Resources that have helped others in similar situations:**")
            for r in recs[:3]:
                duration = f" (~{r['duration_minutes']} min)" if r.get("duration_minutes") else ""
                lines.append(f"• **{r['title']}**{duration} — {r.get('reason', '')}")
                if r.get("url"):
                    lines.append(f"  🔗 {r['url']}")

        lines += [
            "",
            "Would you like to try one of these now, or would you prefer to talk more about what's going on?",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Mindfulness Agent
# ─────────────────────────────────────────────────────────────────────────────

class MindfulnessAgent(BaseAgent):
    name = "MindfulnessAgent"
    tools = ["analyze_message", "search_resources", "recommend_resources"]

    def __init__(self):
        super().__init__(reasoner=_MindfulnessReasoner())

    def compile_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        analysis = context.get("analysis", {})
        recs = context.get("recommendations", [])
        emotion = analysis.get("emotion", "stress")

        lines = [
            f"Let's bring your attention gently to the present moment. {emotion.capitalize()} can feel overwhelming, but we can work through this together.",
            "",
            "**Try this right now (takes 2 minutes):**",
            "Close your eyes if comfortable. Take a slow breath in through your nose for 4 counts.",
            "Hold gently for 2. Exhale slowly through your mouth for 6 counts.",
            "Repeat 3 times. Notice how your body feels after.",
            "",
        ]

        if recs:
            lines.append("**Mindfulness exercises recommended for you:**")
            for r in recs[:3]:
                duration = f" ({r['duration_minutes']} min)" if r.get("duration_minutes") else ""
                lines.append(f"• **{r['title']}**{duration}")
                if r.get("url"):
                    lines.append(f"  🔗 {r['url']}")

        lines.append("\nHow are you feeling after the breathing exercise?")
        return "\n".join(lines)


class _MindfulnessReasoner(RuleBasedReasoner):
    async def think(self, agent_name, user_message, observations, available_tools, context):
        already_run = {s["tool"] for s in context.get("tools_run", [])}
        if "analyze_message" not in already_run:
            return ThinkOutput("Analysing message.", "analyze_message", {"text": user_message}, False)
        if "recommend_resources" not in already_run:
            analysis = context.get("analysis", {})
            return ThinkOutput(
                "Finding mindfulness resources.",
                "search_resources",
                {
                    "topics": ["anxiety", "stress", "sleep"],
                    "emotions": [analysis.get("emotion", "fear")],
                    "resource_types": ["exercise", "technique"],
                    "difficulty": "easy",
                    "db": "__db__",
                },
                False,
            )
        return ThinkOutput("Done.", "FINAL_ANSWER", {}, True)


# ─────────────────────────────────────────────────────────────────────────────
# Grief Agent
# ─────────────────────────────────────────────────────────────────────────────

class GriefAgent(BaseAgent):
    name = "GriefAgent"
    tools = ["analyze_message", "get_user_history", "search_resources", "recommend_resources"]

    def compile_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        recs = context.get("recommendations", [])

        lines = [
            "Grief is one of the heaviest things a person can carry. There's no right way to grieve, and no timeline you have to follow.",
            "",
            "What you're feeling is real, and it deserves to be honoured — not rushed.",
            "",
        ]

        if recs:
            lines.append("**Resources that may offer some comfort:**")
            for r in recs[:3]:
                lines.append(f"• **{r['title']}** — {r.get('reason', '')}")

        lines += [
            "",
            "Is there a specific part of your grief you'd like to explore, or would you just like to be heard right now?",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Progress Tracking Agent
# ─────────────────────────────────────────────────────────────────────────────

class ProgressAgent(BaseAgent):
    name = "ProgressAgent"
    tools = ["analyze_message", "track_progress", "get_user_history"]

    def __init__(self):
        super().__init__(reasoner=_ProgressReasoner())

    def compile_answer(self, context: Dict[str, Any], observations: List[str]) -> str:
        progress = context.get("progress", {})
        history = context.get("user_history", {})
        profile = history.get("profile", {})

        trend = progress.get("sentiment_trend", "unknown")
        note = progress.get("progress_note", "")
        crisis_eps = progress.get("crisis_episodes", 0)

        lines = [note, ""]

        if trend == "improving":
            lines += [
                "🌱 Your mood has been trending upward across your recent sessions.",
                "That's real progress — even if it doesn't always feel that way.",
            ]
        elif trend == "declining":
            lines += [
                "I've noticed things have been getting harder for you lately.",
                "That's important information. It might be a good time to talk to a professional who can provide deeper support.",
            ]
        else:
            lines.append("Your mood has been relatively stable — consistency is valuable.")

        if crisis_eps > 0:
            lines += [
                f"",
                f"I also want to acknowledge the difficult moments you've been through. You've kept showing up — that matters.",
            ]

        stress = profile.get("stress_level", 5)
        lines += [
            "",
            f"Your current stress level: **{stress}/10**.",
            "Would you like to talk about what's been happening, or explore some strategies to keep building on your progress?",
        ]
        return "\n".join(lines)


class _ProgressReasoner(RuleBasedReasoner):
    async def think(self, agent_name, user_message, observations, available_tools, context):
        already_run = {s["tool"] for s in context.get("tools_run", [])}
        if "analyze_message" not in already_run:
            return ThinkOutput("Analysing message.", "analyze_message", {"text": user_message}, False)
        if "track_progress" not in already_run:
            return ThinkOutput("Tracking progress.", "track_progress",
                               {"user_id": context.get("user_id"), "db": "__db__"}, False)
        if "get_user_history" not in already_run:
            return ThinkOutput("Getting user history.", "get_user_history",
                               {"user_id": context.get("user_id"), "db": "__db__"}, False)
        return ThinkOutput("Done.", "FINAL_ANSWER", {}, True)
