"""Law Agent LangGraph StateGraph definition.

Graph topology:
    analyze_law -> check_routing -> (parallel) call_tax + call_compliance -> aggregate -> END

The parallel branches are dispatched via LangGraph's Send API so that
specialist sub-agent calls happen concurrently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3
LLM_TIMEOUT_SECONDS = 45


def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class LawState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str


async def _ask_llm(messages: list, fallback: str) -> str:
    """Call the LLM with a bounded wait so A2A requests do not hang."""
    try:
        result = await asyncio.wait_for(
            get_llm().ainvoke(messages),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return result.content
    except Exception as exc:
        logger.warning("LLM call failed or timed out: %s", exc)
        return f"{fallback}\n\n[Fallback used because the LLM was unavailable or too slow: {exc}]"


async def analyze_law(state: LawState) -> dict:
    """LLM analysis from a contract / general law perspective."""
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney specialising in contract law, "
                "tort law, and general business law. Analyse the legal aspects of the question "
                "thoroughly, covering relevant statutes, case law principles, and liability exposure. "
                "Keep the response under 180 words."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    fallback = (
        "Contract breaches can create exposure for expectation damages, foreseeable "
        "consequential damages, specific performance in limited cases, injunctions, "
        "and attorney-fee shifting if the contract or statute allows it."
    )
    return {"law_analysis": await _ask_llm(messages, fallback)}


async def check_routing(state: LawState) -> dict:
    """Determine whether tax and/or compliance sub-agents are needed."""
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping sub-agents", depth)
        return {"needs_tax": False, "needs_compliance": False}

    question_lower = state["question"].lower()
    needs_tax = any(
        kw in question_lower
        for kw in ["tax", "irs", "revenue", "fbar", "fatca", "thuế", "avoids taxes"]
    )
    needs_compliance = any(
        kw in question_lower
        for kw in ["compliance", "regulatory", "regulation", "sec", "sox", "aml", "fcpa"]
    )
    logger.info(
        "Routing decision: needs_tax=%s needs_compliance=%s",
        needs_tax,
        needs_compliance,
    )
    return {"needs_tax": needs_tax, "needs_compliance": needs_compliance}


def route_to_subagents(state: LawState) -> list[Send]:
    """Dispatch parallel Send objects based on routing flags."""
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    if not sends:
        sends.append(Send("aggregate", state))
    return sends


async def call_tax(state: LawState) -> dict:
    """Delegate to the Tax Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("tax_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Tax Agent returned %d chars", len(result))
        return {"tax_result": result}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}


async def call_compliance(state: LawState) -> dict:
    """Delegate to the Compliance Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("compliance_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Compliance Agent returned %d chars", len(result))
        return {"compliance_result": result}
    except Exception as exc:
        logger.exception("call_compliance failed: %s", exc)
        return {"compliance_result": f"[Compliance analysis unavailable: {exc}]"}


async def aggregate(state: LawState) -> dict:
    """Combine law, tax, and compliance analyses into a final answer."""
    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance Analysis\n{state['compliance_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response for the client. Combine the following "
                "analyses into a cohesive answer with clear sections. Avoid redundancy. "
                "Keep the response under 350 words and end with a brief educational disclaimer."
            )
        ),
        HumanMessage(content=combined),
    ]
    fallback = (
        f"{combined}\n\n"
        "## Practical Next Steps\n"
        "Preserve documents, quantify damages, review reporting duties, evaluate tax exposure, "
        "and consult licensed counsel before making admissions or settlement offers."
    )
    return {"final_answer": await _ask_llm(messages, fallback)}


def create_graph():
    """Build and compile the Law Agent StateGraph."""
    graph = StateGraph(LawState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_subagents,
        ["call_tax", "call_compliance", "aggregate"],
    )
    graph.add_edge("call_tax", "aggregate")
    graph.add_edge("call_compliance", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
