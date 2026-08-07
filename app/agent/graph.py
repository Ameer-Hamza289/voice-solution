"""LangGraph conversation graph.

Topology:

    [agent] --(tool calls?)--> [tools] --> [agent]
        \\--(no tool calls)--> END

The agent node is Gemini with the dispatcher system prompt and the business
tools bound. A MemorySaver checkpointer keyed by call id gives each phone
call its own persistent multi-turn memory.
"""
from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.prompts import system_prompt
from app.agent.tools import ALL_TOOLS
from app.config import GEMINI_CHAT_MODEL, GEMINI_FALLBACK_MODELS

logger = logging.getLogger(__name__)


class CallState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_graph():
    # Free-tier Gemini quotas are per model. Chain fallbacks so a quota or
    # availability error on one model transparently retries on the next.
    primary = ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL, temperature=0.3
    ).bind_tools(ALL_TOOLS)
    fallbacks = [
        ChatGoogleGenerativeAI(model=name, temperature=0.3).bind_tools(ALL_TOOLS)
        for name in GEMINI_FALLBACK_MODELS
    ]
    llm_with_tools = primary.with_fallbacks(fallbacks) if fallbacks else primary

    def agent_node(state: CallState) -> dict:
        messages = [SystemMessage(content=system_prompt()), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route(state: CallState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(CallState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


async def respond(call_id: str, user_text: str) -> str:
    """Run one conversational turn for the given call and return the reply."""
    graph = get_graph()
    config = {"configurable": {"thread_id": call_id}, "recursion_limit": 12}
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_text)]}, config
        )
        reply = result["messages"][-1].content
        if isinstance(reply, list):  # Gemini can return multi-part content
            reply = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in reply
            )
        return reply.strip() or "Sorry, could you say that again?"
    except Exception:
        logger.exception("Agent turn failed")
        return (
            "I'm sorry, I'm having a little trouble on my end. "
            "Could you say that one more time?"
        )
