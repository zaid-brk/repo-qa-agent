"""Phase 5 — the ReAct agent.

create_react_agent builds a loop: the LLM sees the question + available tools,
decides to call one, sees the result, and decides whether it can answer or needs
another call. Reason + Act, repeated. This is what turns "retrieve once and hope"
into an agent that can chase an answer across multiple files.
"""
from langgraph.prebuilt import create_react_agent

from llm import get_llm
from tools import search_code, list_files, read_file

SYSTEM = """You are an expert code assistant. Answer questions about a codebase.
You have three tools: search_code (semantic search), list_files (see all files),
and read_file (read a whole file). You MUST use search_code before answering —
never answer from memory. You may call tools multiple times to gather enough
context. Always cite the files you used.
If the answer truly isn't in the code, say so."""

_agent = None


def get_agent():
    """Built once, reused across questions (and lazily, so import needs no API key)."""
    global _agent
    if _agent is None:
        _agent = create_react_agent(get_llm(), [search_code, list_files, read_file], prompt=SYSTEM)
    return _agent


def ask_traced(question: str) -> tuple[str, list[dict]]:
    """Answer, plus the tool calls the agent made getting there.

    The multi-hop retrieval is the whole point of the agent, so surface it
    instead of throwing it away — the trace is what shows a second search or a
    read_file actually happened rather than one blind retrieval.
    """
    result = get_agent().invoke({"messages": [("user", question)]})
    trace = [
        {"tool": call["name"], "input": call["args"]}
        for msg in result["messages"]
        for call in getattr(msg, "tool_calls", []) or []
    ]
    return result["messages"][-1].content, trace


def ask(question: str) -> str:
    return ask_traced(question)[0]


if __name__ == "__main__":
    print(ask("How does the retry mechanism work, and which file implements it?"))
