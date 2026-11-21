"""Phase 5 — the ReAct agent.

create_react_agent builds a loop: the LLM sees the question + available tools,
decides to call one, sees the result, and decides whether it can answer or needs
another call. Reason + Act, repeated. This is what turns "retrieve once and hope"
into an agent that can chase an answer across multiple files.
"""
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from llm import get_llm
from tools import search_code, list_files, read_file, find_symbol

SYSTEM = """You are an expert code assistant. Answer questions about a codebase.
You have four tools: search_code (semantic search), find_symbol (where Python
classes and functions are defined, with line numbers), list_files (see all files),
and read_file (read a whole file). Use find_symbol for "where is X defined" or
"what classes are there"; semantic search is bad at exact names. You MUST use
search_code or find_symbol before answering, never answer from memory. You may call tools multiple times to gather enough
context. Always cite the files you used by the full path the tools show, with
line numbers when the tools give them, like src/pkg/app.py:120 or
src/pkg/app.py:120-140.
If the answer truly isn't in the code, say so."""

_agent = None

# langgraph's default step limit is 10,007, so nothing stopped the loop except the
# model's 128k context overflowing (issue #15). 25 steps is about 12 tool calls,
# plenty for the 5-10 hop questions, and with read_file capped at 20k chars it
# keeps the worst case near 60k tokens. langgraph's docs say running out of steps
# returns a reply, but 1.2.8 raises GraphRecursionError, so ask_traced catches it.
MAX_STEPS = 25
OUT_OF_STEPS = ("I ran out of steps before finding a complete answer. "
                "Try a narrower question, or name the file or class you mean.")


def get_agent():
    """Built once, reused across questions (and lazily, so import needs no API key)."""
    global _agent
    if _agent is None:
        _agent = create_react_agent(get_llm(), [search_code, find_symbol, list_files, read_file], prompt=SYSTEM)
    return _agent


def ask_traced(question: str, repo: str | None = None) -> tuple[str, list[dict]]:
    """Answer, plus the tool calls the agent made getting there.

    The multi-hop retrieval is the whole point of the agent, so surface it
    instead of throwing it away — the trace is what shows a second search or a
    read_file actually happened rather than one blind retrieval.
    """
    # Stream instead of invoke so the steps taken so far survive hitting the limit.
    state = None
    try:
        for state in get_agent().stream(
            {"messages": [("user", question)]},
            config={"recursion_limit": MAX_STEPS, "configurable": {"repo": repo}},
            stream_mode="values",
        ):
            pass
        answer = state["messages"][-1].content
    except GraphRecursionError:
        answer = OUT_OF_STEPS
    trace = [
        {"tool": call["name"], "input": call["args"]}
        for msg in (state or {}).get("messages", [])
        for call in getattr(msg, "tool_calls", []) or []
    ]
    return answer, trace


def ask(question: str) -> str:
    return ask_traced(question)[0]


if __name__ == "__main__":
    print(ask("How does the retry mechanism work, and which file implements it?"))
