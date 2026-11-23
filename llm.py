"""One place for every model choice; other files import these getters.

The getters are lazy (model built on first use, not at import) so every module
imports fine without an API key — handy for selftest.py and for reading the code.
The judge (Phase 6) is deliberately a STRONGER model than the agent: a weaker
judge grading a stronger agent is unreliable.
"""
from contextvars import ContextVar

from dotenv import load_dotenv

load_dotenv()

AGENT_MODEL = "gpt-4o-mini"             # fast + cheap; answers the questions
JUDGE_MODEL = "gpt-4o"                  # stronger; grades the agent's answers
EMBED_MODEL = "text-embedding-3-small"  # ~$0.02/M tokens, 1536 dimensions
EMBED_DIM = 1536

# Bring your own key: the API sets this to the caller's OpenAI key for one
# request. Scripts and the eval leave it unset and use OPENAI_API_KEY from .env.
user_key: ContextVar[str | None] = ContextVar("user_key", default=None)


def _key() -> dict:
    k = user_key.get()
    return {"api_key": k} if k else {}


def get_llm():
    """The Q&A model. temperature=0 = deterministic — right for factual code Q&A."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=AGENT_MODEL, temperature=0, **_key())


def get_judge():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBED_MODEL, **_key())
