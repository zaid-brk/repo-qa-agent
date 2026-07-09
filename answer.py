"""Phase 4 — naive RAG: retrieve once, stuff chunks into the prompt, answer.

Kept around as the baseline the agent (Phase 5) is compared against in the eval.
The two anti-hallucination moves in the prompt matter most: "use ONLY the code
below" grounds the model in retrieved context, and "say you don't know" gives it
permission to admit gaps instead of inventing plausible-sounding answers.
"""
from llm import get_llm
from store import search

PROMPT = """You are a code assistant answering questions about a codebase.
Use ONLY the code snippets below. If the answer isn't in them, say you don't know.
Cite the file name(s) you used.

Question: {question}

Relevant code:
{context}

Answer:"""


def answer_question(question: str) -> str:
    hits = search(question, top_k=5)
    context = "\n\n".join(f"--- {h['file']} ---\n{h['text']}" for h in hits)
    prompt = PROMPT.format(question=question, context=context)
    return get_llm().invoke(prompt).content


if __name__ == "__main__":
    print(answer_question("How does this library handle HTTP retries?"))
