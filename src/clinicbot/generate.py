"""Step 4 of RAG: generate an answer *grounded* in the retrieved chunks.

"Augmented generation" just means: paste the retrieved chunks into the prompt
and tell the LLM to answer from them. The system prompt does the heavy lifting:
  - answer ONLY from the provided context (reduces hallucination),
  - cite sources by number (lets users verify, and lets you debug),
  - say "I don't know" when the context doesn't contain the answer.
"""

import re

from clinicbot.config import CLAUDE_MODEL, LLM_PROVIDER, OLLAMA_MODEL
from clinicbot.store import SearchResult

SYSTEM_PROMPT = """\
You are ClinicBot, the front-desk assistant for Sunrise Family Clinic.

Answer the patient's question using only the numbered context passages provided in \
the user message. Cite the passages you used with their numbers in square brackets, \
for example [2]. If the passages do not contain the answer, say you don't have that \
information and suggest calling the clinic on 011 234 5678; do not guess or use \
outside knowledge about the clinic.

You do not give medical diagnoses. If the question describes an emergency (chest \
pain, difficulty breathing, severe bleeding, stroke signs), tell the patient to call \
1990 or go to the nearest emergency unit before anything else.

Keep answers short and friendly: a few sentences or a short list."""

def format_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {r.source})\n{r.text}" for i, r in enumerate(results, start=1)
    )


def answer(question: str, results: list[SearchResult]) -> str:
    user_message = (
        f"<context>\n{format_context(results)}\n</context>\n\n"
        f"Patient question: {question}"
    )
    if LLM_PROVIDER == "claude":
        return _answer_with_claude(user_message)
    return _answer_with_ollama(user_message)


def _answer_with_ollama(user_message: str) -> str:
    import ollama  # talks to the Ollama server at http://localhost:11434

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            # "/no_think" is Qwen3's in-prompt switch to skip its reasoning step.
            {"role": "user", "content": user_message + "\n\n/no_think"},
        ],
        # Qwen3 can "think" before answering. Off = much faster on a laptop;
        # try True and compare answer quality vs. speed.
        think=False,
        # temperature 0 = pick the most likely token each time, so answers are
        # repeatable and less "creative" (good for factual Q&A).
        options={"temperature": 0},
    )
    # Small models don't always obey the switches, so drop any reasoning that
    # still leaks into the answer (everything up to a closing </think> tag).
    return re.sub(r"^.*?</think>", "", response.message.content, flags=re.S).strip()


def _answer_with_claude(user_message: str) -> str:
    import anthropic  # reads ANTHROPIC_API_KEY from env / .env

    response = anthropic.Anthropic().beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        # Short factual Q&A doesn't need deep reasoning; low effort = faster + cheaper.
        output_config={"effort": "low"},
        # If the model declines a request, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "Sorry, I can't help with that. Please call the clinic on 011 234 5678."
    return "".join(block.text for block in response.content if block.type == "text")
