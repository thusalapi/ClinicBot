"""ClinicBot command line.

    uv run clinicbot ingest          # chunk + embed + store the docs
    uv run clinicbot chunks          # print the chunks (inspect your chunking)
    uv run clinicbot search "query"  # retrieval only, no LLM (free!)
    uv run clinicbot ask "question"  # full RAG: retrieve + generate
    uv run clinicbot chat            # interactive loop
"""

import argparse

from clinicbot import store
from clinicbot.chunking import chunk_directory
from clinicbot.config import DOCS_DIR, TOP_K


def cmd_ingest(_args) -> None:
    chunks = chunk_directory(DOCS_DIR)
    print(f"Chunked {len(list(DOCS_DIR.glob('*.md')))} files into {len(chunks)} chunks.")
    store.reset_collection()  # start fresh so deleted/renamed chunks don't linger
    store.add_chunks(chunks)
    print(f"Stored {store.count()} vectors in the vector DB.")


def cmd_chunks(_args) -> None:
    for chunk in chunk_directory(DOCS_DIR):
        print(f"--- {chunk.id}  ({len(chunk.text)} chars)")
        print(chunk.text)
        print()


def cmd_search(args) -> None:
    for i, r in enumerate(store.search(args.query, args.k), start=1):
        print(f"[{i}] distance={r.distance:.3f}  {r.source}  ::  {r.section}")
        print("    " + r.text.replace("\n", "\n    ")[:400])
        print()


def ask(question: str, k: int, show_sources: bool) -> None:
    from clinicbot.generate import answer  # imported lazily: only needed for ask/chat

    results = store.search(question, k)
    print(answer(question, results))
    if show_sources:
        print("\nSources:")
        for i, r in enumerate(results, start=1):
            print(f"  [{i}] {r.source} :: {r.section} (distance {r.distance:.3f})")


def cmd_ask(args) -> None:
    ask(args.question, args.k, args.sources)


def cmd_chat(args) -> None:
    print("ClinicBot ready. Ask a question (or 'quit').")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print("bot> ", end="")
            ask(question, args.k, args.sources)


def main() -> None:
    parser = argparse.ArgumentParser(prog="clinicbot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    sub.add_parser("chunks").set_defaults(func=cmd_chunks)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=TOP_K)
    p.set_defaults(func=cmd_search)

    for name, func in (("ask", cmd_ask), ("chat", cmd_chat)):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("-k", type=int, default=TOP_K)
        p.add_argument("--sources", action="store_true", help="show retrieved chunks")
        p.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)
