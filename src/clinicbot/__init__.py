"""ClinicBot command line.

    uv run clinicbot ingest                 # chunk + embed + store the docs
    uv run clinicbot chunks                 # print the chunks (inspect your chunking)
    uv run clinicbot search "query"         # retrieval only, no LLM (free!)
    uv run clinicbot search "q" --mode bm25 # try another retrieval strategy
    uv run clinicbot ask "question"         # full RAG: retrieve + generate
    uv run clinicbot chat                   # interactive loop
    uv run clinicbot eval                   # score retrieval for every mode
    uv run clinicbot eval --answers         # score final answers (runs the LLM)
    uv run clinicbot agent                  # Phase 3: agent that can book/cancel
    uv run clinicbot reset-db               # recreate the mock booking database
"""

import argparse
import sys

from clinicbot import store
from clinicbot.chunking import chunk_directory
from clinicbot.config import DOCS_DIR, RETRIEVAL_MODE, TOP_K
from clinicbot.retrieval import MODES, retrieve


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
    for i, r in enumerate(retrieve(args.query, args.k, args.mode), start=1):
        print(f"[{i}] score={r.score:.3f}  {r.source}  ::  {r.section}")
        print("    " + r.text.replace("\n", "\n    ")[:400])
        print()


def ask(question: str, args) -> None:
    from clinicbot.generate import answer  # imported lazily: only needed for ask/chat

    results = retrieve(question, args.k, args.mode)
    print(answer(question, results))
    if args.sources:
        print("\nSources:")
        for i, r in enumerate(results, start=1):
            print(f"  [{i}] {r.source} :: {r.section} (score {r.score:.3f})")


def cmd_ask(args) -> None:
    ask(args.question, args)


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
            ask(question, args)


def cmd_agent(args) -> None:
    from clinicbot.agent import Agent

    def approve(name: str, arguments: dict) -> bool:
        if args.yes:
            return True
        print(f"\n  [approval needed] {name} {arguments}")
        return input("  Allow? [y/N] ").strip().lower() in {"y", "yes"}

    trace = None if args.quiet else (lambda line: print(f"  [tool] {line}"))
    agent = Agent(approve=approve, trace=trace)

    # Each item is one user turn; pass several to script a conversation.
    if args.message:
        for text in args.message:
            print(f"\nyou> {text}")
            print(f"bot> {agent.send(text)}")
        return

    print("ClinicBot agent ready. Try: 'Book me with the dentist on Friday' (or 'quit').")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            print(f"bot> {agent.send(text)}")


def cmd_reset_db(_args) -> None:
    from clinicbot import clinic_db

    clinic_db.reset()
    print(f"Recreated {clinic_db.DB_PATH} with fresh sample bookings.")


def cmd_eval(args) -> None:
    from clinicbot import evaluate

    if args.agent:
        evaluate.eval_agent()
    elif args.answers:
        evaluate.eval_answers(args.k, args.mode)
    else:
        modes = [args.only] if args.only else list(MODES)
        evaluate.eval_retrieval(args.k, modes, args.misses)


def main() -> None:
    parser = argparse.ArgumentParser(prog="clinicbot")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    sub.add_parser("chunks").set_defaults(func=cmd_chunks)

    def add_retrieval_args(p):
        p.add_argument("-k", type=int, default=TOP_K)
        p.add_argument("--mode", choices=MODES, default=RETRIEVAL_MODE)

    p = sub.add_parser("search")
    p.add_argument("query")
    add_retrieval_args(p)
    p.set_defaults(func=cmd_search)

    for name, func in (("ask", cmd_ask), ("chat", cmd_chat)):
        p = sub.add_parser(name)
        if name == "ask":
            p.add_argument("question")
        add_retrieval_args(p)
        p.add_argument("--sources", action="store_true", help="show retrieved chunks")
        p.set_defaults(func=func)

    p = sub.add_parser("agent")
    p.add_argument("-m", "--message", action="append", help="send a message (repeatable)")
    p.add_argument("--yes", action="store_true", help="auto-approve bookings/cancellations")
    p.add_argument("--quiet", action="store_true", help="hide tool call trace")
    p.set_defaults(func=cmd_agent)

    sub.add_parser("reset-db").set_defaults(func=cmd_reset_db)

    p = sub.add_parser("eval")
    add_retrieval_args(p)
    p.add_argument("--answers", action="store_true", help="grade final answers (slow)")
    p.add_argument("--agent", action="store_true", help="grade agent tool use (slow)")
    p.add_argument("--only", choices=MODES, help="evaluate a single mode")
    p.add_argument("--misses", choices=MODES, help="list failed questions for a mode")
    p.set_defaults(func=cmd_eval)

    # Windows consoles default to cp1252, which can't print emoji LLMs like to use.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parser.parse_args()
    args.func(args)
