# ClinicBot: learning notes

## Phase 1: Basic RAG

### The big idea

An LLM only knows what was in its training data. It has never seen *your*
clinic's cancellation policy. You could paste every document into each prompt,
but that gets slow and expensive as the documents grow, and the model gets
distracted by irrelevant text.

**RAG (Retrieval-Augmented Generation)** instead does:

```
          OFFLINE (once, when docs change)                ONLINE (every question)
 ┌────────┐   ┌────────┐   ┌──────────┐   ┌─────────┐      question
 │  docs  │──▶│ chunk  │──▶│  embed   │──▶│ vector  │◀──── embed ──── "Can I cancel?"
 └────────┘   └────────┘   └──────────┘   │   DB    │         │
                                          └─────────┘    top-k chunks
                                                              │
                                                  ┌───────────▼───────────┐
                                                  │ LLM: answer using ONLY │
                                                  │ these chunks + cite    │
                                                  └───────────────────────┘
```

### Where each step lives in the code

| Step | File | What to look at |
|------|------|-----------------|
| Settings | [config.py](src/clinicbot/config.py) | chunk size, overlap, TOP_K, model |
| 1. Chunk | [chunking.py](src/clinicbot/chunking.py) | heading-based splitting, contextual headers |
| 2. Embed | [embeddings.py](src/clinicbot/embeddings.py) | text → 384 numbers |
| 3. Store/search | [store.py](src/clinicbot/store.py) | Chroma, cosine distance |
| 4. Generate | [generate.py](src/clinicbot/generate.py) | grounding system prompt, citations |
| CLI | [\_\_init\_\_.py](src/clinicbot/__init__.py) | `ingest`, `chunks`, `search`, `ask`, `chat` |

### Setup

Answers are written by **Qwen3 4B**, running locally through
[Ollama](https://ollama.com). It's free, private, and works offline. It uses
about 3.2 GB of your RTX 3050's 6 GB of VRAM.

1. Make sure Ollama is running. It starts with Windows; look for the llama icon
   in the system tray.
2. Build the index, which runs steps 1–3:
   ```bash
   uv run clinicbot ingest
   ```
3. Optional: to compare against Claude, copy `.env.example` to `.env`, then set
   `CLINICBOT_PROVIDER=claude` and `ANTHROPIC_API_KEY`.

#### Local LLM lessons
- **Model size vs. hardware.** "4B" means 4 billion parameters. Ollama stores
  them *quantized* to about 4 bits each (Q4), so the model takes ~2.5 GB instead
  of ~8 GB. The model has to fit in VRAM to run on the GPU. Check with
  `ollama ps`: you want `100% GPU`. A 7–8B model would barely fit your 6 GB, and
  part of it would spill to the CPU, which is much slower.
- **Cold start.** The first question after a while takes much longer (~50 s)
  because Ollama loads the model into VRAM. It unloads the model after 5 idle
  minutes.
- **Context window.** `ollama ps` shows `CONTEXT 4096`. That's the most tokens the
  model can read at once: system prompt + chunks + question + answer. A larger
  TOP_K or bigger chunks can overflow it, and then the start of the prompt is
  silently cut off. Small context is one more reason retrieval has to be precise.
- **Small models obey less reliably.** Qwen3 sometimes "thinks out loud" even
  when told not to, so `generate.py` strips that text out. Expect to spend more
  effort on prompts and guard code with small models than with Claude.

### Exercises (do them in order)

**Exercise 1: Look at your chunks.**
```bash
uv run clinicbot chunks
```
Every chunk starts with a breadcrumb like `Fees and Payments > Insurance`. Why
does that help? Try to guess which chunks will be hard to retrieve.

**Exercise 2: Retrieval alone (free, no LLM).**
```bash
uv run clinicbot search "can my kid get the chickenpox shot"
uv run clinicbot search "parking"
uv run clinicbot search "Rs. 500"
```
Look at the `distance` values. Lower means more similar. Notice that "kid" and
"shot" still find the vaccination chunk even though those words never appear in
it. That's semantic search. Then notice that `"Rs. 500"` probably does **badly**.
Embeddings are weak at exact tokens like numbers, codes and names. That's the
motivation for *hybrid search* in Phase 2.

**Exercise 3: Full RAG.**
```bash
uv run clinicbot ask "What's the fee if I cancel 5 hours before?" --sources
uv run clinicbot ask "Which doctor speaks Tamil and sees kids?" --sources
uv run clinicbot ask "Do you do MRI scans?" --sources
```
The last question isn't covered by any document. Check that the bot says so
instead of making something up. That behaviour comes from the system prompt in
`generate.py`.

**Exercise 4: Break it on purpose (this is where the learning is).**
- In `config.py`, set `TOP_K = 1` and re-ask "Which doctor speaks Tamil and sees
  kids?". The answer needs facts from more than one chunk. What happens?
- Set `CHUNK_MAX_CHARS = 150`, re-run `ingest`, then run `chunks`. Look at how
  tables and lists get cut apart. Ask about lab prices.
- In `chunking.py`, remove the breadcrumb (use `text=piece`), re-ingest, and
  compare the `search` distances for "dental extraction aftercare".
- In `generate.py`, delete the "only use the context" sentence from the system
  prompt and ask about MRI scans again.

Put each setting back after you try it.

**Exercise 4b: Local vs. hosted, and thinking vs. not.**
- In `generate.py`, set `think=True` and remove `/no_think`. Is it slower? Are the
  answers better on the tricky question ("Tamil + kids")?
- Try another small model, `ollama pull llama3.2:3b`, by setting
  `CLINICBOT_OLLAMA_MODEL=llama3.2:3b` in `.env`. Compare the answers.
  (Free disk space first: models are 2–3 GB each.)

**Exercise 5: Add your own document.**
Write `data/docs/09_physiotherapy.md` (new service, hours, prices), run
`ingest`, and ask about it. The model learns new facts without any retraining.

### Key terms you should now be able to explain
- **Chunking**, chunk size and overlap, and contextual chunk headers
- **Embedding** and why the query and the docs must use the same model
- **Vector database**, nearest-neighbour search, HNSW, cosine distance
- **Top-k** retrieval and the precision/recall trade-off
- **Grounding**: answering only from context, with citations and a way to refuse
- **Ingestion** (offline) vs. **query time** (online)

---

## Coming next
- **Phase 2:** hybrid search (BM25 + vectors), reranking, and an eval set that
  scores retrieval with recall@k
- **Phase 3:** tools (check availability, book an appointment), the agent loop,
  and MCP
- **Phase 4:** deploying and governing the agent on WSO2 Agent Manager
