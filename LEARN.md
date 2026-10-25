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
uv run clinicbot search "can my kid get the chickenpox shot" --mode vector
uv run clinicbot search "parking" --mode vector
uv run clinicbot search "Rs. 500" --mode vector
```
Look at the `score` values (cosine similarity; higher means more similar). Notice that "kid" and
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
  compare the `search --mode vector` scores for "dental extraction aftercare".
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

## Phase 2: Better RAG, and measuring it

### The big idea

In Phase 1 you judged quality by reading a few answers. That doesn't scale, and
it can't tell you whether a change helped or hurt. Phase 2 does two things:

1. **Builds a test set** ([data/eval/questions.jsonl](data/eval/questions.jsonl)):
   41 questions, each labelled with the section that *should* be retrieved and a
   fact the answer *must* contain.
2. **Adds three more retrieval strategies** and uses the test set to pick a winner.

```
              ┌── vector search (meaning) ──┐
 question ────┤                             ├── RRF fusion ── top 20 ── cross-encoder ── top 4 ── LLM
              └── BM25 (exact keywords) ────┘                           reranker
                 mode: vector | bm25          mode: hybrid             mode: rerank
```

### New files

| File | Concept |
|------|---------|
| [bm25.py](src/clinicbot/bm25.py) | Keyword search written from scratch: TF, IDF, length normalisation |
| [retrieval.py](src/clinicbot/retrieval.py) | One `retrieve()` for all modes; **Reciprocal Rank Fusion** |
| [rerank.py](src/clinicbot/rerank.py) | **Cross-encoder** reranking, two-stage retrieval |
| [evaluate.py](src/clinicbot/evaluate.py) | **recall@k**, **MRR**, answer checks |

### Results on this project (your numbers may vary slightly)

```
uv run clinicbot eval -k 1                uv run clinicbot eval -k 4
mode     recall@1   MRR                   mode     recall@4   MRR
vector      82%     0.82                  vector     100%     0.90
bm25        62%     0.62                  bm25        82%     0.70
hybrid      74%     0.74                  hybrid      95%     0.83
rerank      87%     0.87   <- winner      rerank      97%     0.92   <- winner
```

Final answers with `rerank` and Qwen3 4B: **39/41 = 95%** (`uv run clinicbot eval --answers`).

### What the numbers teach

**1. Hybrid search is not automatically better.** Blog posts often say "always
use hybrid". Here it *lowered* recall, because most test questions are worded
differently from the docs ("come to my house to take blood" vs. "home
collection"). BM25 is bad at those, and RRF gives its bad rankings the same
weight as the good vector rankings. Hybrid *did* win on the keyword questions
(8/9 at k=1 vs. 7/9). The right choice depends on what your users actually ask,
and that's why you measure.

**2. Reranking fixes the ordering.** Look at MRR: the reranker puts the right
chunk *first* much more often. That matters most with small models and small
context windows, where you can only afford a few chunks.

**3. Read the failures, not just the score.** The two answer failures were:
- *"Can I pay with a cheque?"* → the bot said "No, we **don't** accept cheques",
  which is correct. The test looked for "not", so **the eval had a bug**. Evals
  are code too, and they need debugging. (Fixed: facts can list alternatives
  like `not|don't`.)
- *"Can someone come to my house to take blood?"* → retrieval missed the "Home
  sample collection" chunk. The model then said the lab "requires patients to
  come to the clinic", which **no document says**. That's a *faithfulness*
  failure (hallucination), caused by a *retrieval* failure. Bad retrieval
  doesn't just give "I don't know"; it can produce confident wrong answers.

**4. Don't overfit to your test set.** If you tweak settings until all 41
questions pass, you may just be memorising these 41. Real projects keep a
separate held-out set that they check only occasionally.

### Exercises

**Exercise 6: Watch the strategies disagree.**
```bash
uv run clinicbot search "Rs. 500" --mode vector
uv run clinicbot search "Rs. 500" --mode bm25
uv run clinicbot search "Can someone come to my house to take blood?" --mode bm25
uv run clinicbot search "Can someone come to my house to take blood?" --mode rerank
```

**Exercise 7: List the failures.**
```bash
uv run clinicbot eval -k 1 --misses vector
```
For each miss, decide whether it's a chunking problem, a wording mismatch, or a
bad label in the test set.

**Exercise 8: Weighted fusion.** In `reciprocal_rank_fusion`, give each list a
weight, for example vector 1.0 and BM25 0.5:
`fused[r.id] += weight / (RRF_K + rank)`. Re-run `eval`. Can you find weights
where hybrid beats vector at k=1?

**Exercise 9: Fix the home-collection miss *without* editing the eval.** Ideas:
- **Query rewriting:** ask the LLM to rewrite the question into clinic
  vocabulary before searching ("home visit blood test collection").
- **Better chunk headers:** add keywords or a one-line summary to each chunk at
  ingest time.
- **A stronger embedding model** (look up `bge-small-en-v1.5`).
Check that your fix doesn't break other questions, which the eval will show you.

**Exercise 10: Metadata filtering.** Add a `--doc` option to `search` that only
searches one source file. Chroma supports `where={"source": "05_laboratory.md"}`
in `.query()`. When would a real app use this? (Hint: a patient on the "Dental"
page of the website.)

**Exercise 11: LLM-as-judge.** Checking that "1,500" appears in the answer is
crude. Write a judge prompt that gives the model the question, the retrieved
chunks and the answer, and asks: "Is every claim in the answer supported by the
chunks? yes/no". Run it on the home-collection answer. Does it catch the
hallucination?

### Key terms you should now be able to explain
- **BM25**: term frequency, IDF, length normalisation
- **Hybrid search** and **Reciprocal Rank Fusion**
- **Bi-encoder vs. cross-encoder**, and **two-stage retrieval** (retrieve then rerank)
- **recall@k**, **MRR**, and why retrieval metrics come before answer metrics
- **Faithfulness** vs. relevance, and **LLM-as-judge**
- Eval hygiene: debugging the eval itself, and held-out sets

---

## Coming next
- **Phase 3:** tools (check availability, book an appointment), the agent loop,
  and MCP
- **Phase 4:** deploying and governing the agent on WSO2 Agent Manager
