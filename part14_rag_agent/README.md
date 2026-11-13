# Part 14 — Knowledge-Grounded Agent with LangGraph and RAG

This part connects the Part 13 LangGraph tool loop to a scoped internal knowledge base. The `qwen3:4b` model selects tools through constrained JSON and the graph converts those selections into standard LangChain tool calls for `tools_condition` and `ToolNode`; native `bind_tools` remains a fallback for malformed planner output. This change was made after live tests showed native tool calls could be slow, verbose, and unreliable for policy-derived calculations. The agent can choose the calculator, date utility, country lookup, and `search_knowledge_base`, including sequential tool calls. Embeddings use Ollama's `nomic-embed-text` endpoint.

## Recovered work and data safety

The original Part 14 folder already contained the five synthetic policy/product documents, document loader, recursive chunker, embedding client, PostgreSQL adapter, retriever, and ingestion script. The implementation reuses the Parts 9–12 `document_chunks` table and scopes search to the five Part 14 document IDs. It does not create or clear tables and inserts use `ON CONFLICT (chunk_key) DO NOTHING`. Existing research rows are preserved. Run ingestion only after confirming the local database is available; the script verifies required columns before inserting.

Configuration defaults match the existing setup: PostgreSQL at `localhost:5433`, database `agentic_ai`, Ollama at `localhost:11434`, `nomic-embed-text`, and `qwen3:4b`. Environment variables can override these values; see `config.py`. Do not commit credentials or `.env` files.

## Install and run

Use the repository shared virtual environment and add only missing dependencies:

```bash
source ../venv/bin/activate
python -m pip install -r requirements.txt
python ingest.py
python main.py
```

The first run of `main.py` creates this part's independent `checkpoints.sqlite`. Conversation checkpoints are separate from Part 13. Type `clear` for a fresh thread and `exit` to quit.

## Retrieval and policy behavior

Cosine similarity is `1 - (embedding <=> query_embedding)`. `MIN_SIMILARITY` is configurable and currently defaults to `0.67`; live answerable leave questions scored 0.69–0.80, cash-conversion scored about 0.53, and an unrelated Japan query scored about 0.36. Maternity wording can score as high as 0.68 because it shares general leave-policy terms, so a query-specific term-overlap check also rejects results that do not contain the requested subject. These settings are corpus-specific and need more labeled queries and monitoring; they are not universal. The earlier `0.25` value came from a different corpus. Retrieval reports `relevant`, `low_relevance`, `no_results`, or `error` distinctly. A document-ID allowlist prevents unrelated research chunks from becoming policy evidence.

The leave-policy fixtures carry explicit status, effective date, version, and chunker version metadata in `ingest.py`. Where both a current policy and its `_old` sibling are returned, the superseded policy is suppressed. Corrected chunker v2 rows supersede legacy chunks in retrieval without changing or deleting the legacy database rows. This rule complements relevance scores; cosine similarity alone does not establish authority. The final answer step also checks whether evidence explicitly supports the requested subject before answering.

The RAG tool returns compact JSON with evidence and source metadata. The final response's `[S1]` style references are validated against actual `ToolMessage` payloads; displayed filenames, page numbers, chunk IDs, and scores come from that payload. Unknown citation IDs are removed. PDF page numbers are one-based; text and Markdown have no page number.

## Evaluation

`evaluation/metrics.py` implements Precision@K, Recall@K, and reciprocal rank/MRR following Part 12's approach. `evaluation/test_cases.json` separates retrieval, grounding, tool selection, multi-tool use, missing evidence, and infrastructure failure. The leave retrieval case uses a manually verified deterministic chunk key (derived from source, page, index, and content). `evaluate_retrieval.py` reports aggregate metrics only for labeled cases. `run_live_scenarios.py` exercises the seven agent scenarios against Ollama and PostgreSQL and reports observed tools, final answers, and validated citations.

Run isolated tests without either service:

```bash
python -m unittest discover -s tests -v
```

Run the live tests and seven agent scenarios with the local Ollama and PostgreSQL services available:

```bash
RUN_INTEGRATION=1 python -m unittest discover -s tests -v
python evaluation/evaluate_retrieval.py
python evaluation/run_live_scenarios.py
```

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for verified row counts, model availability, retrieval metrics, and scenario results.
