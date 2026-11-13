# Part 14 live validation report

Validation date: 2026-11-13 (Asia/Kolkata)

## Services and configuration

- Ollama `/api/tags` responded successfully. Available models included `qwen3:4b`, `nomic-embed-text:latest`, and `qwen2.5-coder:7b`; no model was downloaded.
- A live `nomic-embed-text` request returned 768 dimensions (norm 1.0).
- A live `qwen3:4b` generation request returned successfully. Native tool calling was slow and unreliable for compound questions; the agent now uses a JSON-constrained LLM tool selector, emits ordinary LangChain tool-call messages, and executes them through the LangGraph `ToolNode` loop. `bind_tools` remains the malformed-plan fallback.
- PostgreSQL container `day9_pgvector` was running. `agentic_ai` connected from both `psql` and the configured Python client using role `pavan` and the existing `pavan` / `day9password` defaults. The suggested `postgres` role does not exist in this container.

## Database compatibility and preservation

- Before this validation's corrected-chunk ingestion, the table contained **1,817 rows**, not 1,807. Ten Part 14 legacy rows (two per document) were already present alongside the earlier corpus.
- `document_chunks` has the expected columns: `id BIGSERIAL`, unique non-null `chunk_key`, document/source/page/index fields, `content`, `embedding vector(768)`, `metadata jsonb`, and `created_at`.
- The `vector` extension is installed. The primary key and unique `chunk_key` B-tree indexes exist. There is no vector similarity index, so searches use a sequential scan; this is acceptable at the current table size but may need attention as the corpus grows.
- The 1,817 existing rows were preserved. Ten corrected chunker v2 rows were added, giving a final count of **1,827**. All 1,827 keys are distinct. Each of the five Part 14 document IDs now has four rows: two untouched legacy chunks and two corrected chunks.
- All ten new rows matched the source text and corrected chunker output exactly. A duplicate-key probe returned `False` for insertion, left the stored content and metadata unchanged, and kept the row count at 1,827.
- No table/schema creation, update, deletion, truncation, or destructive operation was performed. No packages were installed.

## Retrieval and evaluation

- Live embedding and database retrieval worked. Similarity is calculated as `1 - cosine_distance` from pgvector `<=>` results.
- Current policy retrieval selected `employee_leave_policy` metadata (`policy_status=current`, version `2.0`, effective date `2026-01-01`) and suppressed the superseded document when both were candidates.
- `MIN_SIMILARITY=0.67` is configurable. A query-specific term-overlap check catches high-scoring but unsupported subjects such as maternity leave, whose wording can still score around 0.68 because of general leave-policy similarity. Cash-conversion scored around 0.53; Japan around 0.36. Thresholds remain specific to this small corpus.
- The labeled retrieval evaluation has one manually curated chunk key: Precision@5 **0.20**, Recall@5 **1.00**, MRR **1.00**. This is a smoke evaluation, not a broad quality estimate.

## Agent scenarios

All seven scenarios selected the expected tool set across the live run; the maternity scenario was rerun after fixing its unsupported-evidence response.

| Scenario | Observed behavior |
|---|---|
| Company leave policy | Search; answered with current policy and validated `[S1]`, `[S2]` citations. |
| `125 * 48` | Calculator only; answered `6000`, with no fabricated source citation. |
| Current allowance minus 6 days | Search then calculator; answered `14` and cited current policy source `[S1]`. |
| Maternity leave | Search; after the topic-overlap fix, answered that the documents do not provide the information, with no citation. |
| Leave-to-cash conversion | Search; answered that the documents do not provide the information, with no citation. |
| Capital of Japan | No tool; answered Tokyo. |
| Current annual leave allowance | Search; answered 20 paid leave days and cited the current policy. |

Citation validation reads source IDs, chunk IDs, filenames, page numbers, version, and scores from actual `ToolMessage` JSON. Unknown source IDs and non-source values are discarded. Unit tests explicitly reject invented citations. A live graph turn was also recovered from `checkpoints.sqlite`, confirming checkpoint persistence.

## Tests and remaining limits

- Final suite: **20 tests passed** with `RUN_INTEGRATION=1`; no tests were skipped in that run. This included live Ollama embedding and PostgreSQL schema/retrieval checks.
- Offline suite: 15 tests passed and five service integration tests were skipped as designed.
- Retrieval evaluation completed with the metrics above. Python compilation passed.
- The query-specific lexical check is deliberately conservative and can reduce recall for paraphrases that share no distinctive terms with the source. Add curated queries and recalibrate as documents change.
- The table has no vector index. Agent answer quality depends on the local 4B model; the JSON-constrained planner and final-answer generator reduce, but cannot eliminate, model variability.
- Part 13 files/database were not modified. Git remains uncommitted; the only untracked project folder is `part14_rag_agent/`.
