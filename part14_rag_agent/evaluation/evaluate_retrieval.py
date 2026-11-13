import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import evaluate_retrieval, aggregate_metrics
from retriever import retrieve_chunks, STATUS_RELEVANT
from config import TOP_K


def main():
    cases = json.loads((Path(__file__).parent / "test_cases.json").read_text())
    results = []
    for case in cases:
        expected_keys = case.get("expected_chunk_keys", [])
        if case["category"] != "retrieval" or not (case["expected_chunk_ids"] or expected_keys):
            continue
        retrieval = retrieve_chunks(case["question"], top_k=TOP_K)
        chunks = retrieval.chunks if retrieval.status == STATUS_RELEVANT else []
        results.append(evaluate_retrieval(chunks, case["expected_chunk_ids"], TOP_K, expected_keys))
    print(json.dumps(aggregate_metrics(results), indent=2))
    if not results:
        print("No retrieval cases have manually labeled expected_chunk_ids yet.")


if __name__ == "__main__":
    main()
