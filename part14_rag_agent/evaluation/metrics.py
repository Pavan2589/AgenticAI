def evaluate_retrieval(retrieved_chunks, expected_chunk_ids, k, expected_chunk_keys=None):
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved_chunks[:k]
    expected = set(expected_chunk_ids)
    expected_keys = set(expected_chunk_keys or [])
    if not expected and not expected_keys:
        return {"precision_at_k": None, "recall_at_k": None,
                "reciprocal_rank": None, "relevant_chunks": 0,
                "retrieved_chunks": len(top)}
    hits = [chunk.get("id") in expected or chunk.get("chunk_key") in expected_keys for chunk in top]
    rr = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    return {"precision_at_k": sum(hits) / k,
            "recall_at_k": len({(chunk.get("id") if chunk.get("id") in expected else chunk.get("chunk_key")) for chunk, hit in zip(top, hits) if hit}) / len(expected or expected_keys),
            "reciprocal_rank": rr, "relevant_chunks": sum(hits),
            "retrieved_chunks": len(top)}


def aggregate_metrics(results):
    valid = [r for r in results if r["precision_at_k"] is not None]
    if not valid:
        return {"evaluated_queries": 0, "precision_at_k": 0,
                "recall_at_k": 0, "mrr": 0}
    n = len(valid)
    return {"evaluated_queries": n,
            "precision_at_k": round(sum(r["precision_at_k"] for r in valid) / n, 4),
            "recall_at_k": round(sum(r["recall_at_k"] for r in valid) / n, 4),
            "mrr": round(sum(r["reciprocal_rank"] for r in valid) / n, 4)}
