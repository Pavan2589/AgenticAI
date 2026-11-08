
def evaluate_retrieval(retrieved_chunks, expected_chunk_ids, k):

    top_chunks = retrieved_chunks[:k]

    expected = set(expected_chunk_ids)

    if not expected:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
            "relevant_chunks": 0,
            "retrieved_chunks": len(top_chunks)
        }

    retrieved_ids = [
        chunk["id"] for chunk in top_chunks
    ]

    relevant_flags = [
        chunk_id in expected
        for chunk_id in retrieved_ids
    ]

    relevant_count = sum(relevant_flags)

    # Precision@K
    precision = relevant_count / k

    # Recall@K
    recall = len(
        expected.intersection(set(retrieved_ids))
    ) / len(expected)

    # Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(
        relevant_flags, start=1
    ):
        if is_relevant:
            reciprocal_rank = 1 / rank
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
        "relevant_chunks": relevant_count,
        "retrieved_chunks": len(top_chunks)
    }


def aggregate_metrics(results):

    valid_results = [
        result for result in results
        if result["precision_at_k"] is not None
    ]

    if not valid_results:
        return {
            "evaluated_queries": 0,
            "precision_at_k": 0,
            "recall_at_k": 0,
            "mrr": 0
        }

    count = len(valid_results)

    return {
        "evaluated_queries": count,

        "precision_at_k": round(
            sum(r["precision_at_k"] for r in valid_results)
            / count, 4
        ),

        "recall_at_k": round(
            sum(r["recall_at_k"] for r in valid_results)
            / count, 4
        ),

        "mrr": round(
            sum(r["reciprocal_rank"] for r in valid_results)
            / count, 4
        )
    }
