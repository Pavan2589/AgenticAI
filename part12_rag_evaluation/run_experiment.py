
import json
import os
from datetime import datetime, timezone

from config import (
    CONFIGURATIONS,
    DATASET_PATH,
    RESULTS_DIR
)

from retriever import retrieve_chunks

from retrieval_evaluation import (
    evaluate_retrieval,
    aggregate_metrics
)


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def run_configuration(config_name, config, dataset):

    top_k = config["top_k"]

    print(f"\nRunning configuration: {config_name}")
    print(f"Top K: {top_k}")

    query_results = []

    for item in dataset:

        question_id = item["id"]
        question = item["question"]

        print(f"\nEvaluating {question_id}: {question}")

        chunks = retrieve_chunks(
            query=question,
            top_k=top_k
        )

        metrics = evaluate_retrieval(
            retrieved_chunks=chunks,
            expected_chunk_ids=item["expected_chunk_ids"],
            k=top_k
        )
        query_result = {
            "id": question_id,
            "question": question,
            "expected_chunk_ids": item["expected_chunk_ids"],
            "retrieved_chunks": [
                {
                    "id": chunk["id"],
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "similarity": chunk["similarity"],
                    "content": chunk["content"]
                }
                for chunk in chunks
            ],
            "metrics": metrics
        }

        query_results.append(query_result)

        print(
            f"Precision@{top_k}:",
            metrics["precision_at_k"]
        )

        print(
            f"Recall@{top_k}:",
            metrics["recall_at_k"]
        )

        print(
            "Reciprocal Rank:",
            metrics["reciprocal_rank"]
        )

    summary = aggregate_metrics([
        result["metrics"]
        for result in query_results
    ])

    return {
        "configuration": config_name,
        "settings": config,
        "summary": summary,
        "queries": query_results
    }


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_dataset()

    experiment_results = {
        "experiment": "Day 12 RAG Retrieval Evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": {}
    }

    for config_name, config in CONFIGURATIONS.items():

        result = run_configuration(
            config_name,
            config,
            dataset
        )

        experiment_results["results"][config_name] = result

    output_path = os.path.join(
        RESULTS_DIR,
        "retrieval_evaluation.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            experiment_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)

    for name, result in experiment_results["results"].items():
        print(f"\nConfiguration {name}")
        print(json.dumps(result["summary"], indent=4))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
