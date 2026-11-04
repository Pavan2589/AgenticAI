
import json
import os

from config import DATASET_PATH, RESULTS_DIR


def evaluate_context_sufficiency(dataset, generation_results):
    """
    Evaluate whether retrieved chunks contain the
    manually verified expected chunks.

    Unanswerable and unlabeled questions are excluded
    from the context sufficiency rate.
    """

    dataset_map = {
        item["id"]: item
        for item in dataset
    }

    evaluation = {}

    for config_id, config_data in generation_results["results"].items():

        query_results = []

        for result in config_data["queries"]:

            query_id = result["id"]
            dataset_item = dataset_map[query_id]

            answerable = dataset_item["answerable"]

            expected_ids = set(
                dataset_item.get("expected_chunk_ids", [])
            )

            retrieved_ids = {
                chunk["id"]
                for chunk in result["retrieved_chunks"]
            }

            # Case 1: Unanswerable questions.
            if not answerable:

                query_results.append({
                    "id": query_id,
                    "status": "unanswerable",
                    "context_sufficient": None,
                    "context_coverage": None,
                    "expected_chunks": [],
                    "retrieved_expected_chunks": [],
                    "missing_chunks": []
                })

                continue

            # Case 2: Answerable questions without verified labels.
            if not expected_ids:

                query_results.append({
                    "id": query_id,
                    "status": "unlabeled",
                    "context_sufficient": None,
                    "context_coverage": None,
                    "expected_chunks": [],
                    "retrieved_expected_chunks": [],
                    "missing_chunks": []
                })

                continue

            # Case 3: Evaluate labeled answerable questions.
            retrieved_expected = expected_ids & retrieved_ids
            missing_chunks = expected_ids - retrieved_ids

            coverage = (
                len(retrieved_expected) / len(expected_ids)
            )

            context_sufficient = len(missing_chunks) == 0

            query_results.append({
                "id": query_id,
                "status": "evaluated",
                "context_sufficient": context_sufficient,
                "context_coverage": round(coverage, 4),
                "expected_chunks": sorted(expected_ids),
                "retrieved_expected_chunks": sorted(
                    retrieved_expected
                ),
                "missing_chunks": sorted(missing_chunks)
            })

        # Separate evaluated answerable queries.
        evaluated = [
            item
            for item in query_results
            if item["status"] == "evaluated"
        ]

        sufficient_count = sum(
            item["context_sufficient"]
            for item in evaluated
        )

        insufficient_count = (
            len(evaluated) - sufficient_count
        )

        # Calculate context sufficiency rate.
        sufficiency_rate = (
            sufficient_count / len(evaluated)
            if evaluated else 0
        )

        # Count unanswerable and unlabeled questions.
        unanswerable_count = sum(
            item["status"] == "unanswerable"
            for item in query_results
        )

        unlabeled_count = sum(
            item["status"] == "unlabeled"
            for item in query_results
        )

        evaluation[config_id] = {
            "top_k": config_data["top_k"],
            "total_queries": len(query_results),
            "evaluated_queries": len(evaluated),
            "sufficient_contexts": sufficient_count,
            "insufficient_contexts": insufficient_count,
            "unanswerable_queries": unanswerable_count,
            "unlabeled_queries": unlabeled_count,
            "context_sufficiency_rate": round(
                sufficiency_rate, 4
            ),
            "queries": query_results
        }

    return evaluation


def main():

    # Load golden dataset.
    with open(DATASET_PATH, "r") as file:
        dataset = json.load(file)

    # Load generation experiment results.
    results_path = os.path.join(
        RESULTS_DIR,
        "generation_evaluation.json"
    )

    with open(results_path, "r") as file:
        generation_results = json.load(file)

    # Run context sufficiency evaluation.
    evaluation = evaluate_context_sufficiency(
        dataset,
        generation_results
    )

    # Prepare output.
    output = {
        "experiment": "context_sufficiency_evaluation",
        "results": evaluation
    }

    # Ensure results directory exists.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(
        RESULTS_DIR,
        "context_sufficiency.json"
    )

    # Save evaluation results.
    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    # Display results.
    print("\n" + "=" * 60)
    print("CONTEXT SUFFICIENCY EVALUATION")
    print("=" * 60)

    for config_id, config_result in evaluation.items():

        print(f"\nConfiguration: {config_id}")
        print(f"Top K: {config_result['top_k']}")

        print(
            f"Context sufficiency rate: "
            f"{config_result['context_sufficiency_rate']:.2%}"
        )

        print(
            f"Sufficient contexts: "
            f"{config_result['sufficient_contexts']}"
        )

        print(
            f"Insufficient contexts: "
            f"{config_result['insufficient_contexts']}"
        )

        print(
            f"Unanswerable queries: "
            f"{config_result['unanswerable_queries']}"
        )

        print(
            f"Unlabeled queries: "
            f"{config_result['unlabeled_queries']}"
        )

        for query in config_result["queries"]:

            print(f"\n{query['id']}: {query['status']}")

            if query["status"] != "evaluated":
                continue

            print(
                "Context sufficient:",
                query["context_sufficient"]
            )

            print(
                "Context coverage:",
                f"{query['context_coverage']:.2%}"
            )

            print(
                "Retrieved expected chunks:",
                query["retrieved_expected_chunks"]
            )

            print(
                "Missing chunks:",
                query["missing_chunks"]
            )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
