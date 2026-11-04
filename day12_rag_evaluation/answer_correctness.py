
import json
import requests
from pathlib import Path

from config import (
    OLLAMA_URL,
    LLM_MODEL,
    DATASET_PATH,
    RESULTS_DIR
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GENERATION_RESULTS_PATH = Path(RESULTS_DIR) / "generation_evaluation.json"
OUTPUT_PATH = Path(RESULTS_DIR) / "answer_correctness.json"

MAX_RETRIES = 2
TIMEOUT = 300


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# ABSTENTION DETECTION
# --------------------------------------------------

def is_abstained(result):

    if result.get("insufficient_evidence") is True:
        return True

    answer = result.get("answer", "")

    if not isinstance(answer, str) or not answer.strip():
        return True

    abstention_phrases = [
        "insufficient information",
        "insufficient evidence",
        "cannot answer",
        "cannot determine",
        "not enough information",
        "not provided in the context",
        "not available in the context",
        "unable to answer"
    ]

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in abstention_phrases
    )


# --------------------------------------------------
# JUDGE JSON SCHEMA
# --------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key_point": {
                        "type": "string"
                    },
                    "covered": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "key_point",
                    "covered"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "evaluations"
    ],
    "additionalProperties": False
}


# --------------------------------------------------
# LLM-AS-A-JUDGE
# --------------------------------------------------

def call_llm_judge(
    question,
    ground_truth,
    answer,
    expected_key_points
):

    key_points_text = "\n".join(
        f"- {point}"
        for point in expected_key_points
    )

    prompt = f"""
Evaluate the generated answer against each expected key point.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{answer}

EXPECTED KEY POINTS:
{key_points_text}

Rules:
1. Evaluate every key point independently.
2. covered=true only when the answer communicates the correct information.
3. Accept equivalent wording and mathematically equivalent expressions.
4. Do not infer missing information.
5. For mathematical complexity, check that the relevant terms and variables are correct.
6. Copy every key point exactly.
7. Return exactly one evaluation per key point.

Return only the required JSON.
"""

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "format": JUDGE_SCHEMA,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 150
                    }
                },
                timeout=TIMEOUT
            )

            response.raise_for_status()

            response_data = response.json()

            raw_content = (
                response_data
                .get("message", {})
                .get("content", "")
            )

            parsed = json.loads(raw_content)

            evaluations = parsed.get("evaluations")

            if not isinstance(evaluations, list):
                raise ValueError(
                    "Invalid evaluations structure."
                )

            evaluation_map = {}

            for item in evaluations:

                if not isinstance(item, dict):
                    raise ValueError(
                        "Invalid evaluation item."
                    )

                key_point = item.get("key_point")
                covered = item.get("covered")

                if not isinstance(key_point, str):
                    raise ValueError(
                        "key_point must be a string."
                    )

                if not isinstance(covered, bool):
                    raise ValueError(
                        "covered must be boolean."
                    )

                if key_point not in expected_key_points:
                    raise ValueError(
                        f"Unexpected key point: {key_point}"
                    )

                evaluation_map[key_point] = {
                    "key_point": key_point,
                    "covered": covered
                }

            # Ensure every expected key point was evaluated.
            if set(evaluation_map.keys()) != set(expected_key_points):
                raise ValueError(
                    "Judge did not evaluate every expected key point."
                )

            return {
                "success": True,
                "evaluations": [
                    evaluation_map[point]
                    for point in expected_key_points
                ],
                "error": None
            }

        except Exception as error:

            print(
                f"    Judge attempt {attempt + 1} failed: {error}"
            )

    return {
        "success": False,
        "evaluations": [],
        "error": "judge_failed_after_retries"
    }


# --------------------------------------------------
# EVALUATE ONE GENERATED ANSWER
# --------------------------------------------------

def evaluate_answer(result, dataset_item):

    answerable = dataset_item.get("answerable", False)

    answer = result.get("answer", "")

    question = dataset_item.get(
        "question",
        result.get("question", "")
    )

    ground_truth = dataset_item.get(
        "ground_truth",
        ""
    )

    key_points = dataset_item.get(
        "expected_key_points",
        dataset_item.get("key_points", [])
    )

    # ----------------------------------------------
    # Unanswerable questions
    # ----------------------------------------------

    if not answerable:

        abstained = is_abstained(result)

        return {
            "id": dataset_item["id"],
            "question": question,
            "status": "unanswerable",
            "abstained": abstained,
            "correct_abstention": abstained,
            "semantic_coverage": None,
            "evaluations": [],
            "evaluation_error": None
        }

    # ----------------------------------------------
    # Answerable questions
    # ----------------------------------------------

    if is_abstained(result):

        return {
            "id": dataset_item["id"],
            "question": question,
            "status": "abstained_on_answerable_question",
            "abstained": True,
            "correct_abstention": None,
            "semantic_coverage": None,
            "evaluations": [],
            "evaluation_error": None
        }

    # ----------------------------------------------
    # Missing key points
    # ----------------------------------------------

    if not key_points:

        return {
            "id": dataset_item["id"],
            "question": question,
            "status": "missing_expected_key_points",
            "abstained": False,
            "correct_abstention": None,
            "semantic_coverage": None,
            "evaluations": [],
            "evaluation_error": "No expected key points found."
        }

    # ----------------------------------------------
    # Semantic evaluation
    # ----------------------------------------------

    judge_result = call_llm_judge(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        expected_key_points=key_points
    )

    if not judge_result["success"]:

        return {
            "id": dataset_item["id"],
            "question": question,
            "status": "evaluation_error",
            "abstained": False,
            "correct_abstention": None,
            "semantic_coverage": None,
            "evaluations": [],
            "evaluation_error": judge_result["error"]
        }

    evaluations = judge_result["evaluations"]

    covered_count = sum(
        1 for item in evaluations
        if item["covered"]
    )

    total_count = len(key_points)

    semantic_coverage = (
        covered_count / total_count
    ) * 100

    return {
        "id": dataset_item["id"],
        "question": question,
        "status": "evaluated",
        "abstained": False,
        "correct_abstention": None,
        "semantic_coverage": round(
            semantic_coverage, 2
        ),
        "covered_key_points": covered_count,
        "total_key_points": total_count,
        "evaluations": evaluations,
        "evaluation_error": None
    }


# --------------------------------------------------
# AGGREGATE METRICS
# --------------------------------------------------

def calculate_metrics(results):

    evaluated = [
        item for item in results
        if item["status"] == "evaluated"
    ]

    evaluation_errors = [
        item for item in results
        if item["status"] == "evaluation_error"
    ]

    abstention_results = [
        item for item in results
        if item["status"] == "unanswerable"
    ]

    correctly_abstained = sum(
        1 for item in abstention_results
        if item["correct_abstention"]
    )

    semantic_coverages = [
        item["semantic_coverage"]
        for item in evaluated
    ]

    average_coverage = (
        sum(semantic_coverages) / len(semantic_coverages)
        if semantic_coverages
        else 0
    )

    abstention_accuracy = (
        correctly_abstained / len(abstention_results) * 100
        if abstention_results
        else 0
    )

    return {
        "total_questions": len(results),

        "evaluated_answers": len(evaluated),

        "evaluation_errors": len(evaluation_errors),

        "abstained_on_answerable": sum(
            1 for item in results
            if item["status"] == "abstained_on_answerable_question"
        ),

        "unanswerable_questions": len(abstention_results),

        "correctly_abstained": correctly_abstained,

        "average_semantic_coverage": round(
            average_coverage, 2
        ),

        "abstention_accuracy": round(
            abstention_accuracy, 2
        )
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("ANSWER CORRECTNESS EVALUATION")
    print("=" * 60)

    dataset = load_json(DATASET_PATH)

    generation_data = load_json(
        GENERATION_RESULTS_PATH
    )

    dataset_map = {
        item["id"]: item
        for item in dataset
    }

    all_results = {}

    for config_name, config_data in generation_data["results"].items():

        print(f"\nConfiguration: {config_name}")
        print("-" * 60)

        config_results = []

        queries = config_data.get("queries", [])

        for result in queries:

            query_id = result.get("id")

            dataset_item = dataset_map.get(query_id)

            if dataset_item is None:

                print(
                    f"  {query_id}: Dataset entry not found. Skipping."
                )

                continue

            print(
                f"\n  Evaluating {query_id}: "
                f"{dataset_item['question'][:65]}..."
            )

            evaluation = evaluate_answer(
                result,
                dataset_item
            )

            config_results.append(evaluation)

            print(
                f"    Status: {evaluation['status']}"
            )

            if evaluation["status"] == "evaluated":

                print(
                    f"    Semantic coverage: "
                    f"{evaluation['semantic_coverage']}%"
                )

                for item in evaluation["evaluations"]:

                    status = (
                        "Covered"
                        if item["covered"]
                        else "Missing"
                    )

                    print(
                        f"      [{status}] {item['key_point']}"
                    )

            elif evaluation["status"] == "unanswerable":

                print(
                    f"    Correct abstention: "
                    f"{evaluation['correct_abstention']}"
                )

            elif evaluation["status"] == "evaluation_error":

                print(
                    f"    Error: {evaluation['evaluation_error']}"
                )

        metrics = calculate_metrics(config_results)

        all_results[config_name] = {
            "queries": config_results,
            "metrics": metrics
        }

        print("\n  Configuration Summary")
        print("  " + "-" * 35)

        for metric, value in metrics.items():

            print(
                f"  {metric}: {value}"
            )

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    output = {
        "experiment": "Answer Correctness Evaluation",
        "evaluation_method": "LLM-as-a-Judge",
        "judge_model": LLM_MODEL,
        "results": all_results
    }

    Path(RESULTS_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
