
import json
import os
import requests

from config import (
    OLLAMA_URL,
    LLM_MODEL,
    CONFIGURATIONS,
    DATASET_PATH,
    RESULTS_DIR
)

from retriever import retrieve_chunks


def validate_generation(result, chunks, answerable):
    """
    Validate generated answer structure, citations,
    abstention flag, and expected answerability.
    """

    if not isinstance(result, dict):
        return False, "invalid_json_structure"

    answer = result.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        return False, "empty_answer"

    normalized_answer = answer.strip().lower().rstrip(".! ")

    placeholders = {
        "your answer",
        "your response",
        "insert answer here",
        "generated answer",
        "answer here"
    }

    if normalized_answer in placeholders:
        return False, "placeholder_answer"

    citations = result.get("citations")

    if not isinstance(citations, list):
        return False, "invalid_citations"

    if not all(isinstance(c, str) for c in citations):
        return False, "invalid_citation_format"

    valid_ids = {
        f"S{i}"
        for i in range(1, len(chunks) + 1)
    }

    invalid_citations = [
        citation
        for citation in citations
        if citation not in valid_ids
    ]

    if invalid_citations:
        return False, "invalid_citation_ids"

    insufficient_evidence = result.get("insufficient_evidence")

    if not isinstance(insufficient_evidence, bool):
        return False, "invalid_abstention_flag"

    # Check whether the model's abstention decision
    # matches the dataset's expected answerability.
    if answerable and insufficient_evidence:
        return False, "unexpected_abstention"

    if not answerable and not insufficient_evidence:
        return False, "failed_to_abstain"

    # An abstention should not cite sources.
    if insufficient_evidence and citations:
        return False, "citations_present_during_abstention"

    # A non-abstaining answer should cite at least one source.
    if not insufficient_evidence and not citations:
        return False, "missing_citations"

    return True, "valid"


def generate_answer(question, chunks):
    """
    Generate a grounded answer using retrieved context.
    """

    if not chunks:
        return {
            "answer": "Insufficient evidence to answer this question.",
            "citations": [],
            "insufficient_evidence": True,
            "raw_output": None
        }

    # Construct context with source IDs.
    context_parts = []

    for index, chunk in enumerate(chunks, 1):

        context_parts.append(
            f"""
[S{index}]
Source: {chunk['source']}
Page: {chunk['page_number']}
Content: {chunk['content']}
"""
        )

    context = "\n".join(context_parts)

    # Construct generation prompt.
    prompt = f"""
You are a grounded research assistant.

Answer the question using ONLY the supplied context.

STRICT RULES:

1. Never use external knowledge.
2. Never invent information.
3. If the context does not contain enough evidence,
   explicitly abstain.
4. Never return placeholder text.
5. Every factual claim must be supported by the context.
6. Cite only the exact source IDs that support your answer.
7. Do not cite sources that are merely related to the topic.
8. Return valid JSON only.

If you cannot answer, return:

{{
    "answer": "Insufficient evidence to answer this question.",
    "citations": [],
    "insufficient_evidence": true
}}

QUESTION:
{question}

CONTEXT:
{context}

If the context supports the answer, use this structure:

{{
    "answer": "Your evidence-based answer goes here.",
    "citations": ["S1"],
    "insufficient_evidence": false
}}
"""

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
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 400
                }
            },
            timeout=300
        )

        response.raise_for_status()

        response_data = response.json()

        raw_output = (
            response_data
            .get("message", {})
            .get("content", "")
        )

        if not raw_output.strip():
            return {
                "answer": "",
                "citations": [],
                "insufficient_evidence": False,
                "raw_output": raw_output
            }

        # Parse generated JSON.
        try:
            result = json.loads(raw_output)

        except json.JSONDecodeError:
            return {
                "answer": raw_output,
                "citations": [],
                "insufficient_evidence": False,
                "raw_output": raw_output
            }

        if not isinstance(result, dict):
            return {
                "answer": "",
                "citations": [],
                "insufficient_evidence": False,
                "raw_output": raw_output
            }

        # Preserve the raw response for debugging.
        result["raw_output"] = raw_output

        return result

    except requests.RequestException as error:

        print(f"Ollama request failed: {error}")

        return {
            "answer": "",
            "citations": [],
            "insufficient_evidence": False,
            "raw_output": None,
            "generation_error": str(error)
        }


def main():

    # Load evaluation dataset.
    with open(DATASET_PATH, "r") as file:
        dataset = json.load(file)

    all_results = {}

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run generation for each configuration.
    for config_id, config in CONFIGURATIONS.items():

        top_k = config["top_k"]

        print(f"\nRunning generation configuration: {config_id}")
        print(f"Top K: {top_k}")

        config_results = []

        for item in dataset:

            question = item["question"]
            answerable = item["answerable"]

            print(
                f"\nGenerating answer for {item['id']}: "
                f"{question}"
            )

            # Retrieve relevant chunks.
            chunks = retrieve_chunks(
                query=question,
                top_k=top_k
            )

            # Generate answer.
            generated = generate_answer(
                question=question,
                chunks=chunks
            )

            # Validate generated output.
            is_valid, validation_status = validate_generation(
                generated,
                chunks,
                answerable
            )

            result = {
                "id": item["id"],
                "question": question,
                "ground_truth": item["ground_truth"],
                "answerable": answerable,
                "answer": generated.get("answer", ""),
                "citations": generated.get("citations", []),
                "insufficient_evidence": generated.get(
                    "insufficient_evidence"
                ),
                "generation_valid": is_valid,
                "validation_status": validation_status,
                "generation_error": generated.get("generation_error"),
                "raw_output": generated.get("raw_output"),
                "retrieved_chunks": chunks
            }

            config_results.append(result)

            print("\nGenerated answer:")
            print(result["answer"])

            print("Citations:", result["citations"])

            print(
                "Insufficient evidence:",
                result["insufficient_evidence"]
            )

            print("Validation:", validation_status)

        all_results[config_id] = {
            "top_k": top_k,
            "queries": config_results
        }

    # Prepare final experiment output.
    output = {
        "experiment": "generation_evaluation",
        "results": all_results
    }

    # Save results.
    output_path = os.path.join(
        RESULTS_DIR,
        "generation_evaluation.json"
    )

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 60)
    print("GENERATION EXPERIMENT COMPLETED")
    print("=" * 60)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
