
import json
import requests
from pathlib import Path

from config import (
    OLLAMA_URL,
    LLM_MODEL,
    RESULTS_DIR
)


# ==================================================
# CONFIGURATION
# ==================================================

INPUT_PATH = Path(RESULTS_DIR) / "generation_evaluation.json"

OUTPUT_PATH = Path(RESULTS_DIR) / "citation_evaluation.json"

MAX_RETRIES = 2
TIMEOUT = 300
MAX_CLAIMS = 4


# ==================================================
# JUDGE JSON SCHEMA
# ==================================================

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string"
                    },
                    "supporting_sources": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "supported": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "claim",
                    "supporting_sources",
                    "supported"
                ],
                "additionalProperties": False
            }
        },
        "citation_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string"
                    },
                    "supports_answer": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "source_id",
                    "supports_answer"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "claims",
        "citation_assessments"
    ],
    "additionalProperties": False
}


# ==================================================
# LOAD JSON
# ==================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==================================================
# BUILD RETRIEVED CONTEXT
# ==================================================

def build_context(chunks):

    context_parts = []

    for index, chunk in enumerate(chunks, 1):

        source_id = f"S{index}"

        context_parts.append(
            f"""
[{source_id}]
Source: {chunk.get('source', 'Unknown')}
Page: {chunk.get('page_number', 'Unknown')}
Content:
{chunk.get('content', '')}
"""
        )

    return "\n".join(context_parts)


# ==================================================
# VALIDATE CITATION IDS
# ==================================================

def validate_citations(citations, chunks):

    if not isinstance(citations, list):

        return (
            False,
            [],
            "invalid_citation_structure"
        )

    if not all(
        isinstance(item, str)
        for item in citations
    ):

        return (
            False,
            [],
            "invalid_citation_format"
        )

    if len(citations) != len(set(citations)):

        return (
            False,
            [],
            "duplicate_citation_ids"
        )

    valid_ids = {
        f"S{i}"
        for i in range(1, len(chunks) + 1)
    }

    invalid_ids = [
        citation
        for citation in citations
        if citation not in valid_ids
    ]

    if invalid_ids:

        return (
            False,
            invalid_ids,
            "invalid_citation_ids"
        )

    return True, [], "valid"


# ==================================================
# LLM-AS-A-JUDGE
# ==================================================

def call_citation_judge(
    question,
    answer,
    citations,
    chunks
):

    context = build_context(chunks)

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Evaluate whether the generated answer is supported by its cited sources.

QUESTION:
{question}

ANSWER:
{answer}

ACTUAL CITATIONS:
{json.dumps(citations)}

RETRIEVED CONTEXT:
{context}

RULES:

1. Extract at most {MAX_CLAIMS} important factual claims.
2. Merge closely related claims to keep output concise.
3. Identify which ACTUAL CITATIONS directly support each claim.
4. Only sources in ACTUAL CITATIONS can support a claim.
5. A claim is supported only when its cited source provides evidence.
6. Do not use external knowledge.
7. Evaluate every ID in ACTUAL CITATIONS.
8. A citation supports the answer if it supports at least one factual claim.
9. Do not invent citation IDs.
10. Keep claim descriptions short.
11. Return every actual citation ID exactly once in citation_assessments.
12. If a claim has no supporting cited source, use an empty list and false.

Return valid JSON matching the required schema.
"""

    for attempt in range(MAX_RETRIES + 1):

        raw_content = ""

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
                        "num_predict": 1000
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

            if not raw_content.strip():

                raise ValueError(
                    "Empty judge response."
                )

            parsed = json.loads(raw_content)

            claims = parsed.get("claims")
            assessments = parsed.get(
                "citation_assessments"
            )

            if not isinstance(claims, list):

                raise ValueError(
                    "Invalid claims structure."
                )

            if not isinstance(assessments, list):

                raise ValueError(
                    "Invalid citation assessments."
                )

            if len(claims) > MAX_CLAIMS:

                raise ValueError(
                    "Judge returned too many claims."
                )

            cited_ids = set(citations)

            # --------------------------------------
            # VALIDATE CLAIMS
            # --------------------------------------

            for claim in claims:

                if not isinstance(claim, dict):

                    raise ValueError(
                        "Invalid claim item."
                    )

                claim_text = claim.get("claim")

                if (
                    not isinstance(claim_text, str)
                    or not claim_text.strip()
                ):

                    raise ValueError(
                        "Invalid claim text."
                    )

                if not isinstance(
                    claim.get("supported"),
                    bool
                ):

                    raise ValueError(
                        "Invalid supported value."
                    )

                supporting_sources = claim.get(
                    "supporting_sources"
                )

                if not isinstance(
                    supporting_sources,
                    list
                ):

                    raise ValueError(
                        "Invalid supporting sources."
                    )

                if not all(
                    isinstance(source, str)
                    for source in supporting_sources
                ):

                    raise ValueError(
                        "Source IDs must be strings."
                    )

                if len(supporting_sources) != len(
                    set(supporting_sources)
                ):

                    raise ValueError(
                        "Duplicate supporting source IDs."
                    )

                # Separate cited and uncited sources.
                uncited_sources = [
                    source
                    for source in supporting_sources
                    if source not in cited_ids
                ]

                valid_supporting_sources = [
                    source
                    for source in supporting_sources
                    if source in cited_ids
                ]

                claim["uncited_sources"] = uncited_sources

                claim["supporting_sources"] = (
                    valid_supporting_sources
                )

                # A claim cannot be grounded without
                # at least one source actually cited.
                if not valid_supporting_sources:

                    claim["supported"] = False

            # --------------------------------------
            # VALIDATE CITATION ASSESSMENTS
            # --------------------------------------

            assessment_map = {}

            unexpected_assessments = []

            for item in assessments:

                if not isinstance(item, dict):

                    raise ValueError(
                        "Invalid citation assessment item."
                    )

                source_id = item.get("source_id")

                supports_answer = item.get(
                    "supports_answer"
                )

                if not isinstance(source_id, str):

                    raise ValueError(
                        "Invalid source ID."
                    )

                if not isinstance(
                    supports_answer,
                    bool
                ):

                    raise ValueError(
                        "Invalid supports_answer value."
                    )

                # Ignore sources that were not cited
                # by the original generator.
                if source_id not in cited_ids:

                    unexpected_assessments.append(
                        source_id
                    )

                    continue

                if source_id in assessment_map:

                    raise ValueError(
                        f"Duplicate assessment: {source_id}"
                    )

                assessment_map[source_id] = {
                    "source_id": source_id,
                    "supports_answer": supports_answer
                }

            # Missing assessments are treated as
            # unsupported, never as useful citations.
            for source_id in citations:

                if source_id not in assessment_map:

                    assessment_map[source_id] = {
                        "source_id": source_id,
                        "supports_answer": False
                    }

            return {
                "success": True,
                "claims": claims,
                "citation_assessments": [
                    assessment_map[source_id]
                    for source_id in citations
                ],
                "unexpected_assessments": list(
                    dict.fromkeys(unexpected_assessments)
                ),
                "error": None
            }

        except Exception as error:

            print(
                f"    Judge attempt {attempt + 1} failed: {error}"
            )

            if raw_content:

                print("    Raw judge output:")

                print(raw_content[:1000])

    return {
        "success": False,
        "claims": [],
        "citation_assessments": [],
        "unexpected_assessments": [],
        "error": "judge_failed_after_retries"
    }


# ==================================================
# EVALUATE ONE ANSWER
# ==================================================

def evaluate_citation(result):

    query_id = result.get("id")

    question = result.get("question", "")

    answer = result.get("answer", "")

    citations = result.get("citations", [])

    chunks = result.get("retrieved_chunks", [])

    abstained = result.get("insufficient_evidence")

    base_result = {
        "id": query_id,
        "question": question,
        "answer": answer,
        "citations": citations
    }

    # ----------------------------------------------
    # ABSTENTION
    # ----------------------------------------------

    if abstained is True:

        return {
            **base_result,
            "status": "abstained",
            "citation_validity": None,
            "groundedness": None,
            "citation_precision": None,
            "claims": [],
            "citation_assessments": []
        }

    # ----------------------------------------------
    # ANSWER VALIDATION
    # ----------------------------------------------

    if (
        not isinstance(answer, str)
        or not answer.strip()
    ):

        return {
            **base_result,
            "status": "invalid_answer",
            "citation_validity": False,
            "groundedness": None,
            "citation_precision": None,
            "claims": [],
            "citation_assessments": []
        }

    # ----------------------------------------------
    # CITATION VALIDATION
    # ----------------------------------------------

    valid, invalid_ids, validation_status = (
        validate_citations(
            citations,
            chunks
        )
    )

    if not valid:

        return {
            **base_result,
            "status": "invalid_citations",
            "citation_validity": False,
            "invalid_citation_ids": invalid_ids,
            "validation_status": validation_status,
            "groundedness": None,
            "citation_precision": None,
            "claims": [],
            "citation_assessments": []
        }

    if not citations:

        return {
            **base_result,
            "status": "missing_citations",
            "citation_validity": True,
            "groundedness": None,
            "citation_precision": None,
            "claims": [],
            "citation_assessments": []
        }

    # ----------------------------------------------
    # CALL JUDGE
    # ----------------------------------------------

    judge_result = call_citation_judge(
        question=question,
        answer=answer,
        citations=citations,
        chunks=chunks
    )

    if not judge_result["success"]:

        return {
            **base_result,
            "status": "evaluation_error",
            "citation_validity": True,
            "groundedness": None,
            "citation_precision": None,
            "evaluation_error": judge_result["error"],
            "claims": [],
            "citation_assessments": []
        }

    claims = judge_result["claims"]

    assessments = judge_result["citation_assessments"]

    # ----------------------------------------------
    # GROUNDEDNESS
    # ----------------------------------------------

    total_claims = len(claims)

    supported_claims = sum(
        1
        for claim in claims
        if claim["supported"]
    )

    groundedness = (
        supported_claims / total_claims * 100
        if total_claims > 0
        else None
    )

    # ----------------------------------------------
    # CITATION PRECISION
    # ----------------------------------------------

    total_citations = len(assessments)

    useful_citations = sum(
        1
        for item in assessments
        if item["supports_answer"]
    )

    citation_precision = (
        useful_citations / total_citations * 100
        if total_citations > 0
        else None
    )

    return {
        **base_result,
        "status": (
            "evaluated"
            if total_claims > 0
            else "no_factual_claims"
        ),
        "citation_validity": True,
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": (
            total_claims - supported_claims
        ),
        "groundedness": (
            round(groundedness, 2)
            if groundedness is not None
            else None
        ),
        "total_citations": total_citations,
        "useful_citations": useful_citations,
        "citation_precision": (
            round(citation_precision, 2)
            if citation_precision is not None
            else None
        ),
        "unexpected_assessments": judge_result.get(
            "unexpected_assessments",
            []
        ),
        "claims": claims,
        "citation_assessments": assessments
    }


# ==================================================
# AGGREGATE METRICS
# ==================================================

def calculate_metrics(results):

    evaluated = [
        item
        for item in results
        if item["status"] == "evaluated"
    ]

    abstained = [
        item
        for item in results
        if item["status"] == "abstained"
    ]

    errors = [
        item
        for item in results
        if item["status"] == "evaluation_error"
    ]

    total_claims = sum(
        item["total_claims"]
        for item in evaluated
    )

    supported_claims = sum(
        item["supported_claims"]
        for item in evaluated
    )

    total_citations = sum(
        item["total_citations"]
        for item in evaluated
    )

    useful_citations = sum(
        item["useful_citations"]
        for item in evaluated
    )

    groundedness = (
        supported_claims / total_claims * 100
        if total_claims > 0
        else 0
    )

    citation_precision = (
        useful_citations / total_citations * 100
        if total_citations > 0
        else 0
    )

    return {
        "total_questions": len(results),
        "evaluated_answers": len(evaluated),
        "abstained_answers": len(abstained),
        "evaluation_errors": len(errors),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": (
            total_claims - supported_claims
        ),
        "groundedness": round(groundedness, 2),
        "total_citations": total_citations,
        "useful_citations": useful_citations,
        "citation_precision": round(
            citation_precision, 2
        )
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 60)

    print("CITATION CORRECTNESS EVALUATION")

    print("=" * 60)

    generation_data = load_json(INPUT_PATH)

    all_results = {}

    for config_id, config_data in generation_data["results"].items():

        print(
            f"\nConfiguration: {config_id}"
        )

        print("-" * 60)

        config_results = []

        for result in config_data.get("queries", []):

            query_id = result.get("id")

            print(
                f"\n  Evaluating {query_id}..."
            )

            evaluation = evaluate_citation(result)

            config_results.append(evaluation)

            print(
                f"    Status: {evaluation['status']}"
            )

            print(
                f"    Citations: {evaluation['citations']}"
            )

            if evaluation["status"] == "evaluated":

                print(
                    f"    Groundedness: "
                    f"{evaluation['groundedness']}%"
                )

                print(
                    f"    Citation precision: "
                    f"{evaluation['citation_precision']}%"
                )

                for claim in evaluation["claims"]:

                    status = (
                        "SUPPORTED"
                        if claim["supported"]
                        else "UNSUPPORTED"
                    )

                    print(
                        f"      [{status}] {claim['claim']}"
                    )

                    if claim["uncited_sources"]:

                        print(
                            "        Uncited sources returned by judge:",
                            claim["uncited_sources"]
                        )

                if evaluation["unexpected_assessments"]:

                    print(
                        "    Unexpected citation assessments:",
                        evaluation["unexpected_assessments"]
                    )

            elif evaluation["status"] == "evaluation_error":

                print(
                    f"    Error: {evaluation['evaluation_error']}"
                )

        metrics = calculate_metrics(config_results)

        all_results[config_id] = {
            "queries": config_results,
            "metrics": metrics
        }

        print(
            "\n  Configuration Summary"
        )

        print(
            "  " + "-" * 35
        )

        for metric, value in metrics.items():

            print(
                f"  {metric}: {value}"
            )

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    output = {
        "experiment": "Citation Correctness Evaluation",
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

    print("CITATION EVALUATION COMPLETED")

    print("=" * 60)

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()
