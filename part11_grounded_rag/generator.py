
import json
import requests

from config import OLLAMA_URL, LLM_MODEL


SYSTEM_PROMPT = """
You are a grounded question-answering assistant.

Answer using ONLY the retrieved context.

Rules:
1. Never use external or pretrained knowledge.
2. Never invent facts or statistics.
3. If evidence is insufficient, explicitly say so.
4. If only part of the question is answerable,
   explain what is supported and what is missing.
5. Every factual claim must have supporting source IDs.
6. Only cite source IDs present in the context.
7. Mention conflicting evidence if sources disagree.

Return valid JSON with this structure:

{
    "answer": "Your answer",
    "insufficient_evidence": false,
    "citations": ["S1"]
}

If evidence is insufficient, set insufficient_evidence
to true and use an empty citations list if nothing
supports the answer.
"""


def generate_answer(question, context):

    url = f"{OLLAMA_URL}/api/chat"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
Retrieved Context:

{context}

Question:

{question}

Answer using only the retrieved context.
"""
        }
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 400
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    message = data.get("message", {})
    raw_content = message.get("content", "").strip()

    # Diagnostic information if Qwen returns empty content.
    if not raw_content:
        thinking = message.get("thinking", "")

        raise ValueError(
            "Ollama returned empty message content. "
            f"Thinking field: {thinking[:500]}. "
            f"Full response: {data}"
        )

    try:
        result = json.loads(raw_content)

    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON returned by Ollama: {raw_content}"
        )

    required_fields = [
        "answer",
        "insufficient_evidence",
        "citations"
    ]

    for field in required_fields:
        if field not in result:
            raise ValueError(
                f"Missing field in LLM response: {field}"
            )

    if not isinstance(result["answer"], str):
        raise ValueError("Answer must be a string.")

    if not isinstance(result["insufficient_evidence"], bool):
        raise ValueError(
            "insufficient_evidence must be a boolean."
        )

    if not isinstance(result["citations"], list):
        raise ValueError("Citations must be a list.")

    return result
