
import json
from typing import Literal

import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MODEL = "qwen3:4b"
MAX_ATTEMPTS = 3


# 1. Define the structure and validation rules.

class SupportTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: str | None = Field(
        description="Customer name, or null if not provided."
    )

    issue_summary: str = Field(
        min_length=5,
        description="A concise summary of the customer's issue."
    )

    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Priority inferred from the issue and urgency."
    )

    status: Literal["open", "in_progress", "resolved"] = Field(
        description="Current status explicitly supported by the text."
    )


# 2. Ask the model to extract structured information.

def extract_ticket(text: str) -> SupportTicket:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract a support ticket from the user's text. "
                "Use only information supported by the text. "
                "Do not invent customer details. "
                "If the customer name is missing, use null. "
                "Use status 'open' unless the text clearly establishes "
                "that work is in progress or the issue is resolved. "
                "Infer priority from the urgency and impact described."
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    schema = SupportTicket.model_json_schema()
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                format=schema,
                think=False,
                options={"temperature": 0},
            )

            raw_output = response.message.content

            # 3. Parse JSON and validate against the Pydantic model.
            ticket = SupportTicket.model_validate_json(raw_output)

            return ticket

        except ValidationError as error:
            last_error = error

            if attempt == MAX_ATTEMPTS:
                break

            # 4. Ask the model to correct the invalid output.
            messages.append({
                "role": "assistant",
                "content": raw_output,
            })
            messages.append({
                "role": "user",
                "content": (
                    "The previous output failed validation. "
                    "Correct it to satisfy the required schema. "
                    f"Validation feedback: {error}"
                ),
            })

            print(
                f"Validation failed on attempt {attempt}; retrying..."
            )

    raise ValueError(
        f"Could not produce a valid support ticket "
        f"after {MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )


# 5. Run the extractor.

def main():
    text = input("Enter a customer support message:\n> ").strip()

    if not text:
        print("Error: input cannot be empty.")
        return

    try:
        ticket = extract_ticket(text)

        print("\nValidated support ticket:")
        print(ticket.model_dump_json(indent=2))

        print("\nPython object:")
        print(ticket)

    except ValidationError as error:
        print("\nValidation error:")
        print(error)

    except ValueError as error:
        print(f"\nExtraction failed: {error}")

    except Exception as error:
        # Handles connection errors and unexpected runtime failures.
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
