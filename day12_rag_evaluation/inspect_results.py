
import json

with open("results/generation_evaluation.json", "r") as file:
    data = json.load(file)

for config in ["A", "B"]:

    print("\n" + "=" * 70)
    print(f"CONFIGURATION {config}")
    print("=" * 70)

    for query_id in ["q3", "q4"]:

        result = next(
            item
            for item in data["results"][config]["queries"]
            if item["id"] == query_id
        )

        print(f"\n{query_id}: {result['question']}")
        print("Generated answer:", result["answer"])
        print("\nRetrieved chunks:")

        for index, chunk in enumerate(result["retrieved_chunks"], 1):

            print(
                f"\nS{index} | Chunk ID: {chunk['id']}"
                f" | Similarity: {chunk['similarity']:.4f}"
            )

            print(chunk["content"][:500])
