
from db import get_chunk_count
from retriever import retrieve_chunks


print("Total database chunks:", get_chunk_count())

query = "How does the Proof-of-QK Stake consensus mechanism work?"

for top_k in [5, 10]:

    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL WITH TOP_K = {top_k}")
    print("=" * 60)

    chunks = retrieve_chunks(query, top_k=top_k)

    print("Retrieved chunks:", len(chunks))

    for rank, chunk in enumerate(chunks, start=1):
        print(f"\nRank: {rank}")
        print("Source:", chunk["source"])
        print("Page:", chunk["page_number"])
        print("Chunk ID:", chunk["id"])
        print("Similarity:", round(chunk["similarity"], 4))
        print("Content:", chunk["content"][:300])
