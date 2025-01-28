from index_documents import index_documents

# Example documents
texts = [
    "The sky is blue because of Rayleigh scattering of sunlight.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth completes one rotation around its axis in approximately 24 hours."
]

metadata = [
    {"source": "science_book", "topic": "atmosphere"},
    {"source": "science_book", "topic": "physics"},
    {"source": "science_book", "topic": "astronomy"}
]

# Index the documents
index_documents(texts, metadata)

# Query the RAG agent
import asyncio
from rag_agent import query_rag_agent

async def main():
    question = "what is Gourmet Haven?"
    answer = await query_rag_agent(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main()) 