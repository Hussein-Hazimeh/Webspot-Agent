# RAG Agent with PydanticAI, OpenAI, and Pinecone

This project implements a Retrieval-Augmented Generation (RAG) agent using PydanticAI for the agent framework, OpenAI for embeddings and LLM capabilities, and Pinecone as the vector database.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
```

3. Make sure to create an index in Pinecone with the appropriate dimension size (1536 for OpenAI's text-embedding-3-small model).

## Usage

The RAG agent provides a simple interface to query your knowledge base:

```python
import asyncio
from rag_agent import RAGAgent

async def main():
    rag = RAGAgent()
    result = await rag.query("Your question here")
    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")
    print(f"Confidence: {result.confidence}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- Uses PydanticAI for type-safe agent implementation
- Leverages OpenAI's GPT-4 for question answering
- Uses OpenAI's text-embedding-3-small for generating embeddings
- Stores and retrieves context from Pinecone vector database
- Returns structured responses with answer, sources, and confidence score
