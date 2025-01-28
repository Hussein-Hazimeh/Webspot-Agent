import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import pinecone
from openai import OpenAI

load_dotenv()

@dataclass
class RAGDependencies:
    pinecone_index: pinecone.Index
    openai_client: OpenAI
    
class SearchResult(BaseModel):
    answer: str = Field(description='The answer to the user query')
    sources: List[str] = Field(description='The sources used to generate the answer')
    confidence: float = Field(description='Confidence score of the answer', ge=0, le=1)

class RAGAgent:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        self.index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
        
        # Initialize PydanticAI agent
        self.agent = Agent(
            'openai:gpt-4',
            deps_type=RAGDependencies,
            result_type=SearchResult,
            system_prompt=(
                'You are a helpful assistant that answers questions based on the retrieved context. '
                'Always provide accurate information and cite your sources.'
            )
        )
        
        self.deps = RAGDependencies(
            pinecone_index=self.index,
            openai_client=self.openai_client
        )

    async def _get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the input text using OpenAI."""
        response = await self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    @Agent.tool
    async def search_context(self, ctx: RunContext[RAGDependencies], query: str) -> List[dict]:
        """Search for relevant context in the vector database."""
        query_embedding = await self._get_embeddings(query)
        results = ctx.deps.pinecone_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        return [
            {
                'content': match.metadata.get('content', ''),
                'source': match.metadata.get('source', 'Unknown'),
                'score': match.score
            }
            for match in results.matches
        ]

    async def query(self, question: str) -> SearchResult:
        """Query the RAG system with a question."""
        result = await self.agent.run(question, deps=self.deps)
        return result.data

async def main():
    # Example usage
    rag = RAGAgent()
    result = await rag.query("What is the capital of France?")
    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")
    print(f"Confidence: {result.confidence}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
