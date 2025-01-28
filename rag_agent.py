from dataclasses import dataclass
from typing import List
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pinecone import Pinecone
from config import (
    OPENAI_API_KEY, 
    PINECONE_API_KEY, 
    PINECONE_ENVIRONMENT, 
    PINECONE_INDEX_NAME
)

# Initialize Pinecone with new client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

@dataclass
class Deps:
    openai: AsyncOpenAI
    pinecone_index: Pinecone.Index

# Initialize the agent
rag_agent = Agent(
    "openai:gpt-4",
    system_prompt=(
        "You are a helpful assistant that answers questions based on the retrieved context. "
        "Always use the retrieved information to formulate your answers. "
        "If you cannot find relevant information in the context, say so."
    ),
    deps_type=Deps
)

@rag_agent.tool
async def retrieve(context: RunContext[Deps], query: str) -> str:
    """Retrieve relevant documents based on the query using vector similarity search.
    
    Args:
        context: The run context containing dependencies
        query: The search query
    
    Returns:
        str: Retrieved context from the vector database
    """
    # Generate embedding for the query
    embedding_response = await context.deps.openai.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = embedding_response.data[0].embedding
    
    # Search Pinecone
    search_results = context.deps.pinecone_index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    # Format results
    contexts = []
    for match in search_results.matches:
        if match.score > 0.7:  # Only include relevant matches
            contexts.append(f"Context (relevance: {match.score:.2f}):\n{match.metadata['text']}")
    
    return "\n\n".join(contexts) if contexts else "No relevant context found."

async def query_rag_agent(question: str) -> str:
    """Query the RAG agent with a question.
    
    Args:
        question: The question to ask
        
    Returns:
        str: The agent's response
    """
    # Initialize dependencies
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    deps = Deps(openai=openai_client, pinecone_index=index)
    
    # Run the agent
    result = await rag_agent.run(
        f"Use the retrieve tool to find relevant information and answer this question: {question}",
        deps=deps
    )
    
    return result.data 