from typing import List
from openai import OpenAI
from pinecone import Pinecone
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of texts using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = []
    
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    
    return embeddings

def index_documents(texts: List[str], metadata: List[dict]):
    """Index documents in Pinecone."""
    # Initialize Pinecone with new client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine'
        )
    
    # Get the index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Create embeddings
    embeddings = create_embeddings(texts)
    
    # Prepare vectors for upsert
    vectors = []
    for i, (embedding, text, meta) in enumerate(zip(embeddings, texts, metadata)):
        meta["text"] = text  # Include the text in metadata for retrieval
        vectors.append((str(i), embedding, meta))
    
    # Upsert to Pinecone
    index.upsert(vectors=vectors) 