import json
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
"""
this script is used to add a jsonl from EDA CORPUS  file to a qdrant collection using openai embeddings.
"""
# --- Configuration ---
JSONL_FILE_PATH = "/home/german/Desktop/cse291/camel/query_dataset.jsonl"  # Path to your source JSONL
QDRANT_PATH = "vector_db/"  # Path to your Qdrant data directory
COLLECTION_NAME = "documents_collection"  # The collection name you used
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI's embedding model
OPENAI_BATCH_SIZE = 20  # OpenAI recommends smaller batches for embeddings
QDRANT_BATCH_SIZE = 100  # For Qdrant uploads
QDRANT_TEXT_PAYLOAD_KEY = "text"  # Payload field in Qdrant that stores the text
EMBEDDING_DIMENSION = 3072  # Dimension for text-embedding-ada-002
# ---------------------
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_jsonl(file_path):
    """Load JSONL file and return list of items."""
    items = []
    if not os.path.exists(file_path):
        print(f"Error: JSONL file '{file_path}' not found.")
        return items
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line: {line[:100]}...")
        print(f"Loaded {len(items)} items from '{file_path}'")
        return items
    except Exception as e:
        print(f"Error reading JSONL file '{file_path}': {e}")
        return []

def get_openai_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI's API."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [embedding_data.embedding for embedding_data in response.data]
    except Exception as e:
        print(f"Error getting embeddings from OpenAI: {e}")
        raise

def batch_get_embeddings(texts: List[str], batch_size: int = OPENAI_BATCH_SIZE) -> List[List[float]]:
    """Process texts in batches to avoid rate limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Getting embeddings for batch {i//batch_size + 1} ({len(batch)} items)")
        try:
            batch_embeddings = get_openai_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            # Small delay to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.5)
        except Exception as e:
            print(f"Error on batch starting at index {i}: {e}")
            # Add placeholder embeddings (all zeros) for failed batch
            all_embeddings.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch))
    return all_embeddings

def ingest_to_qdrant():
    """Ingest JSONL content into Qdrant using OpenAI embeddings."""
    print(f"Starting ingestion of '{JSONL_FILE_PATH}' to Qdrant collection '{COLLECTION_NAME}'")
    
    # 1. Load data from JSONL
    items = load_jsonl(JSONL_FILE_PATH)
    if not items:
        print("No items loaded. Aborting.")
        return
    
    # 2. Initialize Qdrant client
    try:
        client = QdrantClient(path=QDRANT_PATH)
        print(f"Connected to Qdrant at '{QDRANT_PATH}'")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return
    
    # 3. Check if collection exists, create if it doesn't
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME in collection_names:
            # Check if vector size matches
            collection_info = client.get_collection(COLLECTION_NAME)
            existing_vector_size = collection_info.config.params.vectors.size
            if existing_vector_size != EMBEDDING_DIMENSION:
                print(f"Error: Existing collection has vector size {existing_vector_size}, but OpenAI embeddings have size {EMBEDDING_DIMENSION}")
                print("To use OpenAI embeddings, you need to create a new collection with matching vector size.")
                return
            print(f"Found existing collection '{COLLECTION_NAME}' with {collection_info.points_count} points")
        else:
            # Create new collection
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Error checking/creating collection: {e}")
        return
    
    # 4. Prepare items for embedding
    item_strings = [json.dumps(item) for item in items]
    print(f"Getting embeddings for {len(item_strings)} items using OpenAI API...")
    
    # 5. Get embeddings from OpenAI in batches
    try:
        embeddings = batch_get_embeddings(item_strings)
        if len(embeddings) != len(items):
            print(f"Warning: Got {len(embeddings)} embeddings for {len(items)} items")
            print("Proceeding with available embeddings only")
    except Exception as e:
        print(f"Fatal error getting embeddings: {e}")
        return
    
    # 6. Process items and upsert to Qdrant
    batch = []
    total_processed = 0
    
    for i, (item_str, embedding) in enumerate(zip(item_strings, embeddings)):
        try:
            # Skip items with zero embeddings (failed to embed)
            if np.all(np.array(embedding) == 0):
                print(f"Skipping item {i} due to embedding failure")
                continue
                
            # Create point
            point = models.PointStruct(
                id=i+1,  # Use index+1 as ID
                vector=embedding,
                payload={
                    QDRANT_TEXT_PAYLOAD_KEY: item_str
                }
            )
            batch.append(point)
            
            # Process batch if batch size reached
            if len(batch) >= QDRANT_BATCH_SIZE:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                print(f"Processed {total_processed + len(batch)} / {len(items)} items")
                total_processed += len(batch)
                batch = []
        except Exception as e:
            print(f"Error processing item {i}: {e}")
    
    # Process remaining batch
    if batch:
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            total_processed += len(batch)
        except Exception as e:
            print(f"Error processing final batch: {e}")
    
    print(f"Ingestion complete. Successfully processed {total_processed} / {len(items)} items.")
    
    # 7. Verify results
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' now has {collection_info.points_count} points")
    except Exception as e:
        print(f"Error getting collection info: {e}")

if __name__ == "__main__":
    ingest_to_qdrant()