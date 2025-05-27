import os
import argparse
from dotenv import load_dotenv

# CAMEL components are not strictly needed for basic Qdrant client inspection,
# but if we want to do a test query with embeddings, we'll need the embedding model.
from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType

# Direct Qdrant client for more detailed inspection
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScrollRequest, Filter

"""
para desbloquear el lock de qdrant
ps aux | grep qdrant
kill -9 
ps -fp <PID>

"""


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def setup_openai_api_key():
    """Checks if the OpenAI API key is available in the environment."""
    return bool(OPENAI_API_KEY)

def inspect_qdrant_collection(
    qdrant_path: str,
    collection_name: str,
    embedding_model_name: str = EmbeddingModelType.TEXT_EMBEDDING_3_LARGE,
    num_sample_points: int = 5,
    test_query: str = None
):
    """
    Inspects a Qdrant collection by showing point count, sample points,
    and optionally performing a test query.
    """
    print(f"--- Inspecting Qdrant Collection ---")
    print(f"Path: {qdrant_path}")
    print(f"Collection: {collection_name}\n")

    try:
        # For local file-based storage, QdrantClient can connect directly to the path
        client = QdrantClient(path=qdrant_path)
    except Exception as e:
        print(f"Error connecting to Qdrant at path '{qdrant_path}': {e}")
        print("Ensure the path is correct and Qdrant storage files exist there.")
        return

    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' found.")
        print(f"  Status: {collection_info.status}")
        print(f"  Points count: {collection_info.points_count}")
        print(f"  Vectors count: {collection_info.vectors_count}") # May differ if multiple named vectors
        print(f"  Segments count: {collection_info.segments_count}")
        # print(f"  Config: {collection_info.config}") # Can be verbose
    except Exception as e:
        print(f"Error getting info for collection '{collection_name}': {e}")
        print("Make sure the collection name is correct and was created by the processing script.")
        client.close()
        return

    if collection_info.points_count == 0:
        print("The collection is empty. No documents have been processed into it.")
        client.close()
        return

    # View sample points
    print(f"\n--- Sample Points (first {num_sample_points}) ---")
    try:
        scroll_response = client.scroll(
            collection_name=collection_name,
            scroll_filter=None, # No filter, get any points
            limit=num_sample_points,
            with_payload=True,
            with_vectors=False # Set to True if you want to see the raw vectors
        )
        if scroll_response[0]: # scroll_response is a tuple (points, next_page_offset)
            for i, point in enumerate(scroll_response[0]):
                print(f"\nPoint {i+1}:")
                print(f"  ID: {point.id}")
                print(f"  Payload (metadata): {point.payload}")
                # print(f"  Vector: {point.vector}") # Uncomment if with_vectors=True
        else:
            print("No points found with scroll (this shouldn't happen if points_count > 0).")

    except Exception as e:
        print(f"Error scrolling points: {e}")

    # Perform a test query if a query string is provided
    if test_query:
        print(f"\n--- Test Query ---")
        print(f"Query: '{test_query}'")
        if not setup_openai_api_key():
            print("Cannot perform test query without OpenAI API key for embeddings.")
            client.close()
            return

        try:
            embedding_instance = OpenAIEmbedding(model_type=embedding_model_name)
            query_vector = embedding_instance.get_embedding(test_query)

            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=3, # Get top 3 results
                with_payload=True
            )
            if search_results:
                print("Search Results:")
                for i, hit in enumerate(search_results):
                    print(f"\nResult {i+1}:")
                    print(f"  ID: {hit.id}")
                    print(f"  Score: {hit.score:.4f}")
                    print(f"  Payload: {hit.payload}")
            else:
                print("No results found for the test query.")

        except Exception as e:
            print(f"Error performing test query: {e}")

    client.close()
    print("\n--- Inspection Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Inspect a Qdrant collection.")
    parser.add_argument(
        "--qdrant_path",
        type=str,
        default="my_vectors",
        help="Path to the local Qdrant storage directory (default: my_vectors).",
    )
    parser.add_argument(
        "--qdrant_collection",
        type=str,
        default="or_rag_docs",
        help="Name of the Qdrant collection to inspect (default: or_rag_docs).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of sample points to display (default: 3)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="An optional test query to run against the collection."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE,
        help=f"OpenAI embedding model for test query. Default: {EmbeddingModelType.TEXT_EMBEDDING_3_LARGE}",
    )
    args = parser.parse_args()

    inspect_qdrant_collection(
        qdrant_path=args.qdrant_path,
        collection_name=args.qdrant_collection,
        embedding_model_name=args.embedding_model,
        num_sample_points=args.samples,
        test_query=args.query
    )

if __name__ == "__main__":
    main()