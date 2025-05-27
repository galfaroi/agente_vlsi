import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScrollRequest

# --- Configuration ---
QDRANT_PATH = "my_vectors"  # Path to your Qdrant data directory
COLLECTION_NAME = "or_rag_docs"  # The collection name you used
JSONL_FILE_PATH = "/home/german/Desktop/cse291/camel/query_dataset.jsonl" # Path to your source JSONL
# Path to the directory containing your original markdown source files
MARKDOWN_SOURCE_DIR = "/home/german/Desktop/cse291/camel/processed_output" # FIXME: Update this path
# Payload field in Qdrant that stores the text. CAMEL usually uses "text".
QDRANT_TEXT_PAYLOAD_KEY = "text"
# ---------------------

def get_qdrant_data(client, collection_name):
    """Retrieves all text data from the specified Qdrant collection."""
    qdrant_texts = set()
    qdrant_point_ids = set()
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        total_points = collection_info.points_count
        print(f"Qdrant collection '{collection_name}' has {total_points} points.")

        if total_points == 0:
            return qdrant_texts, qdrant_point_ids, total_points

        # Scroll through all points. Adjust limit if you have extremely large collections,
        # but for typical fine-tuning datasets, a single scroll might be enough.
        # Qdrant's scroll API handles pagination internally with the offset.
        offset = None
        while True:
            response, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=None, # No filter, get all
                limit=1000, # How many to retrieve per request
                offset=offset,
                with_payload=True,
                with_vectors=False # We don't need vectors for this check
            )
            for hit in response:
                if hit.payload and QDRANT_TEXT_PAYLOAD_KEY in hit.payload:
                    qdrant_texts.add(hit.payload[QDRANT_TEXT_PAYLOAD_KEY])
                    qdrant_point_ids.add(hit.id)
                else:
                    print(f"Warning: Point ID {hit.id} in Qdrant has no payload or no '{QDRANT_TEXT_PAYLOAD_KEY}' key.")
            
            if not next_offset:
                break
            offset = next_offset
            
        return qdrant_texts, qdrant_point_ids, total_points
        
    except Exception as e:
        if "Not found: Collection" in str(e) or "doesn't exist" in str(e):
            print(f"Error: Qdrant collection '{collection_name}' not found at path '{QDRANT_PATH}'.")
        else:
            print(f"Error connecting to or retrieving data from Qdrant: {e}")
        return None, None, 0

def get_jsonl_data(file_path):
    """Reads all lines from the JSONL file."""
    jsonl_texts = set()
    if not os.path.exists(file_path):
        print(f"Error: JSONL file '{file_path}' not found.")
        return None
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                # Assuming each line is a JSON string that was stored.
                # We strip it because Qdrant might store the stripped version,
                # or the original line might have trailing whitespace.
                jsonl_texts.add(line.strip())
        print(f"Read {len(jsonl_texts)} unique lines from '{file_path}'.")
        return jsonl_texts
    except Exception as e:
        print(f"Error reading JSONL file '{file_path}': {e}")
        return None

def get_markdown_file_count(directory_path):
    """Counts the number of markdown (.md) files in the specified directory."""
    if not os.path.exists(directory_path):
        print(f"Error: Markdown source directory '{directory_path}' not found.")
        return None
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        return None
    
    md_file_count = 0
    try:
        for item in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, item)) and item.lower().endswith(".md"):
                md_file_count += 1
        print(f"Found {md_file_count} markdown (.md) files in '{directory_path}'.")
        return md_file_count
    except Exception as e:
        print(f"Error counting markdown files in '{directory_path}': {e}")
        return None

def verify_data():
    print("--- Starting Data Verification ---")

    # 0. Get count of original markdown files
    print(f"\n--- Verifying Markdown Source Files: {MARKDOWN_SOURCE_DIR} ---")
    markdown_files_count = get_markdown_file_count(MARKDOWN_SOURCE_DIR)
    if markdown_files_count is None:
        print("Could not determine the count of markdown source files. Continuing...")
        # You might choose to return here if this count is critical

    # 1. Get data from JSONL file
    print(f"\n--- Verifying JSONL File: {JSONL_FILE_PATH} ---")
    source_texts = get_jsonl_data(JSONL_FILE_PATH)
    if source_texts is None:
        print("Cannot proceed without source JSONL data.")
        return

    # 2. Get data from Qdrant
    print(f"\n--- Verifying Qdrant Collection: {COLLECTION_NAME} at {QDRANT_PATH} ---")
    try:
        qdrant_client = QdrantClient(path=QDRANT_PATH)
    except Exception as e:
        print(f"Failed to initialize Qdrant client for path '{QDRANT_PATH}': {e}")
        print("Please ensure Qdrant is accessible or the path is correct.")
        return

    stored_texts, _, qdrant_total_points = get_qdrant_data(qdrant_client, COLLECTION_NAME)

    if stored_texts is None: # An error occurred connecting or collection not found
        print("Cannot proceed with Qdrant data verification due to previous errors.")
        return

    if qdrant_total_points == 0 and len(source_texts) > 0:
        print("\nWARNING: Qdrant collection is empty, but the source JSONL file has data.")
        print("This likely means the data ingestion into Qdrant failed or was not performed for this collection.")
        print("Please check the output of your `vector_retriever.process()` step for errors (like 'Failed to partition' or 'No elements extracted').")
        # return # Optional: stop here if Qdrant is empty

    # 3. Compare the data
    print("\n--- Comparison Results ---")
    if markdown_files_count is not None:
        print(f"Number of source Markdown (.md) files: {markdown_files_count}")
    print(f"Number of unique entries in JSONL file: {len(source_texts)}")
    print(f"Number of unique text payloads in Qdrant: {len(stored_texts)}")
    print(f"Total points reported by Qdrant collection info: {qdrant_total_points}")

    if len(source_texts) != qdrant_total_points and qdrant_total_points > 0 :
         print("Note: The number of JSONL entries and Qdrant points differ.")
         print("If CAMEL's VectorRetriever did not use additional text splitting (i.e., no `text_splitter_kwargs`),")
         print("each line in your JSONL should ideally correspond to one point in Qdrant.")
         print("Differences could also arise if Qdrant stores non-unique text payloads as distinct points with different IDs,")
         print("or if the JSONL file contains duplicate lines (which `set` would de-duplicate for `source_texts`).")


    missing_in_qdrant = source_texts - stored_texts
    if missing_in_qdrant:
        print(f"\nFound {len(missing_in_qdrant)} entries from JSONL that are MISSING in Qdrant's text payloads:")
        # Print a few examples
        for i, text in enumerate(list(missing_in_qdrant)[:5]):
            print(f"  Example {i+1} (JSONL): {text[:100]}...") # Print first 100 chars
    else:
        if len(source_texts) > 0 and qdrant_total_points > 0 and len(source_texts) == len(stored_texts) :
             print("\nAll unique entries from JSONL appear to be present in Qdrant's text payloads.")

    extra_in_qdrant = stored_texts - source_texts
    if extra_in_qdrant:
        print(f"\nFound {len(extra_in_qdrant)} text payloads in Qdrant that are NOT in the JSONL file:")
        # Print a few examples
        for i, text in enumerate(list(extra_in_qdrant)[:5]):
            print(f"  Example {i+1} (Qdrant): {text[:100]}...") # Print first 100 chars
        print("These could be from previous ingestions or other sources if the collection was not cleared.")
    else:
        if len(source_texts) > 0 and qdrant_total_points > 0 and len(source_texts) == len(stored_texts):
            print("No unexpected extra text payloads found in Qdrant (based on comparison with current JSONL).")
            
    if len(source_texts) == 0 and qdrant_total_points == 0:
        print("\nBoth the source JSONL file and the Qdrant collection appear to be empty.")

    print("\n--- Verification Complete ---")

# Run the verification
if __name__ == "__main__":
    verify_data()