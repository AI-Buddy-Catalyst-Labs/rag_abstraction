#!/usr/bin/env python3
"""
Script to clear all data from insta_rag_test_collection in Qdrant.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import urllib.parse

# Load environment variables
load_dotenv()

# Get Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "insta_rag_test_collection"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY not found in environment variables")

# Parse URL
parsed = urllib.parse.urlparse(QDRANT_URL)
host = parsed.hostname or parsed.netloc
port = parsed.port or (443 if QDRANT_URL.startswith("https://") else 6333)
https = QDRANT_URL.startswith("https://")

print(f"Connecting to Qdrant at {host}:{port}...")

# Initialize Qdrant client
client = QdrantClient(
    host=host,
    port=port,
    api_key=QDRANT_API_KEY,
    timeout=60,
    prefer_grpc=False,
    https=https,
)

try:
    # Get collection info before clearing
    print(f"\nChecking collection '{COLLECTION_NAME}'...")
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    vectors_count = collection_info.points_count

    print(f"Collection found: {COLLECTION_NAME}")
    print(f"Current vectors count: {vectors_count}")

    if vectors_count == 0:
        print("Collection is already empty!")
    else:
        # Delete all points from collection by deleting and recreating it
        print(f"\nClearing all data from '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"✓ Collection '{COLLECTION_NAME}' deleted successfully")

        # Recreate collection with same settings
        from qdrant_client.models import VectorParams, Distance

        vector_size = collection_info.config.params.vectors.size
        distance = collection_info.config.params.vectors.distance

        print(f"\nRecreating collection with {vector_size} dimensions and {distance} distance metric...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"✓ Collection '{COLLECTION_NAME}' recreated successfully")

        # Verify
        new_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"\n✓ Collection is now empty with {new_info.points_count} vectors")
        print("✓ All data has been cleared!")

except Exception as e:
    print(f"✗ Error: {str(e)}")
    raise
