# Quick Fix for Qdrant Connection Timeout

## The Problem

Your Qdrant server at `qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com` is timing out.

## Solution 1: Use Local Qdrant (Fastest, Recommended for Testing)

### Start Local Qdrant with Docker

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Update .env

Edit `/home/macorov/Documents/GitHub/insta_rag/.env`:

```env
# Comment out remote Qdrant (add # at start)
# QDRANT_URL=https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/
# QDRANT_API_KEY=edfBd7pP251ev2uiRcjcBGt7QXJe1P70

# Add local Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### Test It

```bash
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]}}
```

### Run Your API

```bash
cd /home/macorov/Documents/GitHub/insta_rag/testing_api
./run.sh
```

**Done!** Your tests will now use local Qdrant with zero latency.

______________________________________________________________________

## Solution 2: Fix Remote Qdrant Connection

If you need to use the remote Qdrant server:

### Step 1: Test Basic Connectivity

```bash
# Test if server is reachable
curl -I --max-time 10 "https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/"
```

**If this times out:** The server is not accessible from your network.

- Check if server is running
- Check firewall settings
- Try from different network

### Step 2: Increase Timeout Significantly

Edit `/home/macorov/Documents/GitHub/insta_rag/src/insta_rag/vectordb/qdrant.py`:

```python
def __init__(
    self,
    url: str,
    api_key: str,
    timeout: int = 300,  # Change from 60 to 300 (5 minutes)
    prefer_grpc: bool = False,
):
```

### Step 3: Add Retry Logic

Edit `/home/macorov/Documents/GitHub/insta_rag/src/insta_rag/vectordb/qdrant.py`, find the `_initialize_client` method:

```python
def _initialize_client(self):
    """Initialize Qdrant client."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # Store for later use
        self.Distance = Distance
        self.VectorParams = VectorParams

        # Try multiple times with increasing timeout
        for attempt in range(3):
            try:
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout * (attempt + 1),  # Increasing timeout
                    prefer_grpc=self.prefer_grpc,
                )
                # Test connection
                self.client.get_collections()
                print(f"✓ Qdrant connected (attempt {attempt + 1})")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    import time

                    time.sleep(2)
                else:
                    raise

    except ImportError as e:
        raise VectorDBError(
            "Qdrant client not installed. Install with: pip install qdrant-client"
        ) from e
    except Exception as e:
        raise VectorDBError(f"Failed to initialize Qdrant client: {str(e)}") from e
```

______________________________________________________________________

## Solution 3: Use Qdrant Cloud (Free Tier)

### Get Free Qdrant Cloud Account

1. Go to https://cloud.qdrant.io
1. Sign up (free tier available)
1. Create a cluster
1. Copy the URL and API key

### Update .env

```env
QDRANT_URL=https://your-cluster-xyz.cloud.qdrant.io
QDRANT_API_KEY=your_api_key_from_qdrant_cloud
```

______________________________________________________________________

## Quick Test Commands

### Test if Qdrant is accessible

```bash
# Test remote Qdrant
curl --max-time 10 "https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/collections" \
  -H "api-key: edfBd7pP251ev2uiRcjcBGt7QXJe1P70"

# Test local Qdrant (if running)
curl http://localhost:6333/collections
```

### Test from Python

```python
from qdrant_client import QdrantClient

# Test remote
try:
    client = QdrantClient(
        url="https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/",
        api_key="edfBd7pP251ev2uiRcjcBGt7QXJe1P70",
        prefer_grpc=False,
        timeout=120,
    )
    print("Remote:", client.get_collections())
except Exception as e:
    print("Remote failed:", e)

# Test local
try:
    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
    print("Local:", client.get_collections())
except Exception as e:
    print("Local failed:", e)
```

______________________________________________________________________

## My Recommendation

**Use local Qdrant for testing:**

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Update .env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# 3. Test
curl http://localhost:6333/collections

# 4. Run your app
cd testing_api && ./run.sh
```

**Why?**

- ✅ Zero latency
- ✅ No timeout issues
- ✅ Works offline
- ✅ Perfect for development
- ✅ Free

You can always switch back to remote Qdrant later when connectivity is fixed!
