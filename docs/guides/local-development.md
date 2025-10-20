# Guide: Local Development Setup

This guide provides tips for setting up a smooth and efficient local development environment for `insta_rag`, including how to run a local instance of Qdrant to avoid network latency and connection issues.

## Using a Local Qdrant Instance

For development and testing, running a local Qdrant instance via Docker is highly recommended. It's fast, free, and eliminates network-related issues.

### 1. Start Qdrant with Docker

Run the following command in your terminal to start a Qdrant container. This command also mounts a local volume to persist your data between container restarts.

```bash
# This command will download the Qdrant image and run it in the background.
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

*   `-p 6333:6333`: Maps the HTTP REST API port.
*   `-p 6334:6334`: Maps the gRPC port.
*   `-v $(pwd)/qdrant_storage:/qdrant/storage`: Persists data in a `qdrant_storage` directory in your current folder.

### 2. Update Your `.env` File

Modify your `.env` file to point to your local instance. Comment out the remote/cloud Qdrant variables and add the local ones.

```env
# .env file

# Comment out the remote Qdrant configuration
# QDRANT_URL=https://your-remote-qdrant-url.com/
# QDRANT_API_KEY=your-remote-api-key

# Add the local Qdrant configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

Leave `QDRANT_API_KEY` blank, as the default local instance does not require one.

### 3. Verify the Connection

You can quickly check if your local Qdrant is running correctly.

```bash
# Use curl to check the collections endpoint
curl http://localhost:6333/collections
```

You should see a response like: `{"result":{"collections":[]},"status":"ok","time":...}`.

### 4. Access the Qdrant Dashboard

Qdrant provides a web dashboard to view your collections, search points, and monitor the instance. Access it at:

**http://localhost:6333/dashboard**

## Troubleshooting Connection Issues

If you are having trouble connecting to a remote Qdrant server, here are a few steps to debug the issue.

### 1. Test Basic Connectivity

Use `curl` to see if the server is reachable from your network. A timeout here indicates a network or firewall issue, not a problem with the library itself.

```bash
# Test if the server responds to a basic request (10-second timeout)
curl -I --max-time 10 "https://your-remote-qdrant-url.com/"
```

### 2. Increase Client Timeout

For slow remote connections, you can increase the timeout directly in the client configuration. In `src/insta_rag/core/config.py`, you can add a `timeout` parameter to the `VectorDBConfig`.

```python
# src/insta_rag/core/config.py

@dataclass
class VectorDBConfig:
    provider: str = "qdrant"
    url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120 # Increase timeout to 120 seconds
```

### 3. Use a Free Qdrant Cloud Instance

If your self-hosted remote server is unreliable, consider using the free tier from [Qdrant Cloud](https://cloud.qdrant.io). It's a quick and easy way to get a stable remote vector database for development.

1.  Sign up and create a free cluster.
2.  Copy the URL and generate an API key.
3.  Update your `.env` file with the new credentials.

```