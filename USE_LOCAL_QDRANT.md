# Using Local Qdrant for Testing

If you're experiencing connection issues with remote Qdrant, you can easily run a local instance for testing.

## Quick Start with Docker

### 1. Run Qdrant Container

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Without Docker volume (data will be lost on restart):**
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Update .env File

```env
# Comment out or replace remote Qdrant
# QDRANT_URL=https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/
# QDRANT_API_KEY=edfBd7pP251ev2uiRcjcBGt7QXJe1P70

# Use local Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### 3. Test Connection

```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Should return: {"result":{"collections":[]}}
```

### 4. Run Your Tests

```bash
cd testing_api
./run.sh
```

Now all tests will use your local Qdrant instance!

## Alternative: Use Docker Compose

### 1. Create docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
```

### 2. Start Services

```bash
docker-compose up -d
```

### 3. Stop Services

```bash
docker-compose down
```

## Verify Local Qdrant is Working

```bash
# Test with curl
curl http://localhost:6333/collections

# Test with Python
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://localhost:6333', prefer_grpc=False)
print('Collections:', client.get_collections())
print('✓ Local Qdrant working!')
"
```

## Access Qdrant Dashboard

Open in browser: **http://localhost:6333/dashboard**

You can:
- View collections
- Browse points
- Run queries
- Monitor performance

## Switching Back to Remote Qdrant

When you want to use remote Qdrant again:

1. Stop local Qdrant:
```bash
docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)
```

2. Restore .env:
```env
QDRANT_URL=https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/
QDRANT_API_KEY=edfBd7pP251ev2uiRcjcBGt7QXJe1P70
```

## Benefits of Local Qdrant

- ✅ No network latency
- ✅ No connection timeouts
- ✅ Free and unlimited
- ✅ Full control
- ✅ Works offline
- ✅ Perfect for development and testing

## Troubleshooting Local Qdrant

### Port already in use

```bash
# Find what's using the port
lsof -i :6333

# Kill the process or use different ports
docker run -d -p 7333:6333 qdrant/qdrant
# Then use QDRANT_URL=http://localhost:7333
```

### Docker not installed

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in
```

**macOS:**
```bash
brew install docker
```

### Permission denied

```bash
sudo docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or add user to docker group (Ubuntu):
```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Production Considerations

For production, consider:
- Using persistent volumes
- Setting up authentication
- Configuring backups
- Using Qdrant Cloud for managed service

## Summary

Local Qdrant is perfect for:
- Development and testing
- Learning and experimentation
- When remote Qdrant has connection issues
- Offline development

It's **not recommended** for:
- Production deployments
- Shared/team environments
- When you need remote access
