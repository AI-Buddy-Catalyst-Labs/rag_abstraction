#!/usr/bin/env python
"""Diagnose Qdrant connection issues."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def test_basic_connectivity():
    """Test basic HTTP connectivity to Qdrant."""
    print("=" * 60)
    print("Test 1: Basic HTTP Connectivity")
    print("=" * 60)

    import socket
    from urllib.parse import urlparse

    qdrant_url = os.getenv("QDRANT_URL")
    print(f"Testing: {qdrant_url}")

    # Parse URL
    parsed = urlparse(qdrant_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    print(f"Host: {host}")
    print(f"Port: {port}")

    # Test DNS resolution
    try:
        ip = socket.gethostbyname(host)
        print(f"✓ DNS resolved: {ip}")
    except Exception as e:
        print(f"✗ DNS resolution failed: {e}")
        return False

    # Test port connectivity
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            print(f"✓ Port {port} is open and accessible")
        else:
            print(f"✗ Port {port} is not accessible (error code: {result})")
            return False
    except Exception as e:
        print(f"✗ Port test failed: {e}")
        return False

    return True


def test_http_request():
    """Test HTTP request to Qdrant."""
    print("\n" + "=" * 60)
    print("Test 2: HTTP Request")
    print("=" * 60)

    try:
        import requests

        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        # Test basic endpoint
        url = f"{qdrant_url.rstrip('/')}/collections"
        headers = {"api-key": api_key}

        print(f"GET {url}")
        print(f"Timeout: 30 seconds")

        response = requests.get(url, headers=headers, timeout=30, verify=True)

        if response.status_code == 200:
            print(f"✓ HTTP request successful (status: {response.status_code})")
            data = response.json()
            print(f"✓ Collections found: {len(data.get('result', {}).get('collections', []))}")
            return True
        else:
            print(f"✗ HTTP request failed (status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print("✗ HTTP request timed out (30 seconds)")
        print("\nPossible causes:")
        print("  - Qdrant server is slow or overloaded")
        print("  - Network latency is high")
        print("  - Firewall blocking the connection")
        return False
    except requests.exceptions.SSLError as e:
        print(f"✗ SSL/TLS error: {e}")
        print("\nTrying without SSL verification...")
        return test_http_request_no_verify()
    except Exception as e:
        print(f"✗ HTTP request failed: {e}")
        return False


def test_http_request_no_verify():
    """Test HTTP request without SSL verification."""
    try:
        import requests
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        url = f"{qdrant_url.rstrip('/')}/collections"
        headers = {"api-key": api_key}

        response = requests.get(url, headers=headers, timeout=30, verify=False)

        if response.status_code == 200:
            print("✓ HTTP request successful (SSL verification disabled)")
            print("⚠ Note: SSL verification is disabled - use with caution")
            return True
        else:
            print(f"✗ HTTP request failed even without SSL verification")
            return False

    except Exception as e:
        print(f"✗ Still failed: {e}")
        return False


def test_qdrant_client_http():
    """Test Qdrant client with HTTP."""
    print("\n" + "=" * 60)
    print("Test 3: Qdrant Client (HTTP)")
    print("=" * 60)

    try:
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        print("Initializing QdrantClient with HTTP...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=api_key,
            prefer_grpc=False,
            timeout=60,
        )

        print("Getting collections...")
        collections = client.get_collections()

        print(f"✓ Qdrant client successful")
        print(f"✓ Collections: {len(collections.collections)}")
        for col in collections.collections:
            print(f"  - {col.name}")
        return True

    except Exception as e:
        print(f"✗ Qdrant client failed: {e}")
        print("\nTrying alternative connection method...")
        return test_qdrant_client_alternative()


def test_qdrant_client_alternative():
    """Test Qdrant client with alternative settings."""
    try:
        from qdrant_client import QdrantClient

        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        # Remove trailing slash if present
        qdrant_url = qdrant_url.rstrip("/")

        print(f"\nAlternative connection settings:")
        print(f"  - URL: {qdrant_url}")
        print(f"  - gRPC: False")
        print(f"  - Timeout: 120 seconds")
        print(f"  - HTTPS: True")

        client = QdrantClient(
            url=qdrant_url,
            api_key=api_key,
            prefer_grpc=False,
            timeout=120,
            https=True,
        )

        collections = client.get_collections()
        print(f"✓ Alternative connection successful!")
        print(f"✓ Collections: {len(collections.collections)}")
        return True

    except Exception as e:
        print(f"✗ Alternative connection also failed: {e}")
        return False


def test_local_qdrant():
    """Test if local Qdrant is available."""
    print("\n" + "=" * 60)
    print("Test 4: Check for Local Qdrant")
    print("=" * 60)

    try:
        from qdrant_client import QdrantClient

        # Try local Qdrant
        print("Checking for local Qdrant instance...")
        client = QdrantClient(host="localhost", port=6333, timeout=5)

        collections = client.get_collections()
        print("✓ Local Qdrant instance found!")
        print(f"✓ Collections: {len(collections.collections)}")
        print("\n💡 You can use local Qdrant for testing")
        return True

    except Exception as e:
        print("✗ No local Qdrant instance found")
        print("💡 To run local Qdrant:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        return False


def provide_solutions():
    """Provide solutions based on test results."""
    print("\n" + "=" * 60)
    print("Solutions")
    print("=" * 60)

    print("\n1. Use Local Qdrant (Recommended for Testing)")
    print("-" * 60)
    print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
    print("\n   Then update .env:")
    print("   QDRANT_URL=http://localhost:6333")
    print("   QDRANT_API_KEY=  (leave empty for local)")

    print("\n2. Check Network/Firewall")
    print("-" * 60)
    print("   - Verify the Qdrant server is running")
    print("   - Check if your network/firewall allows HTTPS connections")
    print("   - Try from a different network")

    print("\n3. Verify Qdrant URL")
    print("-" * 60)
    print("   Current URL:", os.getenv("QDRANT_URL"))
    print("   - Ensure URL is correct and accessible")
    print("   - Try accessing in browser")

    print("\n4. Use Qdrant Cloud")
    print("-" * 60)
    print("   - Sign up at https://cloud.qdrant.io")
    print("   - Create a free cluster")
    print("   - Update QDRANT_URL and QDRANT_API_KEY in .env")

    print("\n5. Increase Timeout (Temporary Fix)")
    print("-" * 60)
    print("   In src/insta_rag/vectordb/qdrant.py:")
    print("   Change timeout=60 to timeout=300 (5 minutes)")


def main():
    """Run all diagnostics."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Qdrant Connection Diagnostics" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    results = []

    results.append(("Basic Connectivity", test_basic_connectivity()))
    results.append(("HTTP Request", test_http_request()))
    results.append(("Qdrant Client", test_qdrant_client_http()))
    results.append(("Local Qdrant", test_local_qdrant()))

    # Summary
    passed = sum(1 for _, result in results if result)

    if passed == 0:
        print("\n" + "=" * 60)
        print("⚠ All connection tests failed")
        print("=" * 60)
        provide_solutions()
    elif any(name == "Qdrant Client" and result for name, result in results):
        print("\n" + "=" * 60)
        print("✓ Connection successful!")
        print("=" * 60)
        print("\nYour Qdrant connection is working. The testing API should work now.")
    else:
        print("\n" + "=" * 60)
        print("⚠ Partial success")
        print("=" * 60)
        print("\nSome tests passed but Qdrant client failed.")
        provide_solutions()


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    main()
