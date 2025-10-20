"""Check your Azure OpenAI deployments to find the correct deployment name."""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("Azure OpenAI Configuration Check")
print("=" * 80)

# Check current config
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT")

print("\nCurrent Configuration:")
print(f"  Endpoint: {endpoint}")
print(f"  API Key: {'***' + api_key[-8:] if api_key else 'NOT SET'}")
print(f"  Embedding Deployment: {embedding_deployment}")
print(f"  LLM Deployment: {llm_deployment}")

print("\n" + "=" * 80)
print("Testing Deployments...")
print("=" * 80)

try:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=api_key, api_version="2024-02-01", azure_endpoint=endpoint
    )

    # Test embedding deployment
    print(f"\n1. Testing Embedding Deployment: '{embedding_deployment}'")
    try:
        response = client.embeddings.create(model=embedding_deployment, input="test")
        print("   ✅ SUCCESS - Embedding deployment works!")
        print(f"   Dimensions: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"   ❌ FAILED - {str(e)}")

    # Test LLM deployment
    print(f"\n2. Testing LLM Deployment: '{llm_deployment}'")
    try:
        response = client.chat.completions.create(
            model=llm_deployment,
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=10,
        )
        print("   ✅ SUCCESS - LLM deployment works!")
        print(f"   Response: {response.choices[0].message.content}")
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ FAILED - {error_msg}")

        if "DeploymentNotFound" in error_msg or "404" in error_msg:
            print(
                f"\n   ⚠️  The deployment '{llm_deployment}' does NOT exist in your Azure resource!"
            )
            print("\n   🔍 To find your actual deployment name:")
            print("      1. Go to: https://oai.azure.com/")
            print("      2. Click 'Deployments' in the left sidebar")
            print("      3. Look for GPT-4 or GPT-3.5 model deployments")
            print("      4. Copy the exact 'Deployment name' column value")
            print("      5. Update AZURE_LLM_DEPLOYMENT in .env with that name")
            print("\n   Common deployment names:")
            print("      - gpt-4")
            print("      - gpt-4-turbo")
            print("      - gpt-35-turbo")
            print("      - gpt-4-32k")

except ImportError:
    print("\n❌ OpenAI library not installed. Install with: pip install openai")
except Exception as e:
    print(f"\n❌ Configuration error: {e}")

print("\n" + "=" * 80)
print("Recommendations:")
print("=" * 80)

print("\n1. For HyDE to work:")
print("   - Find your correct LLM deployment name in Azure Portal")
print("   - Update AZURE_LLM_DEPLOYMENT in .env")

print("\n2. For now (without HyDE):")
print("   - HyDE automatically falls back to original query")
print("   - Vector search still works perfectly!")
print("   - Just set enable_hyde=false in API calls if you prefer")

print("\n3. For retrieval to work:")
print("   - NEVER use score_threshold=1.0 (filters everything!)")
print("   - Use score_threshold=null or 0.15-0.3")
print("   - Typical scores are 0.15-0.40 for semantic search")

print("\n" + "=" * 80)
