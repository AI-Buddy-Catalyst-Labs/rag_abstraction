"""HyDE (Hypothetical Document Embeddings) query generator."""

import json
from typing import Dict

from insta_rag.utils.exceptions import QueryGenerationError


class HyDEQueryGenerator:
    """
    Generate optimized queries using HyDE (Hypothetical Document Embeddings).

    HyDE improves retrieval by generating a hypothetical answer to the query,
    then using that answer's embedding for search. Research shows 20-30%
    improvement in retrieval quality.

    Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """

    def __init__(self, llm_config):
        """
        Initialize HyDE query generator.

        Args:
            llm_config: LLMConfig with Azure OpenAI settings
        """
        self.llm_config = llm_config
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            if self.llm_config.provider == "azure_openai":
                from openai import AzureOpenAI

                self.client = AzureOpenAI(
                    api_key=self.llm_config.api_key,
                    api_version=self.llm_config.api_version or "2024-02-01",
                    azure_endpoint=self.llm_config.api_base,
                )
            else:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.llm_config.api_key)

        except ImportError as e:
            raise QueryGenerationError(
                "OpenAI library not installed. Install with: pip install openai"
            ) from e
        except Exception as e:
            raise QueryGenerationError(
                f"Failed to initialize LLM client: {str(e)}"
            ) from e

    def generate_queries(self, query: str) -> Dict[str, str]:
        """
        Generate optimized standard query and HyDE query.

        This method makes a single LLM call to generate:
        1. Optimized standard query (cleaned, expanded)
        2. HyDE query (hypothetical answer to the question)

        Args:
            query: Original user query

        Returns:
            Dictionary with:
            - "standard": Optimized query for vector search
            - "hyde": Hypothetical document/answer for better retrieval

        Example:
            >>> generator = HyDEQueryGenerator(llm_config)
            >>> result = generator.generate_queries("What is semantic chunking?")
            >>> result["standard"]
            "semantic chunking text splitting method"
            >>> result["hyde"]
            "Semantic chunking is a method of dividing text into meaningful
             segments based on semantic similarity..."
        """
        try:
            # Prepare the prompt for query generation
            system_prompt = """You are a search query optimization assistant.
Your task is to generate two types of queries to improve document retrieval:

1. STANDARD QUERY: An optimized version of the user's query
   - Remove stop words
   - Expand abbreviations
   - Add relevant synonyms
   - Keep it concise (5-10 words)

2. HYDE QUERY: A hypothetical document that would answer the query
   - Write 2-3 sentences as if you're answering the question
   - Use technical terms and domain-specific language
   - Be specific and detailed
   - This will be used for semantic search

Return your response as JSON with keys "standard" and "hyde"."""

            user_prompt = f"""Original query: {query}

Generate an optimized standard query and a HyDE (hypothetical answer) query."""

            # Call LLM with JSON mode for structured output
            if self.llm_config.provider == "azure_openai":
                response = self.client.chat.completions.create(
                    model=self.llm_config.deployment_name or self.llm_config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.llm_config.temperature,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.llm_config.temperature,
                )

            # Parse JSON response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Validate and return
            if "standard" in result and "hyde" in result:
                return {
                    "standard": result["standard"],
                    "hyde": result["hyde"],
                }
            else:
                # Fallback if JSON structure is wrong
                raise QueryGenerationError(
                    f"Invalid response structure from LLM: {result}"
                )

        except json.JSONDecodeError as e:
            # Fallback to original query if JSON parsing fails
            print(f"Warning: Failed to parse LLM response: {e}")
            return {"standard": query, "hyde": query}

        except Exception as e:
            # Fallback to original query on any error
            print(f"Warning: Query generation failed: {e}")
            return {"standard": query, "hyde": query}
