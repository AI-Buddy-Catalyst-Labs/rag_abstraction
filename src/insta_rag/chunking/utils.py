"""Utility functions for chunking operations."""

import re
from typing import List


def count_tokens(text: str) -> int:
    """Count tokens in text using simple approximation.

    This is a fast approximation. For production use with specific models,
    consider using tiktoken or the model's tokenizer.

    Args:
        text: Input text

    Returns:
        Approximate token count
    """
    # Simple approximation: split on whitespace and punctuation
    # Average token/word ratio is ~1.3 for English
    words = len(text.split())
    return int(words * 1.3)


def count_tokens_accurate(text: str, model: str = "gpt-4") -> int:
    """Count tokens accurately using tiktoken.

    Args:
        text: Input text
        model: Model name for tokenizer

    Returns:
        Exact token count
    """
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to approximate counting if tiktoken not available
        return count_tokens(text)
    except Exception:
        # Fallback on any error
        return count_tokens(text)


def split_text_by_tokens(
    text: str, max_tokens: int, overlap_tokens: int = 0
) -> List[str]:
    """Split text into chunks by token count.

    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    if not words:
        return []

    # Approximate words per chunk (tokens ~= words * 1.3)
    words_per_chunk = int(max_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)

    chunks = []
    start_idx = 0

    while start_idx < len(words):
        end_idx = min(start_idx + words_per_chunk, len(words))
        chunk_words = words[start_idx:end_idx]
        chunks.append(" ".join(chunk_words))

        if end_idx >= len(words):
            break

        # Move start position for next chunk (with overlap)
        start_idx = end_idx - overlap_words

    return chunks


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Use regex to split on sentence boundaries
    # This handles common cases but may not be perfect for all texts
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = sentence_endings.split(text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text: Input text

    Returns:
        List of paragraphs
    """
    # Split on double newlines
    paragraphs = re.split(r"\n\s*\n", text)

    # Clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def validate_chunk_quality(chunk: str) -> bool:
    """Validate chunk quality.

    Checks for:
    - Minimum length
    - Not mostly garbled text
    - Not mostly special characters

    Args:
        chunk: Text chunk to validate

    Returns:
        True if chunk passes quality checks
    """
    # Check minimum length
    if len(chunk) < 10:
        return False

    # Check for excessive special characters (possible garbled text)
    special_char_ratio = sum(
        1 for c in chunk if not c.isalnum() and not c.isspace()
    ) / len(chunk)

    if special_char_ratio > 0.5:  # More than 50% special characters
        return False

    # Check for reasonable alphanumeric content
    alphanumeric_ratio = sum(1 for c in chunk if c.isalnum()) / len(chunk)

    if alphanumeric_ratio < 0.3:  # Less than 30% alphanumeric
        return False

    return True


def add_overlap_to_chunks(
    chunks: List[str], overlap_percentage: float = 0.2
) -> List[str]:
    """Add overlap between consecutive chunks.

    Args:
        chunks: List of text chunks
        overlap_percentage: Percentage of chunk to overlap (0-1)

    Returns:
        List of chunks with overlap
    """
    if len(chunks) <= 1:
        return chunks

    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk stays as is
            overlapped_chunks.append(chunk)
        else:
            # Add overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_words = int(len(prev_chunk.split()) * overlap_percentage)

            if overlap_words > 0:
                prev_words = prev_chunk.split()
                overlap_text = " ".join(prev_words[-overlap_words:])
                overlapped_chunks.append(f"{overlap_text} {chunk}")
            else:
                overlapped_chunks.append(chunk)

    return overlapped_chunks


def merge_small_chunks(chunks: List[str], min_chunk_size: int = 100) -> List[str]:
    """Merge chunks that are smaller than minimum size.

    Args:
        chunks: List of text chunks
        min_chunk_size: Minimum chunk size in tokens

    Returns:
        List of chunks with no chunks smaller than min_size
    """
    if not chunks:
        return chunks

    merged_chunks = []
    current_chunk = chunks[0]

    for i in range(1, len(chunks)):
        current_tokens = count_tokens(current_chunk)

        if current_tokens < min_chunk_size:
            # Merge with next chunk
            current_chunk = f"{current_chunk} {chunks[i]}"
        else:
            # Current chunk is large enough, save it
            merged_chunks.append(current_chunk)
            current_chunk = chunks[i]

    # Add the last chunk
    merged_chunks.append(current_chunk)

    return merged_chunks
