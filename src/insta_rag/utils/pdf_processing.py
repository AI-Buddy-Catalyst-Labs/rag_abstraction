"""PDF processing utilities."""

from pathlib import Path
from typing import Union

from .exceptions import PDFCorruptedError, PDFEmptyError, PDFEncryptedError


def extract_text_from_pdf(
    pdf_path: Union[str, Path], parser: str = "pdfplumber"
) -> str:
    """Extract text from PDF file.

    Args:
        pdf_path: Path to PDF file
        parser: Parser to use (pdfplumber, pypdf2)

    Returns:
        Extracted text

    Raises:
        PDFEncryptedError: If PDF is password-protected
        PDFCorruptedError: If PDF is corrupted
        PDFEmptyError: If PDF has no extractable text
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise PDFCorruptedError(f"PDF file not found: {pdf_path}")

    # Try primary parser
    if parser == "pdfplumber":
        text = _extract_with_pdfplumber(pdf_path)
    elif parser == "pypdf2":
        text = _extract_with_pypdf2(pdf_path)
    else:
        # Default to pdfplumber
        text = _extract_with_pdfplumber(pdf_path)

    # If primary parser fails or returns empty, try fallback
    if not text or len(text.strip()) < 10:
        print(f"Primary parser '{parser}' returned empty text, trying fallback...")
        if parser != "pypdf2":
            text = _extract_with_pypdf2(pdf_path)

    # Final validation
    if not text or len(text.strip()) < 10:
        raise PDFEmptyError(f"No extractable text found in PDF: {pdf_path}")

    return text


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    try:
        import pdfplumber

        text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            # Check if PDF is encrypted
            if pdf.metadata.get("Encrypt"):
                raise PDFEncryptedError(f"PDF is password-protected: {pdf_path}")

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)

    except PDFEncryptedError:
        raise
    except ImportError as e:
        raise ImportError(
            "pdfplumber not installed. Install with: pip install pdfplumber"
        ) from e
    except Exception as e:
        raise PDFCorruptedError(
            f"Failed to extract text with pdfplumber: {str(e)}"
        ) from e


def _extract_with_pypdf2(pdf_path: Path) -> str:
    """Extract text using PyPDF2.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))

        # Check if PDF is encrypted
        if reader.is_encrypted:
            raise PDFEncryptedError(f"PDF is password-protected: {pdf_path}")

        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n\n".join(text_parts)

    except PDFEncryptedError:
        raise
    except ImportError as e:
        raise ImportError(
            "PyPDF2 not installed. Install with: pip install PyPDF2"
        ) from e
    except Exception as e:
        raise PDFCorruptedError(f"Failed to extract text with PyPDF2: {str(e)}") from e


def validate_pdf(pdf_path: Union[str, Path]) -> bool:
    """Validate PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if PDF is valid and readable
    """
    try:
        extract_text_from_pdf(pdf_path)
        return True
    except (PDFEncryptedError, PDFCorruptedError, PDFEmptyError):
        return False
