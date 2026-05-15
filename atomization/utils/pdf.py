"""PDF document processing utilities.

This module provides utilities for intelligent PDF truncation, section detection,
and encoding for LLM processing. Used primarily by the blog generation stage.
"""

import io
import base64
import re
from typing import List
from pypdf import PdfReader, PdfWriter

# Section detection patterns (case-insensitive, must appear at/near start of line)
REFERENCES_PATTERN = re.compile(
    r'^\s*(?:\d+\.?\s+)?(?:References|Bibliography)\s*$',
    re.IGNORECASE | re.MULTILINE
)
APPENDIX_PATTERN = re.compile(
    r'^\s*(?:[A-Z]\.?\s+)?(?:Appendix|Supplementary(?:\s+Materials?)?)\s*$',
    re.IGNORECASE | re.MULTILINE
)


def _find_section_page(reader, pattern) -> int | None:
    """
    Find the first page where a section pattern matches.

    Args:
        reader: PdfReader instance
        pattern: Compiled regex pattern

    Returns:
        0-based page index where pattern first matches, or None
    """
    for page_idx in range(len(reader.pages)):
        try:
            text = reader.pages[page_idx].extract_text()
            if text and pattern.search(text):
                return page_idx
        except Exception:
            # Page text extraction failed, skip this page
            continue
    return None


def truncate_pdf_bytes(pdf_bytes: bytes, max_pages: int) -> tuple[bytes, dict]:
    """
    Truncate a PDF intelligently by detecting References/Appendix sections.

    Args:
        pdf_bytes: Raw PDF file bytes
        max_pages: Maximum number of pages to keep (fallback limit)

    Returns:
        Tuple of (truncated_bytes, metadata_dict) where metadata contains:
        - original_pages: Total page count before truncation
        - kept_pages: Actual pages in result
        - truncation_reason: "references" | "appendix" | "max_pages" | "none"
        - references_page: Page number where References found (1-indexed, or None)
        - appendix_page: Page number where Appendix found (1-indexed, or None)

    Raises:
        Exception: If PDF processing fails (fail-fast approach)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    # Search for section markers
    references_page = _find_section_page(reader, REFERENCES_PATTERN)
    appendix_page = _find_section_page(reader, APPENDIX_PATTERN)

    # Determine cutoff (includes the page where section starts for two-column layouts)
    if references_page is not None:
        cutoff = min(references_page + 1, max_pages, total_pages)
        reason = "references"
    elif appendix_page is not None:
        cutoff = min(appendix_page + 1, max_pages, total_pages)
        reason = "appendix"
    else:
        cutoff = min(max_pages, total_pages)
        reason = "max_pages" if max_pages < total_pages else "none"

    # Build metadata
    metadata = {
        "original_pages": total_pages,
        "kept_pages": cutoff,
        "truncation_reason": reason,
        "references_page": references_page + 1 if references_page is not None else None,
        "appendix_page": appendix_page + 1 if appendix_page is not None else None
    }

    # No truncation needed
    if cutoff >= total_pages:
        return pdf_bytes, metadata

    # Create truncated PDF
    writer = PdfWriter()
    for page_num in range(cutoff):
        writer.add_page(reader.pages[page_num])

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue(), metadata


def pdf_bytes_to_base64_data_uri(pdf_bytes: bytes) -> str:
    """
    Convert PDF bytes to a base64 data URI for LiteLLM.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Base64-encoded data URI (e.g., "data:application/pdf;base64,...")
    """
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def compute_truncation_stats(truncation_data: List[dict]) -> dict:
    """
    Compute aggregate statistics from truncation metadata.

    Args:
        truncation_data: List of truncation metadata dicts from blog generation results

    Returns:
        Dictionary with aggregate statistics:
        - total_original_pages: Total pages before truncation
        - total_kept_pages: Total pages after truncation
        - total_removed_pages: Total pages removed
        - removal_percentage: Percentage of pages removed
        - avg_original_pages: Average original pages per paper
        - avg_kept_pages: Average kept pages per paper
        - reasons: Dict mapping truncation reason to count
    """
    if not truncation_data:
        return {
            'total_original_pages': 0,
            'total_kept_pages': 0,
            'total_removed_pages': 0,
            'removal_percentage': 0.0,
            'avg_original_pages': 0.0,
            'avg_kept_pages': 0.0,
            'reasons': {}
        }

    total_original = sum(t['original_pages'] for t in truncation_data)
    total_kept = sum(t['kept_pages'] for t in truncation_data)
    total_removed = total_original - total_kept

    # Count truncation reasons
    reasons = {}
    for t in truncation_data:
        reason = t['truncation_reason']
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        'total_original_pages': total_original,
        'total_kept_pages': total_kept,
        'total_removed_pages': total_removed,
        'removal_percentage': (total_removed / total_original * 100) if total_original > 0 else 0,
        'avg_original_pages': total_original / len(truncation_data),
        'avg_kept_pages': total_kept / len(truncation_data),
        'reasons': reasons
    }
