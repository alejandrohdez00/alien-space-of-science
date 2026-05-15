"""Core pipeline utilities.

This module provides fundamental utilities for the atomization pipeline:
paper ID generation, input file loading, resume capability, and formatting.
"""

import sys
import json
from pathlib import Path
from typing import List


def load_papers_from_file(file_path: str) -> List[tuple]:
    """
    Load papers from a TSV file (paper_id<TAB>pdf_url per line).

    Returns:
        List of (paper_id, pdf_url) tuples
    """
    try:
        papers = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    print(f"Warning: Invalid line format (expected TSV): {line[:50]}...")
                    continue
                paper_id = parts[0].replace("/", "__")
                papers.append((paper_id, parts[1]))
        return papers
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def get_paper_completion_status(paper_id: str, base_path: str = "papers") -> dict:
    """
    Check which output files exist for a paper.

    Args:
        paper_id: The paper hash identifier
        base_path: Base directory for papers output

    Returns:
        dict with keys 'blog', 'ideas', 'refined_ideas' - each True/False
    """
    paper_dir = Path(base_path) / paper_id

    return {
        "blog": (paper_dir / "blog.md").exists(),
        "ideas": (paper_dir / "ideas.json").exists(),
        "refined_ideas": (paper_dir / "refined_ideas.json").exists(),
    }


def load_existing_output(paper_id: str, output_type: str, base_path: str = "papers"):
    """
    Load existing output file content for a paper.

    Args:
        paper_id: The paper hash identifier
        output_type: One of 'blog', 'ideas', 'refined_ideas'
        base_path: Base directory for papers output

    Returns:
        str for blog, dict for ideas/refined_ideas, or None if file not found
    """
    filenames = {
        "blog": "blog.md",
        "ideas": "ideas.json",
        "refined_ideas": "refined_ideas.json"
    }

    if output_type not in filenames:
        raise ValueError(f"output_type must be one of {list(filenames.keys())}")

    filepath = Path(base_path) / paper_id / filenames[output_type]

    if not filepath.exists():
        return None

    content = filepath.read_text()

    if output_type == "blog":
        return content
    else:
        return json.loads(content)
