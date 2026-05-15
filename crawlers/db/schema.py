"""SQLite schema for the paper and author metadata used by availability datasets."""

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    keywords TEXT,
    pdf_url TEXT NOT NULL,
    conference TEXT,
    venue_year INTEGER NOT NULL,
    venue_track TEXT,
    openreview_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS authors (
    author_id TEXT PRIMARY KEY,
    display_name TEXT,
    email TEXT
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    author_position INTEGER,
    PRIMARY KEY (paper_id, author_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX IF NOT EXISTS idx_paper_authors_paper ON paper_authors(paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_conference ON papers(conference);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(venue_year);
CREATE INDEX IF NOT EXISTS idx_papers_track ON papers(venue_track);
CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(display_name);
"""


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize a SQLite database with the metadata schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open an existing SQLite metadata database."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
