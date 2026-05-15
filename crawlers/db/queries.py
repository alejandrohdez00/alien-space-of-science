"""SQLite queries used by the dataset builders."""

from typing import Any, Optional

from .schema import get_connection


def get_all_papers(
    db_path: str,
    conference: Optional[str] = None,
    year: Optional[int] = None,
    track: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return papers, optionally filtered by conference, year, or track."""
    conn = get_connection(db_path)
    try:
        conditions = []
        params = []

        if conference:
            conditions.append("conference = ?")
            params.append(conference)
        if year:
            conditions.append("venue_year = ?")
            params.append(year)
        if track:
            conditions.append("venue_track = ?")
            params.append(track)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        cursor = conn.execute(
            f"SELECT * FROM papers WHERE {where_clause} ORDER BY venue_year DESC, title ASC",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
