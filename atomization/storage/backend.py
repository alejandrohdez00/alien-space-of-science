"""Local filesystem storage for atomization outputs."""

import asyncio
from pathlib import Path


class StorageBackend:
    """Small local-only storage helper.

    Output files follow the paper pipeline layout:
    ``{base_dir}/{paper_id}/blog.md``, ``ideas.json``, and
    ``refined_ideas.json``.
    """

    def __init__(self, base_dir: str = "papers"):
        self.base_dir = base_dir
        self.mode = "local"

    async def save(self, content: str, paper_id: str, file_type: str) -> dict:
        filename = {
            "blog": "blog.md",
            "ideas": "ideas.json",
            "refined_ideas": "refined_ideas.json",
        }.get(file_type)
        if filename is None:
            raise ValueError(f"Unknown file_type: {file_type}")

        path = Path(self.base_dir) / paper_id / filename
        await asyncio.to_thread(self._save_local, content, path)
        return {"local": str(path)}

    @staticmethod
    def _save_local(content: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
