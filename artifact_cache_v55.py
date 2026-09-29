"""Cross-run artifact and precomputation cache with atomic file replacement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


class ArtifactCache:
    """Quản lý lưu trữ cache đĩa cho các bước tính toán lặp lại qua nhiều lần chạy."""

    def __init__(self, cache_dir: Path | str = ".cache_v55"):
        """Thực thi __init__."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, key: str) -> Path:
        """Thực thi _entry_path."""
        return self.cache_dir / f"{key}.json"

    def get(self, key: str, expected_fingerprint: str) -> dict[str, Any] | None:
        """Đọc entry nếu tồn tại, có dấu COMPLETE và khớp fingerprint."""
        path = self._entry_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("COMPLETE", False):
                return None
            if data.get("fingerprint") != expected_fingerprint:
                return None
            return data.get("payload")
        except Exception:
            return None

    def put(self, key: str, fingerprint: str, payload: dict[str, Any]) -> None:
        """Ghi entry nguyên tử thông qua file tạm."""
        path = self._entry_path(key)
        temp_path = self.cache_dir / f"{key}.tmp.{os.getpid()}"
        data = {
            "COMPLETE": True,
            "fingerprint": fingerprint,
            "payload": payload,
        }
        try:
            temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(path)
        except Exception:
            if temp_path.is_file():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
