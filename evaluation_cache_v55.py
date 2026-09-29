"""Evaluation Cache for storing reverse ray traces across identical prescriptions."""

from __future__ import annotations

import collections
import hashlib
import pickle
import threading
from typing import Any, Callable


class ReverseTraceEvaluationCache:
    """Bộ nhớ cache cho các lượt trace reverse cùng hệ quang và bundle tia."""

    def __init__(self, max_entries: int = 2, max_bytes_mib: float = 256.0):
        """Thực thi __init__."""
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1024, int(float(max_bytes_mib) * (1024 ** 2)))
        self._cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()

    def make_key(self, payload: dict[str, Any]) -> str:
        """Tạo khóa SHA-256 an toàn từ nội dung prescription và rays."""
        try:
            serialized = pickle.dumps(payload, protocol=5)
            return hashlib.sha256(serialized).hexdigest()
        except Exception as exc:
            raise ValueError(f"PAYLOAD_NOT_SERIALIZABLE_FOR_CACHE: {exc}") from exc

    def get_or_compute(self, key: str, compute_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Lấy kết quả từ cache hoặc thực hiện tính toán mới."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                raw_bytes = self._cache[key]
                return pickle.loads(raw_bytes)

        # Compute outside lock
        result = compute_fn()

        try:
            payload_bytes = pickle.dumps(result, protocol=5)
        except Exception:
            # If not serializable, return raw result without caching
            return result

        entry_size = len(payload_bytes)
        if entry_size > self.max_bytes:
            # Single entry exceeds total byte budget
            return result

        with self._lock:
            if key in self._cache:
                old_bytes = self._cache.pop(key)
                self._current_bytes -= len(old_bytes)

            while (len(self._cache) >= self.max_entries or
                   (self._current_bytes + entry_size) > self.max_bytes) and self._cache:
                _, evicted_bytes = self._cache.popitem(last=False)
                self._current_bytes -= len(evicted_bytes)

            self._cache[key] = payload_bytes
            self._current_bytes += entry_size

        return result

    def clear(self) -> None:
        """Xóa toàn bộ nội dung trong cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    @property
    def current_bytes(self) -> int:
        """Thực thi current_bytes."""
        with self._lock:
            return self._current_bytes

    @property
    def entry_count(self) -> int:
        """Thực thi entry_count."""
        with self._lock:
            return len(self._cache)
