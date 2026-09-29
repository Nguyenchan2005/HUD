"""Authenticated bounded local IPC for HUD GPU requests."""

from __future__ import annotations

import hashlib
import hmac
import pickle
import socket
import struct
import time
from typing import Any


HEADER = struct.Struct("!Q")
MAC_SIZE = hashlib.sha256().digest_size


def _remaining(deadline: float) -> float:
    """Thực thi _remaining."""
    value = deadline - time.monotonic()
    if value <= 0.0:
        raise TimeoutError("GPU_IPC_DEADLINE_EXCEEDED")
    return value


def _read_exact(
    sock: socket.socket,
    size: int,
    deadline: float,
) -> bytes:
    """Thực thi _read_exact."""
    result = bytearray()
    while len(result) < size:
        sock.settimeout(_remaining(deadline))
        block = sock.recv(min(size - len(result), 1024 * 1024))
        if not block:
            raise EOFError("GPU_IPC_CONNECTION_CLOSED")
        result.extend(block)
    return bytes(result)


def send_message(
    sock: socket.socket,
    value: Any,
    authkey: bytes,
    deadline: float,
    max_bytes: int,
) -> None:
    """Thực thi send_message."""
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > max_bytes:
        raise ValueError("GPU_IPC_MESSAGE_TOO_LARGE")

    header = HEADER.pack(len(payload))
    signature = hmac.new(
        authkey, header + payload, hashlib.sha256
    ).digest()

    sock.settimeout(_remaining(deadline))
    sock.sendall(header + signature)
    sock.settimeout(_remaining(deadline))
    sock.sendall(payload)


def receive_message(
    sock: socket.socket,
    authkey: bytes,
    deadline: float,
    max_bytes: int,
) -> Any:
    """Thực thi receive_message."""
    header = _read_exact(sock, HEADER.size, deadline)
    size = HEADER.unpack(header)[0]
    if size > max_bytes:
        raise ValueError("GPU_IPC_MESSAGE_TOO_LARGE")

    signature = _read_exact(sock, MAC_SIZE, deadline)
    payload = _read_exact(sock, size, deadline)
    expected = hmac.new(
        authkey, header + payload, hashlib.sha256
    ).digest()

    if not hmac.compare_digest(signature, expected):
        raise PermissionError("GPU_IPC_AUTHENTICATION_FAILED")

    # Chỉ unpickle sau khi xác thực từ process cùng run.
    return pickle.loads(payload)


def request(
    endpoint: tuple[str, int, bytes],
    message: dict[str, Any],
    timeout_seconds: float,
    max_bytes: int,
) -> dict[str, Any]:
    """Thực thi request."""
    host, port, authkey = endpoint
    if host != "127.0.0.1":
        raise ValueError("GPU_IPC_REQUIRES_LOOPBACK")

    deadline = time.monotonic() + float(timeout_seconds)

    with socket.create_connection(
        (host, int(port)),
        timeout=_remaining(deadline),
    ) as sock:
        send_message(
            sock, message, authkey, deadline, max_bytes
        )
        response = receive_message(
            sock, authkey, deadline, max_bytes
        )

    if not isinstance(response, dict):
        raise TypeError("GPU_IPC_RESPONSE_NOT_DICT")
    return response
