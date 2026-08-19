"""Read one scene out of the HDR Photographic Survey's 14 GB RAW archive.

Fairchild publishes every NEF of the survey as a single zip of 14.3 GB. Reaching
one scene's eighteen files does not require downloading it: the server answers
byte-range requests and the archive is organised one folder per scene, so a scene
costs about 220 MB on the wire.

What does require work is that the archive was written without ZIP64 records
although it is far larger than 4 GB, so every offset inside it wrapped modulo
2**32. `zipfile` and the off-the-shelf remote-zip readers follow those offsets and
land in the middle of some other member's data. Two corrections fix it: the
central-directory offset needs a known number of wraps added back, and per-entry
offsets are repaired by counting the points where the sequence decreases, which
works because entries are stored in order.
"""

import hashlib
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from fundamentals_lab.config import (
    HDRPS_CD_BYTES,
    HDRPS_CD_OFFSET_RAW,
    HDRPS_CD_WRAPS,
    HDRPS_RAW_ARCHIVE_URL,
)

WRAP = 2**32
_CHUNK = 4 << 20


@dataclass(frozen=True)
class Member:
    """One file inside the archive, with its offset already repaired."""

    name: str
    method: int  # 0 stored, 8 deflate
    compressed_size: int
    size: int
    offset: int

    @property
    def scene(self) -> str:
        parts = self.name.split("/")
        return parts[1] if len(parts) > 2 else ""


def _get_range(start: int, end: int) -> bytes:
    """One byte range of the archive, inclusive of both ends."""
    request = urllib.request.Request(
        HDRPS_RAW_ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 — fixed http URL
        if response.status != 206:
            raise OSError(
                f"server answered {response.status}, not 206 — it stopped honouring ranges"
            )
        return response.read()


def central_directory() -> list[Member]:
    """Every member of the archive, with wrapped offsets corrected."""
    start = HDRPS_CD_OFFSET_RAW + HDRPS_CD_WRAPS * WRAP
    blob = _get_range(start, start + HDRPS_CD_BYTES - 1)
    if not blob.startswith(b"PK\x01\x02"):
        raise OSError(
            "central directory not where the wrap correction says it is — "
            "the archive changed; re-measure HDRPS_CD_* in config.py"
        )
    members: list[Member] = []
    raw_offsets: list[int] = []
    pos = 0
    while pos + 46 <= len(blob) and blob[pos : pos + 4] == b"PK\x01\x02":
        method = int.from_bytes(blob[pos + 10 : pos + 12], "little")
        csize = int.from_bytes(blob[pos + 20 : pos + 24], "little")
        size = int.from_bytes(blob[pos + 24 : pos + 28], "little")
        name_len = int.from_bytes(blob[pos + 28 : pos + 30], "little")
        extra_len = int.from_bytes(blob[pos + 30 : pos + 32], "little")
        comment_len = int.from_bytes(blob[pos + 32 : pos + 34], "little")
        raw_offsets.append(int.from_bytes(blob[pos + 42 : pos + 46], "little"))
        name = blob[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        members.append(Member(name, method, csize, size, offset=0))
        pos += 46 + name_len + extra_len + comment_len

    # Entries are stored in order, so a decrease means the 32-bit field wrapped.
    wraps = 0
    repaired: list[Member] = []
    for member, raw_offset in zip(members, raw_offsets, strict=True):
        if repaired and raw_offset < raw_offsets[len(repaired) - 1]:
            wraps += 1
        repaired.append(
            Member(member.name, member.method, member.compressed_size, member.size,
                   raw_offset + wraps * WRAP)
        )
    logger.info(f"archive index: {len(repaired)} members, {wraps} offset wraps corrected")
    return repaired


def scene_members(scene: str, suffix: str = ".NEF") -> list[Member]:
    """The files of one scene folder, in name order."""
    prefix = f"HDRPS Raws/{scene}/"
    found = [
        m for m in central_directory()
        if m.name.startswith(prefix) and m.name.upper().endswith(suffix.upper())
    ]
    if not found:
        raise LookupError(f"no {suffix} members under {prefix!r} — check the scene name")
    return sorted(found, key=lambda m: m.name)


def fetch(member: Member, destination: Path) -> str:
    """Download and inflate one member. Returns its sha256."""
    header = _get_range(member.offset, member.offset + 29)
    if not header.startswith(b"PK\x03\x04"):
        raise OSError(f"no local header at {member.offset} for {member.name}")
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    data_start = member.offset + 30 + name_len + extra_len
    data_end = data_start + member.compressed_size - 1

    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if member.method == 8 else None
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as out:
        for chunk_start in range(data_start, data_end + 1, _CHUNK):
            chunk = _get_range(chunk_start, min(chunk_start + _CHUNK - 1, data_end))
            plain = decompressor.decompress(chunk) if decompressor else chunk
            out.write(plain)
            digest.update(plain)
            written += len(plain)
        if decompressor:
            tail = decompressor.flush()
            out.write(tail)
            digest.update(tail)
            written += len(tail)
    if written != member.size:
        raise OSError(f"{member.name}: expected {member.size} bytes, wrote {written}")
    return digest.hexdigest()
