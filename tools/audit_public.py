"""Audit a public staging tree without network/Git operations.

The optional kit/guide inputs remain external and read-only. Binary media also
require an existing ffprobe for container/stream tag inspection; raw bytes are
scanned for sensitive ASCII and UTF-16 strings independently of that tool.
"""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

DENIED_PARTS = {"sim", "model", ".venv", "__pycache__", "evidence", ".git"}
DENIED_NAMES = {"run.py", "evaluate.py", "install.py", "servo_controller.py",
                "torque_controller.py", "collision_detector.py", "guide_extracted.txt"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".sha256", ".html", ".yml", ".toml"}
MEDIA_SUFFIXES = {".mp4", ".gif"}
MEDIA_ROOT = PurePosixPath("artifacts/media")
MEDIA_TOTAL_LIMIT = 30_000_000
ALLOWED_MEDIA_TAGS = {"encoder", "language", "handler_name", "vendor_id", "major_brand",
                      "minor_version", "compatible_brands"}
# The course interface signature is an API contract, not prose redistribution.
GUIDE_ALLOWED_SIGNATURE = "def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):"


def entries_under(root):
    """Walk without following links; prune only the real root .git directory."""
    root = Path(root)
    pending, entries = [root], []
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.parent == root and path.name == ".git" and path.is_dir() and not path.is_symlink():
                continue
            entries.append(path)
            if not path.is_symlink() and path.is_dir():
                pending.append(path)
    return sorted(entries)


def files_under(root):
    return [p for p in entries_under(root) if p.is_symlink() or not p.is_dir()]


def text_errors(text):
    patterns = {"machine home path": "/" + "home/",
                "private workflow reference": "agent" + "-workflow",
                "internal task identifier": r"TASK-\d{3,}",
                "internal evidence reference": "local/" + "evidence/",
                "contact/email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
                "phone contact": r"(?<!\d)(?:\+82[- .]?)?(?:010|02|0[3-6]\d)[- .]\d{3,4}[- .]\d{4}(?!\d)",
                "private key": "-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                "API credential": r"sk-" + r"(?:proj-)?[A-Za-z0-9_-]{24,}"}
    return [label for label, pattern in patterns.items() if re.search(pattern, text)]


def binary_errors(data, include_email=True, compressed_payload=False):
    """Examine printable ASCII and UTF-16 strings without decoding arbitrary bytes."""
    runs = [m.group().decode("ascii") for m in re.finditer(rb"[\x20-\x7e]{5,}", data)]
    for encoding, pattern in (("utf-16-le", rb"(?:[\x20-\x7e]\x00){5,}"),
                              ("utf-16-be", rb"(?:\x00[\x20-\x7e]){5,}")):
        runs.extend(m.group().decode(encoding) for m in re.finditer(pattern, data))
    found = set()
    user = getpass.getuser()
    for value in runs:
        if compressed_payload:
            # High-specificity exact markers only: generic e-mail/phone/path
            # patterns can arise by chance in entropy-coded pixels.
            if "/" + "home/" + user + "/" in value:
                found.add("local home path")
            if "agent" + "-workflow" in value or "local/" + "evidence/" in value:
                found.add("private workflow reference")
            if re.search(r"TASK-\d{3,}", value):
                found.add("internal task identifier")
            if re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", value):
                found.add("private key")
            if re.search(r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}", value):
                found.add("API credential")
        else:
            found.update(text_errors(value))
        if len(user) >= 4 and re.search(re.escape(user), value, re.I):
            found.add("local username")
        if not compressed_payload and re.search(r"/(?:Users|tmp|mnt|media|var|opt)/[^\s\x00]+", value):
            found.add("absolute local path")
    if not include_email:
        found.discard("contact/email")
    return sorted(found)


def media_metadata_bytes(data, suffix):
    """Extract container metadata, excluding compressed image/video payloads.

    This avoids mistaking coincidental byte patterns in compressed pixels for
    actual e-mail addresses while still scanning GIF extensions and all MP4
    top-level boxes other than mdat. ffprobe independently checks parsed tags.
    """
    if suffix == ".mp4":
        parts, offset = [], 0
        while offset < len(data):
            if offset + 8 > len(data):
                raise ValueError("truncated MP4 box")
            size = int.from_bytes(data[offset:offset+4], "big")
            kind = data[offset+4:offset+8]
            header = 8
            if size == 1:
                if offset + 16 > len(data):
                    raise ValueError("truncated MP4 extended box")
                size = int.from_bytes(data[offset+8:offset+16], "big")
                header = 16
            elif size == 0:
                size = len(data) - offset
            if size < header or offset + size > len(data):
                raise ValueError("invalid MP4 box length")
            if kind != b"mdat":
                parts.append(data[offset:offset+size])
            offset += size
        return b"\n".join(parts)
    if suffix != ".gif" or len(data) < 13 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        raise ValueError("invalid GIF header")
    flags = data[10]
    offset = 13 + (3 * (2 ** ((flags & 7) + 1)) if flags & 128 else 0)
    parts = [data[:13]]
    while offset < len(data):
        marker = data[offset]
        offset += 1
        if marker == 0x3b:
            parts.append(data[offset:])
            return b"\n".join(parts)
        if marker == 0x21:
            if offset >= len(data):
                raise ValueError("truncated GIF extension")
            offset += 1  # extension label
            capture = True
        elif marker == 0x2c:
            if offset + 9 > len(data):
                raise ValueError("truncated GIF image descriptor")
            packed = data[offset+8]
            offset += 9 + (3 * (2 ** ((packed & 7) + 1)) if packed & 128 else 0)
            offset += 1  # LZW minimum code size
            capture = False
        else:
            raise ValueError("unknown GIF block")
        block = bytearray()
        while True:
            if offset >= len(data):
                raise ValueError("truncated GIF sub-block")
            length = data[offset]
            offset += 1
            if length == 0:
                break
            if offset + length > len(data):
                raise ValueError("invalid GIF sub-block length")
            if capture:
                block.extend(data[offset:offset+length])
            offset += length
        if capture:
            parts.append(bytes(block))
    raise ValueError("missing GIF trailer")


def normalized_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def copied_blocks(public_root, kit_root, length=6):
    if kit_root is None:
        return []
    originals = {}
    for source in kit_root.rglob("*.py"):
        if ".venv" in source.parts or ".git" in source.parts or source.is_symlink():
            continue
        lines = normalized_lines(source)
        for i in range(len(lines) - length + 1):
            originals.setdefault("\n".join(lines[i:i+length]), source.relative_to(kit_root).as_posix())
    matches = []
    for path in files_under(public_root):
        if path.is_symlink() or not path.is_file() or path.suffix != ".py":
            continue
        lines = normalized_lines(path)
        for i in range(len(lines) - length + 1):
            block = "\n".join(lines[i:i+length])
            if block in originals:
                matches.append({"file": path.relative_to(public_root).as_posix(),
                                "kit_file": originals[block], "normalized_start": i + 1})
    return matches


def guide_overlaps(public_root, guide, n=10, minimum_chars=60):
    """Detect reflowed guide text; ignore the documented public compute signature."""
    if guide is None:
        return []
    words = re.findall(r"\S+", Path(guide).read_text(encoding="utf-8"))
    chunks = {" ".join(words[i:i+n]) for i in range(len(words)-n+1)
              if len(" ".join(words[i:i+n])) >= minimum_chars}
    matches = []
    for path in files_under(public_root):
        if path.is_symlink() or not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        tokens = re.findall(r"\S+", path.read_text(encoding="utf-8"))
        for i in range(len(tokens)-n+1):
            snippet = " ".join(tokens[i:i+n])
            if snippet in chunks and snippet != GUIDE_ALLOWED_SIGNATURE:
                # Do not copy the guide text into the audit report.
                matches.append({"file": path.relative_to(public_root).as_posix(), "token_start": i+1})
                break
    return matches


def media_tag_errors(path, ffprobe):
    """Reject unreadable media, unexpected tags, and private tag values."""
    if not ffprobe:
        return ["ffprobe is required for media metadata inspection"]
    try:
        proc = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags:stream_tags",
                               "-of", "json", str(path)], text=True, capture_output=True,
                              timeout=60, check=False)
        if proc.returncode:
            return ["ffprobe metadata inspection failed"]
        data = json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ["ffprobe metadata inspection failed"]
    errors = set()
    tag_maps = [data.get("format", {}).get("tags", {})]
    tag_maps.extend(s.get("tags", {}) for s in data.get("streams", []))
    for tags in tag_maps:
        for key, value in tags.items():
            if key.lower() not in ALLOWED_MEDIA_TAGS:
                errors.add("unexpected media metadata tag")
            errors.update(binary_errors(str(value).encode("utf-8")))
    return sorted(errors)


def audit(root, kit=None, guide=None, ffprobe=None, require_checksums=True, require_media=False):
    root = Path(root).resolve()
    paths = files_under(root)
    errors, listing = [], []
    media = []
    media_metadata_checked_count = 0
    for path in paths:
        rel = path.relative_to(root)
        name = rel.as_posix()
        if path.is_symlink():
            errors.append(f"symlink: {name}")
            continue
        if not path.is_file():
            errors.append(f"special file: {name}")
            continue
        if set(rel.parts) & DENIED_PARTS or path.name in DENIED_NAMES:
            errors.append(f"excluded kit/environment/raw file: {name}")
        if (path.suffix.lower() in {".npz", ".npy", ".stl", ".xml", ".log"} or
                (path.suffix.lower() != ".mp4" and path.stat().st_size > 8*1024**2)):
            errors.append(f"raw/large/model artifact: {name}")
        listing.append({"file": name, "bytes": path.stat().st_size})
        is_text = path.suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE", "THIRD_PARTY_NOTICES", "SHA256SUMS"}
        if is_text:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                errors.append(f"invalid UTF-8 text: {name}")
            else:
                errors.extend(f"{name}: {error}" for error in text_errors(text))
        else:
            raw = path.read_bytes()
            is_media = path.suffix.lower() in MEDIA_SUFFIXES
            errors.extend(f"{name}: {error} in binary strings"
                          for error in binary_errors(raw, include_email=not is_media, compressed_payload=is_media))
            if is_media:
                try:
                    metadata = media_metadata_bytes(raw, path.suffix.lower())
                except ValueError:
                    errors.append(f"{name}: invalid media container structure")
                else:
                    errors.extend(f"{name}: {error} in binary metadata" for error in binary_errors(metadata))
        if path.suffix.lower() in MEDIA_SUFFIXES:
            media.append(path)
            if not PurePosixPath(name).is_relative_to(MEDIA_ROOT):
                errors.append(f"media outside artifacts/media: {name}")
            if path.suffix.lower() == ".gif" and path.stat().st_size > 6_000_000:
                errors.append(f"GIF exceeds 6,000,000 bytes: {name}")
            tag_errors = media_tag_errors(path, ffprobe)
            if not tag_errors:
                media_metadata_checked_count += 1
            errors.extend(f"{name}: {error}" for error in tag_errors)
        elif PurePosixPath(name).is_relative_to(MEDIA_ROOT) and name != "artifacts/media/media_manifest.json":
            errors.append(f"unexpected file in media directory: {name}")
    media_bytes = sum(p.stat().st_size for p in media)
    if media_bytes > MEDIA_TOTAL_LIMIT:
        errors.append("media total exceeds 30,000,000 bytes")
    if require_media:
        names = {p.relative_to(root).as_posix() for p in media}
        stems = ("01_servo_no_collision", "02_torque_no_collision",
                 "03_servo_place_collision", "04_servo_transport_collision")
        expected_media = {f"artifacts/media/{stem}.{suffix}" for stem in stems for suffix in ("mp4", "gif")}
        if names != expected_media or not (root/"artifacts/media/media_manifest.json").is_file():
            errors.append("required four MP4/GIF pairs or media manifest missing/unexpected")
    kit_valid = kit is None or (Path(kit).is_dir() and not Path(kit).is_symlink())
    guide_valid = guide is None or (Path(guide).is_file() and not Path(guide).is_symlink())
    if not kit_valid:
        errors.append("kit source for comparison missing or unsafe")
    if not guide_valid:
        errors.append("guide source for comparison missing or unsafe")
    try:
        matches = copied_blocks(root, Path(kit) if kit is not None and kit_valid else None)
    except (OSError, UnicodeError):
        errors.append("kit source comparison failed to read valid UTF-8 files")
        matches = []
        kit_valid = False
    errors.extend(f"kit source block: {m['file']} from {m['kit_file']}" for m in matches)
    try:
        overlaps = guide_overlaps(root, guide if guide_valid else None)
    except (OSError, UnicodeError):
        errors.append("guide comparison failed to read valid UTF-8 files")
        overlaps = []
        guide_valid = False
    errors.extend(f"guide text overlap: {m['file']}" for m in overlaps)
    manifest = root / "artifacts/SHA256SUMS"
    if require_checksums:
        if not manifest.is_file() or manifest.is_symlink():
            errors.append("missing/unsafe checksum manifest")
        else:
            expected = {}
            try:
                manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                manifest_lines = []
                errors.append("checksum manifest is not readable UTF-8")
            for line in manifest_lines:
                try:
                    digest, name = line.split("  ", 1)
                    pure = PurePosixPath(name)
                    if (pure.is_absolute() or name != pure.as_posix() or "\\" in name or
                            ".." in pure.parts or name in expected or name == "artifacts/SHA256SUMS" or
                            not re.fullmatch("[0-9a-f]{64}", digest)):
                        raise ValueError
                    expected[name] = digest
                except ValueError:
                    errors.append("malformed checksum manifest")
            actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths if p != manifest and not p.is_symlink() and p.is_file()}
            if actual != expected:
                errors.append("checksum content/coverage mismatch")
    return {"status": "pass" if not errors else "FAIL", "files": listing,
            "file_count": len(listing), "total_bytes": sum(f["bytes"] for f in listing),
            "media_files": len(media), "media_bytes": media_bytes, "errors": errors,
            "copied_source_blocks": matches, "guide_overlaps": overlaps,
            "kit_comparison_performed": kit is not None and kit_valid,
            "guide_comparison_performed": guide is not None and guide_valid,
            "media_metadata_checked": bool(media and media_metadata_checked_count == len(media)),
            "media_metadata_checked_count": media_metadata_checked_count,
            "required_media_checked": require_media,
            "scope": "finite path/text/binary/metadata/checksum, 6-line kit and 10-token guide checks plus human review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--guide", type=Path)
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--require-media", action="store_true", help="require the four video/GIF pairs and manifest")
    parser.add_argument("--out", type=Path, help="report outside staging; avoid self-checksum cycle")
    args = parser.parse_args()
    result = audit(args.root, args.kit, args.guide, args.ffprobe, require_media=args.require_media)
    content = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(content)
    print(content, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
