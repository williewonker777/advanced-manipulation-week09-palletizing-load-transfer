#!/usr/bin/env python3
"""Deterministic public-link checks and narrow archived-README repair proof.

The link checker performs no network access.  It validates only local file and
directory targets; heading anchors are deliberately outside its scope.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import html
import os
import re
from urllib.parse import unquote, urlsplit


ARCHIVE_PATHS = (
    "docs/R1_README.md",
    "docs/R2_README.md",
    "docs/R3_README.md",
)
ORIGINAL_SHA256 = {
    "docs/R1_README.md": "b60f462586204b2db2d0b380c1fd12a57a6dfe5a89e139f6a040d7bb9a07637f",
    "docs/R2_README.md": "e868de9e717a90a64c8e7d3b362e821781d3672ee0b6699d7684906db9764479",
    "docs/R3_README.md": "00fac6faa7ce1959dd8ab3a95514a1b9e31395b6df03a771889d99a6c8d10bf5",
}
EXPECTED_REPAIRS = {
    "docs/R1_README.md": 9,
    "docs/R2_README.md": 31,
    "docs/R3_README.md": 20,
}

_INLINE = re.compile(r"]\(\s*(?:<([^>]+)>|([^\s)]+))")
_REFERENCE = re.compile(r"(?m)^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*(?:<([^>]+)>|([^\s]+))")
_HTML_ATTRIBUTE = re.compile(r"(?is)\b(?:href|src)\s*=\s*(['\"])(.*?)\1")
_HTML_CODE = re.compile(r"(?is)<(?:pre|code)\b[^>]*>.*?</(?:pre|code)>")
_DESTINATION = re.compile(r"(?<=\]\()([^)]+)(?=\))")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _without_code(text: str) -> str:
    """Blank fenced Markdown and HTML code blocks while preserving line count."""
    output = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if fence is None and match:
            fence = match.group(1)[0]
            output.append("\n" if line.endswith("\n") else "")
        elif fence is not None:
            if re.match(r"^[ \t]{0,3}" + re.escape(fence) + r"{3,}[ \t]*$", line.rstrip("\r\n")):
                fence = None
            output.append("\n" if line.endswith("\n") else "")
        else:
            output.append(line)
    return _HTML_CODE.sub("", "".join(output))


def _targets(text: str) -> list[str]:
    visible = _without_code(text)
    found = []
    for pattern in (_INLINE, _REFERENCE):
        found.extend((match.group(1) or match.group(2)) for match in pattern.finditer(visible))
    found.extend(match.group(2) for match in _HTML_ATTRIBUTE.finditer(visible))
    return found


def _documents(root: Path, document_paths) -> list[Path]:
    root = root.resolve()
    if document_paths is not None:
        paths = [root / Path(name) for name in document_paths]
    else:
        paths = []
        excluded = {"logs", "publish", "__pycache__"}
        for directory, names, files in os.walk(root, followlinks=False):
            names[:] = [name for name in names if not name.startswith(".") and name not in excluded
                        and not (Path(directory) / name).is_symlink()]
            paths.extend(Path(directory) / name for name in files
                         if Path(name).suffix.lower() in {".md", ".markdown", ".html", ".htm"})
    selected = []
    for path in paths:
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("document escapes public root: " + str(path))
        relative = resolved.relative_to(root)
        # On a work tree, never recurse into a nested publication or local state.
        if relative.parts[:2] == ("publish", "repo") \
                or any(part.startswith(".") or part == "logs" for part in relative.parts[:-1]):
            continue
        if not resolved.is_file():
            raise FileNotFoundError("public document is absent: " + relative.as_posix())
        selected.append(resolved)
    return sorted(set(selected), key=lambda path: path.relative_to(root).as_posix())


def _local_target(root: Path, document: Path, target: str):
    target = html.unescape(target.strip())
    if not target or target.startswith("#"):
        return None, "anchor"
    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("//"):
        return None, "external"
    path_text = unquote(parsed.path)
    if not path_text:
        return None, "anchor"
    candidate = root / path_text.lstrip("/") if path_text.startswith("/") \
        else document.parent / path_text
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        return resolved, "root_escape"
    return resolved, "local"


def check_public_links(root, document_paths=None) -> dict:
    """Check local file links in public Markdown/HTML without network access."""
    root = Path(root).resolve()
    documents = _documents(root, document_paths)
    missing = []
    relative_links = external = anchors = 0
    for document in documents:
        relative = document.relative_to(root).as_posix()
        for target in _targets(document.read_text(encoding="utf-8")):
            resolved, kind = _local_target(root, document, target)
            if kind == "external":
                external += 1
                continue
            if kind == "anchor":
                anchors += 1
                continue
            relative_links += 1
            if kind == "root_escape" or not resolved.exists():
                entry = {"document": relative, "target": target,
                         "reason": kind if kind == "root_escape" else "missing"}
                if kind != "root_escape":
                    entry["resolved"] = resolved.relative_to(root).as_posix()
                missing.append(entry)
    return {
        "schema": "R4_PUBLIC_LINK_CHECK.v1",
        "documents": [path.relative_to(root).as_posix() for path in documents],
        "document_count": len(documents),
        "relative_links": relative_links,
        "external_links_skipped": external,
        "anchor_only_links_skipped": anchors,
        "heading_anchor_validation": "out_of_scope",
        "missing": missing,
        "passed": not missing,
    }


def _destination_spans(text: str):
    return [(match.start(1), match.end(1), match.group(1))
            for match in _DESTINATION.finditer(text)]


def _same_archive_target(root: Path, relative: str, old: str, new: str):
    """Relocation changes a base directory, never the referenced resource."""
    root = Path(root).resolve()
    if not isinstance(old, str) or not isinstance(new, str):
        raise ValueError("archive destinations must be strings")
    old_url, new_url = (urlsplit(html.unescape(value.strip())) for value in (old, new))
    if (old_url.query, old_url.fragment) != (new_url.query, new_url.fragment):
        raise ValueError("archive query/fragment changed: " + relative)
    original, original_kind = _local_target(root, root / "README.md", old)
    current, current_kind = _local_target(root, root / relative, new)
    if original_kind != "local" or current_kind != "local" or original != current:
        raise ValueError("archive repair must keep the same repository target: " + relative)


def build_archive_link_proof(root, originals: dict[str, bytes]) -> dict:
    """Build proof that only Markdown destinations changed in three archives."""
    root = Path(root)
    if set(originals) != set(ARCHIVE_PATHS):
        raise ValueError("archive proof requires exactly R1/R2/R3 README originals")
    documents = []
    for relative in ARCHIVE_PATHS:
        original = originals[relative]
        if not isinstance(original, bytes) or _sha(original) != ORIGINAL_SHA256[relative]:
            raise ValueError("archive original hash differs: " + relative)
        current = (root / relative).read_bytes()
        try:
            old_text, new_text = original.decode("utf-8"), current.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("archive README must be UTF-8: " + relative) from exc
        old_spans, new_spans = _destination_spans(old_text), _destination_spans(new_text)
        if len(old_spans) != len(new_spans):
            raise ValueError("archive link count changed: " + relative)
        changes = [{"ordinal": index, "old": old[2], "new": new[2]}
                   for index, (old, new) in enumerate(zip(old_spans, new_spans), 1)
                   if old[2] != new[2]]
        if len(changes) != EXPECTED_REPAIRS[relative]:
            raise ValueError("unexpected archive repair count: " + relative)
        for change in changes:
            _same_archive_target(root, relative, change["old"], change["new"])
        reconstructed = new_text
        for change in reversed(changes):
            start, end, value = new_spans[change["ordinal"] - 1]
            if value != change["new"]:
                raise ValueError("archive repair ordinal differs: " + relative)
            reconstructed = reconstructed[:start] + change["old"] + reconstructed[end:]
        if reconstructed.encode("utf-8") != original:
            raise ValueError("archive content outside link destinations changed: " + relative)
        documents.append({"path": relative, "original_sha256": ORIGINAL_SHA256[relative],
                          "current_sha256": _sha(current), "link_count": len(new_spans),
                          "repair_count": len(changes), "changes": changes})
    return {"schema": "R4_DOCUMENT_LINK_REPAIRS.v1", "documents": documents,
            "repair_count": sum(item["repair_count"] for item in documents),
            "url_only_change": True}


def verify_archive_link_proof(root, proof) -> dict:
    """Verify archive repairs from current files and compact proof, without Git."""
    if not isinstance(proof, dict) or proof.get("schema") != "R4_DOCUMENT_LINK_REPAIRS.v1" \
            or proof.get("url_only_change") is not True:
        raise ValueError("invalid archive-link proof schema")
    documents = proof.get("documents")
    if not isinstance(documents, list) or [item.get("path") for item in documents] != list(ARCHIVE_PATHS):
        raise ValueError("archive-link proof inventory differs")
    total = 0
    for item in documents:
        relative = item["path"]
        if item.get("original_sha256") != ORIGINAL_SHA256[relative] \
                or item.get("repair_count") != EXPECTED_REPAIRS[relative]:
            raise ValueError("archive-link proof baseline differs: " + relative)
        current = (Path(root) / relative).read_bytes()
        if _sha(current) != item.get("current_sha256"):
            raise ValueError("archive-link current hash differs: " + relative)
        text = current.decode("utf-8")
        spans = _destination_spans(text)
        if item.get("link_count") != len(spans):
            raise ValueError("archive-link count differs: " + relative)
        changes = item.get("changes")
        if not isinstance(changes, list) or len(changes) != EXPECTED_REPAIRS[relative]:
            raise ValueError("archive-link changes differ: " + relative)
        ordinals = [change.get("ordinal") for change in changes]
        if ordinals != sorted(set(ordinals)) or any(not isinstance(value, int) or value < 1
                                                    or value > len(spans) for value in ordinals):
            raise ValueError("archive-link ordinals differ: " + relative)
        reconstructed = text
        for change in reversed(changes):
            start, end, value = spans[change["ordinal"] - 1]
            if value != change.get("new") or not isinstance(change.get("old"), str):
                raise ValueError("archive-link destination differs: " + relative)
            _same_archive_target(Path(root), relative, change["old"], change["new"])
            reconstructed = reconstructed[:start] + change["old"] + reconstructed[end:]
        if _sha(reconstructed.encode("utf-8")) != ORIGINAL_SHA256[relative]:
            raise ValueError("archive content outside approved URLs differs: " + relative)
        total += len(changes)
    if proof.get("repair_count") != total or total != sum(EXPECTED_REPAIRS.values()):
        raise ValueError("archive-link total differs")
    return {"schema": "R4_DOCUMENT_LINK_REPAIRS_CHECK.v1", "documents": len(documents),
            "repairs": total, "url_only_change": True,
            "same_repository_targets": True, "query_fragment_unchanged": True,
            "passed": True}
