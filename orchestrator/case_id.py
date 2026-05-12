"""Stable case-ID derivation.

Many clinical filelists contain paths whose basenames collide:
    /data/.../RHEUM/PT0001/CT_CHEST/CORONAL
    /data/.../LUPUS/PT0002/CT_CHEST/CORONAL
                         ^^^^^^^^^^ same basename, different case

Using basename as the case ID corrupts the state tracker (one entry overwrites
another) and the per-case log filenames. Derive the ID from the path relative
to a known root so it's unique by construction.
"""

from __future__ import annotations

from typing import Iterable

from config.constants import (
    DICOM_ROOT,
    SEGMENTATION_ROOT,
    SRC_PREFIX,
    DST_PREFIX,
)


def derive_case_id(case_path: str) -> str:
    """Return a stable, filesystem-safe ID derived from a case directory path.

    Strips the longest matching known root prefix, then replaces remaining
    slashes with ``__``. Result is unique per input path under the same root.

    Examples:
        /data/soumitri/segmentations_3d/RHEUM/PT0001/CT_ABD_PELVIS
            → "RHEUM__PT0001__CT_ABD_PELVIS"
        /data/RAD/RHEUM/PT0001/CT_CHEST/CORONAL
            → "RHEUM__PT0001__CT_CHEST__CORONAL"

    Note: the reverse (``case_id_to_relpath``) round-trips correctly as long
    as no original directory name contains the literal substring "__".
    """
    p = str(case_path).rstrip("/")

    # Order matters: try longest/most-specific prefixes first.
    candidates = [DST_PREFIX, SRC_PREFIX, f"{SEGMENTATION_ROOT}/", f"{DICOM_ROOT}/"]
    for prefix in candidates:
        if prefix and p.startswith(prefix):
            p = p[len(prefix):]
            break

    return p.strip("/").replace("/", "__")


def case_id_to_relpath(case_id: str) -> str:
    """Inverse of ``derive_case_id`` (ignoring the stripped root prefix).

    Returns the slash-separated path relative to whichever root the case_id
    was derived under. Caller is responsible for joining with a root to
    obtain an absolute path.

    Example:
        "RHEUM__PT0001__CT_ABD_PELVIS" → "RHEUM/PT0001/CT_ABD_PELVIS"
    """
    return case_id.replace("__", "/")


def case_id_to_abspath(case_id: str, root: str) -> str:
    """Reconstruct an absolute path from a case_id and a known root.

    Example:
        case_id_to_abspath("RHEUM__PT0001__CT_ABD_PELVIS", SEGMENTATION_ROOT)
            → "/data/soumitri/segmentations_3d/RHEUM/PT0001/CT_ABD_PELVIS"
    """
    rel = case_id_to_relpath(case_id)
    return f"{root.rstrip('/')}/{rel}"


def assert_unique_ids(case_paths: Iterable[str]) -> dict:
    """Map each path to its case ID, raising ValueError on any collision.

    Returns the {case_id: path} dict so callers can use it directly.

    Raises on:
      - Two different paths producing the same case ID (under-specified roots).
      - The same path appearing twice in the filelist (user typo / merge bug).
    Both are worth telling the user about at 24K-volume scale.
    """
    out: dict = {}
    collisions: dict = {}
    duplicates: list = []
    for p in case_paths:
        cid = derive_case_id(p)
        if cid in out:
            if out[cid] == p:
                duplicates.append(p)
            else:
                collisions.setdefault(cid, [out[cid]]).append(p)
        else:
            out[cid] = p
    if collisions:
        first_cid, first_paths = next(iter(collisions.items()))
        raise ValueError(
            f"Duplicate case IDs after derivation ({len(collisions)} collisions). "
            f"Example id={first_cid!r} from paths {first_paths!r}. "
            f"Check that filelist paths are unique under the configured roots."
        )
    if duplicates:
        raise ValueError(
            f"Filelist contains {len(duplicates)} duplicate path(s); "
            f"first duplicate: {duplicates[0]!r}. Deduplicate the filelist before running."
        )
    return out
