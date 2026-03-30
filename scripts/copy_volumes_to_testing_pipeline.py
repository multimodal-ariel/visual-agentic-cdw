import os
from pathlib import Path
from typing import List, Optional
import shutil

def load_manifest(manifest_path: Optional[str]) -> List[str]:
    if manifest_path is None:
        return []
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        print(f"[WARN] Manifest not found: {manifest_path}")
        return []

    with open(manifest_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths


def resolve_from_symlink(link_path: Path) -> Optional[Path]:
    """
    Resolve the true source path from a symlink-like file.
    Returns None if it cannot be resolved to an existing file.
    """
    try:
        real_path = Path(os.path.realpath(link_path))
        if real_path.exists():
            return real_path
    except Exception:
        pass
    return None


def infer_source_from_manifest(case_dir_name: str, manifest_paths: List[str]) -> Optional[Path]:
    """
    Fallback matching using case folder name against global manifest entries.
    This is intentionally conservative for now.
    """
    # crude tokenization; can refine later
    tokens = case_dir_name.split("_")

    # keep stronger tokens only
    strong_tokens = [t for t in tokens if len(t) >= 6]

    candidates = []
    for p in manifest_paths:
        score = sum(tok in p for tok in strong_tokens)
        if score > 0:
            candidates.append((score, p))

    if not candidates:
        return None

    # highest token match wins
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_path = Path(candidates[0][1])

    if best_path.exists():
        return best_path

    return None


def find_cases(pipeline_root: str) -> List[Path]:
    root = Path(pipeline_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Pipeline root not found: {pipeline_root}")

    case_dirs = []
    for item in sorted(root.iterdir()):
        if item.is_dir():
            link_path = item / "image_nifti.nii.gz"
            if link_path.exists() or link_path.is_symlink():
                case_dirs.append(item)
    return case_dirs


def main(pipeline_root: str, manifest_path: Optional[str] = None) -> None:
    manifest_paths = load_manifest(manifest_path)
    case_dirs = find_cases(pipeline_root)

    print(f"[INFO] Found {len(case_dirs)} case directories\n")

    for case_dir in case_dirs:
        case_name = case_dir.name
        link_path = case_dir / "image_nifti.nii.gz"
        dest_path = case_dir / "volume.nii.gz"

        source_path = resolve_from_symlink(link_path)

        if source_path is None and manifest_paths:
            source_path = infer_source_from_manifest(case_name, manifest_paths)

        print("=" * 100)
        print(f"CASE   : {case_name}")
        print(f"LINK   : {link_path}")
        print(f"SOURCE : {source_path if source_path is not None else 'NOT FOUND'}")
        print(f"DEST   : {dest_path}")
        print(f"STATUS : {'OK' if source_path is not None else 'FAILED'}")

        shutil.copy2(source_path, dest_path)


if __name__ == "__main__":
    # Example usage:
    pipeline_root = "/data/soumitri/test_pipeline_new_2_llm"
    manifest_path = None  # e.g. "/data/soumitri/all_original_image_paths.txt"
    main(pipeline_root, manifest_path)