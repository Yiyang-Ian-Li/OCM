from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_BENCHMARKS_DIR = REPO_ROOT / "raw_benchmarks"


def ensure_git_repo(path: Path, repo_url: str, expected_child: str) -> Path:
    path = Path(path).expanduser()
    if path.exists():
        if not (path / expected_child).exists():
            raise FileNotFoundError(
                f"{path} exists but does not look like the expected repository; "
                f"missing {expected_child!r}."
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(path)],
        check=True,
    )
    if not (path / expected_child).exists():
        raise FileNotFoundError(
            f"Cloned {repo_url} into {path}, but missing expected path {expected_child!r}."
        )
    return path
