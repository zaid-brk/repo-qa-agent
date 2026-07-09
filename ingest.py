"""Phase 1 — get a repo's code onto disk as text.

Shallow clone (--depth 1 = latest commit only, no history — a big repo's history
can be gigabytes we don't need), then walk every file and keep only real code.
"""
import subprocess
import tempfile
from pathlib import Path

# File types worth indexing. Add more as needed.
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
                   ".rs", ".c", ".cpp", ".h", ".md", ".json", ".yaml", ".yml"}
# Folders that are noise, not code.
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}


def clone_repo(github_url: str, dest: str) -> str:
    """Shallow-clone a repo (--depth 1 = just latest commit, faster)."""
    subprocess.run(["git", "clone", "--depth", "1", github_url, dest], check=True)
    return dest


def collect_files(repo_path: str):
    """Yield (relative_path, file_contents) for every code file."""
    repo_path = Path(repo_path)
    for path in repo_path.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in CODE_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue  # skip binaries / unreadable files
        rel = str(path.relative_to(repo_path))
        yield rel, text


if __name__ == "__main__":
    # Quick test: clone a small repo and count files
    dest = tempfile.mkdtemp()
    clone_repo("https://github.com/psf/requests", dest)
    files = list(collect_files(dest))
    print(f"Collected {len(files)} files")
    print("Sample:", files[0][0])
