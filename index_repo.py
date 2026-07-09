"""Phase 3 — wire ingest -> chunk -> store into one command.

Usage: python index_repo.py https://github.com/psf/requests
"""
import sys
import tempfile

from ingest import clone_repo, collect_files
from chunk import chunk_file
from store import store_chunks, collection_for, repo_name_from_url, set_active_repo


def index_repo(github_url: str):
    repo = repo_name_from_url(github_url)
    collection = collection_for(repo)
    dest = tempfile.mkdtemp()
    clone_repo(github_url, dest)
    total = 0
    batch = []
    for rel_path, text in collect_files(dest):
        batch.extend(chunk_file(rel_path, text))
        if len(batch) >= 100:          # embed in batches of 100 to save API calls
            total += store_chunks(batch, collection)
            batch = []
    if batch:
        total += store_chunks(batch, collection)
    set_active_repo(repo)              # questions now go to this repo
    print(f"Indexed {total} chunks from {github_url} into '{collection}'")
    return total


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/psf/requests"
    index_repo(url)
