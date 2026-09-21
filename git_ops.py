"""
git_ops.py — Pure-Python git operations for LunarAlign using dulwich.
Handles init, add, commit since git CLI is not installed.
"""
import os
import time
from dulwich.repo import Repo
from dulwich.objects import Blob, Tree, Commit
from dulwich import porcelain

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def init_and_commit(message):
    """Initialize repo if needed and commit all tracked files."""
    try:
        repo = Repo(REPO_ROOT)
        print(f"  Repo already exists at {REPO_ROOT}")
    except Exception:
        repo = Repo.init(REPO_ROOT)
        print(f"  Initialized new repo at {REPO_ROOT}")
    
    # Stage all project files (exclude .venv, __pycache__, .gemini)
    exclude_dirs = {'.venv', '__pycache__', '.git', '.gemini', 'demo_fallback'}
    
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # Prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in filenames:
            if fname.endswith('.pyc'):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, REPO_ROOT).replace('\\', '/')
            try:
                porcelain.add(REPO_ROOT, paths=[rel])
            except Exception:
                pass
    
    commit_id = porcelain.commit(
        REPO_ROOT,
        message=message.encode('utf-8'),
        author=b"LunarAlign <lunaralign@sih2026>",
        committer=b"LunarAlign <lunaralign@sih2026>",
    )
    print(f"  Committed: {commit_id.decode()}")
    return commit_id.decode()

if __name__ == '__main__':
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "auto-commit"
    commit_hash = init_and_commit(msg)
    print(f"Commit hash: {commit_hash}")
