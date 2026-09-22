"""
git_commit_v2.py — Pure-Python git commit using only stdlib.
Works without dulwich by writing git objects directly.
Commits ALL tracked files in the repo (excluding .venv, __pycache__, etc.)
against the current HEAD tree, producing a proper incremental commit.

Usage: python git_commit_v2.py "commit message"
"""
import os, sys, zlib, hashlib, time, struct

REPO = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', '.gemini', 'demo_fallback', 'data\\scratch', 'data/scratch'}
EXCLUDE_NAMES = {'.pyc', '.pyo'}

def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def write_git_object(data: bytes) -> str:
    sha = sha1_hex(data)
    d1, d2 = sha[:2], sha[2:]
    obj_dir = os.path.join(REPO, '.git', 'objects', d1)
    obj_path = os.path.join(obj_dir, d2)
    if not os.path.exists(obj_path):
        os.makedirs(obj_dir, exist_ok=True)
        with open(obj_path, 'wb') as f:
            f.write(zlib.compress(data, level=1))
    return sha

def make_blob(content: bytes) -> str:
    header = f'blob {len(content)}\x00'.encode()
    return write_git_object(header + content)

def make_tree(entries: list) -> str:
    """entries: list of (mode_str, name_str, sha_hex)"""
    tree_data = b''
    for mode, name, sha in sorted(entries, key=lambda e: e[1]):
        entry = f'{mode} {name}\x00'.encode() + bytes.fromhex(sha)
        tree_data += entry
    header = f'tree {len(tree_data)}\x00'.encode()
    return write_git_object(header + tree_data)

def make_commit(tree_sha, parent_sha, message, author, ts_epoch, tz_offset='+0530'):
    """Create a commit object."""
    lines = [f'tree {tree_sha}']
    if parent_sha:
        lines.append(f'parent {parent_sha}')
    ts = f'{int(ts_epoch)} {tz_offset}'
    lines += [
        f'author {author} {ts}',
        f'committer {author} {ts}',
        '',
        message,
    ]
    body = '\n'.join(lines) + '\n'
    body_bytes = body.encode('utf-8')
    header = f'commit {len(body_bytes)}\x00'.encode()
    return write_git_object(header + body_bytes)

def read_git_object(sha):
    path = os.path.join(REPO, '.git', 'objects', sha[:2], sha[2:])
    with open(path, 'rb') as f:
        raw = zlib.decompress(f.read())
    null = raw.index(b'\x00')
    hdr = raw[:null].decode()
    obj_type, _ = hdr.split(' ')
    return obj_type, raw[null+1:]

def read_tree_entries(tree_sha):
    """Returns list of (mode, name, sha_hex) for a tree object."""
    _, data = read_git_object(tree_sha)
    entries = []
    i = 0
    while i < len(data):
        sp = data.index(b' ', i)
        mode = data[i:sp].decode()
        nl = data.index(b'\x00', sp)
        name = data[sp+1:nl].decode()
        sha_bytes = data[nl+1:nl+21]
        entries.append((mode, name, sha_bytes.hex()))
        i = nl + 21
    return entries

def build_tree_from_disk(base_dir, rel_path=''):
    """Recursively build git tree objects from disk. Returns tree sha."""
    entries = []
    scan_dir = os.path.join(base_dir, rel_path) if rel_path else base_dir
    
    for name in sorted(os.listdir(scan_dir)):
        full = os.path.join(scan_dir, name)
        rel = os.path.join(rel_path, name).replace('\\', '/') if rel_path else name
        
        # Skip excluded
        if name in {'.git', '.venv', '__pycache__', '.gemini', 'demo_fallback', 'notebooks'}:
            continue
        if name.endswith(('.pyc', '.pyo')):
            continue
        if 'scratch' in rel.split('/'):
            continue
        
        if os.path.isdir(full):
            sub_sha = build_tree_from_disk(base_dir, rel)
            if sub_sha:
                entries.append(('40000', name, sub_sha))
        elif os.path.isfile(full):
            with open(full, 'rb') as f:
                content = f.read()
            blob_sha = make_blob(content)
            entries.append(('100644', name, blob_sha))
    
    if not entries:
        return None
    return make_tree(entries)

def get_head():
    head_path = os.path.join(REPO, '.git', 'HEAD')
    with open(head_path) as f:
        head = f.read().strip()
    if head.startswith('ref:'):
        ref = head.split(' ')[1]
        ref_path = os.path.join(REPO, '.git', ref)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                return f.read().strip(), ref_path
        return None, ref_path
    return head, None

def commit_all(message, author='LunarAlign <lunaralign@sih2026>'):
    parent_sha, ref_path = get_head()
    print(f'  Parent commit: {parent_sha}')
    
    print(f'  Building tree from disk...')
    tree_sha = build_tree_from_disk(REPO)
    print(f'  Tree SHA: {tree_sha}')
    
    ts = time.time()
    commit_sha = make_commit(tree_sha, parent_sha, message, author, ts)
    print(f'  Commit SHA: {commit_sha}')
    
    # Update ref
    if ref_path:
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        with open(ref_path, 'w') as f:
            f.write(commit_sha + '\n')
    
    # Update reflog
    log_path = os.path.join(REPO, '.git', 'logs', 'HEAD')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts_int = int(ts)
    entry = f'{parent_sha or "0"*40} {commit_sha} {author} {ts_int} +0530\t{message}\n'
    with open(log_path, 'a') as f:
        f.write(entry)
    
    return commit_sha

if __name__ == '__main__':
    msg = sys.argv[1] if len(sys.argv) > 1 else 'auto-commit'
    print(f'Committing: "{msg}"')
    sha = commit_all(msg)
    print(f'Done. New HEAD: {sha}')
