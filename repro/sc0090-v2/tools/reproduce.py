#!/usr/bin/env python3
"""Verify and materialize exact experiment sources without touching training branches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def safe_child(root, name):
    p = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError(f'Invalid package path: {name}')
    return p

def snapshot_id(name):
    index = read(PACKAGE / 'snapshots.json')
    if name in index:
        return name
    return next(m['snapshot'] for m in read(PACKAGE / 'models.json')['models'] if m['id'] == name)

def snapshot(name):
    sid = snapshot_id(name)
    entry = read(PACKAGE / 'snapshots.json')[sid]
    path = safe_child(PACKAGE, entry['manifest'])
    if digest(path) != entry['manifest_sha256']:
        raise ValueError(f'Snapshot manifest mismatch: {sid}')
    return read(path)

def verify_tree(manifest, root):
    for name, entry in manifest['files'].items():
        p = safe_child(root, name)
        if not p.is_file() or p.is_symlink() or digest(p) != entry['sha256']:
            raise ValueError(f'Source mismatch: {name}')
    # Extra numerical modules would also alter the trainer source identity.
    expected = {n for n in manifest['files'] if n.startswith('src/') and Path(n).suffix in ('.py', '.cpp')}
    actual = {str(p.relative_to(root)) for p in (root/'src').rglob('*') if p.suffix in ('.py', '.cpp')}
    if expected != actual:
        raise ValueError('Unexpected numerical source files')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--artifact-root', type=Path, help='Optional original-layout checkpoint root')
    m = sub.add_parser('materialize')
    m.add_argument('model_or_snapshot')
    m.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--source', type=Path, required=True)
    r.add_argument('--python', type=Path, required=True)
    r.add_argument('script_and_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command == 'verify':
        checked = set()
        for sid in read(PACKAGE/'snapshots.json'):
            for entry in snapshot(sid)['files'].values():
                h = entry['sha256']
                if h in checked:
                    continue
                path = PACKAGE/'objects'/h[:2]/h
                if not path.is_file() or digest(path) != h:
                    raise ValueError(f'Object mismatch: {h}')
                checked.add(h)
        for entry in read(PACKAGE/'evidence-index.json').values():
            if digest(safe_child(PACKAGE, entry['path'])) != entry['sha256']:
                raise ValueError(f'Evidence mismatch: {entry["path"]}')
        models = read(PACKAGE/'models.json')['models']
        if a.artifact_root:
            for model in models:
                cp = safe_child(a.artifact_root, model['artifact_relative_path'])
                if digest(cp) != model['checkpoint_sha256']:
                    raise ValueError(f'Checkpoint mismatch: {model["id"]}')
        print(json.dumps({'status':'PASS','source_objects':len(checked),'models':len(models),'checkpoint_bytes_checked':bool(a.artifact_root)}))
    elif a.command == 'materialize':
        manifest = snapshot(a.model_or_snapshot)
        out = a.output.resolve()
        if out.exists():
            raise FileExistsError('Choose a new source directory; existing trees are never overwritten')
        out.mkdir(parents=True)
        for name, entry in manifest['files'].items():
            h = entry['sha256']; src = PACKAGE/'objects'/h[:2]/h
            if digest(src) != h:
                raise ValueError(f'Object mismatch: {name}')
            dst = safe_child(out, name); dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst); dst.chmod(0o755 if entry['executable'] else 0o644)
        (out/'.repro-snapshot.json').write_text(json.dumps({'snapshot':manifest['id']})+'\n')
        verify_tree(manifest, out)
        print(out)
    else:
        root = a.source.resolve()
        manifest = snapshot(read(root/'.repro-snapshot.json')['snapshot'])
        verify_tree(manifest, root)
        argv = a.script_and_args
        if argv and argv[0] == '--':
            argv = argv[1:]
        if not argv:
            p.error('Supply a script and its arguments after --')
        env = os.environ.copy()
        env['SC0090_V2_PYTHON'] = str(a.python.absolute())
        cache = Path(env.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))/'microduck-sc0090-v2'
        env.setdefault('TORCHINDUCTOR_CACHE_DIR', str(cache/'inductor-no-fma'))
        env.setdefault('TRITON_CACHE_DIR', str(cache/'triton-no-fma'))
        # Keep the historical wrapper bytes exact; only machine-specific paths are overridden.
        raise SystemExit(subprocess.call(['bash',str(root/'scripts/v2_python.sh'),*argv],cwd=root,env=env))

if __name__ == '__main__':
    main()
