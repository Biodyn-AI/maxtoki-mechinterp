#!/usr/bin/env python3
"""Build one folder per agent for the isolated Haiku re-run (Amendment 3).

Each folder mirrors one Study A task package: real sub-folders and one
symbolic link per package file, so files an agent writes stay in its own
folder. Folder names are opaque: 'u' + 6 digits from a salted FNV-1a hash of
the agent label, so the workflow can compute them (same function in JS). Writes the mapping
{label: folder} to studyA_haiku2/folder_map.json.
"""
import json, os

EV = '<EVAL_ROOT>'
AT = '<CODE_ROOT>/analysis-tasks'
OUT = os.path.join(EV, 'studyA_haiku2')
TASKS = {'T1': ('w4891', 'w6252'), 'T2': ('w3637', 'w3318'), 'T3': ('w6895', 'w9668'), 'T4': ('w7421', 'w3033')}
ROLES = ['exec', 'review-generic', 'review-checklist', 'repair-generic', 'repair-checklist',
         'rgrade-generic', 'rgrade-checklist'] + [f'{c}:grade{g}' for c in ['none', 'generic', 'checklist'] for g in (1, 2, 3)]
SALT = 1
taken = set(os.listdir(AT))


def fnv(s):
    h = 0x811c9dc5
    for b in s.encode():
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h


def name_for(lab):
    return f'u{100000 + fnv(f"{SALT}|{lab}") % 900000}'


def mirror(src, dst):
    os.makedirs(dst)
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        rel = os.path.relpath(root, src)
        tgt = dst if rel == '.' else os.path.join(dst, rel)
        os.makedirs(tgt, exist_ok=True)
        for f in files:
            if f.startswith('._') or f == '.DS_Store':
                continue
            os.symlink(os.path.join(root, f), os.path.join(tgt, f))
            n += 1
    return n


os.makedirs(OUT, exist_ok=True)
fmap, nlinks = {}, 0
for t, (paper, contract) in TASKS.items():
    for arm, pkg in (('paper', paper), ('contract', contract)):
        for rep in range(1, 6):
            for role in ROLES:
                lab = f'AI:{t}:{arm}:{rep}:{role}'
                name = name_for(lab)
                assert name not in taken, name
                taken.add(name)
                nlinks += mirror(os.path.join(AT, pkg), os.path.join(AT, name))
                fmap[lab] = os.path.join(AT, name)
json.dump(fmap, open(os.path.join(OUT, 'folder_map.json'), 'w'), indent=1)
print(f'{len(fmap)} folders, {nlinks} links -> {OUT}/folder_map.json')
