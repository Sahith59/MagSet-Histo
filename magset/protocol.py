"""Audit the provider's five trial files without executing its legacy script."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile
import urllib.request
from .data import case_id, cases_from_rows, file_hash, read_manifest

FOLD_URL = 'https://www.inf.ufpr.br/lesoliveira/download/mkfold.tar.gz'


def audit_fold(text, rows):
    lookup = {r['filename']: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Filenames must be globally unique for provider matching')
    assignments, roles, stem_roles = {}, defaultdict(set), defaultdict(set)
    for line in text.splitlines():
        parts = line.split('|')
        if len(parts) != 4:
            raise ValueError('Unexpected provider fold row')
        filename, mag, trial, role = (p.strip() for p in parts)
        if filename not in lookup or filename in assignments or role not in {'train', 'test'}:
            raise ValueError('Unknown/duplicate image or unsupported role in provider fold')
        row = lookup[filename]
        if int(mag) != int(row['magnification']):
            raise ValueError('Magnification differs between fold and manifest')
        assignments[filename] = role
        roles[case_id(row)].add(role)
        stem_roles[row['patient_group_conservative']].add(role)
    if set(assignments) != set(lookup):
        raise ValueError('Provider fold does not cover the manifest exactly')
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError('Provider case crosses train/test')
    case_roles = {k: next(iter(v)) for k, v in sorted(roles.items())}
    return {'image_counts': dict(Counter(assignments.values())),
            'case_counts': dict(Counter(case_roles.values())),
            'provider_case_overlap': 0,
            'numeric_stem_overlap': sum(len(v) > 1 for v in stem_roles.values()),
            'case_roles': case_roles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fold-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch-folds', action='store_true')
    args = parser.parse_args()
    if not args.fold_archive.exists():
        if not args.fetch_folds:
            parser.error('Fold archive missing; use --fetch-folds')
        args.fold_archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FOLD_URL, timeout=60) as response:
            body = response.read(5 * 1024 * 1024 + 1)
        if len(body) > 5 * 1024 * 1024:
            raise ValueError('Unexpectedly large fold metadata archive')
        args.fold_archive.write_bytes(body)
    rows = read_manifest(args.manifest)
    cases = cases_from_rows(rows)
    report = {'manifest_sha256': file_hash(args.manifest), 'fold_archive_sha256': file_hash(args.fold_archive),
              'fold_source': FOLD_URL, 'provider_case_count': len(cases),
              'unit': 'provider-defined subtype-qualified case; independent clinical identity not reverified',
              'cases_per_subtype': dict(Counter(c['subtype'] for c in cases.values())), 'trials': {}}
    with tarfile.open(args.fold_archive, 'r:gz') as bundle:
        for trial in range(1, 6):
            name = 'dsfold%d.txt' % trial
            item = bundle.getmember(name)
            if not item.isfile() or item.size > 2 * 1024 * 1024:
                raise ValueError('Invalid fold metadata member')
            report['trials'][str(trial)] = audit_fold(bundle.extractfile(item).read().decode('utf-8'), rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'provider_case_count': len(cases), 'cases_per_subtype': report['cases_per_subtype'],
                      'trials': {k: {a:b for a,b in v.items() if a != 'case_roles'} for k,v in report['trials'].items()}}, indent=2))


if __name__ == '__main__':
    main()
