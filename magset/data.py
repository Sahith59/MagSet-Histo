"""Manifest identities; a case is not silently equated with a verified patient."""
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

MAGNIFICATIONS = (40, 100, 200, 400)
CLASS_NAMES = ('adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma',
               'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Manifest must have unique, nonempty relative image paths')
    for r in rows:
        if r['subtype'] not in CLASS_NAMES or int(r['magnification']) not in MAGNIFICATIONS:
            raise ValueError('Unsupported subtype or magnification')
    return rows


def case_id(row):
    # This reconstructs the subtype-qualified identity used by the provider folds.
    prefix = row['filename'].split('-')[0]
    return prefix + '-' + row['case_id_raw']


def cases_from_rows(rows):
    cases = {}
    for row in rows:
        key = case_id(row)
        label = CLASS_NAMES.index(row['subtype'])
        if key not in cases:
            cases[key] = {'case_id': key, 'label': label, 'subtype': row['subtype'],
                          'stem_group': row['patient_group_conservative'], 'indices': defaultdict(list)}
        if cases[key]['label'] != label:
            raise ValueError('Case has contradictory labels: ' + key)
        cases[key]['indices'][int(row['magnification'])].append(row['relative_path'])
    return cases
