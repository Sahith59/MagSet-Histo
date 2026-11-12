#!/usr/bin/env python3
"""Download/index BreakHis with standard-library dependencies; run large work via SLURM."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tarfile
from datetime import datetime, timezone

SOURCE_URL = 'https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz'
PATTERN = re.compile(r'^SOB_(B|M)_(A|F|PT|TA|DC|LC|MC|PC)-(.+)-(40|100|200|400)-(\d+)\.png$', re.I)
BENIGN = {'A', 'F', 'PT', 'TA'}
SUBTYPES = {'A': 'adenosis', 'F': 'fibroadenoma', 'PT': 'phyllodes_tumor',
            'TA': 'tubular_adenoma', 'DC': 'ductal_carcinoma',
            'LC': 'lobular_carcinoma', 'MC': 'mucinous_carcinoma', 'PC': 'papillary_carcinoma'}
EXPECTED_MAG = {40: 1995, 100: 2081, 200: 2013, 400: 1820}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_filename(name):
    match = PATTERN.fullmatch(name)
    if match is None:
        raise ValueError('Unrecognized BreakHis image filename: ' + name)
    broad, subtype, raw_id, mag, sequence = match.groups()
    broad, subtype = broad.upper(), subtype.upper()
    if (subtype in BENIGN) != (broad == 'B'):
        raise ValueError('Broad/subtype labels disagree: ' + name)
    # Preserve exact identifiers. Stripping trailing section letters is a conservative
    # grouping proposal only; it must be reconciled with the original protocol.
    conservative = re.sub(r'[A-Za-z]+$', '', raw_id)
    return {'biopsy_method': 'SOB', 'broad_class': 'benign' if broad == 'B' else 'malignant',
            'subtype': SUBTYPES[subtype], 'subtype_code': subtype,
            'case_id_raw': raw_id, 'patient_group_conservative': conservative,
            'group_rule': 'strip-trailing-section-letters-v1-UNVERIFIED',
            'magnification': int(mag), 'sequence': int(sequence)}


def png_header(path):
    with Path(path).open('rb') as stream:
        header = stream.read(29)
    if len(header) != 29 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Invalid PNG header: ' + str(path))
    width, height = struct.unpack('>II', header[16:24])
    return width, height, header[24], header[25]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def extract_archive(archive, destination, digest):
    destination = Path(destination)
    marker = destination / '.archive_sha256'
    if marker.exists():
        if marker.read_text().strip() != digest:
            raise ValueError('Existing extraction belongs to a different archive; refusing overwrite')
        return
    stage = destination.with_name(destination.name + '.extracting')
    if destination.exists():
        raise ValueError('Unmarked destination exists; inspect it before extraction')
    stage.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        total = 0
        for member in bundle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in member.name:
                raise ValueError('Unsafe archive member: ' + member.name)
            if not member.isdir() and not member.isfile():
                raise ValueError('Only regular files/directories are allowed: ' + member.name)
            total += member.size
            if total > 32 * 1024**3:
                raise ValueError('Archive exceeds expected extraction budget')
            target = stage.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(stage.resolve()):
                raise ValueError('Archive target escapes extraction directory')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # A previously interrupted extraction can safely resume by overwriting
                # only this isolated staging directory's regular member files.
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError('Cannot read archive member: ' + member.name)
                with source, target.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    (stage / '.archive_sha256').write_text(digest + '\n')
    os.replace(stage, destination)


def index_images(images_root, output_dir):
    images_root, output_dir = Path(images_root), Path(output_dir)
    files = sorted(images_root.rglob('*.png'))
    if not files:
        raise ValueError('No PNG images found under ' + str(images_root))
    rows, widths, by_mag, by_subtype = [], Counter(), Counter(), Counter()
    raw_coverage, conservative_coverage = defaultdict(set), defaultdict(set)
    raw_labels, conservative_labels = defaultdict(set), defaultdict(set)
    raw_ids_per_group, hashes = defaultdict(set), defaultdict(list)
    for path in files:
        parsed = parse_filename(path.name)
        width, height, depth, color_type = png_header(path)
        digest = sha256(path)
        row = {'relative_path': path.relative_to(images_root).as_posix(), 'filename': path.name,
               **parsed, 'sha256': digest, 'bytes': path.stat().st_size,
               'width': width, 'height': height, 'png_bit_depth': depth, 'png_color_type': color_type}
        rows.append(row)
        widths[(width, height)] += 1
        by_mag[parsed['magnification']] += 1
        by_subtype[parsed['subtype']] += 1
        raw, group = parsed['case_id_raw'], parsed['patient_group_conservative']
        raw_coverage[raw].add(parsed['magnification'])
        conservative_coverage[group].add(parsed['magnification'])
        raw_labels[raw].add(parsed['subtype'])
        conservative_labels[group].add(parsed['subtype'])
        raw_ids_per_group[group].add(raw)
        hashes[digest].append(row['relative_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / 'manifest.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_dir / 'manifest.csv')
    collisions = {g: sorted(ids) for g, ids in raw_ids_per_group.items() if len(ids) > 1}
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'images': len(rows),
             'magnification_counts': dict(sorted(by_mag.items())),
             'subtype_counts': dict(sorted(by_subtype.items())),
             'dimensions': {str(k): v for k, v in widths.items()},
             'raw_case_count': len(raw_coverage), 'conservative_group_count': len(conservative_coverage),
             'raw_cases_with_all_four': sum(len(v) == 4 for v in raw_coverage.values()),
             'conservative_groups_with_all_four': sum(len(v) == 4 for v in conservative_coverage.values()),
             'merged_raw_ids_under_conservative_rule': collisions,
             'conflicting_raw_case_labels': {k: sorted(v) for k, v in raw_labels.items() if len(v) > 1},
             'conflicting_conservative_group_labels': {k: sorted(v) for k, v in conservative_labels.items() if len(v) > 1},
             'identical_file_groups': [v for v in hashes.values() if len(v) > 1],
             'standard_counts_match': dict(by_mag) == EXPECTED_MAG,
             'manifest_sha256': sha256(output_dir / 'manifest.csv'),
             'grouping_status': 'UNVERIFIED: reconcile identifiers before creating train/test splits',
             'decode_status': 'PNG headers checked; full image decoding/CRC validation remains pending'}
    atomic_json(output_dir / 'audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['images', 'magnification_counts', 'raw_case_count',
                                           'conservative_group_count', 'standard_counts_match',
                                           'grouping_status']}, indent=2), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Download official archive and safely extract')
    parser.add_argument('--images-root', type=Path, help='Index existing images instead of downloading')
    args = parser.parse_args()
    if args.download and args.images_root:
        parser.error('--download and --images-root are mutually exclusive')
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    images_root = args.images_root
    if args.download:
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise RuntimeError('Need at least 20 GiB free on the dataset volume')
        archive = root / 'archives' / 'BreaKHis_v1.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            partial = archive.with_suffix(archive.suffix + '.part')
            subprocess.run(['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '30',
                            '--continue-at', '-', '--output', str(partial), '--url', SOURCE_URL], check=True)
            os.replace(partial, archive)
        digest = sha256(archive)
        atomic_json(root / 'archive_provenance.json', {'source_url': SOURCE_URL,
                    'bytes': archive.stat().st_size, 'sha256': digest,
                    'checked_utc': datetime.now(timezone.utc).isoformat(),
                    'verification': 'Local hash recorded; no publisher-provided checksum verified'})
        images_root = root / 'raw'
        extract_archive(archive, images_root, digest)
    if images_root is None:
        parser.error('Use --download or --images-root')
    index_images(images_root, root / 'processed')


if __name__ == '__main__':
    main()
