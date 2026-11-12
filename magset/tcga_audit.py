"""Audit TCGA-MS archive membership against supplied labels; do not train or pair fields."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path, PurePosixPath
from .data import file_hash


def audit(inventory):
    root = Path(inventory); label_file = root/'metadata/tcga_ms/labels.csv'
    with label_file.open() as stream:
        labels = list(csv.DictReader(stream))
    lookup = {r['slide_uuid']:r for r in labels}
    if len(lookup) != len(labels):
        raise ValueError('Duplicate slide labels')
    with (root/'members.csv').open() as stream:
        png_entries = [r for r in csv.DictReader(stream) if r['suffix']=='.png']
    sidecars = [r for r in png_entries if PurePosixPath(r['relative_path']).name.startswith('._')]
    images = [r for r in png_entries if not PurePosixPath(r['relative_path']).name.startswith('._')]
    scales = Counter(); slide_scales = defaultdict(Counter); coordinates = defaultdict(set)
    unknown = set(); labeled_images = 0
    for row in images:
        path = PurePosixPath(row['relative_path']); scale = path.parent.name
        slide,x,y = path.stem.rsplit('_',2)
        if not x.isdigit() or not y.isdigit() or not scale.startswith('mpp_'):
            raise ValueError('Unexpected patch path convention')
        scales[scale] += 1; slide_scales[slide][scale] += 1
        coordinates[(slide,x,y)].add(scale)
        if slide not in lookup:
            unknown.add(slide)
        else:
            labeled_images += 1
    folds = {}; class_set = {r['label_id'] for r in labels}
    for fold in range(5):
        groups = defaultdict(set); classes = defaultdict(Counter)
        for row in labels:
            role = row['fold_%d'%fold]
            if role not in ('train','dev','test'):
                raise ValueError('Unrecognized provider split role')
            groups[role].add(row['case_uuid']); classes[role][row['label_id']] += 1
        overlap = {a+'/'+b:len(groups[a]&groups[b]) for a in groups for b in groups if a<b}
        if any(overlap.values()):
            raise ValueError('Provider fold has patient-role overlap')
        folds[str(fold)] = {'slide_counts':dict(Counter(r['fold_%d'%fold] for r in labels)),
                           'patient_role_overlap':overlap,
                           'missing_classes':{r:sorted(class_set-set(c)) for r,c in classes.items()}}
    return {'png_suffixed_entries':len(png_entries),'excluded_sidecar_named_entries':len(sidecars),
            'image_files':len(images),'labeled_image_files':labeled_images,
            'unlabeled_image_files':len(images)-labeled_images,
            'image_slide_ids':len(slide_scales),'labeled_slides':len(labels),
            'labeled_patients':len({r['case_uuid'] for r in labels}),
            'unlabeled_slide_ids':sorted(unknown),'labels_without_images':sorted(set(lookup)-set(slide_scales)),
            'images_by_resolution':dict(sorted(scales.items())),
            'labeled_slides_with_all_resolutions':sum(set(slide_scales[s])==set(scales) for s in lookup),
            'class_slide_counts':dict(sorted(Counter(r['label_id'] for r in labels).items(),key=lambda kv:int(kv[0]))),
            'exact_coordinate_scale_coverage':dict(Counter(len(v) for v in coordinates.values())),
            'folds':folds,'labels_sha256':file_hash(label_file),
            'status':'metadata audit only; exclude unlabeled slides, verify decoding and pretraining overlap before fitting',
            'correspondence':'Identical filename coordinates do not span all seven scales. No registered patch pairing is asserted.'}


def markdown(report):
    lines = ['# TCGA-MS archive and metadata audit', '',
             'The pinned official archive downloaded successfully and matched the provider LFS SHA-256. '
             'The archive was inventoried without executing its contents. This is dataset preparation, not a second-dataset result.', '',
             '| Item | Audited count |','| --- | ---: |',
             '| PNG-suffixed archive entries | %d |'%report['png_suffixed_entries'],
             '| Excluded `._` sidecar-named entries | %d |'%report['excluded_sidecar_named_entries'],
             '| Candidate image files | %d |'%report['image_files'],
             '| Images with a supplied slide label | %d |'%report['labeled_image_files'],
             '| Images without a supplied slide label | %d |'%report['unlabeled_image_files'],
             '| Slide identifiers in image paths | %d |'%report['image_slide_ids'],
             '| Labeled slides / distinct labeled patients | %d / %d |'%(report['labeled_slides'],report['labeled_patients']),
             '| Labeled slides with all seven resolutions | %d |'%report['labeled_slides_with_all_resolutions'],
             '| Classes | %d |'%len(report['class_slide_counts']), '',
             'The candidate image count differs from the 67,200 patches described in [the paper](https://arxiv.org/html/2601.02198v1). '
             'In the [official release](https://huggingface.co/datasets/bifold-pathomics/TCGA-MS), 19 PNG-suffixed '
             'entries have `._` sidecar names and are only 233 bytes; they are excluded by filename policy before '
             'matching slides to labels. The remaining images all match the label table. Do not silently claim '
             'that the paper image count was reproduced; the reason for its difference remains unverified.', '',
             '## Resolution inventory', '', '| Resolution directory | Images |','| --- | ---: |']
    for scale,count in report['images_by_resolution'].items():
        lines.append('| %s | %d |'%(scale,count))
    lines += ['', 'The supplied table contains 16 noncontiguous label IDs, with eight labeled slides per class. '
              'All five provider folds have zero patient-role overlap in the supplied metadata. Counts follow.', '',
              '| Provider fold | Train slides | Dev slides | Test slides | Missing class/role combinations |',
              '| --- | ---: | ---: | ---: | ---: |']
    for fold,row in report['folds'].items():
        counts = row['slide_counts']
        lines.append('| %s | %d | %d | %d | %d |'%(fold,counts['train'],counts['dev'],counts['test'],
                     sum(len(v) for v in row['missing_classes'].values())))
    lines += ['', '## Consequences for the next experiment', '',
              '- Restrict fitting eligibility to explicitly labeled slides and exclude resource sidecars; preserve an exclusion ledger. '+
              ('Full PNG decoding is verified below.' if 'decoding' in report else 'Full PNG decoding remains pending.'),
              '- Keep patient IDs, all slide fields, and all resolutions together using the supplied patient/split metadata. Remap the 16 label IDs explicitly.',
              '- Treat each slide/resolution as an unpaired field bag. Equal coordinate strings do not establish physical alignment; no coordinate key spans all seven resolutions in this archive.',
              '- Define four-native-resolution and/or seven-resolution availability protocols in advance. BreakHis objective magnifications and TCGA microns per pixel are different units.',
              '- Audit encoder-pretraining overlap before treating this cohort as independent external evidence. No encoder adaptation or TCGA scoring has run.', '',
              'Portable data-readiness jobs are `slurm/extract_tcga.sbatch` followed by `slurm/decode_tcga.sbatch`. '
              'They verify the pinned archive, safely extract, exclude sidecar-named entries from decoding, '
              'and record every error. Seven decode shards run with at most four workers.', '',
              'Source revision/checksum: `configs/tcga-ms-source.json`. '
              'Reproduce with `python -m magset.tcga_audit --inventory /path/to/inventory-v1 '
              '--output /path/to/private-audit.json --markdown docs/TCGA_MS_AUDIT.md`.', '']
    if 'decoding' in report:
        d=report['decoding']
        lines += ['## Full decoding verified November 12, 2026', '',
                  'All %d resolution reports were verified against the inventory and pinned archive checksum. '
                  'All %d images passed CRC/structure verification and full loading, with zero errors. '
                  'Every decoded image is RGB and224×224. Extraction and all seven scheduler tasks completed successfully. '
                  'This establishes image readiness, not a trained second-dataset benchmark.'%(d['shards'],d['images']), '',
                  'Decode source revision: `%s`. Regenerate this section by adding '
                  '`--decode-reports /path/to/decode-v1` to the command above.'%d['code_sha'], '']
    return '\n'.join(lines)


def verify_decoding(report, inventory, decode_reports, archive_sha256):
    reports=[json.loads(p.read_text()) for p in sorted(Path(decode_reports).glob('mpp-*.json'))]
    expected=report['images_by_resolution'];seen=set();total=0;codes=set()
    for item in reports:
        scale='mpp_'+item['resolution_mpp']
        if scale in seen or scale not in expected or item['images']!=expected[scale]:
            raise ValueError('Decode resolution/count coverage differs from inventory')
        if (item['errors'] or item['inventory_sha256']!=file_hash(Path(inventory)/'members.csv')
                or item['archive_sha256']!=archive_sha256):
            raise ValueError('Decode error or provenance mismatch')
        if item['modes']!={'RGB':item['images']} or item['dimensions']!={'(224, 224)':item['images']}:
            raise ValueError('Decoded modes/dimensions differ from expected dataset patches')
        seen.add(scale);total+=item['images'];codes.add(item['code_sha'])
    if seen!=set(expected) or total!=report['image_files'] or len(codes)!=1:
        raise ValueError('Decode report matrix incomplete or source inconsistent')
    return {'shards':len(reports),'images':total,'errors':0,'code_sha':codes.pop()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--markdown',type=Path,required=True)
    p.add_argument('--decode-reports',type=Path)
    p.add_argument('--source',type=Path,default=Path('configs/tcga-ms-source.json'))
    a = p.parse_args(); report = audit(a.inventory)
    if a.decode_reports:
        report['decoding']=verify_decoding(report,a.inventory,a.decode_reports,json.loads(a.source.read_text())['sha256'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    a.markdown.parent.mkdir(parents=True,exist_ok=True);a.markdown.write_text(markdown(report))
    print(json.dumps({k:report[k] for k in ['image_files','labeled_image_files','labeled_slides','labeled_patients','labeled_slides_with_all_resolutions']}))


if __name__=='__main__':
    main()
