"""Independent CPU replay of the registered matched three-way comparison."""
import argparse
from itertools import combinations
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magset.five_collect import assert_public_safe, read_json, require_equal

MAGNIFICATIONS = (40, 100, 200, 400)
SUBSETS = tuple(subset for size in range(1, 5) for subset in combinations(range(4), size))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integers(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in 'iu', name+': integer shape/dtype differs')
    return value


def probabilities(value, shape, name, tolerance=1e-10):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind == 'f' and np.isfinite(value).all()
            and np.all((value >= 0) & (value <= 1))
            and np.allclose(value.sum(-1), 1, rtol=0, atol=tolerance), name+': invalid normalized probabilities')
    return value.astype(np.float64)


def field_views(joint5, field_cases, field_magnifications, cases):
    """One equal-field mean per case/view; never assume matched field indices."""
    cases = integers(cases, (len(cases),), 'requested cases')
    require(len(cases) > 0 and len(set(cases.tolist())) == len(cases), 'Requested cases duplicate or empty')
    joint = probabilities(joint5, (len(joint5), 5), 'field joint five probabilities')
    owners = integers(field_cases, (len(joint),), 'field case membership')
    mags = integers(field_magnifications, (len(joint),), 'field magnification membership')
    require(set(owners) == set(cases) and set(mags) <= set(MAGNIFICATIONS), 'Field case/view coverage differs')
    views = np.empty((len(cases), 4, 5), dtype=np.float64)
    for row, case in enumerate(cases):
        for view, mag in enumerate(MAGNIFICATIONS):
            selected = (owners == case) & (mags == mag)
            require(selected.any(), 'Expected every case to have all four nonempty views')
            views[row, view] = joint[selected].mean(0)
    return views


def pool_subsets(views):
    """Equal active-view means of joint probabilities, then the fixed cascade."""
    views = probabilities(views, (len(views), 4, 5), 'case view probabilities')
    pooled = np.stack([views[:, subset, :].mean(1) for subset in SUBSETS], axis=1)
    pooled /= pooled.sum(-1, keepdims=True)
    malignant = pooled[:, :, 1:].sum(-1)
    predictions = np.where(malignant >= .5, 1+pooled[:, :, 1:].argmax(-1), 0)
    return pooled, predictions, malignant


def classification(truth, prediction, classes, weights=None):
    truth = integers(truth, (len(truth),), 'metric truth')
    prediction = integers(prediction, truth.shape, 'metric prediction')
    require(len(truth) > 0 and np.all((truth >= 0) & (truth < classes))
            and np.all((prediction >= 0) & (prediction < classes)), 'Metric label range differs')
    weights = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=np.float64)
    require(weights.shape == truth.shape and np.isfinite(weights).all() and np.all(weights >= 0)
            and weights.sum() > 0, 'Invalid metric weights')
    matrix = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(matrix, (truth, prediction), weights)
    support, called, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    f1 = np.divide(2*correct, support+called, out=np.zeros(classes), where=support+called > 0)
    recall = np.divide(correct, support, out=np.zeros(classes), where=support > 0)
    return {'records': len(truth), 'weight_sum': float(weights.sum()), 'correct': float(correct.sum()),
            'accuracy': float(correct.sum()/weights.sum()), 'macro_f1': float(f1.mean()),
            'macro_recall': float(recall.mean()), 'confusion_matrix': matrix.tolist(),
            'class_support': support.tolist(), 'class_correct': correct.tolist(), 'class_recall': recall.tolist()}


def single_subset_summary(truth, prediction, weights=None):
    five = classification(truth, prediction, 5, weights)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    binary = classification((truth > 0).astype(int), (prediction > 0).astype(int), 2, weights)
    weights = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=np.float64)
    malignant = truth > 0
    conditional = malignant & (prediction > 0)
    mal_mass = float(weights[malignant].sum())
    conditional_mass = float(weights[conditional].sum())
    correct = float(weights[malignant & (prediction == truth)].sum())
    return {'five_class': five, 'binary': binary,
            'end_to_end_malignant': {'correct': correct, 'denominator': mal_mass,
                                       'recall': correct/mal_mass if mal_mass else None},
            'conditional_malignant': {'correct': correct, 'denominator': conditional_mass,
                                      'accuracy': correct/conditional_mass if conditional_mass else 0.0}}


def case_subset_summary(truth, predictions, weights=None):
    truth = integers(truth, (len(truth),), 'case truth')
    predictions = integers(predictions, (len(truth), 15), 'case/subset predictions')
    subsets = {'+'.join(str(MAGNIFICATIONS[v]) for v in subset):
               single_subset_summary(truth, predictions[:, column], weights)
               for column, subset in enumerate(SUBSETS)}
    return {'cases': len(truth), 'case_subset_events': len(truth)*15,
            'mean_subset_macro_f1': float(np.mean([s['five_class']['macro_f1'] for s in subsets.values()])),
            'mean_subset_accuracy': float(np.mean([s['five_class']['accuracy'] for s in subsets.values()])),
            'worst_subset_macro_f1': float(min(s['five_class']['macro_f1'] for s in subsets.values())),
            'worst_subset_accuracy': float(min(s['five_class']['accuracy'] for s in subsets.values())),
            'all_four': subsets['40+100+200+400'], 'subsets': subsets}


def paired_group_bootstrap(truth, predictions, groups, *, seed=812, replicates=2000):
    """Resample groups, keeping every case and all masks paired across models."""
    truth = integers(truth, (len(truth),), 'bootstrap truth')
    groups = np.asarray(groups)
    require(groups.shape == truth.shape and groups.dtype.kind in 'US' and all(groups.tolist()),
            'Bootstrap requires nonempty conservative group labels')
    names = sorted(predictions)
    require(len(names) == 3, 'Paired bootstrap requires exactly three registered models')
    for name in names:
        integers(predictions[name], (len(truth), 15), 'bootstrap predictions/'+name)
    unique = sorted(set(groups.tolist()))
    require(len(unique) >= 2 and type(seed) is int and type(replicates) is int and replicates >= 2,
            'Insufficient group bootstrap units/budget')
    membership = np.asarray([unique.index(group) for group in groups])
    rng = np.random.RandomState(seed)
    draws = {name: np.empty((replicates, 2), dtype=np.float64) for name in names}
    missing_class = 0
    for draw in range(replicates):
        multiplicity = np.bincount(rng.randint(len(unique), size=len(unique)), minlength=len(unique))
        weights = multiplicity[membership].astype(np.float64)
        missing_class += int(np.any(np.bincount(truth, weights=weights, minlength=5) == 0))
        for name in names:
            scores = []
            for mask in range(15):
                matrix = np.zeros((5, 5), dtype=np.float64)
                np.add.at(matrix, (truth, predictions[name][:, mask]), weights)
                support, called, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
                f1 = np.divide(2*correct, support+called, out=np.zeros(5), where=support+called > 0)
                scores.append((float(f1.mean()), float(correct.sum()/weights.sum())))
            draws[name][draw] = np.mean(scores, axis=0)
    pairwise = {}
    for left, right in combinations(names, 2):
        difference = draws[left]-draws[right]
        pairwise[left+'_minus_'+right] = {
            metric: {'lower': float(np.percentile(difference[:, column], 2.5, method='linear')),
                     'upper': float(np.percentile(difference[:, column], 97.5, method='linear'))}
            for column, metric in enumerate(('mean_subset_macro_f1', 'mean_subset_accuracy'))}
    return {'unit': 'conservative group', 'groups': len(unique), 'cases': len(truth),
            'replicates': replicates, 'seed': seed, 'percentiles': [2.5, 97.5], 'percentile_method': 'linear',
            'draws_missing_any_truth_class': missing_class,
            'class_policy': 'fixed five labels, including absent truth classes', 'pairwise': pairwise,
            'interpretation': 'paired exploratory group bootstrap; all masks and cases within each group stay together; '
                              'small reused development holdout, no independent clinical cohort or multiplicity correction'}


def reconstruct_roles(cases, *, require_version=True):
    """Independent labels-only reconstruction, including deterministic tie codes."""
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold
    if require_version:
        require(sklearn.__version__ == '1.6.1', 'Pinned sklearn split version differs')
    labels = np.asarray([case['truth5'] for case in cases], dtype=np.int64)
    groups = np.asarray([case['group'] for case in cases])
    unique = sorted(set(groups.tolist()))
    permutation = np.random.RandomState(812).permutation(len(unique))
    group_codes = {group: int(code) for group, code in zip(unique, permutation)}
    codes = np.asarray([group_codes[group] for group in groups])
    fold = np.full(len(cases), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=False)
    for number, (_, held) in enumerate(splitter.split(np.zeros((len(cases), 1)), labels, codes)):
        require(np.all(fold[held] == -1), 'Duplicate split fold membership')
        fold[held] = number
    require(np.all(fold >= 0), 'Incomplete split fold coverage')
    for test in range(5):
        for validation in range(5):
            if validation == test:
                continue
            roles = {'train': np.flatnonzero((fold != test) & (fold != validation)),
                     'validation': np.flatnonzero(fold == validation), 'test': np.flatnonzero(fold == test)}
            if any(np.any(np.bincount(labels[ix], minlength=5) == 0) for ix in roles.values()):
                continue
            if any(len(set(groups[roles['train']][labels[roles['train']] == label])) < 2 for label in range(5)):
                continue
            for left, right in combinations(roles, 2):
                require(not set(groups[roles[left]]) & set(groups[roles[right]]), 'Conservative groups cross roles')
            return roles, fold, {'test_fold': test, 'validation_fold': validation,
                                 'group_code_permutation': permutation.tolist()}
    raise ValueError('No registered class-complete role pair; no alternative seed permitted')


def checked_hashes(folder, hashes, required=()):
    import re
    from magset.data import file_hash
    require(isinstance(hashes, dict) and set(required) <= set(hashes), 'Required artifact receipts missing')
    for name, digest in hashes.items():
        path = Path(name)
        require(isinstance(name, str) and name and not path.is_absolute() and '..' not in path.parts
                and isinstance(digest, str) and re.fullmatch('[a-f0-9]{64}', digest) is not None,
                'Invalid artifact hash receipt')
        require_equal(file_hash(Path(folder)/path), digest, 'artifact/'+name, 0)
    return hashes


def read_records(path):
    import json
    # Reuse strict duplicate-key/nonfinite JSON parsing, one record at a time.
    from magset.five_collect import _unique_object, _invalid_constant
    return [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            for line in Path(path).read_text().splitlines() if line.strip()]


def replay_field_archive(path, audit, role):
    """Replay every declared role field and every case/mask, with private IDs."""
    from scipy.special import logsumexp
    cases, fields = audit['cases'], audit['fields']
    role_cases = np.asarray(audit['roles'][role], dtype=np.int64)
    wanted = np.asarray([f['index'] for f in fields if f['case_index'] in set(role_cases)], dtype=np.int64)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'field_indices', 'joint_probabilities5'}, 'Field archive schema differs')
        indices = saved['field_indices'].copy()
        joint = saved['joint_probabilities5'].copy()
    require(indices.dtype == np.dtype('int64') and np.array_equal(indices, wanted), 'Role field coverage/order differs')
    require(joint.dtype == np.dtype('float64'), 'Canonical joint probabilities must be float64')
    joint = probabilities(joint, (len(wanted), 5), 'role field probabilities', tolerance=1e-10)
    owners = np.asarray([fields[index]['case_index'] for index in indices], dtype=np.int64)
    mags = np.asarray([MAGNIFICATIONS[fields[index]['view']] for index in indices], dtype=np.int64)
    views = field_views(joint, owners, mags, role_cases)
    pooled, predictions, scores = pool_subsets(views)
    records = []
    for row, case in enumerate(role_cases):
        for column, subset in enumerate(SUBSETS):
            probability = pooled[row, column]
            stable = np.maximum(probability, np.finfo(np.float64).tiny)
            stable /= stable.sum()
            log5 = np.log(stable)
            log_malignant = logsumexp(log5[1:])
            log_conditional = log5[1:]-log_malignant
            records.append({'role': role, 'case_id': cases[case]['case_id'], 'truth': cases[case]['truth8'],
                            'truth5': cases[case]['truth5'], 'prediction': int(predictions[row, column]),
                            'magnifications': [MAGNIFICATIONS[v] for v in subset],
                            'probabilities5': stable.tolist(), 'log_probabilities5': log5.tolist(),
                            'binary_score': float(scores[row, column]),
                            'binary_log_probabilities': [float(log5[0]), float(log_malignant)],
                            'malignant_probabilities4': np.exp(log_conditional).tolist(),
                            'malignant_log_probabilities': log_conditional.tolist()})
    # Metamorphic checks act on saved probabilities only: no encoder/model work.
    permuted = np.arange(len(indices))[::-1]
    reverse_views = field_views(joint[permuted], owners[permuted], mags[permuted], role_cases)
    require(np.allclose(reverse_views, views, rtol=0, atol=1e-12), 'Field permutation invariance failed')
    for column, subset in enumerate(SUBSETS):
        changed = views.copy()
        for inactive in set(range(4))-set(subset):
            changed[:, inactive] = np.array([1., 0, 0, 0, 0])
        changed_pooled, changed_predictions, _ = pool_subsets(changed)
        require(np.array_equal(changed_pooled[:, column], pooled[:, column])
                and np.array_equal(changed_predictions[:, column], predictions[:, column]),
                'Missing-view invariance failed')
    truth = np.asarray([cases[index]['truth5'] for index in role_cases], dtype=np.int64)
    return records, case_subset_summary(truth, predictions), predictions


def source_context(args):
    import re
    from magset.data import file_hash
    from magset import matched_data
    config = matched_data.config(args.config)
    context = matched_data.context(args.code_sha, args.config)
    require(re.fullmatch('[a-f0-9]{64}', context['source_archive_sha256']) is not None,
            'Immutable source archive digest missing')
    require(context['config_sha256'] == file_hash(args.config), 'Source configuration binding differs')
    require(config['models'] == ['midnight_p', 'cnn', 'swin'] and config['split']['seed'] == 812
            and config['split']['folds'] == 5 and config['split']['shuffle'] is False
            and config['training']['epochs'] == 40 and config['selection']['opportunities_per_model'] == 4,
            'Registered three-way scope differs')
    return context, config


def accepted_audit(args, context, config):
    import re
    from magset.data import CLASS_NAMES, file_hash
    from magset.five_metrics import MAP8_TO5
    folder = Path(args.audit)
    private, public = read_json(folder/'private.json'), read_json(folder/'public.json')
    require_equal(public.get('private_audit_sha256'), file_hash(folder/'private.json'), 'private audit binding', 0)
    require_equal(private.get('config_sha256'), context['config_sha256'], 'audit/configuration', 0)
    require_equal(private.get('manifest_sha256'), config['manifest_sha256'], 'audit/canonical manifest', 0)
    require_equal(private['implementation_sha256']['magset/matched_data.py'],
                  context['implementation_sha256']['magset/matched_data.py'], 'audit/data implementation', 0)
    require_equal(private['indices_sha256'], file_hash(folder/'indices.npz'), 'audit/indices bytes', 0)
    require(private.get('status') == 'accepted matched grouping and input audit'
            and private.get('new_fits') == 0 and private.get('new_predictions') == 0, 'Input audit status differs')
    require_equal(private['training_weights_sha256'], file_hash(folder/'training-weights.npz'), 'audit/training weights bytes', 0)
    cases, fields = private['cases'], private['fields']
    require(len(cases) == 82 and len(fields) == 7909, 'Expected full82-case/7909-image dataset')
    require([c['case_id'] for c in cases] == sorted({c['case_id'] for c in cases})
            and len({c['group'] for c in cases}) == 70, 'Canonical case ordering/group coverage differs')
    require([f['relative_path'] for f in fields] == sorted({f['relative_path'] for f in fields}),
            'Canonical field ordering/uniqueness differs')
    lookup = {case['case_id']: i for i, case in enumerate(cases)}
    for case in cases:
        require(type(case['truth8']) is int and case['truth8'] in range(8)
                and type(case['truth5']) is int and case['truth5'] == MAP8_TO5[case['truth8']]
                and isinstance(case['group'], str) and case['group'], 'Audited case truth/group differs')
    view_coverage = [set() for _ in cases]
    for index, field in enumerate(fields):
        label = CLASS_NAMES.index(field['subtype'])
        key = field['filename'].split('-')[0]+'-'+field['case_id_raw']
        require(key in lookup, 'Unknown field provider identity')
        case = cases[lookup[key]]
        view = MAGNIFICATIONS.index(int(field['magnification']))
        require(field['index'] == index and field['case_index'] == lookup[key] and field['view'] == view
                and case['truth8'] == label and case['group'] == field['patient_group_conservative']
                and re.fullmatch('[a-f0-9]{64}', field['sha256']) is not None,
                'Canonical field identity/truth/view/digest differs')
        view_coverage[lookup[key]].add(view)
    require(all(views == set(range(4)) for views in view_coverage), 'All cases need all four views')
    roles, folds, choice = reconstruct_roles(cases)
    require_equal(private['roles'], {r: ix.tolist() for r, ix in roles.items()}, 'independent role membership', 0)
    require_equal(private['fold_assignment'], folds.tolist(), 'independent group folds', 0)
    require_equal(private['split_choice'], choice, 'first valid role-pair selection', 0)
    expected_arrays = {'field_case_index': np.asarray([f['case_index'] for f in fields]),
                       'field_view': np.asarray([f['view'] for f in fields]),
                       'case_y5': np.asarray([c['truth5'] for c in cases]),
                       'case_y8': np.asarray([c['truth8'] for c in cases]), 'case_fold': folds,
                       **{role+'_cases': indices for role, indices in roles.items()}}
    with np.load(folder/'indices.npz', allow_pickle=False) as saved:
        require(set(saved.files) == set(expected_arrays), 'Audit split archive schema differs')
        for key, wanted in expected_arrays.items():
            actual = integers(saved[key], wanted.shape, 'audit/'+key)
            require(np.array_equal(actual, wanted), 'Audit array differs: '+key)
    training_fields = [field for field in fields if field['case_index'] in set(roles['train'])]
    counts_by_view = {}
    for field in training_fields:
        key = field['case_index'], field['view']
        counts_by_view[key] = counts_by_view.get(key, 0)+1
    fitting_truth = np.asarray([cases[i]['truth5'] for i in roles['train']])
    binary_counts = np.bincount((fitting_truth > 0).astype(int), minlength=2)
    malignant_counts = np.bincount(fitting_truth[fitting_truth > 0]-1, minlength=4)
    expected_weights = {'binary': np.zeros(len(fields)), 'malignant': np.zeros(len(fields))}
    for field in training_fields:
        label = cases[field['case_index']]['truth5']
        denominator = 4*counts_by_view[field['case_index'], field['view']]
        expected_weights['binary'][field['index']] = len(training_fields)/(2*binary_counts[int(label > 0)]*denominator)
        if label > 0:
            expected_weights['malignant'][field['index']] = len(training_fields)/(4*malignant_counts[label-1]*denominator)
    with np.load(folder/'training-weights.npz', allow_pickle=False) as saved:
        require(set(saved.files) == set(expected_weights), 'Training weight schema differs')
        for name, wanted in expected_weights.items():
            require(saved[name].dtype == np.dtype('float64') and np.allclose(saved[name], wanted, rtol=0, atol=1e-12),
                    'Independent fitting-only sample weights differ: '+name)
    counts = {}
    for role, indices in roles.items():
        selected = set(indices.tolist())
        counts[role] = {'cases': len(indices), 'groups': len({cases[i]['group'] for i in indices}),
                        'class_cases': np.bincount([cases[i]['truth5'] for i in indices], minlength=5).tolist(),
                        'fields': sum(field['case_index'] in selected for field in fields)}
    require_equal({r: [v['cases'], v['groups'], v['fields']] for r, v in counts.items()},
                  {'train': [49, 42, 4621], 'validation': [17, 15, 1584], 'test': [16, 13, 1704]},
                  'Registered shared role counts', 0)
    require_equal({role: value['class_cases'] for role, value in counts.items()},
                  {role: value['class_cases'] for role, value in config['expected_roles'].items()},
                  'Registered shared final-target supports', 0)
    return private, counts


def verify_context(receipt, context, audit_sha, model=None):
    for key, expected in context.items():
        require_equal(receipt.get(key), expected, 'receipt/context/'+key, 0)
    require_equal(receipt.get('input_audit_sha256'), audit_sha, 'receipt/input audit', 0)
    if model is not None:
        require_equal(receipt.get('model'), model, 'receipt/model', 0)


def verify_reported_summary(actual, replay):
    """Accept producer classification claims only after independent record replay."""
    for key in ('mean_subset_macro_f1', 'mean_subset_accuracy', 'worst_subset_macro_f1'):
        require_equal(actual.get(key), replay[key], 'independent summary/'+key, 2e-12)
    require_equal(actual.get('distinct_cases'), replay['cases'], 'distinct case denominator', 0)
    require(set(actual.get('subsets', {})) == set(replay['subsets']), 'Reported subset coverage differs')
    for name, wanted in replay['subsets'].items():
        reported = actual['subsets'][name]
        for key in ('records', 'accuracy', 'macro_f1', 'confusion_matrix', 'class_support', 'class_recall'):
            require_equal(reported.get(key), wanted['five_class'][key], 'independent subset/'+name+'/'+key, 2e-12)
        binary = actual['binary_summary']['subsets'][name]
        require_equal(binary.get('accuracy'), wanted['binary']['accuracy'], 'independent binary/'+name, 2e-12)
    require_equal(actual['all_view'], actual['subsets']['40+100+200+400'], 'reported full-view alias', 0)


def verify_prediction_set(folder, prefix, audit, role, reported):
    records, summary, predictions = replay_field_archive(Path(folder)/(prefix+'fields.npz'), audit, role)
    require_equal(read_records(Path(folder)/(prefix+'records.jsonl')), records,
                  'independent field-to-case records/'+role, 1e-10)
    verify_reported_summary(reported, summary)
    return summary, predictions


def verify_history(path, config, model):
    value = read_json(path)
    require_equal(value.get('candidate_epochs'), [10, 20, 30, 40], 'fixed epoch candidates', 0)
    require_equal(value.get('field_batches_per_epoch'), 289, 'fitting field batches', 0)
    rows = value.get('epochs')
    require(isinstance(rows, list) and len(rows) == 40, 'Incomplete forty-epoch neural trajectory')
    initial = None
    for index, row in enumerate(rows):
        require_equal(row.get('epoch'), index+1, 'history epoch order', 0)
        require_equal(row.get('optimizer_updates'), 289*(index+1), 'fixed optimizer updates', 0)
        for key in ('learning_rate', 'loss', 'binary_loss', 'malignant_loss'):
            require(isinstance(row.get(key), (float, int)) and not isinstance(row[key], bool)
                    and math.isfinite(row[key]) and row[key] >= 0, 'Nonfinite/negative neural history')
        if initial is None:
            initial = config['training']['learning_rates'][model]
        expected = initial*(.01+.99*.5*(1+math.cos(math.pi*index/39)))
        require_equal(row['learning_rate'], expected, 'fixed half-cosine learning-rate schedule', 1e-10)
    import csv
    with Path(path).with_name('progress.csv').open(newline='') as handle:
        progress = list(csv.DictReader(handle))
    require(len(progress) == 40, 'Neural CSV progress is incomplete')
    for actual, wanted in zip(progress, rows):
        require(set(actual) == set(wanted) and all(math.isclose(float(actual[key]), wanted[key], rel_tol=0, abs_tol=1e-12)
                                                  for key in wanted), 'Neural CSV progress/history differs')
    return {'epochs': 40, 'optimizer_updates': 40*289, 'candidate_epochs': [10, 20, 30, 40]}


def verify_pixels(args, context, audit_sha):
    from magset.data import file_hash
    folder = Path(args.root)/'pixels'
    receipt = read_json(folder/'receipt.json')
    require(receipt.get('status') == 'accepted common three-crop pixels'
            and receipt.get('new_fits') == 0 and receipt.get('new_predictions') == 0,
            'Common crop preparation did not pass')
    require_equal(receipt.get('config_sha256'), context['config_sha256'], 'pixels/configuration', 0)
    require_equal(receipt['implementation_sha256']['magset/matched_data.py'],
                  context['implementation_sha256']['magset/matched_data.py'], 'pixels/data implementation', 0)
    require_equal(receipt.get('input_audit_sha256'), audit_sha, 'pixels/input audit', 0)
    for key, wanted in {'images': 7909, 'dtype': 'uint8', 'shape': [7909, 3, 224, 224, 3]}.items():
        require_equal(receipt.get(key), wanted, 'common crop pixels/'+key, 0)
    digest = file_hash(folder/'pixels.npy')
    require_equal(receipt.get('pixels_sha256'), digest, 'common raw crop bytes', 0)
    pixels = np.load(folder/'pixels.npy', mmap_mode='r', allow_pickle=False)
    require(pixels.shape == (7909, 3, 224, 224, 3) and pixels.dtype == np.uint8,
            'Shared three-crop pixel cache shape/dtype differs')
    del pixels
    return {'input_pixels_receipt_sha256': file_hash(folder/'receipt.json'), 'pixels_sha256': digest}


def verify_head_archive(candidate, audit):
    with np.load(Path(candidate)/'validation-heads.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'field_indices', 'binary_probabilities2', 'malignant_probabilities4'},
                'Validation head archive schema differs')
        wanted = np.asarray([field['index'] for field in audit['fields']
                             if field['case_index'] in set(audit['roles']['validation'])], dtype=np.int64)
        require(saved['field_indices'].dtype == np.dtype('int64') and np.array_equal(saved['field_indices'], wanted),
                'Validation head field membership/order differs')
        binary = probabilities(saved['binary_probabilities2'], (len(wanted), 2), 'validation binary head', tolerance=1e-5)
        malignant = probabilities(saved['malignant_probabilities4'], (len(wanted), 4), 'validation malignant head', tolerance=1e-5)
    composed = np.column_stack((binary[:, 0], binary[:, 1, None]*malignant))
    composed /= composed.sum(1, keepdims=True)
    with np.load(Path(candidate)/'validation-fields.npz', allow_pickle=False) as saved:
        require(np.allclose(saved['joint_probabilities5'], composed, rtol=0, atol=1e-10),
                'Independent binary/malignant joint composition differs')


def review_selection(args, model, context, audit, audit_sha, config, pixels):
    from magset.data import file_hash
    folder = Path(args.root)/'models'/model
    selection = read_json(folder/'selection.json')
    verify_context(selection, context, audit_sha, model)
    require(selection.get('status') == 'selected validation candidate' and selection.get('test_predictions') == 0,
            'Validation selection was not sealed before test prediction')
    result = read_json(folder/'result.json')
    verify_context(result, context, audit_sha, model)
    require(result.get('status') == 'complete training', 'Incomplete model training/selection')
    required_training = {'selection.json'}
    if model != 'midnight_p':
        required_training |= {'history.json', 'progress.csv', 'architecture.json', 'environment.json', 'started.json'}
    checked_hashes(folder, result.get('artifact_sha256'), required=required_training)
    settings = config['midnight_p']['temperatures'] if model == 'midnight_p' else [10, 20, 30, 40]
    hashes = selection.get('candidate_artifact_sha256')
    require(isinstance(hashes, list) and len(hashes) == 4, 'Incomplete four-candidate selection matrix')
    scores, candidates = [], []
    for index, setting in enumerate(settings):
        candidate = folder/'candidates'/('candidate%d' % index)
        required = {'validation-fields.npz', 'validation-records.jsonl', 'metrics.json'}
        if model != 'midnight_p':
            required |= {'weights.weights.h5', 'validation-heads.npz'}
        require(set(hashes[index]) == required, 'Candidate artifact matrix differs')
        checked_hashes(candidate, hashes[index], required)
        if model != 'midnight_p':
            verify_head_archive(candidate, audit)
        metrics = read_json(candidate/'metrics.json')
        require_equal(metrics.get('setting'), setting, 'candidate setting', 0)
        summary, _ = verify_prediction_set(candidate, 'validation-', audit, 'validation', metrics['validation'])
        scores.append(summary['mean_subset_macro_f1'])
        candidates.append({'index': index, 'setting': setting, 'validation': summary})
    require_equal(selection.get('candidate_validation_scores'), scores, 'independent validation selection scores', 2e-12)
    best = int(np.argmax(scores))
    require_equal(selection.get('selected_index'), best, 'first maximum validation candidate', 0)
    require_equal(selection.get('selected_setting'), settings[best], 'selected setting', 0)
    if model == 'midnight_p':
        require_equal(selection.get('support_case_indices'), audit['roles']['train'], 'fit-only P support cases', 0)
        require(selection.get('selected_checkpoint_sha256') is None, 'P unexpectedly reused neural checkpoint')
        history = None
        architecture = None
    else:
        require_equal(selection.get('selected_checkpoint_sha256'), hashes[best]['weights.weights.h5'], 'selected neural checkpoint', 0)
        for key, digest in pixels.items():
            require_equal(result.get(key), digest, 'neural/shared crop binding/'+key, 0)
            require_equal(selection.get(key), digest, 'selection/shared crop binding/'+key, 0)
        started = read_json(folder/'started.json')
        verify_context(started, context, audit_sha, model)
        for key, digest in pixels.items():
            require_equal(started.get(key), digest, 'started/shared crop binding/'+key, 0)
        require(started.get('status') == 'training started' and started.get('test_predictions') == 0
                and result.get('test_predictions') == 0, 'Neural pre-gate test-access receipt differs')
        history = verify_history(folder/'history.json', config, model)
        architecture = verify_architecture(folder, model, config)
        verify_environment(folder/'environment.json', config)
        smoke_path = Path(args.root)/'technical'/model/'smoke.json'
        smoke = read_json(smoke_path)
        for key, wanted in context.items():
            require_equal(smoke.get(key), wanted, 'same-source technical smoke/'+key, 0)
        require(smoke.get('status') == 'accepted synthetic matched model smoke' and smoke.get('model') == model
                and smoke.get('full_batch_fields') == 16 and smoke.get('partial_batch_fields') == 5
                and smoke.get('synthetic_updates') == 2 and smoke.get('real_validation_predictions') == 0
                and smoke.get('real_test_predictions') == 0 and isinstance(smoke.get('weight_reload_max_abs'), (float, int))
                and math.isfinite(smoke['weight_reload_max_abs']) and 0 <= smoke['weight_reload_max_abs'] <= 1e-5,
                'Actual full/partial-batch synthetic model smoke did not pass')
    return selection, {'selection_sha256': file_hash(folder/'selection.json'), 'selected_index': best,
                       'selected_setting': settings[best], 'candidate_validation_scores': scores,
                       'selected_validation': candidates[best]['validation'], 'history': history, 'architecture': architecture}


def verify_gate(args, context, audit_sha, selections):
    from magset.data import file_hash
    gate = read_json(Path(args.root)/'test-gate.json')
    verify_context(gate, context, audit_sha)
    require(gate.get('status') == 'sealed all three validation selections' and gate.get('test_predictions') == 0,
            'All-three-model validation gate is incomplete')
    expected = {model: file_hash(Path(args.root)/'models'/model/'selection.json') for model in selections}
    require_equal(gate.get('selections'), expected, 'all-three selection hash gate', 0)
    return file_hash(Path(args.root)/'test-gate.json')


def review_test(args, model, context, audit, audit_sha, gate_sha, selection, config, pixels):
    from magset.data import file_hash
    folder = Path(args.root)/'models'/model/'test'
    receipt = read_json(folder/'test.json')
    verify_context(receipt, context, audit_sha, model)
    require(receipt.get('status') == 'complete selected test inference', 'Incomplete selected test inference')
    require_equal(receipt.get('selection_sha256'), file_hash(folder.parent/'selection.json'), 'test/selected validation receipt', 0)
    require_equal(receipt.get('gate_sha256'), gate_sha, 'test/all-three sealed gate', 0)
    required = {'fields.npz', 'records.jsonl', 'metrics.json'}
    if model != 'midnight_p':
        required |= {'checkpoint_reload.json', 'environment.json'}
    checked_hashes(folder, receipt.get('artifact_sha256'), required)
    summary, predictions = verify_prediction_set(folder, '', audit, 'test', read_json(folder/'metrics.json'))
    if model != 'midnight_p':
        for key, digest in pixels.items():
            require_equal(receipt.get(key), digest, 'test/shared crop binding/'+key, 0)
        verify_environment(folder/'environment.json', config)
        reload = read_json(folder/'checkpoint_reload.json')
        first = [f['index'] for f in audit['fields'] if f['case_index'] in set(audit['roles']['validation'])][:16]
        for key, wanted in {'field_indices': first, 'images': 16, 'atol': 1e-5, 'role': 'validation',
                            'checkpoint_sha256': selection['selected_checkpoint_sha256']}.items():
            require_equal(reload.get(key), wanted, 'validation checkpoint reload/'+key, 0)
        require(all(isinstance(reload.get(k), (float, int)) and not isinstance(reload[k], bool)
                    and math.isfinite(reload[k]) and 0 <= reload[k] <= 1e-5
                    for k in ('maximum_binary_abs', 'maximum_subtype_abs')), 'Selected checkpoint reload parity failed')
    return summary, predictions


def verify_environment(path, config):
    value = read_json(path)
    for name, version in config['runtime'].items():
        require_equal(value.get(name), version, 'pinned neural runtime/'+name, 0)
    require_equal(value.get('gpu_count'), 1, 'one-GPU neural execution', 0)
    require(isinstance(value.get('python'), str) and value['python'], 'Python runtime receipt missing')
    return value


def verify_architecture(folder, model, config):
    architecture = read_json(Path(folder)/'architecture.json')
    for key, wanted in {'kind': model, 'input_shape': [None, 3, 224, 224, 3],
                        'input_scale': 'raw RGB 0..255', 'outputs': {'binary_output': 2, 'subtype_output': 4},
                        'subtype_binary_stop_gradient': model == 'swin',
                        'descriptor_dimensions': 512 if model == 'cnn' else 768}.items():
        require_equal(architecture.get(key), wanted, 'matched architecture/'+key, 0)
    parameters = architecture.get('parameters', {})
    require(set(parameters) == {'all', 'trainable', 'nontrainable'}
            and all(type(value) is int and value > 0 for value in parameters.values())
            and parameters['all'] == parameters['trainable']+parameters['nontrainable'],
            'Actual architecture parameter receipt differs')
    if model == 'cnn':
        require_equal(architecture.get('pretrained_sha256'), config['weights']['cnn_sha256'], 'original CNN weights', 0)
        require_equal(architecture.get('transformer_blocks'), 4, 'original CNN transformer depth', 0)
        flags = architecture.get('backbone_layer_flags')
        require(isinstance(flags, list) and len(flags) >= 60
                and all(type(row['trainable']) is bool for row in flags)
                and [row['trainable'] for row in flags] == [False]*(len(flags)-60)+[True]*60,
                'CNN frozen/final60 layer flags differ')
    else:
        require_equal(architecture.get('pretrained_revision'), config['weights']['swin_revision'], 'original Swin revision', 0)
        require_equal(architecture.get('pretrained_files', {}).get('tf_model.h5'), config['weights']['swin_tf_sha256'], 'original native TF Swin weights', 0)
        require_equal(architecture.get('embeddings_trainable'), False, 'Swin frozen embeddings', 0)
        require_equal(architecture.get('stage_trainable'), [False, False, True, True], 'Swin stage trainability', 0)
        require_equal(architecture.get('integer_window_reverse'), True, 'Swin integer window compatibility', 0)
        loading = architecture.get('pretrained_loading', {})
        for key, wanted in {'verified_position_buffers': 12, 'unresolved_missing_weights': 0, 'mismatched_weights': 0}.items():
            require_equal(loading.get(key), wanted, 'Swin pretrained loading/'+key, 0)
        require(loading.get('missing_deterministic_buffers') in (0, 12), 'Swin deterministic buffer exemption differs')
    return {'parameters': parameters, 'descriptor_dimensions': architecture['descriptor_dimensions'],
            'subtype_binary_stop_gradient': architecture['subtype_binary_stop_gradient']}


def p_geometry_replay(args, audit, config, selected):
    """Replay frozen descriptors/support geometry; no raw-image encoder or fitting.

    The reviewed prior_centered_model remains a declared mathematical dependency.
    Field ordering, pooling, routing, metrics and selection are independent above.
    """
    from scipy.special import logsumexp
    from magset.bags import load_bags
    from magset.data import file_hash
    from magset import prior_centered_model as geometry
    root = Path(__file__).resolve().parents[1]
    coverage_path = root/config['midnight_p']['feature_config']
    require_equal(file_hash(coverage_path), config['midnight_p']['feature_config_sha256'], 'registered P coverage configuration', 0)
    require_equal(audit['coverage_config_sha256'], file_hash(coverage_path), 'audit/P coverage configuration', 0)
    coverage = read_json(coverage_path)
    cache = Path(audit['coverage_cache'])
    for mag in MAGNIFICATIONS:
        shard = cache/('midnight-coverage-v1-%d.npz' % mag)
        receipts = {'npz_sha256': shard, 'json_sha256': shard.with_suffix('.json'),
                    'components_sha256': shard.with_name(shard.stem+'-components.npz')}
        for key, path in receipts.items():
            digest = file_hash(path)
            require_equal(digest, audit['coverage_cache_sha256'][str(mag)][key], 'audit/P cache/'+key, 0)
            require_equal(digest, coverage['cache_provenance'][str(mag)][key], 'registered/P cache/'+key, 0)
    cases, bags, _ = load_bags(audit['fields'], cache, 'midnight-coverage-v1', config['manifest_sha256'])
    require_equal([case['case_id'] for case in cases], [case['case_id'] for case in audit['cases']], 'P cached provider order', 0)
    require_equal([case['label'] for case in cases], [case['truth8'] for case in audit['cases']], 'P cached original truth', 0)
    labels = np.asarray([case['truth5'] for case in audit['cases']], dtype=np.int64)
    support = np.asarray(audit['roles']['train'], dtype=np.int64)
    fields = audit['fields']
    folder = Path(args.root)/'models'/'midnight_p'
    maximum = 0.
    for role in ('validation', 'test'):
        settings = list(config['midnight_p']['temperatures']) if role == 'validation' else [selected['selected_setting']]
        saved = []
        for index, setting in enumerate(settings):
            path = (folder/'candidates'/('candidate%d' % index)/'validation-fields.npz') if role == 'validation' else folder/'test'/'fields.npz'
            with np.load(path, allow_pickle=False) as archive:
                saved.append({int(key): value for key, value in zip(archive['field_indices'], archive['joint_probabilities5'])})
        queries = audit['roles'][role]
        require(not set(queries) & set(support), 'P query/support roles overlap')
        for query in queries:
            prepared = geometry.prepare_case(bags, labels, support, query, leave_query_case_out=False)
            require_equal(list(prepared.support_indices), support.tolist(), 'P fitting-only support eligibility', 0)
            for candidate, setting in enumerate(settings):
                logs = geometry.field_log_probabilities(prepared, setting, 'dual', 'head_specific')
                for view, values in enumerate(logs):
                    indices = [f['index'] for f in fields if f['case_index'] == query and f['view'] == view]
                    expected = np.exp(values-logsumexp(values, axis=1, keepdims=True))
                    actual = np.asarray([saved[candidate][index] for index in indices])
                    require(expected.shape == actual.shape and np.isfinite(expected).all(), 'P geometry replay field coverage differs')
                    difference = float(np.max(np.abs(expected-actual)))
                    maximum = max(maximum, difference)
                    require(difference <= 1e-10, 'P fitting-only centered KDE replay differs')
    return {'status': 'accepted fit-only cached-descriptor geometry replay',
            'mathematical_dependency': 'reviewed prior_centered_model', 'raw_encoder_forwards': 0,
            'support_cases': len(support), 'maximum_field_probability_abs': maximum}


def collect(args):
    import traceback
    from magset.data import file_hash
    from magset.five_class import write_json
    output = Path(args.output)
    require(not output.exists(), 'Refusing existing three-way collector output')
    result = {'status': 'failed or incomplete matched three-way comparison', 'collector_new_fits': 0,
              'collector_new_image_inference': 0,
              'interpretation': 'Matched reused development comparison; conservative groups are kept together, '
              'all fifteen masks within a case remain dependent, and no globally fresh clinical test is claimed. '
              'Pretrained readers and learning algorithms differ; four validation choices do not equalize compute.'}
    private = {'errors': {}}
    try:
        context, config = source_context(args)
        audit, counts = accepted_audit(args, context, config)
        audit_sha = file_hash(Path(args.audit)/'private.json')
        pixels = verify_pixels(args, context, audit_sha)
        selections, models = {}, {}
        for model in config['models']:
            selections[model], models[model] = review_selection(args, model, context, audit, audit_sha, config, pixels)
        gate_sha = verify_gate(args, context, audit_sha, selections)
        predictions = {}
        for model in config['models']:
            summary, predictions[model] = review_test(args, model, context, audit, audit_sha, gate_sha, selections[model], config, pixels)
            models[model]['test'] = summary
        p_geometry = p_geometry_replay(args, audit, config, selections['midnight_p'])
        test_cases = audit['roles']['test']
        truth = np.asarray([audit['cases'][i]['truth5'] for i in test_cases], dtype=np.int64)
        groups = [audit['cases'][i]['group'] for i in test_cases]
        bootstrap = paired_group_bootstrap(truth, predictions, groups, seed=config['budget'].get('bootstrap_seed', 812),
                                            replicates=config['budget']['bootstrap_replicates'])
        result.update(context, status='complete accepted matched three-way comparison', input_audit_sha256=audit_sha,
                      test_gate_sha256=gate_sha, shared_pixels=pixels, roles=counts, models=models, p_geometry_replay=p_geometry, paired_group_bootstrap=bootstrap)
    except Exception as error:
        private['errors']['review'] = {'message': str(error), 'traceback': traceback.format_exc()}
        result['review'] = {'status': 'invalid_or_incomplete', 'error_type': type(error).__name__}
    assert_public_safe(result)
    output.mkdir(parents=True)
    write_json(output/'private_review.json', private)
    result['private_review_sha256'] = file_hash(output/'private_review.json')
    write_json(output/'result.json', result)
    lines = ['# Matched three-way comparison', '', 'Status: '+result['status']+'.', '']
    if 'models' in result:
        for model, value in result['models'].items():
            summary = value['test']
            lines.append(f"{model}: mean15 macro F1 {summary['mean_subset_macro_f1']:.6f}; "
                         f"mean15 accuracy {100*summary['mean_subset_accuracy']:.2f}%; "
                         f"all-four accuracy {100*summary['all_four']['five_class']['accuracy']:.2f}%.")
    else:
        lines.append('No model performance result is accepted.')
    lines += ['', result['interpretation'], '']
    (output/'summary.md').write_text('\n'.join(lines))
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'audit', 'config', 'output'):
        value.add_argument('--'+name, type=Path, required=True)
    value.add_argument('--code-sha', required=True)
    return value


if __name__ == '__main__':
    print(collect(parser().parse_args())['status'])
