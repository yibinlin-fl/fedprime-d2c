"""CPU-only audit: frozen Axis-I configs, legacy JTT and normalization.

Creates evidence about readiness; never trains, edits an experiment or uploads.
Run with the project's existing PyTorch environment. Output is a dated snapshot.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fedprime.methods.spurious_baselines import jtt_reweighted_loss
from scripts import run_cle_v2_spurious_baselines as screen
from scripts.run_cle_jtt_formal import jtt_config
from scripts.run_cle_v2_spurious_final import final_arm_config

OUT = Path(__file__).resolve().parent
BASES = [
    ROOT / 'outputs/openi_downloads/cle_v2_spurious_final_map2_formal/extracted',
    ROOT / 'outputs/openi_downloads/m1_map2_multiseed/extracted_seed1',
    ROOT / 'outputs/openi_downloads/m1_map2_multiseed/extracted_seed2',
]
ARMS = ['erm', 'cvar_dro', 'pew_groupdro', 'pew_ber']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, prefix+'.'+key if prefix else key))
        return result
    return {prefix: value}


def compare(a, b):
    aa, bb = flatten(a), flatten(b)
    aa = {key: value.replace('\\', '/') if isinstance(value, str) else value for key, value in aa.items()}
    bb = {key: value.replace('\\', '/') if isinstance(value, str) else value for key, value in bb.items()}
    return [{'field': key, 'frozen': aa.get(key, '<missing>'), 'candidate': bb.get(key, '<missing>')}
            for key in sorted(aa.keys() | bb.keys()) if aa.get(key) != bb.get(key)]


def relocate(config, package, output):
    result = copy.deepcopy(config)
    old = result['data']['private_root'].rsplit('/data/', 1)[0]
    for container, key in [(result['data'], 'private_root'), (result['data'], 'public_root'),
                           (result['method']['strict_fit_audit'], 'split_path'),
                           (result['checkpoints'], 'load_dir')]:
        container[key] = container[key].replace(old, package.as_posix(), 1)
    result['output_root'] = str(output)
    result['device'] = 'cpu'
    return result


def main():
    target = OUT / 'AUDIT_EVIDENCE.json'
    if target.exists():
        raise FileExistsError('Preserving dated evidence; use a new audit revision')
    package = ROOT / 'local_runs/audit_only_map2_package_not_a_training_input'
    output = ROOT / 'local_runs/jtt_fairness_audit_20261104'
    output.mkdir(parents=True, exist_ok=True)
    configs, sources = {}, []
    for seed, base in enumerate(BASES):
        for arm in ARMS:
            path = base / f'configs/map2_{arm}_trainseed{seed}.json'
            config = json.loads(path.read_text(encoding='utf-8'))
            assert config['seed'] == seed
            assert config['train']['rounds'] == 40
            assert config['train']['max_local_batches'] == 16
            assert config['train']['batch_size'] == 64
            assert config['method']['communication'] == 'asymhfl_val'
            assert config['method']['strict_fit_audit']['loader_seed'] == 20260909+seed
            assert config['data']['scenario_id'] == 'cle_hfl_v2_cross_map2_seed0_split0'
            assert config['method']['local_loader_mode'] == 'standard'
            assert config['method']['lambda_jsd'] == 0
            configs[(seed, arm)] = config
            sources.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
    frozen = relocate(configs[(0, 'erm')], package, output)
    current = final_arm_config('erm', package_root=package, mode='formal', device='cpu',
                               output_root=output, train_seed=0)
    current['experiment_name'] = frozen['experiment_name']
    legacy = jtt_config('erm', package_root=package, mode='formal', output_root=output,
                        config_root=output/'configs', device='cpu')
    legacy['experiment_name'] = frozen['experiment_name']
    legacy_jtt = jtt_config('jtt', package_root=package, mode='formal', output_root=output,
                            config_root=output/'configs', device='cpu')
    assert Path(legacy_jtt['checkpoints']['load_dir']).resolve() == Path(frozen['checkpoints']['load_dir']).resolve()
    assert legacy_jtt['checkpoints']['require_all'] and legacy_jtt['checkpoints']['strict']

    # Full-data repeated-example risk differs from a mean of batch ratios.
    losses = torch.tensor([1., 4., 4., 4.], dtype=torch.float64)
    mask = torch.tensor([True, False, False, False])
    weights = torch.where(mask, 20., 1.).to(torch.float64)
    global_risk = (weights*losses).sum()/weights.sum()
    pairs = list(itertools.combinations(range(4), 2))
    minibatch_risks = [jtt_reweighted_loss(losses[list(pair)], mask[list(pair)], upweight=20.)
                       for pair in pairs]
    expected_batch_ratio = torch.stack(minibatch_risks).mean()
    global_normalizer = weights.mean()
    fixed_normalized = torch.stack([(weights[list(pair)]*losses[list(pair)]).mean()/global_normalizer
                                    for pair in pairs]).mean()
    torch.testing.assert_close(fixed_normalized, global_risk)
    assert abs(float(expected_batch_ratio-global_risk)) > 1.

    # Exercise the actual mask producer with non-contiguous fit indices; zero training.
    split = output / 'toy_split.npz'
    np.savez(split, client_0_fit=np.array([4, 1, 2]), client_0_audit=np.array([0, 3, 5]))
    visited = []
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs):
            assert kwargs['train'] is True and kwargs['return_corruption'] is False
        def __len__(self):
            return 6
        def __getitem__(self, index):
            visited.append(index)
            return torch.tensor([float(index)]), index % 2
    class ToyModel(torch.nn.Module):
        def forward(self, images):
            idx = images[:, 0].long()
            labels = idx % 2
            predicted = torch.where((idx == 1) | (idx == 5), 1-labels, labels)
            return torch.nn.functional.one_hot(predicted, 2).float()
    toy_config = {'data': {'private_root': str(output), 'num_classes': 2},
                  'models': {'names': ['toy']},
                  'method': {'strict_fit_audit': {'split_path': str(split)}}}
    with patch.object(screen, 'build_models', lambda *a, **kw: {0: ToyModel()}), \
         patch.object(screen, 'load_state', lambda path: {}), \
         patch.object(screen, '_prepared_private_dataset_name', lambda path: 'cifar10'), \
         patch.object(screen, '_private_test_transform', lambda name: None), \
         patch.object(screen, 'CorruptionSkewClientDataset', ToyDataset), \
         patch.object(screen, 'forward_logits', lambda model, images: model(images)):
        produced = screen.generate_jtt_error_masks(erm_config=toy_config, erm_checkpoint_root=output,
                    destination=output/'toy_masks', device=torch.device('cpu'), batch_size=2)
    actual_mask = np.load(output/'toy_masks/client_0_error_mask.npy')
    assert visited == [4, 1, 2]
    assert np.array_equal(actual_mask, [0, 1, 0, 0, 0, 0])
    traces = []
    for seed, base in enumerate(BASES):
        hashes = []
        for arm in ARMS:
            path = base / f'outputs/cle_v2_spurious_final_map2_{arm}_trainseed{seed}/local_batch_trace.jsonl'
            hashes.append(sha(path))
        assert len(set(hashes)) == 1, 'Frozen four arms no longer have matching traces'
        traces.append({'train_seed': seed, 'four_arm_trace_sha256': hashes[0]})
    report = {
        'date': '2026-11-04',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'status': 'BLOCKED_NOT_READY_FOR_JTT_FORMAL_AGAINST_FROZEN_AXIS_I',
        'training_started': False, 'sources': sources,
        'frozen_contract': {'training_seeds': [0,1,2], 'map_seed': 2, 'rounds': 40,
            'local_batches': 16, 'batch_size': 64, 'public_batch_size': 128, 'public_batches_per_round': 4,
            'pretrain_epochs': 0, 'optimizer': configs[(0,'erm')]['train']['optimizer'],
            'skip_nonfinite': configs[(0,'erm')]['train']['skip_nonfinite'],
            'models': configs[(0,'erm')]['models']['names']},
        'frozen_four_arm_traces': traces,
        'current_generic_runner_vs_frozen': compare(frozen, current),
        'legacy_jtt_erm_vs_frozen': compare(frozen, legacy),
        'legacy_jtt_train_seed_support': [0],
        'stage2_initial_weights_same_as_frozen_erm_config': True,
        'minibatch_counterexample': {'N': 4, 'batch_size': 2, 'upweight': 20,
            'losses': losses.tolist(), 'error_mask': mask.tolist(),
            'normalized_full_dataset_repetition_risk': float(global_risk),
            'expected_legacy_batch_normalized_loss': float(expected_batch_ratio),
            'expected_fixed_global_normalized_loss': float(fixed_normalized),
            'importance': 'Exact batch repetition arithmetic does not prove a global minibatch equivalence'},
        'mask_unit_check': {'fit_indices': [4,1,2], 'audit_indices': [0,3,5],
            'visited_indices': visited, 'full_mask': actual_mask.tolist(), 'passed': True,
            'producer_manifest_keys': sorted(produced)},
        'protocol_draft_not_authorized': {'stage1_rounds_candidate': 12,
            'stage2_rounds': 40, 'local_batches': 16, 'upweight_candidate': 20,
            'final_stage_budget_matched': True, 'total_training_budget_equal_to_40_round_erm': False,
            'total_rounds': 52, 'extra_local_updates_fraction': .3,
            'primary_fidelity_option': 'Fixed fit-only error set, resampled stage2, mean CE; new sampler gate required',
            'alternative': 'Same-batch global-normalized reweighting must be labelled an adaptation',
            'hyperparameter_rule': 'No map2 final-test/DSA/true-environment tuning; select and freeze using permitted development protocol'},
        'known_missing_gates': ['map2-specific input and seed identity', 'stage1 early-stop selection protocol',
            'mask provenance/coverage checks', 'global target objective or faithful resampling',
            'sampler-aware pairing audit', 'stage-specific recovery uploads and fail-closed context',
            'matched final analysis and finite-value/completion gates'],
    }
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status','frozen_contract','current_generic_runner_vs_frozen',
           'legacy_jtt_erm_vs_frozen','minibatch_counterexample','mask_unit_check']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
