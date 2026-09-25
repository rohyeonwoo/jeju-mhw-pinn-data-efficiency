"""P0 lambda pilot; preserves the original runner and src modules."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

import setproctitle
setproctitle.setproctitle('yeonwoo')
os.environ['MHW_PROCESS_NAME'] = 'yeonwoo'
ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path('/ceph_data/yeonwoo/MHW_subset')
OUTPUT = STORAGE / 'outputs/data_efficiency'
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch.utils.data import Subset
import train_jeju_hilam as original


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


class BatchStream:
    """Keep a subset iterator alive across validation; reshuffle only at exhaustion."""
    def __init__(self, loader):
        if len(loader) == 0:
            raise ValueError('Empty train loader')
        self.loader = loader
        self.iterator = None
        self.completed_steps = 0
        self.completed_samples = 0
        self.started_passes = 0

    def take(self, count, on_step=None):
        for _ in range(count):
            if self.iterator is None:
                self.iterator = iter(self.loader)
                self.started_passes += 1
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.loader)
                self.started_passes += 1
                batch = next(self.iterator)
            yield batch
            # Called only after the caller has finished the optimizer step.
            self.completed_steps += 1
            self.completed_samples += len(batch['target_index'])
            if on_step is not None:
                on_step(self)


def original_args(lead, seed, mode):
    saved = sys.argv
    try:
        sys.argv = ['train_jeju_hilam.py', '--lead', str(lead), '--seed', str(seed),
                    '--physics-mode', mode, '--no-residual-feature']
        return original.parse_args()
    finally:
        sys.argv = saved


def verify_inputs(seed, lead):
    pre = ROOT / 'docs/server_preflight'
    snapshot = json.loads((pre / 'original_snapshot_sha256.json').read_text())
    for name, expected in snapshot.items():
        if name.startswith(('src/', 'data/', 'outputs/main_experiment_final/')):
            if sha(ROOT / name) != expected:
                raise RuntimeError(f'Protected input changed: {name}')
    audit = json.loads((pre / 'audit_report.json').read_text())
    assert audit['status'] == 'passed' and audit['protected_files_unchanged']
    subset_path = ROOT / f'data/subsets/train_subsets_v2_win44_seed{seed}.json'
    subset = json.loads(subset_path.read_text())
    data = (ROOT / subset['_metadata']['source']['resolved_data_path']).resolve()
    assert data.is_relative_to(STORAGE / 'data')
    assert sha(data) == audit['data_sha256'] == subset['_metadata']['source']['data_sha256']
    assert subset['_metadata']['seed'] == seed
    return data, subset_path, subset, audit['leads'][str(lead)], snapshot


def select_training_data(bundle, payload, ratio=None, use_full=False):
    if use_full:
        if ratio is not None:
            raise ValueError('Full and block-subset ratio are mutually exclusive')
        starts = bundle.train_indices.copy()
        selected = bundle.train
        assert selected.base.stats is bundle.stats
        assert len(selected) == len(starts)
    else:
        starts = np.asarray(payload[str(ratio)]['start_indices'], dtype=np.int64)
        positions = np.searchsorted(bundle.train_indices, starts)
        assert np.all(positions < len(bundle.train_indices))
        np.testing.assert_array_equal(bundle.train_indices[positions], starts)
        assert len(starts) == {10: 376, 25: 987, 50: 1974, 100: 3995}[ratio]
        selected = Subset(bundle.train, positions.tolist())
        assert selected.dataset.base.stats is bundle.stats
    return selected, starts


def gpu_identity():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True)
    rows = list(csv.reader(io.StringIO(result)))
    own = [r for r in rows if int(r[0].strip()) == os.getpid()]
    assert own and all(r[1].strip() == 'yeonwoo' for r in own), own


def finite_training(stats):
    for key in ('loss', 'total_loss', 'data_loss', 'physics_loss', 'rmse'):
        if not math.isfinite(stats[key]):
            raise RuntimeError(f'Nonfinite {key}: {stats[key]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lead', type=int, choices=(7,), required=True)
    p.add_argument('--seed', type=int, choices=(42, 7, 123), required=True)
    data_choice = p.add_mutually_exclusive_group(required=True)
    data_choice.add_argument('--ratio', type=int, choices=(10,))
    data_choice.add_argument('--full', action='store_true', help='All original Full training samples; not the 100% block subset')
    p.add_argument('--physics-mode', choices=('p0_original',), default='p0_original')
    p.add_argument('--lambda-physics', type=float, choices=(0.0,0.01,0.1,0.3), required=True)
    p.add_argument('--batch-size', type=int, choices=(16,), default=16)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='2 blocks of 2 updates, separate from production')
    options = p.parse_args()
    assert pwd.getpwuid(os.geteuid()).pw_name == 'yeonwoo'
    out = options.output_root.resolve()
    assert out.is_relative_to(OUTPUT), 'All outputs must stay in Ceph data_efficiency'
    if options.smoke:
        out = out / '_smoke'
    elif '_smoke' in out.parts:
        raise ValueError('Production cannot use a smoke directory')
    model_id = 'jeju_hilam' if options.physics_mode == 'none' else 'pi_jeju_hilam'
    data_label = 'full' if options.full else f'ratio_{options.ratio}'
    lambda_tag = format(options.lambda_physics, '.6g').replace('.', 'p')
    run_dir = out / ('lambda_' + lambda_tag) / data_label / model_id / f'lead{options.lead}_seed{options.seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    progress_path = run_dir / 'progress.json'
    save(progress_path, {'status': 'preparing', 'pid': os.getpid(), 'started_unix': time.time()})
    args = original_args(options.lead, options.seed, options.physics_mode)
    args.batch_size = options.batch_size
    args.lambda_physics = options.lambda_physics
    assert args.lambda_rres == 10.0 and args.lambda_we_tie == 0.0
    assert args.zero_rres_physics is True
    torch.set_num_threads(4)
    original.set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')
    data, subset_path, payload, full, snapshot = verify_inputs(args.seed, args.lead)
    bundle = original.make_jeju_hilam_bundle(data, None, 30, args.lead, include_residual_feature=False)
    for name in ('train', 'val', 'test'):
        idx = getattr(bundle, name + '_indices')
        assert len(idx) == full['counts'][name]
        assert hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest() == full['split_sha256'][name]
    # The same original Full routine in the same pinned environment must reproduce the cache exactly.
    for key, value in bundle.stats.items():
        np.testing.assert_array_equal(value, full['full_stats'][key])
    selected, starts = select_training_data(bundle, payload, options.ratio, options.full)
    for i in (0, len(selected) // 2, len(selected) - 1):
        sample = selected[i]
        assert int(sample['current_index']) == int(starts[i]) + 29
        assert int(sample['target_index']) == int(starts[i]) + 29 + args.lead
    physics = original.build_physics_args(args)
    if physics is not None:
        physics.physics_rhs_scale = full['full_rhs_scale']
        args.physics_rhs_scale = full['full_rhs_scale']
    model = original.JejuHiLAM(
        graph=bundle.graph, input_dim=len(bundle.train.input_feature_names),
        hidden_dim=args.hidden_dim, temporal_hidden_dim=args.temporal_hidden_dim,
        processor_layers=args.processor_layers, mlp_hidden_layers=args.mlp_hidden_layers,
        prediction_mode=args.prediction_mode, physics_heads=physics is not None,
        we_nonneg=True, we_scale=args.we_scale, rres_scale=args.rres_scale,
        we_softplus_shift=args.we_softplus_shift).to(device)
    gpu_identity()
    original.save_graph_artifacts(bundle.graph, run_dir)
    loader = original.make_loader(selected, args, shuffle=True)
    val_loader = original.make_loader(bundle.val, args, shuffle=False)
    test_loader = original.make_loader(bundle.test, args, shuffle=False)
    stream = BatchStream(loader)
    S = math.ceil(len(bundle.train) / args.batch_size)
    assert full['S'] == math.ceil(full['counts']['train'] / 8)
    assert S == math.ceil(full['counts']['train'] / args.batch_size)
    target_mean, target_std = bundle.stats['target']
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'data_path': str(data), 'output_root': str(out), 'device': str(device),
        'model': model_id, 'model_label': ('P0 auxiliary-only (lambda_physics=0)' if options.lambda_physics == 0 else 'SeaCast + PINN(P0)'),
        'study': 'lambda_pilot_lead7_10pct_full',
        'arm': 'auxiliary_only' if options.lambda_physics == 0 else 'p0_lambda_sweep',
        'lambda_tag': lambda_tag,
        'parent_runner_path': str(ROOT / 'scripts/train_data_efficiency.py'),
        'parent_runner_sha256': sha(ROOT / 'scripts/train_data_efficiency.py'),
        'rng_policy': 'unchanged legacy P0 initialization and DataLoader RNG; no reset after model construction',
        'data_regime': 'full' if options.full else 'block_subset',
        'subset_ratio': options.ratio, 'subset_seed': None if options.full else args.seed, 'training_seed': args.seed,
        'subset_path': None if options.full else str(subset_path),
        'subset_sha256': None if options.full else sha(subset_path),
        'data_sha256': payload['_metadata']['source']['data_sha256'],
        'source_sha256': {k: v for k, v in snapshot.items() if k.startswith('src/')},
        'runner_sha256': sha(__file__), 'full_audit_sha256': sha(ROOT / 'docs/server_preflight/audit_report.json'),
        'split_sha256': full['split_sha256'], 'full_statistics': full['full_stats'],
        'statistics_basis': 'original Full train; includes unselected input and observed target information',
        'statistics_end_date': full['statistics_end_date'],
        'evaluation_reference_end_date': full['evaluation_reference_end_date'],
        'physics_rhs_scale': full['full_rhs_scale'] if physics is not None else None,
        'physics_preset': 'P0' if physics is not None else None,
        'physics_components': 'original' if physics is not None else None,
        'physics_normalize': 'rhs' if physics is not None else None,
        'physics_time_mode': 'daily_rollout' if physics is not None else None,
        'lambda_physics': args.lambda_physics if physics is not None else 0.0,
        'train_samples': len(selected), 'full_train_samples': len(bundle.train),
        'val_samples': len(bundle.val), 'test_samples': len(bundle.test),
        'input_dim': len(bundle.train.input_feature_names), 'input_feature_names': bundle.train.input_feature_names,
        'target_mean': target_mean, 'target_std': target_std, 'graph': bundle.graph.summary(),
        'S': S, 'max_steps': 80*S, 'warmup_steps': 5*S,
        'original_full_batch_size': 8,
        'schedule_basis': 'ceil(original Full train sample count / current batch size)',
        'validation_interval_steps': S, 'early_stopping_patience_validations': 12,
        'smoke_only': options.smoke, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'environment': json.loads((ROOT / 'docs/server_preflight/environment.json').read_text()),
        'process_name': 'yeonwoo',
    }
    save(run_dir / 'jeju_hilam_config.json', config)
    save(run_dir / 'selected_start_indices.json', starts.tolist())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.initial_lr,
                                 weight_decay=args.weight_decay, betas=(0.9, 0.95))
    common = dict(graph=bundle.graph, device=device, target_mean=target_mean,
                  target_std=target_std, precision=args.precision, physics_args=physics,
                  lambda_physics=args.lambda_physics)
    best, best_step, bad = float('inf'), 0, 0
    history = []
    best_path = run_dir / 'jeju_hilam_best.pt'
    windows = 2 if options.smoke else 80
    block_steps = 2 if options.smoke else S
    print(json.dumps({'event': 'start', 'pid': os.getpid(), 'run_dir': str(run_dir),
                      'S': S, 'train_samples': len(selected), 'smoke': options.smoke}), flush=True)
    for epoch in range(1, windows + 1):
        if not options.smoke:
            assert epoch == stream.completed_steps // S + 1
        lr = original.set_epoch_lr(optimizer, args, epoch)
        def progress(s):
            if s.completed_steps % 25 == 0 or s.completed_steps % block_steps == 0:
                state = {'status': 'training', 'pid': os.getpid(), 'step': s.completed_steps,
                         'virtual_epoch': epoch, 'max_steps': 80*S, 'best_step': best_step,
                         'best_val_rmse': best if math.isfinite(best) else None,
                         'elapsed_seconds': time.monotonic()-t0, 'updated_unix': time.time()}
                save(progress_path, state)
                print(json.dumps(state), flush=True)
        train = original.run_epoch(model, stream.take(block_steps, progress),
                                   optimizer=optimizer, epoch=epoch, max_batches=None, **common)
        finite_training(train)
        assert stream.completed_steps == epoch * block_steps
        save(progress_path, {'status': 'validating', 'pid': os.getpid(), 'step': stream.completed_steps,
                             'virtual_epoch': epoch, 'updated_unix': time.time()})
        with torch.no_grad():
            val = original.run_epoch(model, itertools.islice(val_loader, 2) if options.smoke else val_loader,
                                     optimizer=None, epoch=epoch, max_batches=None, **common)
        finite_training(val)
        row = {'epoch': epoch, 'virtual_epoch': epoch, 'step': stream.completed_steps, 'lr': lr,
               'elapsed_seconds': time.monotonic()-t0,
               **{'train_'+k: v for k,v in train.items()}, **{'val_'+k: v for k,v in val.items()}}
        history.append(row)
        original.save_epoch_history(run_dir / 'jeju_hilam_epoch_history.csv', history)
        print(json.dumps(row), flush=True)
        if val['rmse'] < best:
            best, best_step, bad = val['rmse'], stream.completed_steps, 0
            checkpoint = {'model_state': model.state_dict(), 'config': config, 'stats': bundle.stats,
                          'best_epoch': epoch, 'best_step': best_step, 'best_value': best}
            temp = best_path.with_suffix('.tmp.pt')
            torch.save(checkpoint, temp)
            temp.replace(best_path)
        else:
            bad += 1
            if bad >= args.patience:
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    save(progress_path, {'status': 'evaluating', 'pid': os.getpid(), 'step': stream.completed_steps,
                         'best_step': best_step, 'best_val_rmse': best, 'updated_unix': time.time()})
    with torch.no_grad():
        test_stats = original.run_epoch(model, itertools.islice(test_loader, 2) if options.smoke else test_loader,
                                       optimizer=None, epoch=args.physics_warmup_epochs, **common)
    fields = original.collect_predictions(model, itertools.islice(test_loader, 2) if options.smoke else test_loader,
                                           **{k:v for k,v in common.items() if k != 'lambda_physics'})
    expected_targets = bundle.test_indices + 29 + args.lead
    np.testing.assert_array_equal(fields['target_index'], expected_targets[:len(fields['target_index'])])
    if not options.smoke:
        assert len(fields['target_index']) == len(bundle.test)
    fields_path = run_dir / f'{model_id}_lead_{args.lead}_history_30_test_fields.npz'
    np.savez_compressed(fields_path, **fields)
    metrics = original.compute_forecast_metrics(fields['pred_raw'], fields['target_raw'], fields['mask'])
    if not options.smoke:
        original.add_acc_and_sw_metrics(metrics, bundle.ds, fields)
    metrics.update({'test_'+k:v for k,v in test_stats.items()})
    metrics.update({k: config[k] for k in ('model','model_label','lead','history','seed','physics_mode',
                   'physics_preset','physics_components','physics_normalize','physics_rhs_scale',
                   'lambda_physics','lambda_rres','lambda_we_tie','arm','study','subset_ratio','subset_seed','training_seed','test_samples','data_regime')})
    metrics.update(best_epoch=checkpoint['best_epoch'], best_step=best_step,
                   best_selection_metric='val_rmse', best_selection_value=best,
                   actual_steps=stream.completed_steps, validation_count=len(history),
                   fields_path=str(fields_path), checkpoint_path=str(best_path), smoke_only=options.smoke)
    # Original metric code may intentionally emit NaN for an undefined diagnostic.
    original.save_json(run_dir / f'{model_id}_lead_{args.lead}_history_30_metrics.json', metrics)
    for name in ('train', 'val', 'test'):
        idx = getattr(bundle, name + '_indices')
        assert hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest() == full['split_sha256'][name]
    save(progress_path, {'status': 'completed', 'pid': os.getpid(), 'step': stream.completed_steps,
                         'best_step': best_step, 'best_val_rmse': best, 'validation_count': len(history),
                         'elapsed_seconds': time.monotonic()-t0, 'updated_unix': time.time()})
    bundle.ds.close()
    print(json.dumps({'event':'completed','run_dir':str(run_dir),'steps':stream.completed_steps}), flush=True)


if __name__ == '__main__':
    main()
