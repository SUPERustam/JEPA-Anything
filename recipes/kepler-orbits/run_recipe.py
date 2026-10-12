"""Reconstructed Kepler scaling diagnostic; no sustained training by default."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'jepa-anything-core/src'))
from jepa_anything_core import (OrthogonalFactorProjection, audit_basis_geometry,
                                ema_update, jepa_anything_objective)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    mu: float = 1.0
    dt: float = 0.05
    orbits: int = 256
    samples: int = 4096
    validation_fraction: float = 0.2
    batch_size: int = 256
    eval_orbits: int = 32
    eval_samples: int = 16384
    learning_rate: float = 1e-3
    ema: float = 0.99
    orthogonality_weight: float = 0.1
    factor_activity_weight: float = 0.05
    encoder_variance_weight: float = 0.02
    activity_floor: float = 0.1
    smoke: bool = False


def orbit(a, e, orientation, phase, samples, dt=0.05, mu=1.0):
    """Uniform-time [x,y,vx,vy] observations from elliptic Kepler solutions."""
    values = np.asarray([a, e, orientation, phase, dt, mu], dtype=float)
    if not np.isfinite(values).all() or a <= 0 or not 0 <= e < 1 or dt <= 0 or mu <= 0:
        raise ValueError('finite elliptic parameters, positive a, dt and mu required')
    if samples < 4:
        raise ValueError('at least four samples required')
    n = np.sqrt(mu / a**3)
    mean = (phase + n * np.arange(samples) * dt + np.pi) % (2 * np.pi) - np.pi
    eccentric = mean.copy()
    for _ in range(50):
        delta = (eccentric - e * np.sin(eccentric) - mean) / (1 - e * np.cos(eccentric))
        eccentric -= delta
        if np.max(np.abs(delta)) < 1e-13:
            break
    else:
        raise RuntimeError('Kepler solver did not converge')
    c, s = np.cos(eccentric), np.sin(eccentric)
    speed = n / (1 - e * c)
    shape = np.sqrt(1 - e**2)
    position = np.stack((a * (c - e), a * shape * s), axis=-1)
    velocity = np.stack((-a * s * speed, a * shape * c * speed), axis=-1)
    co, si = np.cos(orientation), np.sin(orientation)
    rotation = np.array([[co, -si], [si, co]])
    return np.concatenate((position @ rotation.T, velocity @ rotation.T), axis=-1)


def generate_orbits(config, *, evaluation=False):
    rng = np.random.default_rng(config.seed + int(evaluation))
    count = config.eval_orbits if evaluation else config.orbits
    axes = np.geomspace(0.5, 4, count) if evaluation else rng.uniform(0.5, 4, count)
    metadata = np.column_stack((axes, rng.uniform(0.05, 0.3, count),
                                rng.uniform(0, 2*np.pi, count), rng.uniform(0, 2*np.pi, count)))
    samples = config.eval_samples if evaluation else config.samples
    states = np.stack([orbit(*p, samples, config.dt, config.mu) for p in metadata])
    return states, metadata


def split_and_normalize(states, config):
    if not 0 < config.validation_fraction < 1 or len(states) < 2:
        raise ValueError('split needs at least two orbits and a fraction in (0,1)')
    ids = np.random.default_rng(config.seed).permutation(len(states))
    count = max(1, min(len(states)-1, round(len(states)*config.validation_fraction)))
    validation, training = ids[:count], ids[count:]
    mean = states[training].mean(axis=(0, 1))
    std = states[training].std(axis=(0, 1))
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('training normalization must have finite positive scales')
    normalization = {'mean': mean.tolist(), 'std': std.tolist(),
                     'training_ids': training.tolist(), 'validation_ids': validation.tolist()}
    normalized = torch.tensor((states - mean) / std, dtype=torch.float32)
    return normalized[training], normalized[validation], normalization


class OrbitalJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 128),
                                     nn.GELU(), nn.Linear(128, 32))
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.projection = OrthogonalFactorProjection(32, 4, 8, learnable=True,
                                                     orthogonality_mode='soft_gram')
        self.predictors = nn.ModuleList([nn.Sequential(nn.Linear(32, 8), nn.GELU(),
                                                       nn.Linear(8, 8)) for _ in range(4)])

    def predict_factors(self, state):
        return torch.stack([p(state) for p in self.predictors], dim=-2)

    def objective(self, context, future, config):
        state = self.encoder(context)
        with torch.no_grad():
            target = self.target_encoder(future)
        # Target encoding is frozen; projection retains activity/prediction gradients.
        factors = self.projection(target)
        predicted = self.predict_factors(state)
        return jepa_anything_objective(
            predicted, factors, self.projection.analysis_basis(), state,
            orthogonality_weight=config.orthogonality_weight,
            factor_activity_weight=config.factor_activity_weight,
            encoder_variance_weight=config.encoder_variance_weight,
            factor_min_std=config.activity_floor, encoder_min_std=config.activity_floor)


def transitions(states, batch_size, rng):
    ids = torch.randint(len(states), (batch_size,), generator=rng)
    times = torch.randint(states.shape[1]-1, (batch_size,), generator=rng)
    return states[ids, times], states[ids, times+1]


def spectral_peak(coordinates, dt):
    """Dominant peak and concentration diagnostics; no physical labels used."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if coordinates.ndim != 2 or len(coordinates) < 4 or not np.isfinite(coordinates).all():
        raise ValueError('finite time-by-coordinate samples required')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite dt required')
    centered = coordinates - coordinates.mean(axis=0)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered))[:, None], axis=0))**2
    power = power.sum(axis=1)
    if power[1:].max() <= 0:
        raise ValueError('constant signal has no nonzero spectral peak')
    peak = int(np.argmax(power[1:]) + 1)
    offset = 0.0
    if 1 <= peak < len(power)-1:
        left, middle, right = np.log(np.maximum(power[peak-1:peak+2], np.finfo(float).tiny))
        curvature = left - 2*middle + right
        if curvature < 0:
            offset = float(np.clip(0.5*(left-right)/curvature, -0.5, 0.5))
    return {'frequency': (peak + offset) / (len(centered)*dt),
            'temporal_rms': float(np.sqrt(np.mean(centered**2))),
            'three_bin_power_fraction': float(power[max(1, peak-1):peak+2].sum()/power[1:].sum())}


def strongest_peak(coordinates, dt):
    return spectral_peak(coordinates, dt)['frequency']


def scaling_report(axes, frequencies, mu=1.0):
    x, y = np.log10(axes), np.log10(frequencies)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope*x + intercept)
    total = np.sum((y-y.mean())**2)
    theory = np.sqrt(mu) / (2*np.pi) * axes**-1.5
    return {'slope': float(slope), 'intercept_log10': float(intercept),
            'r_squared_log10': float(1 - np.sum(residual**2)/total) if total else None,
            'mean_absolute_frequency_error': float(np.mean(np.abs(frequencies-theory))),
            'mean_relative_frequency_error': float(np.mean(np.abs(frequencies/theory-1))),
            'median_relative_frequency_error': float(np.median(np.abs(frequencies/theory-1))),
            'max_relative_frequency_error': float(np.max(np.abs(frequencies/theory-1)))}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = Config(**checkpoint['config'])
    model = OrbitalJEPA()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, config, checkpoint


def initial_model(checkpoint, config):
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        model = OrbitalJEPA().eval()
    if 'initial_model' in checkpoint:
        model.load_state_dict(checkpoint['initial_model'])
        source = 'saved zero-step weights'
    else:
        # shortcut: old checkpoints lack initial weights, reconstruct and disclose the library version.
        source = 'seed-reconstructed zero-step weights; original initialization was not saved'
    return model, source


@torch.no_grad()
def evaluate_prediction(model, states):
    """Every held-out transition against one fixed EMA target in factor units."""
    if states.ndim != 3 or states.shape[1] < 2 or not torch.isfinite(states).all():
        raise ValueError('finite orbit-by-time-by-coordinate observations required')
    context, future = states[:, :-1].reshape(-1, 4), states[:, 1:].reshape(-1, 4)
    totals = {name: torch.zeros(4, dtype=torch.float64) for name in
              ('learned', 'teacher_persistence', 'online_persistence')}
    for start in range(0, len(context), 2048):
        x, y = context[start:start+2048], future[start:start+2048]
        state = model.encoder(x)
        target = model.projection(model.target_encoder(y))
        predictions = {'learned': model.predict_factors(state),
                       'teacher_persistence': model.projection(model.target_encoder(x)),
                       'online_persistence': model.projection(state)}
        for name, predicted in predictions.items():
            totals[name] += (predicted-target).double().square().sum(dim=(0, 2)).cpu()
    mse = {name: float(value.sum()/(len(context)*32)) for name, value in totals.items()}
    reference = mse['teacher_persistence']
    return {'trajectories': len(states), 'transitions': len(context), 'mse': mse,
            'factor_mse': {name: (value/(len(context)*8)).tolist() for name, value in totals.items()},
            'improvement_over_teacher_persistence': 1-mse['learned']/reference if reference else None,
            'target': 'final frozen EMA encoder and learned projection',
            'units': 'latent factor squared error; not physical-state forecasting'}


def analyze(checkpoint_path, output=None):
    started = datetime.now(timezone.utc).isoformat()
    model, config, checkpoint = load_checkpoint(checkpoint_path)
    output = output or checkpoint_path.parent
    names = ('frequencies.csv', 'spectral_diagnostics.csv', 'analysis.json',
             'evaluation-provenance.json', 'scaling.png', 'scaling.pdf')
    if any((output/name).exists() for name in names):
        raise FileExistsError('analysis artifacts already exist; use a fresh --output directory')
    output.mkdir(parents=True, exist_ok=True)
    baseline, baseline_source = initial_model(checkpoint, config)
    states, metadata = generate_orbits(config, evaluation=True)
    normalization = checkpoint['normalization']
    normalized = torch.tensor((states - normalization['mean']) / normalization['std'], dtype=torch.float32)
    rows, baseline_rows, spectra = [], [], []
    with torch.no_grad():
        for i, observations in enumerate(normalized):
            readouts = {}
            for name, candidate in (('trained', model), ('untrained', baseline)):
                factors = torch.cat([candidate.projection(candidate.encoder(chunk))
                                     for chunk in observations.split(2048)]).numpy()
                diagnostics = [spectral_peak(factors[:, k], config.dt) for k in range(4)]
                combined = spectral_peak(factors.reshape(len(factors), -1), config.dt)
                readouts[name] = [combined['frequency'], *[p['frequency'] for p in diagnostics]]
                for factor, peak in enumerate([combined, *diagnostics]):
                    spectra.append({'model': name, 'orbit_id': i,
                                    'readout': 'aggregate' if factor == 0 else f'factor_{factor-1}', **peak})
            raw = strongest_peak(states[i], config.dt)
            theory = np.sqrt(config.mu) / (2*np.pi) * metadata[i, 0]**-1.5
            rows.append([i, *metadata[i], theory, raw, *readouts['trained'], *readouts['untrained']])
            baseline_rows.append(readouts['untrained'])
    values = np.asarray(rows)
    axes = metadata[:, 0]
    reports = {'learned_aggregate': scaling_report(axes, values[:, 7], config.mu),
               'learned_factors': [scaling_report(axes, values[:, 8+k], config.mu) for k in range(4)],
               'raw_state_physics_check': scaling_report(axes, values[:, 6], config.mu),
               'optimizer_steps': checkpoint['steps'], 'smoke': config.smoke,
               'scientific_accuracy_validated': False,
               'peak_selection': 'strongest nonzero Hann-windowed power; no theoretical guidance',
               'evaluation_seed': config.seed+1, 'eval_samples': config.eval_samples,
               'frequency_bin_width': 1/(config.eval_samples*config.dt)}
    reports['untrained_aggregate'] = scaling_report(axes, np.asarray(baseline_rows)[:, 0], config.mu)
    reports['untrained_factors'] = [scaling_report(axes, np.asarray(baseline_rows)[:, k+1], config.mu)
                                   for k in range(4)]
    reports['baseline_source'] = baseline_source
    reports['spectral_diagnostics'] = 'temporal RMS and three-bin power fraction; not confidence intervals'
    reports['harmonic_flags'] = {}
    theory = values[:, 5]
    # The law labels peaks only after extraction; never use it to choose or correct them.
    for name, offset in (('trained', 8), ('untrained', 13)):
        reports['harmonic_flags'][name] = []
        for k in range(4):
            ratio = values[:, offset+k]/theory
            multiple = np.rint(ratio).astype(int)
            flags = (multiple >= 2) & (np.abs(ratio/np.maximum(multiple, 1)-1) <= 0.02)
            reports['harmonic_flags'][name].append(np.flatnonzero(flags).tolist())
    reports['harmonic_flag_rule'] = 'within 2% of an integer multiple >= 2 of theory; evaluation flag only'
    reports['basis_geometry'] = audit_basis_geometry(model.projection.analysis_basis()).to_dict()
    training_states, _ = generate_orbits(config)
    ids = normalization['validation_ids']
    training_ids = normalization['training_ids']
    if (not ids or set(ids) & set(training_ids)
            or sorted(ids+training_ids) != list(range(config.orbits))):
        raise ValueError('checkpoint split must partition complete trajectories')
    validation = torch.tensor((training_states[ids]-normalization['mean'])/normalization['std'],
                              dtype=torch.float32)
    reports['prediction_evaluation'] = evaluate_prediction(model, validation)
    reports['interpretation'] = ('Scaling tests retained periodicity; the matched zero-step control '
                                 'tests whether training changes it. Prediction is scored separately.')
    with (output / 'frequencies.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['orbit_id', 'a', 'e', 'orientation', 'phase', 'theory_frequency',
                         'raw_state_frequency', 'learned_aggregate_frequency',
                         *[f'factor_{k}_frequency' for k in range(4)], 'untrained_aggregate_frequency',
                         *[f'untrained_factor_{k}_frequency' for k in range(4)]])
        writer.writerows(rows)
    with (output / 'spectral_diagnostics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectra[0]))
        writer.writeheader()
        writer.writerows(spectra)
    write_json(output / 'analysis.json', reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, panels = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    theory = np.sqrt(config.mu) / (2*np.pi) * axes**-1.5
    for k in range(4):
        panels[0].scatter(axes, values[:, 8+k], s=16, alpha=0.6, label=f'Factor {k}')
    fit = reports['learned_aggregate']
    panels[0].scatter(axes, values[:, 7], marker='x', color='black', label='Aggregate power peak')
    panels[0].plot(axes, 10**fit['intercept_log10'] * axes**fit['slope'], '--', color='black',
                   label=f"Fit: slope {fit['slope']:.4f}, R² {fit['r_squared_log10']:.5f}")
    panels[1].scatter(axes, values[:, 6], s=20, label='Raw position–velocity peak')
    panels[1].scatter(axes, values[:, 12], marker='+', label='Matched zero-step encoder')
    panels[1].scatter(axes, values[:, 7], marker='x', label='Trained encoder aggregate')
    for panel in panels:
        panel.plot(axes, theory, color='red', label='Theory: (2π)⁻¹ a⁻³ᐟ²')
        panel.set(xscale='log', yscale='log', xlabel='Semimajor axis a', ylabel='Frequency (cycles / time)')
        panel.grid(True, which='both', alpha=0.2)
        panel.legend(fontsize=8)
    panels[0].set_title('Learned online-encoder factor coordinates')
    panels[1].set_title('Matched controls: periodicity can survive without learning')
    fig.suptitle(f"Kepler reconstruction · {checkpoint['steps']} optimizer steps" +
                 (' · SMOKE: implementation check only' if config.smoke else ' · scientific accuracy unvalidated'))
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'scaling.{suffix}', dpi=180)
    plt.close(fig)
    sources = [Path(__file__), *sorted((ROOT/'jepa-anything-core/src/jepa_anything_core').glob('*.py'))]
    write_json(output / 'evaluation-provenance.json', {
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'artifact_sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                            for name in names if name != 'evaluation-provenance.json'},
        'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'baseline_source': baseline_source, 'training_torch': checkpoint.get('torch_version'),
        'optimizer_steps': checkpoint['steps'], 'evaluation_seed': config.seed+1})
    return reports


def train(config, steps, output):
    if steps <= 0:
        raise ValueError('steps must be positive')
    # Refuse to overwrite run evidence; use a new directory for each experiment.
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    states, metadata = generate_orbits(config)
    training, validation, normalization = split_and_normalize(states, config)
    model = OrbitalJEPA()
    initialization = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    rng = torch.Generator().manual_seed(config.seed)
    validation_rng = torch.Generator().manual_seed(config.seed+2)
    metrics = []
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.objective(*transitions(training, config.batch_size, rng), config)
        if not torch.isfinite(loss.total):
            raise RuntimeError('non-finite training loss')
        loss.total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('non-finite gradient')
        optimizer.step()
        ema_update(model.target_encoder, model.encoder, config.ema)
        with torch.no_grad():
            val = model.objective(*transitions(validation, config.batch_size, validation_rng), config)
        row = {'step': step, **{f.name: float(getattr(loss, f.name).detach()) for f in fields(loss)},
               'validation_total': float(val.total), 'validation_prediction': float(val.prediction)}
        metrics.append(row)
    write_json(output / 'config.json', asdict(config))
    write_json(output / 'normalization.json', normalization)
    write_json(output / 'orbit_metadata.json', {'columns': ['a', 'e', 'orientation', 'phase'], 'rows': metadata.tolist()})
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    torch.save({'model': model.state_dict(), 'initial_model': initialization, 'optimizer': optimizer.state_dict(),
                'config': asdict(config), 'normalization': normalization, 'steps': steps,
                'torch_version': str(torch.__version__)}, output / 'checkpoint.pt')
    return output / 'checkpoint.pt'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('smoke', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, default=ROOT / 'runs/kepler-orbits' / name)
        if name == 'train':
            command.add_argument('--steps', type=int, required=True)
    command = commands.add_parser('analyze')
    command.add_argument('--checkpoint', type=Path, required=True)
    command.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'analyze':
        torch.set_num_threads(1)
        reports = analyze(args.checkpoint, args.output)
    else:
        config = Config(orbits=16, batch_size=32, smoke=True) if args.command == 'smoke' else Config()
        path = train(config, 5 if args.command == 'smoke' else args.steps, args.output)
        reports = analyze(path)
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
