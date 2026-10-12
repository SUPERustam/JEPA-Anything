import csv
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from run_recipe import (Config, OrbitalJEPA, analyze, ema_update, generate_orbits,
                        evaluate_prediction, initial_model, load_checkpoint, main, orbit,
                        scaling_report, spectral_peak, split_and_normalize, strongest_peak, train)


def test_solver_circular_conservation_and_determinism():
    dt, a = 0.05, 2.0
    states = orbit(a, 0, 0, 0, 1024, dt)
    angle = np.arange(1024)*dt*a**-1.5
    np.testing.assert_allclose(states[:, 0], a*np.cos(angle), atol=1e-12)
    np.testing.assert_allclose(states[:, 1], a*np.sin(angle), atol=1e-12)
    np.testing.assert_allclose(states[:, 2], -a**-0.5*np.sin(angle), atol=1e-12)
    np.testing.assert_allclose(states[:, 3], a**-0.5*np.cos(angle), atol=1e-12)
    states = orbit(a, 0.3, 1.2, 2.1, 4096)
    radius = np.linalg.norm(states[:, :2], axis=1)
    energy = (states[:, 2:]**2).sum(axis=1)/2 - 1/radius
    momentum = states[:, 0]*states[:, 3] - states[:, 1]*states[:, 2]
    np.testing.assert_allclose(energy, -1/(2*a), atol=1e-12)
    np.testing.assert_allclose(momentum, np.sqrt(a*(1-0.3**2)), atol=1e-12)
    config = Config(orbits=8, samples=128)
    first = generate_orbits(config)
    second = generate_orbits(config)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError):
        orbit(1, 1, 0, 0, 64)


def test_split_and_train_only_normalization():
    config = Config(orbits=8, samples=128)
    states, _ = generate_orbits(config)
    training, validation, normalization = split_and_normalize(states, config)
    training_ids = normalization['training_ids']
    validation_ids = normalization['validation_ids']
    assert set(training_ids).isdisjoint(validation_ids)
    assert sorted(training_ids + validation_ids) == list(range(8))
    np.testing.assert_allclose(normalization['mean'], states[training_ids].mean(axis=(0, 1)))
    np.testing.assert_allclose(normalization['std'], states[training_ids].std(axis=(0, 1)))
    altered = states.copy()
    altered[validation_ids] += 1000
    train_after, _, norm_after = split_and_normalize(altered, config)
    assert norm_after == normalization
    torch.testing.assert_close(training, train_after, rtol=0, atol=0)
    assert len(validation) == len(validation_ids)
    torch.testing.assert_close(training.mean(dim=(0, 1)), torch.zeros(4), atol=1e-6, rtol=0)


def test_gradients_ema_and_checkpoint_parity(tmp_path):
    torch.manual_seed(42)
    config = Config()
    model = OrbitalJEPA()
    context, future = torch.randn(32, 4), torch.randn(32, 4)
    loss = model.objective(context, future, config)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.target_encoder.parameters())
    for module in (model.encoder, model.predictors, model.projection):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
    old = [p.clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.01)
    ema_update(model.target_encoder, model.encoder, config.ema)
    for before, target, online in zip(old, model.target_encoder.parameters(), model.encoder.parameters()):
        torch.testing.assert_close(target, before*config.ema + online*(1-config.ema))
        assert not torch.equal(before, target)
    path = tmp_path/'checkpoint.pt'
    torch.save({'config': asdict(config), 'model': model.state_dict()}, path)
    restored, restored_config, _ = load_checkpoint(path)
    assert restored_config == config
    torch.testing.assert_close(restored.projection(restored.encoder(context)),
                               model.projection(model.encoder(context)), rtol=0, atol=0)
    torch.testing.assert_close(restored.objective(context, future, config).total,
                               model.objective(context, future, config).total, rtol=0, atol=0)


def test_spectral_sinusoids_and_circular_scaling():
    dt, samples = 0.05, 16384
    time = np.arange(samples)*dt
    for frequency in (0.021, 0.1537, 0.437):
        signal = np.sin(2*np.pi*frequency*time+1.3)
        assert abs(strongest_peak(signal, dt)/frequency-1) < 0.002
    axes = np.geomspace(0.5, 4, 32)
    recovered = np.array([strongest_peak(orbit(a, 0, 0.7, 1.2, samples), dt) for a in axes])
    report = scaling_report(axes, recovered)
    assert abs(report['slope']+1.5) < 0.02
    assert report['mean_relative_frequency_error'] < 0.002
    with pytest.raises(ValueError):
        strongest_peak(np.ones(128), dt)


def test_cli_requires_explicit_training_steps():
    with pytest.raises(SystemExit) as missing:
        main(['train'])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as empty:
        main([])
    assert empty.value.code == 2


def test_dominant_peak_keeps_harmonics_and_reports_concentration():
    dt, samples, frequency = 0.05, 16384, 0.1537
    time = np.arange(samples)*dt
    signal = np.sin(2*np.pi*frequency*time) + 3*np.sin(4*np.pi*frequency*time)
    peak = spectral_peak(signal, dt)
    assert abs(peak['frequency']/(2*frequency)-1) < 0.002
    assert 0 < peak['three_bin_power_fraction'] <= 1
    assert peak['temporal_rms'] > 0


def test_initialization_control_preserves_rng_and_uses_saved_weights():
    config = Config()
    rng_before = torch.get_rng_state().clone()
    first, source = initial_model({}, config)
    assert 'reconstructed' in source
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    with torch.no_grad():
        first.encoder[0].bias.add_(1)
    restored, source = initial_model({'initial_model': first.state_dict()}, config)
    assert source == 'saved zero-step weights'
    torch.testing.assert_close(restored.encoder[0].bias, first.encoder[0].bias, rtol=0, atol=0)


def test_full_prediction_evaluation_scores_actual_future_and_persistence():
    class KnownTransition:
        def encoder(self, x):
            return x.repeat(1, 8)

        target_encoder = encoder

        def projection(self, state):
            return state.reshape(-1, 4, 8)

        def predict_factors(self, state):
            return self.projection(state+1)

    observations = torch.arange(7, dtype=torch.float32)[None, :, None].repeat(3, 1, 4)
    report = evaluate_prediction(KnownTransition(), observations)
    assert report['trajectories'] == 3 and report['transitions'] == 18
    assert report['mse'] == {'learned': 0.0, 'teacher_persistence': 1.0, 'online_persistence': 1.0}
    assert report['improvement_over_teacher_persistence'] == 1.0


def test_five_step_smoke_and_analysis(tmp_path):
    # This is the only optimizer execution in the suite: exactly five steps.
    output = tmp_path/'smoke'
    config = Config(orbits=16, batch_size=32, smoke=True)
    path = train(config, 5, output)
    reports = analyze(path)
    model, saved_config, checkpoint = load_checkpoint(path)
    assert saved_config == config and checkpoint['steps'] == 5
    torch.manual_seed(config.seed)
    initial = OrbitalJEPA()
    for name, value in initial.state_dict().items():
        torch.testing.assert_close(checkpoint['initial_model'][name], value, rtol=0, atol=0)
    assert not torch.equal(model.projection.raw_basis, initial.projection.raw_basis)
    assert any(not torch.equal(left, right) for left, right in
               zip(model.target_encoder.parameters(), initial.target_encoder.parameters()))
    with (output/'metrics.csv').open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 5 and [int(row['step']) for row in metrics] == list(range(1, 6))
    assert all(np.isfinite(float(value)) for row in metrics for value in row.values())
    assert reports['scientific_accuracy_validated'] is False
    assert reports['optimizer_steps'] == 5 and len(reports['learned_factors']) == 4
    assert reports['baseline_source'] == 'saved zero-step weights'
    assert reports['prediction_evaluation']['transitions'] == 3*(config.samples-1)
    assert len(reports['untrained_factors']) == 4
    assert reports['basis_geometry']['rank'] == 32
    with (output/'frequencies.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 32
    for name in ('config.json', 'normalization.json', 'analysis.json', 'orbit_metadata.json', 'evaluation-provenance.json'):
        json.loads((output/name).read_text())
    with (output/'spectral_diagnostics.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 2*32*5
    for name in ('scaling.png', 'scaling.pdf'):
        assert (output/name).stat().st_size > 1000
    with pytest.raises(FileExistsError):
        train(config, 5, output)
    preserved = (output/'analysis.json').read_bytes()
    with pytest.raises(FileExistsError):
        analyze(path)
    assert (output/'analysis.json').read_bytes() == preserved
    print(f'SMOKE_ARTIFACTS={output}')
