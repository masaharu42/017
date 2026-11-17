"""No-pytest checks for the exploratory pair, without training or GPU updates."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
from ebsd_diffusion.mechanics_runtime import MechanicsRuntime
from scripts.run_mechanics_pair import build_configs, assert_pair


def main():
    protocol = yaml.safe_load(Path('configs/mechanics_cnn_feedback_500_exploratory.yaml').read_text())
    _, configs, _ = build_configs(protocol)
    assert_pair(configs)
    for name in ('C0', 'C1'):
        m = configs[name]['train_diffusion']['mechanics']
        assert m['exploratory_fixed_lambda'] == 0.1 and m['calibration_batches'] == 50
        assert configs[name]['train_diffusion']['max_steps'] == 500
    assert configs['C0']['train_diffusion']['initialization'] == configs['C1']['train_diffusion']['initialization']
    with TemporaryDirectory() as directory:
        runtime = MechanicsRuntime.__new__(MechanicsRuntime)
        runtime.cfg = {'calibration_batches': 10, 'exploratory_fixed_lambda': .1}
        runtime.exploratory, runtime.active = True, [True, True]
        runtime.signature = runtime.decision_sha = 'synthetic'
        runtime.proxy_path = Path(directory) / 'proxy.json'
        runtime.proxy_path.write_text('{}')
        row = {'mse_norm': 1., 'mechanics_norm': 2., 'peak_norm': 2., 'strain_norm': 2.}
        runtime.records = [row] * 3
        try:
            runtime.calibration_result()
        except RuntimeError as e:
            assert 'Too few active' in str(e)
        else:
            raise AssertionError('Inactive probe was accepted')
        runtime.records.append(row)
        result = runtime.calibration_result()
        assert result['lambda'] == .1 and result['optimizer_updates'] == 0
    print(json.dumps({'status':'passed', 'scope':'synthetic configuration and calibration logic only',
                      'lambda_c0':0, 'lambda_c1':0.1, 'steps_each':500,
                      'actual_model_or_data_checked':False}, indent=2))

if __name__ == '__main__':
    main()
