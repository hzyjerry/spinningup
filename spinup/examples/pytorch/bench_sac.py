from spinup.utils.run_utils import ExperimentGrid
from spinup import sac_upn_pytorch, sac_pytorch
from functools import partial
import upn.envs
import torch


if __name__ == '__main__':
    import argparse, yaml
    from dotmap import DotMap

    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', type=str, default=None)
    parser.add_argument('--cloud', default=False, action="store_true")
    args = parser.parse_args()
    assert args.params_file is not None
    params = None
    with open(args.params_file) as f:
        params = DotMap(yaml.load(f))


    eg = ExperimentGrid(name=params.name)
    eg.add('env_name', params.env_name, '', True)
    eg.add('test_env_names', params.test_env_names, "test", False)
    eg.add('seed', params.seed)
    eg.add('epochs', params.epochs)
    eg.add('steps_per_epoch', params.steps_per_epoch)
    eg.add('update_after', params.update_after)
    eg.add('alpha', params.alpha, 'alp', True)
    eg.add('ac_kwargs:hidden_sizes', params.hidden_sizes, 'hid')

    activation = None
    if params.activation == "relu":
        activation = torch.nn.ReLU
    elif params.activation == "tanh":
        activation = torch.nn.Tanh

    if args.cloud:
        params.save_dir = params.cloud_dir

    eg.add('ac_kwargs:activation', activation, '')

    if params.use_upn:
        eg.run(partial(sac_upn_pytorch, env_name=params.env_name, load_dir=params.load_dir, num_procs=params.num_procs), num_cpu=params.cpu, data_dir=f"{params.save_dir}/{params.date}", params=params.toDict(), pickle=params.pickle)
    else:
        eg.run(partial(sac_pytorch, env_name=params.env_name, load_dir=params.load_dir, num_procs=params.num_procs), num_cpu=params.cpu, data_dir=f"{params.save_dir}/{params.date}", params=params.toDict(), pickle=params.pickle)
