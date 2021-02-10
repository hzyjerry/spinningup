from spinup.utils.run_utils import ExperimentGrid
from spinup import sac_upn_pytorch
from functools import partial
import upn.envs
import torch

# env_name = 'PointIRL-v1'
env_name = 'DriveGen0_01-v03'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-upn-pyt-bench')
    eg.add('env_name', env_name, '', True)
    # eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('seed', 0)
    eg.add('epochs', 200)
    eg.add('steps_per_epoch', 4000)
    # eg.add('ac_kwargs:hidden_sizes', [(32, 32), (64,64)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.add('alpha', [0.1, 0.2, 0.4])
    eg.add('ac_kwargs:hidden_sizes', [(32, 32, 32)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(partial(sac_upn_pytorch, env_name=env_name), num_cpu=args.cpu)