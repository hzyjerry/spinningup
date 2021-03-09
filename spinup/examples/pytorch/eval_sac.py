import time
import joblib
import upn.envs
import gym
import os
import os.path as osp
import tensorflow as tf
import torch
import numpy as np
from tqdm import tqdm
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph


def load_policy_and_env(fpath, itr='last', deterministic=False, return_model=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action, model = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action, model = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    #import pdb; pdb.set_trace()
    try:
        # import pdb; pdb.set_trace()
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        import gym
        env = gym.make(state["env_name"])
        # env = state['env']
    except:
        env = None
        # import gym
        # state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        # env = gym.make(state["env_name"])


    if return_model:
        return env, get_action, model
    else:
        return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action, model


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)
    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action, model


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=False, params={}, verbose=False):

    from upn.visualize.render import forward_env
    from numpngw import write_apng


    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    test_envs, test_env_names = [], params["test_env_names"][0]
    for name in test_env_names:
        test_envs.append(gym.make(name))

    logger = EpochLogger()
    for env_name, env in zip(test_env_names, test_envs):
        all_feats = []
        all_rews = []
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        coeff = o[-env.coeff_dim:]
        acs = []
        pbar = tqdm(total=num_episodes)
        while n < num_episodes:
            #import pdb; pdb.set_trace()
            if render:
                env.render()
                time.sleep(1e-3)
            # import pdb; pdb.set_trace()
            a = get_action(o)
            acs.append(a)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1
            if "all_feats" in info.keys():
                all_feats.append(info["all_feats"])

            if d or (ep_len == max_ep_len):
                if verbose:
                    print(f"Coeff: {coeff}")
                    print(f"All feats", np.array(all_feats).sum(axis=0))
                # import pdb; pdb.set_trace()
                logger.store(**{f"{env_name}_EpRet": ep_ret})
                logger.store(**{f"{env_name}_EpLen": ep_len})
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                all_rews.append(ep_ret)
                if verbose:
                    print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                print(f"{env_name}: reward {ep_ret:.03f}")
                if render:
                    frames = forward_env(env, np.array(acs), batch=False, subrender=False, resize=0.4)
                    fps = 10
                    fname = f"{env_name}_{n:02d}_rew_{ep_ret:.03f}.png"
                    #os.makedirs(osp.dirname(fname), exist_ok=True)
                    write_apng(os.path.join(args.folder, fname), frames, delay=1000/fps)

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                o = env.reset()

                all_feats = []
                acs = []
                n += 1
                pbar.update(1)
        print(f"{env_name}: mean reward {np.mean(all_rews):.03f}")
        pbar.close()

        logger.log_tabular(f'{env_name}_EpRet', with_min_and_max=True)
        logger.log_tabular(f'{env_name}_EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse, yaml
    from dotmap import DotMap
    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', type=str, default=None)
    parser.add_argument('--norender', '-nr', action='store_true')

    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=60)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    norender = args.norender
    folder = args.folder
    deterministic = args.deterministic
    # import pdb; pdb.set_trace()
    if args.params_file is not None:
        with open(args.params_file) as f:
            args = DotMap(yaml.load(f))

    if folder is None:
        folder = args.folder
    if not norender:
        args.episodes = args.render_episodes

    params = {}
    with open(osp.join(folder, "params.yaml")) as f:
        params = yaml.load(f)
    env, get_action = load_policy_and_env(folder,
                                          args.itr if args.itr >=0 else 'last',
                                          deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(norender), params=params)
