from copy import deepcopy
import itertools
import numpy as np
import torch
import os, sys
from torch.optim import Adam
import gym
import upn.envs
import datetime
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from spinup.utils.mpi_tools import proc_id
from stable_baselines3.common.vec_env import SubprocVecEnv

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.feats_buf = {}
        self.ptr, self.size, self.max_size = 0, 0, size

    def store_vec(self, obs, act, rew, next_obs, done, feats):
        for i in range(len(obs)):
            self.store(obs[i], act[i], rew[i], next_obs[i], done[i], feats[i])

    @property
    def feats_keys(self):
        return list(self.feats_buf.keys())

    def store(self, obs, act, rew, next_obs, done, feats):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        for key, val in feats.items():
            if key not in self.feats_buf:
                self.feats_buf[key] = np.zeros(self.max_size, dtype=np.float32)
            self.feats_buf[key][self.ptr] = val

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        feats = {}
        for key, val in self.feats_buf.items():
            feats[key] = torch.as_tensor(self.feats_buf[key][idxs], dtype=torch.float32)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        batch = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        batch["feats"] = feats
        return batch

    def get_size(self):
        total_size = 0
        for obj in [self.obs_buf, self.obs2_buf, self.act_buf, self.rew_buf, self.done_buf] + list(self.feats_buf.values()):
            total_size += sys.getsizeof(obj)
        return total_size


def sac_rl(env_fn, env_name, test_env_fns=[], actor_critic=core.MLPActorCritic, **kwargs):
    env = env_fn()
    coeff_dim = env.coeff_dim
    _actor_critic = partial(actor_critic, coeff_dim=coeff_dim)
    return sac(env_fn, env_name, test_env_fns=test_env_fns, actor_critic=_actor_critic, **kwargs)


def sac_upn(env_fn, env_name, test_env_fns=[], actor_critic=core.MLPActorCriticUPNNaive, **kwargs):
    env = env_fn()
    coeff_dim = env.coeff_dim
    _actor_critic = partial(actor_critic, coeff_dim=coeff_dim)
    return sac(env_fn, env_name, test_env_fns=test_env_fns, actor_critic=_actor_critic, **kwargs)


def sac(env_fn, env_name, test_env_fns=[], actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, load_dir=None, num_procs=1, clean_every=200):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    from spinup.examples.pytorch.eval_sac import load_pytorch_policy


    print(f"SAC proc_id {proc_id()}")
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    if proc_id() == 0:
        writer = SummaryWriter(log_dir=os.path.join(logger.output_dir, str(datetime.datetime.now())), comment=logger_kwargs["exp_name"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SubprocVecEnv([partial(env_fn, rank=i) for i in range(num_procs)], "spawn")
    test_env = SubprocVecEnv(test_env_fns, "spawn")
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    if load_dir is not None:
        _, ac = load_pytorch_policy(load_dir, itr="", deterministic=False)
    else:
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing TD feats-losses
    def compute_loss_feats(data):
        o, a, r, o2, d, feats = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data["feats"]

        feats = torch.stack(list(feats.values())).T # (nbatch, nfeats)
        feats1 = ac.q1.predict_feats(o,a)
        feats2 = ac.q2.predict_feats(o,a)

        feats_keys = replay_buffer.feats_keys

        # Bellman backup for feature functions
        with torch.no_grad():
            a2, _ = ac.pi(o2)

            # Target feature values
            feats1_targ = ac_targ.q1.predict_feats(o2, a2)
            feats2_targ = ac_targ.q2.predict_feats(o2, a2)
            feats_targ = torch.min(feats1_targ, feats2_targ)
            backup = feats + gamma * (1 - d[:, None]) * feats_targ

        # MSE loss against Bellman backup
        loss_feats1 = ((feats1 - backup)**2).mean(axis=0)
        loss_feats2 = ((feats2 - backup)**2).mean(axis=0)
        loss_feats = loss_feats1 + loss_feats2

        # Useful info for logging
        feats_info = dict(Feats1Vals=feats1.detach().numpy(),
                          Feats2Vals=feats1.detach().numpy())

        return loss_feats, feats_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, feats_keys):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Feature loss
        loss_feats, feats_info = compute_loss_feats(data)
        keys = [f"LossFeats_{key}" for key in feats_keys]
        logger.store(**dict(zip(keys, loss_feats)))

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)

    def test_agent(feats_keys):
        num_envs = len(test_env_fns)
        env_ep_rets = np.zeros(num_envs)
        for j in range(num_test_episodes):
            o, d  = test_env.reset(), np.zeros(num_envs, dtype=bool)
            ep_len = np.zeros(num_envs)
            while not (np.all(d) or np.all(ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, info = test_env.step(get_action(o, True))
                env_ep_rets += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        for ti in range(num_envs):
            logger.store(**{f"TestEpRet_{ti}": env_ep_rets[ti] / num_test_episodes})

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), np.zeros(num_procs), np.zeros(num_procs)

    # Main loop: collect experience in env and update/log each epoch
    epoch = 0
    update_times, clean_times = 0, 0
    t = 0
    while t <= total_steps:
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = np.stack([env.action_space.sample() for _ in range(num_procs)])

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        if np.all(ep_len == max_ep_len):
            d.fill(False)

        # Store experience to replay buffer
        replay_buffer.store_vec(o, a, r, o2, d, [inf["features"] for inf in info])

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling, assumes all subenvs end at the same time
        if np.all(d) or np.all(ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)

            if clean_every > 0 and epoch // clean_every >= clean_times:
                env.close()
                test_env.close()
                env = SubprocVecEnv([partial(env_fn, rank=i) for i in range(num_procs)], "spawn")
                test_env = SubprocVecEnv(test_env_fns, "spawn")
                clean_times += 1

            o, ep_ret, ep_len = env.reset(), np.zeros(num_procs), np.zeros(num_procs)

        # Update handling
        if t >= update_after and t / update_every > update_times:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, feats_keys=replay_buffer.feats_keys)
            update_times += 1


        # End of epoch handling
        if t // steps_per_epoch > epoch:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                # try:
                logger.save_state({'env_name': env_name}, None)
                    # logger.save_state({'env': env}, None)
                #except:
                    #logger.save_state({'env_name': env_name}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(replay_buffer.feats_keys)

            # Update tensorboard
            if proc_id() == 0:
                log_perf_board = ['EpRet','EpLen','Q1Vals','Q2Vals'] + [f"TestEpRet_{ti}" for ti in range(len(test_env_fns))]
                log_loss_board = ['LogPi', 'LossPi','LossQ'] + [key for key in logger.epoch_dict.keys() if "LossFeats" in key]
                log_board = {'Performance': log_perf_board, 'Loss': log_loss_board}
                for key,value in log_board.items():
                    for val in value:
                        mean, std = logger.get_stats(val)
                        if key=='Performance':
                            writer.add_scalar(key+'/Average'+val, mean, epoch)
                            writer.add_scalar(key+'/Std'+val, std, epoch)
                        else:
                            writer.add_scalar(key+'/'+val, mean, epoch)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            if proc_id() == 0:
                writer.flush()

                import psutil
                # gives a single float value
                cpu_percent = psutil.cpu_percent()
                # gives an object with many fields
                mem_percent = psutil.virtual_memory().percent
                print(f"Used cpu avg {cpu_percent}% memory {mem_percent}%")
                cpu_separate = psutil.cpu_percent(percpu=True)
                for ci, cval in enumerate(cpu_separate):
                    print(f"\t cpu {ci}: {cval}%")
                # buf_size = replay_buffer.get_size()
                # print(f"Replay buffer size: {buf_size//1e6}MB {buf_size // 1e3} KB {buf_size % 1e3} B")
        t += num_procs

    if proc_id() == 0:
        writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    def env_fn():
        import upn.envs
        return gym.make(args.env)

    sac(env_fn, args.env, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
