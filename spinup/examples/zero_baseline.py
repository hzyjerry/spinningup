import upn.envs
import gym
import numpy as np

#env = gym.make("ReacherIRL-v0")
env = gym.make("PointIRL-v0")
done = False
all_rew = 0
obs = env.reset()
import pdb; pdb.set_trace()

print(env.observation_space)
print(env.action_space)
while not done:
    # ac = np.zeros(env.action_space.shape)
    ac = obs[-2:]
    # ac = env.action_space.sample()
    # ac = np.array([1, 1])
    obs, rew, done, _ = env.step(ac)
    print(obs)
    all_rew += rew


#PointIRL-v0 optimal reward -1.12
env.render()
print(all_rew)