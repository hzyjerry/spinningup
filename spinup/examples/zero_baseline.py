import upn.envs
import gym
import numpy as np

#env = gym.make("ReacherIRL-v0")
# env = gym.make("PointIRL-v0")
env = gym.make("DriveGen0_01-v01")
done = False
all_rew = 0
obs = env.reset()
t = 0

print(env.observation_space)
print(env.action_space)
while not done:
    # ac = np.zeros(env.action_space.shape)
    # ac = obs[-2:]
    ac = env.action_space.sample()
    # ac = np.array([1, 1])
    obs, rew, done, _ = env.step(ac)
    print(obs)
    all_rew += rew
    t += 1

#PointIRL-v0 optimal reward -1.12
env.render()
print(f"Rews {all_rew:.03f} t {t}")