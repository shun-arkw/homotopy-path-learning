import gymnasium as gym
from hc_envs.register_env import register_bezier_env
register_bezier_env()

env = gym.make("BezierHomotopyUnivar-v0", degree=10, bezier_degree=2, latent_dim_m=8,
               episode_len_T=8, alpha_z=2.0, failure_penalty=1e6, seed=0)
obs, info = env.reset()
obs, r, term, trunc, info = env.step(env.action_space.sample())
print(r, term, info)
