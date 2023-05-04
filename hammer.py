import gym
import mj_envs
import click
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from normalized_env import *

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_policy.py --env_name door-v0 \n
    $ python visualize_policy.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', default='hammer-v0')
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)


def main(env_name, policy, mode, seed, episodes):
    env = GymEnv(env_name)
    # env = gym.make(env_name)
    # env = NormalizedEnv(env)
    # env.set_seed(seed)
    env.reset()
    episode_steps = 0
    total_reward = 0

    while True:
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        print(reward)
        total_reward += reward
        episode_steps += 1

        if episode_steps >= 500:
            done = True

        if done:
            print("reward is {}".format(total_reward))
            env.reset()
            total_reward = 0
            episode_steps = 0

        env.render()

if __name__ == '__main__':
    # env = gym.make('hammer-v0')
    main()