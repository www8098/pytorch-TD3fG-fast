#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
from gym.wrappers import TimeLimit
from tensorboardX import SummaryWriter
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from normalized_env import *
from evaluator import Evaluator
from td3 import TD3
from mjrl.utils.gym_env import GymEnv

from parameters import get_args
from trainer import *
import mj_envs
from mjrl.utils.gym_env import GymEnv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
args = get_args()

def get_env(rank):
    env = GymEnv(args.env)
    # env = gym.make(args.env)
    # env.set_seed(args.seed + rank)
    # env.seed(args.seed)
    return env


def make_env(rank):
    def _init():
        env = get_env(rank)
        return env
    return _init


if __name__ == "__main__":
    if args.mode != 'BC+FineTune':
        args.output = get_output_folder(args.output, args.env)
    else:
        args.output = get_output_folder('bc_output', args.env)
        if not os.path.exists('bc_output/' + args.env):
            os.mkdir('bc_output/' + args.env)

    # test the model
    if args.resume == 'default':
        args.resume = 'output/{}-run1'.format(args.env)
    args.resume = 'bc_output/pen-expert-filterQ'
    args.resume = 'output/pen-TD3fG-Reply'

    # env = gym.make(args.env)
    env = GymEnv(args.env)
    # env = SubprocVecEnv([make_env(i) for i in range(args.thread)])
    # env = NormalizedEnv(env)
    # env = TimeLimit(env, 200)
    # env = reward_clip(env)
    # go to gym/env/box2d/nipedal_walker to change the probability
    
    writer = SummaryWriter(log_dir='{}/tensorboardx_log'.format(args.output))

    if args.seed > 0:
        env.set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        env.action_space.np_random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    nb_states = env.observation_space.shape[0] * args.window_length
    nb_actions = env.action_space.shape[0]


    # initialize the generator
    # noise_generator = directed_noise(args, nb_states, nb_actions)
    # adress = 'bc_output/door-100-10'
    # adress = 'bc_output/door-v0'
    adress = 'bc_output/pen-expert-filterQ'
    noise = teacher(nb_states, nb_actions, args, adress)
    agent = TD3(nb_states, nb_actions, args, noise=noise)
    evaluate = Evaluator(args.validate_episodes,
                         args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    "BC path"
    # agent.load_weights('bc_output/door-100-10')
    # agent.load_weights('bc_output/HalfCheetah-2.5')
    # agent.load_weights('bc_output/pen-expert-filterQ')


    if args.mode == 'train':
        train(writer, args, agent, env, evaluate, 
            debug=args.debug, num_interm=args.num_interm, visualize=False)

    elif args.mode == 'test':
        test(writer, args.validate_episodes, agent, env, args.window_length, evaluate, args.resume,
            visualize=True, debug=args.debug)
    
    elif args.mode == 'bc':
        bc = behavior_clone(args, nb_states, nb_actions, train_Q=False)
        bc.clone(agent, args.bc_steps)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
