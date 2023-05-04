import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
from tensorboardX import SummaryWriter
from normalized_env import *
from evaluator import Evaluator
from td3 import TD3
from util import *
import pickle
import random
from memory import RingBuffer
from model import (Actor, Critic)
import torch.nn as nn


def train(writer, args, agent, env,  evaluate, debug=False, num_interm=25, visualize=False):
    
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    ob_buffer = RingBuffer(args.window_length)
    best_val_r = float('-inf')
    last_val_r = float('-inf')

    while step < args.train_iter:            # 使用step做遍历限制

        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            for _ in range(args.window_length):
                ob_buffer.append(observation)
            agent.reset(observation)                    # s_t = observation， self.random_process.reset_states()

        # agent pick TD3+TD3+TD3+ActionNoise ...
        if step <= args.warmup:
            # TD3+TD3+TD3+ActionNoise = agent.random_action()
            action = env.action_space.sample()
            agent.a_t = action
            if agent.noise:
                with torch.no_grad():
                    agent.ra_t = agent.noise(observation)
            else:
                agent.ra_t = action.detach().clone()
        else:
            ob = []
            for i in range(args.window_length):
                ob.extend(ob_buffer[i])
            action = agent.select_action(ob)

        # visualize    
        if visualize:
            env.render()
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)                                       # deep copy
        ob_buffer.append(deepcopy(observation2))

        # [optional] 一般环境回自动重置，但是有些环境不会，所以需要手动重置
        if args.max_episode_length and episode_steps >= args.max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)                                   # observe(self, r_t, s_t1, done), 写入memory
        if step > args.warmup:
            agent.update_policy(step)
        
        # [optional] evaluate
        if evaluate is not None and args.validate_steps > 0 and (step+1) % args.validate_steps == 0:
            # done = True # 打断训练
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward, success_rate = evaluate(env, policy, args.window_length, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
            last_val_r = validate_reward
            writer.add_scalar('data/success_rate', success_rate, step)

        # [optional] save intermideate model
        if (step+1) % args.validate_steps == 0:
            if last_val_r >= best_val_r:
                prYellow('find better model')
                best_val_r = last_val_r
                agent.save_model(args.output)
            else:
                prYellow('drop the model')

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done and step > args.warmup + 1: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))
            agent.select_action(ob)
            agent.memory.append(
                observation,
                agent.a_t,
                agent.ra_t,
                0., False
            )
            
            # print('Total reward:' + str(episode_reward))

            writer.add_scalar('data/iter_reward_sum', episode_reward, step)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

    writer.close()


def test(writer, num_episodes, agent, env, window_length, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward, success_rate = evaluate(env, policy, window_length, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{} steps:{}'.format(i, validate_reward, i))
        writer.add_scalar('test_data/iter_reward_sum', validate_reward, i)


# behavior clone
class behavior_clone(object):

    def __init__(self, args, state_dim, action_dim, train_Q=False):
        self.length = args.demonstration_length
        self.demonstration_ratio = args.demonstration_ratio
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.max_ep_len = args.max_episode_length
        self.window_length = args.window_length
        self.env = args.env
        self.train_Q = train_Q
        self.dataset_path = args.demonstration_path
        with open(self.dataset_path, 'rb') as f:
                self.trajectories = pickle.load(f)
        self.traj_lens, self.returns = self.load_demonstration()

    def load_demonstration(self):
        # save all path information into separate lists
        traj_lens, returns = [], []
        for path in self.trajectories:
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())   # save the final reward
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        num_timesteps = sum(traj_lens)

        print('=' * 50)
        print(f'Starting new experiment: {self.dataset_path}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)
        return traj_lens, returns

    # sample a training batch from trajectories
    def get_batch(self, batch_size=64):
        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.demonstration_ratio*num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = self.traj_lens[sorted_inds] / sum(self.traj_lens[sorted_inds])

        # randomly choose demonstration trajectories based on the length
        # prior to choose long trajectory to clone its behavior
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, q = [], [], []
        for i in range(batch_size):
            traj = self.trajectories[int(sorted_inds[batch_inds[i]])]   # extract the trjectory
            si = random.randint(0, traj['rewards'].shape[0] - 1 - self.length)    # randomly choose a state
            
            # get sequences from dataset
            if self.window_length > 1:
                s1 = np.array(traj['observations'][si:si + self.length].reshape(-1, self.state_dim // self.window_length))
                for i in range(1, self.window_length):
                    s2 = np.array(traj['observations'][si + i : si + i + self.length].reshape(-1, self.state_dim // self.window_length))
                    s1 = np.hstack([s1, s2])
                s.extend(s1)
            else:
                s.extend(traj['observations'][si:si + self.length].reshape(-1, self.state_dim))
            a.extend(traj['actions'][si:si + self.length].reshape(-1, self.act_dim))

            if self.train_Q:
                q.extend(traj['Q'][si:si + self.length])
            # important: the TD3+TD3+TD3+ActionNoise is not normalized (because I'm lazy to do that :) )

        if self.window_length > 0:
            s = np.array(s)
            s = torch.from_numpy(s).to(dtype=torch.float32, device='cuda')
        else:
            s = torch.tensor(s).to(dtype=torch.float32, device='cuda')
        a = torch.tensor(a).to(dtype=torch.float32, device='cuda')
        q = torch.tensor(q).to(dtype=torch.float32, device='cuda')

        return s, a, q

    def clone_step(self, agent):
        s, a, q = self.get_batch()
        q = torch.unsqueeze(q, 1)
        criterion = torch.nn.MSELoss(reduce=False, size_average=False)
        state, action_target = torch.clone(s), torch.clone(a)
        if self.train_Q:
            q_target = torch.clone(q)
        action_preds = agent.actor(state)

        actor_loss = criterion(action_preds, action_target).mean()
        agent.actor_optim.zero_grad()
        actor_loss.backward()
        agent.actor_optim.step()

        if self.train_Q:
            q1_preds = agent.critic1([s, a])
            q2_preds = agent.critic2([s, a])
            critic1_loss = criterion(q1_preds, q_target).mean()
            critic2_loss = criterion(q2_preds, q_target).mean()
            agent.critic1_optim.zero_grad()
            agent.critic2_optim.zero_grad()
            critic1_loss.backward()
            critic2_loss.backward()
            agent.critic1_optim.step()
            agent.critic2_optim.step()

        if self.train_Q:
            return actor_loss.detach().cpu().item(), critic1_loss.detach().cpu().item(), critic2_loss.detach().cpu().item()
        else:
            return actor_loss.detach().cpu().item()

    def clone(self, agent, num_steps):
        for i in range(num_steps):
            if self.train_Q:
                ac_loss, q1_loss, q2_loss = self.clone_step(agent)
            else:
                ac_loss = self.clone_step(agent)
            if i % 10000 == 0:
                if self.train_Q:
                    print(f'The {i} steps, actor loss={ac_loss}, critic1 loss={q1_loss}, critic2 loss={q2_loss}')
                else:
                    print(f'The {i} steps, actor loss={ac_loss}')

        agent.save_model('bc_output/{}'.format(self.env))


# directed noise
class directed_noise():

    def __init__(self, args, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.max_ep_len = args.max_episode_length
        self.window_length = args.window_length
        dataset_path = args.demonstration_path
        with open(dataset_path, 'rb') as f:
                self.trajectories = pickle.load(f)
        self.traj_lens, self.returns = self.load_demonstration()
        self.state_demo, self.action_demo = self.demo_batch()       # store the dmeonstration actions and states in numpy array
        
    def load_demonstration(self):
        # save all path information into separate lists
        traj_lens, returns = [], []
        for path in self.trajectories:
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())   # save the final reward
        traj_lens, returns = np.array(traj_lens), np.array(returns)
        num_timesteps = sum(traj_lens)

        print('=' * 50)
        print(f'Starting new experiment: {dataset_path}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)
        return traj_lens, returns
    
    def demo_batch(self, batch_size=64, sample_size=10):
        num_timesteps = sum(self.traj_lens) # when need to shrink the samples, add a num_timsteps ratio
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        p_sample = self.traj_lens[sorted_inds] / sum(self.traj_lens[sorted_inds])

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a = [], []
        for i in range(batch_size):
            traj = self.trajectories[int(sorted_inds[batch_inds[i]])]   # extract the trjectory
            for _ in range(sample_size):
                si = random.randint(0, traj['rewards'].shape[0] - 1)    # randomly choose a state

                # get sequences from dataset
                s.extend(traj['observations'][si].reshape(-1, self.state_dim//self.window_length))
                a.extend(traj['actions'][si].reshape(-1, self.act_dim))

            # important: the TD3+TD3+TD3+ActionNoise is not normalized (because I'm lazy to do that :) )
            
        s = np.array(s)
        a = np.array(a)
        return s, a
    
    def __call__(self, state):
        demo_len = len(self.action_demo)
        l2distance = np.sum((np.array([[state]*demo_len]).squeeze(0) - self.state_demo) ** 2, axis=1)
        sorted_inds = np.argsort(l2distance)
        cloest_action = self.action_demo[sorted_inds[-1]]
        return cloest_action * 0.2

class teacher():
    def __init__(self, nb_states, nb_actions, args, adress):
        super().__init__()
        self.nb_states = nb_states
        self.nb_actions= nb_actions
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor = self.actor.cuda()
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(adress)))
    
    def __call__(self, state):
        action = to_numpy(
            self.actor(to_tensor(np.array([state])))
        ).squeeze(0)
        action = np.clip(action, -1., 1.)
        return action


class teacher_Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=512, hidden3=1024, init_w=3e-3):
        super(teacher_Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc2_5 = nn.Linear(hidden2, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc2_5(out)
            out = self.relu(out)
            out = self.fc3(out)
            out = self.tanh(out)
        return out

