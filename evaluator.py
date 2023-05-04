
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from memory import RingBuffer
from util import *
from copy import deepcopy

@torch.no_grad()
class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None, success_threshold=0):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        self.success_threshold = success_threshold

    def __call__(self, env, policy, window_length, debug=False, visualize=False, save=True):

        self.is_training = False
        observation = None
        result = []
        ob_buf = RingBuffer(window_length)
        observation = env.reset()
        success_count = 0

        for i in range(window_length):
            ob_buf.append(deepcopy(observation))

        for episode in range(self.num_episodes):
            # reset at the start of episode
            observation = env.reset()
            ob_buf.append(deepcopy(observation))
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, TD3+TD3+TD3+ActionNoise ,reward, blablabla ...
                if window_length == 1:
                    action = policy(observation)
                else:
                    ob = []
                    for i in range(window_length):
                        ob.extend(ob_buf[i])
                    action = policy(ob)
                
                observation, reward, done, info = env.step(action)
                ob_buf.append(deepcopy(observation))
                
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    # env.render(mode='human')
                    env.render()
                # update
                episode_reward += reward
                episode_steps += 1

            if episode_reward > self.success_threshold:
                success_count += 1

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])
        success_rate = success_count / self.num_episodes
        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result), success_rate

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
