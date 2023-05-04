from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import pickle
import numpy as np
import d4rl
import gym

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/memory.py
Experience = namedtuple('Experience', 'state0, action, refaction, reward, state1, terminal1')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class DemoRingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 10000
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
            if self.start < 10000:
                self.start = 10000
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):           # 判断是否包含 shape 属性
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):      # 可迭代？
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=True):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):                              # 不可采样
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    # 然而这个function根本没用， 在SequentialMemory里写了个类似功能的模块
    def get_recent_state(self, current_observation):                                            # 从recent_state中抽取， 填充一个window_size的batch
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):    # ignore_episode_boundaries控制terminal是否起效
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:                                                  # recent state 无法填满 window 时， 用 0 填充
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config


# Memory contains recent_observations and  recent_terminals
class SequentialMemory(Memory): 
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)            # 初始化父类 kwargs 为 dict，  存储 window_length and ignore_episode_boundaries
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        # self.actions = RingBuffer(limit)                            # 为actions rewards terminals observations 创建环形缓冲
        # self.refaction = RingBuffer(limit)
        # self.rewards = RingBuffer(limit)                            # maxlen == limit
        # self.terminals = RingBuffer(limit)
        # self.observations = RingBuffer(limit)

        # f = open('data/pen-expert-filterQ.pkl', 'rb')
        # data = pickle.load(f)
        data = d4rl.qlearning_dataset(gym.make('pen-expert-v0'))
        self.actions = DemoRingBuffer(limit)  # 为actions rewards terminals observations 创建环形缓冲
        self.actions.data[:10000] = data['actions'][:10000]
        self.rewards = DemoRingBuffer(limit)  # maxlen == limit
        self.rewards.data[:10000] = data['rewards'][:10000]
        self.terminals = DemoRingBuffer(limit)
        self.terminals.data[:10000] = data['terminals'][:10000]
        self.observations = DemoRingBuffer(limit)
        self.observations.data[:10000] = data['observations'][:10000]
        self.refaction = DemoRingBuffer(limit)
        self.refaction.data[:10000] = data['actions'][:10000]


    def sample(self, batch_size, batch_idxs=None):                                                      # 返回experience存储s0 s1 TD3+TD3+TD3+ActionNoise reward terminal， state为一串observation序列
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)                  # 从observation里随机抽取batch_size个idxs
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:                                                                            
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]                               # nb_entries == len(self.observations) == 当前长度 ??
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False                              # 随机抽取idx，直到上一步 terminal 为 false
            assert 1 <= idx < self.nb_entries

            # ensure that an experience never spans multiple episodes.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset                                                          # 上一步observation 对应s0
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False    # 用于判断是否完成一个 trajectory ？
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])                                        # 导入 idx 之前 window_length 内的 observation
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))                                         # 用0填充 功能和Memory中的get_recent_state类似
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            refaction = self.refaction[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]                                                   # 复制state0[1:], state1 比 state0 滞后一个observation
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, refaction=refaction, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def sample_and_split(self, batch_size, batch_idxs=None):          # 拆分sample获得的experience， 分别写入buffer
        experiences = self.sample(batch_size, batch_idxs)
        state0_batch = []
        reward_batch = []
        action_batch = []
        refaction_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(np.squeeze(e.state0))
            state1_batch.append(np.squeeze(e.state1))
            reward_batch.append(np.squeeze(e.reward))
            action_batch.append(np.squeeze(e.action))
            refaction_batch.append(np.squeeze(e.refaction))
            if type(e.terminal1) == list:
                terminal1_batch.append(np.squeeze([0. if t else 1. for t in e.terminal1]))  # 0代表terminal为true， 为false时写入1
            else:
                terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch).reshape(batch_size,-1)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)
        refaction_batch = np.array(refaction_batch).reshape(batch_size, -1)

        return state0_batch, action_batch, refaction_batch, reward_batch, state1_batch, terminal1_batch

    def append(self, observation, action, refaction, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)  # 更新recent_observations
        
        # This needs to be understood as follows: in `observation`, take `TD3+TD3+TD3+ActionNoise`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)           # 环形缓冲定义的append， 会改变start保证数据的顺序性
            self.actions.append(action)
            self.refaction.append(refaction)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        return len(self.total_rewards)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config
