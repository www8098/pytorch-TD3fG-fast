import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import copy
from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *
from parameters import get_args

args = get_args()
# from ipdb import set_trace as debug

criterion = nn.MSELoss()


class TD3(object):
    def __init__(self, nb_states, nb_actions, args, noise):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        
        self.expl_noise = 0.4

        # for mujoco
        # self.noise_std = 0.1
        # self.noise_clip = 0.3

        # for hand
        self.noise_std = 0.3
        self.final_std = 0.1
        self.noise_clip = 0.1

        self.noise = noise
        self.noise_decay = 1.0 / args.noise_decay

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)                  # input state, hiden1, hiden2, output TD3+TD3+TD3+ActionNoise
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.rate, eps=1e-08, weight_decay=args.L2)

        self.critic1 = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=args.rate, eps=1e-08, weight_decay=args.L2)

        self.critic2 = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=args.rate, eps=1e-08, weight_decay=args.L2)

        hard_update(self.actor_target, self.actor)                                      # Make sure target is with the same weight
        hard_update(self.critic1_target, self.critic1)                                    # copy online model paramets to target model
        hard_update(self.critic2_target, self.critic2)

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.noise_weight = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.ra_t = None # Most recent action from the generator
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()                # 在util中定义，USE_CUDA = torch.cuda.is_available()

    def update_policy(self, step: int):
        # Sample batch
        state_batch, action_batch, refaction_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)        # 采集experience并拆分

        # Prepare for the target q batch
        with torch.no_grad():
            next_action = self.actor_target(to_tensor(next_state_batch))
            noise = torch.Tensor(self.nb_actions).data.normal_(0, self.noise_std).cuda()
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1, 1)
            self.noise_std -= (self.noise_std - self.final_std) / args.train_iter

            target_q1 = self.critic1_target([                                                    # Qt+1 = r + discount * Qt
                to_tensor(next_state_batch),
                next_action,
            ])
            target_q2 = self.critic2_target([                                                    # Qt+1 = r + discount * Qt
                to_tensor(next_state_batch),
                next_action,
            ])
            next_q_values = torch.min(target_q1, target_q2).detach()

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values          # 我就是傻逼！！！为什么要反转terminal！！！

        # Critic1 update
        self.critic1.zero_grad()                                                              # 梯度初始化为0
        q1_batch = self.critic1([to_tensor(state_batch), to_tensor(action_batch)])
        value_loss1 = criterion(q1_batch, target_q_batch)                                         # criterion = nn.MSELoss()
        value_loss1.backward()
        self.critic1_optim.step()

        # Critic2 update
        self.critic2.zero_grad()                                                                 # 梯度初始化为0
        q2_batch = self.critic2([to_tensor(state_batch), to_tensor(action_batch)])
        value_loss2 = criterion(q2_batch, target_q_batch)                                         # criterion = nn.MSELoss()
        value_loss2.backward()
        self.critic2_optim.step()

        # Actor update
        if (step + 1) % args.policy_delay == 0:
            self.actor.zero_grad()
            actor_action_batch = self.actor(to_tensor(state_batch))
            policy_loss = -self.critic1([to_tensor(state_batch), actor_action_batch])
            policy_loss = policy_loss.mean()

            if self.noise:
                supervised_loss = criterion(actor_action_batch, to_tensor(refaction_batch))
                supervised_loss = supervised_loss.mean()

                # hand tasks
                policy_loss = policy_loss*min((1-self.noise_weight), 1) + 3 * supervised_loss*max(self.noise_weight, 0)
                # policy_loss = policy_loss * min((1 - self.noise_weight), 1) + 15 * supervised_loss * max(self.noise_weight, 0)
                # policy_loss = policy_loss*min((1-self.noise_weight) + 0.2, 1) + 2 * supervised_loss*max(self.noise_weight, 0)

            # use Q filter
            # if self.noise:
            #     if self.noise_weight > 0.5:
            #         supervised_loss = criterion(actor_action_batch, to_tensor(refaction_batch))
            #         supervised_loss = supervised_loss.mean()
            #         policy_loss = policy_loss * min((1 - self.noise_weight), 1) + 3 * supervised_loss * max(
            #             self.noise_weight, 0)
            #     else:
            #         with torch.no_grad():
            #             refer_Q = self.critic1([to_tensor(state_batch), to_tensor(refaction_batch)])
            #             refer_Q = refer_Q.mean()
            #         if -policy_loss < refer_Q:
            #             supervised_loss = criterion(actor_action_batch, to_tensor(refaction_batch))
            #             supervised_loss = supervised_loss.mean()
            #             policy_loss = policy_loss * min((1 - self.noise_weight), 1) + 3 * supervised_loss * max(
            #                 self.noise_weight, 0)

            policy_loss.backward()
            self.actor_optim.step()

        # Target update
        with torch.no_grad():
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

    def eval(self):                                                 # test时使用eval(), 使用全局固定BatchNorm， 取消Dropout
        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic1.cuda()
        self.critic1_target.cuda()
        self.critic2.cuda()
        self.critic2_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, self.ra_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):                                        # 全随机action 用于 warm up
        action = np.random.uniform(-1., 1., self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)

        # training模式下，对action添加噪声
        if self.noise:
            # add_noise = self.is_training*max(self.epsilon, 0)*self.random_process.sample()
            add_noise = (np.random.randn(self.nb_actions) * 0.2)
            add_noise = np.clip(add_noise, -self.noise_clip, self.noise_clip)  # for hand
            action = action + add_noise

        action = np.clip(action, -1., 1.)
        # print(action)
        if decay_epsilon:
            self.epsilon -= self.depsilon
            self.noise_weight -= self.noise_decay
        self.a_t = action

        # with torch.no_grad():
        #     self.ra_t = self.noise(s_t)
        # return action * 0.5 + 0.5 * self.ra_t
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic1.load_state_dict(torch.load('{}/critic1.pkl'.format(output)))
        self.critic2.load_state_dict(torch.load('{}/critic2.pkl'.format(output)))

        hard_update(self.actor_target, self.actor)                                      # Make sure target is with the same weight
        hard_update(self.critic1_target, self.critic1)                                    # copy online model paramets to target model
        hard_update(self.critic2_target, self.critic2)

    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic1.state_dict(),
            '{}/critic1.pkl'.format(output)
        )
        torch.save(
            self.critic2.state_dict(),
            '{}/critic2.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
