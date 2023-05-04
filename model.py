
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


"""four layers for manipulator tasks' actor"""
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=512, hidden3=1024, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        self.fc2_5 = nn.Linear(hidden2, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.tanh(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.tanh(out)
        out = self.relu(out)
        out = self.fc2_5(out)
        # out = self.tanh(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=512, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states + +nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        # nn.init.normal_(self.fc1.weight)
        # nn.init.normal_(self.fc2.weight)
        # nn.init.normal_(self.fc3.weight)
    def forward(self, xs):
        x, a = xs
        out = self.fc1(torch.cat([x,a],1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out