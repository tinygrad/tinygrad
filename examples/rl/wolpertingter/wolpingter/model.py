#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import tinygrad
from tinygrad import nn, Tensor, dtypes

def fanin_init(size, fanin=None) -> Tensor:
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return Tensor.uniform(*size, low=float(-v), high=float(v))

class Actor():
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=128, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight = fanin_init(self.fc1.weight.size())
        self.fc2.weight = fanin_init(self.fc2.weight.size())
        self.fc3.weight = Tensor.uniform(
            *(self.fc3.weight.size()), 
            low=float(-init_w), high=float(init_w)
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        out = self.fc1(x.cast(dtypes.float)).relu()
        out = self.fc2(out).relu()
        out = self.fc3(out).softsign()
        return out

class Critic():
    def __init__(self, nb_states, nb_actions, hidden1=319, hidden2=128, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight = fanin_init(self.fc1.weight.size())
        self.fc2.weight = fanin_init(self.fc2.weight.size())
        self.fc3.weight = Tensor.uniform(
            *(self.fc3.weight.size()),
            low=float(-init_w), high=float(init_w)
        )
    
    def __call__(self, xs:Tensor) -> Tensor:
        x, a = xs
        x = x.cast(dtypes.float)
        a = a.cast(dtypes.float)
        out = self.fc1(x).relu()
        # concatenate along columns
        c_in = out.cat(a, dim=len(a.shape)-1)
        out = self.fc2(c_in).relu()
        out = self.fc3(out)
        return out
