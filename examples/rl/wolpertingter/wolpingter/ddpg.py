#!/usr/bin/env python
# -*- coding: utf-8 -*-

# [reference] Use and modified code in https://github.com/ghliu/pytorch-ddpg

import tinygrad
from tinygrad import nn, Tensor, dtypes
from tinygrad.nn.optim import LAMB

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# might have to change this to "load_state_dict" and manually updating state_dict
def hard_update(target, source):
    for target_tensor, tensor in zip(nn.state.get_parameters(target), nn.state.get_parameters(source)):
        tensor.requires_grad = False
        target_tensor.replace(tensor)
        tensor.requires_grad = True

class DDPG(object):
    def __init__(self, args, nb_states, nb_actions):
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states =  nb_states
        self.nb_actions= nb_actions

        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        print(f'Initialized DDPG with actor parameters: {len(nn.state.get_parameters(self.actor))}, lr={args.p_lr}')
        print(f'Initialized DDPG with critic parameters: {len(nn.state.get_parameters(self.critic))}, lr={args.c_lr}')
        
        self.actor_optim = LAMB(params=[self.actor.fc1.weight, self.actor.fc2.weight, self.actor.fc3.weight], lr=args.p_lr, weight_decay=args.weight_decay, adam=True)
        self.critic_optim = LAMB(params=[self.critic.fc1.weight, self.critic.fc2.weight, self.critic.fc3.weight], lr=args.c_lr, weight_decay=args.weight_decay, adam=True)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions,
                                                       theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau_update = args.tau_update
        self.gamma = args.gamma

        # Linear decay rate of exploration policy
        self.depsilon = 1.0 / args.epsilon
        # initial exploration rate
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.continious_action_space = False

    def update_policy(self):
        pass

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            # print(f'typese of memory: s_t: {self.s_t}, a_t: {self.a_t}')
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1., 1., self.nb_actions)
        # self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        # proto action
        if type(s_t) is tuple and s_t[1] == {}:
            s_t = s_t[0]
        #print(f's_t: {s_t}')
        orig_tensor = Tensor([np.array(list(s_t), dtype=np.float32)], dtype=dtypes.float, requires_grad=False)
        action = self.actor(orig_tensor).numpy().squeeze(0)
        action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        # self.a_t = action
        return action

    def reset(self, s_t):
        self.s_t = s_t
        self.random_process.reset_states()

    def load_weights(self, dir):
        if dir is None: return

        # load all tensors to CPU
        ml = lambda storage, loc: storage

        state_dict_actor = nn.state.safe_load(
            'output/{}/actor.safetensors'.format(dir)
        )
        nn.state.load_state_dict(
            self.actor, state_dict_actor
        )

        state_dict_critic = nn.state.safe_load(
            'output/{}/critic.safetensors'.format(dir)
        )
        nn.state.load_state_dict(
            self.critic, state_dict_critic
        )
        print('model weights loaded')


    def save_model(self,output):
        state_dict_actor = nn.state.get_state_dict(self.actor)
        nn.state.safe_save(
            state_dict_actor,
            '{}/actor.safetensors'.format(output)
        )

        state_dict_critic = nn.state.get_state_dict(self.critic)
        nn.state.safe_save(
            state_dict_critic,
            '{}/critic.safetensors'.format(output)
        )

    def seed(self,seed):
        Tensor.manual_seed(seed)
