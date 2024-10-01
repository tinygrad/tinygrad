#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ddpg import DDPG
import action_space
from action_space import Discrete_space, Space
from util import *

import tinygrad
from tinygrad import Tensor, nn, dtypes

import numpy as np
np.bool = np.bool_

def soft_update(target, source, tau_update):
    for target_tensor, tensor in zip(nn.state.get_parameters(target), nn.state.get_parameters(source)):
        tensor.requires_grad = False
        target_tensor.replace(target_tensor * (1.0 - tau_update) + tensor * tau_update)
        tensor.requires_grad = True

def criterion(input: Tensor, target: Tensor) -> Tensor:
    return ((target-input).pow(2)).mean()

class WolpertingerAgent(DDPG):

    def __init__(self, continuous, max_actions, action_low, action_high, nb_states, nb_actions, args, k_ratio=0.1):
        super().__init__(args, nb_states, nb_actions)
        self.experiment = args.id
        # according to the papers, it can be scaled to hundreds of millions
        if continuous:
            self.action_space = Space(action_low, action_high, args.max_actions)
            self.k_nearest_neighbors = max(1, int(args.max_actions * k_ratio))
        else:
            self.action_space = Discrete_space(max_actions)
            self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))


    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def wolp_action(self, s_t, proto_action):
        # get the proto_action's k nearest neighbors
        raw_actions, actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)
        raw_actions = raw_actions.astype(np.float32)
        actions = actions.astype(np.float32)
        
        if not isinstance(s_t, np.ndarray):
            if isinstance(s_t, tuple):
                s_t = np.array(s_t[0]).astype(np.float32)
            else:
                s_t = s_t.numpy().astype(np.float32)
        # make all the state, action pairs for the critic
        s_t = np.tile(s_t, [raw_actions.shape[1], 1])

        s_t = s_t.reshape(len(raw_actions), raw_actions.shape[1], s_t.shape[1]) if self.k_nearest_neighbors > 1 \
            else s_t.reshape(raw_actions.shape[0], s_t.shape[1])
        raw_actions = Tensor(raw_actions, dtype=dtypes.float, requires_grad=False)
        s_t = Tensor(s_t, dtype=dtypes.float, requires_grad=False)

        # evaluate each pair through the critic
        actions_evaluation = self.critic([s_t, raw_actions])

        # find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation.numpy(), axis=1)# 0)[0].repeat(len(actions_evaluation.numpy().flatten()))
        max_index = max_index.reshape(len(max_index),)
        
        raw_actions = raw_actions.numpy().astype(np.float32)
        # return the best action, i.e., wolpertinger action from the full wolpertinger policy
        if self.k_nearest_neighbors > 1:
            return raw_actions[[i for i in range(len(raw_actions))], max_index, [0]].reshape(len(raw_actions),1), \
                   actions[[i for i in range(len(actions))], max_index, [0]].reshape(len(actions),1)
        else:
            return raw_actions[max_index], actions[max_index]

    def select_action(self, s_t, decay_epsilon=True):
        # taking a continuous action from the actor
        proto_action = super().select_action(s_t, decay_epsilon)

        #if type(s_t) is tuple and s_t[1] == {}:
        #    s_t = s_t[0]
        raw_wolp_action, wolp_action = self.wolp_action(s_t, proto_action)
        assert isinstance(raw_wolp_action, np.ndarray)
        self.a_t = raw_wolp_action
        # return the best neighbor of the proto action, this is an action for env step
        return wolp_action[0]  # [i]

    def random_action(self):
        proto_action = super().random_action()
        raw_action, action = self.action_space.search_point(proto_action, 1)
        raw_action = raw_action.astype(np.float32)
        action = action.astype(np.float32)
        
        raw_action = raw_action[0]
        action = action[0]
        assert isinstance(raw_action, np.ndarray)
        self.a_t = raw_action
        return action[0] # [i]

    def select_target_action(self, s_t):
        proto_action = self.actor_target(s_t)
        proto_action = proto_action.clamp(-1.0, 1.0).numpy().astype(np.float32)
        raw_wolp_action, wolp_action = self.wolp_action(s_t, proto_action)
        return raw_wolp_action

    def update_policy(self):
        # Sample batch
        with Tensor.train():
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
    
            # Prepare for the target q batch
            # the operation below of critic_target does not require backward_P
            next_state_batch = Tensor(next_state_batch, dtype=dtypes.float, requires_grad=False)
            next_wolp_action_batch = self.select_target_action(next_state_batch)
            next_q_values = self.critic_target((
                next_state_batch,
                Tensor(next_wolp_action_batch, dtype=dtypes.float, requires_grad=False)
            ))
            # but it requires bp in computing gradient of critic loss
            # next_q_values.volatile = False
    
            # next_q_values = 0 if is terminal states
            target_q_batch = Tensor(reward_batch, dtype=dtypes.float, requires_grad=False) + \
                             self.gamma * \
                             Tensor(terminal_batch.astype(np.float32), dtype=dtypes.float, requires_grad=False) * \
                             next_q_values

            # Critic update
            state_batch = Tensor(state_batch, dtype=dtypes.float, requires_grad=False)
            action_batch = Tensor(action_batch, dtype=dtypes.float, requires_grad=False)
            q_batch = self.critic([state_batch, action_batch])
    
            value_loss = (q_batch-target_q_batch).pow(2).mean()
            self.critic_optim.zero_grad()  # Clears the gradients of all optimized tinygrad.Tensor s.
            value_loss.backward()  # computes gradients
            self.critic_optim.step()  # updates the parameters
    
            # Actor update
    
            # self.actor(to_tensor(state_batch)): proto_action_batch
            self.actor_optim.zero_grad()
            policy_loss = -self.critic([state_batch, self.actor(state_batch)]).mean()
            policy_loss.backward()
            self.actor_optim.step()
    
            # Target update
            soft_update(self.actor_target, self.actor, self.tau_update)
            soft_update(self.critic_target, self.critic, self.tau_update)
