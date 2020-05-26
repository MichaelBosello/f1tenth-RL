import math
import numpy as np
import os
import tensorflow as tf

import state
from state import State

class DeepQNetwork:
    def __init__(self, num_actions, state_size, base_dir, args):
        
        self.num_actions = num_actions
        self.state_size = state_size
        self.history_length = args.history_length

        self.gamma = args.gamma
        self.target_model_update_freq = args.target_model_update_freq
        self.normalize_weights = args.normalize_weights

        self.base_dir = base_dir
        self.save_model_freq = args.save_model_freq
        self.tensorboard_freq = args.tensorboard_logging_freq


        self.behavior_net = self.build_q_net()
        self.target_net = self.build_q_net()

        if args.model is not None:
            pass #load
        else:
            pass #new
    def build_q_net(self):
        pass
        
    def inference(self, state):
        pass
        
    def train(self, batch, step_number):

        old_states = [b.old_state.get_data() for b in batch]
        new_states = [b.new_state.get_data() for b in batch]

        actions = np.zeros((len(batch), self.num_actions))
        updates = np.zeros(len(batch))
        
        for i in range(0, len(batch)):
            actions[i, batch[i].action] = 1
            if batch[i].terminal:
                updates[i] = batch[i].reward
            else:
                updates[i] = batch[i].reward + self.gamma * np.max(y2[i])

        if  step_number % self.tensorboard_freq == 0:
            pass #log

        if step_number % self.target_model_update_freq == 0:
          pass #copy target

        if step_number % self.save_model_freq == 0:
            dir = self.base_dir + '/models'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            # save model
