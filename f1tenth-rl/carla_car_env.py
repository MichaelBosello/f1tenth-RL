import numpy as np
import os
import random
import time

from state import State
from car.carla.carla_env import CarlaEnv

class CarEnv:
    
    def __init__(self, args):
        self.history_length = args.history_length
        self.env = CarlaEnv()
        self.control = self.env.control
        self.sensors = self.env.sensors
        self.safety_control = self.env.safety_control

        # available actions
        self.action_set = [0, 1, 2]

        self.game_number = 0
        self.step_number = 0
        self.is_terminal = False

        self.reset_game()

    def step(self, action):
        self.env.update_view()

        self.is_terminal = False
        self.step_number += 1
        self.episode_step_number += 1

        if self.safety_control.emergency_brake:
            self.safety_control.disable_safety()
            self.control.new_position()
            self.safety_control.enable_safety()
            self.safety_control.unlock_brake()
            reward = -1
            self.is_terminal = True
            self.game_score += reward
            return reward, self.state, self.is_terminal

        reward = 0
        if action == 0:
            self.control.forward()
            reward = 0.08
        elif action == 1:
            self.control.right()
            reward = 0.02
        elif action == 2:
            self.control.left()
            reward = 0.02
        else:
            raise ValueError('`action` should be between 0 and ' + str(len(self.action_set)-1))


        self.state = self.state.state_by_adding_data(self._get_car_state())

        self.game_score += reward
        return reward, self.state, self.is_terminal

    def reset_game(self):
        self.control.stop()

        if self.is_terminal:
            self.game_number += 1
            self.is_terminal = False
        self.state = State().state_by_adding_data(self._get_car_state())
        self.game_score = 0
        self.episode_step_number = 0
        self.car_stop_count = 0

    def _get_car_state(self):
        current_data = list(self.sensors.get_lidar_data())
        return current_data


    def destroy(self):
        self.env.destroy()


    def get_state_size(self):
        return len(self.state.get_data())

    def get_num_actions(self):
        return len(self.action_set)

    def get_state(self):
        return self.state
    
    def get_game_number(self):
        return self.game_number
    
    def get_episode_step_number(self):
        return self.episode_step_number
    
    def get_step_number(self):
        return self.step_number
    
    def get_game_score(self):
        return self.game_score

    def is_game_over(self):
        return self.is_terminal
