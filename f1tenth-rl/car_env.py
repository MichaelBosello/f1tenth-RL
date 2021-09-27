import numpy as np
import os
import random
import time

import rospy

from state import State
from car.car_control import Drive
from car.safety_control import SafetyControl
from car.sensors import Sensors

MAX_STOP = 3

# you can set the reward according to the action performed or according to the linear velocity of the car
USE_VELOCITY_AS_REWARD = True
ADD_LIDAR_DISTANCE_REWARD = True
LIDAR_DISTANCE_WEIGHT = 0.01

VELOCITY_NORMALIZATION = 0.3 # normalize the velocity between 0 and 1 (e.g. max velocity = 1.8 => 1.8*0.55 =~ 1)
REWARD_SCALING = 0.09 # scale the velocity rewards between [0, REWARD_SCALING]. I.e. at max velocity the reward is REWARD_SCALING

class CarEnv:
    
    def __init__(self, args):
        self.history_length = args.history_length
        self.is_simulator = args.simulator
        self.add_velocity = args.add_velocity
        rospy.init_node('rl_driver')
        self.sensors = Sensors(is_simulator=args.simulator, use_back_sensors=args.use_back_sensors)
        self.control = Drive(self.sensors, is_simulator=args.simulator)
        self.safety_control = SafetyControl(self.control, self.sensors, is_simulator=args.simulator)
        time.sleep(4)

        # available actions
        self.action_set = [0, 1, 2, 3]

        self.game_number = 0
        self.step_number = 0
        self.is_terminal = False

        self.reset_game()

    def step(self, action):
        self.is_terminal = False
        self.step_number += 1
        self.episode_step_number += 1

        if self.safety_control.emergency_brake:
            if self.is_simulator:
                self.safety_control.disable_safety()
                time.sleep(0.3)
                self.control.backward_until_obstacle()
                self.safety_control.enable_safety()
                self.safety_control.unlock_brake()
                time.sleep(0.3)
            else:
                self.safety_control.disable_safety()
                time.sleep(0.6)
                self.control.backward_until_obstacle()
                time.sleep(0.4)
                self.safety_control.enable_safety()
                self.safety_control.unlock_brake()
                # if you select right/left from stop state, the real car turn the servo without moving..
                self.control.forward()
                time.sleep(0.4)

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
        elif action == 3:
            self.control.slowdown()
            reward = 0
        elif action == 4:
            self.control.lightly_right()
            reward = 0.01
        elif action == 5:
            self.control.lightly_left()
            reward = 0.01
        elif action == 6:
            self.control.stop()
            reward = -0.01
            self.car_stop_count += 1
        else:
            raise ValueError('`action` should be between 0 and ' + str(len(self.action_set)-1))

        if action != 6:
            self.car_stop_count = 0

        if self.car_stop_count > MAX_STOP * self.history_length:
            self.control.forward()

        self.state = self.state.state_by_adding_data(self._get_car_state())

        if USE_VELOCITY_AS_REWARD:
            reward = self.sensors.get_car_linear_velocity() * VELOCITY_NORMALIZATION * REWARD_SCALING

        if ADD_LIDAR_DISTANCE_REWARD:
            reward += min(list(self.sensors.get_lidar_ranges())) * LIDAR_DISTANCE_WEIGHT

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
        current_data = list(self.sensors.get_lidar_ranges())
        if self.add_velocity:
            current_data.append(self.sensors.get_car_linear_velocity())
        return current_data


    def get_state_size(self):
        if self.add_velocity:
            return len(self.state.get_data()[0])
        else:
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
