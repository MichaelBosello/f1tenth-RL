#!/usr/bin/env python3
#
from threading import Thread
import sys
import numpy as np
import os
import random
import time
import argparse
import datetime
import tensorflow as tf

import replay
import dqn
from car_env import CarEnv
from state import State

# real car: reduce-lidar-data:36, cut-lidar-data: 4
# simulator: reduce-lidar-data:30, cut-lidar-data: 8


#################################
# parameters
#################################

parser = argparse.ArgumentParser()
# real car or simulator
parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
# agent parameters
parser.add_argument("--learning-rate", type=float, default=0.00038, help="learning rate of the NN")
parser.add_argument("--gamma", type=float, default=0.98, help="""gamma [0, 1] is the discount factor. It determines the importance of future rewards.
                                A factor of 0 will make the agent consider only immediate reward, a factor approaching 1 will make it strive for a long-term high reward""")
parser.add_argument("--epsilon", type=float, default=1, help="]0, 1]for epsilon greedy train")
parser.add_argument("--epsilon-decay", type=float, default=0.999934, help="]0, 1] every step epsilon = epsilon * decay, in order to decrease constantly")
parser.add_argument("--epsilon-min", type=float, default=0.1, help="epsilon with decay doesn't fall below epsilon min")
parser.add_argument("--batch-size", type=float, default=32, help="size of the batch used in gradient descent")

parser.add_argument("--observation-steps", type=int, default=400, help="train only after this many steps (1 step = [history-length] frames)")
parser.add_argument("--target-model-update-freq", type=int, default=500, help="how often (in steps) to update the target model")
parser.add_argument("--model", help="tensorflow model directory to initialize from (e.g. run/model)")
parser.add_argument("--history-length", type=int, default=2, help="(>=1) length of history used in the dqn. An action is performed [history-length] time")
parser.add_argument("--repeat-action", type=int, default=0, help="(>=0) actions are repeated [repeat-action] times. Unlike history-length, it doesn't increase the network size")
parser.add_argument("--gpu-time", type=int, default=0.08, help="""waiting time (seconds) between actions when agent is not training (observation steps/evaluation).
                                It should be the amount of time used by your CPU/GPU to perform a training sweep. It is needed to have the same states and rewards as
                                training takes time and the environment evolves indipendently""")
# lidar pre-processing
parser.add_argument("--reduce-lidar-data", type=int, default=30, help="lidar data are grouped by taking the min of [reduce-lidar-data] elements")
parser.add_argument("--cut-lidar-data", type=int, default=8, help="N element at begin and end of lidar data are cutted. Executed after the grouping")
parser.add_argument("--max-distance-norm", type=float, default=20, help="divide lidar elems by [max-distance-norm] to normalize between [0, 1]")
parser.add_argument("--lidar-reduction-method", choices=['avg', 'max', 'min', 'sampling'], default='avg', type=str.lower, help="method used to aggregate lidar data")
parser.add_argument("--lidar-float-cut", type=int, default=-1, help="how many decimals of lidar ranges to take. -1 for no cutting")

parser.add_argument("--lidar-to-image", type=bool, default=False, help="if true, an image of borders is built from lidar ranges and it is used as state")
parser.add_argument("--show-image", type=bool, default=False, help="show the agent view. [lidar-to-image] must be true to have effect")
parser.add_argument("--image-width", type=int, default=84, help="the width of the image built from lidar data. Applicable if [lidar-to-image] is true")
parser.add_argument("--image-height", type=int, default=84, help="the height of the image built from lidar data. Applicable if [lidar-to-image] is true")
parser.add_argument("--image-zoom", type=int, default=2, help="""zoom lidar image to increase border separation.
                                It must be appropriate for the circuit max distance and image size otherwise out-of-bound exception will be casted""")
# train parameters
parser.add_argument("--train-epoch-steps", type=int, default=3500, help="how many steps (1 step = [history-length] frames) to run during a training epoch")
parser.add_argument("--eval-epoch-steps", type=int, default=500, help="how many steps (1 step = [history-length] frames) to run during an eval epoch")
parser.add_argument("--replay-capacity", type=int, default=100000, help="how many states to store for future training")
parser.add_argument("--prioritized-replay", action='store_true', help="prioritize interesting states when training (e.g. terminal or non zero rewards)")
parser.add_argument("--compress-replay", action='store_true', help="if set replay memory will be compressed with blosc, allowing much larger replay capacity")
parser.add_argument("--save-model-freq", type=int, default=2000, help="save the model once per X training sessions")
parser.add_argument("--logging", type=bool, default=True, help="enable tensorboard logging")
args = parser.parse_args()

print('Arguments: ', (args))


#################################
# setup
#################################

base_output_dir = 'run-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(base_output_dir)

tensorboard_dir = base_output_dir + "/tensorboard/"
os.makedirs(tensorboard_dir)
summary_writer = tf.summary.create_file_writer(tensorboard_dir)
with summary_writer.as_default():
    tf.summary.text('params', str(args), step=0)

State.setup(args)

environment = CarEnv(args)
replay_memory = replay.ReplayMemory(base_output_dir, args)
dqn = dqn.DeepQNetwork(environment.get_num_actions(), environment.get_state_size(),
                        replay_memory, base_output_dir, tensorboard_dir, args)

train_epsilon = args.epsilon #don't want to reset epsilon between epoch
start_time = datetime.datetime.now()
train_episodes = 0
eval_episodes = 0
episode_train_reward_list = []
episode_eval_reward_list = []

#################################
# stop handler
#################################

stop = False

def stop_handler():
  global stop
  while not stop:
    user_input = input()
    if user_input == 'q':
      print("Stopping...")
      stop = True
    if user_input == 'r':
      print("Resetting simulator position...")
      environment.control.reset_simulator()

process = Thread(target=stop_handler)
process.start()

#################################
# training cycle definition
#################################

def run_epoch(min_epoch_steps, eval_with_epsilon=None):
    global train_epsilon
    global train_episodes
    global eval_episodes
    global episode_train_reward_list
    global episode_eval_reward_list
    is_training = True if eval_with_epsilon is None else False
    step_start = environment.get_step_number()
    start_game_number = environment.get_game_number()
    epoch_total_score = 0
    stuck_count = 0

    while environment.get_step_number() - step_start < min_epoch_steps and not stop:
        state_reward = 0
        state = None
        
        episode_losses = []
        while not environment.is_game_over() and not stop:
            # epsilon selection and update
            if is_training:
                epsilon = train_epsilon
                if train_epsilon > args.epsilon_min:
                    train_epsilon = train_epsilon * args.epsilon_decay
                    if train_epsilon < args.epsilon_min:
                        train_epsilon = args.epsilon_min
            else:
                epsilon = eval_with_epsilon

            # action selection
            if state is None or random.random() < epsilon:
                action = random.randrange(environment.get_num_actions())
            else:
                action = dqn.inference(state.get_data())

            # we can't skip frames as in a game
            # we need to wait the evolution of the environment, but we don't want to waste GPU time
            # we can use a training sweep (which requires some time) instead of using a sleep
            old_state = state
            for i in range(0, args.history_length * (args.repeat_action + 1)):
                # Make the move
                reward, state, is_terminal = environment.step(action)

                # train
                if is_training and old_state is not None:
                    if environment.get_step_number() > args.observation_steps:
                        batch = replay_memory.draw_batch(args.batch_size)
                        loss = dqn.train(batch, environment.get_step_number())
                        episode_losses.append(loss)
                    else:
                        time.sleep(args.gpu_time)
                else:
                    time.sleep(args.gpu_time)
                
                if is_terminal:
                    break

            # Record experience in replay memory
            if is_training and old_state is not None:
                replay_memory.add_sample(replay.Sample(old_state, action, reward, state, is_terminal))

            if is_terminal:
                state = None

            if reward == -1:
                stuck_count = stuck_count + 1
            else:
                stuck_count = 0
            if stuck_count > 2:
                print("Car stuck, resetting simulator position...")
                environment.control.reset_simulator()
                stuck_count = 0



        #################################
        # logging
        #################################

        episode_time = datetime.datetime.now() - start_time

        if is_training:
            train_episodes += 1
            episode_train_reward_list.insert(0, environment.get_game_score())
            if len(episode_train_reward_list) > 100:
                episode_train_reward_list = episode_train_reward_list[:-1]
            avg_rewards = np.mean(episode_train_reward_list)

            episode_avg_loss = 0
            if episode_losses:
                episode_avg_loss = np.mean(episode_losses)

            log = ('Episode %d ended with score: %.2f (%s elapsed) (step: %d). Avg score: %.2f Avg loss: %.5f' %
                (environment.get_game_number(), environment.get_game_score(), str(episode_time),
                environment.get_step_number(), avg_rewards, episode_avg_loss))
            print(log)
            print("   epsilon " + str(train_epsilon))
            if args.logging:
                with summary_writer.as_default():
                    tf.summary.text('log', log, step=environment.get_game_number())
                    tf.summary.scalar('train episode reward', environment.get_game_score(), step=train_episodes)
                    tf.summary.scalar('train avg reward(100)', avg_rewards, step=train_episodes)
                    tf.summary.scalar('average loss', episode_avg_loss, step=train_episodes)
        else:
            eval_episodes += 1
            episode_eval_reward_list.insert(0, environment.get_game_score())
            if len(episode_eval_reward_list) > 10:
                episode_eval_reward_list = episode_eval_reward_list[:-1]
            avg_rewards = np.mean(episode_eval_reward_list)

            log = ('Eval %d ended with score: %.2f (%s elapsed) (step: %d). Avg score: %.2f' %
                (environment.get_game_number(), environment.get_game_score(), str(episode_time),
                environment.get_step_number(), avg_rewards))
            print(log)
            if args.logging:
                with summary_writer.as_default():
                    tf.summary.text('log', log, step=environment.get_game_number())
                    tf.summary.scalar('eval episode reward', environment.get_game_score(), step=eval_episodes)
                    tf.summary.scalar('eval avg reward(10)', avg_rewards, step=eval_episodes)

        epoch_total_score += environment.get_game_score()
        environment.reset_game()


    if environment.get_game_number() - start_game_number == 0:
        return 0
    return epoch_total_score / (environment.get_game_number() - start_game_number)

#################################
# main
#################################

while not stop:
    avg_score = run_epoch(args.train_epoch_steps) # train
    print('Average epoch training score: %d' % (avg_score))
    avg_score = run_epoch(args.eval_epoch_steps, eval_with_epsilon=.01) # eval
    print('Average epoch eval score: %d' % (avg_score))
