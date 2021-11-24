import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import losses

from models import *

class DeepQNetwork:
    def __init__(self, num_actions, state_size, replay_buffer, base_dir, tensorboard_dir, args):

        self.num_actions = num_actions
        self.state_size = state_size
        self.replay_buffer = replay_buffer
        self.history_length = args.history_length

        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.target_model_update_freq = args.target_model_update_freq

        self.checkpoint_dir = base_dir + '/models/'

        self.lidar_to_image = args.lidar_to_image
        self.image_width = args.image_width
        self.image_height = args.image_height

        self.add_velocity = args.add_velocity

        self.lidar_3d = args.lidar_3d

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


        self.behavior_net = self.__build_q_net()
        self.target_net = self.__build_q_net()

        model_as_string = []
        self.target_net.summary(print_fn=lambda x : model_as_string.append(x))
        "\n".join(model_as_string)

        summary_writer = tf.summary.create_file_writer(tensorboard_dir)
        with summary_writer.as_default():
            tf.summary.text('model', model_as_string, step=0)

        if args.model is not None:
            self.target_net.load_weights(args.model)
            self.behavior_net.set_weights(self.target_net.get_weights())


    def __build_q_net(self):
        if self.lidar_to_image:
            return build_cnn2D(self.image_width, self.image_height, self.history_length, self.num_actions, self.learning_rate)
        elif self.lidar_3d:
            return build_pointnet(self.state_size[0], self.state_size[1], self.history_length, self.num_actions, self.learning_rate)
        else:
            if self.add_velocity:
                return build_cnn1D_plus_velocity(self.state_size, self.history_length, self.num_actions, self.learning_rate)
            else:
                # select from __build_dense or build_cnn1D
                return build_cnn1D(self.state_size, self.history_length, self.num_actions, self.learning_rate)


    def inference(self, state):
        if self.lidar_to_image:
            state = state.reshape((-1, self.image_width, self.image_height, self.history_length))
        elif self.add_velocity:
            state[0] = state[0].reshape((-1, self.state_size, self.history_length))
            state[1] = state[1].reshape((-1, self.history_length))
        elif self.lidar_3d:
            state = state.reshape((-1, self.state_size[0], self.state_size[1] * self.history_length))
        else:
            state = state.reshape((-1, self.state_size, self.history_length))
        return np.asarray(self.behavior_predict(state)).argmax(axis=1)

    def train(self, batch, step_number):
        if self.add_velocity:
            old_states_lidar = np.asarray([sample.old_state.get_data()[0] for sample in batch])
            old_states_acc = np.asarray([sample.old_state.get_data()[1] for sample in batch])
            new_states_lidar = np.asarray([sample.new_state.get_data()[0] for sample in batch])
            new_states_acc = np.asarray([sample.new_state.get_data()[1] for sample in batch])
            actions = np.asarray([sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])

            predicted = self.target_predict({'lidar': new_states_lidar, 'acc': new_states_acc})
            q_new_state = np.max(predicted, axis=1)
            target_q = rewards + (self.gamma*q_new_state * (1-is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)# using tf.one_hot causes strange errors

            loss = self.gradient_train({'lidar': old_states_lidar, 'acc': old_states_acc}, target_q, one_hot_actions)
        else:
            old_states = np.asarray([sample.old_state.get_data() for sample in batch])
            new_states = np.asarray([sample.new_state.get_data() for sample in batch])
            actions = np.asarray([sample.action for sample in batch])
            rewards = np.asarray([sample.reward for sample in batch])
            is_terminal = np.asarray([sample.terminal for sample in batch])

            q_new_state = np.max(self.target_predict(new_states), axis=1)
            target_q = rewards + (self.gamma*q_new_state * (1-is_terminal))
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions)# using tf.one_hot causes strange errors

            loss = self.gradient_train(old_states, target_q, one_hot_actions)

        if step_number % self.target_model_update_freq == 0:
            self.behavior_net.set_weights(self.target_net.get_weights())

        return loss

    @tf.function
    def target_predict(self, state):
        return self.target_net(state)

    @tf.function
    def behavior_predict(self, state):
        return self.behavior_net(state)

    @tf.function
    def gradient_train(self, old_states, target_q, one_hot_actions):
        with tf.GradientTape() as tape:
            q_values = self.target_net(old_states)
            current_q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            loss = losses.Huber()(target_q, current_q)

        variables = self.target_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.target_net.optimizer.apply_gradients(zip(gradients, variables))

        return loss


    def save_network(self):
        print("saving..")
        self.target_net.save_weights(self.checkpoint_dir)
        self.replay_buffer.save()
        print("saved")
