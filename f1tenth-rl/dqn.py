import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers

import state
from state import State

class DeepQNetwork:
    def __init__(self, num_actions, state_size, base_dir, args):
        
        self.num_actions = num_actions
        self.state_size = state_size
        self.history_length = args.history_length

        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.target_model_update_freq = args.target_model_update_freq

        self.checkpoint_dir = base_dir + '/models/'
        self.save_model_freq = args.save_model_freq

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


        self.behavior_net = self.__build_q_net()
        self.target_net = self.__build_q_net()

        model_as_string = []
        self.target_net.summary(print_fn=lambda x : model_as_string.append(x))
        "\n".join(model_as_string)

        summary_writer = tf.summary.create_file_writer(base_dir)
        with summary_writer.as_default():
            tf.summary.text('model', model_as_string, step=0)
            

        if args.model is not None:
            self.target_net.load_weights(args.model)
            self.behavior_net.set_weights(self.target_net.get_weights())


    def __build_q_net(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length))
        x = layers.Conv1D(filters=64, kernel_size=10, activation='relu')(inputs)
        x = layers.MaxPooling1D(pool_size=3)(x)
        x = layers.Conv1D(filters=128, kernel_size=10, activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(self.num_actions, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate, decay=.98))
        model.summary()
        return model


        
    def inference(self, state):
        state = np.asarray(state).reshape((-1, self.state_size, self.history_length))
        q_vals = self.behavior_net.predict(state)[0]
        return q_vals.argmax()

        
    def train(self, batch, step_number):

        old_states = np.asarray([sample.old_state.get_data() for sample in batch])
        new_states = np.asarray([sample.new_state.get_data() for sample in batch])
        actions = np.asarray([sample.action for sample in batch])
        rewards = np.asarray([sample.reward for sample in batch])
        is_terminal = np.asarray([sample.terminal for sample in batch])

        q_new_state = self.target_net.predict(new_states).argmax(axis=1)
        target_q = np.where(is_terminal, rewards, rewards+self.gamma*q_new_state)

        with tf.GradientTape() as tape:
            q_values = self.target_net(old_states)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            current_q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            loss = tf.reduce_mean(tf.square(target_q - current_q))

        variables = self.target_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.target_net.optimizer.apply_gradients(zip(gradients, variables))

        if step_number % self.target_model_update_freq == 0:
            self.behavior_net.set_weights(self.target_net.get_weights())

        if step_number % self.save_model_freq == 0:
            self.target_net.save_weights(self.checkpoint_dir)

        return loss
