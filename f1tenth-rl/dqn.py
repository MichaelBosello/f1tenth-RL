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

        self.checkpoint_dir = base_dir + '/models'
        self.save_model_freq = args.save_model_freq

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


        self.behavior_net = self.__build_q_net()
        self.target_net = self.__build_q_net()

        if args.model is not None:
            self.target_net.load_weights(args.model)
            self.behavior_net.set_weights(self.target_net.get_weights())


    def __build_q_net(self):
        inputs = tf.keras.Input(shape=(self.history_length, self.state_size))
        x = layers.Flatten()(inputs)
        x = layers.Dense(100, activation='relu',
            kernel_initializer='RandomUniform', bias_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(100, activation='relu',
            kernel_initializer='RandomUniform', bias_regularizer=tf.keras.regularizers.l2(0.01))(x)
        predictions = layers.Dense(self.num_actions, activation='relu',
            kernel_initializer='RandomUniform', bias_regularizer=tf.keras.regularizers.l2(0.01))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(self.learning_rate, clipvalue=1, decay=.95, epsilon=.01))
        print(model.summary())
        return model


        
    def inference(self, state):
        state = np.asarray(state).reshape((-1, self.history_length, self.state_size))
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
