import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers

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
            return self.__build_cnn2D()
        else:
            if self.add_velocity:
                return self.__build_cnn1D_plus_velocity()
            else:
                # select from __build_dense or build_cnn1D
                return self.__build_cnn1D()

    def __build_dense(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length))
        x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def __build_cnn1D(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length))
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def __build_cnn1D_plus_velocity(self):
        inputs = tf.keras.Input(shape=(self.state_size, self.history_length), name="lidar")
        input_acceleration = tf.keras.Input(shape=((self.history_length)), name="acc")
        x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
        x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.Flatten()(x)
        x = layers.concatenate([x, input_acceleration])
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=[inputs, input_acceleration], outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model

    def __build_cnn2D(self):
        inputs = tf.keras.Input(shape=(self.image_width, self.image_height, self.history_length))
        x = layers.Lambda(lambda layer: layer / 255)(inputs)
        x = layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        predictions = layers.Dense(self.num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                            loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
        model.summary()
        return model


    def inference(self, state):
        if self.lidar_to_image:
            state = state.reshape((-1, self.image_width, self.image_height, self.history_length))
        elif self.add_velocity:
            state[0] = state[0].reshape((-1, self.state_size, self.history_length))
            state[1] = state[1].reshape((-1, self.history_length))
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
