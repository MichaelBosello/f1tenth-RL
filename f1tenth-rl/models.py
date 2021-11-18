import tensorflow as tf
from tensorflow.keras import layers, initializers, losses, optimizers

def build_dense(state_size, history_length, num_actions, learning_rate):
    inputs = tf.keras.Input(shape=(state_size, history_length))
    x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.Flatten()(x)
    predictions = layers.Dense(num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                        loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
    model.summary()
    return model

def build_cnn1D(state_size, history_length, num_actions, learning_rate):
    inputs = tf.keras.Input(shape=(state_size, history_length))
    x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
    x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    predictions = layers.Dense(num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                        loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
    model.summary()
    return model

def build_cnn1D_plus_velocity(state_size, history_length, num_actions, learning_rate):
    inputs = tf.keras.Input(shape=(state_size, history_length), name="lidar")
    input_acceleration = tf.keras.Input(shape=((history_length)), name="acc")
    x = layers.Conv1D(filters=16, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(inputs)
    x = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.Flatten()(x)
    x = layers.concatenate([x, input_acceleration])
    x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    predictions = layers.Dense(num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    model = tf.keras.Model(inputs=[inputs, input_acceleration], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                        loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
    model.summary()
    return model

def build_cnn2D(image_width, image_height, history_length, num_actions, learning_rate):
    inputs = tf.keras.Input(shape=(image_width, image_height, history_length))
    x = layers.Lambda(lambda layer: layer / 255)(inputs)
    x = layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(filters=8, kernel_size=(2, 2), strides=(1, 1), activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    predictions = layers.Dense(num_actions, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                        loss=losses.Huber()) #loss to be removed. It is needed in the bugged version installed on Jetson
    model.summary()
    return model