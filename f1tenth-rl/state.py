import numpy as np
import math
import time

try:
    import cv2
except ImportError:
    pass
try:
    import blosc
except ImportError:
    pass

class State:

    @staticmethod
    def setup(args):
        State.use_compression = args.compress_replay
        State.history_length = args.history_length
        State.reduce_by = args.reduce_lidar_data
        State.cut_by = args.cut_lidar_data
        State.max_distance_norm = args.max_distance_norm
        State.add_velocity = args.add_velocity

        State.lidar_reduction_method = args.lidar_reduction_method
        State.lidar_float_cut = args.lidar_float_cut
        State.lidar_to_image = args.lidar_to_image
        State.show_image = args.show_image
        State.image_width = args.image_width
        State.image_height = args.image_height
        State.image_zoom = args.image_zoom

    def state_by_adding_data(self, data):
        data = self.process_data(data)
        
        if State.use_compression:
            data = blosc.compress(data.tobytes(), typesize=1)

        new_state = State()
        if hasattr(self, 'data'):
            new_state.data = self.data[:State.history_length -1]
            new_state.data.insert(0, data)
        else:
            new_state.data = []
            for i in range(State.history_length):
                new_state.data.append(data)
        return new_state
    
    def get_data(self):
        if State.use_compression:
            state = []
            for i in range(State.history_length):
                state.append(np.fromstring(
                    blosc.decompress(self.data[i]),
                dtype=np.float32))
        else:
            state = self.data

        if State.lidar_to_image:
            return np.asarray(state).reshape((State.image_width, State.image_height, State.history_length))
        elif State.add_velocity:
            lidar_state = [state[0][0], state[1][0]]
            acc_state = [state[0][1], state[1][1]]
            return [np.asarray(lidar_state).reshape((len(lidar_state[0]), State.history_length)), np.asarray(acc_state)]
        else:
            return np.asarray(state).reshape((len(state[0]), State.history_length))

    def process_data(self, data):
        if State.add_velocity:
            lidar_data, acceleration_value = data[:-1], data[-1]
            data = lidar_data

        if State.lidar_to_image:
            return self.lidar_to_img(data)

        #we have the same max value for sampling errors and max range exceeding
        #thus, we compute the reduction on the values under the max range and then we sobstitute the max value to the empty sets that resulted in 0
        if State.lidar_reduction_method == 'avg':
            data_avg = []
            for i in range(0, len(data), State.reduce_by):
                filtered = list(filter(lambda x:  x <= State.max_distance_norm, data[i:i + State.reduce_by]))
                if len(filtered) == 0:
                    data_avg.append(State.max_distance_norm)
                else:
                    data_avg.append(sum(filtered)/len(filtered))
            data = data_avg
        if State.lidar_reduction_method == 'sampling':
            data = [data[i] for i in range(0, len(data), State.reduce_by)]
        if State.lidar_reduction_method == 'max':
            data = [i if i <= State.max_distance_norm else 0 for i in data]
            data = [max(data[i:i + State.reduce_by]) for i in range(0, len(data), State.reduce_by)]
            data = [i if i > 0 else State.max_distance_norm for i in data]
        if State.lidar_reduction_method == 'min':
            data = [min(data[i:i + State.reduce_by]) for i in range(0, len(data), State.reduce_by)]

        data = data[State.cut_by:-State.cut_by]
        if State.max_distance_norm > 1:
            data = [x / State.max_distance_norm for x in data]
        if State.lidar_float_cut > -1:
            data = [round(x, State.lidar_float_cut) for x in data]

        if State.add_velocity:
            return (data, acceleration_value)
        else:
            return data

    def lidar_to_img(self, data):
        img_array = np.zeros((State.image_width, State.image_height), dtype=np.uint8)
        for i in range(State.cut_by * State.reduce_by, len(data) - (State.cut_by * State.reduce_by)):
            angle = i * 2*math.pi / 1080
            x = int(data[i] * State.image_zoom * math.cos(angle)) +42
            y = int(data[i] * State.image_zoom * math.sin(angle)) +42
            img_array[x,y] = 255
        if State.show_image:
            cv2.imshow('image', cv2.resize(img_array, (500, 500)))
            cv2.waitKey(1)
        return img_array
