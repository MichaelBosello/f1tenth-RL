import numpy as np

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
        return np.transpose(state, axes=(1,0))

    def process_data(self, data):
        data = [sum(data[i:i + State.reduce_by])/State.reduce_by for i in range(0, len(data), State.reduce_by)]
        return data

