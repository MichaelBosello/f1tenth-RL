import numpy as np

class State:

    @staticmethod
    def setup(args):
        State.use_compression = args.compress_replay
        State.step_frames = args.frame
        if State.use_compression:
            import blosc

    def state_by_adding_data(self, data):
        
        if State.use_compression:
            data = blosc.compress(data.tobytes(), typesize=1)

        new_state = State()
        if hasattr(self, 'data'):
            new_state.data = self.data[:State.step_frames -1]
            new_state.data.insert(0, data)
        else:
            new_state.data = []
            for i in range(State.step_frames):
                new_state.data.append(data)
        return new_state
    
    def get_data(self):
        if State.use_compression:
            state = []
            for i in range(State.step_frames):
                state.append(np.fromstring(
                    blosc.decompress(self.data[i]),
                dtype=np.float32))
        else:
            state = self.data
        if State.step_frames == 1:
            return state[0]
        return np.concatenate(state)
