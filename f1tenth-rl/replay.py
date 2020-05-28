import bisect
import math
import random
import os
import pickle

class Sample:
    
    def __init__(self, old_state, action, reward, new_state, terminal):
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.terminal = terminal
        self.weight = 1
        self.cumulative_weight = 1

    def is_interesting(self):
        return self.terminal or self.reward != 0

    def __cmp__(self, obj):
        return self.cumulative_weight - obj.cumulative_weight


class ReplayMemory:
    
    def __init__(self, base_output_dir, args):
        self.save_buffer_dir = base_output_dir + "/models/"
        if not os.path.isdir(self.save_buffer_dir):
            os.makedirs(self.save_buffer_dir)
        self.file = "replay_buffer.dat"
        self.samples = []
        self.max_samples = args.replay_capacity
        self.prioritized_replay = args.prioritized_replay
        self.num_interesting_samples = 0
        self.batches_drawn = 0

        if args.model is not None:
            self.load(args.model + self.file)

    def num_samples(self):
        return len(self.samples)

    def add_sample(self, sample):
        self.samples.append(sample)
        if self.prioritized_replay:
            self._update_weights()
        self._truncate_list_if_necessary()

    def draw_batch(self, batch_size):
        if batch_size > len(self.samples):
            raise IndexError('Too few samples (%d) to draw a batch of %d' % (len(self.samples), batch_size))
        
        if self.prioritized_replay:
            self.batches_drawn += 1
            return self._draw_prioritized_batch(batch_size)
        else:
            return random.sample(self.samples, batch_size)

    def save(self):
        with open(self.save_buffer_dir + self.file, "wb") as f:
            pickle.dump(self.samples, f)

    def load(self, file):
        with open(file, "rb") as f:
            self.samples = pickle.load(f)




    def _truncate_list_if_necessary(self):
        # optimizastion: don't truncate on each added sample since it requires a memcopy of the list
        if len(self.samples) > self.max_samples * 1.05:

            if self.prioritized_replay:
                truncated_weight = 0
                for i in range(self.max_samples, len(self.samples)):
                    truncated_weight += self.samples[i].weight
                    if self.samples[i].is_interesting():
                        self.num_interesting_samples -= 1

            # Truncate the list
            self.samples = self.samples[(len(self.samples) - self.max_samples):]
            
            # Correct cumulativeWeights
            if self.prioritized_replay:
                for sample in self.samples:
                    sample.cumulative_weight -= truncated_weight

    def _draw_prioritized_batch(self, batch_size):
        batch = []
        probe = Sample(None, 0, 0, None, False)
        while len(batch) < batch_size:
            probe.cumulative_weight = random.uniform(0, self.samples[-1].cumulative_weight)
            index = bisect.bisect_right(self.samples, probe, 0, len(self.samples) - 1)
            sample = self.samples[index]
            sample.weight = max(1, .8 * sample.weight)
            if sample not in batch:
                batch.append(sample)

        if self.batches_drawn % 100 == 0:
            cumulative = 0
            for sample in self.samples:
                cumulative += sample.weight
                sample.cumulative_weight = cumulative
        return batch

    def _update_weights(self):
        if len(self.samples) > 1:
            self.samples[-1].cumulative_weight = self.samples[-1].weight + self.samples[-2].cumulative_weight

        if self.samples[-1].is_interesting():
            self.num_interesting_samples += 1
            
            # Boost the neighboring samples.  How many samples?  Roughly the number of samples
            # that are "uninteresting".  Meaning if interesting samples occur 3% of the time, then boost 33
            uninteresting_sample_range = max(1, len(self.samples) / max(1, self.num_interesting_samples))
            uninteresting_sample_range = int(uninteresting_sample_range)
            for i in range(uninteresting_sample_range, 0, -1):
                index = len(self.samples) - i
                if index < 1:
                    break
                # This is an exponential that ranges from 3.0 to 1.01 over the domain of [0, uninterestingSampleRange]
                # So the interesting sample gets a 3x boost, and the one furthest away gets a 1% boost
                boost = 1.0 + 3.0/(math.exp(i/(uninteresting_sample_range/6.0)))
                self.samples[index].weight *= boost
                self.samples[index].cumulative_weight = self.samples[index].weight + self.samples[index - 1].cumulative_weight
