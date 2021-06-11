import time
import os
import concurrent.futures 

class AsyncLogger():
    def __init__(self, base_dir, is_rl_log=False):
        log_dir = base_dir + "/logging/"
        os.makedirs(log_dir)
        self.file_name = time.strftime("%d-%m-%Y_%H:%M:%S", time.localtime())
        self.text_path = log_dir + self.file_name + ".csv"
        self.text_file = open(self.text_path, "a")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 
        if is_rl_log:
            self.log("episode, state, action, reward\n")

    def rl_log(self, episode, state, action, reward):
        self.log('{}, {}, {}, {}\n'.format(episode, state, action, reward))

    def log(self, message):
        self.executor.submit(self.text_file.write, message)
        self.executor.submit(self.text_file.flush)

    def close_logger(self):
        self.text_file.close()