from threading import Thread
import time

MONITOR_WAIT = 1/15

class Monitor():
    def __init__(self, sensors):
        self.sensors = sensors
        self.velocity = None
        self.action = None
        process = Thread(target=self.monitor_runner)
        process.daemon = True
        process.start()

    def update(self, action, autonomous_mode):
        self.action = action
        self.autonomous_mode = autonomous_mode
        self.velocity = self.sensors.get_car_linear_acceleration()

    def monitor_runner(self):
        while True:
            if self.velocity is not None:
                print("{} km/h, action: {}, autonomous: {}".format(self.velocity*3.6, self.action, self.autonomous_mode))
            time.sleep(MONITOR_WAIT)