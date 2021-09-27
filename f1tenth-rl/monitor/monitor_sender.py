from threading import Thread
import socket
import pickle

UDP_IP = "192.168.1.23"
UDP_PORT = 5005

class Monitor():
    def __init__(self, sensors):
        self.sensors = sensors
        self.action = None
        self.autonomous_mode = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        process = Thread(target=self.monitor_runner)
        process.daemon = True
        process.start()

    def update(self, action, autonomous_mode):
        self.action = action
        self.autonomous_mode = autonomous_mode

    def monitor_runner(self):
        while True:
            if self.action is not None:
                data = {
                    "spd": self.sensors.get_car_linear_velocity(),
                    "act": self.action,
                    "auto": self.autonomous_mode,
                    "ldr": self.sensors.lidar_data,
                    "orn": self.sensors.get_car_orientation(),
                    "acc": self.sensors.get_car_linear_acceleration(),
                }
                msg = pickle.dumps(data)
                self.sock.sendto(msg, (UDP_IP, UDP_PORT))