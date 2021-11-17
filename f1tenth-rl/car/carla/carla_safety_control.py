EUCLIDEAN_THRESHOLD = 1.8
FOV_MARGIN = 160

class SafetyControl():
    def __init__(self, drive, sensors):
        self.emergency_brake = False
        self.drive = drive
        self.sensors = sensors
        self.sensors.add_lidar_callback(self.lidar_callback)
        self.safety = True

    def lidar_callback(self, lidar_data):
        if self.sensors.main_sensor == '2d-lidar':
            ranges = lidar_data[FOV_MARGIN:-FOV_MARGIN]
        if self.sensors.main_sensor == '3d-lidar':
            ranges = self.sensors.lidar_distances(lidar_data)
        if self.safety:
            if min(ranges) < EUCLIDEAN_THRESHOLD:
                self.emergency_brake = True

            if self.emergency_brake:
                self.drive.stop()

    def unlock_brake(self):
        self.emergency_brake = False

    def disable_safety(self):
        self.safety = False

    def enable_safety(self):
        self.safety = True