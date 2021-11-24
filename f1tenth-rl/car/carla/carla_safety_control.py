try:
    from car.carla.carla_sensors import *
except:
    from carla_sensors import *

EUCLIDEAN_THRESHOLD = 1.8
EUCLIDEAN_THRESHOLD_MAX_DISTANCE = 10
FOV_MARGIN = 160

class SafetyControl():
    def __init__(self, drive, sensors):
        self.emergency_brake = False
        self.drive = drive
        self.sensors = sensors
        if sensors.main_sensor == '2d-lidar' or '2d-lidar' in sensors.side_sensors:
            self.sensors.add_2d_lidar_callback(self.lidar_callback)
        if 'collision' in sensors.side_sensors:
            self.sensors.add_collision_callback(self.stop_event_callback)
        if 'lane-invasion' in sensors.side_sensors:
            self.sensors.add_lane_invasion_callback(self.stop_event_callback)
        self.safety = True

    def lidar_callback(self, lidar_data):
        ranges = lidar_data[FOV_MARGIN:-FOV_MARGIN]
        ranges = list(filter(lambda i: i < LIDAR_2D_RANGE, ranges))
        if self.safety and len(ranges) > 0:
            if min(ranges) < EUCLIDEAN_THRESHOLD or min(ranges) > EUCLIDEAN_THRESHOLD_MAX_DISTANCE:
                self.emergency_brake = True

            if self.emergency_brake:
                self.drive.stop()

    def stop_event_callback(self, _):
        if self.safety:
            self.emergency_brake = True
            self.drive.stop()


    def unlock_brake(self):
        self.emergency_brake = False

    def disable_safety(self):
        self.safety = False

    def enable_safety(self):
        self.safety = True