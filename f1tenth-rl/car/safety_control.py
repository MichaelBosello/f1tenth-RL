import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import math
import argparse
import time

TTC_THRESHOLD_SIM = 0.37
TTC_THRESHOLD_REAL_CAR = 1.21

EUCLIDEAN_THRESHOLD_SIM = 0.35
EUCLIDEAN_THRESHOLD_REAL_CAR = 0.35

USE_TTC_SIM = False
USE_TTC_REAL_CAR = True

#if your circuit has only an external perimeter barrier and you want to simulate a lane
ONLY_EXTERNAL_BARRIER = False
EXTERNAL_BARRIER_THRESHOLD = 2.73

class SafetyControl():
    def __init__(self, drive, sensors, is_simulator=False):
        self.emergency_brake = False
        self.drive = drive
        self.sensors = sensors
        self.sensors.add_lidar_callback(self.lidar_callback)
        self.safety = True
        if not is_simulator:
            self.ttc_treshold = TTC_THRESHOLD_REAL_CAR
            self.euclidean_treshold = EUCLIDEAN_THRESHOLD_REAL_CAR
            self.use_ttc = USE_TTC_REAL_CAR
        else:
            self.ttc_treshold = TTC_THRESHOLD_SIM
            self.euclidean_treshold = EUCLIDEAN_THRESHOLD_SIM
            self.use_ttc = USE_TTC_SIM

    def lidar_callback(self, lidar_data):
        if self.safety:
            if self.use_ttc:
                acceleration = self.sensors.get_car_linear_velocity()
                if acceleration > 0:
                    for i in range(len(lidar_data.ranges)):
                        angle = lidar_data.angle_min + i * lidar_data.angle_increment
                        proj_velocity = acceleration * math.cos(angle)
                        if proj_velocity != 0:
                            ttc = lidar_data.ranges[i] / proj_velocity
                            if ttc < self.ttc_treshold and ttc >= 0:
                                self.emergency_brake = True
                                break
            
            if min(lidar_data.ranges) < self.euclidean_treshold:
                self.emergency_brake = True

            if ONLY_EXTERNAL_BARRIER and min(lidar_data.ranges) > EXTERNAL_BARRIER_THRESHOLD:
                self.emergency_brake = True

            if self.emergency_brake:
                self.drive.stop()

    def unlock_brake(self):
        self.emergency_brake = False

    def disable_safety(self):
        self.safety = False

    def enable_safety(self):
        self.safety = True

if __name__ == '__main__':
    from sensors import Sensors
    from car_control import Drive

    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()

    rospy.init_node('safety_control_test')
    sensors = Sensors(args.simulator)
    drive = Drive(sensors, args.simulator)
    safety_control = SafetyControl(drive, sensors, args.simulator)
    time.sleep(0.5)
    while not safety_control.emergency_brake:
        safety_control.drive.forward()
        time.sleep(0.01)
    safety_control.drive.stop()
    safety_control.unlock_brake()
    safety_control.sensors.lidar_subscriber.unregister()
    print("safe_brake! exiting..")
  

