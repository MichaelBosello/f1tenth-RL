import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import math
import argparse
import time

TTC_THRESHOLD_SIM = 0.37
TTC_THRESHOLD_REAL_CAR = 0.7

EUCLIDEAN_THRESHOLD_SIM = 0.45
EUCLIDEAN_THRESHOLD_REAL_CAR = 0.29

USE_TTC = False

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
        else:
            self.ttc_treshold = TTC_THRESHOLD_SIM
            self.euclidean_treshold = EUCLIDEAN_THRESHOLD_SIM

    def lidar_callback(self, lidar_data):
        if self.safety:
            if USE_TTC:
                acelleration = self.sensors.get_car_linear_acelleration()
                if acelleration > 0:
                    for i in range(len(lidar_data.ranges)):
                        angle = lidar_data.angle_min + i * lidar_data.angle_increment
                        proj_velocity = acelleration * math.cos(angle)
                        if proj_velocity != 0:
                            ttc = lidar_data.ranges[i] / proj_velocity
                            if ttc < self.ttc_treshold and ttc >= 0:
                                self.emergency_brake = True
                                break
            
            if min(lidar_data.ranges) < self.euclidean_treshold:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()

    rospy.init_node('safety_control_test')
    safety_control = SafetyControl(args.simulator)
    time.sleep(0.5)
    while not safety_control.emergency_brake:
        safety_control.drive.forward()
        time.sleep(0.01)
    safety_control.drive.stop()
    safety_control.unlock_brake()
    safety_control.sensors.lidar_subscriber.unregister()
    print("safe_brake! exiting..")
  

