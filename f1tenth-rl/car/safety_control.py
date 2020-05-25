import rospy
from car.car_control import Drive
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import argparse
import time

TTC_THRESHOLD = 0.8

class SafetyControl():
    def __init__(self, is_simulator=False):
        self.odometry = None
        self.emergency_brake = False
        self.drive = Drive(is_simulator)
        rospy.Subscriber("odom", Odometry, self.odometry_callback)
        self.safety_subscriber = rospy.Subscriber("scan", LaserScan, self.lidar_callback)

    def lidar_callback(self, lidar_data):
        if self.odometry != None and self.odometry.twist.twist.linear.x != 0:
            for i in range(len(lidar_data.ranges)):
                angle = lidar_data.angle_min + i * lidar_data.angle_increment
                proj_velocity = self.odometry.twist.twist.linear.x * math.cos(angle)
                if proj_velocity != 0:
                    ttc = lidar_data.ranges[i] / proj_velocity
                    if ttc < TTC_THRESHOLD and ttc >= 0:
                        self.emergency_brake = True
                        break
        if self.emergency_brake:
            self.drive.stop()

    def odometry_callback(self, odometry):
        self.odometry = odometry

    def unlock_brake(self):
        self.emergency_brake = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()


    rospy.init_node('safety_control_test')
    safety_control = SafetyControl(args.simulator)
    time.sleep(0.01)
    while not safety_control.emergency_brake:
        safety_control.drive.lightly_right()
        time.sleep(0.01)
    safety_control.drive.stop()
    safety_control.unlock_brake()
    safety_control.safety_subscriber.unregister()
    print("safe_brake! exiting..")
  

