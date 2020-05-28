import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time
import math
import argparse

try:
    import Jetson.GPIO as GPIO
except ImportError:
    pass

LX_IR_SENSOR_PIN = 1
RX_IR_SENSOR_PIN = 0

class Sensors():
    def __init__(self, is_simulator=False, add_lidar_callback=None):
        self.add_lidar_callback = add_lidar_callback
        self.lidar_data = None
        self.odometry = None
        self.lidar_subscriber = rospy.Subscriber("scan", LaserScan, self.lidar_callback)
        self.odom_subscriber = rospy.Subscriber("odom", Odometry, self.odometry_callback)

        if not is_simulator:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(LX_IR_SENSOR_PIN, GPIO.IN)
            GPIO.setup(RX_IR_SENSOR_PIN, GPIO.IN)

    def lidar_callback(self, lidar_data):
        self.lidar_data = lidar_data
        if self.add_lidar_callback:
            self.add_lidar_callback(lidar_data)

    def odometry_callback(self, odometry):
        self.odometry = odometry

    def get_lidar_ranges(self):
        return self.lidar_data.ranges

    def get_car_linear_acelleration(self):
        if self.odometry is None or (self.odometry.twist.twist.linear.x == 0 and self.odometry.twist.twist.linear.x == 0):
            return 0
        return math.sqrt(self.odometry.twist.twist.linear.x ** 2 + self.odometry.twist.twist.linear.y ** 2)
    def get_car_angular_acelleration(self):
        return self.odometry.twist.twist.angular

    def back_obstacle(self):
        return (self.GPIO.input(LX_IR_SENSOR_PIN) == self.GPIO.LOW
                    or self.GPIO.input(RX_IR_SENSOR_PIN) == self.GPIO.LOW)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", action='store_true', help="to set the use of the simulator")
    args = parser.parse_args()

    rospy.init_node('sensors_test')
    sensor = Sensors(args.simulator)
    time.sleep(1)
    while True:
        print(sensor.lidar_data)
        print(sensor.odometry)
        print(sensor.get_car_linear_acelleration())
        if not args.simulator:
            print(sensor.back_obstacle())
        time.sleep(0.2)